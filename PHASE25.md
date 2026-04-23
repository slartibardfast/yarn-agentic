# Phase 25: TURBO_KV_4B AVX2 Kernel Design — Non-AVX-512 Microarchitecture Targets

## Status

Design document — **REVISED**. The earlier draft made ISA-availability claims that do not match what the targeted CPUs actually implement; that draft is superseded by this file. See the final "Notes" section for what changed and why. No kernel code written yet.

## Scope

Non-AVX-512 x86 CPUs with AVX2.

AVX-512-capable CPUs fall through to the scalar reference path until a dedicated AVX-512 kernel is written. This is a scope decision: the AVX-512 kernel is future work, not a claim that AVX-512 is slower than AVX2 on modern silicon. On Skylake-SP and Ice Lake-SP AVX-512 did incur measurable frequency-license throttling that could make AVX2 competitive on latency-bound code; on Zen 4 and Sapphire Rapids that penalty is largely absent.

## Target microarchitectures

| # | Microarchitecture | Vendor | Year | Notes |
|---|-------------------|--------|------|-------|
| 1 | Haswell | Intel | 2013 | AVX2 baseline; legacy server (Xeon E5-26xx v3, Xeon E3 v3). |
| 2 | Skylake client | Intel | 2015 | AVX2 only; 6th–10th gen Core consumer/mobile. Skylake-W/SP/X is excluded — those have AVX-512. |
| 3 | Alder Lake / Raptor Lake | Intel | 2021–2022 | P-core AVX-512 fused off at retail. E-cores never shipped AVX-512. Treated as AVX2-only. |
| 4 | Zen 2 | AMD | 2019 | First Zen generation with native 256-bit AVX2 execution. Ryzen 3000, Threadripper 3000, EPYC 7002. |
| 5 | Zen 3 | AMD | 2020 | Refines Zen 2. Ryzen 5000, EPYC 7003. |

**Explicitly excluded:**

- **Zen 4 (Ryzen 7000, EPYC 9004)** — has AVX-512F/BW/DQ/VL/VBMI/VBMI2/VNNI/BF16/VPOPCNTDQ/BITALG/GFNI. Covered by the AVX-512 scope decision above.
- **Skylake-W, Skylake-X, Cascade Lake, Ice Lake-SP, Sapphire Rapids, Emerald Rapids** — all AVX-512.
- **Zen 1 / Zen+** — implement the full AVX2 ISA but decompose 256-bit ops into two 128-bit µops. Consistently slower than Zen 2. Left out to bound initial scope; can be readmitted if real-world hosts turn up.
- **Broadwell** — microarchitecturally close to Haswell; no separate kernel warranted. The Haswell kernel will run on Broadwell unchanged.
- **Goldmont, Tremont, Gracemont (standalone)** — Atom tier, not an LLM hosting target.

## Instruction availability

All 15 instructions below are in AVX2 or earlier. Every AVX2-capable CPU in scope implements all of them. There is no ISA-driven kernel split.

| # | Instruction | Introduced in | Role in TURBO |
|---|-------------|---------------|---------------|
| 1 | VPERM2I128 | AVX2 | RHT butterfly lane swap |
| 2 | VPERMD | AVX2 | RHT sign mask application (32-bit lane-crossing shuffle) |
| 3 | VPERMQ | AVX2 | RHT butterfly (64-bit lane-crossing shuffle) |
| 4 | VPERMPD | AVX2 | RHT butterfly (64-bit float permute) |
| 5 | VPERM2F128 | AVX | 128-bit lane permute |
| 6 | VPMOVZXBW | SSE4.1 / AVX2 YMM | Bit unpack: u8 → u16 |
| 7 | VPMOVSXBW | SSE4.1 / AVX2 YMM | Sign-extend unpack: i8 → i16 |
| 8 | VPMOVMSKB | SSE2 / AVX2 YMM | Extract byte bitmask from YMM |
| 9 | VPSHUFB | SSSE3 / AVX2 YMM | Table-driven byte shuffle (codebook lookup) |
| 10 | VPSLLVD | AVX2 | Variable left shift (bit unpack) |
| 11 | VCVTDQ2PD | SSE2 / AVX2 YMM | int32 → float64 (norm computation) |
| 12 | VEXTRACTF128 | AVX | Extract XMM from YMM |
| 13 | VINSERTF128 | AVX | Insert XMM into YMM |
| 14 | VRSQRTPS | SSE / AVX2 YMM | Reciprocal square root (normalization) |
| 15 | VBLENDVPS / VBLENDVPD | AVX | Mask-driven conditional blend (quantization clamp) |

```
Instruction          Haswell  SKL-client  ADL/RPL  Zen 2  Zen 3
VPERM2I128               ✓         ✓          ✓       ✓      ✓
VPERMD                   ✓         ✓          ✓       ✓      ✓
VPERMQ                   ✓         ✓          ✓       ✓      ✓
VPERMPD                  ✓         ✓          ✓       ✓      ✓
VPERM2F128               ✓         ✓          ✓       ✓      ✓
VPMOVZXBW                ✓         ✓          ✓       ✓      ✓
VPMOVSXBW                ✓         ✓          ✓       ✓      ✓
VPMOVMSKB                ✓         ✓          ✓       ✓      ✓
VPSHUFB                  ✓         ✓          ✓       ✓      ✓
VPSLLVD                  ✓         ✓          ✓       ✓      ✓
VCVTDQ2PD                ✓         ✓          ✓       ✓      ✓
VEXTRACTF128             ✓         ✓          ✓       ✓      ✓
VINSERTF128              ✓         ✓          ✓       ✓      ✓
VRSQRTPS                 ✓         ✓          ✓       ✓      ✓
VBLENDVPS/PD             ✓         ✓          ✓       ✓      ✓
```

## Throughput variation — qualitative, to be quantified

Per-uarch latency, reciprocal throughput, and port-binding data must be extracted directly from Agner Fog's `instruction_tables.ods` and committed as a structured data file before any kernel decisions are made on timing grounds. Specific figures are not reproduced here — see MEMORY feedback "Follow published specs, don't riff" and the `no-riff` note in Design Constraints.

Known qualitative differences worth measuring, from published microarchitectural descriptions (not from invented numbers):

- **VPSLLVD** reciprocal throughput improved materially between Zen 2 and Zen 3. If bit-unpack is a hot path, this is the most likely trigger for a Zen 2-specific variant.
- **VPERMD / VPERMQ** are lane-crossing; they are the RHT bottleneck on all targets and dominate latency regardless of uarch.
- **VRSQRTPS** reciprocal throughput differs between Intel client and Zen.
- **VBLENDVPS/PD** reciprocal throughput is worse on Haswell than on later Intel or Zen.

All of these are throughput differences, not capability differences.

## Kernel strategy

**Baseline: a single AVX2 kernel covering all 5 targets.**

**Rationale:** the ISA is uniform. A single compiled kernel emits the same instructions on every target. A multi-kernel split would add maintenance cost without fixing a concrete problem. The earlier draft's two-kernel split was derived from an ISA matrix that turned out to be wrong.

**What this design does not include ahead of measurement:**

- No Zen 2-specific VPSLLVD-avoidance variant until VPSLLVD is measured as the bottleneck.
- No "VPAND-optimized path" — VPAND/VPANDN are SSE2 (2001) and part of the baseline kernel on every target.
- No emulation paths — all 15 instructions are natively available everywhere in scope.

**Conditional second kernel:** if profiling one representative CPU per target class identifies a single instruction as a dominant bottleneck on a specific uarch, introduce a variant kernel at that point with the measurement as justification. Do not speculate.

## CPUID detection

```c
// AVX2 is required for this kernel.
bool cpu_has_avx2 = (cpuid(7, 0).ebx >> 5) & 1;

// Any AVX-512F implies the AVX-512 scope fallthrough.
bool cpu_has_avx512f = (cpuid(7, 0).ebx >> 16) & 1;

if (!cpu_has_avx2)       return scalar_quantize_row_turbo_kv_4b(...);
if ( cpu_has_avx512f)    return scalar_quantize_row_turbo_kv_4b(...);  // scope decision

// All remaining AVX2-only CPUs run the same kernel, uarch-independent.
```

**Family/model reference (informational — no uarch-specific dispatch yet):**

- Intel, family 6:
  - Haswell: models `0x3C`, `0x3F`, `0x45`, `0x46`
  - Broadwell: models `0x3D`, `0x47`, `0x4F`, `0x56`
  - Skylake client: models `0x4E`, `0x5E`
  - Alder Lake: models `0x97`, `0x9A`
  - Raptor Lake: models `0xB7`, `0xBA`, `0xBE`, `0xBF`
  - (Skylake-SP/W/X is family 6 model `0x55` — has AVX-512, excluded via the AVX-512F bit.)
- AMD:
  - Zen 2: family `0x17`, models include `0x31`, `0x47`, `0x60`, `0x68`, `0x71`, `0xA0`
  - Zen 3: family `0x19`, models `0x00`–`0x0F`, `0x20`–`0x2F`, `0x40`–`0x4F`, `0x50`–`0x5F`
  - Zen 4: family `0x19`, models `0x10`–`0x1F`, `0x60`–`0x6F`, `0x70`–`0x7F`, `0xA0`–`0xAF` (excluded via the AVX-512F bit)

## Design constraints

1. **Test-first:** the kernel must pass all 20 Allium obligations from PHASE24 before it ships. No deferrals.
2. **Surgical changes:** match existing `ggml-turbo-kv.c` style exactly.
3. **No workarounds:** if a target needs something this design did not anticipate, implement it. Do not downgrade types or hack around it.
4. **No riff on tables:** actual Agner Fog figures go into a separate committed data file with row citations. Do not quote timing numbers from memory.
5. **GPU testing (repo rule, noted for context):** GPU tests use Vega only (`GGML_VK_VISIBLE_DEVICES=1`); does not apply to CPU kernels but flagged here so it doesn't fall off the checklist at integration time.

## Next steps

1. Extract latency / reciprocal-throughput / port-binding data for all 15 instructions across the 5 targets from Agner Fog `instruction_tables.ods`. Commit as a structured file (CSV or similar) alongside this doc, with source row numbers cited.
2. Implement the single-kernel AVX2 baseline in `ggml-turbo-kv.c` behind runtime CPUID dispatch.
3. Verify all 20 Allium obligations via `test-backend-ops`.
4. Profile one representative CPU per target class. Introduce a second kernel only if measurement demands it.

## Notes

**Supersedes earlier revision.** The earlier draft contained several ISA-availability claims that do not match reality and were not derived from Agner Fog's tables despite the citation. Concretely:

- It marked **VPSHUFB** as missing on Zen 2 and Zen 3. VPSHUFB is SSSE3 (2006) with an AVX2 YMM form; every AVX2-capable CPU has it, including Zen 1.
- It marked **VBLENDVPS/PD** as missing on Zen 2, Zen 3, and Zen 4. VBLENDVPS/PD is AVX (2011); every AVX-capable CPU has it.
- It marked **VPERMPD** as missing on Zen 4. VPERMPD is AVX2; Zen 4 has the full AVX2 ISA.
- It described **VPAND/VPANDN** as introduced on Zen 4. They are SSE2 (2001).
- It listed **VPOPCNTD** as an AVX2 instruction. `VPOPCNTD` is AVX-512VPOPCNTDQ, not AVX2.
- It treated **Zen 4** as an AVX2-only target. Zen 4 has full AVX-512F/BW/DQ/VL/VBMI/VBMI2/VNNI/BF16/VPOPCNTDQ/BITALG/GFNI.
- CPUID constants were wrong: Zen 4 is family `0x19` (same as Zen 3, distinguished by model), not family `0x1A`; Haswell does not include model `0x4F` (that is Broadwell-EP).

The 2-kernel split in the earlier draft was derived from the incorrect matrix and is abandoned here in favour of a single-kernel baseline plus measurement-driven variants.

**Zen 1 / Zen+** implement full AVX2 (the earlier draft's "Zen 1 has only ~10 AVX2 instructions" claim is wrong), but 128-bit internal execution halves their effective throughput on 256-bit ops. They can be readmitted to scope if a deployment needs them.
