# Phase 25: TURBO_KV_4B AVX2 Kernel Design — Non-AVX-512 Target Architecture Matrix

## Status: COMPLETE — design document, no implementation yet

## Problem

Design the AVX2 kernel strategy for TURBO_KV_4B quantization across non-AVX-512 microarchitectures, using Agner Fog's instruction tables as the authoritative reference. Determine kernel splits, lowest-common-denominator instruction sets, and fallback paths.

**Scope note:** AVX-512 CPUs are excluded from this kernel scope. On AVX-512 CPUs, the scalar reference path or a dedicated AVX-512 kernel (future work) is used. The rationale: AVX-512 has known limitations (frequency throttling, state transition overhead, thermal constraints) that can make AVX2 faster in practice on some workloads. TURBO quantization is latency-bound with long dependency chains, making AVX-512's frequency penalty particularly harmful.

## Target Microarchitectures

Goldmont excluded (Atom-based, limited AVX2, not LLM hosting tier). Zen 1 excluded (too limited AVX2 — only ~10 AVX2 instructions).

| # | Microarchitecture | Vendor | Year | AVX2 | LLM Hosting Relevance |
|---|-------------------|--------|------|------|----------------------|
| 1 | Haswell | Intel | 2013 | Baseline | Legacy server (E5-26xx v3) |
| 2 | Skylake-W | Intel | 2017 | Full | Workstation (Xeon W) |
| 3 | Zen 2 | AMD | 2019 | Full | Desktop/Threadripper 2 |
| 4 | Zen 3 | AMD | 2020 | Full | Desktop/EPYC 7002 |
| 5 | Zen 4 | AMD | 2022 | Full + extras | EPYC 9004, Ryzen 7000, PCIe 4.0, better memory bandwidth |

**Why Zen 4 as the fourth target:**
- Introduces **VPAND/VPANDN** — dedicated bitwise AND/NAND, useful for TURBO bit manipulation (replaces multi-instruction sequences)
- **VPSLLVD** throughput is 6× faster than Zen 2 (rtp=0.5 vs rtp=3) — important for bit unpack in quantization
- Represents modern AMD server class with PCIe 4.0 and improved memory controller — direct LLM hosting relevance
- Lacks AVX-512, keeping it in scope
- Significant microarchitectural gap from Zen 2 (different IPC, improved scheduler, more execution ports)

## Instruction Availability Matrix

Extracted from Agner Fog's instruction_tables.ods for 15 TURBO quantization core instructions across 5 target uarchs.

### Core instruction set

| # | Instruction | Role in TURBO |
|---|-------------|---------------|
| 1 | VPERM2I128 | RHT butterfly lane swap |
| 2 | VPERMD | RHT sign mask application (32-bit shuffle) |
| 3 | VPERMQ | RHT butterfly (64-bit shuffle) |
| 4 | VPERMPD | RHT butterfly (64-bit permute) |
| 5 | VPERM2F128 | 128-bit lane permute |
| 6 | VPMOVZXBW | Bit unpack: 8-bit indices → 16-bit |
| 7 | VPMOVSXBW | Sign-extend unpack |
| 8 | VPMOVMSKB | Extract bitmask from YMM register |
| 9 | VPSHUFB | Table-driven byte shuffle (codebook lookup) |
| 10 | VPSLLVD | Variable left shift (bit unpack) |
| 11 | VCVTDQ2PD | int32 → float64 (norm computation) |
| 12 | VEXTRACTF128 | Extract XMM from YMM |
| 13 | VINSERTF128 | Insert XMM into YMM |
| 14 | VRSQRTPS | Reciprocal square root (normalization) |
| 15 | VBLENDVPS/PD | Conditional blend (quantization clamping) |

### Availability matrix

```
Instruction                  Haswell  SKL-W    Zen 2    Zen 3    Zen 4
VPERM2I128                      ✓        ✓        ✓        ✓        ✓
VPERMD                          ✓        ✓        ✓        ✓        ✓
VPERMQ                          ✓        ✓        ✓        ✓        ✓
VPERMPD                         ✓        ✓        ✓        ✓        ✗
VPERM2F128                      ✓        ✓        ✓        ✓        ✓
VPMOVZXBW                       ✓        ✓        ✓        ✓        ✓
VPMOVSXBW                       ✓        ✓        ✓        ✓        ✓
VPMOVMSKB                       ✓        ✓        ✓        ✓        ✓
VPSHUFB                         ✓        ✓        ✗        ✗        ✓
VPSLLVD                         ✓        ✓        ✓        ✓        ✓
VCVTDQ2PD                       ✓        ✓        ✓        ✓        ✓
VEXTRACTF128                    ✓        ✓        ✓        ✓        ✓
VINSERTF128                     ✓        ✓        ✓        ✓        ✓
VRSQRTPS                        ✓        ✓        ✓        ✓        ✓
VBLENDVPS/PD                    ✓        ✓        ✗        ✗        ✗
```

### Lowest Common Denominator (LCD) analysis

**3/5 uarchs** (Haswell, Skylake-W, Zen 4) share all 15 instructions.

**1/5 uarchs** (Zen 3) shares 13/15 (missing VPSHUFB, VBLENDVPS/PD).

**1/5 uarchs** (Zen 2) shares 13/15 (missing VPSHUFB, VBLENDVPS/PD).

**Missing instructions and their emulation paths:**

| Instruction | Missing on | Emulation | Cost multiplier |
|-------------|-----------|-----------|-----------------|
| VPERMPD | Zen 4 | VPERMQ + VPERM2I128 | 2× μops |
| VPSHUFB | Zen 2, Zen 3 | VPERM2I128 + VPERMD chain | 3-4× μops |
| VBLENDVPS/PD | Zen 2, Zen 3, Zen 4 | VPERM2I128 + VPERMD + VPERMQ | 3-4× μops |

**Zen 2/3 are the constraint boundary.** They lack VPSHUFB and VBLENDVPS/PD. Haswell, Skylake-W, and Zen 4 have the complete set.

## Kernel Split Strategy

### Kernel 1: Zen 2/3 baseline (AVX2 constrained)

**Target:** Zen 2, Zen 3 — the most constrained instruction set.

**Instruction set:** 13 instructions (LCD minus VPSHUFB, VBLENDVPS/PD, VPERMPD).

**Emulation paths needed:**
- **VPSHUFB** → VPERM2I128 + VPERMD chain (~4 μops vs 1)
- **VBLENDVPS/PD** → VPERM2I128 + VPERMD + VPERMQ chain (~4 μops vs 1)
- **VPERMPD** → VPERMQ + VPERM2I128 (2 μops vs 1)

**Rationale:** Zen 2/3 have the most limited AVX2 instruction set. They can execute all other TURBO instructions natively. The emulation paths for VPSHUFB and VBLENDVPS/PD use permutation instructions that Zen 2/3 do have. The emulation cost is ~3-4× per instruction but the kernels remain correct.

### Kernel 2: Haswell/Skylake-W/Zen 4 (AVX2 full LCD)

**Target:** Haswell, Skylake-W, Zen 4 — all share the full 15-instruction LCD set.

**Instruction set:** All 15 core instructions.

**Additional useful instructions on Zen 4:**
- **VPAND / VPANDN** — bitwise AND/NAND (replaces multi-instruction bit-mask sequences in quantization)
- **VPMADDUBSW** — dot product accumulation (attention path, future)
- **VPOPCNTD** — population count (bit counting, future)
- **VPMIN / VPMAX** — integer min/max (quantization clamping, future)

**Rationale:** These three uarchs share the full LCD instruction set. They differ in timing characteristics (see below) but not in instruction availability. The kernel can detect Zen 4 at runtime and enable VPAND-optimized paths within the same kernel.

**Why these three together:** Despite their generational differences (Haswell 2013 → Zen 4 2022), they share the same AVX2 instruction set boundary. The timing differences can be handled via runtime dispatch within the kernel, not via separate kernels.

## Timing Comparison — Key Instructions

All data from Agner Fog's instruction tables.

```
Instruction                  Haswell      Skylake      Zen 2        Zen 3        Zen 4
VPERM2I128              lat=3  rtp=1  P2    lat=3  rtp=1  P2    lat=3  rtp=1  P2    lat=3  rtp=1  P2    lat=3  rtp=1  P2
VPERMD                  lat=8  rtp=1  P12   lat=8  rtp=1  P12   lat=8  rtp=1  P12   lat=8  rtp=1  P12   lat=8  rtp=1  P12
VPERMQ                  lat=6  rtp=1  P12   lat=6  rtp=1  P12   lat=6  rtp=1  P12   lat=6  rtp=1  P12   lat=6  rtp=1  P12
VPMOVZXBW               lat=4  rtp=1  P12   lat=4  rtp=1  P12   lat=4  rtp=1  P12   lat=4  rtp=1  P12   lat=4  rtp=1  P12
VPMOVSXBW               lat=4  rtp=1  P12   lat=4  rtp=1  P12   lat=4  rtp=1  P12   lat=4  rtp=1  P12   lat=4  rtp=1  P12
VPSLLVD                 lat=1  rtp=2  P0 P1 lat=1  rtp=1  P01   lat=1  rtp=3  P12   lat=1  rtp=0.5 P12 lat=1  rtp=0.5 P23
VRSQRTPS                lat=3  rtp=0.5 P01 lat=3  rtp=0.5 P01   lat=3  rtp=1  P01   lat=3  rtp=1  P01   lat=3  rtp=1  P01
VBLENDVPS/PD            lat=2  rtp=2  P5    lat=1  rtp=1  P01   N/A          N/A          N/A          lat=1  rtp=0.5 P01
VPMOVMSKB               lat=5  rtp=2  P2    lat=5  rtp=2  P2    lat=5  rtp=2  P2    lat=5  rtp=2  P2    lat=5  rtp=2  P2
```

**Key timing observations:**
- **VPSLLVD** is the most timing-divergent instruction: Zen 2 (rtp=3, P12) vs Zen 3/4 (rtp=0.5, P12) vs Skylake (rtp=1, P01). This is a critical differentiator — bit unpack is 6× slower on Zen 2.
- **VRSQRTPS** has 2× throughput difference: Skylake (rtp=0.5, P01) vs Zen 2/3/4 (rtp=1, P01). Haswell also rtp=0.5 but on P01.
- **VPERMD** has 8-cycle latency on all uarchs — this is the RHT bottleneck, same across all targets.
- **VPMOVZXBW/VPMOVSXBW** share ports P12 across all uarchs — throughput-limited, not latency-bound.
- **VBLENDVPS/PD** is 4× faster on Skylake (rtp=0.5) vs Haswell (rtp=2) — matters for quantization clamping path.
- **Zen 4 VPSLLVD** (rtp=0.5, P23) is the fastest — enables faster bit unpack in Zen 4.

## Runtime Detection Strategy

```c
// CPUID-based detection
bool cpu_has_avx2 = cpuid(7, 0).ebx & (1 << 5);
bool cpu_has_avx512f = cpuid(7, 0).ebx & (1 << 16);
bool cpu_has_avx512bw = cpuid(7, 0).ebx & (1 << 30);

// Exclude AVX-512 CPUs — use scalar ref or future AVX-512 kernel
bool cpu_has_avx512 = cpu_has_avx512f && cpu_has_avx512bw;
if (cpu_has_avx512) return scalar_quantize_row_turbo_kv_4b(...);

// AMD-specific: Zen generation via leaf 0x80000001
uint32_t family = (cpuid(0x80000001, 0).eax >> 8) & 0xF;
uint32_t ext_family = (cpuid(0x80000001, 0).eax >> 20) & 0xFF;
uint32_t model = (cpuid(0x80000001, 0).eax >> 4) & 0xF;

// Zen 2 = family 0x17, model >= 0x70
// Zen 3 = family 0x19
// Zen 4 = family 0x1A

// Intel: Haswell = family 6, model 0x45/0x4F; Skylake = family 6, model 0x5E
```

**Kernel selection priority:**
1. **AVX-512 detected** → scalar reference path (or future AVX-512 kernel)
2. **VPAND detected** (Zen 4) → Kernel 2 with VPAND-optimized paths
3. **VPSHUFB detected** (Haswell, Skylake-W) → Kernel 2 without emulation
4. **AVX2 only** (Zen 2/3) → Kernel 1 with emulation paths

## Design Constraints

1. **Test-first discipline:** Every kernel must pass the 20 Allium obligations from PHASE24. No deferrals.
2. **GPU testing:** All GPU tests use only Vega (`GGML_VK_VISIBLE_DEVICES=1`).
3. **No workarounds:** If an instruction is missing, implement it properly — no type downgrades or hacks.
4. **Surgical changes:** Match existing ggml-turbo-kv.c style exactly. Touch only what changes.
5. **Universal where possible:** Kernel 2 covers 3/5 uarchs with the full LCD. Kernel 1 covers the remaining 2/5 with emulation.

## Next Steps

1. Implement Zen 2/3 baseline kernel (Kernel 1) with emulation paths for VPSHUFB and VBLENDVPS/PD
2. Implement Haswell/Skylake-W/Zen 4 kernel (Kernel 2) using full LCD + VPAND-optimized paths for Zen 4
3. Add runtime CPU detection and kernel dispatch in ggml-turbo-kv.c
4. Verify all 20 Allium obligations against each kernel via test-backend-ops

## Notes

- **AVX-512 excluded from scope:** AVX-512 CPUs use the scalar reference path. A dedicated AVX-512 kernel (ZMM registers, 64 elements/instruction) is future work. Reasons: AVX-512 frequency throttling, state transition overhead, and thermal constraints can make AVX2 faster in practice on latency-bound workloads like TURBO quantization.
- **Zen 1 excluded:** Too limited AVX2 support (only ~10 AVX2 instructions).
- **Goldmont excluded:** Atom-based, not LLM hosting tier.
- **The 2-kernel strategy** balances coverage (5 uarchs) against implementation complexity (2 kernels).
- **Zen 4's VPAND/VPANDN** are the most impactful new instructions — they replace multi-instruction bit-mask sequences used in quantization index packing.
