# Phase 25: TURBO_KV_4B AVX2 Kernel Design — Non-AVX-512 Microarchitecture Targets

## Status

Kernel baseline implemented; three AVX2 bugs found and fixed during propagation; PBT coverage complete over both `turbo-kv-4b.allium` and the sibling `mul_mat_cpu.allium` distilled during this phase. Profiling (Next Steps step 3) is the active step.

## Scope

Non-AVX-512 x86 CPUs with AVX2.

AVX-512-capable CPUs fall through to the scalar reference path until a dedicated AVX-512 kernel is written. This is a scope decision: the AVX-512 kernel is future work, not a claim that AVX-512 is slower than AVX2 on modern silicon. On Skylake-SP and Ice Lake-SP AVX-512 did incur measurable frequency-license throttling that could make AVX2 competitive on latency-bound code; on Zen 4 and Sapphire Rapids that penalty is largely absent.

## Spec-code alignment (prerequisite)

Before any kernel work begins, `turbo-kv-4b.allium` must be a high-fidelity description of `llama.cpp/ggml/src/ggml-turbo-kv.c`. A weed pass on 2026-04-23 (following the HQ-agent spec consolidation of `93b6ec2`) compared the spec against the scalar C reference and closed the divergences below.

### Divergences found

1. **`normalize`** — spec had unguarded `1.0 / norm`. Code guards against `norm < 1e-10` (`ggml-turbo-kv.c:256`).
2. **`QuantizeBlock`** — spec had unguarded `config.cent_max / max_abs(rotated)`. Code clamps `max_abs` to `1.0` when below `1e-10` (`ggml-turbo-kv.c:273-274`).
3. **`reconstruct_codebook` / `DequantizeBlock`** — spec had unguarded `1.0 / block.inv_std`. Code falls back to `sqrtf(dim)` when `inv_std < 1e-10` (`ggml-turbo-kv.c:311-312`).
4. **`ReconstructionPreservesNorm`** — spec claimed a universal 10 % relative-error bound. The bound is distribution-conditional; it holds for near-Gaussian post-RHT inputs (PHASE24 measured `rel=0.083` on unit Gaussian) but can be exceeded on adversarial inputs.
5. **`VectorDot` modelled the single-block primitive only** — code composes N per-block dots into one score per K position via `turbo_kv_4b_attention_multi` (`ggml-turbo-kv.c:362-399`).

### Resolutions

**Spec-side (landed in `46bf05e` and `8b179fb`):**

1. `normalize` now `requires: norm > 0.0`.
2. `QuantizeBlock` now `requires: L2_norm(tensor.data) > 0.0` — a single precondition that covers both the `normalize` entry (`norm > 0`) and the `cent_max / max_abs(rotated)` division (`max_abs > 0` follows from `L2_norm(input) > 0` via RHT orthogonality).
3. `reconstruct_codebook` and `DequantizeBlock` both `requires: block.inv_std > 0.0`.
4. `ReconstructionPreservesNorm` now carries a prose comment restricting its domain to near-Gaussian post-RHT inputs; the expression form is unchanged.
5. New rule **`MultiBlockVectorDot(query, blocks)`** models the multi-block composition, with `@guidance` documenting the equivalence to `dot_product(query, dequantize_row(blocks))` via RHT orthogonality within each block and linearity across blocks. The optimized implementation (pre-rotate query once per row, sum per-block dots in RHT-rotated space) is recorded in the `@guidance` as the intended kernel path.

**Code-side (deferred cleanup item for this phase):**

- `dequantize_block_turbo_kv_4b` at `ggml-turbo-kv.c:311-312` falls back to `inv_std = sqrtf((float) dim)` when `inv_std < 1e-10`. With the tightened spec (`requires: block.inv_std > 0.0`), this branch handles undefined-behaviour input only. The `sqrtf(dim)` value itself is unmotivated; decision: either document *why* the fallback is `sqrt(dim)` or replace with a simpler sentinel (`1.0f`, or a hard assert). Decision tracked alongside the kernel implementation (Next Steps step 1).

### Final spec state

- `allium check`: 0 errors, 0 warnings on both `turbo-kv-4b.allium` and `mul_mat_cpu.allium`.
- `allium plan`: **46 obligations total** — 41 from `turbo-kv-4b.allium` plus 5 from `mul_mat_cpu.allium` (the SIMD-equivalence spec distilled during this phase; see Implementation progress below). Combined breakdown:

  | Category | Count |
  |----------|-------|
  | entity_fields | 4 (Tensor, TurboBlock, FloatTensor, QuantizedRow) |
  | config_default | 7 (5 for turbo-kv-4b, 2 for mul_mat_cpu — `simd_equivalence_{rel,abs}_tol`) |
  | contract_signature | 1 (VecDot.invoke) |
  | rule_success | 11 |
  | rule_failure | 14 (one per `requires:` clause) |
  | rule_entity_creation | 2 |
  | invariant | 7 |
  | **Total** | **46** |

  PHASE24's "20 of 20 obligations verified" statement is superseded by the post-consolidation, post-weed, post-distill spec set. Re-verification against all 46 is folded into Next Steps (step 2).

### Intentional gaps (by design)

- **Internal zero-padding for `dim < block_size`** (`ggml-turbo-kv.c:260-262, 329-331`). The public API always passes `dim = block_size`; padding is defensive for an unreachable internal state. The spec's `requires: tensor.count = config.block_size` captures the contract at the API boundary.
- **Block-layout artifacts** (`residual_norm`, `_pad`, fp16 representation of `norm` / `inv_std`). Wire format; the spec models behaviour.
- **Multi-block `head_dim ≤ 4 * block_size` cap** (`turbo_kv_4b_attention_multi` returns early if `n_blocks > 4`). An implementation limit, not a behavioural one.

## Implementation progress

The kernel outlined below is live in the llama.cpp submodule. This section records what shipped, the bugs surfaced by the Allium lifecycle, and the regression guard that now protects future work.

### Files added/modified

- `ggml/src/ggml-cpu/arch/x86/turbo_kv_4b_avx2.h` — AVX2 inner kernel. Processes 32 elements per iteration via `_mm256_shuffle_epi8` (VPSHUFB) against a duplicated int8 codebook, with eight XMM accumulators for ILP. Replaces the previous scalar fallback on AVX2 hosts.
- `ggml/src/ggml-cpu/arch/x86/turbo_kv_4b_sse.h` — gains a `turbo_kv_4b_sse_single_block_dot` wrapper paralleling the AVX2 one, so both SIMD paths expose the same per-block API.
- `ggml/src/ggml-cpu/ggml-cpu.c` — rewritten `ggml_vec_dot_turbo_kv_4b_f32_cpu` dispatch: pre-rotate the query once per call via `turbo_kv_rotate_query`, loop per block summing `single_block_dot` outputs, arch-dispatch at the per-block granularity.
- `ggml/src/ggml.c` — adds `GGML_TYPE_TURBO_KV_4B` case to `ggml_quantize_chunk`, without which `test-backend-ops` could not construct the test tensor.
- `tests/test-backend-ops.cpp` — adds `GGML_TYPE_TURBO_KV_4B` to `all_types[]`. Primary correctness gate per the llama.cpp repo's testing CLAUDE.md.
- `tests/test-turbo-kv-pbt.cpp` — 26 PBT properties, all passing 100 RapidCheck runs each.

### Bugs surfaced by the lifecycle

All three were silent: none tripped `test-backend-ops -b CPU` (which compares the CPU backend against itself with `use_ref` toggled — doesn't bypass SIMD), and none tripped the existing test-turbo-kv-4b-attn (which compares two paths that both go through the scalar `turbo_kv_4b_single_block_dot`). Only a test that compared the AVX2 dispatch against an **independent** scalar reference (the ggml-base `ggml_vec_dot_turbo_kv_4b_f32`) caught them.

1. **Codebook buffer over-read** (submodule `588dd9bdd`). The int8 codebook table was declared `[16]` bytes but loaded via `_mm256_loadu_si256` (32-byte load). `VPSHUFB` treats each 128-bit YMM lane independently, so the upper-lane lookups sampled whatever `.bss` happened to sit after the array. On the test host those bytes were zero, so upper-lane contributions silently dropped out of every per-iteration accumulation instead of being garbage. Fix: 32-byte table with the 16 centroids duplicated.
2. **Inverted `per_block_scale`** (submodule `70036900a`). Both the SSE and AVX2 per-block wrappers computed `(127 / CENT_MAX) / inv_std` but the inner kernels expect its reciprocal `(CENT_MAX / 127) / inv_std`. The SSE kernel's own doc comment had the correct formula; the new wrappers contradicted it. Scores were inflated by `(127/CENT_MAX)² ≈ 2160×`.
3. **AVX2 upper-lane interleave** (same commit). The interleave step pulled elements 16..31 of each 32-element iter from upper-lane VPSHUFB results that contained duplicated `codebook[0]` values (because `bytes_16`'s upper lane was zero-padded). The correct lookups for all 32 elements were entirely in the lower lane — split by the low/high nibble unpack — but the code was reading the wrong half. Fix: replace `unpacklo(hi_low, hi_high)` with `unpackhi(lo_low, lo_high)`.

### Regression guard

`mul_mat_cpu.allium` was distilled during this phase (not a prerequisite, a product). Its `SIMDEquivalence` contract asserts that any registered SIMD variant must produce the same score as the scalar reference on the same inputs, within a mixed absolute/relative tolerance. Its propagated PBT property in `test-turbo-kv-pbt.cpp` (`property_SIMDEquivalence_turbo_kv_4b` + `property_SIMDEquivalence_multi_row`) runs on every build. The memory entry `feedback_simd_needs_independent_reference.md` documents why `test-backend-ops -b CPU` alone is insufficient and codifies the pattern for future SIMD kernel work.

## Target microarchitectures

| # | Microarchitecture | Vendor | Year | Notes |
|---|-------------------|--------|------|-------|
| 1 | Haswell | Intel | 2013 | AVX2 baseline; legacy server (Xeon E5-26xx v3, Xeon E3 v3). |
| 2 | Skylake client | Intel | 2015 | AVX2 only; 6th–10th gen Core consumer/mobile. Skylake-W/SP/X is excluded — those have AVX-512. |
| 3 | Alder Lake / Raptor Lake | Intel | 2021–2022 | P-core AVX-512 fused off at retail. E-cores never shipped AVX-512. Treated as AVX2-only. |
| 4 | Zen 1 / Zen+ | AMD | 2017–2018 | Full AVX2 ISA, but 128-bit internal SIMD execution decomposes 256-bit ops into two µops — effective 256-bit throughput is ~halved vs Zen 2. Ryzen 1000/2000 desktop, Threadripper 1000/2000, EPYC 7001 Naples. |
| 5 | Zen 2 | AMD | 2019 | First Zen generation with native 256-bit AVX2 execution. Ryzen 3000, Threadripper 3000, EPYC 7002. |
| 6 | Zen 3 | AMD | 2020 | Refines Zen 2. Ryzen 5000, EPYC 7003. |

**Explicitly excluded:**

- **Zen 4 (Ryzen 7000, EPYC 9004)** — has AVX-512F/BW/DQ/VL/VBMI/VBMI2/VNNI/BF16/VPOPCNTDQ/BITALG/GFNI. Covered by the AVX-512 scope decision above.
- **Skylake-W, Skylake-X, Cascade Lake, Ice Lake-SP, Sapphire Rapids, Emerald Rapids** — all AVX-512.
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
Instruction          Haswell  SKL-client  ADL/RPL  Zen 1/+  Zen 2  Zen 3
VPERM2I128               ✓         ✓          ✓        ✓       ✓      ✓
VPERMD                   ✓         ✓          ✓        ✓       ✓      ✓
VPERMQ                   ✓         ✓          ✓        ✓       ✓      ✓
VPERMPD                  ✓         ✓          ✓        ✓       ✓      ✓
VPERM2F128               ✓         ✓          ✓        ✓       ✓      ✓
VPMOVZXBW                ✓         ✓          ✓        ✓       ✓      ✓
VPMOVSXBW                ✓         ✓          ✓        ✓       ✓      ✓
VPMOVMSKB                ✓         ✓          ✓        ✓       ✓      ✓
VPSHUFB                  ✓         ✓          ✓        ✓       ✓      ✓
VPSLLVD                  ✓         ✓          ✓        ✓       ✓      ✓
VCVTDQ2PD                ✓         ✓          ✓        ✓       ✓      ✓
VEXTRACTF128             ✓         ✓          ✓        ✓       ✓      ✓
VINSERTF128              ✓         ✓          ✓        ✓       ✓      ✓
VRSQRTPS                 ✓         ✓          ✓        ✓       ✓      ✓
VBLENDVPS/PD             ✓         ✓          ✓        ✓       ✓      ✓
```

## Throughput variation — qualitative, to be quantified

Per-uarch latency, reciprocal throughput, and port-binding data must be extracted directly from Agner Fog's `instruction_tables.ods` and committed as a structured data file before any kernel decisions are made on timing grounds. Specific figures are not reproduced here — see MEMORY feedback "Follow published specs, don't riff" and the `no-riff` note in Design Constraints.

Known qualitative differences worth measuring, from published microarchitectural descriptions (not from invented numbers):

- **Zen 1 / Zen+** decomposes every 256-bit AVX2 op into two 128-bit µops; effective YMM throughput is systematically ~halved relative to Zen 2+. This is the tightest target in scope and the most likely to drive a uarch-specific variant (e.g. an XMM-width path) if measurement demands one.
- **VPSLLVD** reciprocal throughput improved materially between Zen 2 and Zen 3. If bit-unpack is a hot path, this is the most likely trigger for a Zen 2-specific variant.
- **VPERMD / VPERMQ** are lane-crossing; they are the RHT bottleneck on all targets and dominate latency regardless of uarch.
- **VRSQRTPS** reciprocal throughput differs between Intel client and Zen.
- **VBLENDVPS/PD** reciprocal throughput is worse on Haswell than on later Intel or Zen.

All of these are throughput differences, not capability differences.

## Kernel strategy

**Baseline: a single AVX2 kernel covering all 6 targets.**

**Rationale:** the ISA is uniform. A single compiled kernel emits the same instructions on every target. A multi-kernel split would add maintenance cost without fixing a concrete problem.

**What this design does not include ahead of measurement:**

- No Zen 1/+ 128-bit XMM-width variant, even though 256-bit ops are split internally on that uarch.
- No Zen 2-specific VPSLLVD-avoidance variant until VPSLLVD is measured as the bottleneck.
- No "VPAND-optimized path" — VPAND/VPANDN are SSE2 (2001) and part of the baseline kernel on every target.
- No emulation paths — all 15 instructions are natively available everywhere in scope.

**Conditional second kernel:** measurement happens on one host only — this Zen 2 machine. Performance on the other 5 targets is extrapolated from Agner Fog tables for the hot instructions identified on Zen 2. If that extrapolation predicts a target (most likely Zen 1/+ due to 256-bit decomposition) would fall below the performance goal, introduce a variant kernel at that point. No speculation ahead of measurement, but also no per-target profiling rig.

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
  - Zen 1: family `0x17`, models `0x01` (Summit Ridge desktop, Naples EPYC), `0x11` (Raven Ridge APU), `0x20` (Dali/Pollock)
  - Zen+: family `0x17`, models `0x08` (Pinnacle Ridge desktop, Colfax Threadripper), `0x18` (Picasso APU)
  - Zen 2: family `0x17`, models include `0x31`, `0x47`, `0x60`, `0x68`, `0x71`, `0xA0`
  - Zen 3: family `0x19`, models `0x00`–`0x0F`, `0x20`–`0x2F`, `0x40`–`0x4F`, `0x50`–`0x5F`
  - Zen 4: family `0x19`, models `0x10`–`0x1F`, `0x60`–`0x6F`, `0x70`–`0x7F`, `0xA0`–`0xAF` (excluded via the AVX-512F bit)

## Design constraints

1. **Test-first:** the kernel must satisfy all 41 Allium obligations generated from the aligned spec (see Spec-code alignment above) before it ships. No deferrals.
2. **Surgical changes:** match existing `ggml-turbo-kv.c` style exactly.
3. **No workarounds:** if a target needs something this design did not anticipate, implement it. Do not downgrade types or hack around it.
4. **No riff on tables:** actual Agner Fog figures go into a separate committed data file with row citations. Do not quote timing numbers from memory.
5. **GPU testing (repo rule, noted for context):** GPU tests use Vega only (`GGML_VK_VISIBLE_DEVICES=1`); does not apply to CPU kernels but flagged here so it doesn't fall off the checklist at integration time.

## Next steps

1. **DONE** — Single-kernel AVX2 baseline implemented in `llama.cpp/ggml/src/ggml-cpu/arch/x86/turbo_kv_4b_avx2.h` + dispatch in `ggml-cpu/ggml-cpu.c`. Landed across four llama.cpp submodule commits (`0d7f16ffe`, `4a0297f56`, `588dd9bdd`, `70036900a`); the three bugs found along the way are documented in the Implementation progress section above. **Open sub-item:** the `sqrtf(dim)` fallback at `ggml-turbo-kv.c:311-312` is still present and still unmotivated — tracked but not resolved here. The tightened `requires: block.inv_std > 0.0` in the spec makes it dead code under valid input; either replace with `1.0f` or leave the defensive behaviour with a comment explaining what it's defensive against. Low urgency.
2. **DONE** — PBT coverage against both specs. 26 properties in `tests/test-turbo-kv-pbt.cpp` directly test 30 of 46 obligations; 2 more are structurally implied; 14 `rule_failure.*` obligations are consciously skipped per the `feedback_simd_needs_independent_reference` memory entry (defensive clamps outside the spec contract). All 26 × 100 RapidCheck runs pass every build. `test-backend-ops -b CPU -o MUL_MAT` also passes 1113/1113 including 10 `turbo_kv_4b` cases — but that gate alone would not have caught the three kernel bugs (it compares CPU backend against itself with `use_ref` toggled, which doesn't bypass SIMD).
3. **Active** — Profile the baseline on this Zen 2 host. Identify which of the 15 core instructions from the Instruction availability matrix actually dominate cycle count for the `turbo_kv_4b_avx2_single_block_dot` hot path. Expected candidates based on the kernel's structure: `_mm256_shuffle_epi8` (codebook lookup, once per 32 elements), `_mm_cvtepi8_epi32` (int8 → int32 conversion, 8× per iteration), `_mm_add_ps` / `_mm_mul_ps` (FMA-equivalent accumulation). Expect 2-4 instructions to account for >80% of cycles.
4. For the instructions flagged hot in step 3 (and only those), extract per-uarch latency, reciprocal throughput, and port-binding data from Agner Fog `instruction_tables.ods` for all 6 targets. Commit as a structured data file alongside this doc, with source row numbers cited.
5. Combine the Zen 2 hot-path counters with the Agner figures to extrapolate expected per-target performance. Introduce a variant kernel only if the extrapolation predicts a target would fall below the goal.

## Notes

**Zen 1 / Zen+ is the floor for performance targeting.** If the baseline kernel meets latency goals on Zen 1/+, it will meet them on every other target. Conversely, if measurement shows the 256-bit-split penalty dominates, the natural response is an XMM-width variant for that uarch rather than per-instruction workarounds.
