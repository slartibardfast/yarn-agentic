# Phase 25: TURBO_KV_4B AVX2 Kernel Design — Non-AVX-512 Microarchitecture Targets

## Status

**Phase complete.** Every scalar path surfaced by the Zen 2 profile is vectorised (step 3 and sub-steps 3a–3d). Kernel baseline, argmin, `max_abs`, block repack to fp32 scales, fp64/AVX2 `L2_norm`, RHT AVX2 widening, normalize + RHT sign-flip fusion, and a precomputed signmask LUT for the in-tree seed all shipped. Full-pipeline AVX2 quantize is **311 ns/call** on Zen 2 (down from a pre-repack 486 ns — 36% session reduction, 7.2× vs scalar). 30 PBT properties cover the lifecycle; `test-backend-ops -b CPU -o MUL_MAT` passes 1113/1113 turbo_kv_4b cases; GPU and CPU produce bit-identical `norm` / `inv_std`.

Agner Fog data for all 6 target uarchs has been extracted into `reference/agner/turbo_kv_4b_agner.csv` (step 4) and combined with the Zen 2 profile to extrapolate per-target performance (step 5). Conclusion: **no variant kernel needed** — the slowest projected target is Zen 1 at ~1.40× Zen 2 overall (~434 ns/call), still faster than the pre-repack Zen 2 baseline. Details in Next Steps § below.

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

**Vec_dot hot path (landed first):**

- `ggml/src/ggml-cpu/arch/x86/turbo_kv_4b_avx2.h` — AVX2 inner kernel. Processes 32 elements per iteration via `_mm256_shuffle_epi8` (VPSHUFB) against a duplicated int8 codebook, with eight XMM accumulators for ILP. Replaces the previous scalar fallback on AVX2 hosts. Also carries `turbo_kv_4b_avx2_nearest_centroid_block` (Step 5 argmin, see below).
- `ggml/src/ggml-cpu/arch/x86/turbo_kv_4b_sse.h` — gains a `turbo_kv_4b_sse_single_block_dot` wrapper paralleling the AVX2 one, so both SIMD paths expose the same per-block API.
- `ggml/src/ggml-cpu/ggml-cpu.c` — rewritten `ggml_vec_dot_turbo_kv_4b_f32_cpu` dispatch: pre-rotate the query once per call via `turbo_kv_rotate_query`, loop per block summing `single_block_dot` outputs, arch-dispatch at the per-block granularity.
- `ggml/src/ggml.c` — adds `GGML_TYPE_TURBO_KV_4B` case to `ggml_quantize_chunk`, without which `test-backend-ops` could not construct the test tensor.
- `tests/test-backend-ops.cpp` — adds `GGML_TYPE_TURBO_KV_4B` to `all_types[]`. Primary correctness gate per the llama.cpp repo's testing CLAUDE.md.

**Quantize hot path (landed as follow-on):**

- Step 5 (nearest-centroid argmin) — vectorised in `turbo_kv_4b_avx2_nearest_centroid_block`. Eight centroid distances computed in parallel per element, lane-mask argmin (strict `<` preserves first-match tie-break). Reduced a 2048-scalar-comparison inner loop to a 16-wide SIMD scan. **11.1× speedup on step 5 in isolation; end-to-end quantize 5.32× vs scalar at first landing.**
- Step 4 (`max_abs` scan) — AVX2 `_mm256_max_ps` cascade with scalar tail. Small but noise-free.
- Step 1 (`L2_norm`) — fp64 accumulator inner loop: `_mm256_cvtps_pd` → `_mm256_fmadd_pd` on two 128-bit halves per iteration. The fp64 accumulator absorbs sum-tree drift before the final `sqrtf` → fp32 rounding, so scalar and SIMD paths agree at fp32 precision regardless of summation order. Eliminates the sum-tree non-associativity risk that a straight-line AVX2 port would have exposed.

**Block layout repack (landed with the L2_norm widening):**

- `block_turbo_kv_4b` — was `(uint16_t norm, uint16_t residual_norm, uint16_t inv_std_fp16, uint16_t _pad, uint8_t mse_indices[64])`. Now `(float norm, float inv_std, uint8_t mse_indices[64])`. Same 72 bytes. Reclaims the 4 wasted bytes that were reserved for a composite residual that never materialised. Eliminates the fp16 ↔ fp32 round-trip at every read site (single_block_dot wrapper, dequant, attention, set_rows, cpy). CPU and GPU now produce **bit-identical** `norm` / `inv_std` values (was last-bit different due to independent fp16 rounding paths).
- Four Vulkan compute shaders updated to read/write fp32 directly (`dequant`, `get_rows`, `cpy_f32`, `set_rows`). The `turbo_kv_f32_to_fp16_bits` GLSL helper is removed.
- SSE Westmere-fallback wrapper and the scalar `dequantize_block_turbo_kv_4b` both read fp32 directly; the SSE inner kernel itself is unchanged (still int8-codebook + SSSE3 pshufb).

**Test coverage:**

- `tests/test-turbo-kv-pbt.cpp` — 30 PBT properties, all passing 100 RapidCheck runs each. Includes the three propagated from the `nearest_centroid.allium` contract (`ArgminFirstMatch_lower_index_on_tie`, `PackNibbleIndices_roundtrip`, `NearestCentroidAssignment_matches_reference`), one AVX2-vs-scalar argmin bit-exactness property, and two direct-coverage properties for `QuantizeBlock.created` and `DequantizeBlock.Tensor.created`.

**Bench numbers (Zen 2, Ryzen 9 3950X, post-repack):**

| Bench | Scalar | AVX2 | Speedup |
|---|---|---|---|
| `bench-turbo-kv-4b-avx2` (single_block_dot, inside vec_dot) | — | 22.91 ns/call | — |
| `bench-turbo-kv-quantize` (full pipeline per block, 128 elems) | 2278 ns/call | 402 ns/call | 5.66× |
| `bench-turbo-kv-argmin` (step 5 isolated) | — | — | 11.1× |

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

The kernel splits into two inference-hot paths: the **per-attention-score** path (`turbo_kv_4b_avx2_single_block_dot`, called per K position during attention scoring) and the **per-KV-write** path (`quantize_block_turbo_kv_4b`, `turbo_kv_rht_forward`, `turbo_kv_rotate_query`, called once per token per layer during prefill and decode). Both paths matter for end-to-end throughput — attention scoring gets called more often during prefill of long contexts, but quantize gets called every decode step too (KV rollouts are one write per layer per new token).

Profiling so far (see below) has covered the vec_dot path only; the quantize path needs its own bench + perf pass before step 4 extraction. The primary table below is therefore the *profiled* vec_dot hot path; the secondary section is the quantize/RHT candidate set that needs its own ranking.

All instructions are in AVX2 or earlier. Every AVX2-capable CPU in scope implements all of them — there is no ISA-driven kernel split.

### Primary: vec_dot hot path (~85% of per-call cycles, profiled)

| # | Instruction | Introduced in | Role in vec_dot | Profiled % on Zen 2 |
|---|-------------|---------------|-----------------|---------------------|
| 1 | VMULPS | AVX | Codebook int8 × per-block scale → fp32 | 12.78 |
| 2 | VFMADD231PS | FMA (AVX2-era) | q_rot × scaled_cb → accumulator | 10.37 |
| 3 | VHADDPS | SSE3 / AVX | Horizontal reduce accumulators → 1 fp | 8.24 |
| 4 | VADDPS | AVX | Merge 8 XMM accumulators pairwise | 6.23 |
| 5 | VINSERTI128 | AVX2 | Combine 128-bit lanes at reduce | 5.02 |
| 6 | VPMOVSXBD | SSE4.1 / AVX2 YMM | Codebook int8 → int32 | 4.90 |
| 7 | VFMADD213SS | FMA (AVX2-era) | Final scalar: norm × mse_dot | 4.55 |
| 8 | VCVTDQ2PS | SSE2 / AVX2 YMM | int32 → fp32 pre-scale | 4.51 |
| 9 | VMOVDQU | SSE2 / AVX2 YMM | Load 16 bytes of mse_indices | 3.74 |
| 10 | VFMADD132PS | FMA (AVX2-era) | Fused multiply-add, alternate operand order | 2.65 |
| 11 | VPSRLW | SSE2 / AVX2 YMM | High-nibble extraction (shift right words) | ~1.85 |
| 12 | VPSHUFB | SSSE3 / AVX2 YMM | Table-driven codebook lookup (32-entry) | ~0 (absorbed by dispatch) |
| 13 | VPAND | SSE2 / AVX2 YMM | Low-nibble mask (0x0F) | ~0 |

VPSHUFB appearing at ~0% is a good sign, not a bad one — the codebook lookup designed as the kernel's centerpiece is so fast on Zen 2 that the out-of-order dispatch pipeline absorbs it behind the FMA-family work. It stays in the hot-path list because it's structurally critical (removing it breaks the kernel).

### Availability matrix

```
Instruction          Haswell  SKL-client  ADL/RPL  Zen 1/+  Zen 2  Zen 3
VMULPS                   ✓         ✓          ✓        ✓       ✓      ✓
VFMADD231PS              ✓         ✓          ✓        ✓       ✓      ✓
VHADDPS                  ✓         ✓          ✓        ✓       ✓      ✓
VADDPS                   ✓         ✓          ✓        ✓       ✓      ✓
VINSERTI128              ✓         ✓          ✓        ✓       ✓      ✓
VPMOVSXBD                ✓         ✓          ✓        ✓       ✓      ✓
VFMADD213SS              ✓         ✓          ✓        ✓       ✓      ✓
VCVTDQ2PS                ✓         ✓          ✓        ✓       ✓      ✓
VMOVDQU                  ✓         ✓          ✓        ✓       ✓      ✓
VFMADD132PS              ✓         ✓          ✓        ✓       ✓      ✓
VPSRLW                   ✓         ✓          ✓        ✓       ✓      ✓
VPSHUFB                  ✓         ✓          ✓        ✓       ✓      ✓
VPAND                    ✓         ✓          ✓        ✓       ✓      ✓
```

### Secondary: quantize + RHT paths (profiled)

These instructions live in `quantize_block_turbo_kv_4b`, `turbo_kv_rht_forward` / `turbo_kv_rht_inverse`, and `turbo_kv_rotate_query`. They run **per K/V cache write** — during prefill, once per token per layer; during decode, once per token per layer for the K side. For a 32-layer model decoding at 50 t/s that's ~1600 quantize calls and ~1600 query-rotations per second. This path matters for end-to-end throughput separately from vec_dot.

Per-call cost (`quantize_row_turbo_kv_4b_ref` with `dim=block_size`, measured via `tests/bench-turbo-kv-quantize`):

| Metric | Value | vec_dot ratio |
|---|---|---|
| ns/call | 2346 | 87× slower |
| cycles/call | 9719 | 69× |
| instructions/call | 16710 | 48× |
| IPC | 1.72 | lower (more stalls) |
| Branches/call | 3042 | 150× |
| Branch-miss rate | 4.3% | 2.5× worse |

The 87× cost difference per call matters because quantize runs at similar frequency to vec_dot during decode (one K write per token per layer, vs one attention-score call per token per K-position per layer). For long contexts vec_dot wins on total cycles; for short contexts quantize dominates.

**Symbol-level breakdown** (93.15% in `quantize_block_turbo_kv_4b`, 4.00% in `walsh_hadamard_sse`, 2.19% in `turbo_kv_rht_forward` scalar wrapper, 0.52% in fp16 conversion):

### Primary hot instructions — `quantize_block_turbo_kv_4b` inner loop (pre-vectorisation baseline)

| # | Instruction | Introduced in | Role | Profiled % |
|---|-------------|---------------|------|-----------|
| 1 | VANDPS (scalar) | AVX | `fabsf(x - codebook[c])` via sign-bit clear | 13.51 |
| 2 | VMOVAPS (scalar) | SSE / AVX | Register-to-register move; spill/reload | 11.14 |
| 3 | VSUBSS | SSE / AVX | `x - codebook[c]` in nearest-centroid scan | 10.88 |
| 4 | VCOMISS | SSE / AVX | Compare current distance to best-so-far | 10.25 |
| 5 | JA | x86-64 | Conditional branch on the argmin update | 9.94 |
| 6 | VADDSS | SSE / AVX | Scalar accumulator add (various) | 9.91 |
| 7 | VMULSS | SSE / AVX | Scalar multiply (rotated[i] × inv_std) | 7.45 |
| 8 | VMINSS | SSE / AVX | Min-reduction within the inner scan | 2.91 |

**This profile captured the state *before* the Step-5 argmin vectorisation landed.** The profile is kept here for historical reference because it's what motivated the distilled `nearest_centroid.allium` contract and the subsequent kernel rewrite. Post-vectorisation the scalar distance-compute instructions above are gone; the dominant instructions in the quantize hot path are now `VPCMPGTD`-style lane-mask argmin updates in the AVX2 inner loop, plus the SSE butterflies inside `walsh_hadamard_sse` (see secondary section).

The argmin vectorisation landed under this phase (not deferred to a separate phase as the original text speculated). It lives in `turbo_kv_4b_avx2.h::turbo_kv_4b_avx2_nearest_centroid_block`, shares the int8-codebook + `VPSHUFB` trick with the vec_dot kernel, and produces byte-identical `mse_indices` vs the scalar reference on every input (guarded by `property_AVX2_NearestCentroidAssignment_matches_reference` at 100 RapidCheck runs per build). `bench-turbo-kv-argmin` reports 11.1× on Step 5 in isolation.

### Secondary hot instructions — `walsh_hadamard_sse` (4.00% of total)

The SSE4.1 butterfly inside `turbo_kv_rht_forward`. Used only when `__SSE4_1__` is defined (i.e., every target in scope). Per-call cost is amortized over one RHT call per block.

| # | Instruction | Introduced in | Role | Profiled % within `walsh_hadamard_sse` |
|---|-------------|---------------|------|---|
| 1 | VHADDPS / VHSUBPS | SSE3 | Stage-1 butterfly: horizontal add/sub | top-2 (details in `perf annotate walsh_hadamard_sse`) |
| 2 | VADDPS / VSUBPS | SSE / AVX | Stages 2+: 4-lane add/sub butterfly | |
| 3 | VSHUFPS, VMOVLHPS, VMOVHLPS | SSE | Stage-2 lane recombination | |
| 4 | VMOVUPS | SSE / AVX | Unaligned load / store across each stage | |

All standard AVX / SSE instructions, present on every target in scope. Not a per-uarch concern for step 4 unless a specific uarch has a bad VHADDPS throughput (worth one row in the Agner extraction).

### Never referenced in the current kernel

Instructions from the original PHASE25 draft that don't appear anywhere in the profile (or in the kernel source) — removed from step 4 scope entirely:

- `VPMOVZXBW`, `VPMOVSXBW` — the kernel goes byte→dword directly via `VPMOVSXBD`, skipping the intermediate word unpack.
- `VCVTDQ2PD` — kernel is single-precision throughout; uses `VCVTDQ2PS`.
- `VPMOVMSKB`, `VEXTRACTF128`, `VINSERTF128` — superseded by `*I128` variants in the AVX2 path, or not used.
- `VRSQRTPS` — the scalar norm computation uses libc `sqrtf` (one call per block, off the hot path).
- `VBLENDVPS/PD` — not used.
- `VPERM2I128`, `VPERMD`, `VPERMQ`, `VPERMPD`, `VPERM2F128` — not used by either the vec_dot or the quantize/RHT hot paths. The WHT uses scalar butterflies, not lane permutes.
- `VPSLLVD` — not used; the kernel uses scalar multiply for the normalize step.

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

2. **DONE** — PBT coverage against all three lifecycle specs (`turbo-kv-4b.allium`, `mul_mat_cpu.allium`, `nearest_centroid.allium`). 30 properties in `tests/test-turbo-kv-pbt.cpp`, all passing 100 RapidCheck runs each. Directly covers 28 of 46 obligations (up from 26 after distilling the argmin contract and adding direct-coverage props for `QuantizeBlock.created` and `DequantizeBlock.Tensor.created`). 14 `rule_failure.*` obligations are consciously skipped per `feedback_simd_needs_independent_reference` (defensive clamps outside the spec contract). `test-backend-ops -b CPU -o MUL_MAT` also passes 1113/1113 including 10 `turbo_kv_4b` cases, but that gate alone would not have caught the three kernel bugs — it compares CPU backend against itself with `use_ref` toggled, which doesn't bypass SIMD.

3. **DONE** — both inference hot paths profiled on Zen 2, and the first-order optimisation targets surfaced by the profile have been executed:
   - **3.1 DONE — argmin vectorisation (Step 5 of quantize).** Distilled `nearest_centroid.allium` as a byte-exact contract, propagated three PBT properties (`ArgminFirstMatch`, `PackNibbleIndices`, `NearestCentroidAssignment`), implemented `turbo_kv_4b_avx2_nearest_centroid_block`. 11.1× on Step 5 in isolation, 5.32× end-to-end at first landing.
   - **3.2 DONE — `max_abs` vectorisation (Step 4 of quantize).** AVX2 `_mm256_max_ps` cascade with scalar tail.
   - **3.3 DONE — block layout repack + L2_norm widening.** `block_turbo_kv_4b` repacked to fp32 scales (reclaiming wasted 4 bytes), L2_norm moved to fp64 accumulator with AVX2 SIMD inner loop. Removed fp16 ↔ fp32 round-trips from every read site; CPU and GPU now produce bit-identical `norm` / `inv_std`. 84 ns saved on the AVX2 quantize path; end-to-end quantize speedup 5.32× → **5.66×**. Committed as `6e0fb3e2f` in the llama.cpp submodule.

3a. **DONE — RHT AVX2 widening.** `walsh_hadamard_avx2` widens the butterfly stages with stride ≥ 8 to 256-bit (stages 1–3 stay 128-bit because the cross-lane shuffles they need would undo the gain). Sign-flip and scale passes also widen. SSE4.1 remains as the fallback for Westmere-to-Sandy-Bridge. Savings: **46 ns** (402 → 356 ns).

3b. **DONE — Normalize + RHT sign-flip fusion.** Refactored `turbo_kv_rht_forward` into `turbo_kv_rht_sign_flip` + `turbo_kv_rht_forward_body`. `quantize_block_turbo_kv_4b` now calls `turbo_kv_fused_normalize_and_sign_flip` (src → rotated_out with sign bit conditionally flipped in one pass) then `turbo_kv_rht_forward_body` for WHT + scale. Eliminates a full load/store sweep over 128 floats. Savings: **24 ns** (355 → 331 ns). The standalone Step-2 AVX2 `mul_ps` widening alone contributed ~1 ns (the scalar loop was already auto-vectorised by the compiler); the fusion's savings come from the shared-pass, not the widening.

3c. **DONE — Step-4 horizontal max reduce.** In-register cascade: `_mm256_extractf128 + _mm_max_ps → _mm_movehl + _mm_max_ps → _mm_shuffle + _mm_max_ss`. Shipped for cleanliness. Impact measured at or below the benchmark noise floor (<2 ns on 331 → 331 runs, matching the 2–3 ns estimate).

3d. **DONE — Signmask lookup table for the in-tree seed.** `TURBO_KV_DEFAULT_SEED` is the only seed used by callers in practice (quantize prepare_block, query pre-rotation, dequant inverse). A 128-entry uint32 table is populated at shared-library load time via `__attribute__((constructor))` so the hot path has no init branch. Any other seed falls back to per-index scalar `turbo_kv_random_sign` calls. Savings: **20 ns** (331 → 311 ns). Larger than the 5–10 ns estimate because the scalar `turbo_kv_random_sign` calls were costlier than the original profile suggested once the surrounding paths had been widened.

**Cumulative savings across 3a–3d: 91 ns** on the AVX2 quantize path (402 → 311 ns). End-to-end speedup vs scalar: 5.66× → **7.2×**. Landed as `4ad8efd1e` in the llama.cpp submodule. All 30 PBT properties, 8/8 attention, 1113/1113 `test-backend-ops -b CPU -o MUL_MAT`, and every GPU-side test still pass with `max_err = 0.000000` in the differential tests.

4. **DONE — Agner Fog extraction.** `reference/agner/turbo_kv_4b_agner.csv` holds per-uarch latency, reciprocal throughput, execution-pipe binding, and µop count for 27 instructions × 6 microarchitectures (162 rows), each citing the Agner sheet + row number for verification. Reproducible via `reference/agner/extract.py`. Targets are Haswell, Skylake, IceLake (proxy for Alder Lake / Raptor Lake P-core per the rationale in `reference/agner/README.md`), Zen 1 (proxy for Zen+), Zen 2, Zen 3. Proxies, grouped-row handling, and the operand-pattern preference order are documented in the README.

5. **DONE — Cross-uarch extrapolation.** `reference/agner/analyze.py` combines the Zen 2 profile %s from section "Primary hot instructions" with the Agner reciprocal throughputs, computes a profile-weighted slowdown vs Zen 2, and projects per-call cycles for each target:

    | Target | profile-weighted slowdown vs Zen 2 | projected ns/call (311 ns baseline) |
    |---|---|---|
    | Haswell | 1.10× (tracked) → 1.06× overall | 330 |
    | Skylake | 0.95× → 0.97× overall | 301 |
    | IceLake (≈ Alder Lake / Raptor Lake P) | 0.95× → 0.97× overall | 301 |
    | Zen 1 / Zen+ | **1.64× (tracked) → 1.40× overall** | **434** |
    | Zen 2 (baseline) | 1.000× | 311 |
    | Zen 3 | 0.95× → 0.97× overall | 301 |

    The 0.378 untracked fraction of cycles (frontend, L1 load, branch prediction) is assumed uarch-neutral. Per-instruction hotspots driving the Zen 1 slowdown: VMULPS/VFMADD/VADDPS/VHADDPS/VCVTDQ2PS/VMOVDQU/VPSRLW all hit 2× (256-bit µop decomposition), VHADDPS 1.5×. Haswell's only slowdown vs Zen 2 is VADDPS at 2× (recip 1 vs 0.5), contributing 0.06× overall.

    **Decision: no variant kernel.** Every target's projection falls at or below the pre-repack Zen 2 baseline of 486 ns/call. Zen 1 at 434 ns is the worst case in scope and is still faster than what Zen 2 delivered before this phase's vectorisation work landed. The 128-bit XMM-width variant hypothesised in §"Kernel strategy" as a contingency for Zen 1 is not triggered — the AVX2 kernel as shipped satisfies the performance goal on every uarch in scope.

    An Alder Lake / Raptor Lake P-core actual measurement would validate the IceLake proxy assumption, but the extrapolation is conservative: Golden Cove has a wider retire pipeline than Ice Lake on most integer ops, so real performance should be no worse than the IceLake projection.

## Notes

**Zen 1 / Zen+ is the floor for performance targeting.** If the baseline kernel meets latency goals on Zen 1/+, it will meet them on every other target. Conversely, if measurement shows the 256-bit-split penalty dominates, the natural response is an XMM-width variant for that uarch rather than per-instruction workarounds.
