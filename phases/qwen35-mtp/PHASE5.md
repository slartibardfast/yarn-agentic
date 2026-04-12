# Phase 5: FP16 Accumulation (Rapid Packed Math) for mul_mat_vec on Vega

## Target

Drop `mul_mat_vec_q5_k` from **VGPR=256 / 1 wave / 10% occupancy** to **VGPR=64 / 4 waves / 40% occupancy** on Vega 64 using FP16 packed accumulation. Expected generation throughput uplift: 1.5–2× on the vocab head kernel (15.1% of total GPU time).

## Measured baseline (overnight profiling data)

| | Vega 64 (current) | Navi 21 (reference) |
|---|---|---|
| Kernel variant | `mul_mat_vec_q5_k_f32_f32` | `mul_mat_vec_q5_k_q8_1_f32` |
| VGPR | **256** | 40 |
| Waves/SIMD | 1 / 10 | 16 / 16 |
| Occupancy | 10% | 100% |
| Vocab head GFLOPS | 471 | 810 |

Navi uses the IDP (`q8_1`) path which Vega lacks. Vega's equivalent lever is RPM (`v_pk_fma_f16`) — packed FP16 FMA, 2 ops per cycle, halving VGPR and doubling peak FLOPS.

## Why FP16 accumulation is safe here

- q5_K weights: 5-bit values × FP16 scale. FP16's 10-bit mantissa is 2× wider than the data.
- The **batched matmul** (`matmul_q5_k_f32_f16acc_aligned_m`) already runs FP16 accumulation on the same weights on Vega at **VGPR=64**. Proven to be correct in production.
- Activations are RMS-normalized; no overflow risk in the dot product range.

## Occupancy table (Vega GCN gfx906, 256 VGPR pool, wave64)

| VGPR | Waves/SIMD | Occupancy | vs current |
|---:|---:|---:|---:|
| 256 | 1 | 10% | 1× |
| 128 | 2 | 20% | 2× |
| **64** | **4** | **40%** | **4×** |
| 48 | 5 | 50% | 5× |
| 32 | 8 | 80% | 8× |

## The gap

The batched `matmul_*_f16acc` variants exist. The single-token `mul_mat_vec` path has no `f16acc` variant. The infrastructure (FLOAT_TYPE, ACC_TYPE, fp16 loop in shader-gen) exists upstream — the single-token path just never got the same treatment. This is a **registration gap**, not a capability gap.

## Root cause of VGPR=256

`mul_mat_vec_q5_k.comp` has **hardcoded `vec4` and `vec2` intermediates** that bypass `FLOAT_TYPE`:

```glsl
const vec4 scale_0_4_l_f = vec4(unpack8(...));    // 4 × F32 = 4 VGPR
const vec4 qs0_16_lo4 = vec4(unpack8(...));        // 4 × F32 = 4 VGPR
vec2 by10 = vec2(data_b_v2[...]);                  // 2 × F32 = 2 VGPR × 8 reads
```

With 4 scale `vec4`s + 4 quant `vec4`s + 8 `vec2` B-reads + accumulators + loop state, the compiler can't fit below ~256 VGPR at F32.

With `f16vec4` / `f16vec2`, each value packs 2 into one VGPR. The same intermediates fit in ~128 VGPR, and with compiler scheduling, the target of 64 is reachable (the batched matmul already proves this).

## Implementation plan

### 1. Shader prep: `FLOAT_TYPEV4` define

Add to `mul_mat_vec_q5_k.comp` (and the generic `mul_mat_vec.comp` / `mul_mat_vec_base.glsl`):

```glsl
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable

#ifndef FLOAT_TYPEV4
#define FLOAT_TYPEV4 vec4
#endif
```

The `GL_EXT_shader_explicit_arithmetic_types_float16` extension is **critical** — without it, `float16_t` arithmetic may silently promote to F32 even if storage is FP16. This ensures RADV's ACO compiler actually emits `v_pk_fma_f16`.

### 2. Replace hardcoded F32 types in `mul_mat_vec_q5_k.comp`

Every `vec4` intermediate → `FLOAT_TYPEV4`. Every `vec2` → `FLOAT_TYPEV2`. The `fma()` and `dot()` calls work identically on `float16_t` — no logic changes, only type substitutions.

Backward compatibility: when `FLOAT_TYPEV4 = vec4` (the default), the shader compiles identically to today. The f16acc path only activates when `FLOAT_TYPEV4 = f16vec4` via the shader-gen defines.

### 3. Register f16acc variant in `vulkan-shaders-gen.cpp`

After the existing `_f16_f32` emission (line ~716), add for each K-quant type:

```cpp
string_to_spv("mul_mat_vec_" + tname + "_f16_f16", shader,
    merge_maps(base_dict_f16acc, {{data_a_key, "1"},
        {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"},
        {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}));
```

Where `base_dict_f16acc` sets `FLOAT_TYPE=float16_t`, `FLOAT_TYPEV2=f16vec2`, `FLOAT_TYPEV4=f16vec4`.

Start with q5_k only. Extend to other K-quants after validation.

### 4. Add pipeline array + selection in `ggml-vulkan.cpp`

```cpp
// New pipeline array (alongside existing f32 and f16_f32)
vk_pipeline pipeline_dequant_mul_mat_vec_f16_f16[DMMV_WG_SIZE_COUNT][GGML_TYPE_COUNT][mul_mat_vec_max_cols];
```

Selection logic in `ggml_vk_get_dequantize_mul_mat_vec()`:

```cpp
// Prefer f16acc on GCN (fp16 but no integer dot product)
if (ctx->device->fp16 && !ctx->device->integer_dot_product) {
    auto p = ctx->device->pipeline_dequant_mul_mat_vec_f16_f16[wg][a_type][cols-1];
    if (p) return p;
}
```

This gates on `fp16 && !integer_dot_product` — fires on Vega (GCN), not on Navi (RDNA2 has IDP). Navi keeps its existing fast `q8_1` path.

### 5. Verify VGPR

```bash
GGML_VK_VISIBLE_DEVICES=1 GGML_VK_PIPELINE_STATS=mul_mat_vec_q5_k_f16_f16 \
  llama-server -m model.gguf -c 4096 -ngl 99 --no-warmup
# drive one completion, then:
# Expected: VGPRs: 64
```

If VGPR > 64: investigate with `ACO_DEBUG=live-info` and apply vec2 pairing (rewrite scalar `fma(a, b, c)` as `fma(f16vec2(a0, a1), f16vec2(b0, b1), f16vec2(c0, c1))`) to force the compiler's hand.

### 6. Verify correctness

- Output fingerprint: same prompt, same seed, compare first 200 chars between f32acc and f16acc
- For q5_K (5-bit data), expect identical tokens for at least the first 100+ tokens

### 7. Verify throughput

```bash
GGML_VK_PERF_LOGGER=1 drive.py 3 --label q4km-vega-f16acc
```

Compare `MUL_MAT_VEC q5_K 248320×4096` timing and GFLOPS against step 1 baseline (471 GFLOPS, 3263 ms over 755 calls). Target: 800+ GFLOPS, closing the gap with Navi's 810.

## Risk mitigation

| Risk | Detection | Mitigation |
|---|---|---|
| RADV doesn't emit `v_pk_fma_f16` | VGPR > 64 in pipeline stats | Explicit `f16vec2` pairing in the FMA chain |
| FP16 overflow on unusual weights | Perplexity regression or NaN | Add `clamp()` before accumulation (performance cost is negligible) |
| Shared memory bank conflicts | Throughput below 1.3× baseline | Pad shmem layout or switch to subgroup-only reduction |
| The "smaller variant" at 64 VGPR gets selected for the wrong dimensions | Wrong pipeline stats entry | Check the pipeline hash matches the 248320-wide dispatch |

## VGPR target ladder

| VGPR | Waves/SIMD | Occupancy | Commit |
|---:|---:|---:|---|
| 256 | 1 | 10% | current (baseline) |
| **64** | **4** | **40%** | **Phase 5a — first commit** |
| 32 | 8 | 80% | Phase 5b — if profiling shows occupancy still bottlenecked |
| 24 | 10 | 100% | Phase 5c — stretch, requires careful register scheduling |

24 VGPR is NOT out of the question — `mul_mat_vec_q8_0` already hits
36 VGPR on Vega (small variant), and simple ops like `fused_gate_prep`
hit 8. But the payoff curve flattens: 4→8 waves is the second biggest
step (after 1→4). The 8→10 wave delta is small and only helps if the
kernel is purely latency-limited (unlikely for matvec which is
bandwidth-dominated). Profile first, then chase lower VGPR only if the
data says it's worth it.

## Scope and phasing

### Phase 5a — q5_k only (one commit)

Target `mul_mat_vec_q5_k` as the first kernel: it's the vocab head,
the #2 hotspot, and the only pipeline at VGPR=256. Changes touch 4
files (see below). Verify VGPR=64, throughput uplift, correctness.

### Phase 5b — extend to all quant types (one commit)

The full audit identified **18 files** needing the same treatment.
The pattern is identical in every case: replace hardcoded `vec4` /
`vec2` / `float` intermediates with `FLOAT_TYPEV4` / `FLOAT_TYPEV2` /
`FLOAT_TYPE`, then register f16acc variants in the shader-gen loop.

**Tier 1 — K-quant mul_mat_vec shaders (each has its own .comp file):**

| File | Hardcoded vec4 | Hardcoded vec2 | Notes |
|---|---|---|---|
| `mul_mat_vec_q2_k.comp` | 4 (qs_u32_*) | 9 (b*, dm) | sccache uses FLOAT_TYPE already |
| `mul_mat_vec_q3_k.comp` | 8 (hmk_*, qs_u32_*) | 8 (b*) | sccache uses FLOAT_TYPE already |
| `mul_mat_vec_q4_k.comp` | 6 (scale_*, qs*) | — | uses vec4 for B reads too (by*) |
| `mul_mat_vec_q5_k.comp` | 6 (scale_*, qs*) | 8 (by*) | **Phase 5a target** |
| `mul_mat_vec_q6_k.comp` | 8 (q0-q3, by*) | — | sccache uses FLOAT_TYPE already |

**Tier 2 — generic mul_mat_vec (legacy quants via dequant_funcs.glsl):**

| File | Hardcoded types | Dependency |
|---|---|---|
| `mul_mat_vec.comp` | vec4 (bv*, v, v2), vec2 (dm, v) | `dequant_funcs.glsl` return types |

This file handles Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, BF16, IQ4_NL, MXFP4.
The fix requires ALSO changing `dequant_funcs.glsl` (see critical deps).

**Tier 3 — IQ-quant mul_mat_vec shaders:**

| File | Hardcoded vec4 | Hardcoded float | Notes |
|---|---|---|---|
| `mul_mat_vec_iq1_m.comp` | b_vals[8], fbits_v*, delta_v, sum_v | delta_cache[] | complex bit-unpacking |
| `mul_mat_vec_iq1_s.comp` | b_val_*, fbits*, delta_v, sum_v | dl, delta_val | similar to iq1_m |
| `mul_mat_vec_iq2_s.comp` | grid*, b0, b4 | db | |
| `mul_mat_vec_iq2_xs.comp` | grid*, b0, b4 | db_vals, d, scale | |
| `mul_mat_vec_iq2_xxs.comp` | grid*, b0, b4 | d, db | |
| `mul_mat_vec_iq3_s.comp` | grid*, b0, b4 | d, dscale | |
| `mul_mat_vec_iq3_xxs.comp` | grid*, b0, b4 | d, db | |

**Tier 4 — MMVQ path (integer dot product, lower priority):**

| File | Hardcoded types | Notes |
|---|---|---|
| `mul_mat_vecq.comp` | vec2 (cache_b_ds) | MMVQ path — only fires on Navi (IDP). Lower priority for f16acc since Navi already fast. |
| `mul_mat_vecq_funcs.glsl` | vec2 (get_dm*, mul_q8_1 params), float (get_d_scale, intermediates) | Would need full type parameterization for f16acc MMVQ. |

**Tier 5 — special-layout shaders (lowest priority):**

| File | Status |
|---|---|
| `mul_mat_vec_nc.comp` | Hardcodes `#define FLOAT_TYPE float`. Needs parameterization. |
| `mul_mat_vec_p021.comp` | Same. Standalone, no quant support. |

### Critical shared dependencies

**`dequant_funcs.glsl`** — used by Tier 1 + Tier 2 shaders:

- `get_dm()` returns hardcoded `vec2` (~12 definitions)
- `dequantize()` returns hardcoded `vec2` (~30 definitions)
- `dequantize4()` returns hardcoded `vec4` (~20 definitions)

All need `vec2` → `FLOAT_TYPEV2` and `vec4` → `FLOAT_TYPEV4`. This
is the single largest changeset. It affects every quant type that uses
the generic mul_mat_vec.comp path.

**`mul_mat_vec_base.glsl`** — used by all Tier 1/2/3 shaders:

Already uses `FLOAT_TYPE` throughout. Only needs the
`GL_EXT_shader_explicit_arithmetic_types_float16` extension added.
No hardcoded type fixes needed.

## Full file change table

| File | Phase | Change type |
|---|---|---|
| `mul_mat_vec_q5_k.comp` | 5a | vec4→FLOAT_TYPEV4, vec2→FLOAT_TYPEV2, add f16 ext |
| `vulkan-shaders-gen.cpp` | 5a | base_dict_f16acc, emit f16_f16 variants |
| `ggml-vulkan.cpp` | 5a | pipeline array + creation + selection |
| `mul_mat_vec_base.glsl` | 5a | add f16 ext, add FLOAT_TYPEV4 default |
| `mul_mat_vec_q4_k.comp` | 5b | vec4→FLOAT_TYPEV4 (6 sites) |
| `mul_mat_vec_q6_k.comp` | 5b | vec4→FLOAT_TYPEV4 (8 sites) |
| `mul_mat_vec_q3_k.comp` | 5b | vec4→FLOAT_TYPEV4 (8), vec2→FLOAT_TYPEV2 (8) |
| `mul_mat_vec_q2_k.comp` | 5b | vec4→FLOAT_TYPEV4 (4), vec2→FLOAT_TYPEV2 (9) |
| `dequant_funcs.glsl` | 5b | vec2→FLOAT_TYPEV2 (~42), vec4→FLOAT_TYPEV4 (~20) |
| `mul_mat_vec.comp` | 5b | vec4→FLOAT_TYPEV4 (6), vec2→FLOAT_TYPEV2 (2) |
| `mul_mat_vec_iq1_m.comp` | 5b | vec4→FLOAT_TYPEV4 (8+), float→FLOAT_TYPE (2+) |
| `mul_mat_vec_iq1_s.comp` | 5b | vec4→FLOAT_TYPEV4 (8+), float→FLOAT_TYPE (2+) |
| `mul_mat_vec_iq2_s.comp` | 5b | vec4→FLOAT_TYPEV4 (6+), float→FLOAT_TYPE (2+) |
| `mul_mat_vec_iq2_xs.comp` | 5b | vec4→FLOAT_TYPEV4 (6+), float→FLOAT_TYPE (3+) |
| `mul_mat_vec_iq2_xxs.comp` | 5b | vec4→FLOAT_TYPEV4 (6+), float→FLOAT_TYPE (2+) |
| `mul_mat_vec_iq3_s.comp` | 5b | vec4→FLOAT_TYPEV4 (4+), float→FLOAT_TYPE (3+) |
| `mul_mat_vec_iq3_xxs.comp` | 5b | vec4→FLOAT_TYPEV4 (4+), float→FLOAT_TYPE (2+) |
| `mul_mat_vecq_funcs.glsl` | 5c | vec2→FLOAT_TYPEV2 (10+), float→FLOAT_TYPE (15+) |
| `mul_mat_vecq.comp` | 5c | vec2→FLOAT_TYPEV2 (2) |
| `mul_mat_vec_nc.comp` | 5d | parameterize FLOAT_TYPE, vec4→FLOAT_TYPEV4 |
| `mul_mat_vec_p021.comp` | 5d | parameterize FLOAT_TYPE, vec4→FLOAT_TYPEV4 |

### Shader-gen registration (Phase 5b)

The current gen loop at line ~710 iterates `type_names` and emits
`_f32_f32` + `_f16_f32` per type. Phase 5b adds a third variant
`_f16_f16` (B=f16, acc=f16) per type, using `base_dict_f16acc`:

```cpp
base_dict_f16acc = {
    {"FLOAT_TYPE", "float16_t"},
    {"FLOAT_TYPEV2", "f16vec2"},
    {"FLOAT_TYPEV4", "f16vec4"},
    {"FLOAT16", "1"},
};
```

This gives ~25 new shader variants (one per type × 3 subgroup modes).
All registered under `pipeline_dequant_mul_mat_vec_f16_f16`.

### Pipeline selection (Phase 5a, extended in 5b)

In `ggml_vk_get_dequantize_mul_mat_vec()`:

```cpp
// Prefer f16acc on devices with RPM but no IDP
if (ctx->device->fp16 && !ctx->device->integer_dot_product) {
    auto p = ctx->device->pipeline_dequant_mul_mat_vec_f16_f16[wg][a_type][cols-1];
    if (p) return p;
}
// Existing: prefer q8_1 on IDP-capable devices (Navi)
// Existing: fall back to f32/f16 accumulation
```

Both paths remain available: Vega takes f16acc (RPM), Navi takes
q8_1 (IDP). Neither path is forced — the selection is
device-capability-driven with graceful fallback.

## Implementation sequence

| Commit | Scope | Risk | Verification |
|---|---|---|---|
| **5a** | q5_k only (4 files) | Low — isolated, one kernel | VGPR=64, fingerprint match, GFLOPS ≥700 |
| **5b-tier1** | q2_k through q6_k (5 K-quant .comp files) | Low — same pattern | VGPR ≤64 per type, test-backend-ops MUL_MAT passes |
| **5b-tier2** | dequant_funcs.glsl + mul_mat_vec.comp (legacy quants) | Medium — large shared dependency | All legacy quant types pass test-backend-ops |
| **5b-tier3** | IQ-quant shaders (7 files) | Medium — complex bit-unpacking | IQ quant test-backend-ops passes |
| **5c** | MMVQ funcs + shader (2 files) | Low priority — Navi already fast | Optional, only if f16acc MMVQ shows further gains |
| **5d** | nc + p021 (2 files) | Low priority — rarely used | Optional |

## Success criteria

### Phase 5a (q5_k)

1. `GGML_VK_PIPELINE_STATS` shows `mul_mat_vec_q5_k_f16_f16` at **VGPR ≤ 64**
2. Output fingerprint matches F32 baseline for the first 200 chars at seed=42
3. `MUL_MAT_VEC q5_K 248320×4096` GFLOPS on Vega ≥ **700** (currently 471; Navi reference 810)
4. Generation t/s on Vega improves from 35.9 to ≥ **40** (net ~12% global speedup from a single kernel fix)

### Phase 5b (all types)

5. Every K-quant + legacy quant mul_mat_vec pipeline compiles at **VGPR ≤ 128** on Vega (most should hit 64)
6. test-backend-ops passes for `MUL_MAT` across all quant types on Vulkan0
7. Overall generation throughput on Qwen3.5-9B improves by ≥ **20%** (all hot mul_mat_vec kernels benefit, not just vocab head)
8. No perplexity regression vs F32 accumulation baseline

### Stretch (Phase 5c — VGPR=24-32)

9. If post-5b profiling shows occupancy is still the bottleneck, push q5_k to **VGPR ≤ 32** (8 waves) via explicit vec2 pairing and tighter intermediate scheduling
10. Measure diminishing returns — if 8 waves shows <5% over 4 waves, stop there
