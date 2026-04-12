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

## Scope

Phase 5 is one commit, targeting one kernel (`mul_mat_vec_q5_k`). If VGPR=64 is confirmed and throughput improves, a follow-up commit extends to all K-quant types (q2_k through q6_k) and legacy quants (q4_0, q5_0, q8_0) using the same pattern.

No architectural changes. No new ops. No new fusion. Just a shader type substitution + registration + pipeline selection. The batched matmul already proves the approach works.

## Files to modify

| File | Change |
|---|---|
| `vulkan-shaders/mul_mat_vec_q5_k.comp` | Add `GL_EXT_shader_explicit_arithmetic_types_float16`, add `FLOAT_TYPEV4` default, replace `vec4`/`vec2` with `FLOAT_TYPEV4`/`FLOAT_TYPEV2` |
| `vulkan-shaders/mul_mat_vec_base.glsl` | Add `GL_EXT_shader_explicit_arithmetic_types_float16` |
| `vulkan-shaders/vulkan-shaders-gen.cpp` | Add `base_dict_f16acc`, emit `_f16_f16` variants, add SPV array generation |
| `ggml-vulkan.cpp` | Add `pipeline_dequant_mul_mat_vec_f16_f16` array, creation, selection logic |

## Success criteria

1. `GGML_VK_PIPELINE_STATS` shows `mul_mat_vec_q5_k_f16_f16` at **VGPR ≤ 64**
2. Output fingerprint matches F32 baseline for the first 200 chars at seed=42
3. `MUL_MAT_VEC q5_K 248320×4096` GFLOPS on Vega ≥ **700** (currently 471; Navi reference 810)
4. Generation t/s on Vega improves from 35.9 to ≥ **40** (net ~12% global speedup from a single kernel fix)
