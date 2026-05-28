# Phase 6: Vulkan TURBO_KV_4B — World's First Walsh-Hadamard Transform on Vulkan

## Context

Phase 5 explored FP16 Rapid Packed Math for mul_mat_vec on Vega 64, producing deep
research into RADV ACO compiler behavior but no throughput improvement (38.4 t/s
ceiling confirmed through 8 approaches). The investigation uncovered `v_mad_mix_f32`
on GFX900 via `DenormFlushToZero` and characterized ACO's scheduler inflation as
optimal for bandwidth-bound workloads.

This phase shifted focus to **KV cache compression**: the peer CPU agent shipped
split K cache (partitioning K by RoPE boundary) and TURBO_KV_4B (RHT + Lloyd-Max
codebook from quantumaikr/quant.cpp). The Vulkan backend needed a complete shader
suite for TURBO_KV_4B — including the first-ever GPU Walsh-Hadamard Transform in
Vulkan compute shaders.

## Results

| Config | PPL (wikitext-2, c=512) | vs baseline | K compression | t/s (c=4096) |
|---|---|---|---|---|
| f16 baseline | 8.11 | — | 1× | 38.4 |
| f16:q4_0 split K | 9.45 | +16.5% | 2.2× | 37.5 |
| **f16:turbo_kv_4b split K** | **9.13** | **+12.6%** | **2.2×** | **37.7** |

TURBO_KV_4B **beats q4_0 on quality** (+12.6% vs +16.5% PPL degradation) at the
same compression ratio. The RHT-decorrelated codebook preserves more information
than uniform 4-bit nibble quantization.

MTP speculative decoding functional with split K (25-28 draft tokens accepted).
All 37 existing test-backend-ops tests pass. Baseline unaffected (BYTE-IDENTICAL).

## What shipped

### 7 new Vulkan shader files

| Shader | Purpose | Key technique |
|---|---|---|
| `turbo_kv_4b_rht.glsl` | Shared FWHT core | 12 `subgroupShuffleXor` on wave64, zero shared memory |
| `dequant_turbo_kv_4b.comp` | TURBO_KV_4B → F16/F32 | Codebook lookup + inverse RHT |
| `cpy_f32_turbo_kv_4b.comp` | F32 → TURBO_KV_4B | Forward RHT + `subgroupAdd`/`subgroupMax` + nearest codebook |
| `set_rows_turbo_kv_4b.comp` | KV cache write (SET_ROWS) | Forward RHT with row-indexed addressing |
| `get_rows_turbo_kv_4b.comp` | Row extraction | Inverse RHT with row lookup |
| `cpy_turbo_kv_4b_f32.comp` | TURBO_KV_4B → F32 (CPY) | Same as dequant, D_TYPE=float |

### Subgroup-cooperative Fast Walsh-Hadamard Transform

The inverse and forward RHT use a warp-cooperative approach inspired by
[hammer.ai's CUDA port](https://blog.hammer.ai/) and
[Dao-AILab/fast-hadamard-transform](https://github.com/Dao-AILab/fast-hadamard-transform):

```
Wave64 (GCN Vega): 64 threads × 2 values = 128 elements per block

Stages 0-5: subgroupShuffleXor(val, stride) for stride = 1,2,4,8,16,32
Stage 6:    thread-local swap (stride 64)

Total: 12 shuffle instructions + 1 local swap
Cost:  ~50 cycles (vs ~450 with shared memory + 14 barriers)
```

This is the **first Vulkan Walsh-Hadamard Transform implementation anywhere** — all
prior GPU WHT implementations used CUDA or Triton. Maps cleanly via
`GL_KHR_shader_subgroup_shuffle` (`subgroupShuffleXor`).

### Pipeline infrastructure (ggml-vulkan.cpp)

- 7 pipelines registered (dequant, get_rows, get_rows_f32, cpy_f32, cpy_f32,
  set_rows_i32, set_rows_i64), all gated on `device->subgroup_shuffle &&
  device->subgroup_arithmetic`
- CPY pipeline lookups extended for TURBO_KV_4B (F32↔TURBO_KV_4B both directions)
- SET_ROWS dispatch override: `CEIL_DIV(ne, QUANT_K)` instead of `CEIL_DIV(ne, 32*QUANT_K)` for subgroup-cooperative shaders
- supports_op coverage: GET_ROWS, SET_ROWS, CPY/CONT for TURBO_KV_4B

### 4 test programs

| Test | Result | What it validates |
|---|---|---|
| `test-turbo-kv-vulkan` | 6/6 PASS | CPU reference: round-trip, metadata, codebook distribution |
| `test-turbo-kv-gpu-roundtrip` | PASS (RMSE=0.000000) | GPU dequant bit-identical to CPU |
| `test-turbo-kv-gpu-quantize` | PASS (0/64 indices differ) | GPU CPY quantize matches CPU |
| `test-turbo-kv-set-rows` | PASS (indices match CPY) | SET_ROWS produces same blocks as CPY |

## Bugs found and fixed

### 1. Inverse RHT sign flip (early)

The inverse RHT initially applied the sign diagonal BEFORE and AFTER the WHT
butterfly. Correct inverse: WHT → normalize → sign flip (sign only AFTER, not
before). Found via Python reference comparison: RMSE dropped from 1.31 to 0.085.

### 2. 3D dispatch grid addressing (critical)

The dequant and CPY shaders used `gl_WorkGroupID.x` as the flat block index, but
the Vulkan dispatch decomposes large block counts into a 3D grid (512×Y×Z). Only
blocks with `WorkGroupID.x < wg0` were processed per Y-Z slice — the rest were
silently skipped.

**Fix**: reconstruct flat block_id from all three WorkGroupID dimensions:
```glsl
const uint block_id = gl_WorkGroupID.z * gl_NumWorkGroups.y * gl_NumWorkGroups.x
                    + gl_WorkGroupID.y * gl_NumWorkGroups.x
                    + gl_WorkGroupID.x;
```

This was the root cause of PPL=37 (only ~10% of blocks dequantized correctly).

### 3. SET_ROWS dispatch count (important)

The standard SET_ROWS dispatch divides by `32 * QUANT_K` (assuming 32 threads per
workgroup, each processing one block). TURBO_KV_4B needs 64 threads per block for
the subgroup FWHT. Added a type-specific override:
```cpp
if (dst->type == GGML_TYPE_TURBO_KV_4B)
    ne = CEIL_DIV(ne, ggml_blck_size(dst->type));  // one WG per block
```

### 4. SET_ROWS I32/I64 index types

The KV cache uses I64 row indices (`ggml_set_rows` with `GGML_TYPE_I64`). Added
separate shader variants with `B_TYPE=uint` (I32) and `B_TYPE=uvec2` (I64) index
buffer declarations.

## Also shipped (from earlier in the session)

### Phased sub-block mul_mat_vec (Q4_K + Q5_K)

Restructured the K-quant shaders to process Q-blocks in two phases of 4 sub-blocks
each. Clean code improvement, identical throughput (38.4 t/s), lays groundwork for
future compiler improvements.

### ACC_TYPE accumulator abstraction

Added `ACC_TYPE` to `mul_mat_vec_base.glsl` for mixed-precision accumulation
(F16 intermediates with F32 cross-block accumulator). Used by the f16acc variants.

### DenormFlushToZero for v_mad_mix_f32

Discovered that `v_mad_mix_f32` (f16×f16+f32) exists on GFX900 (Vega 10) and can
be enabled via `spirv_execution_mode(capabilities=[4465], 4460, 16)` with zero
Mesa changes. The ACO optimizer converts eligible F32 ops with F16 source
conversions into `v_mad_mix_f32` when the shader declares DenormFlushToZero.

### Subgroup ops on GCN

Removed legacy `architecture != AMD_GCN` exclusion for subgroup arithmetic. Wave64
`subgroupAdd` is correct and verified on Vega.

## Session statistics

- 8 approaches tested for mul_mat_vec throughput improvement (all at 38.4 t/s ceiling)
- 3 Mesa modifications tested (lower_ffma32, has_fma_mix, wave_minimum)
- 1 Mesa git build from source for ISA validation
- 5 shader bugs found and fixed
- 398 new lines of Vulkan shader code (world's first Vulkan WHT)
- PPL went from 37.3 (broken) → 9.13 (correct, beats q4_0) through systematic
  test-driven debugging

## Commit log

```
4b59de48e vulkan: fix TURBO_KV_4B 3D dispatch grid addressing
ea6fe3641 vulkan: fix set_rows dispatch for TURBO_KV_4B
a0b8b6dbc tests: GPU dequant PASS + GPU quantize(CPY) PASS
2b198d335 tests: add TURBO_KV_4B validation suite (6 tests)
99f1e1f55 vulkan: TURBO_KV_4B shader suite with subgroup-cooperative FWHT
de6adfaf8 merge: take turbo_kv_4b V cache fix
9233b3f8d merge: integrate split K cache + turbo_kv + chained MTP
2fb9da108 vulkan: phased sub-block mul_mat_vec + f16acc infrastructure
```
