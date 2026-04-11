# mul_mat_vec Dispatch: Fork vs Upstream Comparison

## Function Signatures

| Aspect | Fork | Upstream |
|--------|------|----------|
| Function | `ggml_vk_mul_mat_vec_q_f16(ctx, subctx, src0, src1, dst, dryrun)` | `ggml_vk_mul_mat_vec_q_f16(ctx, subctx, cgraph, node_idx)` |
| Tensor access | Direct tensor pointers | Derived from `cgraph->nodes[node_idx]` |
| Dryrun handling | Explicit `bool dryrun` parameter, early return | Separate dryrun pass during graph build |

## Buffer/Subbuffer Construction

| Aspect | Fork | Upstream |
|--------|------|----------|
| d_Qx/d_Qy | Manual: `src0_buf_ctx->dev_buffer` + `vk_tensor_offset(src0) + src0->view_offs` | `ggml_vk_tensor_subbuffer(ctx, src0)` → `{buffer, offset, ggml_nbytes(tensor)}` |
| d_D | Manual: `dst_buf_ctx->dev_buffer` + `vk_tensor_offset(dst) + dst->view_offs` | `ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[node_idx + num_additional_fused_ops])` |
| UMA path | Has explicit UMA host buffer detection | No UMA code (handled elsewhere) |
| d_X (no dequant) | `{d_Qx, qx_buf_offset, x_sz}` where x_sz = aligned | `d_Qx` (subbuffer with ggml_nbytes as size) |
| d_Y (no dequant) | `{d_Qy, qy_buf_offset, y_sz}` | `d_Qy` (subbuffer) |
| d_X (dequant) | `{prealloc_x, 0, VK_WHOLE_SIZE}` via ggml_vk_cpy_to_contiguous | `{prealloc_x, 0, prealloc_x->size}` |

**Key difference**: Fork constructs `vk_subbuffer{buffer, offset, size}` inline at dispatch time. Upstream uses pre-constructed `vk_subbuffer` from `ggml_vk_tensor_subbuffer`. The descriptor **range** differs:
- Fork: `x_sz = ggml_vk_align_size(type_size * x_ne / blck_size, minStorageBufferOffsetAlignment)` — alignment-rounded
- Upstream: `ggml_nbytes(tensor)` — exact tensor size

For bf16 m=16,k=1: both give 32 bytes. **No numerical difference for this test case.**

## Pipeline Selection

| Aspect | Fork | Upstream |
|--------|------|----------|
| Function | `ggml_vk_get_dequantize_mul_mat_vec(ctx, a_type, b_type, num_cols)` | `ggml_vk_get_dequantize_mul_mat_vec(ctx, a_type, b_type, num_cols, m, k)` |
| MMQ (Q8_1) | **Not supported** — no `quantize_y` path | Supported — tries Q8_1 first if integer_dot_product available |
| Workgroup size | Always default (single dim in pipeline array) | Heuristic selects `DMMV_WG_SIZE_LARGE` vs `DMMV_WG_SIZE_SUBGROUP` based on M/K and vendor |
| 64-bit indexing | **Not supported** | Falls back to 64b indexing pipeline for large buffers |
| Pipeline array | `pipeline_dequant_mul_mat_vec_f32_f32[a_type][cols-1]` (2D) | `pipeline_dequant_mul_mat_vec_f32_f32[wg_size][a_type][cols-1]` (3D) |

**Impact on AMD**: The upstream workgroup size heuristic only triggers for NVIDIA post-Turing and Intel. On AMD/Vega, `dmmv_wg = DMMV_WG_SIZE_SUBGROUP` always. Same pipeline selected. **No difference on our hardware.**

## Descriptor Bindings

| Aspect | Fork | Upstream |
|--------|------|----------|
| Non-ID bindings | 3: A(0), B(1), D(2) | 5: A(0), B(1), D(2), Fuse0(3), Fuse1(4) |
| ID bindings | 4: A(0), B(1), D(2), IDS(3) | 6: A(0), B(1), D(2), Fuse0(3), Fuse1(4), IDS(5) |
| Dispatch call | `{d_X, d_Y, d_D}` | `{d_X, d_Y, d_D, d_F0, d_F1}` |
| Fuse buffers | N/A | `d_F0 = d_F1 = d_D` when no fusion |

**This is the ONLY structural SPIR-V difference.** Fork shaders don't declare `data_fuse0`/`data_fuse1` buffers at all. Upstream declares them even when `fusion_flags=0` and never reads them. The descriptor set layout differs, producing different SPIR-V. Fork SPIR-V is smaller (16604 vs 17960 bytes for bf16).

## Push Constants

| Field | Fork value | Upstream value | Match? |
|-------|-----------|---------------|--------|
| ncols (offset 0) | ne00 | ne00 | YES |
| stride_a (offset 4) | ne10 | ne10 | YES |
| stride_b (offset 8) | ne10 | ne10 | YES |
| stride_d (offset 12) | ne01 | ne01 | YES |
| batch_stride_a (16) | stride_batch_x | stride_batch_x | YES (same formula) |
| batch_stride_b (20) | stride_batch_y | stride_batch_y | YES (same formula) |
| batch_stride_d (24) | stride_batch_d | stride_batch_d | YES (same formula) |
| fusion_flags (28) | 0 | 0 (or flags) | YES (no fusion in tests) |
| base_work_group_y (32) | 0 | 0 (incremented in loop) | YES for ne12*ne13=1 |
| ne02 (36) | ne02 | ne02 | YES |
| ne12 (40) | ne12 | ne12 | YES |
| broadcast2 (44) | r2 | r2 | YES |
| broadcast3 (48) | r3 | r3 | YES |

**Identical for all test cases with ne12*ne13 <= maxComputeWorkGroupCount[1].**

## Dispatch Geometry

| Aspect | Fork | Upstream |
|--------|------|----------|
| groups_x | `ne01` (clamped with groups_z if >maxWgCount) | Same |
| groups_y | `ne12 * ne13` (single dispatch) | `min(ne12*ne13 - base, maxWgCount[1])` (loop) |
| groups_z | 1 (or 64 if ne01>maxWgCount) | Same |
| Descriptor sets | Always 1 | `CEIL_DIV(ne12*ne13, maxWgCount[1])` |
| base_work_group_y | Always 0 | Incremented per loop iteration |

**For test cases with ne12*ne13=1**: identical. The loop difference only matters for huge batch counts exceeding `maxComputeWorkGroupCount[1]` (typically 65535).

## Sync/Barriers

| Aspect | Fork | Upstream |
|--------|------|----------|
| Pre-dispatch sync | `ggml_vk_sync_buffers(subctx)` always | No sync before dispatch |
| Non-contig sync | Calls cpy_to_contiguous with VK_WHOLE_SIZE | Calls with exact prealloc size, conditional sync based on need_sync flags |
| prealloc_y caching | None | Caches `last_pipeline_used` / `last_tensor_used` to skip redundant quantize/copy |
| Post-dispatch sync flags | None | Sets `prealloc_x_need_sync` / `prealloc_y_need_sync` |

**The fork's extra `ggml_vk_sync_buffers` before dispatch adds an unnecessary pipeline barrier but should not cause numerical errors.**

## Shader: reduce_result

| Aspect | Fork | Upstream |
|--------|------|----------|
| Variants | 1: shmem-only | 3: subgroup-only, subgroup+shmem, shmem-only |
| Parameter qualifier | `const in FLOAT_TYPE temp[...]` | `FLOAT_TYPE temp[...]` (no qualifier, shmem path) |
| Fusion logic | None (no fusion_flags checks) | Has BIAS0/BIAS1/SCALE0/SCALE1 checks guarded by fusion_flags |

**The `const in` vs no qualifier was tested and ruled out — no effect.** The fusion logic is guarded by `fusion_flags != 0`, and fork always sets fusion_flags=0, so the upstream fusion code is dead code. **No computational difference.**

## Shader: get_offsets (non-ID path)

Fork and upstream are now **identical** after Round 2 alignment:
- Both use `batch_idx_a * (p.batch_stride_a / QUANT_K)` for a_offset
- Both use `gl_WorkGroupID.y + p.base_work_group_y` for batch_idx

## Shader: iter / compute_outputs / main

**Byte-for-byte identical** between fork and upstream. No differences.

---

## Summary of Actionable Differences

### Differences that COULD cause bugs:

1. **Descriptor binding count (3 vs 5)**: Different SPIR-V bytecode. Could produce different register allocation, different instruction scheduling, or different driver behavior on RADV. This is the ONLY difference that makes the fork's compiled shader binary different from upstream's.

2. **Missing prealloc_y caching**: Fork may redundantly re-quantize/copy Y data. Unlikely to cause wrong results but indicates the fork is behind upstream.

3. **Missing MMQ (quantize_y) support**: Performance difference only. Fork always uses f32 B matrix.

### Differences that CANNOT cause bugs:

- Push constant values: identical
- Dispatch geometry: identical for test cases
- Buffer ranges: identical for test cases
- Shader math: identical
- Extra sync barrier: can only slow down, not corrupt

---

## Fix Plan

### Root Cause Hypothesis

The descriptor binding count (3 vs 5) is the only difference that affects the SPIR-V binary. Since the shader logic is identical, the bug must be in how RADV compiles the different SPIR-V. This is a driver-level issue that we can work around by **matching upstream's SPIR-V structure**.

### Proposed Fix: Add Fuse0/Fuse1 bindings to fork shaders

1. **Add buffer declarations** to `mul_mat_vec_base.comp`:
   ```glsl
   layout (binding = 3) readonly buffer Fuse0 {D_TYPE data_fuse0[];};
   layout (binding = 4) readonly buffer Fuse1 {D_TYPE data_fuse1[];};
   #ifdef MUL_MAT_ID
   layout (binding = 5) readonly buffer IDS {int data_ids[];};
   #else
   // binding 3 → 5 for IDS
   #endif
   ```

2. **Add fusion_flags constant definitions** (even though they're dead code):
   ```glsl
   #define MAT_VEC_FUSION_FLAGS_BIAS0 0x1
   #define MAT_VEC_FUSION_FLAGS_BIAS1 0x2
   ```

3. **Update C++ dispatch** to pass 5 bindings:
   ```cpp
   ggml_vk_dispatch_pipeline(ctx, subctx, dmmv,
       { vk_subbuffer{d_X, x_buf_offset, x_sz},
         vk_subbuffer{d_Y, y_buf_offset, y_sz},
         vk_subbuffer{d_D, d_buf_offset, d_sz},
         vk_subbuffer{d_D, d_buf_offset, d_sz},  // Fuse0 = D (unused)
         vk_subbuffer{d_D, d_buf_offset, d_sz},  // Fuse1 = D (unused)
       },
       pc, { groups_x, (uint32_t)(ne12 * ne13), groups_z });
   ```

4. **Update pipeline creation** to use 5 descriptor bindings for mul_mat_vec pipelines.

### Alternative Fix: Port upstream dispatch wholesale

Since `ggml_vk_mul_mat_vec_q_f16` is confirmed NOT split-mode code, we can port the entire upstream function. This would also bring:
- MMQ support (integer dot product)
- 64-bit indexing
- Workgroup size heuristics
- prealloc_y caching
- Proper dispatch loop for large batch counts

**Recommended**: Port upstream wholesale. It's lower risk than surgical patches because we get a known-working implementation, and it doesn't touch any split-mode code paths.

### Verification

1. Rebuild and run `test-backend-ops -o MUL_MAT` — bf16 k=1 must pass
2. Run full `test-backend-ops` 3x on Vega — 0 failures
3. Run full `test-backend-ops` on 6800 XT — no regressions
4. Run inference with ngl=99 — no regression
