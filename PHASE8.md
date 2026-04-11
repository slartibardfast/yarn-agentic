# PHASE 8: Double-Buffered Staging

**Goal**: Eliminate shared staging buffer corruption during concurrent cross-device copies by introducing a per-copy staging pool.

## Status: COMPLETE

## Problem

Phase 7 introduced async cross-device copies, but all copies from the same source device shared a single `sync_staging` buffer and `device->fence`. When multiple cross-device copies were submitted before `synchronize()`:

1. The second `vkQueueSubmit` reused the fence that was already pending from the first copy — Vulkan error
2. The second copy overwrote staging buffer contents before the first copy's data was consumed — silent corruption

This didn't manifest in Phase 7 testing because the graph split pattern happened to avoid concurrent copies from the same device.

## Changes Made

**`ggml/src/ggml-vulkan.cpp`** (~60 lines net):

1. Added `xdev_staging_slot` struct to `vk_device_struct`:
   ```cpp
   struct xdev_staging_slot {
       vk_buffer buffer;
       vk::Fence fence;
       bool in_use = false;
   };
   std::vector<xdev_staging_slot> xdev_staging_pool;
   ```

2. Added per-copy staging and fence fields to `vk_pending_xdev_copy`:
   ```cpp
   struct vk_pending_xdev_copy {
       vk_device src_device;
       vk_context src_transfer_ctx;
       vk_buffer staging;      // per-copy staging buffer
       vk::Fence fence;        // per-copy fence
       vk_buffer dst_buf;
       size_t dst_offset;
       size_t nbytes;
   };
   ```

3. Staging pool acquisition in `cpy_tensor_async`:
   - Search for a free slot (`in_use == false`) in the source device's pool
   - If none available, create a new slot with fresh buffer and fence
   - Mark slot `in_use = true`, store references in pending copy

4. Staging slot return in `synchronize()`:
   - Wait on per-copy fence (not shared device fence)
   - Execute deferred memcpy from per-copy staging buffer
   - Mark slot `in_use = false` for reuse
   - Staging buffers persist across calls — no per-transfer allocation

## Design Notes

- **Pool, not double-buffer**: The original plan called for rotating between two fixed staging buffers. A pool is simpler and handles any number of concurrent copies without capacity limits.
- **Lazy allocation**: Slots are created on-demand and never freed. For typical graph splits (2-5 cross-device copies), the pool stays small.
- **lavapipe unaffected**: Host-visible source path uses direct memcpy, bypasses staging entirely.

## Test Hardware

Same deliberately terrible hardware as Phase 7:

- **CPU**: Intel Xeon X5650 (Westmere, 2010) — 6C/12T, no AVX, no AVX2, DDR3-1333 triple-channel
- **GPU 0**: AMD Radeon Pro WX 2100 (Polaris 12, 2017) — 512 stream processors, 2 GB GDDR5, PCIe 3.0 x16
- **GPU 1**: lavapipe (Mesa llvmpipe) — CPU-based Vulkan software rasterizer

## Results

### Phase 7 baseline
```
prompt eval:  862 ms / 2 tokens (2.32 tok/s)
eval:        2241 ms / 4 runs   (1.78 tok/s)
total:       3110 ms / 6 tokens
```

### Phase 8 (per-copy staging pool)
```
prompt eval:  893 ms / 2 tokens (2.24 tok/s)
eval:        2147 ms / 4 runs   (1.86 tok/s)
total:       3051 ms / 6 tokens
```

### Analysis

| Metric | Phase 7 | Phase 8 | Change |
|--------|---------|---------|--------|
| Prompt eval | 862 ms | 893 ms | ~same |
| Token generation | 560 ms/tok | 537 ms/tok | ~4% faster |
| Total | 3110 ms | 3051 ms | ~2% faster |

Performance is essentially unchanged, which is expected — Phase 8 is a **correctness fix**, not a performance optimization. The per-copy staging pool:

1. **Prevents corruption** when multiple cross-device copies overlap (the real win)
2. **Provides infrastructure** for future phases that submit more concurrent transfers
3. **Eliminates the shared fence hazard** that would have caused Vulkan validation errors in more aggressive graph split patterns

## Verification

- Single-GPU inference produces correct output (no regression)
- Multi-GPU inference produces coherent text across both devices
- Multiple concurrent cross-device copies from same source device work correctly
- No Vulkan validation errors
