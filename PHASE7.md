# PHASE 7: Async Cross-Device Copy Pipeline

**Goal**: Eliminate the synchronous host-side block during cross-device tensor copies.

## Status: COMPLETE

## Problem

`ggml_backend_vk_cpy_tensor_async()` performed cross-device copies as:

1. `ggml_vk_buffer_read()` — **synchronous**: submits GPU A → host transfer, calls `waitForFences()`, blocks until complete
2. `ggml_vk_buffer_write_async()` — async: batches host → GPU B transfer into transfer context

Step 1 stalled the calling thread while GPU A's DMA engine drained to host memory. No other work could proceed on either device during this window.

## Changes Made

**`ggml/src/ggml-vulkan.cpp`** (~80 lines net):

1. Added `vk_pending_xdev_copy` struct to track deferred cross-device copies:
   - Source device reference (for fence wait)
   - Source transfer context (holds the async read command buffer)
   - Destination buffer, offset, and size

2. Added `pending_xdev_copies` vector to `ggml_backend_vk_context`

3. Rewrote cross-device path in `cpy_tensor_async`:
   - **Host-visible source** (lavapipe): direct `memcpy` from `src_buf->ptr` into destination async write — zero staging overhead
   - **Device-local source** (real GPU): submit async read via source device's transfer queue using `sync_staging`, push to `pending_xdev_copies`, return immediately without fence wait

4. Updated `synchronize()` to process pending copies before normal sync:
   - Wait on source device's fence (the read)
   - Execute deferred staging memcpys
   - Submit async write to destination device
   - Clear pending list, then proceed with normal transfer context flush

## Test Hardware

Deliberately terrible hardware to stress-test the implementation at the low end:

- **CPU**: Intel Xeon X5650 (Westmere, 2010) — 6C/12T, no AVX, no AVX2, DDR3-1333 triple-channel
- **GPU 0**: AMD Radeon Pro WX 2100 (Polaris 12, 2017) — 512 stream processors, 2 GB GDDR5, PCIe 3.0 x16
- **GPU 1**: lavapipe (Mesa llvmpipe) — CPU-based Vulkan software rasterizer, JIT-compiles SPIR-V to LLVM IR

If it works here, it works anywhere. The Polaris 12 is a passively-cooled workstation card with roughly the compute power of a 2013 midrange GPU. lavapipe has no actual GPU hardware — every Vulkan operation runs on CPU.

## Results

### Phase 6 baseline (synchronous cross-device copy)
```
prompt eval: 15378 ms / 2 tokens (0.13 tok/s)
eval:         2742 ms / 4 runs   (1.46 tok/s)
total:       18127 ms / 6 tokens
```

### Phase 7 (async cross-device copy pipeline)
```
prompt eval:   862 ms / 2 tokens (2.32 tok/s)
eval:         2241 ms / 4 runs   (1.78 tok/s)
total:        3110 ms / 6 tokens
```

### Improvement
| Metric | Phase 6 | Phase 7 | Speedup |
|--------|---------|---------|---------|
| Prompt eval | 15378 ms | 862 ms | **17.8x** |
| Token generation | 685 ms/tok | 560 ms/tok | **1.2x** |
| Total | 18127 ms | 3110 ms | **5.8x** |

The prompt eval improvement is dramatic because the synchronous fence wait was blocking the entire pipeline per copy. With async submission, the source device's DMA engine works concurrently with other operations.

## Verification

- Single-GPU inference produces correct output (no regression)
- Multi-GPU inference produces coherent text across both devices
- Weights distributed: Vulkan0 = 211 MiB, Vulkan1 = 227 MiB
- KV cache split: Vulkan0 = 24 MiB, Vulkan1 = 20 MiB
- Graph splits = 3
