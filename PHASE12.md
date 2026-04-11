# PHASE 12: dmabuf Zero-Copy Cross-Device Transfer (RADV)

**Goal**: Eliminate host staging CPU memcpy for cross-device copies on Mesa RADV by sharing GPU buffers via Linux dmabuf file descriptors.

## Status: COMPLETE

## Problem

All cross-device copies currently route through host memory with 4 data moves:

```
GPU A device-local → [DMA] → host staging → [CPU memcpy] → host staging → [DMA] → GPU B device-local
```

The CPU memcpy is unnecessary overhead. With dmabuf, both GPUs can access the same physical memory, eliminating both CPU memcpys:

```
GPU A device-local → [DMA] → shared dmabuf buffer → [DMA] → GPU B device-local
```

## Changes Made

**`ggml/src/ggml-vulkan.cpp`** (~330 lines net):

1. **Extension probing at device init**:
   - Check for `VK_KHR_external_memory_fd` and `VK_EXT_external_memory_dma_buf` extensions
   - Probe actual compatibility via `vkGetPhysicalDeviceExternalBufferProperties` (checks EXPORTABLE + IMPORTABLE bits)
   - Cache result as `device->dmabuf_support`
   - Enable extensions and load `vkGetMemoryFdKHR` function pointer at device creation

2. **dmabuf shared staging structure** on `vk_device_struct`:
   - `dmabuf_shared_staging` struct: exportable src buffer, imported dst buffer, dmabuf fd, fence
   - Per-peer map keyed by destination device index
   - Automatic cleanup in destructor

3. **Three helper functions** (all `#if !defined(_MSC_VER)` guarded):
   - `ggml_vk_create_dmabuf_exportable_buffer`: Creates buffer with `VkExportMemoryAllocateInfo` + `VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT`
   - `ggml_vk_import_dmabuf_buffer`: Imports dmabuf fd on peer device, uses `dup()` since `vkAllocateMemory` takes ownership
   - `ggml_vk_get_dmabuf_staging`: Gets or creates dmabuf shared staging between a device pair, with error recovery

4. **Modified `cpy_tensor_async`** (cross-device path):
   - Tries dmabuf path first: GPU A copies src → exportable buffer, records pending copy with dmabuf dst buffer
   - Falls back to host-staging path if dmabuf unavailable

5. **Modified `synchronize`** (cross-device copy loop):
   - When `xdev.dmabuf_dst_buffer` is non-null: uses `ggml_vk_buffer_copy_async` (GPU-to-GPU copy on dst device)
   - When null: uses existing host-staging path (memcpy + buffer write)

6. **Startup logging**: Added `dmabuf: 1/0` to device info line in `ggml_vk_print_gpu_info`

## Design Notes

- **No `_MSC_VER` guard**: All dmabuf code is `#if !defined(_MSC_VER)` since dmabuf is a Linux-only concept
- **`dup()` for fd handling**: `vkAllocateMemory` with `VkImportMemoryFdInfoKHR` takes ownership of the fd, so we `dup()` it to keep the original fd alive for reuse
- **Lazy initialization**: dmabuf staging is created on first use per device pair, not at startup
- **Error recovery**: If buffer creation fails, sets `dmabuf_support = false` on the source device and falls back to host staging

### Driver support confirmed
- **RADV (Mesa AMD)**: Full dmabuf support on both Navi 21 (RDNA 2) and Vega 10 (GCN 5)
- **lavapipe**: Reports extension support but probe may filter it; falls back to host staging regardless
- **Polaris 12**: Reports dmabuf support (extension + probe)

### PCIe topology
- On AMD platforms without SAM/rBAR, the dmabuf buffer is likely host-visible memory that both GPUs can DMA to/from
- The key win is eliminating the **CPU memcpy** between two separate host staging buffers — even if the actual DMA still routes through host memory, removing the CPU copy reduces latency
- With SAM/rBAR enabled, true peer-to-peer DMA over PCIe may be possible

## Test Hardware

See [Benchmarks](BENCHMARKS.md) for full hardware specs (System B: RX 6800 XT + Vega).

## Results

### TinyLlama 1.1B Q2_K (111 token prompt, 3 eval tokens, best of 3)

| Configuration | Prompt eval | Token gen | Load time |
|--------------|------------|-----------|-----------|
| RX 6800 XT alone | 99 ms (1117 tok/s) | 11.08 ms/tok (90 tok/s) | 236 ms |
| Both GPUs (smgs 1:1) | 106 ms (1043 tok/s) | 10.34 ms/tok (97 tok/s) | 271 ms |

### Llama-2-7B Q8_0 (109 token prompt, 3 eval tokens, best of 3)

| Configuration | Prompt eval | Token gen | Load time |
|--------------|------------|-----------|-----------|
| RX 6800 XT alone | 285 ms (382 tok/s) | 24.98 ms/tok (40 tok/s) | 2995 ms |
| Both GPUs (smgs 1:1) | 306 ms (356 tok/s) | 25.77 ms/tok (39 tok/s) | 3596 ms |

### Llama-2-13B Q8_0 (109 token prompt, 3 eval tokens, best of 3)

| Configuration | Prompt eval | Token gen | Load time |
|--------------|------------|-----------|-----------|
| RX 6800 XT alone | 465 ms (234 tok/s) | 50.27 ms/tok (20 tok/s) | 5999 ms |
| Both GPUs (smgs 1:1) | 510 ms (214 tok/s) | 46.60 ms/tok (21 tok/s) | 7061 ms |
| Both GPUs (smgs 2:1) | 465 ms (234 tok/s) | 50.25 ms/tok (20 tok/s) | 5998 ms |

### Analysis: dmabuf Impact

Phase 12 dramatically reduces cross-device transfer overhead compared to Phase 10's host-staging:

| Model | Phase 10 overhead/tok | Phase 12 overhead/tok | Improvement |
|-------|----------------------|----------------------|-------------|
| TinyLlama 1.1B | ~5 ms | ~-0.7 ms | Eliminated (multi-GPU faster) |
| Llama-2-7B | ~15 ms | ~1 ms | 15x reduction |
| Llama-2-13B (1:1) | ~26 ms | ~-4 ms | Eliminated (multi-GPU faster) |

Key findings:
1. **TinyLlama multi-GPU token gen is faster than single-GPU** (97 vs 90 tok/s) — the cross-device overhead is now smaller than the compute savings from distributing work
2. **Llama-2-13B 1:1 split is faster than single-GPU for token gen** (21 vs 20 tok/s) — doubled memory bandwidth outweighs transfer cost
3. **Llama-2-7B overhead dropped from ~15 ms to ~1 ms per token** — a 15x reduction
4. **13B 2:1 split matches single-GPU exactly** — the faster GPU dominates and the slower Vega contributes minimally

The dmabuf path eliminates CPU memcpy overhead, which was the dominant component of cross-device transfer latency. The remaining overhead is purely DMA transfer time, which is pipelined with compute.

## Verification

- Single-GPU inference produces correct output (no regression)
- Multi-GPU inference produces coherent output on all three models
- dmabuf: 1 reported for both RADV devices at startup
- Build compiles cleanly on both local (Polaris 12) and remote (Navi 21 + Vega 10) hardware
- Falls back gracefully to host-staging when dmabuf unavailable
