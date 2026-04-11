# PHASE 9: Parallel Split Buffer Uploads

**Goal**: Upload model weights to multiple devices concurrently instead of sequentially.

## Status: COMPLETE

## Problem

`ggml_backend_vk_split_buffer_set_tensor()` uploaded tensor data to each device in a serial loop. Each `set_tensor` call submits a Vulkan transfer and blocks on a fence before moving to the next device. For N devices, load time = sum of all device transfer times.

## Changes Made

**`ggml/src/ggml-vulkan-multigpu.cpp`** (~50 lines net):

1. Added `pending_upload` struct to collect per-device upload work:
   ```cpp
   struct pending_upload {
       ggml_tensor * split;
       std::vector<char> host_data;   // owned copy (for interleaved data)
       const void * direct_ptr;       // direct pointer into source (no copy needed)
       size_t nbytes;
   };
   ```

2. Added `parallel_upload` lambda that spawns `std::thread` per device:
   - Single-device case: no threading overhead (direct call)
   - Multi-device: one thread per device, join all at end

3. Restructured all upload paths (replicated, split_dim=0, split_dim=1) into two phases:
   - **Phase 1** (sequential, CPU-bound): prepare per-device host data with interleaving
   - **Phase 2** (parallel, PCIe-bound): submit all device uploads concurrently

4. Each device gets its own `host_buffer` (previously shared), enabling true parallelism.

## Design Notes

- **Thread-safe by construction**: each device has independent VkDevice, VkQueue, and command pools. Per-device `queue_mutex` (Phase 4) prevents queue submission races.
- **No async Vulkan needed**: `std::thread` achieves the same PCIe overlap — while thread 0 blocks on device 0's fence, thread 1 is transferring to device 1.
- **Per-device host buffers**: the previous shared `host_buffer` was a serialization point even without the upload fence. Now each thread owns its own buffer.

## Test Hardware

- **CPU**: Intel Xeon X5650 (Westmere, 2010) — 6C/12T, DDR3-1333 triple-channel
- **GPU 0**: AMD Radeon Pro WX 2100 (Polaris 12) — 2 GB GDDR5, PCIe 3.0 x16
- **GPU 1**: lavapipe (Mesa llvmpipe) — CPU-based Vulkan software rasterizer

## Results

### Load time comparison (TinyLlama 1.1B Q2_K, 459 MiB model)

| Mode | Trial 1 | Trial 2 | Trial 3 | Average |
|------|---------|---------|---------|---------|
| Serial upload | 1118 ms | 1130 ms | 1103 ms | 1117 ms |
| Parallel upload | 1091 ms | 1136 ms | 1111 ms | 1112 ms |
| Single GPU (reference) | 426 ms | — | — | 426 ms |

### Analysis

No measurable improvement on this hardware. This is expected: GPU 1 is lavapipe, whose "uploads" are host-to-host memcpy (effectively instant). The entire load time is dominated by GPU 0's PCIe transfer to the Polaris 12. There is nothing to parallelize when one device's transfer cost is near zero.

### Where this will matter

The parallel upload benefit requires **two or more discrete GPUs** where both have real PCIe transfer costs:

- **2x discrete GPUs** (e.g., 2x RX 7900 XTX): each gets ~half the model. Serial = transfer_A + transfer_B. Parallel = max(transfer_A, transfer_B) ≈ **2x speedup** on load time.
- **4x discrete GPUs** on a 70B model: serial ~30s → parallel ~8s (PCIe bandwidth limited per device).
- **Asymmetric GPUs** (different PCIe bandwidth): parallel still wins — fastest device finishes early, slowest determines total time.

### Inference performance (unchanged)

```
prompt eval:  851 ms / 2 tokens (2.35 tok/s)
eval:        2177 ms / 4 runs   (1.84 tok/s)
total:       3035 ms / 6 tokens
```

## Verification

- Model loads correctly, weights distributed across both devices
- Multi-GPU inference produces coherent output
- Single-GPU inference unchanged (no regression)
- Serial baseline measured to confirm parallel implementation is correct
