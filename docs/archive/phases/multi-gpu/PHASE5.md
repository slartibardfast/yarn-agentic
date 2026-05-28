# PHASE 5: Topology Discovery and Performance Tuning

**Goal**: Add topology awareness for informed tensor placement and transfer cost estimation.

## Status: COMPLETE

## Changes Made

**`ggml/src/ggml-vulkan.cpp`** (~25 lines):
- Added multi-GPU topology logging in `ggml_vk_instance_init`
- When >1 device is detected, logs each device's name, type (discrete/integrated/cpu/virtual), and VRAM size
- Uses `vk::PhysicalDeviceProperties` for device type and `vk::PhysicalDeviceMemoryProperties` for VRAM

## Design Notes

### What's already handled by existing infrastructure
- **VRAM-based split ratios**: `llama.cpp` already auto-calculates split ratios proportional to free memory when `tensor_split` is NULL (via `llama_get_device_memory` → `ggml_backend_vk_get_device_memory`)
- **Device type awareness**: The backend already handles UMA detection (`device->uma`), CPU device warnings, and driver priority selection
- **Device grouping**: `ggml_vk_instance_init` already deduplicates devices by UUID and selects optimal drivers per vendor

### What this phase adds
- **Startup topology log**: When multiple Vulkan devices are detected, a clear summary is printed showing device names, types, and VRAM. This helps users verify their multi-GPU configuration without needing debug flags.
- **lavapipe identification**: CPU-type devices are explicitly identified as "cpu (lavapipe)" in the topology log, since transfers to/from lavapipe are effectively zero-copy memcpy operations.

### Bandwidth estimation (deferred)
Runtime bandwidth measurement was considered but deferred. The host-staging bandwidth (~6 GB/s effective for PCIe 4.0 x16) is predictable and doesn't vary enough between runs to warrant runtime calibration. If asymmetric interconnects are encountered in practice, this can be added as a follow-up.

## Verification
- Full clean build with clang (zero errors, zero warnings from our files)
- `libggml.so` and `libllama.so` built successfully
