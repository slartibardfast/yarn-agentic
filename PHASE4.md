# PHASE 4: Async Execution and Pipeline Parallelism

**Goal**: Remove cross-device serialization so the scheduler can run Vulkan backends in parallel.

## Status: COMPLETE

## Changes Made

**`ggml/src/ggml-vulkan.cpp`** (~20 lines):
- Added `std::mutex queue_mutex` to `vk_device_struct` (per-device, replaces global)
- Added `vk_device device` field to `vk_context_struct` so `ggml_vk_submit` can access the per-device mutex
- Updated `ggml_vk_create_context` to store device reference in context
- Updated `ggml_vk_create_temporary_context` to accept and store device reference
- Updated all 4 callers of `ggml_vk_create_temporary_context` to pass device
- Updated 3 `queue_mutex` usage sites (`ggml_vk_submit` x2, `event_record` x1) to use `ctx->device->queue_mutex`
- Removed global `static std::mutex queue_mutex`

## Design Notes
- The global `queue_mutex` serialized all queue submissions across all devices. With per-device mutexes, each VkDevice can submit independently.
- `graph_compute` already has good intra-device pipelining (batched submits with fence wait only on last batch).
- The scheduler's parallel execution path (`ggml-backend.cpp:2160-2380`) uses async functions + events, which are all wired (Phase 1) and functional (Phase 3 cross-device copy).
- Double-buffered staging (Phase 3 enhancement) deferred — the current synchronous staging is sufficient for initial multi-GPU operation.

## Verification
- Full clean build with clang (zero errors, zero warnings from our files)
- `libggml.so` and `libllama.so` built successfully
