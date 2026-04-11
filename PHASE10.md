# PHASE 10: True Async Graph Compute

**Goal**: Allow the scheduler to overlap graph execution across multiple Vulkan backends.

## Status: COMPLETE

## Problem

`ggml_backend_vk_graph_compute()` blocks until the compute fence signals:

1. Submits all compute work via `vkQueueSubmit`
2. Spin-waits on fence with `YIELD()` pauses
3. Only returns after all GPU work is done

This means the scheduler cannot submit work to GPU B while GPU A is still computing. The scheduler's parallel execution path (`ggml-backend.cpp:2160-2380`) requires async `graph_compute` + explicit `synchronize()` to overlap backends.

## Changes Made

**`ggml/src/ggml-vulkan.cpp`** (~40 lines net):

1. **Added `compute_pending` and `pending_compute_exit_ctx` to `ggml_backend_vk_context`**:
   - `compute_pending`: flag indicating graph_compute submitted work but fence not yet waited on
   - `pending_compute_exit_ctx`: holds the exit subctx alive so deferred `out_memcpys` survive until synchronize

2. **Added `ggml_vk_sync_compute()` helper** (after `ggml_vk_graph_cleanup`):
   - Waits on the pending compute fence via `ggml_vk_wait_for_fence`
   - Processes deferred `out_memcpys` from the exit context
   - Calls `ggml_vk_graph_cleanup` to reset command pools and release resources
   - Clears the `compute_pending` flag

3. **Modified exit tensor path in `ggml_vk_build_graph`**:
   - Still submits last batch with `ctx->fence` (unchanged)
   - No longer waits on fence at exit tensor — deferred to synchronize
   - Saves exit subctx as `ctx->pending_compute_exit_ctx` instead of processing `out_memcpys`

4. **Modified `ggml_backend_vk_graph_compute`**:
   - Safety check at start: `ggml_vk_sync_compute(ctx)` ensures previous async compute is complete
   - Perf logger path: waits for fence synchronously before reading timestamps, then cleans up (unchanged behavior)
   - Non-perf-logger path: sets `compute_pending = true` and returns immediately (no fence wait, no cleanup)

5. **Modified `ggml_backend_vk_synchronize`**:
   - Calls `ggml_vk_sync_compute(ctx)` at the top, completing any pending async compute before processing cross-device copies

6. **Modified `ggml_backend_vk_cpy_tensor_async`** (cross-device path):
   - Calls `ggml_vk_sync_compute(ctx_src)` on source device before reading from it
   - Ensures source device's compute is complete before cross-device copy

## Design Notes

- **`synchronize` already wired** (Phase 1): the vtable entry was set in Phase 1. Phase 10 just adds compute fence handling to the existing function.
- **Scheduler contract**: `ggml_backend_sched_graph_compute` calls `graph_compute_async` + `sched_synchronize` (which synchronizes all backends). The final sync is guaranteed.
- **Sequential scheduler path**: between splits, `cpy_tensor_async` triggers `sync_compute` on the source device before reading its output. This ensures correctness without explicit barriers.
- **Parallel scheduler path** (OpenMP/std::barrier): barriers sync CPU threads; `sync_compute` in `cpy_tensor_async` handles GPU synchronization.
- **Thread safety**: `ggml_vk_sync_compute` may be called from a different thread than the one that ran `graph_compute` (e.g., the copy thread in the parallel scheduler). This is safe because the computing thread is at a barrier, and all accessed state is per-context.
- **Perf logger compatibility**: when `vk_perf_logger_enabled`, graph_compute waits synchronously and cleans up before returning, preserving the old behavior needed for timestamp reads.

## Test Hardware

See [Benchmarks](BENCHMARKS.md) for full hardware specs (System B: RX 6800 XT + Vega).

## Results

### Llama-2-7B Q8_0 (102 token prompt, 3 eval tokens, best of 3)

| Configuration | Prompt eval | Token gen | Load time |
|--------------|------------|-----------|-----------|
| RX 6800 XT alone | 141 ms (724 tok/s) | 17.55 ms/tok (57.0 tok/s) | 2023 ms |
| Both GPUs (smgs 1:1) | 295 ms (345 tok/s) | 26.03 ms/tok (38.4 tok/s) | 3594 ms |

### Llama-2-13B Q8_0 (102 token prompt, 3 eval tokens)

| Configuration | Prompt eval | Token gen | Load time |
|--------------|------------|-----------|-----------|
| RX 6800 XT alone | 259 ms (395 tok/s) | 32.23 ms/tok (31.0 tok/s) | 4245 ms |
| Both GPUs (smgs 1:1) | 496 ms (206 tok/s) | 47.28 ms/tok (21.2 tok/s) | 7061 ms |
| Both GPUs (smgs 2:1) | 452 ms (226 tok/s) | 50.84 ms/tok (19.7 tok/s) | 6010 ms |

### Analysis

Phase 10 reduces the multi-GPU overhead ratio compared to single-GPU:

| Model | Multi/Single ratio (pre-Phase 10) | Multi/Single ratio (Phase 10) |
|-------|----------------------------------|------------------------------|
| Llama-2-7B | 1.58x | 1.48x |
| Llama-2-13B | 1.69x | 1.47x |

The improvement comes from the CPU being able to prepare the next graph split's inputs while the current split is still executing on the GPU. The fence wait is deferred until the output is actually needed (at the next cross-device copy or final synchronize).

## Verification

- Single-GPU inference produces correct output (no regression)
- Multi-GPU inference produces coherent output
- Model loads and runs correctly on both single and multi-GPU configurations
- Build compiles cleanly on both local and remote hardware
