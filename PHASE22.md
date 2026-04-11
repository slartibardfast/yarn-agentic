# Phase 22: Fix get_tensor_async Race Condition on rBAR Devices

## Status: COMPLETE

## Problem

Inference produces garbage output on AMD Vega 64 (RADV VEGA10) when `ngl > n_layers` (output layer on GPU). The same model and settings produce correct output on RX 6800 XT (RADV NAVI21). Backend-ops tests pass on both GPUs — only end-to-end inference is affected.

### Symptoms

- **Vega (ngl=99)**: `"The capital of France is0.\n91.\n7.\n2.\n6."`
- **6800 XT (ngl=99)**: `"The capital of France is Paris."`
- **Vega (ngl ≤ n_layers)**: correct output (output matmul stays on CPU)

## Root Cause

A race condition in `get_tensor_async` for devices with resizable BAR (rBAR/SAM).

### The data flow

1. `graph_compute` → submits GPU compute work, returns with `compute_pending = true`
2. Scheduler calls `get_tensor_async` → reads output tensor from GPU
3. Scheduler calls `synchronize()` → waits for GPU compute to finish

### The bug

`get_tensor_async` calls `buffer_read_2d_async` which checks if the source buffer is host-visible. On discrete GPUs with rBAR, device-local VRAM is host-visible (mapped via PCIe BAR). The function does a **synchronous memcpy** from the mapped GPU pointer — but GPU compute may still be in flight (step 2 happens before step 3).

On 6800 XT (faster GPU), the compute usually finishes before the CPU reads the buffer — the race is won by luck. On Vega (slower GPU), the CPU reads stale/uninitialized data.

### Why the synchronous path has a UMA guard

The synchronous `ggml_vk_buffer_read` (called from `buffer_get_tensor`) checks **both** `eHostVisible` and `uma` before doing a direct memcpy. For non-UMA discrete GPUs, it falls through to the staging buffer path (which is properly synchronized). But `buffer_read_2d_async` only checked `eHostVisible` — missing the UMA guard.

### Misleading clue

Inserting `submit + fence_wait` mid-graph (inside the output matmul dispatch) "fixed" Vega. This forced the GPU compute to complete before `get_tensor_async` ran, hiding the race. The output matmul's logits were verified correct via debug readback on Vega — confirming the shaders are fine.

## Fix

For host-visible non-UMA buffers, `get_tensor_async` now records a deferred memcpy in a new `pending_host_memcpys` vector on the backend context instead of copying immediately. `synchronize()` processes these after `sync_compute` waits for the compute fence.

### Why not use `transfer_ctx->out_memcpys`?

First attempt stored the deferred memcpy in the transfer context's `out_memcpys`. This failed because `sync_compute` → `graph_cleanup` clears `gc.contexts`, which holds the only strong reference to the transfer context. The weak_ptr `ctx->transfer_ctx` expires, and `synchronize()` exits early without processing the memcpys.

### Changes

**`ggml/src/ggml-vulkan.cpp`**:

1. Added `std::vector<vk_staging_memcpy> pending_host_memcpys` field to `ggml_backend_vk_context`
2. In `get_tensor_async`: for `eHostVisible && !uma` buffers, record deferred memcpy in `pending_host_memcpys` instead of calling `buffer_read_async`
3. In `synchronize()`: after `sync_compute` (which waits for the compute fence), process and clear `pending_host_memcpys`

**`tests/test-backend-ops.cpp`**:

4. Added large-M MUL_MAT tests (m=32000/128256/151936, realistic lm_head dimensions) to cover output matmul shapes. These all pass — the bug was in the host transfer path, not in the shaders.

## Verification

```
# Vega 64 — was garbage, now correct
GGML_VK_VISIBLE_DEVICES=0 llama-cli -m Qwen2.5-Coder-1.5B-Q8_0.gguf -ngl 99 \
  -p "The capital of France is" -n 20
→ "The capital of France is Paris. Paris is the largest city in the country..."

# 6800 XT — still correct
GGML_VK_VISIBLE_DEVICES=1 llama-cli -m Qwen2.5-Coder-1.5B-Q8_0.gguf -ngl 99 \
  -p "The capital of France is" -n 20
→ "The capital of France is Paris. Paris is the largest city in France..."

# Multi-GPU — correct
llama-cli -m Qwen2.5-Coder-1.5B-Q8_0.gguf -ngl 99 \
  -p "The capital of France is" -n 20
→ "The capital of France is Paris. Paris is a city located in the country of France..."

# Nemotron 4B on Vega — correct
GGML_VK_VISIBLE_DEVICES=0 llama-cli -m Nemotron-Nano-4B-Q4_K_M.gguf -ngl 99 \
  -p "The capital of France is" -n 20
→ "The capital of France is Paris."

# Backend-ops: same 6 pre-existing failures (prealloc contamination, unrelated)
```

## Affected Configurations

Any discrete GPU with rBAR/SAM where `eHostVisible` is set on device-local memory. The race is timing-dependent — faster GPUs may appear to work but are still technically racy. The fix is correct for all configurations:

- **UMA devices**: `buffer_read_2d_async` direct memcpy still used (no race — shared memory)
- **Discrete without rBAR**: staging buffer path used (properly synchronized via command buffer ordering)
- **Discrete with rBAR**: now uses `pending_host_memcpys` (deferred until after compute fence)
