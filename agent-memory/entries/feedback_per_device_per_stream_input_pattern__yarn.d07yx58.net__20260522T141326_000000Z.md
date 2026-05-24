---
name: feedback-per-device-per-stream-input-pattern
description: "Under graph-split (CUDA_Split) + per-stream dispatch, allocate per-(device, stream) input tensors each pinned to its consuming backend via ggml_backend_sched_set_tensor_backend. Single shared inputs + scheduler cross-device cpy hits a NULL-backend leaf-copy bug on freshly-built graphs under reset multi-device sched."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

# Per-(device, stream) input pattern under graph-split

**Rule:** For any per-stream graph input under split_mode == GRAPH (CUDA_Split buffer type), allocate N_DEVICE × N_STREAM independent input tensors, each pinned to its consuming CUDA backend via `ggml_backend_sched_set_tensor_backend`. The populator writes the same per-stream slice to every device's copy. Do not use:
- a single 1D input + per-stream views (the view's data pointer is captured at view-creation time, before the parent is host-allocated — scheduler cross-device cpy reads stale pointers);
- n_stream separate inputs not pinned to a device (scheduler tries to cross-device-cpy to per-device leaf copies, and those copies allocate with `[NULL]` backend on freshly-built graphs after `sched_reset` under multi-device).

**Why:** ggml's `ggml_backend_cuda_cpy_tensor_async` → `cudaMemcpyPeerAsync` path crashes with "invalid argument" when the source-side leaf copy's backend assignment is missing. The scheduler's `ggml_backend_sched_split_graph` only adds per-backend input copies to `graph_copy->leafs` when `n_copies > 1` (line 1908 in ggml-backend.cpp). Under default n_copies=1 with multi-device split, the per-backend leaf copies (`CUDAx#leaf_Y#0`) get `tensor_backend_id == -1` and aren't allocated by galloc. Cross-device cpy then reads from a NULL-data tensor.

Pinning each input tensor to its consuming backend at build time avoids the cross-device cpy entirely — each op reads from a local-device input. Eliminates the broken class of paths.

**How to apply:** When building per-stream inputs (e.g., in `build_k_shift`, future `build_defrag`, anywhere with similar shape):

```cpp
const uint32_t n_device = (split_mode == LLAMA_SPLIT_MODE_GRAPH)
                              ? model.splits.size() : 1u;

// 2D storage indexed [device_id][stream_id]
std::vector<std::vector<ggml_tensor *>> inp_per_dev_stream(
    n_device, std::vector<ggml_tensor *>(n_stream, nullptr));

for (uint32_t d = 0; d < n_device; ++d) {
    for (uint32_t s = 0; s < n_stream; ++s) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx0, dtype, size);
        ggml_set_input(t);
        if (n_device > 1 && d < lctx.backends.size()) {
            ggml_backend_sched_set_tensor_backend(sched, t, lctx.backends[d]);
        }
        inp_per_dev_stream[d][s] = t;
    }
}
```

`lctx.backends` is ordered `[CUDA0, CUDA1, ..., CPU]` so `backends[d]` matches the d-th model split.

The populator uses `ggml_backend_tensor_set(t, staging, 0, nbytes)` (host-staging-aware) rather than direct `t->data` memcpy — the per-device inputs may live on CUDA device buffers, not host pinned memory.

**Companion rule:** the intermediate `tmp` tensor (cast / projection / etc.) consuming such inputs must also be pinned to the SAME device as its data source — see [[feedback-tmp-tensor-backend-pinning-under-graph-split]].

**Established:** PHASE_NSTREAM_KV_PERF T3.6.I.c1.x2 (2026-05-22). Per-(device, stream) `inp_K_shift_per_stream` in `src/llama-decoder-internal.h` + `src/llama-build-context.cpp:build_k_shift` + `src/llama.cpp:llama_set_k_shift`. Verified on Qwen 3.6 27B production target.
