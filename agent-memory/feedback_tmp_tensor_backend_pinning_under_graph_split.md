---
name: feedback-tmp-tensor-backend-pinning-under-graph-split
description: "Under graph-split, an intermediate tmp tensor (e.g. F32 cast of a Q4_0 K view for rope) MUST be pinned to the SAME device as its source K split — not to backends[0] via the legacy 'find first backend supporting layer's buft' pattern. The legacy pattern always picks backends[0] under CUDA_Split since every CUDA backend reports supporting CUDA_Split."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

# `tmp` tensor backend pinning under graph-split

**Rule:** When building a graph that operates on per-device K/V splits under `split_mode == GRAPH`, any intermediate tensor (cast, projection, accumulator) MUST be pinned to the SAME device as its source split. Use `ggml_backend_sched_set_tensor_backend(sched, tmp, lctx.backends[device_id])` directly with the device-specific backend.

**Anti-pattern (legacy, looks reasonable but always picks backends[0]):**

```cpp
// WRONG under graph-split — always pins to backends[0].
for (auto * backend : lctx.backends) {
    if (ggml_backend_supports_buft(backend, lctx.model.buft_layer[il].buft)) {
        ggml_backend_sched_set_tensor_backend(sched, tmp, backend);
        break;
    }
}
```

Under graph-split the layer's `buft` is `CUDA_Split`, and every CUDA backend (CUDA0, CUDA1, …) reports `ggml_backend_supports_buft(buft = CUDA_Split) = true`. The loop matches CUDA0 first and breaks → tmp lives on CUDA0 regardless of which split we're processing. When the consuming op operates on split[d=1] (CUDA1 storage), cast/rope/cpy crosses devices and either crashes (`cudaMemcpyPeerAsync` invalid arg) or silently slow-paths.

**Correct pattern:**

```cpp
// Per-device loop over kl_extra->splits[id]:
for (int id = 0; id < kl_extra->n_device; ++id) {
    auto * split = kl_extra->splits[id];
    if (!split) continue;
    ggml_backend_t split_backend = (id < (int)lctx.backends.size())
        ? lctx.backends[id] : nullptr;
    // ... build tmp ...
    if (split_backend) {
        ggml_backend_sched_set_tensor_backend(sched, tmp, split_backend);
    }
    // ... rope, cpy_back, build_forward_expand ...
}
```

This pattern keeps cast / rope / cpy all on the SAME device as the K split they're processing. Combined with per-(device, stream) inputs ([[feedback-per-device-per-stream-input-pattern]]) the graph has zero cross-device cpy.

**Generalisation:** `lctx.backends[id]` corresponds to the d-th GPU under graph-split (backends are pushed in `cparams.devices` order, CPU appended last). For any code that iterates over per-device splits, the d-th backend is the right tmp host.

**Why this didn't bite before:** In production, K-shift was gated off (IMROPE) so this code path never ran. The defrag path is gated on `kv_self.n_stream == 1` (assertion). The standard decode path doesn't create intermediate F32 tmp tensors from K splits — projection and rope are all done on Q/K/V directly on the per-device matmul outputs, not on K splits. So the existing graph builders sidestep the issue.

**Established:** PHASE_NSTREAM_KV_PERF T3.6.I.c1.x2 (2026-05-22). `build_one_rope` in `src/llama-build-context.cpp:build_k_shift` accepts a `backend_override` parameter; the call site at the per-device branch passes `lctx.backends[id]`.

**Apply to:** `build_defrag` (T3.6.I.c2), and any future per-device-split graph builder that creates intermediate compute tensors.
