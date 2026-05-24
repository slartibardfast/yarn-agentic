---
name: project-t3-6-full-grid-closure
description: "PHASE_NSTREAM_KV_PERF T3.6 closure on production/2026-q2-next 2026-05-22. Full grid (I.b + I.c1 + I.c2 + M) green; n_stream==1 asserts lifted for all multi-stream KV paths under both LAYER and GRAPH split on Qwen 3.6 27B production target with Q4_0_AR16 (Hadamard) KV. I.b.2 (bailout drop) closed as a design decision rather than landed code."
metadata:
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

# T3.6 — Drop bailout + lift n_stream==1 guards — CLOSED

**Fact:** T3.6 closed on `production/2026-q2-next` 2026-05-22 across the full audit grid: I.b (SET_ROWS pass-through + bailout-drop design decision), I.c1.x2 (multi-stream `build_k_shift` LAYER + GRAPH), I.c2 (multi-stream `build_defrag` LAYER + GRAPH + generic CUDA Q→Q same-type cpy), M (graph-pool VRAM probe). The full T3.6 closure gate is green: verify-production-determinism PASS at NP={1,2,4,8} multi-GPU, r5-probe-c4 = 0/20, all three per-stream synthetic tests GREEN under both split modes, all three Allium specs + TLA+ MC modules clean, DFlash composition GREEN.

**Why:** PHASE_NSTREAM_KV_PERF Tier 3 unified-stream dispatch needed all `n_stream == 1` asserts and bailouts lifted so multi-stream graph emission could light up correctness paths on the production target with `--k-cache-hadamard --v-cache-hadamard` (Q4_0_AR16 KV). Without T3.6 closure, the codebase had T3.5's multi-seq dispatch in place but the downstream graph builders (K-shift, defrag, can_reuse) would either crash or silently corrupt on multi-stream inputs.

**How to apply:**
- T3.6's structural pattern — per-(device, stream) input tensors pinned via `ggml_backend_sched_set_tensor_backend` + `backend_override` on intermediate tmp + 3D-per-stream views into `splits[id]` — is now the load-bearing template for any future multi-stream graph builder. See [[feedback-per-device-per-stream-input-pattern]] and [[feedback-tmp-tensor-backend-pinning-under-graph-split]] for the K-shift specifics; see [[feedback-cuda-cpy-q-q-same-type-pattern]] for the defrag-specific CUDA cpy enhancement.
- The graph-reuse n_stream>1 bailout (`src/llama.cpp:629`) is INTENTIONALLY KEPT. The inline comment at 610–628 documents that dropping it would expose a single-seq cross-stream bug in `build_std_attention` (the single-seq K view bakes `kqv_stream_id * nb[3]` into its offset and is not reuse-safe across streams) while yielding no real reuse uplift (the multi-seq dispatch path already trips the n_tokens>1 MTP gate at reason=2 before reaching the n_stream check). If a future change makes the single-seq path reuse-safe or shifts workloads off the MTP gate, revisit the drop — VRAM probe + graph-reuse spec are in place to bind the comparison.
- T3.6.M VRAM probe is permanent. `~ggml_backend_cuda_context: have N graphs (M nodes, X KB host, Y KB device)` runs on every context teardown. LAYER vs GRAPH split shapes are distinguishable in the trace (LAYER aggregates into few large graphs, GRAPH split fragments into many smaller per-device sub-graphs). Use this trace as the "is the cache growing without bound" signal.

**Established:** PHASE_NSTREAM_KV_PERF T3.6 closure, 2026-05-22. Submodule commits `c9ef4e57` (T3.6.T synthetic tests), `4210e5b8` (I.b.1 SET_ROWS pass-through), `583c279d` (I.c1 multi-stream build_k_shift), `b62765be` (I.c1.x input-layer restructure), `69027ced` (I.c1.x2 GRAPH-split full fix), `1c84345d` (I.c2 multi-stream defrag), `8eb74b5a` (M VRAM probe). The next blocker on the PHASE doc is T3.8 (perf gate GP3.i) — out of scope for the T3 correctness phase; addressed in a separate session.
