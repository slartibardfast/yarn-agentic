---
name: project-t3-6-kshift-graph-split-closure
description: "PHASE_NSTREAM_KV_PERF T3.6.I.c1 + .x + .x2 chain landed 2026-05-22 — per-(device, stream) inp_K_shift + backend_override on intermediate tmp = full K-shift correctness under both LAYER and GRAPH split on Qwen 3.6 27B. Removes a long-standing limitation hidden behind the IMROPE gate."
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

# T3.6 K-shift GRAPH-split closure (2026-05-22)

**Branch:** `production/2026-q2-next`.
**Final submodule HEAD:** the T3.6.I.c1.x2 commit on top of `b62765be`.

**What landed (chain):**
- **T3.6.I.c1** — per-stream rope loop in `build_k_shift` (lifts `kv_self.n_stream == 1` assert) + IMROPE gate lift in `get_can_shift` (matches upstream's stance that IMROPE supports K-shift via the NEOX rope workaround).
- **T3.6.I.c1 follow-up** — split-aware `build_k_shift` that iterates `kl_extra->splits[id]` under graph-split, plus defensive graph-split gate (clean rc=1, no CUDA crash).
- **T3.6.I.c1.x** — restructured `inp_K_shift` from single 1D + per-stream views to `n_stream` separate input tensors (still failed under graph-split, ruled out view-aliasing as root cause).
- **T3.6.I.c1.x2** — restructured further to per-(device, stream) input tensors, each pinned to its consuming backend; added `backend_override` param to `build_one_rope` so the intermediate F32 tmp tensor lives on the same device as its K split. **Removed the graph-split gate.** Both modes now fully correct.

**Why:** the GRAPH-split + K-shift code path had been latent-broken because the IMROPE gate prevented it from ever running on Qwen 3.x text models. Lifting the IMROPE gate exposed two underlying bugs simultaneously:
1. The scheduler's leaf-input allocation for fresh-graph builds on a reset multi-device scheduler under `n_copies==1` produces `[NULL]` backend assignments — `cudaMemcpyPeerAsync` then crashes with "invalid argument" (see [[feedback-per-device-per-stream-input-pattern]]).
2. The intermediate F32 tmp tensor was always pinned to `backends[0]` via the legacy `for(auto * backend : lctx.backends) ... break;` pattern, because every CUDA backend reports supporting CUDA_Split. tmp on CUDA0 + K view on CUDA1 → cast/rope/cpy crosses devices (see [[feedback-tmp-tensor-backend-pinning-under-graph-split]]).

**How to apply:**
- When implementing per-stream multi-device graph operations (next up: `build_defrag`), use the per-(device, stream) input pattern from the start. Don't share inputs across devices.
- When emitting any intermediate compute tensor that consumes per-device-split data, pin it via `ggml_backend_sched_set_tensor_backend(sched, tmp, lctx.backends[id])` — not via the legacy buft-match loop.
- The defensive both-mode `tests/spec/test-kv-shift-per-stream.cpp` is the regression gate; any future refactor must keep both `[LAYER]` and `[GRAPH]` branches GREEN.

**Verification:**
- Defensive test: `[LAYER]` and `[GRAPH]` both PASS on Qwen 3.6 27B production target with `pos_max[seq2]: 3→8`, isolation preserved.
- `bin/test-dflash-np-multislot` GREEN — slot-0 byte-identical across NP={1,2,4,8} on lm_head-f16 recast target.
- `bin/test-dflash-np-invariance` GREEN — 4/4 seeds, drafter_forward np-invariant.
- `bin/test-dflash-closure` GREEN — 8/8 prompts argmax-equivalent vs vLLM.
- `bash scripts/verify-production-determinism.sh` ACCEPTANCE PASS at DEVICE=CUDA0,CUDA1, NP={1,2,4,8}.

**Remaining T3.6 work:**
- T3.6.I.c2 — multi-stream `build_defrag` (same per-(device, stream) pattern will apply; defrag has its own n_stream==1 assert to lift).
- T3.6.I.b.2 — bailout drop in `can_reuse_graph` (separate cross-stream issue; needs `prev->kqv_stream_id` tracking).
- T3.6.M — VRAM + reuse perf probe.
- T3.6.C — closure with PHASE + MEMORY commits and final verify.
