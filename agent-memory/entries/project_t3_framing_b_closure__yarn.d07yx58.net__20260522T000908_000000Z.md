---
name: project-t3-framing-b-closure
description: "PHASE_NSTREAM_KV_PERF Tier 3 correctness closure (Framing B) landed 2026-05-22 — split_equal unified-stream dispatch in process_batch_tokens, multi-seq K/V WRITE via ggml_set_rows, DFlash composition GREEN. T3.6 bailout drop + T3.8 perf gate remain open."
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

# Tier 3 correctness closure landed 2026-05-22

**Branch:** `production/2026-q2-next`.
**Parent HEAD:** `51c86bb` (T3.5 submodule bump) + `aa1fa02` (PHASE reopen) + the closing PHASE commit on top.
**Submodule HEAD:** `be5d756e` (T3.5 split_equal dispatch) on top of `80de82d3` (T3.3-followup K/V WRITE via ggml_set_rows).

**Why:** PHASE_NSTREAM_KV_PERF's correctness closure path. After T3.3 was reopened (the K/V WRITE side was previously unaddressed), T3.3-followup integrated `ggml_set_rows` into the K/V WRITE path and added per-(token, head) global row indices via a new `inp_kv_idxs` input tensor. T3.5 then activated the multi-seq dispatch from the server side via split_equal grouping in `process_batch_tokens`.

**How to apply:**
- The multi-seq build path is now LIVE in production: `verify-production-determinism.sh` shows 55/64 dispatches use the multi-seq path during the np=8 segment via the permanent `dispatch_multi_seq_count` server-internal metric.
- When debugging multi-seq dispatch issues, check the metric in the server log first (LLAMA_LOG_INFO emit every 64 dispatches). If `multi_seq=0`, the dispatch path is bypassed and you're hitting the legacy single-seq fallback.
- Multi-seq grouping requires seq_ids contiguous from 0 in order (`[0, n_seq_in_batch)`). Non-contiguous seq sets fall back to single-seq dispatch — this is by design (the K/V view's parent->nb[3] stride only maps each ne[3] index to a contiguous stream slice).
- Per-device n_head_q under graph-split is derived empirically as `ggml_nelements(Qcur) / (n_embd_head_k * n_tokens)`. The build-context member `n_head` is the FULL n_head from hparams — using it for the per-device reshape under graph-split gives a wrong result. See `build_std_attention` for the empirical derivation; never substitute back to the member.
- T3.6 (drop the n_stream>1 bailout in `can_reuse_graph`) and T3.8 (perf gate) remain pending. The current state has multi-seq graphs build fresh per call (no graph reuse for multi-seq). Per the previous A/B, graph reuse delivered ~0% at production shape, so this is a correctness rather than perf step.
