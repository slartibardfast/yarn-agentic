---
name: feedback-t3-per-device-n-head-empirical
description: "Under graph-split multi-device build, derive per-device n_head from ggml_nelements(tensor) / (head_dim * n_tokens), NOT from the build-context member n_head or from split_wq->ne[1] / n_embd_head_k."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

# Per-device n_head derivation under graph-split

**Rule:** When reshaping a per-device Q (or K, V) tensor in `build_std_attention` under graph-split mode, derive the per-device head count empirically as `ggml_nelements(tensor) / (head_dim * n_tokens)`. Do NOT use the build-context member `n_head` (it is the FULL n_head from hparams). Do NOT use `split_wq->ne[1] / n_embd_head_k` either — the matmul layout doesn't necessarily map to the head count via ne[1].

**Why:** Two incorrect attempts during PHASE_NSTREAM_KV_PERF T3.5 (2026-05-21/22):
1. First using `n_head` (full) → `ggml_reshape_4d` nelements assertion at ggml.c:8519 because per-device Qcur has only half the elements (graph-split with 2 devices).
2. Second using `split_wq->ne[1] / n_embd_head_k` → same assertion fires because that calculation also gives the full count (the matmul output dim doesn't split — only ne[0] of the per-device matmul is reduced).

The empirical derivation works because `Qcur` is the actual tensor in this function call and its `ggml_nelements()` is the source of truth for what the reshape can map to. Diag verified: Qwen 3.6 27B on 2-GPU graph-split shows `Qcur ne=[256, 12, 2, 1]` with `n_head(member)=24` — per-device n_head = 12 (half of full 24, matching n_device=2). The empirical formula gives 12; both alternatives gave 24.

**How to apply:** When working in `build_std_attention` (or any other multi-device split path that reshapes per-device tensors to 4D for the unified-stream FA dispatch), compute the local head count from the tensor's actual element count. Add the assertion `qcur_nelem % (head_dim * n_tokens) == 0` as a defensive check.
