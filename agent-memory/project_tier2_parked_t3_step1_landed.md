---
name: tier2-parked-t3-step1-landed
description: "PHASE_NSTREAM_KV_PERF — Tier 2 (read-view patching) parked after deep diag, Tier 3 Step 1 (PSKV kernel nb33 mask addressing) landed as foundation"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

`production/2026-q2-next` 2026-05-21 (submodule c2a142a4): Tier 2 simple read-view patching mechanism is wired end-to-end (FA op carries K/V slot in op_params[14]/[15], ggml-cuda hook populates GPU table from FA->src[1]/src[2] per tick, PSKV kernel redirects via slot) but does NOT produce byte-identical output with the n_stream>1 bailout dropped. NPC fails IDENTICALLY across three indirection variants (no indirection / VIEW op_params / FA op_params) AND with GGML_CUDA_DISABLE_GRAPHS=1.

Deep diagnostic confirms:
- K_view->data IS being patched correctly per-tick (alternates between stream bases via launcher debug).
- Kernel reads K_direct = patched value (kernel-side printf confirms K == K_direct, K_eq=1).
- ne[1] is bucketed to 256 (FA padding) → constant within reuse window.

Root cause NOT localised. The simple read-view patching of Tier 2 is structurally insufficient. Pivoted to Tier 3 unified-stream dispatch (the PHASE_NSTREAM_KV_PERF.md plan's structurally-correct path).

T3 Step 1 landed: PSKV singlewarp kernel (`fattn-per-slot-kv-singlewarp-sm75.cu`) now takes `nb33` and addresses mask as `mask + nb33*seq + nb31*tok`. At ne[3]=1 (current production) seq=0 so nb33*0=0 — backwards-compatible. Foundation for unified-stream dispatch where mask is [n_kv, n_tokens_per_seq, 1, n_stream].

Remaining T3 work (substantial, future session):
- find_slot multi-seq support (allocate cells per token's seq_id into correct stream's slice).
- llama_decode multi-seq batch handling (n_kv per-stream, head per-stream).
- Build context: 4D K/V/mask when n_stream > 1 AND batch has multiple seqs.
- KQ_mask 4D [n_kv, n_tok_per_seq, 1, n_stream].
- Q reshape to 4D [head_dim, n_head, n_tok_per_seq, n_stream] in build path.
- server-context: NOT split process_batch_tokens by seq_id (unified batch).
- Drop n_stream>1 bailout in can_reuse_graph.
- NPC + perf gates per PHASE_NSTREAM_KV_PERF.md GP3.a–GP3.n.

Why: Production NP=8 = 27.73 t/s aggregate. Per-slot wait time at 8 slots ≈ 250ms (~9x slower than NP=1 single-slot 30ms). Unified-stream dispatch parallelizes all 8 slots in ONE graph evaluation → expected near-8x speedup.

How to apply: Next session, start from submodule c2a142a4. PSKV kernel mask addressing is done; start with find_slot multi-seq extension. Plan in PHASE_NSTREAM_KV_PERF.md §"Tier 3 — Unified-stream dispatch + ragged FA". Upstream llama.cpp reference at `/home/llm/yarn-agentic/llama.cpp/src/llama-kv-cache.cpp:1383` (4D K view) and `:849` (split_equal ubatch).
