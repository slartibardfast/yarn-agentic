---
name: fattn-per-slot-kv-p2-landed-kernel-only
description: P2 (wmma_f16-pb1 dispatch for GGML_OP_FLASH_ATTN_EXT_PER_SLOT_KV) is landed on production/2026-q2-next 2026-05-15. FA kernel-level NP-invariance bound. Server-level NP>1 byte-identity NOT delivered — non-FA ops contribute. Step 7 stays OPEN.
metadata: 
  node_type: memory
  type: project
  originSessionId: 9b52ee2b-d969-442c-83ba-98396525f6fa
---

Per `specs/deltanet/fattn-per-slot-kv-sm75.md §15.6 / §15.7 / §15.8 / §15.9` and `PLAN.md` Empirical results section.

**What landed on production/2026-q2-next 2026-05-15** (under `LLAMA_FATTN_PER_SLOT_KV_ENABLE=1`):

- ggml op `GGML_OP_FLASH_ATTN_EXT_PER_SLOT_KV` src[5] takes `per_row_k_bound` (i32, length q->ne[1]).
- Build-graph emits it for all Qwen 3.5/3.6 production-shape FA calls (Dq=Dv=256, gqa<=16); no ne[1] gate.
- llama.cpp set_inputs populates per-row bound = max(seq_pos_max(seq_ids of row i)) + 1.
- CUDA dispatcher routes the new op to `ggml_cuda_flash_attn_ext_wmma_f16_case_pb1<256, 256, 8, half>` (NEW thin wrapper in `fattn-wmma-f16.cuh` that pins `parallel_blocks=1` regardless of heuristic).
- The bespoke fattn-per-slot-kv-sm75.cu kernels are DEPRECATED from production routing, kept compiled for unit-test reference.

**Per_row_k_bound is plumbed but unused for correctness.** The existing per-row mask already pushes past-bound K positions to -inf bit-identically (proof in §15.7). The bound is available for a future perf optimization (loop-tail trim).

**Bound status**:

- FA kernel-level NP-invariance: GREEN (`test-fattn-per-slot-kv-dispatch-np-invariance` unit test PASS; E3 intra-batch agreement at NP=4 concurrent decode confirms at the server).
- Server-level same-prompt byte-identity across NP={1,2,4,8}: RED. NP=1 fully reproducible (E1 PASS); NP=4 sequential split (2/3 match NP=1, 1/3 diverges); NP=4 concurrent has 1 solo slot matching NP=1 and 3 batched slots producing a mutually-identical-but-different output.

**Remaining divergence sources** (not addressed by P2):

- Non-FA shape-dependent ops: Q/K/V projection matmuls, RoPE, RMSNorm, MLP, output projection — all have heuristics that pick tile / launch shape based on M (token dim). Solo decode (ne[1]=1) vs batched decode (ne[1]>1) produces different floats.
- CUDA graph cache warm-up at NP>1 — E2 NP=4 sequential showed run2 differing from run1+run3 (all serial, identical-shape requests by hypothesis), suggesting some per-request state varies that doesn't at NP=1.

**Step 7 stays OPEN** per CLAUDE.md §4 "no follow-up cover". The kernel-level binding is delivered; the server-level broader contract requires shape-independence across many ops and is a separate workstream.

Related: [[project_mtp_multislot_determinism_investigation_failed]] (prior session found "the real bug is somewhere else" after burning ~50 iterations on the multi-slot determinism problem — consistent with §15.8 findings).
Related: [[feedback_no_skipping_lessening]] (don't declare structural; instrument before deciding broader work is infeasible).
Related: [[project_continuous_batching_vs_perslot_dispatch]] (multi-slot architecture analysis at ik_llama).
