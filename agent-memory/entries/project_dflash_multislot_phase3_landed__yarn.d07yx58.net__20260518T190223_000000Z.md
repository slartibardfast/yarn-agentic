---
name: dflash-multislot-phase3-landed
description: "DFlash multi-slot Phase 3 (cb_eval per-seq demux) landed on production/2026-q2-next 2026-05-18; n_seq_max=1 byte-identical to pre-Phase-3; per-seq demux verified by new test-dflash-extract-multi-seq; NPC PASS at NP={1,2,4,8}; Phase 4 (multi-slot dispatch) is next"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

DFlash multi-slot libllama API extension Phase 3 on `production/2026-q2-next` on 2026-05-18.

**Plan:** `data/dflash-multi-slot-impl-plan-2026-05-18.md` (6 phases).
**Pickup brief used:** `data/dflash-phase3-pickup-2026-05-18.md`.
**Prior phases:** [[dflash-multislot-phase12-landed]].

## Phase 3 landed (commits `c33f75da` submodule, `32f5c69` parent)

**The Phase 3 problem:** at n_seq_max>1 multiple seq_ids share a target ubatch; the cb_eval hook (`llama_dflash_extract_cb_eval` at `src/llama.cpp:9661`) was appending ALL rows of `l_out-<il>` to a single flat per-source-layer buffer with no seq_id information. Downstream `stage_target_hiddens` had no way to demux, so MAL anchors were corrupt at np>1.

**Phase 3 chosen design (Option A from the pickup brief):**
- Storage `dflash_extract_buf[80]` becomes nested `std::vector<std::vector<float>>` per source-layer slot; inner dim sized to `cparams.n_seq_max` at `llama_set_dflash_extract_layers` time (memory bounded by `n_seq_max`, not 80×80).
- New `default_decoder.dflash_ubatch_row_seq` (`vector<llama_seq_id>`) populated by `llama_decode_internal` after `u_batch.seq_id` is set, before `sched_graph_compute_async`. Source of truth for each row's primary seq_id (`u_batch.seq_id[i][0]`).
- cb_eval reads row_seq[row] and appends per-row to `buf[slot][sid]`. F32 path uses per-row `ggml_backend_tensor_get`; F16 path single tensor_get + per-row `ggml_fp16_to_fp32_row`.
- `stage_target_hiddens` accepts `llama_seq_id seq_id` (current single-slot dispatch passes 0); `trim_extract` iterates per-seq buffers.
- New public API `llama_get_dflash_extract_data_seq(ctx, idx, seq_id, dst, max)`. Existing `llama_get_dflash_extract_data` becomes a trampoline targeting seq_id=0.

**Key source-read finding:** ik_llama.cpp has no `llama_ubatch` type; the decode driver uses `u_batch` of `llama_batch` (built at `src/llama.cpp:5229` inside `llama_decode_internal`'s `for cur_token` loop). Per-row primary seq_id is `u_batch.seq_id[i][0]`. The plumbing site is right after `u_batch.seq_id` finalization at line ~5290, before any compute.

## Verifications passed at landing

1. Build clean: llama, llama-server, llama-batched-bench, test-dflash-closure, test-dflash-extract-multi-seq.
2. `test-dflash-closure`: 8/8 prompts argmax-equivalent vs vLLM dump — confirms n_seq_max=1 single-slot path is byte-identical to pre-Phase-3.
3. `test-dflash-extract-multi-seq` (new): n_seq_max=2 with 2 seq_ids × 7 tokens — per-seq buffer counts both equal 7×n_embd=35840 floats. Pre-Phase-3 would have yielded seq0=71680 floats, seq1=0 floats.
4. `scripts/verify-production-determinism.sh`: NPC harness PASS at NP={1,2,4,8} multi-GPU byte-identical.

## What Phase 4 still needs to do

Phase 4 lifts `const int N_slots = 1` at `src/llama-dflash.cpp:621` (now obsolete given Phase 2 sized scratch for n_slots_cap and Phase 3 demuxed extract). `llama_dflash_draft_batch` should:
- Stage target hiddens for each active slot's seq_id (calling `stage_target_hiddens` per seq_id, or batching into one upload covering all N_slots).
- Run combine/inject/forward/lm_head with `N_slots = n_slots` from the API arg.
- Argmax per slot, writing slot-major into `out_candidates`.

Phase 5 wires `common_speculative_state_dflash` to pass seq_id at construction (line 1281 of `common/speculative.cpp` currently discards it).
