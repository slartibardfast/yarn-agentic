---
name: dflash-multislot-phase12-landed
description: "DFlash multi-slot API extension Phases 1 (C surface) and 2 (per-slot scratch sizing) landed on production/2026-q2-next 2026-05-18; n_slots==1 byte-identical to pre-Phase-1; Phase 3 (cb_eval seq_id demux) is next"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

DFlash multi-slot libllama API extension progress on `production/2026-q2-next` on 2026-05-18.

**Plan:** `data/dflash-multi-slot-impl-plan-2026-05-18.md` (6 phases).
**Pickup brief for next phase:** `data/dflash-phase3-pickup-2026-05-18.md`.

## Phases landed this session

**Phase 1 — C API surface** (commits `8008feaf` submodule, `d708299` parent):
- Added `llama_dflash_draft_batch(ctx, n_slots, anchor_token_ids[], anchor_positions[], seq_ids[], out_candidates[], max_total_candidates)` to `include/llama.h:1808`.
- Stub in `src/llama-dflash.cpp:708`: n_slots==1 trampolines to existing `llama_dflash_draft` (byte-identical); n_slots>1 returns `LLAMA_DFLASH_NP_GT_1` (first site to actually return that enum).
- Stub variant for non-CUDA build returns `LLAMA_DFLASH_NOT_IMPLEMENTED`.

**Phase 2 — Per-slot scratch sizing** (commits `fa3e50c7` submodule, `7d997a5` parent):
- `llama_dflash_ctx_state.n_slots_cap` field added.
- `allocate_ctx_scratch(st, dw, seq_len_cap, mal_cap, n_slots_cap)` scales all slot-major buffers by n_slots_cap. Layouts match the kernel cuh comments (K/V cache `[L_d, N_slots, SeqLen, H_kv, D_h]`; ctx_states/anchor_pos/input_emb/etc `[N_slots, ...]`).
- `llama_set_dflash` captures `n_slots_cap = max(cparams.n_seq_max, 1u)`.
- Per-layer KV stride in dispatch updated to `n_slots_cap * SeqLen * H_kv * D_h`.
- Single-slot dispatch still passes `N_slots=1` to kernels (only the allocation expands).
- At n_slots_cap==1 (default for non-spec contexts), Phase 2 is a no-op for memory.

## Verifications passed at handover

1. `scripts/verify-production-determinism.sh` PASS at NP={1,2,4,8} multi-GPU (re-run 2026-05-18T17:33, pre-Phase-1).
2. `test-dflash-closure`: 8/8 prompts PASS argmax-equivalent against the vLLM dump after each phase landing. Confirms kernel-pipeline binding is byte-identical to pre-Phase-1 at n_slots_cap=1.
3. Build clean: `cmake --build build -j 32 --target llama llama-server llama-batched-bench test-dflash-closure`.

## Why Phase 3 is the architectural risk

The `cb_eval` hook at `src/llama.cpp:9661` (`llama_dflash_extract_cb_eval`) fires once per ubatch and appends ALL rows of `l_out-<il>` to `ctx->default_decoder.dflash_extract_buf[slot]` without any seq_id information. At np>1, rows from multiple slots interleave into that single buffer with no demux info.

The current MAL accounting (`MAL = anchor_pos` at `src/llama-dflash.cpp:614`) implicitly assumes one sequence per context.

Phase 3 must wire seq_id awareness through the cb_eval path — the ubatch's `seq_id[j]` per row is the source of truth, but it's not visible to the cb_eval hook.
