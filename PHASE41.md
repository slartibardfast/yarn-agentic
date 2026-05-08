# Phase 41 — DeltaNet K-Slot Extension (Foundation)

Branch: `phase41-tree-foundation` off `phase40-tree-fanout` (parent + ik_llama.cpp).

## Goal

Extend ik_llama.cpp's DeltaNet recurrent state to support K extra slots per
main `seq_id`, so Phase 40's existing tree-K=2 implementation (depth=1)
runs without crashing on `qnext_state_slots`. Validate end-to-end on
Qwen 3.6 27B; close on measured data.

## Context

Phase 40 implemented top-K=2 fan-out tree drafting end-to-end on Phase 38
base. Two close reasons:

1. **Probe data**: α(top-2) − α(top-1) = 0.06 at long-context X02 → projected
   +3.2% lift. Marginal at depth=1.
2. **Architectural blocker**: Qwen 3.6's DeltaNet hybrid recurrent layers
   fail `GGML_ASSERT(s < qnext_state_slots)` because the state slot buffer
   is sized to `n_parallel`, so transient branch seq_ids overflow it.

Phase 41 unblocks the assert. Phase 42 lifts to depth=2 K=2 and is gated
on Stage 2 probes (α₂ ≥ 0.85, batch=2 ≤ 22ms — strict gate per senior review).

## Critical files (verified against `ad576545` submodule HEAD)

- `ik_llama.cpp/src/llama.cpp:745-746` — `llama_qwen3next_state_slots` formula
- `ik_llama.cpp/src/llama.cpp:969` — `s_l` allocation (qnext recurrent path)
- `ik_llama.cpp/src/llama.cpp:2033-2045` — `llama_kv_cache_seq_cp` qnext branch
- `ik_llama.cpp/common/speculative.cpp:1601` — DRAFT_GEN `llama_batch_init(1, 0, 1)`
- `ik_llama.cpp/examples/server/server-context.cpp:29` — `server_tree_k`
- `ik_llama.cpp/examples/server/server-context.cpp:44` — `TREE_BRANCH_BASE = 1024`

## Steps

1. **Plumb `n_seq_max + (K-1)*n_parallel` extra slots** at server startup.
   Server reads `LLAMA_MTP_TREE_K`, bumps `cparams.n_seq_max` before
   context creation. State buffer allocates with extended slot count.

2. **Extend `llama_kv_cache_seq_cp`** to also issue `ggml_cpy` from
   `s_l[layer][src_slot]` to `s_l[layer][dst_slot]` per layer for hybrid
   models.

3. **Extend `llama_kv_cache_seq_rm`** to also clear `s_l[layer][slot]`
   rows for cleared seq_ids.

4. **Reserve branch slots in `server_tree_branch_seq`**: switch from
   `TREE_BRANCH_BASE = 1024` to compact in-range allocation:
   `branch_seq = n_parallel + slot.id*(K-1) + (branch_idx-1)`.

5. **Validate**: rerun `scripts/probe-tree-k2.sh` at K=2 short context,
   then 256K X02. Expect:
   - No `qnext_state_slots` assert
   - α(accept) at K=2 ≥ probe-measured 0.93 (probe ceiling 0.95 minus 2pp)
   - tg(K=2) within ±3% of tg(K=1) — depth=1 K=2 alone is marginal

## Closure criterion

Phase 41 `[x]` requires *all of*:

- K=2 d=1 runs clean on Qwen 3.6 27B at 256K X02 (no asserts, no crash)
- α(accept) at K=2 ≥ 0.93
- Diff-byte greedy parity at K=1 (tree-mode disabled) at temp=0 against
  pre-Phase-41 build

Throughput is **not** Phase 41 evidence. Phase 41 binds on foundation
correctness, not uplift. Phase 42 binds on uplift.

## Forward direction

Phase 41 closes → Stage 2 probes (α₂, batch=K) → Stage 2.5 gate decision:

- Pass: α₂ ≥ 0.85 AND batch=2 ≤ 22ms → Phase 42 implementation begins
- Fail either: workstream closes at Phase 41 evidence, Phase 42 parked

If α₁ at production ≥ 0.97 (above probe), depth-1 K=2 alone may justify
production swap; closure document re-measures on production prompts.

## Plan source

Full workstream plan at `~/.claude/plans/wild-sleeping-storm.md`. Phase 42
detail (Stage 3.A–3.G + closure) lives there until Phase 41 closes and
Stage 2.5 gate passes.
