# Phase 45 — Decompose `llama_context`

Permanent fork. Not upstreamable. We can have fun here.

## Goal

Replace `llama_context`-as-bag-of-state with three composable types:

- `llama_session` — model-aligned sequence state (transformer K/V, cells, positions, defrag).
- `llama_decoder` — parameterized executor (role, graph builder, scheduler, recurrent state, output buffers, batch tracking, perf).
- `llama_spec_loop` — orchestrator for verify + N draft decoders, owns the accept/reject algorithm + KV transactions.

Multi-slot MTP (np=3 × 256K) lands as the first user. PHASE38 D/E/F are superseded — the parent_ctx alias is no longer needed; sessions are shared by construction.

## Why

`llama_context` conflates ~10 concerns. Spec decoding wants two execution roles sharing transformer K/V but holding their own recurrent state and scheduler. The current code fakes this with parent_ctx aliasing. The right abstraction makes the sharing native.

Verify and draft are the same type (`llama_decoder`) parameterized by role — there is no asymmetry to preserve.

Fork-mode unlocks: no compatibility shims, no preserved API names, aggressive removal of dead paths, honest naming throughout.

## Architecture

```
llama_model              read-only: weights, hparams, vocab
  │
  ▼
llama_session            per-tenant: transformer K/V, cells, positions
  │
  ├── llama_decoder      role-parameterized: graph, sched, recurrent, output
  │     role | builder | sched | s_l[slots][layers] | logits | perf
  │
  └── llama_kv_txn       RAII reservation for speculative writes;
                         commit() folds in, rollback() drops.

llama_spec_loop          orchestrator: 1 verify + N draft decoders;
                         owns accept/reject; spawns kv_txns; coordinates batches.
```

`llama_context` deleted. Server, common, profiles migrate to the new API.

## Roadmap

| Step | What | Verifier |
|---|---|---|
| D1 | Audit current `llama_context` — every field → destination (session / decoder / deleted) | report file `PHASE45_FIELD_AUDIT.md` |
| D2 | Audit `llama_kv_cache_init` paths — keep used (hybrid + split_cache + qwen-mtp), drop dead | report `PHASE45_KV_PATHS.md` |
| D3 | Audit ik_llama-specific accretions on `llama_context` (MTP, INLINE_KV, Hadamard, fused state) | report `PHASE45_ACCRETIONS.md` |
| D4 | Inventory external callsites of `llama_context` (server, common, examples we ship) | report `PHASE45_CALLSITES.md` |
| D5 | Header sketches: `llama-session.h`, `llama-decoder.h`, `llama-kv-txn.h`, `llama-spec-loop.h` (~500 LoC stubs) | builds, no behavior |
| D6 | End-to-end CPU forward through new types (no spec, no multi-slot) | greedy parity vs old API |
| D7 | CUDA single-slot through new types | `--fast` GREEN |
| D8 | Spec decoding via `llama_spec_loop` (single-slot MTP, draft 3) | multi-turn parity ≥ +19% C config |
| D9 | Multi-slot MTP (np=3 × 256K) | 48 GB fit; per-slot tg ≥ 29.6 t/s |
| D10 | Delete `llama_context`. Update server, common, scripts, profiles | `git grep llama_context src/` → 0 |
| D11 | Honest renames: kv_self → session.transformer_kv, etc. | reads cleanly |

## Estimated cost

~150-220k tokens across multiple sessions. D1-D4 parallelizable via subagents (~30k). D5-D11 sequential.

## Success criteria

1. np=3 × 256K MTP fits in 48 GB VRAM and runs without OOM.
2. Multi-turn agentic bench tg ≥ baseline (≥ 29.6 t/s per slot).
3. `git grep llama_context src/` returns 0 hits in the new world.
4. Test parity at every Dx checkpoint (no regressions land mid-branch).

## Branch

`phase45-decompose`. Single coherent body of work; doesn't merge until coherent end-to-end. Multi-slot is the binding closure criterion.

## Supersedes

- **PHASE38 D** (parent_ctx alias) — abandoned; sessions share KV by construction.
- **PHASE38 E** (seed-source via persist host) — folded into decoder construction.
- **PHASE38 F** (Hadamard absorption into weights) — orthogonal; lands separately after PHASE45.
- **PHASE38_D_PLAN.md** — historical, stale. Refer to this doc.

## Anti-goals

- ❌ Compatibility wrapper for `llama_context` — fork is permanent, just delete it.
- ❌ Phased extraction with stable releases between — branch lives until coherent end-to-end.
- ❌ Preserving upstream-friendly names — rename for clarity.
- ❌ Multi-slot via parent_ctx as a fallback — commit to D9 fully or revert the branch.
