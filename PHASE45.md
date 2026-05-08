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

| Step | What | Verifier (binding test) |
|---|---|---|
| D1 | Audit current `llama_context` — every field → destination (session / decoder / deleted) | report `PHASE45_FIELD_AUDIT.md` exists, every field classified |
| D2 | Audit `llama_kv_cache_init` paths — keep used (hybrid + split_cache + qwen-mtp), drop dead | report `PHASE45_KV_PATHS.md` exists, dead paths named |
| D3 | Audit ik_llama-specific accretions on `llama_context` (MTP, INLINE_KV, Hadamard, fused state) | report `PHASE45_ACCRETIONS.md` exists, dead PHASE38 E fields named |
| D4 | Inventory external callsites of `llama_context` (server, common, examples we ship) | report `PHASE45_CALLSITES.md` exists, ~90 API methods classified |
| D5 | Header sketches: `llama-session.h`, `llama-decoder.h`, `llama-kv-txn.h`, `llama-spec-loop.h` (~500 LoC stubs) | `gcc -fsyntax-only` clean on all four headers |
| D6 | End-to-end CPU forward through new types (no spec, no multi-slot) | `main.cpp` greedy-decode 50 tokens, byte-identical output vs old API on Qwen 3.6 27B |
| D7 | CUDA single-slot through new types | `scripts/bench-multiturn-pre-port.sh --fast` GREEN at 0.95 floor (Phase 36 recalibrated gate) |
| D8 | Spec decoding via `llama_spec_loop` (single-slot MTP, draft 3) | multi-turn agentic bench tg ≥ +19% vs nomtp baseline (the measured C config: `-mtp --draft 3` + INLINE_KV) |
| D9 | Multi-slot MTP (np=3 × 256K) | profile `qwen36-27b-x3-mtp.sh` boots; 48 GB VRAM fit; per-slot tg ≥ 29.6 t/s; no OOM over 1000-token soak |
| D10 | Delete `llama_context`. Update server, common, scripts, profiles | `git grep -l llama_context src/ common/ examples/server/` returns 0 |
| D11 | Honest renames: kv_self → session.transformer_kv, etc. | code reads cleanly (judgment, no automated test) |

## Architectural decisions (locked before D6)

These were ambiguous coming out of D1-D4; locking them now so D6+ doesn't re-litigate.

### Recurrent-state rollback

Decoder-internal mechanism. `kv_txn` covers transformer K/V; recurrent state (DeltaNet `s_l`) is per-decoder and rolls back via PHASE36's per-step checkpoint, owned by the decoder. **No new public type for recurrent rollback** — the decoder's `decode()` implementation handles it on accept/reject signals from spec_loop.

### Sampling

Fully external to the engine. Existing `llama_sampler_*` API handles chains; PHASE45 does not add sampling state to session, decoder, or spec_loop's owned state. `llama_spec_loop_params.sampler` is a borrowed pointer — caller owns its lifetime. Removes the long-standing ambiguity about whether `llama_context` "owns" sampling.

### Server slot ↔ engine type mapping

**One session shared across slots; one verify decoder + one draft decoder processing all slots together via seq_id-batched forward.** For the np=3 × 256K MTP target:

```
session.n_seq_max = 3       (transformer K/V holds 3 sequences)
verify_decoder    = role VERIFY     — full forward, batches all 3 slots in one ubatch
draft_decoder     = role DRAFT_MTP  — MTP head, batches all 3 slots in one ubatch
spec_loop         orchestrates verify + draft across all 3 slots simultaneously
```

This matches today's batched architecture (one forward with multi-seq_id batches). **The alternative — one session per slot — was rejected: it would re-create the per-slot K/V duplication PHASE45 exists to fix.** **The alternative — per-slot decoders — was rejected: it would prevent batched compute, which is the multi-slot throughput win.**

Each slot is a `seq_id` partition inside the shared decoders' batches.

### PHASE39 collapsed-context MTP wrapping (not superseded)

PHASE39 ported upstream's inline MTP head (`build_mtp_head_qwen35`). PHASE45 wraps it:

- `DRAFT_MTP` decoder builds the same inline path PHASE39 ported.
- `VERIFY` decoder builds the transformer-only forward (layers 0..N-2).
- They share `session.transformer_kv`.
- Layer N-1 (the MTP head's K/V slot) is written exclusively by the draft decoder — single-canonical-writer, no race, no INLINE_KV hook needed.
- PHASE39's lift target (+2.5× upstream evidence) remains the binding number for D8.

## Migration housekeeping

Mechanical work that's not architecturally interesting but mustn't be missed:

- **CMakeLists.txt** — D6 adds `src/llama-session.cpp`, `src/llama-decoder.cpp`, `src/llama-kv-txn.cpp`, `src/llama-spec-loop.cpp` to the library target. Header install rules under `include/`.
- **Profiles** — `profiles/qwen36-27b-x1.sh` keeps working through D6-D9 (single-slot path). New profile `profiles/qwen36-27b-x3-mtp.sh` lands at D9 for the multi-slot bench.
- **Diagnostic env knobs** — `IK_PRINT_TIMING`, `LLAMA_MTP_FUSED_DIAG`, `LLAMA_MTP_INPUT_CHECKSUM`, etc. — read at decoder construction time (per-decoder), not per-call. The migration is mechanical: each `getenv` at lifecycle time becomes a `decoder_params` field set at create.
- **Submodule branch** — D5's commit landed on `ik_llama.cpp`'s `phase36-mtp-throughput` branch. D6+ commits in the submodule continue on a `phase45-decompose` branch (created from current HEAD). Parent repo's `phase45-decompose` branch tracks submodule's `phase45-decompose`.
- **PHASE39.md update** — one-line note: "PHASE45 wraps this work; the inline MTP head becomes the DRAFT_MTP decoder's graph builder. Not superseded."
- **MEMORY.md entry** — committed separately per CLAUDE.md §6: "PHASE45 supersedes PHASE38 D/E/F via decomposition; permanent fork unlocks no-shim approach."

## Estimated cost

~150-220k tokens across multiple sessions. D1-D4 parallelizable via subagents (~30k). D5-D11 sequential.

## Success criteria

1. np=3 × 256K MTP fits in 48 GB VRAM and runs without OOM.
2. Multi-turn agentic bench tg ≥ baseline (≥ 29.6 t/s per slot).
3. `git grep -l llama_context src/ common/ examples/server/` returns 0 hits.
4. Test parity at every Dx checkpoint per the Roadmap table's binding-test column.

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
