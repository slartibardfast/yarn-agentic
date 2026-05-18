# DFlash multi-slot libllama API extension — pickup brief (2026-05-18)

## State at handover

- **Multi-GPU NPC verified PASS at NP={1,2,4,8}** on
  `production/2026-q2-next` via `scripts/verify-production-determinism.sh`
  (run 2026-05-18T17:33). All slots byte-identical to NP=1 at every NP
  value; all 6 cross-NP slot-0 comparisons byte-identical. The NP=4/8
  vanilla drift named in T9 / PHASE_DFLASH future-work item (1) is
  empirically resolved.
- PSKV singlewarp FA kernel has 4-way K-loop ILP + lb=8 landed
  (commits `c37c161d` submodule, `a5c58aa` parent). TG +2.95% / PP
  +9.17% over HEAD. Touches the target's verify path, not DFlash's
  drafter. Doesn't address DFlash's named kernel bottlenecks.
- DFlash workpath decision 2026-05-18: **multi-slot API first, then
  kernel optimization** (option B over option A; see
  `project_dflash_workpath_2026_05_18` auto-memory).

## Scope per PHASE_DFLASH

> DFlash multi-slot libllama API extension (~115–185k tokens).
> Current `llama_dflash_draft(ctx, anchor_token, anchor_pos, …)` is
> single-slot. The kernels are written with `N_slots × Q` grid (T7)
> but the C entry, `common_speculative_state_dflash`, and the server
> `np==1` gate all need multi-slot variants.

Four components implied:

1. **C entry: multi-slot `llama_dflash_draft` variant.**
   `include/llama.h:1801`. Current signature takes
   `(ctx_tgt, anchor_token_id, anchor_pos, out_candidates, max_candidates)` —
   no slot/seq_id parameter. Needs either a new function (e.g.
   `llama_dflash_draft_slot`) taking `seq_id` and producing
   per-slot block, or batched form
   (e.g. `llama_dflash_draft_batch`) producing N slots' worth of
   blocks per call.

2. **`common_speculative_state_dflash` adapter for multi-slot.**
   `common/speculative.cpp:963-1057`. Current state is per-context
   singleton; one drafter, one cycle per `draft()` call. For
   multi-slot the adapter needs to either:
   - Own one drafter shared across slots and dispatch per-slot, OR
   - Be instantiated per-slot (matches how MTP / draft model adapters
     work — see `common_speculative_state_mtp` at line 1261 with its
     `seq_id` parameter), OR
   - Hold a batched API call that produces N slots per cycle.

3. **Server `np==1` gate lift.** The `LLAMA_DFLASH_NP_GT_1 = -7`
   status code exists in `include/llama.h:1758` but I couldn't find
   its reference site in `src/`, `common/`, or `examples/server/`.
   Server-side handling of DFlash multi-slot may already be partially
   wired or absent entirely; the gate is enforced inside
   `llama_set_dflash` (returning the enum) or inside the C entry.
   **First investigation:** grep for the actual rejection site.

4. **Harness for multi-slot DFlash.** No existing test fixture
   exercises DFlash at np>1; the closest is `tests/dflash-speculative/
   test-dflash-np-invariance.cpp` (T7, kernel-level only). Multi-slot
   needs an end-to-end harness analogous to `test-cy-np2-multi-step-decode.cpp`
   but driving the DFlash adapter through `common_speculative` at np=8.

## Internal state shape — known gotchas

The kernels' `N_slots × Q` grid (T7) refers to **drafter source-layer
slots** indexed by `llama_set_dflash_extract_layers`, NOT to NP slots.
See comment at `src/llama-dflash.cpp:523`:

> NOT by target layer id. The cb_eval hook stores at the slot
> ordinal of llama_set_dflash_extract_layers' input array.

So per-NP-slot DFlash state needs to be carefully partitioned:

- `llama_context` already has per-`seq_id` KV cache (multi-slot).
- The drafter's per-context state (cb_eval extract buffers, scratch
  tensors, anchor token state) is single-instance per context. For
  multi-slot, EACH NP slot needs its own scratch + anchor state.
  Either:
  - Replicate the drafter state N times in `llama_set_dflash`
    (linear in N_slots), OR
  - Batch the drafter's forward over N_slots in one launch (kernels
    already support `N_slots × Q` grid per T7).

Batched approach is the perf-conscious one; per-slot replication is
simpler scaffolding but loses the kernel's N_slots parallelism.

## Key unknowns to investigate first

Before writing code, the next session should resolve:

1. **Where is `LLAMA_DFLASH_NP_GT_1` returned?** Not grep-visible.
   May be enforced via `cparams.n_seq_max > 1` check inside
   `llama_set_dflash` or the C entry. Locating this tells us the
   exact gate to lift.

2. **What does the drafter forward expect at `N_slots > 1`?** Read
   `dflash_drafter_forward_launch` in `ggml/src/ggml-cuda/dflash/
   dflash-drafter-forward.cu`. Verify the grid dimension and what
   per-slot state it expects.

3. **What's the slot-major vs anchor-major layout?** The drafter
   processes `BLOCK_SIZE` anchor positions per cycle. At np>1, we
   either:
   - Process all N slots × BLOCK_SIZE positions in one kernel call
     (batched), OR
   - Make N separate cycle calls (serial slot loop).
   T7's "N_slots × Q grid" suggests the batched form is implemented;
   need to confirm.

4. **MTP-style per-slot adapter or DFlash-style shared adapter?**
   MTP uses one `common_speculative_state` per slot
   (`common/speculative.cpp:1261`). DFlash currently uses one per
   context (because the drafter is a shared resource). For multi-slot
   DFlash we either match MTP's per-slot adapter pattern or extend
   the single shared adapter to handle a slot vector. The choice
   affects how the server-side speculative orchestrator integrates.

## Suggested first probe (sub-10k tokens)

Read these in order to confirm scope before scoping the rewrite:

1. `include/llama.h:1740-1825` — full DFlash public API.
2. `src/llama-dflash.cpp` — find `LLAMA_DFLASH_NP_GT_1` enforcement
   site, the cb_eval extract buffer layout, and `llama_dflash_draft`
   internal call shape.
3. `common/speculative.cpp:963-1057` — current adapter.
4. `common/speculative.cpp:1261-1296` — MTP per-slot adapter (template
   for what DFlash per-slot adapter should look like).
5. `ggml/src/ggml-cuda/dflash/dflash-drafter-forward.cu` launcher —
   confirm N_slots grid support.

That probe should answer:
- Is the drafter's GPU state per-context or per-slot?
- Is the kernel grid already N_slots × Q or stub?
- Where exactly is the np>1 rejection?

With those answers, the actual API extension plan can be scoped
properly. Without them, scope estimates have ±50% error.

## Verification gates for the multi-slot DFlash work

The work isn't done until:

1. `scripts/verify-production-determinism.sh` still PASSES at
   NP={1,2,4,8} with DFlash bound (i.e. multi-slot DFlash preserves
   the NPC contract).
2. A new harness (e.g. `tests/dflash-speculative/test-dflash-np-multislot.cpp`)
   exercises DFlash at np=8 end-to-end and matches the np=1 output
   byte-for-byte per slot.
3. A bench at npp=200 ntg=64 npl=8 with `--spec dflash --draft 4`
   produces a non-degenerate t/s number (compared against the
   `bench-spec-none.json` / `bench-spec-mtp.json` baselines in
   `data/phase_dflash_t8/`).

## Out-of-scope reminders

- DFlash kernel optimization (lm_head + GEMM rewrites). That's the
  PHASE_DFLASH future-work item (3); explicitly deferred until
  multi-slot API lands.
- Server profile flip to enable DFlash. Production stays on the
  current deterministic profile until the multi-slot DFlash path
  validates.
- Drift fix (PHASE_DFLASH item 1). Empirically resolved; no further
  work needed unless re-regression observed.

## Pointers to current-session context

- Auto-memory: `[[pskv-ilp-recovery-landed]]`, `[[dflash-workpath-2026-05-18]]`,
  `[[launch-bounds-non-monotonic]]`, `[[latency-bound-vs-bandwidth-bound]]`.
- Session writeup: `data/pskv-ilp-recovery-2026-05-18.md`.
- Ledger: `data/perf-ralph-pskv-ledger.md` (iters 14-19 cover the
  ILP intervention chain).
- PHASE_DFLASH (snapshot from 2026-05-14, somewhat stale per the
  drift-resolved finding above): `docs/phases/70-dflash/PHASE_DFLASH.md`.
