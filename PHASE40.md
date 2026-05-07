# Phase 40 — Top-K Tree MTP Drafting on Phase 38 Base (consuming existing top-2 infra)

## Hypothesis

Phase 38's MTP architecture (separate `ctx_mtp` + fused dispatch +
`MTP_OP_DRAFT_GEN`) already includes a **top-2 device-side argmax kernel
variant** plus an α(top-2) measurement probe (`edc1f6a3 — mtp: durable
env-gated profiling probes + top-2 kernel variant`). The infrastructure
to read top-K=2 candidates without recomputing logits ALREADY EXISTS.
Phase 40 picks up this surviving Phase 38 work, measures α(top-2), and
— if positive — wires the existing top-2 device cache into a tree-shaped
verify batch on the target context's verify path.

This is **our own work**, on the Phase 38 architecture. Phase 39 (the
upstream port) is parked on its branch as a record but does not gate this
work.

## Why Phase 38 base, not Phase 39

| Aspect | Phase 38 base (this work) | Phase 39 (upstream port, parked) |
|---|---|---|
| MTP head dispatch | Separate `ctx_mtp` fused decode (`MTP_OP_DRAFT_GEN`) | Inline in main forward (`build_mtp_head_qwen35`) |
| Top-2 device kernel | YES (`mtp-argmax.cu` top-2 variant) | DELETED at Phase 39.D |
| `llama_arm_draft_top2` API | YES (`llama.h:1118-1122`) | API survived, but no consumer wired |
| `llama_get_draft_top2` accessor | YES | API survived, but device cache not populated |
| α(top-2) probe (`LLAMA_PROBE_TOP2`) | LIVE — armed during DRAFT_GEN, pushed at draft time, consumed at verify | DEAD — `mtp_speculative_gen_draft` early-returns before probe code |
| Fast-argmax cache (`8a95fcb4`) | YES — saves D2H sync per draft | Replaced by post-compute extraction (different path) |

**The deciding factor:** Phase 38's `mtp_speculative_gen_draft` runs an
explicit `llama_decode(ctx_mtp, mtp_batch)` per draft; the top-2 kernel
emits both top-1 and top-2 IDs to a device cache that's read host-side via
`llama_get_draft_top2`. Tree-K=2 just consumes the existing cache — no
new graph or kernel work needed. Phase 39's inline MTP emits stacked
logits via a different path; the top-2 kernel doesn't participate.

## Schedule (token-budget framing per CLAUDE.md §8)

Phase 40 budget: ~50-70k tokens, gated on the 40.0 probe result.

### 40.0 — α(top-2) probe measurement (~5k)
**Run before any tree implementation work.** Build Phase 38 base, run
`LLAMA_PROBE_TOP2=1` against the slow harness with the existing -mtp
path (LLAMA_MTP_FUSED, draft=3). The probe pushes (top-1, top-2) per
draft, consumes at verify, and prints `α(top1)`, `α(top2)`,
`Δ = α(top2) - α(top1)` per 100 decodes.

**Decision rule:**
- Δ ≥ 0.15 → tree-K=2 will deliver +10% or more, build the rest.
- 0.05 ≤ Δ < 0.15 → marginal; weigh against complexity. Default: build
  if profile-derived cost projection still positive.
- Δ < 0.05 → α(top-2) ceiling is at or below tree's break-even cost on
  this hardware. Close phase as negative with the data; do not build.

**Verify by:** 100+ decodes of slow harness produce coherent α numbers;
α(top1) reproduces Phase 38's measured accept rate (~0.63 baseline).

### 40.1 — Plumb top-2 into the draft return (~10k)

Conditional on 40.0 result. The top-2 device cache is already populated
during DRAFT_GEN. Two changes:

- `mtp_speculative_gen_draft`: when `LLAMA_MTP_TREE_K=2`, after sampling
  top-1 from `common_sampler_sample_speculative`, also read
  `llama_get_draft_top2(ctx, 0, &t2)`. Append both as siblings.
- Mark the slot's drafts as tree-shaped (via `slot.tree_mode` or by
  the env directly).

**Verify by:** at K=2, draft.size() == 2 reliably; the two tokens are
distinct; both are non-EOG.

### 40.2 — Server-side tree verify batch + tree accept (~25-35k)

In `examples/server/server-context.cpp::add_sampled_tokens`:
- If `slot.tree_mode`, K drafts go at the same position (`pos_next()`)
  on K transient seq_ids. The first branch reuses `slot.id`; branches
  1..K-1 get unique transient seq_ids in a high reserved range.
- For branches 1..K-1: `llama_kv_cache_seq_cp(ctx, slot.id,
  branch_seq, 0, pos_next())` shares the prompt prefix cells via
  seq_id-list extension (no data copy).
- Drafts are NOT pushed to `slot.cache_tokens` (they're transient until
  accepted).

In the accept block (`Phase A`):
- New `common_sampler_sample_and_accept_tree(ctx_sampling, ctx, idxs,
  drafts, &winning_branch)` samples target from idxs[0]; finds the
  branch index k where draft[k] == target (≤1 match at depth=1);
  returns {target} or {target, after-target}.

In `Phase B` cleanup:
- If winning branch ≥ 0: `seq_cp(winning_seq, slot.id, pos_next, pos_next+1)`
  promotes the winning cell to slot.id's seq list.
- For all branches: `seq_rm(branch_seq, -1, -1)` releases transient
  cells and seq_ids.

**Verify by:**
- 1000-cycle run: KV cell count steady (no leak), seq_id count steady.
- α(accept) at K=2 matches probe-measured α(top-2) ± 2pp
  (proves the tree path actually realizes the probe's ceiling).
- At `LLAMA_MTP_TREE_K=1` (default): byte-for-byte parity with the
  pre-Phase-40 build for greedy decode (proves the K=1 path is a clean
  no-op vs Phase 38).

### 40.3 — Harness measurement + closure (~10k)

Update `scripts/test-fused-harness.sh` to A/B K=1 vs K=2 in `--fast` and
`--slow`. Closure binds on:
- `--slow tg(K=2) / tg(K=1) ≥ 1.10` (effective +10% from tree fan-out
  on top of whatever K=1 baseline lands at on Phase 38 base).
- Greedy parity at K=1 (verifying we didn't break the linear path).
- α(accept) at K=2 within 2pp of probe-measured ceiling (verifying the
  tree path is correctly wired).

If any of these fail, close as negative result with the specific failure
naming the gap. Production swap deferred until measurement evidence
binds and user authorizes.

## Architectural notes

**Branch seq_id allocation.** Reserve seq_ids `[1024, 1024 + max_slots
* MAX_TREE_K)` for tree branches. Slot S's branch i is `1024 + S *
MAX_TREE_K + i`. Cleanup is unconditional at end of each accept cycle,
so collisions across slots only matter within a single decode — not
across decodes.

**KQ_mask correctness.** Each branch's query at `pos_next()` has
seq_id = branch_seq. After `seq_cp(slot.id, branch_seq, 0, pos_next)`,
prefix cells [0..pos_next) all contain branch_seq in their seq_id list,
so the query attends to them. The cell at `pos_next` for sibling
branches has seq_id list = {sibling_branch_seq} (singleton); the
query's seq_id = own branch_seq doesn't match, so siblings are masked.
ik_llama.cpp's existing KQ_mask construction (per-cell-per-query
seq_id intersection) handles this without modification.

**Top-2 device cache lifetime.** The top-2 IDs live in a per-batch-
position device buffer populated by the kernel and consumed via
`llama_get_draft_top2` between dispatches. The DRAFT_GEN dispatch is
batch=1, so there's exactly one top-2 entry to read per cycle.

## What is explicitly OUT of scope

- Tree depth > 1 (chain rollout > 1 in tree shape). Phase 38's MTP
  fused dispatch can run multiple chain iters, but each one's MTP
  head cost adds linearly; profile data from Phase 39 closure projects
  this is a regression on this hardware. K-wide depth=1 is the only
  sweet spot.
- Top-K with K > 2. The existing CUDA kernel emits exactly top-2.
  Extending to K > 2 would need kernel work; out of scope until K=2
  proves to deliver measured uplift.
- Phase 39 architecture. Parked.
- Production swap. Decision held by user, after closure measurement.

## Backout

Phase 40 builds on `phase40-tree-fanout` branch (off Phase 38 closure
`f7006f2a`). If 40.0 probe is negative or 40.3 measurement fails to
bind, the branch parks alongside `phase39-collapsed-mtp` as a record.
Production stays on no-MTP. The Phase 39 +4.7% uplift is reachable from
the Phase 39 branch if needed (also parked from production).
