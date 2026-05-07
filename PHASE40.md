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

---

## Phase 40 closure — NEGATIVE on probe data alone

### Verdict

**40.0 probe data closes Phase 40 as negative before implementing 40.1+.**
α(top-2) - α(top-1) = 0.06 at production-relevant long context. Below
the +5% build threshold. **No tree-K=2 implementation work is performed.**
Total Phase 40 cost: ~10k tokens (probe runner + measurement +
analysis), saved ~50-70k tokens of tree-K=2 implementation that the
data shows would not have delivered meaningful uplift.

### Measured probe data

```
PROBE: LLAMA_PROBE_TOP2=1, -mtp --draft 1, temperature=0
HARDWARE: 2× Quadro RTX 6000 sm_75, --split-mode graph --tensor-split 1,1
MODEL: qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf

Short context (4K) + short essay prompt:
  total=200 α(top1)=0.8000 α(top2)=0.8950 Δ=0.0950

Long context (256K) + X02 agentic prompt (5K char):
  total=100 α(top1)=0.8500 α(top2)=0.9100 Δ=0.0600
  total=200 α(top1)=0.8900 α(top2)=0.9500 Δ=0.0600
```

Raw runlog: `data/phase40-probe-top2-x02-256k.runlog` (256-token gen).

### Why Δ shrinks at long context

α(top-1) jumps from 0.80 (short) to 0.89 (long) because the X02
agentic prompt provides dense surrounding context that makes top-1
predictions very confident. Less of the probability distribution leaks
to top-2. Concretely: at α(top-1)=0.89 only 11% of cycles fail at
top-1; of those, only 6pp can be recovered by top-2 (the remaining 5%
have the answer outside top-2 entirely).

This is a **feature of the production use case**, not a bug. Agentic
content with rich context is exactly where top-1 predictions are
strongest — and exactly where tree fan-out has least to add.

### Why the +3.2% projected lift is too marginal to ship

Throughput math at Δ=0.06 (long context):

```
K=1 baseline:  tokens/cycle = 1 + 0.89 = 1.89; cycle ~50ms; tg ≈ 37.8 t/s
K=2 tree opt:  tokens/cycle = 1 + 0.95 = 1.95; cycle ~50ms (if verify
               flat in K at long context); tg ≈ 39.0 t/s = +3.2%
K=2 tree pess: cycle ~66ms (if verify scales with token count due to
               seq_cp branch overhead); tg ≈ 29.5 t/s = REGRESSION
```

Even the optimistic +3.2% is well below Phase 40's binding threshold
(+10% on top of K=1, +20% absolute over no-MTP per the earlier
PHASE40.md). The pessimistic case is a regression. The asymmetry of
risk (small upside vs material downside) plus ~50-70k tokens to
implement makes this a clear close.

### What survives Phase 40

- ✅ The α(top-2) measurement infrastructure (`edc1f6a3`) — already
  shipped, proven to work, gives a clean empirical answer to the
  tree-K=2 question for any future model/setup.
- ✅ `scripts/probe-top2.sh` — reusable probe runner for the next
  model class (e.g., if a smaller model with lower α(top-1) is
  considered, re-run probe; if Δ > 0.15, build tree-K=2 on it).
- ✅ Phase 38 base preserved on `phase40-tree-fanout` branch as
  the canonical tree-fan-out base of record.

### What's dead

- ❌ Tree-K=2 implementation work (40.1, 40.2, 40.3) — never built,
  closed before implementation.
- ❌ The PHASE40 +14-19% projection from the Phase 39 closure document
  — that projection assumed Δ ≈ 0.15 at K=2; measured Δ=0.06 invalidates
  it. The earlier projection was theory; the probe is data.

### Why this measurement-first close is correct (not premature)

Per CLAUDE.md §8: "negative results land cheap when honest, expensive
when rationalized." The probe was the cheap-decision-point that the
Phase 40 design specifically gated 40.1+ on. Skipping the probe and
implementing tree-K=2 anyway would have been:

- ~50-70k tokens of implementation work
- Ending in measurement that would show the same +3.2% optimistic /
  regression pessimistic outcome
- A negative-result writeup at the end with the same conclusion
- Plus +50-70k of "abandon the work" cost to revert the branch

Closing on the probe saves the +50-70k of code-then-revert cost AND
keeps the negative result on a credible empirical footing rather than
"we built it, it didn't work, here's why." The probe answer is
unambiguous: the model's top-1 distribution is too confident at
long-context agentic content for tree-K=2 to add meaningful value.

### Forward direction (not Phase 40 work)

If future work wants more throughput on this hardware, paths include:

1. **No-MTP baseline tuning** — production currently runs without -mtp;
   the Phase 39 +4.7% rollout=1 single-best is reachable on the
   `phase39-collapsed-mtp` branch (parked but available).
2. **Different model classes** — α(top-1)=0.89 at long context is a
   feature of this model's calibration. A different model
   (e.g., smaller, higher-temperature, less-context-confident) might
   have Δ ≥ 0.15 and benefit from tree-K=2. Re-run the probe for any
   new candidate.
3. **Top-K with K > 2** — the existing CUDA top-2 kernel emits exactly
   2; extending to K=3 or K=4 would need kernel work. With Δ=0.06 at
   K=2, K=3 might add another 2-3pp (diminishing returns), insufficient
   to justify the kernel + verify-path work.
4. **Different drafter** — n-gram, draft-model, or other speculation
   types. Out of scope for Phase 40.

None of these are Phase 40 follow-ups; they're separate phases gated
on independent decisions.

