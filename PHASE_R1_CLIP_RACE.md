# PHASE_R1_CLIP_RACE — localize the CLIP cross-encode race

**Opened:** 2026-05-28 10:35Z
**Phase A executed:** 2026-05-28 11:08-12:37Z
**Status:** Phase A DONE — race structurally characterized + both
mitigations shipped. The original "delete the workaround" framing was
based on incorrect hypothesis; empirical evidence supersedes.
**Parent:** [PHASE_PERF_R3_FOLLOWUP](PHASE_PERF_R3_FOLLOWUP.md)

## Outcome

CLIP cross-encode non-determinism has **two independently load-bearing
failure modes**, each requiring its own structural fix:

| failure mode | mechanism | fix |
|---|---|---|
| **F1: cross-stream timing variance** | Multi-device kernel completion order varies across encodes; downstream reads see different intermediate states | **Per-node sync fence** — default-on no-op eval callback on CLIP sched, forces `ggml_backend_synchronize` between every node |
| **F2: stale-read from gallocr-reused buffers** | Some kernel partial-writes its output region; next encode reads un-overwritten bytes from whichever tensor previously occupied that slot | **B.5e activation-buffer clear** — gated on `zero_on_reset` flag (default true), LM opts out for perf |

Empirical matrix (10-encode response sha256 across 10 chat requests):

| sync fence | clear | result |
|---|---|---|
| OFF | OFF | 10/10 distinct |
| OFF | ON  | 8/10 vs 2/10 |
| ON  | OFF | 3 distinct |
| **ON** | **ON**  | **10/10 IDENTICAL** ✓ |

Per-node hash bisect (with fence active, `CLIP_CAPTURE_HASH=...`):
**1714/1714 nodes byte-identical** across back-to-back encodes. CLIP's
kernels are deterministic given identical inputs and identical timing;
both fixes together establish that condition.

## What this corrects in the original phase doc

The doc opened with the goal "remove the interim workaround." That was
based on the hypothesis that one fix subsumed the other. Empirical
test 2026-05-28 12:17Z falsified it: with sync fence ON and clear OFF,
**3 distinct sha256s**, not 10/10 identical. The clear covers a
disjoint failure mode the fence doesn't.

So the corrected deliverable is: **both fixes shipped, both required.**

## Performance impact

| metric | result |
|---|---|
| CLIP encode median latency, fence ON + clear ON | 10466 ms |
| CLIP encode median latency, prior (clear only) | 10392 ms |
| Sync fence overhead | +0.7% (within rep noise) |
| LM TG R1 tax at ctx=256k | -7.9% (unchanged — LM opts out of clear) |
| LM TG determinism | 18/18 byte-identical |

The sync fence cost is much lower than predicted (~20-30%) because the
multi-device CLIP graph already has cross-split syncs at split
boundaries; adding per-node sync within a device's split is mostly
absorbed by what the dispatcher already does.

## Submodule commit chain

Three commits land Phase A:

```
4f0a045f clip: per-node sync fence by default + retain B.5e buffer-clear
44f81ad1 ggml-backend: ggml_backend_sched_set_zero_on_reset opt-out (PHASE_PERF_R3_FOLLOWUP — kept)
af41d2b0 llama: O(1) per-seq max-pos cache replaces full-pool scan
```

`af41d2b0` and `44f81ad1` landed in production earlier; `4f0a045f`
adds the sync fence and updates the rationale comments. Both fixes
are documented as covering disjoint failure modes.

## Escape hatches

- `CLIP_DISABLE_SYNC_FENCE=1` — reverts the per-node sync fence on the
  CLIP sched. For measurement / rollback only; restores the
  10/10-distinct-embeddings state.
- `CLIP_DEBUG_SCHED=1` — installs the heavier debug callback with
  per-node IMA logging and optional hash dump via `CLIP_CAPTURE_HASH`.
  Overrides the default sync-fence callback.
- `CLIP_FORCE_EVAL_CB_NOOP=1` — same effect as default; legacy knob
  kept for back-compat.
- `CLIP_FORCE_EVAL_CB_DTOH=1` — heavier per-node DtoH callback for
  diagnostic.

## What remains as legitimate future work

The sync fence + buffer-clear is a structural fix, but each is a
heavy hammer:

- **Sync fence is per-node.** Phase 46's experiments showed
  reduce-only sync wasn't sufficient. A narrower set of "must-sync"
  nodes is theoretically possible but Phase 46 couldn't enumerate it.
  At current cost (+0.7%) the perf gain from narrowing is small.
- **Buffer-clear is whole-allocation.** The actual partial-writing
  kernel was never localized. Finding it would let us delete the clear
  by patching the kernel to fully overwrite its output. Phase 46
  team tried hashes / `CLIP_CAPTURE_SKIP_OPS` and could not pin it.
  Re-attempting on the current binary is feasible but not blocking.

If/when production cost pressures push us, those are the optimization
paths. Today's empirical evidence says: both fixes shipped, costing
under 1% combined CLIP overhead. Done.

## Phase A artifacts

`data/r1-clip-race/phaseA-cliplm-discriminator-20260528T110833/`
- Initial CLIP-vs-LM discriminator (10 chat requests, embedding hash hook)
- Showed 10/10 distinct embedding hashes across encodes (with LM in between)

`/tmp/clip-nodewise-20260528T115801/`
- Per-node hash file from the bisect (3428 lines = 1714 nodes × 2 encodes)
- 0 diffs across 1714 nodes with sync fence active
- (Not committed to repo — large; available locally until cleared)

Bench logs:
- `/tmp/clip-determinism-syncfence.log` — fence ON + clear ON, 10/10 identical
- `/tmp/clip-determinism-clean.log` — fence ON + clear OFF, 3 distinct
- `/tmp/lm-final.log` — LM R1 sweep + 18/18 determinism check

## What does NOT close with this phase

- Localization of the specific partial-writing kernel (deferred —
  not blocking, but the only path to deleting the buffer-clear)
- A narrower sync-fence policy (deferred — current cost is acceptable)

## Acceptance — closed

- [x] CLIP cross-encode race structurally characterized (two failure modes)
- [x] Both fixes shipped and proven independently load-bearing
- [x] 10/10 byte-identical CLIP responses with both fixes active
- [x] 18/18 byte-identical LM TG reps (LM perf preserved via opt-out)
- [x] Submodule pushed, parent pointer to be bumped + deployed in same commit
- [x] Phase doc reflects empirical reality (not initial hypothesis)
