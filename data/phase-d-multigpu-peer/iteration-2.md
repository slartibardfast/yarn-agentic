# CY.F.19 Investigation — iteration 2 (REVERSAL)

## What changed since iteration 1

Iteration 1 claimed:
- NP=1 was byte-identical across 3 runs.
- CTX_CHECKPOINTS=0 makes the race vanish (3/3 PASS).
- Therefore checkpoints are the source.

**Both claims are now suspect**.

## New evidence

### 1. NP=1 baseline isn't stable across sweeps

Sampling SHA256 of `np1.txt` from the 10 most recent server-harness runs:

```
run-20260517T013439  <distinct SHA>
run-20260517T012800  <distinct SHA>
run-20260517T012353  <distinct SHA>
run-20260517T011947  <distinct SHA>
run-20260517T011740  457a44e863ad2c667b2037ad0ea66b8df505615010fec8fc7e97f3241d7acb84
```

5 unique SHAs across 10 runs. The NP=1 baseline drifts between server restarts.

(Earlier in the session, three NP_LIST="1"-only invocations did produce identical SHA `3683a5f83ea2f90b...`. That window's reproducibility may have been coincidence, or may depend on a specific GPU state at the time. Either way, the baseline is NOT reliably reproducible across server restarts.)

### 2. CTX_CHECKPOINTS=0 doesn't fix the race

5-run sweep with `CTX_CHECKPOINTS=0` + long prompt: **0/5 PASS**.

Server log grep confirmed NO checkpoint creation when CTX_CHECKPOINTS=0 — the
disable mechanism is working.

The earlier 3/3 PASS with `CTX_CHECKPOINTS=0` was likely coincidence
(perhaps a lucky baseline match window, or just the small-sample variance the
short-prompt case also showed at 3/5).

## What this means

The race is NOT primarily about checkpoints. It's about something more
fundamental at the server-level multi-slot path. Possible sources:

- GPU state persisting (or NOT properly resetting) across server processes.
- Genuine per-process initialization non-determinism (some path that depends
  on dynamic memory addresses, kernel launch ordering at warmup).
- Continuous batching specifically — slot dispatch ordering can vary.

## Iteration 3 plan (probe priority)

The right next probe is to **test baseline reproducibility directly**:

1. **P-1**: Start one server, fire 3 successive NP=1 completions, kill server.
   Are the 3 outputs byte-identical? (Tests intra-process determinism.)

2. **P-2**: Start server, fire NP=1 completion, kill, repeat ×3. Compare
   outputs. (Tests cross-process determinism.)

3. **P-3**: Same as P-2 but with `CUDA_VISIBLE_DEVICES=0` (single GPU) — does
   the cross-process race only manifest under multi-GPU?

4. **P-4**: If P-1 passes and P-2 fails: the race is at server init / GPU
   state across processes. If both fail: the race is deeper (cont-batching
   internal state).

Decision rule: if P-1 fails, the race is intra-process and we go back to
audit kernel/state mutations in the multi-slot path. If P-1 passes and P-2
fails, the race is at server-process boundary (GPU state, CUDA driver,
something fresh per process).

## Honest correction

Per CLAUDE.md §4 and `feedback_verify_test_mechanism_before_trusting`:
iteration 1's claim "CTX_CHECKPOINTS=0 fixes the race" was an over-reach
from a 3-run sample. Should have run 5-10 before reporting. The PHASE doc
and any commits referencing this claim need correction.
