# PHASE_CUDA_NATIVE_DISPATCH — C14 LIVE verification

**RUN_ID:** 20260527T190528
**Date:** 2026-05-27
**Host:** xeon (2× Quadro RTX 6000, NV2 NVLink, clocks locked 1455 MHz)
**Submodule HEAD tested:** `4465a7d1` initially, then `(gate-fix)` for B.7 rerun
**Production binary (untouched):** Phase-46 closure `1db6c2eb` at `/opt/llm-server/`

## Summary

The C-arc unit tests (7/7) PASSED on the build tree. The live integration
verification revealed two findings — one architectural (closed by a follow-
up patch this session) and one pre-existing (out of scope for the C-arc).

| Gate | Result | Notes |
|---|---|---|
| G3.a NP={1,2,4,8} ×2 reps + Phase-46 baseline | ⚠️ NP=8 single-slot flake | Pre-existing; reproduces on Phase-46 closure too |
| G3.c single-GPU NP=2 × 20 iters | ✅ Rate=0% | All 20 byte-identical |
| B.7 CLIP latency × 10 + Phase-46 baseline | ✅ PASS (after gate fix) | median 14440 ms vs baseline 14421 (+0.13%) |
| Production restart | ✅ /health=200 | Phase-46 closure binary untouched |

## Finding 1 — outer capture incompatible with `GGML_SCHED_MAX_COPIES=1`

**Crashed B.7 on first run:**

```
processing image...
CUDA error: operation not permitted when stream is capturing
  current device: 0, in function ggml_backend_cuda_synchronize at
  /home/dconnolly/yarn-agentic/ik_llama.cpp/ggml/src/ggml-cuda.cu:4516
  cudaDeviceSynchronize()
```

Root cause: `ggml_backend_sched_copy_inputs` falls back to
`ggml_backend_synchronize` whenever `sched->events[backend_id][cur_copy]`
is NULL. The events are allocated only when `sched->n_copies > 1`. The
build flag `-DGGML_SCHED_MAX_COPIES=1` means n_copies is always 1, so
events are never allocated, so copy_inputs always synchronizes —
illegal under `cudaStreamBeginCapture`.

**Fix landed this session (defensive gate):**

```cpp
const bool try_outer_capture = (n_cuda > 0)
                               && cpu_is_prefix_only
                               && first_cuda_split < sched->n_splits
                               && sched->n_copies > 1   // ← added
                               && !sched->outer_capture_disabled;
```

With `n_copies > 1` gate, outer capture only fires when the build has
events enabled. On the current build (`MAX_COPIES=1`), capture is
effectively dead code — dispatch falls back to the C1 eager path. CLIP
encoder now runs deterministically through C1's single-threaded
dispatch (no captured graph). B.7 rerun PASSed at 14440 ms median.

**Implication for the phase:** C4/C5/C7's captured-graph perf benefit
is NOT realized on this build. To unlock it, the project would need
to bump `GGML_SCHED_MAX_COPIES` to 2 (or higher) at build time. That
change has ABI/VRAM implications (events × n_backends × n_copies extra
storage) and is out of scope for this verification.

## Finding 2 — NP=8 single-slot flake pre-exists the C-arc

Four reps of `test-production-np-determinism.sh` with `NP_LIST="1 2 4 8"`:

| Binary | Rep | NP={1,2,4} | NP=8 | Failing slot | Divergent text (first 80 chars) |
|---|---|---|---|---|---|
| C12 build-tree (`4465a7d1`) | 1 | OK | FAIL | slot 7 | `code generation, though they remain probabilistic systems that can produce inac` |
| C12 build-tree (`4465a7d1`) | 2 | OK | FAIL | slot 4 | `code generation, though they remain probabilistic systems that can produce inac` |
| C12 + gate fix | 3 | OK | FAIL | slot 6 | `code generation, though they remain probabilistic systems that can produce inac` |
| **Phase-46 closure `1db6c2eb`** | baseline | OK | FAIL | slot 7 | `code generation, though they remain probabilistic systems that can produce inac` |

The Phase-46 closure build (production's current binary) shows the same
flake. **The C-arc did not introduce this and did not address it.**
Cross-NP slot-0 comparisons are byte-identical in every run (NP=1 vs
NP=2 vs NP=4 vs NP=8 at slot 0); the divergence is concentrated in one
of the 8 slots, stochastically chosen, with the same divergent content
every time (a known race signature).

Per `project_np8_localized_openmp_cuda_mismatch.md` and the earlier
diagnostic chain: the race lives somewhere in the multi-GPU dispatch
that isn't the openmp parallel path C1 deleted, and isn't single-GPU
(G3.c PASS confirms). The prior auto-memory claim that
`GGML_SCHED_EVAL_SERIALIZE=1` closed the flake appears to have been a
single-rep false-pass; under sustained testing the flake reproduces
even on the SERIALIZE path's structural successor (C1's single-thread
dispatch).

**Live impact**: production runs `--parallel 1` so this latent flake
does not affect the live service.

## Phase-46 baseline preservation

- CLIP encode median: **14440 ms** (Phase-46 closure baseline 14421, +19 ms, +0.13%)
- p95: 14456 ms
- 10/10 encodes completed; assistant response present
- `evict_pressure` events: 0

## Decisions

1. **NOT deploying C12 to production.** The C-arc unit tests pass and
   the CLIP path now also passes after the gate fix, but the captured-
   graph perf benefit doesn't realize on `MAX_COPIES=1`. Production
   stays on Phase-46 closure (`1db6c2eb`) until either (a) the build
   is rebumped to `MAX_COPIES=2` and re-verified, or (b) the C-arc is
   declared safe for deploy on its determinism merits alone.
2. **Gate fix committed** (next commit) so the build doesn't crash if
   any downstream consumer enables outer capture without rebuilding
   the events plumbing.
3. **NP=8 flake opened as a separate work item.** Not in the C-arc's
   scope. Will need a fresh PD pass: enumerate the multi-GPU dispatch
   paths under `--parallel 8`, instrument them, and bisect against
   single-GPU.

## Files in this artifact

- `unit-test-sweep.txt` (already present from initial C14 artifact)
- `pre-health.json` — pre-window /health
- `G3a-rep1.log`, `G3a-rep2.log` — initial NP determinism battery on C12
- `G3c.log` — single-GPU NP=2 × 20 iters (PASS)
- `B7-verify.log` — first B.7 run (crashed — capture+sync incompatibility)
- `B7-verify-rerun.log` — B.7 after gate fix (PASS)
- `G3a-rep3-postgate.log` — NP determinism after gate fix
- `G3a-phase46-baseline.log` — NP determinism against `/opt/llm-server/`
  (Phase-46 closure); shows same flake → pre-existing

## Next steps (user direction)

1. Authorize commit + push of the gate fix.
2. Decide on production deploy path: stay on Phase-46 closure, or
   bump `MAX_COPIES=2` and re-run the verification battery.
3. Open a new phase (or PD pack) for the NP=8 single-slot flake.
