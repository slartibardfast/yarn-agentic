# Phase 1 — R2 inflection sweep findings

**Run:** 2026-05-28 08:31Z
**Config:** ctx=262144, NP=1, ubatch=256, RT chain (mlockall + SCHED_FIFO 50 + cpu-mask 0xF0 + threads=4)
**Build:** `/home/dconnolly/yarn-agentic/ik_llama.cpp/build/bin/llama-server` (b2cf8fbf C-arc + RT)
**N reps per depth:** 3 — rep-to-rep TG noise ~0.1%

## Result table

| target depth | actual n_pp | mean TG t/s | min | max |
|---:|---:|---:|---:|---:|
| 3000  | 2221 | 16.53 | 16.41 | 16.62 |
| 4000  | 2901 | 15.21 | 15.19 | 15.23 |
| 5000  | 3581 | 14.05 | 14.03 | 14.06 |
| 6000  | 4431 | 13.01 | 13.01 | 13.02 |
| 8000  | 5791 | 11.63 | 11.63 | 11.63 |
| 10000 | 7321 | 10.29 | 10.27 | 10.30 |
| 12000 | 8851 |  9.25 |  9.24 |  9.25 |

## Decay shape

Per-additional-token slope (Δtg / Δn_pp) between successive points:

| segment | Δtg/Δn_pp |
|---|---:|
| 2221→2901 | -0.00194 |
| 2901→3581 | -0.00171 |
| 3581→4431 | -0.00122 |
| 4431→5791 | -0.00101 |
| 5791→7321 | -0.00088 |
| 7321→8851 | -0.00068 |

**Slope per-additional-token decreases monotonically with depth.** This is
the opposite of a cliff — it's a smooth concave-down curve, matching the
shape of attention cost growing with KV size. Phase E's earlier
"peak at 2901 then drop to 7.5 at 12081" is fully consistent with this
curve under sparse 3-point sampling.

## Closure decision

Per the Phase 1 decision tree in PHASE_PERF_R3_FOLLOWUP.md:

> Smooth decay across the range → R2 is misframed; not actually an
> inflection, just observation variance at Phase E sparse points.
> Close R2.

**R2 is closed.** No Phase 2 nsys diff needed.

## Calibration check

The 2026-05-28 published-curve calibration predicted Q4_0 cliffs are at
32K+ depth, not 3K-12K, and that smooth decay below 24K matches the
published shape. **The calibration was correct.** The Hadamard-path
kernel-threshold hypothesis is unnecessary; the standard attention-cost
shape explains the observation.

## Remaining R3-followup work

- Phase 2: **not run** (R2 closed without nsys diff).
- Phase 3: R1 ctx-allocation tax sweep + T5.9 effectiveness sub-test —
  this is now the load-bearing investigation.
- Phase 4: R1 nsys diff (depends on Phase 3 outcome).
- Phase 5: R3 NP=2 ctx=524k reproducer (gated on Phase 3/4).

## Artifacts

- `results.csv` — full 21-row TG/PP data
- `server.log` — llama-server stdout/stderr for the sweep
- `req-d*.json` — request bodies per depth
- `resp-d*-r*.json` — raw response bodies per (depth, rep)
