# Phase 3 — R1 ctx-allocation tax + T5.9 effectiveness findings

**Run:** 2026-05-28 08:49Z
**Config:** NP=1, ubatch=256, RT chain, q4_0+Hadamard KV, 200t prompt, N_PREDICT=128
**Build:** `/home/dconnolly/yarn-agentic/ik_llama.cpp/build/bin/llama-server` (b2cf8fbf C-arc + RT)
**N reps per config:** 3 — rep-to-rep TG noise ~0.1%

## R1 primary sweep (production allocator: --cache-ram 40960 --ctx-checkpoints 64)

| ctx | mean TG t/s | min | max | Δ vs ctx=8k |
|---:|---:|---:|---:|---:|
|   8192 | 19.34 | 19.33 | 19.35 | baseline |
|  32768 | 18.68 | 18.66 | 18.70 |  -3.4% |
| 131072 | 16.50 | 16.48 | 16.53 | -14.6% |
| 262144 | 14.34 | 14.33 | 14.35 | -25.9% |

Per-doubling-of-ctx degradation slopes:
- 8k → 32k (4×): -3.4%
- 32k → 128k (4×): -11.6% (i.e. -3.4 → -14.6%)
- 128k → 256k (2×): -13.2% (i.e. -14.6 → -25.9%)

**Slope grows with ctx — increasing marginal cost per allocated block.**
Consistent with FA dequant scratch + attention scratch dominating at
large ctx (TurboQuant Discussion #20969 documents the same shape for
Q4_0 dequant-scratch).

## R1 magnitude correction

The earlier Phase E (PHASE_PERF_R3_NP1) headline cited a -37% TG hit
from ctx=8k to ctx=256k at 200t workload. **The clean measurement on
the production allocator is -25.9%, not -37%.** The 12-point gap is
likely due to differences in the Phase E harness config (different
`--cache-ram`, `--ctx-checkpoints`, or rep timing window) that weren't
fully controlled. The -25.9% figure is the load-bearing number going
forward.

## T5.9 effectiveness sub-test — DESIGN ERROR, NOT A FINDING

| ctx | prod allocator | --cache-ram 0 --ctx-checkpoints 0 | delta |
|---:|---:|---:|---:|
|   8192 | 19.34 t/s | 19.46 t/s | -0.6% (within noise) |
| 262144 | 14.34 t/s | 14.40 t/s | -0.5% (within noise) |

**Correction recorded here:** my sub-test design conflated two
distinct layers.

- `--cache-ram N` / `--ctx-checkpoints N` are **host-side context
  checkpoint caching knobs**. They affect how many saved KV
  checkpoints are kept in host RAM for context-shift recovery.
- **T5.9 paged-KV** is the **block-major GPU tensor layout**
  `[head_dim, BLOCK_SIZE_TOKENS, n_head_kv, total_pool_blocks]` baked
  into the build via the paged allocator code path. It is **always
  active** on this binary; no runtime flag toggles it.

What the sub-test actually measured: **host-side checkpoint caching is
a no-op for a single 181t prompt with no context shifts.** That is
expected — there is nothing to checkpoint. The result does not tell us
whether T5.9 is paying back the GPU-side allocation tax.

A true T5.9 A/B requires a pre-T5.9 build (revert to N1/N2 4D layout).
That is out of scope for this phase.

## What R1 looks like with the correction

- Magnitude: -25.9% (clean measurement)
- Shape: monotonic, slope grows with log(ctx) — consistent with FA
  dequant scratch shape
- T5.9 effectiveness: **unknown** without a pre-T5.9 build
- Remaining lever: kernel-level diff (Phase 4 nsys) to identify which
  kernel(s) account for the 25.9% gap at ctx=256k. If FA-related,
  successor phase scopes a tile/warp tuning patch. If allocator
  scratch dominates (e.g. scratch zeroing or pool walks), that's a
  different lever.

## Calibration check

Published Q4_0 vs F16 shape from TurboQuant Discussion #20969 was
"per-token dequant cost grows with depth at constant ctx allocation."
Our measurement is the inverse — **per-step cost grows with allocated
ctx at constant 200t depth**. The two shapes share the same kernel
(FA dequant scratch) but the load varies along a different axis. The
calibration's "-37% magnitude is in-band" prediction was about the
right order of magnitude; the actual measurement is -25.9% which is
even more comfortably in-band.

## Artifacts

- `results.csv` — full 18-row TG/PP data (6 configs × 3 reps)
- `*-server.log` — per-config server stdout/stderr
- `*-resp*.json` — raw response bodies per (config, rep)
- `req-body.json` — the 200t prompt request body (deterministic, seed=1)
