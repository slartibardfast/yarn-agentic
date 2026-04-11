# Fusion A/B — Qwen3.5-9B q4km, Vega 64, default F16 KV

Three runtime disable knobs tested against the step 1 baseline. Same
fixed prompt, same seeds 42/43/44, 3 runs each, median values. Baseline
is `per-op/q4km-vega-f16kv-2026-04-11T224817Z.*`.

## Baseline

- Total Vulkan GPU time captured: 21.67 s over 758 graph blocks
- Prompt eval: **291.7 t/s** (median)
- Generation: **35.88 t/s** (median)
- MTP draft acceptance: **69.91%** (309 / 442 over 3 runs)

## Runs vs. baseline

| Run            | Total GPU Δ     | Prompt t/s | Predict t/s (Δ%) | MTP accept |
|----------------|----------------:|-----------:|-----------------:|-----------:|
| fusion on (baseline) | —          | 291.7      | 35.88            | 69.9%      |
| `DISABLE_FUSION=1`    | +598 ms (+2.76%) | 289.3  | 34.88 (-2.80%)   | 69.9%      |
| `DISABLE_MULTI_ADD=1` | +3 ms (+0.01%)   | 291.9  | 35.86 (-0.08%)   | 69.9%      |
| `DISABLE_GRAPH_OPTIMIZE=1` | +796 ms (+3.68%) | 288.7 | 35.30 (-1.64%) | 73.5% |

## Findings

### 1. `GGML_VK_DISABLE_FUSION` is NOT a full kill-switch

It only disables scheduler-level pattern fusion. Graph-builder-time
fusions (from the Phase 3 `GGML_OP_FUSED` work — fused ops emitted in
`ggml.c` during graph construction) are always on and cannot be
disabled via env var.

Evidence: the `MUL_MAT_VEC q4_K + MUL_MAT_VEC q4_K` (double gate),
`MUL_MAT_VEC q5_K + ARGMAX`, and `MUL_MAT_VEC q5_K + SCALE + GET_ROWS
+ GET_ROWS` entries are present in both the baseline and the no-fusion
run with essentially identical timing.

What `DISABLE_FUSION` actually turns off:

- `CPY, RMS_NORM_MUL` split into `CPY, RMS_NORM` + separate `MUL`
  (visible in the "only in baseline" / "only in run" lists — the
  RMS_NORM_MUL block is replaced with un-fused RMS_NORM, and an
  additional standalone `ADD` op doubles from 44 ms to 143 ms to
  pick up the work that used to be folded into the chain).
- The RMS_NORM + MUL pair accounts for the full ~600 ms regression.

**Implication:** the 2.76% regression from `DISABLE_FUSION` is a
lower bound on Phase 3/4 fusion's value — the upper bound would
require a rebuild without the graph-builder-level fused ops.

### 2. `DISABLE_MULTI_ADD` has no measurable effect on this graph

The multi-add fusion isn't firing for Qwen3.5-9B. Its graph doesn't
have the 3+ broadcast-add pattern that multi-add targets. If we ever
switch to a model that does (e.g. a Mixtral variant with lots of
expert routing adds), this knob becomes relevant.

### 3. `DISABLE_GRAPH_OPTIMIZE` is the biggest hit, and it doubles the ARGMAX cost

Per-op deltas from the `no_graph_opt` run:

- **ARGMAX**: 5.9 ms → 1507.6 ms (**+1501 ms** / 255× slowdown).
  This is the graph-optimize pass folding `MUL_MAT_VEC q5_K 248320×4096 + ARGMAX`
  into a single dispatch. When the optimizer is off, ARGMAX runs as
  a standalone kernel over the full 248K-entry logits, which is
  catastrophically slow.
- **MUL_MAT_VEC q6_K 4096×4096**: 286 → 876 ms (+590 ms). Some
  fused chain around this op is also a graph-optimize feature.
- **RMS_NORM_MUL (4096,1,1,1)**: 547 → 233 ms (-314 ms). The op
  survives but with a smaller call count — the rest of the work
  moved into a new fused chain `RMS_NORM_MUL + SCALE + GET_ROWS + GET_ROWS + CPY`
  at 223 ms.
- **CONCAT**: 2.2 → 170 ms (+168 ms). Another graph-optimize folding.

Net: graph-optimize is worth +3.68% of total GPU time, dominated by
the ARGMAX fusion for the vocab head.

### 4. MTP draft acceptance is mildly sensitive to graph-optimize

`DISABLE_GRAPH_OPTIMIZE=1` raised MTP acceptance from 69.9% → **73.5%**
(309 → 324 accepted out of ~442 drafted). The drafted counts differ
by 1 across runs (442 vs 441) so the 15-token accept-rate increase
is coming from numerically-different intermediate values, not a
different drafting path.

This is small enough to write off as noise, but it is evidence that
graph-optimize changes the order-of-operations of some ops enough to
perturb low-order bits of the softmax path. Worth noting in case
someone later debugs a draft-acceptance regression and hits the same
sensitivity.

## Phase 3/4 value, in one sentence

The `DISABLE_FUSION` and `DISABLE_GRAPH_OPTIMIZE` knobs together
account for ~6.4% of total GPU work on a representative Qwen3.5-9B
q4km generation. That's the lower bound — the graph-builder fusions
from Phase 3 (visible in every run as the `+` concatenated op names)
are not measured here but are likely at least as valuable again.

## Artifacts

- `no-fusion-q4km-vega-2026-04-11T225140Z.{stderr,json,drive.txt}`
- `no-multi-add-q4km-vega-2026-04-11T225330Z.{stderr,json,drive.txt}`
- `no-graph-opt-q4km-vega-2026-04-11T225520Z.{stderr,json,drive.txt}`
- `delta-2026-04-11T225700Z.json` — machine-readable delta table
- `build_delta.py` — the script that produced the delta JSON
