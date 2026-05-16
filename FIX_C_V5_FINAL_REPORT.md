# FIX-C v5 (singlewarp) — final investigation report

**Date**: 2026-05-16
**Branch**: `production/2026-q2-next`
**Kernel**: `ggml/src/ggml-cuda/fattn-per-slot-kv-singlewarp-sm75.cu`
**Dispatch**: env-gated via `LLAMA_PSKV_MODE=singlewarp` (default still `wmma` for now)

## TL;DR

A new bespoke FA kernel for the per-slot-kv production path (Qwen 3.6 27B, Dq=Dv=256, Q4_0 KV) that delivers **full FA batch-invariance** at **1-4% perf cost** vs the current production wmma kernel. All same-prompt slots in a batched decode produce byte-identical FA output and l_out residuals. Multi-run determinism verified across 5 runs × 512 captured residuals (2560 sha256 comparisons, 0 divergences). Closes spec §15.17's "run-to-run non-determinism at concurrent NP>=4" concern at the FA kernel level.

## Architecture

```
Grid:  (n_tokens, n_heads_q, n_seqs)
Block: 32 threads = 1 warp
Each thread holds: 8 Q values (fp32), 8 VKQ accumulators (fp32)
SMEM:  none (0 bytes)
Regs:  95 per thread, 0 spills, 0 stack frame
```

Algorithm per CTA:
1. Load Q row into registers (8 fp32 per thread × 32 threads = 256 = Dq).
2. K-loop iterates `[0, ne11)` in canonical order. For each k:
   - Each thread computes 8-element dot-product partial. `warp_reduce_sum` aggregates to one scalar KQ score.
   - Add mask (slope * maskh[k]) for ALiBi/visibility.
   - Online Welford softmax update of kqmax, kqsum.
   - Multiply VKQ accumulator by scale_correction.
   - Accumulate V[k] * phi into VKQ.
3. Normalize VKQ /= kqsum and write output.

Critically: **no cross-warp reductions**. Single warp = no partition-dependent partial sums.

## Why it's batch-invariant

For same-prompt slots:
- K cache contents at slot's valid positions are byte-identical (verified TRACE-3).
- Q row data is byte-identical (verified TRACE-2).
- For each k in [0, ne11):
  - If mask[k] = 0 (visible): KQ score, softmax update, VKQ accumulation all depend only on K[k] and V[k] values, which are byte-identical between slots.
  - If mask[k] = -∞ (masked): KQ + mask = -∞ → phi = 0; `kqsum *= 1`, `VKQ *= 1`, `VKQ += V * 0`. All fp32 no-ops per IEEE 754 (x+0=x and x*1=x bit-exactly).
- Each slot's CTA produces byte-identical state after the K-loop.

## Validation matrix

| Test | Setup | Result | Source |
|---|---|---|---|
| V2 TRACE-6 NP=2 | same prompt, intra-layer-3 chain | FA-out + l_out-3 **0 diffs**, max\|Δ\|=0.0 | `data/fixc-v5-bench-2026-05-16/` |
| V3 TRACE-1 NP=1 | single slot | byte-identical run-to-run | `data/fixc-v5-trace1-2026-05-16/np1` |
| V3 TRACE-1 NP=2 | all slots same prompt | **2/2 slots byte-identical at every layer** | `data/fixc-v5-trace1-2026-05-16/np2` |
| V3 TRACE-1 NP=4 | all slots same prompt | **4/4 slots byte-identical at every layer** | `data/fixc-v5-trace1-2026-05-16/np4` |
| V3 TRACE-1 NP=8 | all slots same prompt | **8/8 slots byte-identical at every layer** | `data/fixc-v5-trace1-2026-05-16/np8` |
| V4 production harness NP=4 | same prompt, 4 concurrent, 64 decode steps | **4/4 slots byte-identical** to each other (intra-NP) | `data/fixc-v5-prod-harness-2026-05-16/` |
| V5 multi-run stability | NP=8 × 5 back-to-back runs | **0/2560 file divergences (100% sha256-identical)** | `data/fixc-v5-multirun-2026-05-16/` |
| V6 perf | wmma vs singlewarp at NP={1,2,4,8} | 1-4% slowdown | `data/fixc-v5-prod-harness-2026-05-16/V6_perf_findings.md` |
| DATA-4 long context | n_kv up to 2940 at NP=2 | 0/64 divergences | `data/fixc-v5-long-ctx-2026-05-16/` |
| DATA-5 ptxas | resource profile | 95 regs/thread, 0 spills, 16 CTAs/SM max | `data/fixc-v5-prod-harness-2026-05-16/DATA5_ptxas_profile.md` |
| DATA-7 unit-test gate | existing FA correctness suite | **2/2 tests PASS** (ncols-invariance + dispatch-np-invariance) | regression infra |

## Comparison to baseline (wmma)

| Metric | wmma (current default) | singlewarp (FIX-C v5) |
|---|---|---|
| Intra-NP FA byte-identity (NP=4) | FAIL (slot-parity bug per TRACE-1) | **GREEN (V3)** |
| Intra-NP token byte-identity (NP=4 × 64 steps) | partial | **GREEN (V4)** |
| Run-to-run determinism (NP=8 × 5 runs) | "shifts pattern across runs" per §15.17 | **GREEN (V5)** |
| Decode rate vs baseline | reference | -1 to -4% |
| Aggregate throughput @ NP=8 | 28.88 t/s | 28.54 t/s (-1.2%) |

## Cross-NP gap (NOT in scope for FIX-C v5)

V4 shows 2/14 cross-NP slots match NP=1 baseline. The other 12 slots run a different trajectory because of the **F32-vs-F16 residual storage path split** at layer 0 (per `data/deltanet/d2-first-divergent-layer.json` 2026-05-14). This is in `src/llama-build-context.cpp` at conditional branches on `cur->ne[1] == 1` (vs `<= 32` vs `> 32`) at lines 789, 1375, 1387, 1407, 1491, 1505, 2779, 2902, etc.

Fixing the cross-NP gap requires a separate workstream: making the build graph batch-shape invariant in all paths, not just FA. Likely the right approach is to force the dtype path uniformly to F32 (or F16) regardless of `n_tokens`. Out of scope for FIX-C v5.

## Recommended production rollout

1. **Phase 1 (now)**: keep dispatcher env-gated default `wmma`. Opt-in via `LLAMA_PSKV_MODE=singlewarp` in profile scripts for production workloads that need NP determinism.

2. **Phase 2 (after field validation)**: change dispatcher default to `singlewarp`. The 1-4% perf cost is well within tolerances for production NP determinism.

3. **Phase 3 (cross-NP work)**: address the F32-vs-F16 storage path split in the build graph. This unlocks 14/14 NP-cross byte-identity.

## Files added/modified

```
ggml/src/ggml-cuda/fattn-per-slot-kv-singlewarp-sm75.cu  NEW (~270 LOC)
ggml/src/ggml-cuda/fattn-per-slot-kv-sm75.cu             MODIFIED (4-way env dispatcher)
ggml/src/ggml-cuda/fattn-vec-f16.cuh                     MODIFIED (Q4_0 extern decl at hs=256)
ggml/src/ggml-cuda/fattn-vec-f32.cuh                     MODIFIED (Q4_0 extern decl at hs=256)
ggml/src/ggml-cuda/template-instances/fattn-vec-f16-instance-hs256-q4_0-q4_0.cu  NEW
ggml/src/ggml-cuda/template-instances/fattn-vec-f32-instance-hs256-q4_0-q4_0.cu  NEW
scripts/test-production-np-determinism.sh                MODIFIED (LLAMA_PSKV_MODE env passthrough)
```

## Open follow-ups

- **F32-vs-F16 storage path** in build graph (the cross-NP gap) — separate workstream.
- **NP=2 slot 1 / NP=8 slot 1** late-decode divergence: in V4 these slots' tokens differ from slot 0 at decode step ~60. Within-step FA is byte-identical (V3); the divergence comes from autoregressive accumulation over many decode steps. Likely a non-FA op with subtle slot-position-dependence over time. Worth a separate trace.
- **Singlewarp at non-Qwen-3.6 shapes**: kernel is hardcoded to Dq=Dv=256. Other models with different head_dim would need new template instances or a generic dispatcher.

## Source links

- [Defeating Nondeterminism in LLM Inference — Thinking Machines Lab](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- [SGLang Deterministic Inference](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/)
- [llama.cpp PR #16016 draft](https://github.com/ggml-org/llama.cpp/pull/16016)
- [ssiu/flash-attention-turing](https://github.com/ssiu/flash-attention-turing)
- [Turing Tuning Guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html)
