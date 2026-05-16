# V6 — decode-rate perf bench (singlewarp vs wmma)

**Date**: 2026-05-16
**Harness**: `scripts/test-production-np-determinism.sh` × 2 runs (LLAMA_PSKV_MODE=singlewarp + =wmma)
**Workload**: prompt 12 tokens, 64 decode tokens, NP={1,2,4,8}, q4_0 KV cache + Hadamard, multi-GPU split

## Per-slot decode rate (t/s)

| NP | singlewarp | wmma | Slowdown |
|---|---|---|---|
| 1 | 19.97 | 20.89 | 4.4% |
| 2 mean | 11.49 | 11.71 | 1.9% |
| 4 mean | 6.70 | 6.82 | 1.8% |
| 8 mean | 3.57 | 3.61 | 1.1% |

## Aggregate decode rate (sum of per-slot t/s)

| NP | singlewarp | wmma | Δ |
|---|---|---|---|
| 1 | 19.97 | 20.89 | -0.92 t/s |
| 2 | 22.97 | 23.41 | -0.44 t/s |
| 4 | 26.79 | 27.26 | -0.47 t/s |
| 8 | 28.54 | 28.88 | -0.34 t/s |

## Interpretation

The spec §15.13 cited "~12× slower per FA call" for vec_f32. Singlewarp uses
the same per-row CTA architecture (32 threads, single warp) with INLINE
Q4_0 dequant. The measured end-to-end decode slowdown is **1-4%**, not 12×.

Reasons singlewarp comes out so well:
1. FA is a small share of decode time (~3-5% per d2 finding). 12× slower FA
   would yield <50% total slowdown; here it's 1-4%.
2. Singlewarp's per-CTA work is roughly the same as wmma's per-CTA work at
   ne[1]=1 (wmma cpb=8 wastes 7/8 cols; singlewarp has no waste).
3. At NP=N>1 wmma launches 1 CTA per head; singlewarp launches N CTAs.
   Per-CTA work is N× smaller. Total tensor-core work similar; memory
   bandwidth is the same (K/V cache reads dominate).

## Comparison to wmma's determinism gap

wmma achieves 1-4% better t/s but FAILS the determinism contract (TRACE-1..6
chain proved slot-parity bug at every full-attn layer).

Singlewarp delivers FULL batch-invariance at the FA kernel level (V2/V3 PASS)
for a 1-4% perf cost.

## Shippability

Singlewarp is shippable as default for the per-slot-kv path. The 1-4% cost
is well within acceptable perf budget for the determinism guarantee.

Open: V4 production harness still shows cross-NP slot-vs-NP1 divergence
(2/14). That's the F32-vs-F16 storage path split, not FA. Independent fix.
