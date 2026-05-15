# Replacement perf — new FA op at production shape

Date: 2026-05-14
Branch: production/2026-q2-next
Capture binary: `test-np-validity-vanilla` (post Hadamard + new FA op dispatch)
Config: NP=1, n_gen=32, Q4_0 KV + Hadamard, production target GGUF.

## Production FA kernel observed (Stage 2.3 on-device path)

```
fattn_per_slot_kv_sm75_stage23_kernel(...)   — 80 µs/call avg
fattn_per_slot_kv_sm75_combine_kernel(...)   — small (Dv-thread reduction)
fattn_per_slot_kv_sm75_compute_pb_kernel(...) — small (per-slot fp ops)
fattn_per_slot_kv_sm75_init_meta_kernel(...)  — small (clear -INF)
```

Grid at decode NP=1: `(1, 4, MAX_PB_HARD_CAP=16) = 64 CTAs`. CTAs at
ip >= pb_for_slot return early after sentinel meta write. CUDA graph
capture compatible (no host sync).

## Headline comparison vs baseline-prod/wmma_f16 (small n_kv ≈ 24-56)

| Metric | wmma_f16 baseline | Stage 2.3 on-device | Δ |
|---|---:|---:|---:|
| Avg time per call | 20.4 µs | 80.5 µs | +295% slower |
| Instances | 1024 | 1024 | — |
| % of total decode time | 0.8% | 3.2% | +2.4 pp |
| Grid CTAs | 96 | 64 | -33% |

Iteration on Stage 2.3 from Stage 2.2b (single-pass): 82 → 80 µs. Minimal
delta at THIS shape because n_kv is small (24-56 during n_gen=32 from a
24-token prompt) → pb=1 → split-K provides zero benefit. 15 of 16 grid.z
CTAs are sentinel-returns.

## Why the gap persists at small n_kv

At small n_kv per slot, the kernel is launch-overhead and SMEM-staging bound.
Per-CTA work is dominated by:
- Cooperative Q SMEM load (8 KiB)
- Cooperative K SMEM load (8 KiB per K-block)
- Cooperative V SMEM load (8 KiB per K-block)
- mma + cross-warp D reduction
- Output write

wmma_f16's path is leaner per CTA (255 regs/thread, smaller SMEM staging
shadow), gets work done in fewer cycles even though it does ~16x more wasted
mma rows. The new kernel's Approach C pack (6/16 useful rows = 37.5%) is
better mma utilization than wmma_f16's ~12.5%, but doesn't compensate for
the SMEM staging overhead at this n_kv.

## Long-context measurement needed

The perf comparison above is at **small n_kv**. At long-context decode
(n_kv = 4096+), Stage 2.3 split-K actually splits: pb = 16 → grid = (1, 4, 16)
= 64 CTAs with REAL work per CTA. The mma's dominate; SMEM staging amortizes.
That's where the new kernel design pays off.

We don't yet have a measurement at production-typical n_kv (1k-10k+). The
existing test runs at n_gen=32 starting from 24-token prompts, so max n_kv
stays under 60. Long-context capture is a follow-up.

## End-to-end throughput

Production smoke at NP=1 MTP draft=3:
- Stage 2.2b dispatch: 51 t/s
- Stage 2.3 on-device dispatch: 50.5 t/s
- (wmma_f16 baseline production was historically ~55 t/s)

~9% absolute throughput cost at the production smoke shape, attributed to
the +60 µs per FA call × 16 FA layers × ~30 decoded tokens/sec.

## What's locked

- **Determinism (structural)**: GREEN. Per-row CTA + per-slot K-loop bound +
  no cross-block atomics. Unit test scenario C 464/464.
- **Production integration**: GREEN. End-to-end run on production model.
- **NP={2,4,8} validity**: GREEN. All slots PASS.
- **CUDA graph capture compatibility**: GREEN. No host sync; works with
  graph_reuse=true.

## What's still open

- **Decode perf at long n_kv vs wmma_f16**: NOT MEASURED. Expected to be
  the regime where Stage 2.3 pays off; capture needed at n_kv ~4k.
- **Decode perf at small n_kv vs wmma_f16**: 4× slower. Need to either
  reduce SMEM staging cost OR accept the gap as the cost of determinism.
- **End-to-end byte-id NP=2 ≡ NP=4 ≡ NP=8 on identical prompts**: NOT
  MEASURED at production shape.

## Data files

- `nsys-stage23-on-device-np1.nsys-rep` — current run with on-device Stage 2.3
- `nsys-vanilla-np1-q4_0-hadamard-new-fa-op.nsys-rep` — prior run with
  Stage 2.2b (preserved for diff)
