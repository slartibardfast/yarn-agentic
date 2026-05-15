# Replacement perf — new FA op vs wmma_f16 (small + long n_kv)

Date: 2026-05-15
Branch: production/2026-q2-next

## Two captures, same shape, two FA paths

Both runs use `test-np-validity-vanilla` at NP=1, n_gen=64, Q4_0 KV + Hadamard,
production target. The `LLAMA_FATTN_PER_SLOT_KV_DISABLE=1` env var forces the
build-graph to emit ggml_flash_attn_ext (wmma_f16) instead of the new
ggml_flash_attn_ext_per_slot_kv (Stage 2.3). Apples-to-apples.

### Small n_kv (24-token prompt, n_gen=32 → n_kv ≈ 24-56)

| FA path | Per-call avg | Instances |
|---|---:|---:|
| wmma_f16 (`flash_attn_ext_f16`) | 20.4 µs | 1024 |
| Stage 2.3 (`fattn_per_slot_kv_sm75_stage23_kernel`) | 80 µs | 1024 |

Ratio: Stage 2.3 is **4× slower** per call.

### Long n_kv (~1200-token prompt, n_gen=64 → n_kv ≈ 1200-1264)

| FA path | Per-call avg | Instances |
|---|---:|---:|
| wmma_f16 (`flash_attn_ext_f16` + `stream_k_fixup`) | 24.5 + 9.2 = 33.7 µs | 2048 |
| Stage 2.3 (`fattn_per_slot_kv_sm75_stage23_kernel` + `combine_kernel`) | 408 µs + small | 2048 |

Ratio: Stage 2.3 is **12× slower** per call.

**The gap WIDENS with n_kv (4× → 12×).** The earlier hypothesis "long context
closes the gap" was wrong.

## Why the gap widens

Both paths split-K at decode shape:
- wmma_f16: cols_per_block=8, parallel_blocks=4 → grid (4×3, 24, 1) = 288 — actually 96 per nsys; depends on ncols
- Stage 2.3: MAX_PB=16, gqa=6 packed → grid (1, 4, 16) = 64 CTAs

Per-CTA work:
- wmma_f16 CTA: 1 (head, query-row, ip) tuple. mma fragments × n_kv/parallel_blocks K-positions. Light SMEM (~17 KiB).
- Stage 2.3 CTA: 6 (head, query-row) tuples via Approach C pack. mma fragments × n_kv/pb_for_slot K-positions. PER-ROW softmax × 6 + V-accum × 6. Heavy SMEM staging (Q 8 KiB + K 8 KiB + V 8 KiB + KQ 2 KiB ≈ 26 KiB).

The Approach C pack was supposed to be a win at decode (37.5% mma util vs
12.5% for wmma_f16). In wall-clock terms it's a LOSS because the per-CTA work
scales with the number of packed rows and SMEM staging amortizes poorly.

## What's locked

- **Determinism (structural)**: GREEN. Algorithm is correct + batch-invariant.
- **Production integration**: GREEN. End-to-end smoke runs, NP={2,4,8} validity GREEN.
- **CUDA graph capture compatibility**: GREEN. No host sync.

## What's NOT locked

- **Decode FA per-call perf vs wmma_f16**: **RED at every shape measured**.
  4× slower at small n_kv, 12× slower at long n_kv. Per-call FA time is 3-3.2%
  of total decode wall-clock; 4-12× slowdown on that 3% bucket = +9-36% total
  decode regression on the FA bucket. Real-world tg t/s: 50.5 vs ~55 baseline
  ≈ -8%.

## Why the design lost

Approach C decode pack was the structural call from Q3 (3c). The mma
utilization argument (6/16 useful rows vs wmma_f16's 1/8) IS correct on a
per-instruction-issued basis. But the cost of packing — per-row softmax,
per-row VKQ rescale, larger SMEM staging — dominates the gain at our shapes.

wmma_f16's design philosophy: small per-CTA work, lots of CTAs, hide latency
via occupancy. Our design philosophy: pack more work per CTA, fewer CTAs,
better mma utilization. wmma_f16 wins.

## Concrete remediation paths

In order of likely impact, smallest LOC first:

1. **Disable the new op auto-routing** until perf parity is shown.
   Production stays on wmma_f16. Determinism story relies on the unit
   test's structural argument (which still holds). One-line change.

2. **Re-route only at NP > 1** (where the actual production bug fires).
   At NP=1 the bug doesn't trigger (batch invariance with single slot is
   trivial), so wmma_f16 at NP=1 = no determinism issue. New op only when
   it's needed; perf regression scope narrows. ~5-LOC change in
   build_std_attention.

3. **Redesign Stage 2.3 without Approach C pack** (revert to Approach A).
   Eats the m=16 mma at decode (1/16 useful), wins back the per-CTA work
   savings. Substantial kernel rewrite — same as taking Stage 2.2a as the
   default decode kernel + split-K wrapper. ~200 LOC.

4. **Accept the perf gap as the cost of determinism**, ship as-is.
   Production users see -8% decode t/s in exchange for np>1 working
   correctly. Honest tradeoff but worth surfacing.

## Data files

- `nsys-stage23-on-device-np1.nsys-rep` — Stage 2.3 small n_kv
- `nsys-stage23-on-device-longctx.nsys-rep` — Stage 2.3 long n_kv
- `nsys-wmma_f16-longctx.nsys-rep` — wmma_f16 long n_kv (LLAMA_FATTN_PER_SLOT_KV_DISABLE=1)
- (Reference: `../baseline-prod/nsys-vanilla-np1-q4_0-hadamard.nsys-rep` for wmma_f16 small n_kv)
