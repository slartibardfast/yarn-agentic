# T6.2.b ncu deep-dive — `mul_mat_q_split_k` family

Profiled the dominant kernels surfaced by T6.2 nsys (`data/t6.2-nsys-prod-20260523T214629/`). Three variants captured in one application-replay session at the same production decode shape (NP=2, ctx 524288, Q4_0 KV + Hadamard, --parallel 2, --replay-mode application to work around the 27B-model context-save limit).

## Per-kernel summary

### 1. `mul_mat_q_split_k<Q4_0_AR16 (type 159), 8, 8, 0, 4>` — the 6.6% kernel (Hadamard-rotated weights variant)

| Metric | Value |
|---|---:|
| Grid × Block | 40×1×4 × 32×8×1 = 40,960 threads |
| Duration | 34.1 µs |
| DRAM Throughput | **47.2%** |
| Compute (SM) Throughput | **21.5%** |
| L1/TEX Cache Throughput | 44.6% |
| L2 Cache Throughput | 19.3% |
| Registers per Thread | **128** |
| Dynamic Shared Mem per Block | **45,056 B** (out of 64 KiB) |
| Block Limit by Shared Mem | **1** |
| Block Limit by Registers | 2 |
| Theoretical Occupancy | **25.0%** |
| Achieved Occupancy | 25.0% |
| Achieved Active Warps per SM | 8 (of theoretical 32) |

### 2. `mul_mat_q_split_k<Q4_0, 8, 8, 0, 4>` — the 31.0% dominant kernel

| Metric | Value |
|---|---:|
| Grid × Block | 24×1×4 × 32×8×1 = 24,576 threads |
| Duration | 41.2 µs |
| DRAM Throughput | **44.3%** |
| Compute (SM) Throughput | **17.5%** |
| L1/TEX Cache Throughput | 37.7% |
| L2 Cache Throughput | 14.6% |
| Registers per Thread | 125 |
| Dynamic Shared Mem per Block | **40,960 B** (out of 64 KiB) |
| Block Limit by Shared Mem | **1** |
| Theoretical Occupancy | **25.0%** |
| Achieved Occupancy | 25.5% |
| Achieved Active Warps per SM | 8 (of theoretical 32) |

### 3. `mul_mat_q_split_k_fixup<8, 8, 0, 4>` — the 1.4% reduction-tail kernel

| Metric | Value |
|---|---:|
| Duration | 1.9 µs (tiny) |
| DRAM Throughput | 4.6% |
| Compute Throughput | 0.6% |
| Achieved Occupancy | 15.5% (theoretical 100%) |

## Diagnosis: shared-memory-occupancy-bound, not bandwidth or compute

Both dominant kernels show:

- **DRAM at 44-47% of peak** — half-saturated.
- **Compute at 17-22% of peak** — well below saturation.
- **Occupancy stuck at 25%** because `Block Limit Shared Mem = 1` (40-45 KiB per block out of 64 KiB shared-mem-per-SM cap).

Neither memory bandwidth nor compute is the dominant limiter individually. Both are partially used because **occupancy is too low to hide DRAM latency**. With only 1 block per SM (= 8 warps per SM, vs sm_75's 32-warp capacity), warp schedulers stall waiting for memory and can't fill the gap with other warps.

This is a **latency-bound regime caused by shared memory pressure**, not a compute or bandwidth ceiling. The TU102 peak (672 GB/s DRAM, 16.3 TFLOPS FP32, 33 TFLOPS FP16 tensor-core) is far from utilized.

### Estimated upside if shared mem could be halved

If the kernel were re-engineered to use <32 KiB shared mem per block, occupancy could rise to 50% (2 blocks/SM). With 2× the warps in flight, DRAM latency hiding improves; the kernel would move from 44% → ~70-75% DRAM throughput. Estimated speedup on this kernel ≈ **1.6×**.

The dominant kernel is 31% of GPU time. A 1.6× speedup there → ~12% reduction in total decode wall time. Combined with the 6.6% Q4_0_AR16 kernel at similar potential, total ≈ **15% T6 perf headroom** on matmul alone.

This sets the realistic ceiling for matmul-kernel optimization on sm_75 without rewriting to use tensor cores in the dequantize path. vLLM's Marlin int4 kernel achieves higher occupancy because int4 weights pack more per shared-mem-byte and use tensor cores to keep compute saturated.

## What this means for the 6.37× vLLM gap (revised attribution)

From T6.2 nsys + this ncu pass:

| Source | Estimated factor | Lever |
|---|---:|---|
| Precision (BF16+Q4_0 vs int4-Marlin) | ~1.7× | Out of scope (model file change) |
| `mul_mat_q_split_k` shared-mem occupancy limit | ~1.6× | T7 kernel rewrite candidate (reduce shared mem usage) |
| NCCL AllReduce overhead at small-batch decode | ~1.4× | T6.2.c: probe `--split-mode layer` (eliminates AllReduce by keeping each layer on one GPU) |
| Other (norms, casts, scheduler, sampling) | ~1.1-1.2× | T6.4 (admission), T6.9 (dispatch) cover some |
| **Combined (multiplicative)** | **~4.5×** | |

This accounts for ~71% of the 6.37× gap. Remaining ~1.4× is likely workload-shape (gate0 varied prompts vs the hot-path optimized for identical-prompt benches) and HTTP/server scheduling overhead (not in this bench-target trace; relevant to T6.4).

## Next steps (T6 follow-ons)

- **T6.2.c (probe)** — run a sibling profile with `--split-mode layer` instead of `graph`, re-bench at gate0 NP=2. Hypothesis: eliminates the 25.6% AllReduce cost; possible whole-decode speedup of 20-30%. Quick to set up (just a flag), 1-bench to confirm.
- **T6.7** — PSKV singlewarp deep-dive at lower priority (3.2% of time, not the dominant cost). Still owed unconditionally per T6 discipline.
- **T7 candidate: `mul_mat_q_split_k` shared-mem reduction.** Largest single matmul lever identified by this pass. Requires rewriting the per-block tile stride to use less shared mem.
- **T6.3** — DFlash characterisation remains the highest-priority unconditional follow-on (driven by T6.1 finding, not changed by T6.2).
