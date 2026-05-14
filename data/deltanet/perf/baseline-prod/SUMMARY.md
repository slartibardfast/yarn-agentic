# Production baseline — wmma_f16 at Q4_0 + Hadamard (P1/P2 captures)

Date: 2026-05-14
Branch: production/2026-q2-next
Capture binary: `tests/dflash-speculative/test-np-validity-vanilla.cpp` with
2-LOC patch flipping `cparams.k_cache_hadamard = cparams.v_cache_hadamard = true`
(production-supported cparams, no new code path).
Config: NP=1, n_gen=32, Q4_0 KV + Hadamard, --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1
Model: `/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf`

## Files

- `nsys-vanilla-np1-q4_0-hadamard.nsys-rep` — full per-kernel timeline (P1, 12.5 MB)
- `nsys-vanilla-np1-q4_0-hadamard.sqlite` — nsys SQLite export (35 MB)
- `ncu-fa-vanilla-np1-q4_0-hadamard.ncu-rep` — ncu application-replay metrics on flash_attn_ext_f16 (P2, 34 MB)

## Production-confirmed shape

`flash_attn_ext_f16<256, 256, 1, 8, 2, 1, false, false>` — template params:
- Dk (HEAD_DIM_Q) = 256
- Dv (HEAD_DIM_V) = 256  ← **corrects prior spec lock of 128**
- ncols = 1 (decode shape)
- cols_per_block = 8
- VKQ_stride = 2
- parallel_blocks = 1
- KQ_acc_t = float (fp32 accumulators)
- use_softcap = false

GGUF metadata confirms: `qwen35.attention.key_length = 256`, `qwen35.attention.value_length = 256`,
`qwen35.attention.head_count = 24`, `qwen35.attention.head_count_kv = 4` (gqa_ratio = 6).

## P1 nsys per-kernel summary (decode, n_kv ≈ 24-56)

Top-time FA-related kernels:
| Kernel | % of total | Instances | Avg time |
|---|---:|---:|---:|
| flash_attn_ext_f16<256, 256, 1, 8, 2, 1, false, false> | 0.8% | 1024 | 20.4 µs |
| flash_attn_stream_k_fixup<256, 1, 8> | 0.2% | 1024 | 4.0 µs |
| flash_attn_mma_ext_f16<256, 32, 2, 4, 32, 2, false> | 0.1% | 64 | 43.5 µs |
| flash_attn_mma_stream_k_fixup<256, 32, 2, 32> | 0.1% | 64 | 28.8 µs |

Top non-FA kernels for context:
- ncclDevKernel_AllReduce_Sum_f32_RING_LL: 28.6% (! dominant)
- mul_mat_vec_q<q4_0, 1, 4>: 15.1% (projection GEMMs)
- fused_mul_mat_vec_q<q4_0, 1, 4>: 14.0% (fused MLP)
- ncclDevKernel_AllReduce_Sum_f16_RING_LL: 9.5%

**Observation**: NCCL all-reduce dominates decode time at 38% combined. The
FA kernel itself is only ~1% of decode wall-clock. The DeltaNet replacement's
perf binding must account for this: even a 10× speedup on FA only reduces
total decode time by ~0.7%. Determinism is the primary win; perf must not
regress, but the SoTA mandate's % of HBM/tensor-core peak applies at the
KERNEL level, not at the decode-step level.

## P2 ncu per-kernel metrics (wmma_f16 decode floor)

| Metric | Value | Notes |
|---|---:|---|
| Block size | 64 threads (2 warps) | Smaller than spec's 128-thread CTA target |
| Grid size | 96 CTAs | Roughly n_heads_q × ip per kernel call |
| Threads | 6144 | grid × block |
| **Registers / thread** | **255** | **MAXED OUT (Turing cap = 255)** |
| Dynamic SMEM / block | 16.98 KiB | Driver-set |
| Theoretical occupancy | 18.75% | Limited by **shared memory** (3 blocks/SM SMEM cap) |
| Achieved occupancy | 7.9% | Workload imbalance — some SMs do 90% more work than others |
| Waves per SM | 0.44 | Kernel grid too small to fill GPU |
| Duration | 26 µs / call | Matches nsys average |
| **DRAM throughput** | **3.7%** | ~23 GB/s of 624 GB/s peak — well below memory bound |
| **Compute (SM) throughput** | **0.63%** | ~0.8 TFLOPs of 130.5 TFLOPs peak |
| L1/TEX cache throughput | 6.5% | |
| L2 cache throughput | 1.3% | |

### Headroom analysis

At n_kv ≈ 24-56 (early decode), wmma_f16 is **launch-overhead-bound + register-pressure-limited**:

1. **Compute SM throughput 0.63%** says compute is not the bottleneck.
2. **DRAM throughput 3.7%** says memory is not the bottleneck.
3. **Registers / thread = 255** + theoretical occupancy = 18.75% suggests
   the kernel is dominated by per-thread state (255 regs × 64 threads ≈
   16 KiB of register file per block). Block limit from registers = 4
   blocks/SM; block limit from SMEM = 3 blocks/SM.
4. **Waves per SM = 0.44** means the kernel doesn't even fill the GPU once.
   At grid=96, 72 SMs, 0.44 waves implies a fraction of SMs do work and
   the rest are idle for this kernel.

**Replacement targets** (against this baseline):
- Beat 26 µs/call at decode n_kv ≈ 50 (SoTA mandate; this is the production
  shape that runs 32 times per decoded token).
- ≤ 64 regs/thread (1/4 the current 255) → 2 blocks/SM, 25% theoretical occupancy.
- At long-context decode (n_kv=4096), target 60% of HBM = 374 GB/s effective.
  Memory traffic at Dv=256 = 16 MiB K+V; floor = 16/624 = 25.6 µs; target ≤ 35 µs.
- At prefill (n_tokens=1024), target ≥ 50% of fp16 tensor-core peak.

## ⚠️ Patch state and capture provenance

The 2-LOC Hadamard patch to `test-np-validity-vanilla.cpp` (cparams flips
only — no new code) is held LOCAL until the new kernel lands. Rationale:
the patch is exclusively in service of capturing this baseline data; it
will commit alongside the new-kernel S2.5.d replacement-capture batch so
the diff history reads cleanly: one commit setting up production-shape
profiling + capturing baseline + replacement.

If the user wants a single-commit baseline snapshot before kernel work,
the patch can be committed against this directory's data without code
implications.

## Discrepancy from earlier baseline

`data/deltanet/perf/baseline/SUMMARY.md` captured F16 KV at small batch via
llama-bench. That baseline measured tg1=29.7 t/s, pp512+ saturating at
385 t/s. Those numbers are still useful as F16-KV reference but are NOT
the production config. This `baseline-prod/` capture IS the production
config (Q4_0 + Hadamard) and is the authoritative pre-S2.5 floor.
