# Replacement perf — new FA op at production shape

Date: 2026-05-14
Branch: production/2026-q2-next
Capture binary: `test-np-validity-vanilla` (post Hadamard + new FA op dispatch)
Config: NP=1, n_gen=32, Q4_0 KV + Hadamard, production target GGUF.

## Production FA kernel observed

The new dispatcher routes to **Stage 2.2b** (single-pass Approach C decode pack).
Stage 2.3 split-K is deferred (host sync incompatible with CUDA graph capture
in production cparams; on-device pb computation is a follow-up).

```
fattn_per_slot_kv_sm75_stage22b_kernel(...)
```

## Headline comparison vs baseline-prod/wmma_f16

| Metric | wmma_f16 baseline (26 µs) | Stage 2.2b new (82 µs) | Δ |
|---|---:|---:|---:|
| Avg time per call | 20.4 µs | 82.3 µs | **+303% slower** |
| Instances | 1024 | 1024 | — |
| % of total decode time | 0.8% | 3.0% | +2.2 pp |
| Block size | 64 threads (2 warps) | 128 threads (4 warps) | + |
| Grid | 96 CTAs | 4 CTAs (Approach C decode pack) | -96× CTAs |

**The new kernel is 4× slower at decode shape.** Determinism + integration
goals are GREEN; perf binding is in the RED.

## Why the perf gap

1. **No split-K**: Stage 2.2b runs at parallel_blocks=1. At NP=1 decode, only
   `n_seqs × n_kv_heads × 1 = 1 × 4 × 1 = 4 CTAs` per kernel call, vs wmma_f16's
   96 CTAs (24 heads × 4 parallel-block splits). 24× less SM-fill.

2. **Stage 2.2b SMEM staging**: Q + K + V + KQ + partials all in SMEM. At small
   n_kv (decode early steps, n_kv ≈ 24-56 in this capture), the SMEM staging
   overhead dominates. wmma_f16 reads K/V from registers/L2 directly.

3. **Approach C pack at gqa=6**: 6 / 16 useful mma rows = 37.5%. wmma_f16's
   cols_per_block=8 = 1/8 = 12.5% useful at decode shape; ours is better at
   the kernel level but the grid-fill loss dominates.

## Follow-up: closing the perf gap

The perf gap is **expected and fixable**, not structural. Per Q&A 12a, Stage
2.3 split-K is the design path that closes it. The blocker is that the current
Stage 2.3 launcher needs a host sync to compute max_pb (for grid.z), which
breaks CUDA graph capture in production.

**Concrete next step**: rewrite the launcher to compute pb_per_slot and
max_pb on the device (tiny reduction kernel), use a hard upper bound for
grid.z (e.g., 32), have CTAs with ip >= pb_for_slot write sentinel and return.
This unlocks split-K under graph capture. At NP=1 long-context (n_kv=4096),
pb=16 → grid = (1, 4, 16) = 64 CTAs vs wmma_f16's 96 → comparable SM fill,
mma + pool allocation tradeoff dominates.

## Top-time decode kernels for context (this capture)

| Kernel | % | Avg time |
|---|---:|---:|
| ncclDevKernel_AllReduce_Sum_f32 | 32.5% | 107 µs |
| mul_mat_vec_q<q4_0,1,4> | 13.7% | 26.8 µs |
| fused_mul_mat_vec_q<q4_0,1,4> | 12.6% | 85.9 µs |
| ncclDevKernel_AllReduce_Sum_f16 | 9.0% | 995 µs |
| cutlass_75_wmma_tensorop | 6.9% | 62.9 µs |
| **fattn_per_slot_kv_sm75_stage22b** | **3.0%** | **82.3 µs** |

NCCL all-reduce remains the dominant ~42% (f32 + f16 combined). FA is only
3% of decode time, so the 4× per-call slowdown moves total decode by ~2 pp.
Measured throughput: 51 t/s (production smoke). Need to compare against a
fresh baseline at same conditions; if wmma_f16 baseline was 54-55 t/s, the
new op costs ~5% absolute throughput at decode shape pending Stage 2.3.

## Stage 3 closure status

- Determinism (per-slot byte-id under unit test): **GREEN** — 464/464 across NP
- Production integration (boot + brief inference): **GREEN** — 51 t/s MTP draft=3
- NP validity at NP={2,4,8}: **GREEN** — all slots PASS per data/deltanet/perf/np-binding/
- Per-kernel perf vs wmma_f16 baseline: **RED — 4× slower at decode** (above)
- End-to-end byte-id NP=2 ≡ NP=4 ≡ NP=8 on identical prompts: **NOT MEASURED** (follow-up)

Honest assessment per CLAUDE.md §4 "no follow-up cover":  Phase 2 ships
determinism + integration. Phase 2 does NOT yet ship competitive decode perf.
Follow-up is Stage 2.3 with on-device pb computation.
