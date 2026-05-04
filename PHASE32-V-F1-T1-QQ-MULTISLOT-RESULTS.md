# V-F1.T1.qq Multi-Slot Results — post-detector strand

`/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless.gguf`
(17.93 GiB, 5.72 BPW, Q4_0_AR16 attn + FP16 MTP head; PPL 7.0169 ± 0.046).

ik_llama.cpp at `phase32-q4_0_ar16-integration` (commit d7917cf0 — Phase B
runtime-single-seq kernel + Phase D contiguous-block coalescing landed).

Hardware: 2× Quadro RTX 6000 (TU102, sm_75, 24 GiB each), CUDA 13.2,
`--device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 -ngl 999 -fa on`,
`-c 4096 * np`, `--batch-size 2048 --ubatch-size 512`,
`--no-context-shift`, deterministic temp=0, 8 distinct prompts × 40
predicted tokens each.

## Aggregate throughput (sum across np concurrent slots)

| np | mtp t/s | nomtp t/s | mtp ratio vs np=1 | nomtp ratio vs np=1 |
|----|---------|-----------|-------------------|----------------------|
| 1  | 35.36   | 34.09     | 1.000×            | 1.000×               |
| 2  | 30.86   | 31.33     | 0.873×            | 0.919×               |
| 4  | 27.73   | 34.62     | 0.784×            | **1.016×**           |
| 8  | 28.79   | 35.59     | 0.814×            | **1.044×**           |

## Per-slot throughput

| np | mtp t/s/slot | nomtp t/s/slot |
|----|--------------|-----------------|
| 1  | 35.36        | 34.09           |
| 2  | 15.43        | 15.67           |
| 4  | 6.93         | 8.66            |
| 8  | 3.60         | 4.45            |

## Notes

1. **MTP is a single-user latency feature, not a multi-user throughput
   feature.** At np=1 MTP wins by 3.7% (35.36 vs 34.09 t/s); the
   crossover happens between np=1 and np=2, after which nomtp leads.
   At np=8 each nomtp slot gets 4.45 t/s vs 3.60 for mtp — a 24%
   per-user advantage with MTP off. The MTP verify+draft overhead
   amplifies in multi-slot due to extra reduce traffic per produced
   token.

2. **nomtp at np≥4 slightly EXCEEDS np=1 aggregate.** Scheduler
   slack + draft elimination lets multi-slot keep the GPUs more
   evenly fed than single-stream.

3. **Fallback counter == 0 across every cell.** The qnext mixed-seq
   detector never fires; the engine routes everything through the
   per-block fast path. Phase B + D landed cleanly.

4. **Per-slot scaling falls off due to PCIe-reduce contention.**
   nsys at np=8: `k_reduce_add_T` (cross-GPU split-K reduce) is
   29.2% of GPU time. With Quadro RTX 6000 NVLink 2.0 (~100 GB/s
   bidirectional vs PCIe Gen3 x8 ~16 GB/s), this kernel's cost
   should drop by 5-10×, plausibly lifting mtp-np=8 above np=1.

## Recommended production policy

- **Default**: `-mtp --parallel 1` for single-user interactive workloads
  (best latency at 35 t/s).
- **Multi-user agentic**: `-no-mtp --parallel 4` for 8.66 t/s/user
  with 4 concurrent agents (vs 6.93 t/s/user with MTP on).
- **Heavy concurrency**: `-no-mtp --parallel 8` at 4.45 t/s/user.
- Re-evaluate once NVLink lands; the mtp-np≥2 ratio should
  significantly improve, possibly flipping the crossover point.
