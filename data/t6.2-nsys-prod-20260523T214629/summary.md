# T6.2 nsys decode-region trace — kernel attribution

**Trace:** `bench.nsys-rep` (78 MiB)
**Workload:** `llama-batched-bench` at production no-spec config (NP=2 prompts × 256 TG tokens, --parallel 2, ctx 524288, Q4_0 KV + Hadamard, FA on, no DFlash).
**Capture window:** delay 15s, duration 12s (steady-state decode).
**Clocks:** 1455 MHz (locked-class P5 idle).

## Top-N kernels by GPU time

| % | Total (ms) | Instances | Avg (µs) | Kernel | Bucket |
|---:|---:|---:|---:|---|---|
| **31.0** | 4461 | 80986 | 55.1 | `mul_mat_q_split_k<Q4_0, 8, 8, 0, 4>` | **Q4_0 matmul (FFN/MoE/proj)** |
| **25.6** | 3683 | 45532 | 80.9 | `ncclDevKernel_AllReduce_Sum_f32_RING_LL` | **Cross-GPU AllReduce** |
| 8.3 | 1189 | 45628 | 26.0 | `cutlass_75_wmma_tensorop_h161616gemm` | f16 GEMM via cutlass (probably lm_head / f16 path) |
| 8.0 | 1145 | 11994 | 95.5 | `fused_mul_mat_vec_q<Q4_0, 2, 4, 1>` | Q4_0 mat-vec (decode-shape) |
| 6.6 | 952 | 22697 | 42.0 | `mul_mat_q_split_k<type 159, 8, 8, 0, 4>` | Q4_0_AR16 matmul variant |
| 3.8 | 550 | 6274 | 87.7 | `fused_mul_mat_vec_q<Q4_0, 1, 4, 1>` | Q4_0 mat-vec (single-token) |
| **3.2** | 459 | 4331 | 106.0 | `flash_attn_per_slot_kv_singlewarp_kernel<256, 256, Q4_0, Q4_0>` | **PSKV singlewarp (attention)** |
| 2.7 | 391 | 54056 | 7.2 | `cpy_flt<float,float>` | Copy (defrag/K-shift/set_rows) |
| 1.5 | 210 | 45438 | 4.6 | `fused_rms_norm_f32<1024>` | RMS norm |
| 1.4 | 206 | 103681 | 2.0 | `mul_mat_q_split_k_fixup<8,8>` | Split-K reduction tail |
| 1.2 | 171 | 22697 | 7.5 | `delta_net_recurrent_f32<128, 256>` | DeltaNet recurrent (hybrid layers) |
| 0.9 | 131 | 72324 | 1.8 | `quantize_mmq_q8_1` | Q8_1 quant (matmul input) |
| 0.7 | 106 | 22697 | 4.7 | `concat_f32_dim0` | Concat |
| 0.6 | 86 | 45394 | 1.9 | `cublasLt splitKreduce_kernel<32>` | cuBLAS reduction tail |
| 0.5 | 78 | 45394 | 1.7 | `l2_norm_f32<256>` | L2 norm |
| 0.5 | 71 | 45628 | 1.6 | `convert_unary<float, __half>` | fp32↔fp16 cast |
| 0.4 | 64 | 45628 | 1.4 | `convert_unary<__half, float>` | fp16↔fp32 cast |

## Bucket summary

| Bucket | % | Notes |
|---|---:|---|
| **Q4_0 matmul (all variants + fixup + quantize)** | **~62%** | mul_mat_q_split_k (31+6.6) + fused_mul_mat_vec_q (8+3.8) + cutlass h161616 (8.3, probably lm_head) + fixup (1.4) + quantize (0.9) + splitKreduce (0.6) + casts (1) |
| **NCCL AllReduce** | **25.6%** | Cross-GPU sync, graph-split overhead |
| **Attention (PSKV singlewarp)** | **3.2%** | The kernel T3 closures cited as the bottleneck |
| **Copy / set_rows / cpy_flt** | **2.7%** | Defrag + K-shift + SET_ROWS |
| **Norms (RMS + L2)** | **2.0%** | |
| **DeltaNet recurrent / SSM** | **~1.5%** | Qwen 3.5 hybrid layers |
| Other (concat, silu, hadamard inline, rope, ...) | ~3% | |

## Headline findings

### 1. NCCL AllReduce is the second-biggest GPU consumer at 25.6%

Graph-split with 2 GPUs requires an AllReduce after every layer's tensor-parallel computation. At NP=2 decode (1-2 token batches per tick), each AllReduce launches a small-message kernel that pays full RING-LL latency. 45,532 invocations over 12 seconds = ~3,800 AllReduces/sec. The previously-invisible cost of TP-sync on small batches.

**Implication:** The "graph-split is free" assumption from earlier phase work is wrong — it's the single largest non-matmul cost. Reducing AllReduce-per-token is a 25% potential win. Levers:
- Larger NVLink-fused AllReduce (we have peer access)
- AllReduce batching across layers (single bigger msg vs many small)
- Going back to **--split-mode layer** (no AllReduce; replaces with whole-layer-on-one-GPU) — only viable if a model layer fits on one 24 GiB GPU. For Qwen 3.6 27B at Q4_0 weights, a single transformer layer fits comfortably; this is a real option to evaluate in T6.

### 2. PSKV singlewarp attention is only 3.2% of GPU time

The kernel that earlier T3/T5 closure docs identified as the bottleneck contributes ~1/8 the cost of the dominant matmul kernel and ~1/8 the cost of AllReduce. Past optimization effort spent on PSKV ILP / register tuning was real but targeted the wrong dominant cost.

**Implication:** Future kernel-level optimization should target the Q4_0 matmul (31% of time, 80,986 invocations, avg 55µs) before PSKV. T6.7 (per-slot-kv FA characterisation) should still happen, but with the understanding that the kernel is not the dominant cost.

### 3. Q4_0 matmul kernels dominate at ~60% combined

Five distinct Q4_0 matmul kernels appear in the top 10:
- `mul_mat_q_split_k<Q4_0, 8, 8, 0, 4>` — 31.0% — likely the FFN/MoE expert matmul (PP-shape compute)
- `fused_mul_mat_vec_q<Q4_0, 2, 4, 1>` — 8.0% — Q4_0 mat-vec (decode-shape compute)
- `mul_mat_q_split_k<type 159, ...>` — 6.6% — Q4_0_AR16 variant (Hadamard-rotated weights for attention proj?)
- `fused_mul_mat_vec_q<Q4_0, 1, 4, 1>` — 3.8% — single-token mat-vec
- `mul_mat_q_split_k_fixup` — 1.4% — split-K reduction tail

Plus `cutlass_75_wmma_tensorop_h161616gemm` at 8.3% (f16, probably lm_head's F16 output).

**Implication:** The matmul throughput is the dominant single dimension. ik_llama's mul_mat_q_split_k on sm_75 may be ~1.5-2× slower than vLLM's Marlin int4 kernel (precision + kernel quality combined). Most of the 6.37× vLLM gap likely attributes here, in combination with AllReduce.

## Estimated attribution of the 6.37× vLLM gap

A coarse model, to be refined by T6.3-T6.9:

| Source | Estimated factor |
|---|---:|
| Precision (BF16+Q4_0 vs int4-Marlin) | ~1.7× |
| AllReduce overhead at small-batch decode | ~1.4× |
| mul_mat_q_split_k vs Marlin kernel quality on sm_75 | ~1.5× |
| Other (dispatcher / sampling / norms) | ~1.2× |
| **Combined (multiplicative)** | **~4.3×** |

This leaves ~1.5× unaccounted; candidates include workload-shape differences, KV layout cost, MoE expert routing overhead, and scheduler/admission cost (not present in this bench-target trace; bears on T6.0.a's 6.37× which IS the server measurement).

## Next steps (T6 follow-ons)

- **T6.2.b** — ncu deep-dive on `mul_mat_q_split_k<Q4_0, 8, 8, 0, 4>` (the 31% kernel). Per-CTA register/occupancy budget; what's the saturation factor against sm_75 peak FLOPs? Targeted invocations only (1 in 10000).
- **T6.2.c** — AllReduce-batching feasibility probe. Is the NCCL ring-LL choice optimal for sm_75 with peer-access? Probe `--split-mode layer` as a sibling profile and compare end-to-end t/s.
- **T6.7** — PSKV singlewarp deep-dive, scoped now as "what's the cost surface of the 3.2% it contributes" rather than "what's blocking it". Lower priority than expected.
- **T6.3** — DFlash characterisation (already highest-priority per T6.1 closure; T6.2 doesn't change that ordering).
