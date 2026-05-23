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

### 1. NCCL AllReduce is the second-biggest GPU consumer at 25.6% — PCIe-bound

Graph-split with 2 GPUs requires an AllReduce after every layer's tensor-parallel computation. At NP=2 decode (1-2 token batches per tick), each AllReduce launches a small-message kernel that pays full RING-LL latency. 45,532 invocations over 12 seconds = ~3,800 AllReduces/sec.

**Critical context (2026-05-23):** This host currently has **no NVLink** between the two RTX 6000s — AllReduce travels via PCIe-peer-access, which is small-message-latency-poor. NVLink is being installed tomorrow. After install, the same RING-LL AllReduce will benefit from ~10× lower latency on small messages, dramatically reducing this 25.6% share. The cheapest possible "fix" is the hardware install itself; T6.2 must be re-run post-NVLink to establish the new baseline.

**Implication:** The "graph-split is free" assumption from earlier phase work was wrong on PCIe-only hardware. On NVLink it may become much closer to free. The recommended action is to RE-MEASURE T6.2 after NVLink lands — that result gates whether any in-software AllReduce-reduction work is justified. Levers within graph-split, if needed post-NVLink, would include AllReduce batching across layers and reducing AllReduce participation per token; --split-mode layer is OUT by user lock 2026-05-23 regardless of NVLink.

### 2. PSKV singlewarp attention is now 3.2% of GPU time — the ILP recovery worked

The PSKV singlewarp 4-way ILP recovery (2026-05-18, see `[[project-pskv-ilp-recovery-landed]]`) delivered TG +2.95% / PP +9.17% with ncu per-CTA −32.7%, NPC PASS at NP={1,2,4,8}. T6.2 measures the kernel's share at **3.2% post-optimization** — the recovery successfully shrank the kernel's footprint in the decode region. What looks unimpressive in retrospect is the right outcome of focused kernel work: the kernel is no longer the cost it was.

The earlier T3/T5 closure docs framed PSKV as the bottleneck because it WAS more prominent before ILP recovery. T6.2's headline isn't "PSKV was the wrong target" — it's "PSKV was a real target, was addressed, and what's now dominant is the next layer of the cost surface that became visible after PSKV was reduced."

**Implication:** Future kernel-level optimization should target the Q4_0 matmul (31% of time, 80,986 invocations, avg 55µs) — the next-tier cost that became prominent. T6.7 (per-slot-kv FA characterisation) is still owed as an unconditional deep-dive, but with the understanding that the kernel's share is already small.

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

- **T6.2.b** — ncu deep-dive on `mul_mat_q_split_k<Q4_0, 8, 8, 0, 4>` (the 31% kernel). DONE — see `data/t6.2-ncu-20260523T215358/summary.md`. Shared-memory-occupancy-bound at 25%; ~1.6× upside if shared mem can be halved.
- **T6.2.d (NEW)** — re-run T6.2 nsys after NVLink install (2026-05-24). Re-measure the AllReduce share; this is the cheapest possible AllReduce optimization (hardware install rather than software change). T6.2's kernel attribution is the pre-NVLink baseline; the post-NVLink measurement is what informs T7 prioritisation.
- **T6.7** — PSKV singlewarp deep-dive. Scope is now "characterise the 3.2% residual after the T3.5 ILP recovery" rather than "what's blocking it". Lower priority than expected.
- **T6.3** — DFlash characterisation (highest-priority unconditional follow-on per T6.1 closure; T6.2 doesn't change that ordering).
- **T7 candidate** — `mul_mat_q_split_k` shared-memory reduction (largest single matmul lever). Requires kernel rewrite. NOT gated on NVLink.
- **T7 candidate** — NCCL AllReduce reduction within graph-split. Conditional on post-NVLink T6.2.d result; if NVLink alone takes the share to <10%, this becomes low-priority.
