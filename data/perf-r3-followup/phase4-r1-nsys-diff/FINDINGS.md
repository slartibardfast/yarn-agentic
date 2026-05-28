# Phase 4 — R1 nsys kernel diff findings

**Run:** 2026-05-28 08:57Z
**Config:** NP=1, ubatch=256, RT chain, q4_0+Hadamard KV, 200t prompt, N_PREDICT=128
**Build:** `ik_llama.cpp/build/bin/llama-server` (b2cf8fbf C-arc + RT)
**Per ctx:** server boot + 1 warmup rep + 1 traced rep, single nsys trace each side
**Note:** .nsys-rep files (~129 MB each) NOT committed; available at
`/tmp/perf-r3-followup-phase4-20260528T085711/` until cleared.

## Traced TG (matches Phase 3 within instrumentation overhead)

| ctx | TG t/s | vs Phase 3 (untraced) |
|---:|---:|---:|
|   8192 | 16.20 | 19.34 (-16% nsys overhead) |
| 262144 | 12.47 | 14.34 (-13% nsys overhead) |

Ratio preserved: 12.47/16.20 = **-23.0%** vs Phase 3's -25.9%. R1 is real in
the trace; the gap to investigate is intact.

## What kern-sum and api-sum showed — and what they didn't

| view | ctx=8k total | ctx=256k total | Δ |
|---|---:|---:|---:|
| All 36 GPU kernels (cuda_gpu_kern_sum) | 17.0 s | 17.0 s | **+17 ms (+0.1%)** |
| All CUDA API calls (cuda_api_sum) | 14.39 s | 14.49 s | +97 ms (+0.7%) |
| GPU memset (cuda_gpu_mem_time_sum) | 303 ms | 443 ms | +139 ms (+46%) |
| Memcpy HtoD/DtoH/Peer | identical | identical | ~0 |

**Largest individual deltas at the kernel level** (top 5):

| kernel | ctx=8k_ns | ctx=256k_ns | Δns | Δ% |
|---|---:|---:|---:|---:|
| ncclDevKernel_AllReduce_Sum | 2,799,346,651 | 2,813,145,370 | +13,798,719 | +0.5% |
| mul_mat_q_split_k Q4_0 | 6,432,746,140 | 6,437,778,794 | +5,032,654 | +0.1% |
| fused_mul_silu_f32 | 48,004,409 | 45,764,839 | -2,239,570 | -4.7% |
| flash_attn_per_slot_kv | 1,255,632,093 | 1,255,740,538 | +108,445 | +0.0% |
| hadamard_f32 | 63,573,410 | 63,744,685 | +171,275 | +0.3% |

**Every GPU kernel does essentially the same work at the same speed.**
Sum of all 36-kernel deltas is ~+17 ms over ~17 s of total kernel work.
That cannot explain a +23% wall-time delta.

## OSRT — where the gap actually lives

OSRT (OS-runtime / syscall) totals across all server threads:

| syscall | ctx=8k thread-time | ctx=256k thread-time | Δ |
|---|---:|---:|---:|
| `pthread_cond_wait` | 274.0 s | 350.0 s | **+76.0 s (+27.7%)** |
| `poll` | 200.8 s | 243.0 s | **+42.1 s (+21.0%)** |
| `pthread_cond_timedwait` | 65.5 s | 84.0 s | **+18.5 s (+28.2%)** |
| `pthread_cond_clockwait` | 18.1 s | 23.2 s | +5.1 s (+27.9%) |
| `accept4` | 18.2 s | 23.3 s | +5.1 s (+27.8%) |
| `ioctl` | 335 ms | 375 ms | +40 ms (+12.0%) |
| **Total OSRT** | **579 s** | **725 s** | **+147 s (+25.4%)** |

(These are thread-seconds — summed across ~40 server threads — not
wall-clock. The signal is in the **uniform ~25-28% increase** across every
blocking primitive.)

## What this tells us

- Kernel exec on the GPU is identical at both ctx sizes.
- CUDA API calls on the host take essentially the same time.
- Every blocking syscall waits ~25-28% longer at ctx=256k.
- The +25-28% OSRT delta matches the +25.9% TG delta (Phase 3) and
  +23.0% TG delta (Phase 4 traced) — three independent measurements
  converge.

The only model consistent with all three: **kernels execute equally fast
on the GPU but inter-kernel gaps grow** at ctx=256k. CPU threads wait
longer per submit-launch-wait cycle. That stretches every cv_wait, every
poll, every futex park by the same factor — exactly what OSRT shows.

## Root-cause hypothesis (localized but not yet bound)

**Per-step host-side bookkeeping scales with allocated ctx-size**, and
manifests as larger GPU-idle gaps between kernel completions. Leading
suspects:

1. **Paged-KV block-table walks** — `total_pool_blocks = nbps × n_stream`
   grows with `--ctx-size`; every step traverses some portion of this.
   See `ik_llama.cpp/src/llama.cpp:967-968` for the per-stream block
   table identity setup.
2. **ggml-backend scheduler per-step graph work** — scanning the
   activation graph; if any pass scans the allocated KV pool, it grows
   with ctx.
3. **gallocr walks** across the activation buffers.

The Phase 3 T5.9 sub-test was supposed to discriminate this. It misfired
because the flags probed (`--cache-ram` / `--ctx-checkpoints`) are
host-side context-checkpoint caching, not the T5.9 GPU paged-KV layout
toggle. A true T5.9 A/B requires a pre-T5.9 build.

## What rules itself out (negative findings)

- ✗ FA kernel tile/warp tuning (FA exec time identical)
- ✗ cuBLAS algo retuning (Cutlass + cublasLt kernels identical)
- ✗ Hadamard path tuning (Hadamard exec time identical)
- ✗ NCCL workspace / topology (NCCL exec time identical)
- ✗ MMQ Q4_0 dequant kernel work (mul_mat_q exec time identical)
- ✗ Memcpy bandwidth saturation (HtoD/DtoH/Peer identical)
- ✗ CUDA API launch path (cudaLaunchKernel, cuLaunchKernel essentially identical)

## Methodology caveat

Phase 4 ran with `--capture-range=none`, so the kern-sum / api-sum /
mem-sum / osrt-sum totals cover the *entire* trace duration — including
server boot and warmup, not just the traced rep. Boot at ctx=256k is
itself longer (more KV to allocate and zero). The relative deltas above
should still be load-bearing because all three views grow uniformly,
but a clean per-traced-rep number would require either nvtx bracketing
or `--capture-range=cudaProfilerApi` with code instrumentation.

## Phase 4-refined (user-authorized 2026-05-28 09:00Z)

To turn the hypothesis into a bound localization, the next sub-phase
needs:

1. **CPU sampling enabled** (`--sample=process-tree`,
   `--cpuctxsw=process-tree`). Current trace turned both off; with them
   on we get function-level CPU profiles showing WHICH ggml/sched/paged
   functions consume per-step time.
2. **Trace bracketed to just the traced rep** — either via `--delay` /
   `--duration` with empirical warmup-end timing, or a small `cudaProfilerStart()`/
   `cudaProfilerStop()` instrumentation in server.cpp around the slot
   decode loop. The bracketing fixes the boot-time contamination caveat.
3. **Diff per-function CPU time** at ctx=8k vs ctx=256k. If a single
   function dominates the +25% growth, that's the binding finding.

## Artifacts

- `ctx{8k,256k}-{kern,api,mem-time,mem-size,osrt,kex}.csv` — raw nsys
  stats CSV exports for each side
- `kern-diff.csv` — kernel-level Δns / Δ% diff
- Per-rep response bodies + server logs at
  `/tmp/perf-r3-followup-phase4-20260528T085711/`
- `.nsys-rep` files (~129 MB each) at the same tmp path; not committed
