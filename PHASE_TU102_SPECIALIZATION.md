# PHASE_TU102_SPECIALIZATION — Qwen 3.6 27B production kernel ranking

**Opened:** 2026-05-19.
**Branch:** `production/2026-q2-next`.
**Scope:** Rank kernel-level TU102 (sm_75, dual Quadro RTX 6000) specialization
targets for Qwen 3.6 27B production inference, both phases (prefill + decode),
excluding DFlash speculative decoding (closed separately in
`PHASE_DFLASH_BATCHED_PINNED.md`).
**State:** Ranking landed. No kernel work started — each ranked target is a
separate downstream workstream with its own ncu pass + kernel-design lock.

---

## Goal

Identify the highest-yield kernel targets for TU102 specialization on the
current `production/2026-q2-next` binary, with byte-identical NPC preserved
across NP={1,2,4,8} multi-GPU (the six baked NPC fixes per
`PHASE_NPC4_FIX_AUDIT.md` are sacrosanct).

Deliverable: ranked Top 3 / Top 5 / Top 10 with current cost (ms + %),
achievable ceiling (where measurable), and the specific TU102 architectural
hook each target exploits.

---

## Data source

- Existing nsys captures at `data/nsys-perf-2026-05-17/` (4 traces):
  - `head-npl{1,8}.nsys-rep` — `production/2026-q2-next` @ bb5c37eb,
    six NPC fixes + F.4.1' baked in
  - `prenpc-npl{1,8}.nsys-rep` — `production/2026-q2` @ b07d0bbe,
    no NPC fixes (ceiling reference)
- Bench shape (both sides identical):
  ```
  llama-batched-bench -m qwen3.6-27b-V-F1.T1.qq-...gguf
    --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1
    -ngl 999 -fa on -c 4096 --cache-type-k q4_0 --cache-type-v q4_0
    -b 2048 -ub 512 --threads 16 -npp 200 -ntg 64 -npl {1|8}
  ```
- Per-kernel breakdown extracted via `nsys stats --report cuda_gpu_kern_sum`
  to `data/nsys-perf-2026-05-17/stats/*.csv`.
- No Hadamard rotation in this bench (llama-batched-bench flag parse
  mismatch; documented in `reference_production_kv_cache_config`). Both
  sides identical, so the ranking is internally consistent.

---

## Refreshed kernel breakdown (current production, 2026-05-19)

**Captured against the option C binary** (F16 lm_head GGUF + cuBLAS
`ALGO0_TENSOR_OP` baked + `LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH=0`) at the
active profile shape: `-c 524288 -npl 2 -khad -vhad` (np=2 × 256k/slot,
Hadamard on, q4_0 KV). Plus a `-c 262144 -npl 1` companion for the
prefill-heavy view. Bench: same canonical `-npp 200 -ntg 64`.

Captures: `data/nsys-perf-2026-05-19-prod/prod-np{1,2}-ctx256k.nsys-rep`.

Headline t/s (option C + F16 lm_head, production runtime path):

| | np=1 × 256k | np=2 × 256k |
|---|---:|---:|
| Wall time | 5.10 s | 26.00 s |
| PP t/s | 99.28 | 20.94 |
| TG t/s | 20.74 | 18.56 |
| Aggregate | 51.76 | 20.31 |

### Top kernels — np=2 × 256k (active production shape)

| # | Kernel | Time | % | Per-call | Calls |
|---|---|---:|---:|---:|---:|
| 1 | `mul_mat_q[Q4_0, ncols=8]` | 16.85 s | **38.8%** | 76.6 µs | 219 904 |
| 2 | `NCCL_AllReduce_f32` | 9.11 s | **21.0%** | 72.9 µs | 125 020 |
| 3 | `fused_mmvq[Q4_0, nc=1, F.4.1']` | 4.41 s | 10.2% | 86.7 µs | 50 946 |
| 4 | `mul_mat_q[Q4_0_AR16, ncols=8]` | 2.70 s | 6.2% | 53.2 µs | 50 688 |
| 5 | `cutlass_75_wmma_h161616` (incl. lm_head) | 2.69 s | 6.2% | 25.6 µs | 104 975 |
| 6 | `PSKV-singlewarp-FA` | 2.32 s | 5.4% | 156.5 µs | 14 848 |
| 7 | `cpy_flt[fp32→fp32]` | 0.90 s | 2.1% | 6.9 µs | 131 264 |
| 8 | `fused_mmvq[Q4_0, nc=2]` | 0.79 s | 1.8% | 95.1 µs | 8 318 |
| 9 | `fused_rms_norm` | 0.60 s | 1.4% | 4.8 µs | 125 485 |
| 10 | `quantize_q8_1` | 0.36 s | 0.8% | 1.9 µs | 190 778 |
| 11 | `delta_net_recurrent_f32` | 0.34 s | 0.8% | 6.7 µs | 50 784 |

Top 10 covers **~94%** of GPU time at np=2 × 256k.

### Top kernels — np=1 × 256k (prefill-heavy)

| # | Kernel | Time | % | Per-call | Calls |
|---|---|---:|---:|---:|---:|
| 1 | `NCCL_AllReduce_f32` | 2.79 s | **33.5%** | 164.9 µs | 16 892 |
| 2 | `mul_mat_q[Q4_0, ncols=8]` | 2.22 s | 26.7% | 77.3 µs | 28 676 |
| 3 | `fused_mmvq[Q4_0, nc=1, F.4.1']` | 0.71 s | 8.5% | 86.7 µs | 8 194 |
| 4 | `mul_mat_q[Q4_0, ncols=112]` (prefill tile) | 0.54 s | 6.5% | 781.5 µs | 696 |
| 5 | `cutlass_75_wmma_h161616` | 0.49 s | 5.9% | 9.2 µs | 53 825 |
| 6 | `mul_mat_q[Q4_0_AR16, ncols=8]` | 0.33 s | 3.9% | 53.2 µs | 6 144 |
| 7 | `PSKV-singlewarp-FA` | 0.27 s | 3.3% | 128.8 µs | 2 112 |

### What changed vs the 2026-05-17 baseline

| Kernel | 2026-05-17 (BF16, default algo, npl=8 × 4k) | 2026-05-19 (option C, F16, np=2 × 256k) |
|---|---:|---:|
| `cuBLAS_gemvx_bf16` (lm_head fp32 fallback) | 8.8 s, 5.3% | **GONE** — folded into `cutlass_75_wmma` at 25.6 µs/call |
| `mul_mat_q[Q4_0, nc=8]` | 30.4% | 38.8% (relative share grew with lm_head out) |
| `NCCL_AllReduce_f32` | 20.4% | 21.0% |
| `PSKV-singlewarp-FA` | 18.4% (597 µs/call at npl=8) | 5.4% (156 µs/call at np=2) — less concurrent, shorter wall |
| `cutlass_75_wmma_h161616` | 0.9% (3.8 µs/call) | 6.2% (25.6 µs/call) — lm_head moved here |

| # | Kernel | NP=8 decode-dominant | NP=1 prefill-dominant | Per-call NP=8 | Calls NP=8 |
|---|---|---:|---:|---:|---:|
| 1 | `mul_mat_q[Q4_0, ncols=8]` | 30.4% (50.7 s) | 23.8% (1.75 s) | 61.6 µs | 822 778 |
| 2 | `NCCL_AllReduce_f32` | 20.4% (34.0 s) | 36.3% (2.68 s) | 72.7 µs | 468 124 |
| 3 | `PSKV-singlewarp-FA` | 18.4% (30.7 s) | 5.0% (370 ms) | 597.6 µs | 51 326 |
| 4 | `fused_mmvq[Q4_0, nc=1, F.4.1']` | 10.5% (17.5 s) | 9.6% (704 ms) | 85.9 µs | 203 778 |
| 5 | `mul_mat_q[Q4_0_AR16, nc=8]` | 5.4% (9.1 s) | 3.7% (272 ms) | 44.8 µs | 202 737 |
| 6 | `cuBLAS_gemvx_bf16` (lm_head) | 5.3% (8.8 s) | 3.7% (271 ms) | 4174.8 µs | 2 105 |
| 7 | `cpy_flt[fp32→fp32]` | 1.6% (2.75 s) | 1.2% (87 ms) | 5.4 µs | 508 306 |
| 8 | `fused_rms_norm` | 1.1% (1.82 s) | 0.9% (67 ms) | 3.9 µs | 468 026 |
| 9 | `fused_mmvq[Q4_0, nc=8]` | 1.0% (1.64 s) | — | 197.5 µs | 8 318 |
| 10 | `cutlass_75_wmma_h161616` | 0.9% (1.54 s) | 2.3% (171 ms) | 3.8 µs | 408 354 |
| 11 | `delta_net_recurrent_f32` | 0.8% (1.26 s) | 1.0% (73 ms) | 6.2 µs | 202 827 |
| 12 | `mul_mat_q[Q4_0, ncols=112]` (prefill tile) | — | 5.7% (420 ms) | 608.1 µs | 690 |

Top 10 covers **95.8%** of NP=8 decode GPU time; top 12 covers **96.6%**.

### Pre-option-C breakdown above is stale; refer to the refreshed table for the current production binary.

### Pre-NPC ceiling reference (same bench)

| Kernel | HEAD NP=8 | Pre-NPC NP=8 | Δ |
|---|---:|---:|---:|
| MMQ Q4_0 family (#1 + #5) | 59.8 s | mul_mat_vec_q 21.1 s + cutlass 14.4 s = 35.5 s | **+24.3 s** |
| `PSKV-singlewarp-FA` (#3) | 30.7 s | `flash_attn_ext_f16` 1.16 s | **+29.5 s** |
| `fused_mmvq[Q4_0, nc=1]` (#4) | 17.5 s | mul_mat_vec_q (same shape) 17.5 s | ±0 |
| `cuBLAS_gemvx_bf16` (#6) | 8.8 s | 6.65 s | +2.1 s |
| `NCCL_AllReduce_f32` (#2) | 34.0 s | 35.6 s | -1.6 s (NP=8); at NP=1: +0.84 s + f16 path eliminated |

The pre-NPC ceiling is **not the optimization target** for this workstream —
the six baked NPC fixes are a binding requirement. Pre-NPC is shown only
to bound how much *of the regression* a non-NPC-breaking specialization can
plausibly recover. The TU102 specialization targets below aim *beyond* the
pre-NPC ceiling by exploiting sm_75 tensor cores for the same shapes.

---

## Top 3 — clear absolute wins

### #1 — `mul_mat_q[Q4_0,*]` → split-K MMQ — CLOSED 2026-05-19

**Closed via Lever B (split-K MMQ).** Final measured result at
production np=2 × 256k shape with F16 lm_head + cuBLAS ALGO0 baseline:

| | Before split-K | After split-K | Δ |
|---|---:|---:|---:|
| TG NP=2 × 256k | 18.56 t/s | **22.28 t/s** | **+20%** |
| PP NP=2 × 256k | 20.94 t/s | 23.60 t/s | +13% |
| Aggregate NP=2 | 20.31 t/s | 23.26 t/s | +15% |
| NPC matrix | PASS | PASS | unchanged |

Lever A (`launch_bounds(*,1)→(*,2)`) stacked on the split-K kernel
2026-05-19 — ncu shows 125 regs/thread (under the 128 cap), 0 spills,
but theoretical occupancy stays at 25% capped by **shared memory**,
not registers. So Lever A didn't unlock more warps as projected; the
shmem footprint of the load-tiles path is the new binding constraint.
Measured +0.7–1.5% across np=1/np=2 bench lines — small consistent
positive, kept. Final TG NP=2 × 256k = **22.44 t/s** (split-K + A).

**Lever C (shmem-reduction via `mmq_y=128 → 64`) attempted and reverted
2026-05-19.** The 38 KB `shmem_x` buffer is the binding occupancy
constraint. Halving `mmq_y` to 64 would drop shmem to ~19 KB and unlock
2 blocks/SM = 50% occupancy. Blocked by a hard architectural invariant
in `mmq.cuh:3805`:
```cpp
static_assert(nwarps*mma_C::I == mmq_y, "nwarps*mma_C::I != mmq_y");
```
The write-back maps `nwarps=8` warps × `mma_C::I=16` rows-per-warp = 128
output rows = `mmq_y`. Lowering `mmq_y` to 64 requires either nwarps=4
(half block; MMQ_NWARPS is a macro touched widely) or a smaller MMA
fragment (`mma_int_C_I8J8`, restructuring `vec_dot_q4_0_q8_1_mma`).

**Lever D — I=8 fragment ground-up rewrite — SHIPPED 2026-05-19** at
`f8fa3928` (submodule). Adds `mma_int_A_I8K4`, `mma_int_A_I8K8`,
`mma_int_C_I8J8` and a parallel `mul_mat_q_split_k_i8` kernel running
at `mmq_y=16 nwarps=2`. Drops shmem to ~6.4 KB/CTA, bids
`launch_bounds(64, 16)` for 100% theoretical occupancy. Hardware
delivers ~9 CTAs/SM = 56% theoretical (still shmem-capped, but 2.24×
boost over the prior 25%). REG:64 STACK:0 LOCAL:0 — 0 spills on Q4_0
and Q4_0_AR16, both `need_check` variants. Engaged only for sm_75 +
decode (mmq_x ≤ 16) + Q4_0/Q4_0_AR16 weights.

| | After split-K + Lever A | After Lever D (I=8) | Δ |
|---|---:|---:|---:|
| TG NP=2 × 256k | 22.44 t/s | **23.57 t/s** | **+5.04%** |
| PP NP=2 × 256k | 23.93 t/s | 25.71 t/s | +7.4% |
| 3-run noise | ±0.08 | **±0.01** | tight |
| NPC matrix | PASS | PASS | unchanged |

Realized win mechanism: shmem-reduction unlocks higher achieved
occupancy (latency hiding) AND smaller per-CTA shmem-bank contention.
Wave overhead increases (1.97 vs 1.11 waves at decode shape) but
the latency-hiding gain dominates.

**Reframed 2026-05-19 after ncu probe + source survey. Original framing
(engage int8 IMMA TC) was wrong — kernel already uses int8 IMMA on
Turing.**

- **Aggregate cost.** 16.85 s NP=2 × 256k (38.8%) + 9.1 s sibling
  `Q4_0_AR16` (6.2%) = **~45% of GPU time** with the bundle.
- **What's actually true.** `INT8_MMA_AVAILABLE` is defined for sm_75+,
  and `mul_mat_q` already uses `mma_int_A_I16K4` → `mma_K4` →
  `mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32` (Turing's int8 TC
  intrinsic). The kernel is on tensor cores. ncu confirms: 0 register
  spills, IMMA fragments dispatching as expected.
- **The actual bottleneck (ncu, prod-np2-ctx256k):**
  - Grid size **4 CTAs** at decode shape (40 CTAs at bigger), **0.56
    full waves across all SMs**. Most of TU102's 72 SMs idle.
  - Theoretical occupancy **25%** capped by register + shared-memory
    limits. `__launch_bounds__(WARP_SIZE*nwarps, 1)` (line 4187) sets
    `minBlocksPerSM=1` on Volta+, forcing 1 block/SM ⇒ 8 warps/SM ⇒
    25% of TU102's 32-warps/SM peak.
  - DRAM throughput **4.95%**, SM Compute Throughput **2.23%** —
    nowhere near any peak. Kernel is **latency-bound on grid + warp
    under-utilization**, not bandwidth or TC compute.
  - Effective compute: 0.56 waves × 0.25 occupancy ≈ **14% of peak**.
- **Why grid is small.** Stream-K decomposition is **already
  implemented** in `mmq.cuh` (line 4178) but **disabled by env-gate**
  (line 4394): `GGML_CUDA_MMQ_ENABLE_STREAM_K` defaults off because
  the stream-K fixup reduction order is M-dependent, which would
  reproduce the same NP-cluster-partition NPC failure mode we just
  fixed for cuBLAS (2026-05-19 ALGO0 audit). Without stream-K the
  kernel uses xy-tiling with `block_nums(nty, ntx)` where `nty=ne01/mmq_y`
  and `ntx=ne11/mmq_x`. At decode `ne11=1` ⇒ ntx=1; nty ≈ 40 in the
  bigger case, ⇒ 40 CTAs total, ⇒ 0.56 waves on 72 SMs.

#### Optimization vectors (both compatible)

**Lever A — `launch_bounds(*, 1)` → `(*, 2)` on Turing.** Low-risk,
small change. Lets the compiler target 2 blocks/SM ⇒ 50% theoretical
occupancy (vs 25% today). `feedback_launch_bounds_non_monotonic`
binds: bumping `minBlocksPerSM` may force register spills if the
compiler can't keep working state in the smaller reg budget. Today's
ncu shows **0 spills at `, 1)`**, so the compiler is using max regs.
At `, 2)`, expect one of:
  - Spill-free re-ILP'd kernel → ~1.3–1.8× speedup
  - Spilling kernel → regression, revert
Gate: ncu reg/thread + spill count + perf. Cheap measurement, ~1 hr to
prototype + test both Turing settings.

**Lever B — NPC-safe stream-K fixup.** Higher leverage, more design.
Stream-K splits K across `nsm` (=72) CTAs unconditionally, then a
fixup kernel reduces partial results. The current fixup's reduction
order depends on the actual tile count (which depends on M=ne11) —
that's the NPC-breaking dispatcher branch. Fix options:
  - Pad partial-tile count to NSM regardless of M, so the reduction
    order is M-independent. Pad-tiles contribute 0 — costs one extra
    smem buffer but unlocks full-grid stream-K.
  - Or: canonical reduction tree (associative within partials, no
    dependence on tile count).
Expected: 72-CTA grid → ~3–5× speedup on the decode shapes. Effort
higher; needs spec amendment + verify-production-determinism.sh GREEN
across NP={1,2,4,8} after the fixup change.

**Combined ceiling.** A + B → 4–9× per-kernel speedup ceiling. Real
gain bounded by other kernels (NCCL #2 catches up; eventually a
different kernel becomes top).

#### Spec template (per `feedback_kernel_replacements_must_be_sota_sm75`)

A kernel rewrite spec lands before code:
  - Register budget per thread (target ≤ 64 to allow 2 blocks/SM on
    Turing).
  - Shared memory per block (target ≤ 16 KB to allow 2 blocks/SM on
    Turing — TU102 has 64 KB smem/SM).
  - Occupancy target: 50% theoretical (2 blocks/SM × 8 warps/block =
    16 warps/SM / 32 warps/SM peak).
  - Grid target: ≥ NSM = 72 CTAs per launch (post-stream-K).
  - % of TU102 peak: project against 65 TOPS int8 TC × occupancy ×
    waves.
  - NPC contract: K-iteration order fixed; if stream-K, fixup reduction
    order canonical and M-independent.
  - Committed ncu before/after on the prod-np2-ctx256k shape with
    DRAM/SM throughput, occupancy, regs/thread, spill count.

- **Bundles with target #5 (`Q4_0_AR16`)** since AR16 packing only
  changes the dequant inner loop — both go through the same MMQ
  kernel infrastructure.

### #2 — `NCCL_AllReduce_f32` → deterministic f16 reduce
- **Aggregate cost.** 34.0 s NP=8 decode (20.4%) + 2.68 s NP=1 prefill
  (**36.3% — biggest single line at NP=1**).
- **TU102 hook.** Quadro RTX 6000 has NVLink (50 GB/s/dir aggregate per
  pair, ~100 GB/s bidirectional). The f32 ring-LL all-reduce is moving
  twice the data versus f16. Pre-NPC at NP=1 had a mixed `f16+f32` path
  (1.84 s combined) vs HEAD's f32-only 2.68 s.
- **NPC contract.** NCCL ring-LL chunk order is fully deterministic per
  call when `NCCL_NCHANNELS`/`NCCL_PROTO` are pinned. f16 sum is not
  associative, but with fixed order it's deterministic. Validation gate:
  the existing `scripts/verify-production-determinism.sh` matrix.
- **Risk.** f16 underflow on MoE expert weight reductions — needs a
  rigorous PPL sweep before merge. Mitigation: f16 reduce only for
  activation paths, keep f32 for expert-routing logits.
- **Effort.** **Medium.** Mostly NCCL env / runtime config + bind point
  change in `ggml-cuda.cu` reduce dispatch. No new kernel.

### #3 — `cuBLAS_gemvx_bf16` lm_head → F16 + cuBLAS algo pin

**Status (2026-05-19): CLOSED via cuBLAS `ALGO0_TENSOR_OP` pin, not via
pinned-HMMA dispatch. F16 GGUF switch is in. Measured: TG NP=8 21.41 →
25.42 t/s (+18.7%); NPC FAIL → PASS.**

The originally projected path (F16 lm_head + pinned-HMMA batched
dispatch) turned out to be slower at the typical F16 HMMA shape mix —
pinned-HMMA at M=1 wins on lm_head but loses 5.7× on 408k *other* F16
GEMMs (small post-norm projections, MoE expert outputs). Global pinned
dispatch cost -21% TG vs default cuBLAS HGEMM.

The actual fix turned out to be simpler and orthogonal: pin the cuBLAS
gemm algorithm to `CUBLAS_GEMM_ALGO0_TENSOR_OP` unconditionally
(removing the `LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH` dependency at both
algo-selection sites in `ggml-cuda.cu`). This:
1. Closes the NP-cluster-partition NPC failure that the audit found in
   production runtime (NP=2 diverged with F16 lm_head, NP=4 with BF16).
2. Adds +18.7% TG at NP=8 because the default-adaptive algo had been
   falling back to non-TC at small M; ALGO0 engages tensor cores
   unconditionally.

Verified at the actual production runtime config (env=0): NPC GREEN
across full NP={1,2,4,8} slot byte-identity + cross-NP slot-0 matrix.

- **Aggregate cost (BF16 baseline).** 8.8 s NP=8 decode (5.3%) + 271 ms
  NP=1 prefill (3.7%). Per-call **4174 µs** at NP=8 — by far the most
  expensive single call on the system.
- **TU102 hook.** sm_75 has **no BF16 tensor cores**. cuBLAS BF16 gemvx
  runs on fp32 CUDA cores (16.3 TFLOPS) vs fp16 TC (130 TFLOPS). With
  F16 lm_head, the `bf16_native_eligible` carve-out at `src1_ncols == 1`
  in `ggml_cuda_op_mul_mat_cublas` (ggml-cuda.cu:1733) is bypassed
  entirely; dispatch routes to the F16 HMMA path at line 1818+ (cuBLAS
  HGEMM CUBLAS_R_16F).
- **GGUF switch (landed 2026-05-19).** Production paths now point at
  `qwen3.6-27b-V-F1.T1.lm_head-f16.gguf` (recast from
  `qq-tool1lossless-vocab-fix.gguf`; only `output.weight` BF16 → F16,
  all 865 other tensors byte-identical). Files updated:
  `profiles/qwen36-27b-x{1,3,8}*.sh`,
  `scripts/test-production-np-determinism.sh`.
- **Dispatch-side (pending).** The cuBLAS HGEMM F16 path is the
  first-order win; the pinned-HMMA path
  (`ggml_cuda_mul_mat_f16_pinned`, NPC-by-construction across batch
  composition) is the stretch target if cuBLAS HGEMM has residual
  batch-shape sensitivity at the lm_head shape (V=248320, M=1 per slot).
  Currently the pinned dispatch is gated on
  `s_force_sid_cublas` (env `LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH=1`)
  at ggml-cuda.cu:1902 — needs to be baked default-on if the reprofile
  shows cuBLAS HGEMM is not landing the win cleanly.
- **NPC contract.** Identical to DFlash's pinned-HMMA path
  (NPC-by-construction). Already proven across 5 gates in
  `PHASE_DFLASH_BATCHED_PINNED.md`.
- **Projected yield.** 2105 calls × ~3500 µs saving each = ~7 s
  recoverable at NP=8 (assumes lm_head goes from 4174 → ~700 µs/call at
  the pinned batched shape). To be measured.
- **Lowest-effort high-yield target on the list.**

---

## Top 5 — add

### #4 — `fused_mmvq[Q4_0, nc=1]` (F.4.1') dispatch split
- **Aggregate cost.** 17.5 s NP=8 (10.5%) + 704 ms NP=1 (9.6%).
- **TU102 hook.** F.4.1' is already best-in-class under the NPC contract
  (86 µs/call unchanged from pre-NPC). The route here is **not** "rewrite
  the kernel" but **"split the dispatch"**: this kernel handles both
  single-token decode rows AND MoE expert-routing fan-out. The expert
  fan-out path can run on a looser-NPC kernel using DP4A int8 dot with
  s32 accumulate, recovering ~50% on the MoE-routed portion without
  touching the activation-path kernel.
- **NPC contract.** Per-target-kernel; the decode-row path stays on the
  current F.4.1' kernel byte-identically.
- **Effort.** **Medium.** Dispatch site triage + new kernel TU for the
  MoE-expert path.

### #5 — `mul_mat_q[Q4_0_AR16, ncols=8]`
- **Aggregate cost.** 9.1 s NP=8 (5.4%) + 272 ms NP=1 (3.7%).
- **TU102 hook.** Falls out of target #1's int8 IMMA dispatch. AR16
  (16-element 4-bit AutoRound-aware blocks, `GGML_TYPE_Q4_0_AR16 = 159`)
  is structurally identical to Q4_0 with a different dequant inner-loop
  block size. Bundle as a single workstream with #1.

---

## Top 10 — add

### #6 — `mul_mat_q[Q4_0, ncols=112]` prefill big-tile param tuning
- **Aggregate cost.** 420 ms NP=1 prefill (5.7%), 608 µs/call, 690 calls.
- **TU102 hook.** Already uses `cutlass_75_wmma_h161616` under the hood.
  The cutlass tile is 16x16x128 stage-2 transpose-normal align-8 — may
  not be optimal for Qwen 3.6's 5120-hidden shapes. Try 64x64x32 and
  32x32x64 variants; ncu to confirm SM occupancy & shared-memory
  pressure. **Lower leverage** — pre-NPC didn't beat this much.
- **Effort.** **Low.** cutlass template parameter sweep.

### #7 — `fused_rms_norm` fold into prior matmul epilogue
- **Aggregate cost.** 1.82 s NP=8, 468 026 calls, **3.9 µs/call ≈ pure
  launch overhead**.
- **TU102 hook.** Latency-bound, not bandwidth-bound. Fold RMSNorm into
  the source matmul tile's epilogue (compute `out * rsqrt(sum_sq/n + eps)
  * weight[d]` directly in the writeback). cutlass-style epilogue. Saves
  ~468 k kernel launches × ~3 µs launch tax = ~1.4 s recoverable at NP=8.
- **NPC contract.** Reduction tree is per-row, single-CTA — byte-identical
  by construction.
- **Effort.** **Medium-high.** Requires epilogue extension in the MMQ
  kernel; downstream kernels (the F.4.1' kernel, PSKV-FA) also consume
  RMSNorm output, so the fusion must respect dispatch composition.

### #8 — `cpy_flt[fp32→fp32]` fusion
- **Aggregate cost.** 2.75 s NP=8 (1.6%), 508 306 calls, 5.4 µs/call.
- **TU102 hook.** Same launch-overhead story as #7. These copies are
  mostly KV-cache row stages and elementwise op intermediates. Identify
  dispatch sites in `ggml-cuda.cu` — if they're staging before/after
  `mul_mat`, fold into source/dest tensor reshape. Estimated
  **1.5–2 s recoverable** at NP=8.
- **Effort.** **Medium.** Dispatch-site triage + per-site fusion.

### #9 — `delta_net_recurrent_f32` HMMA inner-state matmul
- **Aggregate cost.** 1.26 s NP=8 (0.8%) + 73 ms NP=1 (1.0%), 202 827
  calls, 6.2 µs/call.
- **TU102 hook.** Qwen 3.6 27B's hybrid layers run scalar fp32 recurrent
  state updates. Inner state-update matmul is 128×256 — fits fp16 HMMA
  m16n8k8 fragment. Estimated ~50% per-call recovery.
- **NPC contract.** Per-row, single-CTA — byte-identical by construction.
- **Effort.** **Medium.** New kernel template; the existing scalar f32
  kernel is the reference.

### #10 — `quantize_mmq_q8_1` + `convert_unary` elision
- **Aggregate cost.** `quantize_mmq_q8_1`: 1.07 s NP=8 (0.6%), 720 642
  calls. `convert_unary<fp16↔fp32>` pairs: ~1.05 s combined NP=8
  (~408 k each direction). Combined ~2.1 s recoverable.
- **TU102 hook.** Not architectural — purely op-graph cleanup. Fuse the
  q8_1 quantize with upstream RMSNorm/RoPE output (write q8_1 directly).
  Eliminate `convert_unary` pairs that bracket cutlass tile paths by
  promoting matched-dtype neighbours.
- **Effort.** **Medium.** Cross-kernel dispatch surgery.

---

## Notes

- **PSKV singlewarp FA (table row #3, 18.4% NP=8)** is scoped as an
  existing workstream per the 2026-05-17 ralph-loop spec (Variant D WMMA
  TC path with 25× per-call headroom). It is **not** repeated as a new
  specialization target here. If that workstream lands, the dominant NP=8
  kernel falls from PSKV-FA to MMQ Q4_0 (#1 here).
- **Decode vs prefill phase split.** Top-3 ranking is robust across both
  phases:
  - NP=1 prefill-heavy: NCCL 36.3% + MMQ 23.8% = **60.1% combined** (top 2).
  - NP=8 decode-dominant: MMQ 30.4% + NCCL 20.4% = **50.8% combined**.
  The lm_head gemvx is more punishing at decode (per-slot loop fires
  every token); prefill amortizes it.
- **Effort / yield tiers** (independent of rank):
  - **Lowest effort:** #3 lm_head F16 + pinned (recast tooling exists).
  - **Highest yield, highest effort:** #1 + #5 MMQ int8 IMMA bundle.
  - **Highest yield NCCL:** #2 if f16 reduce path stays NPC.

---

## Verification plan (per target, when picked up)

For each target before kernel-design lock:

1. **ncu pass** on the target kernel at the same bench shape:
   ```
   ncu --set full --target-processes all --launch-skip 5000 --launch-count 200 \
       --kernel-name regex:<kernel> --export <target>.ncu-rep <bench-cmd>
   ```
   Capture: registers/thread, SM occupancy, achieved fp16/int8 TC %, DRAM
   throughput, L2 hit rate, IPC. Confirm latency-bound vs bandwidth-bound
   diagnosis matches the proposed TU102 hook.
2. **Kernel-design lock** as a spec amendment (this repo's
   `specs/dflash/kernel-design.md` is the pattern reference; non-DFlash
   kernels need a sibling spec).
3. **Three gates per landed change.**
   - Build: `cmake --build build -j 32` zero warnings on sm_75.
   - NPC: `bash scripts/verify-production-determinism.sh` GREEN across
     NP={1,2,4,8} multi-GPU slot-0.
   - Perf: `llama-batched-bench` with the four-trace shape above; capture
     new kernel breakdown and confirm ≥ 80% of projected recovery.
4. **No env-gates** per `feedback_bake_measurement_env_gates` — bake the
   verified behaviour or revert.
5. **No phase nomenclature in code** per `feedback_no_host_concerns_in_code`
   — kernel names use content-descriptive form
   (`mul_mat_q_imma_q4_0_*`, not `phase_tu102_*`).

---

## Out of scope

- DFlash speculative decoding — closed in `PHASE_DFLASH_BATCHED_PINNED.md`.
- PSKV singlewarp FA — separately scoped ralph-loop workstream.
- Single-GPU profile (this is a multi-GPU production workload).
- Hadamard rotation paths — bench did not exercise them (llama-batched-bench
  flag mismatch). Hadamard is on at production runtime; its per-call cost
  appears under `cutlass_75_wmma_h161616` and adjacent rotation kernels.
  Sub-1% in the breakdown above; not a primary target.
- vLLM porting / multi-engine — different workstream.

---

## Open items

| Item | What |
|---|---|
| ncu top-3 | Per-kernel arch metrics to lock the speedup vector for each of MMQ, NCCL, lm_head |
| Hadamard-on profile | Re-run the same shape with Hadamard enabled (via `llama-server`, not `llama-batched-bench`) to confirm sub-1% claim |
| Long-prompt prefill | npp=2048 capture for the prefill big-tile (#6) ceiling — current data is npp=200 only |
| MoE dispatch split for #4 | Identify dispatch sites where `fused_mmvq[Q4_0,nc=1]` is called for expert routing vs decode rows |
