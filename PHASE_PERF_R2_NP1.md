# PHASE_PERF_R2_NP1 — kernel-cost re-rank at np=1 × 256k vanilla shape

**Opened:** 2026-05-24.
**Branch:** `production/2026-q2-next` (parent: main).
**Scope:** Re-rank kernel-level hotspots at the **newly-deployed production
shape** (np=1 × 256k, no DFlash, Hadamard ON, q4_0 KV, sm=graph) and decide
which optimization levers to pull next. Successor to
`PHASE_TU102_SPECIALIZATION.md` (which ranked at the np=2 × 256k + DFlash +
Hadamard shape that production no longer runs).
**State:** Re-rank measured and recorded. No kernel work started — this
file is the planning artifact; downstream lever work lands as its own
PHASE_ doc.

---

## Why this re-rank exists

Two facts forced it:

1. **Production shape changed 2026-05-24.** Production migrated from
   `qwen36-27b-x2-dflash.sh` (np=2 × 256k, DFlash drafter, Hadamard ON) to
   `qwen36-27b-x1-vanilla.sh` (np=1 × 256k, **no DFlash**, Hadamard ON).
   The np=2 + DFlash + multi-slot dispatch configuration was the one that
   `PHASE_TU102_SPECIALIZATION.md` ranked against on 2026-05-19; that
   ranking no longer reflects what production runs.

2. **TU102_SPEC's top-2 targets shipped.** Target #1 (MMQ Q4_0 → split-K
   + Lever D I=8 fragment rewrite) shipped 2026-05-19 at submodule
   `f8fa3928`. Target #3 (cuBLAS lm_head F16 + ALGO0_TENSOR_OP pin)
   shipped 2026-05-19. Those wins are baked into the binary used for
   this re-rank — the residual kernel mix is what's left to attack.

The fresh nsys capture lives at
`/tmp/nsys-nvlink/nvlink-vanilla-np1-2026-05-24.nsys-rep` (will be
relocated to `data/nsys-perf-2026-05-24-vanilla/` before close).

---

## Capture methodology

Matches `PHASE_TU102_SPECIALIZATION.md` discipline so the diff is
apples-to-apples on bench command, deviating only on shape.

| | TU102_SPEC 2026-05-19 | This re-rank 2026-05-24 |
|---|---|---|
| Binary | `production/2026-q2-next @ bb5c37eb` | `production/2026-q2-next @ 711212a6` (+ split-K + Lever D + ALGO0 pin baked) |
| Model | `qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf` (BF16 lm_head) | `qwen3.6-27b-V-F1.T1.lm_head-f16.gguf` (F16 lm_head per ALGO0 close) |
| Bench tool | `llama-batched-bench` | `llama-batched-bench` (identical) |
| Shape flags | `-npp 200 -ntg 64 -npl {1\|2\|8}`, `-c 4096`, `-fa on`, `-ctk q4_0 -ctv q4_0` | `-npp 200 -ntg 64 -npl 1`, `-c 4096`, `-fa on`, `-ctk q4_0 -ctv q4_0`, **`-khad -vhad`** |
| Hadamard | Off (claimed flag-parse mismatch — *falsified* 2026-05-24: bench accepts `-khad -vhad`; the TU102 note is stale) | **On** — matches production runtime |
| Clocks | Locked 1455 MHz | Default clocks (xeon doesn't enforce locked-clocks discipline yet; flagged as a follow-up) |
| Tracer | nsys `-t cuda,nvtx,osrt` (no gpu-metrics — `ERR_NVGPUCTRPERM` on xeon) | identical |

Bench result this run (the headline numbers the kernel ranking explains):

| | np=1 × 4k Hadamard ON (this run, vanilla) | np=1 × 256k Hadamard ON (TU102_SPEC, F.4.1' baked, no split-K) | np=2 × 256k Hadamard ON (TU102_SPEC, split-K + Lever D shipped) |
|---|---:|---:|---:|
| Wall | 3.69 s | 5.10 s | n/a (multi-slot wall not directly comparable) |
| PP t/s | **181.14** | 99.28 | 25.71 |
| TG t/s | **24.77** | 20.74 | 23.57 |
| Aggregate t/s | **71.59** | 51.76 | n/a |

TG at the new shape (24.77) is **+19% over the TU102_SPEC np=1 reference
(20.74)**. That's the split-K + Lever D + ALGO0 + F.4.1' + (np reduction
from 2→1) bundle landing visibly. The remaining headroom is what this
phase plans against.

---

## Refreshed kernel breakdown — np=1 × 4k Hadamard ON, vanilla

Top kernels by GPU time (3.69s wall, ~7s aggregate across 2 GPUs):

| # | Kernel | Time | % | Per-call | Calls | Note |
|---|---|---:|---:|---:|---:|---|
| 1 | `mul_mat_q_split_k[Q4_0, ncols=8]` | 1.36 s | **29.9%** | 47.8 µs | 28 382 | post-Lever-D rewrite; std 21 µs (workload variance) |
| 2 | `fused_mmvq[Q4_0, nc=1, F.4.1']` | 714 ms | **15.7%** | 87.1 µs | 8 194 | std 0.5 µs (tight); F.4.1' kernel for ncols≤8 path |
| 3 | `cutlass_75_wmma_h161616gemm` | 446 ms | **9.8%** | 8.3 µs | 53 621 | F16 GEMM body, post-ALGO0 |
| 4 | `NCCL_AllReduce_Sum_f32_RING_LL` | 413 ms | **9.1%** | 24.4 µs | 16 892 | cross-GPU collective, RING_LL is NVLink-optimal already |
| 5 | `mul_mat_q_split_k[Q4_0, ncols=112]` (prefill big tile) | 333 ms | **7.3%** | 482.5 µs | 690 | grew vs np=2 because -npp 200 is bigger share of np=1 wall |
| 6 | `mul_mat_q_split_k[Q4_0_AR16(159), ncols=8]` | 222 ms | **4.9%** | 36.1 µs | 6 138 | type 159 = AR16 recast format, MMQ_Q4_0_AR16 Phase A/B/C |
| 7 | `flash_attn_per_slot_kv_singlewarp[256,256,Q4_0,Q4_0]` | 214 ms | **4.7%** | 105.2 µs | 2 032 | PSKV-singlewarp-FA (separately scoped per ralph-loop spec) |
| 8 | `cpy_flt[fp32→fp32]` | 86 ms | 1.9% | 5.1 µs | 16 712 | launch-overhead-bound staging copies (TU102_SPEC #8) |
| 9 | `cublasLt::splitKreduce_kernel` | 77 ms | 1.7% | 1.4 µs | 53 556 | paired with cutlass at row #3 — identical count |
| 10 | `delta_net_recurrent_f32` | 75 ms | 1.7% | 11.9 µs | 6 324 | Mamba2 SSM recurrent state (TU102_SPEC #9) |
| 11 | `fused_rms_norm_f32<1024>` | 65 ms | 1.4% | 3.85 µs | 16 861 | latency-bound; epilogue-fusion candidate (TU102_SPEC #7) |
| 12 | `convert_unary<f32→f16>` | 64 ms | 1.4% | 1.20 µs | 53 621 | pair-#1 of the cutlass cast trio |
| 13 | `mul_mat_q_split_k[Q4_0_AR16(159), ncols=112]` | 60 ms | 1.3% | 622 µs | 96 | AR16 prefill big tile |
| 14 | `convert_unary<f16→f32>` | 57 ms | 1.2% | 1.06 µs | 53 621 | pair-#2 of the cutlass cast trio |
| 15 | `mul_mat_q_split_k_fixup[8,8]` | 55 ms | 1.2% | 1.6 µs | 34 520 | split-K tail reduction (post-Lever D) |
| | **Top 15 covers** | **3.94 s** | **86.9%** | | | |
| | `hadamard_f32<256>` (the sub-1% claim test) | 12.9 ms | **0.3%** | 1.59 µs | 8 128 | TU102_SPEC sub-1% claim holds at np=1 too |

Total accounted GPU time across all kernels above (and remaining tail):
~4.54 s — sanity-aligned with 3.69 s wall × ~2 GPUs.

---

## Diff against TU102_SPEC ranking

| Kernel | TU102 np=2 × 256k % | This (np=1 × 4k vanilla) % | Shift | Interpretation |
|---|---:|---:|---:|---|
| MMQ Q4_0 ncols=8 | 38.8% | **29.9%** | −8.9 pp | Lever D already cut absolute time +20% TG; remaining share drops as np=1 reduces decode-row pressure |
| NCCL AllReduce | 21.0% | **9.1%** | **−11.9 pp** | np=1 has less cross-GPU collective traffic per generated token. AsyncReduce Option B ceiling halved |
| fused_mmvq F.4.1' nc=1 | 10.2% | **15.7%** | **+5.5 pp** | At np=1, this kernel handles a larger relative share of decode-row + MoE-expert paths |
| cutlass_75_wmma | 6.2% | **9.8%** | +3.6 pp | F16 GEMM body cost amortizes less at smaller batch |
| Q4_0_AR16 ncols=8 | 6.2% | **4.9%** | −1.3 pp | ~stable; the AR16 path is balanced with Q4_0 |
| PSKV-singlewarp-FA | 5.4% | **4.7%** | −0.7 pp | ~stable |
| MMQ Q4_0 ncols=112 (prefill) | not listed | **7.3%** | NEW | Prefill big tile climbs as -npp 200 is a bigger share of np=1 wall |
| delta_net_recurrent_f32 | 0.8% | **1.7%** | +0.9 pp | Mamba2 SSM cost amortizes less at np=1 |
| convert_unary pair | not separately ranked | **2.6% combined** | NEW visibility | The cutlass cast trio (rows 3 + 9 + 12 + 14) — same trio TU102_SPEC #10 flagged |
| `hadamard_f32` | claimed sub-1% (no measurement) | **0.3%** measured | confirmed | TU102 claim verified at np=1; Hadamard not a primary target |

**Headline reframing**: the dominant kernel at the new shape is still MMQ
Q4_0 ncols=8 (29.9%) and that kernel was already best-in-class'd by Lever
D. The **second-biggest single line is now fused_mmvq F.4.1' (15.7%)** —
not NCCL (which roughly halved going np=2 → np=1) and not lm_head (which
moved into the cutlass row at ALGO0 close). F.4.1' becoming the dominant
**addressable** target is the headline shift.

---

## Optimization candidates ranked at the new shape

### Candidate A — fused_mmvq F.4.1' kernel-level investigation

**FRAMING CORRECTED 2026-05-24 — see "Candidate A: premise falsification"
section below.** Originally framed as "dispatch split" (decode rows vs
MoE expert routing), inheriting TU102_SPEC #4. That framing is wrong for
the qwen35 architecture at this production shape. Reframed as a
kernel-level investigation.

**Cost (unchanged).** 714 ms (15.7%), 8 194 calls @ 87.1 µs (tight,
std 0.5 µs).

**What the kernel actually is.** Per code-reading at
`ik_llama.cpp/src/llama-build-context.cpp:1323` and the dispatcher at
`ik_llama.cpp/ggml/src/ggml-cuda/mmvq-templates.cuh:356`, every one of
the 8 194 calls comes from a single source: the **standard FFN block's
`ggml_fused_up_gate(up, gate, cur, unary_op)` invocation**, fired once
per FFN per token. With 65 layers (qwen35) × ~125 token-equivalents per
bench (200 prefill + 64 generate, decode-shape pressure dominating
ncols_y=1 path) the count matches.

**There is no dispatch split.** The kernel template
`fused_mul_mat_vec_q<...>` fires ONLY when `args.vx_u && args.vx_g`
(both up and gate weight pointers set), which is uniquely satisfied by
`ggml_cuda_op_fused_mul_mat_vec_q_id`. The non-fused projection path
(Q/K/V/O attention) fires the sibling `mul_mat_vec_q<...>` template — a
**different kernel by name**, separate row in nsys (not in top 15).
MoE expert routing would fire MUL_MAT_ID, which also routes through the
non-fused MMVQ template, not this one. Hence there are no "decode rows
vs MoE" sub-populations to split.

**Possible levers (require measurement before committing).**

1. **Kernel rewrite Lever-D-style.** The kernel uses
   `__launch_bounds__(nwarps*WARP_SIZE, 1)` at `mmvq-templates.cuh:281,304`
   — `minBlocksPerSM=1` caps occupancy at 25% on Turing, same constraint
   that drove the original Lever D rewrite of `mul_mat_q_split_k`. The
   MMVQ kernel uses fewer registers and less shmem per CTA than MMQ
   (smaller M=1 path), so the shmem ceiling may not be the binding
   constraint that blocked Lever C for MMQ — there may be room to bump
   `minBlocksPerSM` to 2+ on sm_75.

2. **NPC-safe stream-K analog.** The kernel grid is (nrows_x/rpcb_fused,
   ne2, 1). For ncols_y=1, rpcb=1 → nrows_x blocks on Y axis × 8 experts
   (ne2). At hidden_dim=5120 nrows_x is in the thousands, so grid
   already saturates the 72 SMs without stream-K. Not a likely lever
   for this kernel.

3. **FFN fusion.** Fold `ffn_norm` (preceding RMSNorm) into the
   fused_up_gate kernel epilogue — equivalent to TU102_SPEC #7 but at
   the input side. Saves ~16 861 × 3.85 µs ≈ 65 ms (1.4%) launch tax;
   but the modification has to respect the cross-device REDUCE the
   norm participates in. Higher effort.

4. **AR16 expansion.** Q4_0_AR16 (type 159) is 4.9% of kernel time
   currently — it's the recast format that some layers use. If the
   AR16 variant of `fused_mul_mat_vec_q` is faster per byte (the AR16
   MMQ phase shipped per `PHASE_MMQ_Q4_0_AR16.md`), and more layers
   could be moved to AR16 weight type, total time on this kernel
   family might drop. Requires per-tensor recast tooling — orthogonal
   workstream.

**Verification (ncu before any kernel-level commit).**
```
ncu --set full --target-processes all \
    --launch-skip 5000 --launch-count 200 \
    --kernel-name regex:fused_mul_mat_vec_q \
    --export fused_mmvq_baseline.ncu-rep \
    /opt/llm-server/bin/llama-batched-bench \
        -m /opt/models/recast-out/qwen3.6-27b-V-F1.T1.lm_head-f16.gguf \
        --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
        -ngl 999 -fa on -c 4096 -ctk q4_0 -ctv q4_0 -khad -vhad \
        -b 2048 -ub 512 --threads 16 -npp 200 -ntg 64 -npl 1
```

Capture: registers/thread, SM occupancy theoretical and achieved,
DRAM throughput %, achieved DP4A %, shmem/CTA, IPC. The diagnostic
answers "is this kernel hardware-limited at 87 µs or is there
headroom" — same shape of question Lever C asked about MMQ that
revealed shmem as the binding constraint.

**Effort.** Diagnostic ncu pass: ~20-30k tokens (~30 min wall on
xeon). Kernel rewrite (if ncu shows headroom): ~80-150k tokens.
Total to a binding perf-delta gate: ~100-200k tokens.

**Per CLAUDE.md §8 (measured-mode-on-diagnosis).** No kernel rewrite
without ncu evidence first. The MoE-dispatch-split premise was the
exact failure mode §8 warns about ("speculation is cheap until you
find out it's wrong, and the cost-to-undo is 50k+ per false-claim
cascade") — this re-frame is the cheap end of that lesson.

### Candidate B — AsyncReduce Option B (already specced)

**Cost.** 413 ms (9.1%), 16 892 calls @ 24.4 µs.

**Lever per `PHASE_ASYNC_REDUCE.md`.** Replace synchronous F32
cross-device reduce with async F32 reduce on dedicated comm stream +
event-based consumer wait. Allium spec drafted (12 invariants), TLA+
spec drafted (safety + liveness + deadlock-freedom), DESIGN.md
projects transfer fits in <2% of compute window — should hide entirely.

**Recover target.** ~80-90% of the 413 ms if overlap actually works at
TU102 (Open Question #2 in PHASE_ASYNC_REDUCE). Ceiling: **~8% wall**.
Less than Candidate A's ceiling? Wait —

| Lever | Reclaim ceiling (kernel time) | Reclaim ceiling (wall) |
|---|---:|---:|
| A (F.4.1' dispatch split) | ~7-8% of kernel time | ~3% wall |
| B (AsyncReduce overlap) | ~9% of kernel time | ~4-5% wall |

B has a higher wall-time ceiling at this shape **and** has its spec
already drafted (Allium + TLA+) per the prior phase's discipline. It
also de-risks faster — the TLC verification is a one-day exercise from
the existing spec.

**NPC contract.** Already byte-identical via the existing Option A sync
F32 reduce; B preserves the same arithmetic (F32 sum, same order) and
just relocates it to a comm stream. The TLA+ `SafetyConsumeAfterCompute`
property is the formal guarantee.

**Effort.** Medium — but front-loaded. T1 (spec lockdown + TLC run) is
~20-30k tokens. T3 CUDA backend implementation is ~30-50k. T4-T6 wiring +
nsys overlap verification is ~30-40k. Total ~80-120k tokens.

**Per CLAUDE.md §8 (bold on design, measured on diagnosis).** Open
Question #2 — "Does `cudaMemcpyPeerAsync` actually overlap with kernel
launches on TU102?" — must be answered empirically *before* T3. A 2-hour
microbench (memcpy_peer_async on stream A while compute kernel runs on
stream B, nsys trace, check timeline overlap) decides whether B is
viable. If no overlap on TU102, B degrades to Option A bandwidth cost
(still byte-deterministic, still ships, just no perf win) — known
failure mode, scoped in PHASE_ASYNC_REDUCE Risks table.

### Candidate C — MMQ Q4_0 ncols=8 dispatch shape audit

**Cost.** 1.36 s (29.9%), 28 382 calls @ 47.8 µs, **std 21.7 µs on
47.8 µs mean** — the variance signal.

**Lever.** Kernel is already best-in-class'd (Lever D, shipped). The
variance suggests per-call workload differs — different M sizes hit
suboptimal tiles. Either fix dispatcher to pick per-shape, or accept the
variance as workload-intrinsic.

**Per CLAUDE.md §8**: ncu pass first. Instrument the dispatcher to log
(M, N, K) per call; histogram offline. If unimodal, the kernel really is
near-optimal — no easy win, close with evidence. If bimodal, ship a
shape-aware tile dispatcher. **Decision branch, not a commit-to-work.**

**Ceiling if bimodal-fix succeeds.** Var → ~5 µs std would mean
~15-20 µs/call recovered on the high-tail calls. If high-tail is ~⅓ of
calls, ~10 000 × 17 µs = 170 ms ≈ **3.8% wall**.

**Effort.** Low for the diagnostic pass (~10-15k tokens). High if it
turns out a shape-aware dispatcher is needed (~50-80k).

### Candidate D — convert_unary cast trio elision (refined 2026-05-24)

**Cost (the full HGEMM-with-cast trio).** 446 ms cutlass + 78 ms
splitKreduce + 64 ms convert_unary<f32,half> + 57 ms convert_unary<half,f32>
= **645 ms (14.2%)** combined. Identical instance counts (53 621 each
for cutlass/casts; 53 556 splitKreduce — paired one-to-one).

**Call site identified.** `ik_llama.cpp/ggml/src/ggml-cuda.cu:1901-1955` —
the `ggml_cuda_op_mul_mat_cublas` F16 HGEMM path:

```cpp
// line 1901-1907 — entry cast: src1 (activations) F32 → F16
if (src1->type != GGML_TYPE_F16) {
    const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src1->type);
    to_fp16_cuda(src1_ddf_i, src1_as_f16.get(), src1_ncols, ne10, stream);
}

// line 1944-1951 — cuBLAS HGEMM (the cutlass_75_wmma kernel + splitKreduce)
cublasGemmEx(..., CUDA_R_16F src0, CUDA_R_16F src1, CUDA_R_16F dst,
             CUBLAS_COMPUTE_16F, s_cublas_algo /*ALGO0_TENSOR_OP*/);

// line 1953-1954 — exit cast: dst F16 → F32
const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_F16);
to_fp32_cuda(dst_f16.get(), dst_dd_i, row_diff, src1_ncols, stream);
```

**Trio anatomy (per-call attribution).** At avg ~8.3 µs cutlass +
~1.5 µs splitKreduce + ~1.2 µs entry-cast + ~1.1 µs exit-cast =
**~12 µs/call**. Cast pair is **~19% of per-call time**; HGEMM body is
the other ~81%. The casts are NOT the dominant cost per call — but
their *aggregate* is meaningful because the count is 53k.

**The src0 weight cast does NOT contribute.** Per the comment block at
`ggml-cuda.cu:1842`, `src0` (weight) goes through a persistent
`fp16_cache` — one-shot conversion per unique weight pointer, then
cached forever (or until OOM fallback). Confirmed by the convert_unary
instance counts: 53 621 per direction matches `src1_ncols` calls + dst
writebacks exactly. **All cast time is on src1 (activations) and dst.**

#### Sub-lever D1 — exit cast elision via CUDA_R_32F output (low-risk)

cuBLAS `cublasGemmEx` supports `CUDA_R_16F` inputs + `CUBLAS_COMPUTE_16F`
+ **`CUDA_R_32F` output** as a single supported combination on Volta+.
Tensor-core compute path is unchanged (still ALGO0_TENSOR_OP, still
HMMA); only the writeback dtype differs.

**Surgical change.** ~5-line diff at `ggml-cuda.cu:1938-1954`:

```cpp
// before:
ggml_cuda_pool_alloc<half> dst_f16(ctx.pool(id), row_diff*src1_ncols);
const half alpha_f16 = 1.0f, beta_f16 = 0.0f;
cublasGemmEx(..., CUDA_R_16F, ..., dst_f16.get(), CUDA_R_16F, ldc,
             CUBLAS_COMPUTE_16F, s_cublas_algo);
to_fp32_cuda(dst_f16.get(), dst_dd_i, row_diff, src1_ncols, stream);

// after:
const float alpha_f32 = 1.0f, beta_f32 = 0.0f;
cublasGemmEx(..., CUDA_R_16F, ..., dst_dd_i,    CUDA_R_32F, ldc,
             CUBLAS_COMPUTE_32F, s_cublas_algo_32f);
// (no exit cast)
```

**Caveats.**

1. **`CUBLAS_COMPUTE_32F` with F16 inputs uses F32 accumulators.** That's
   actually *better* numerics than CUBLAS_COMPUTE_16F (F16 accumulators)
   which the current code uses. NPC contract must be re-verified: the
   change *should* be byte-identical to a freshly-tuned ALGO that picks
   tensor-core m16n16k16 with F32 accumulator (CUBLAS_GEMM_ALGO0
   variants typically engage this).
2. **Algorithm choice.** The current code pins `CUBLAS_GEMM_ALGO0_TENSOR_OP`
   for the F16 case. There may be a CUBLAS_GEMM_ALGO0_TENSOR_OP_32F
   sibling that needs to be selected for the F32-output flavour — needs
   verification by ALGO survey, not assumption.
3. **The pinned-F16 path stays as-is.** Line 1923's `s_force_sid_cublas`
   branch writes F32 directly through a custom kernel. That branch
   path isn't engaged by default (TU102_SPEC #3 documented it as
   globally slower) — leave it alone.

**Ceiling for D1.** 57 ms (1.3%) wall reclaim if the cuBLAS internal
writeback is as fast as our explicit `convert_unary` cast. Could be
slightly faster (in-kernel writeback) or wash (same cost, just moved).
Empirical question, easy to measure.

**Effort.** Low. ~5-line diff, rebuild (~10 min), NPC matrix verify
(scripts/verify-production-determinism.sh at NP={1,2,4,8} multi-GPU),
re-bench at np=1 vanilla shape. ~30-50k tokens.

#### Sub-lever D2 — entry cast elision (higher effort, dependent on graph build)

The src1 F32→F16 cast at line 1907 needs the *producer* of src1 to
write F16. That's a graph-build change in `llama-build-context.cpp` —
mark the residual / norm output as F16 instead of F32. Affects every
downstream consumer of that tensor (which may include CPU paths, REDUCE
collectives, etc).

**Cost.** ~64 ms (1.4%) wall reclaim ceiling for the entry cast.
**Effort.** Higher; full op-graph change. Not in scope this round
unless D1 lands clean and we want more.

**Combined D1+D2 ceiling.** ~2.7% wall. D1 alone is ~1.3%.

### Out of scope this round

- **PSKV-singlewarp-FA (4.7%)** — separate ralph-loop workstream.
- **Prefill big tile MMQ ncols=112 (7.3%)** — TU102_SPEC #6, "lower
  leverage". Climbs at np=1 prefill but still bounded; the gain ceiling
  is small (the kernel already uses cutlass under the hood).
- **fused_rms_norm + cpy_flt + quantize_q8_1** — launch-overhead-bound
  per TU102_SPEC #7+#8+#10. Real but each ~1-2% wall; fold-into-MMQ
  epilogue is a larger workstream that should follow a kernel rewrite
  decision, not lead one.
- **Hadamard** — measured 0.3% wall; sub-1% claim from TU102_SPEC
  verified, not a target.

---

## Candidate A: premise falsification (2026-05-24)

This phase opened on 2026-05-24 with Candidate A scoped as "F.4.1'
dispatch split — keep decode rows on the F.4.1' kernel, route
MoE-expert-routing fan-out to a looser-NPC DP4A int8 kernel." The
framing was inherited directly from `PHASE_TU102_SPECIALIZATION.md`
section #4:

> *"this kernel handles both single-token decode rows AND MoE
> expert-routing fan-out. The expert fan-out path can run on a
> looser-NPC kernel using DP4A int8 dot with s32 accumulate,
> recovering ~50% on the MoE-routed portion without touching the
> activation-path kernel."*

Code-reading on 2026-05-24, before any instrumentation work, falsified
the framing in two independent ways:

**1. The model is not MoE.** GGUF metadata
(`/opt/models/recast-out/qwen3.6-27b-V-F1.T1.lm_head-f16.gguf`) inspected
via `gguf_dump.py --no-tensors`:
```
general.architecture        = 'qwen35'
qwen35.block_count          = 65
qwen35.embedding_length     = 5120
qwen35.feed_forward_length  = 17408
qwen35.attention.head_count = 24
qwen35.attention.head_count_kv = 4
qwen35.ssm.conv_kernel      = 4
qwen35.ssm.state_size       = 128
qwen35.full_attention_interval = 4
```
No `n_expert` / `expert_count` / `expert_used` field — the model is
**not an MoE**, it's a Mamba-2-SSM + attention hybrid with full
attention every 4th layer. The "27B-MoE" framing from TU102_SPEC was
either for a sibling model variant or a documentation error; the
deployed model has no expert routing.

**2. The kernel template fires from one source only.** The dispatcher
at `ik_llama.cpp/ggml/src/ggml-cuda/mmvq-templates.cuh:356` predicates
the `fused_mul_mat_vec_q<...>` kernel launch on
`args.vx_u && args.vx_g && args.unary_op != GGML_UNARY_OP_COUNT` —
i.e. both up and gate weight pointers must be set. This is uniquely
satisfied by `ggml_cuda_op_fused_mul_mat_vec_q_id`, which is itself
invoked from a single call site:
`ik_llama.cpp/src/llama-build-context.cpp:1323` —
`ggml_fused_up_gate(ctx, up, gate, cur, unary_op)` in the standard
FFN block builder. Fired once per FFN per token.

All other MMVQ paths — non-fused decode projections (Q, K, V, O
attention) and MUL_MAT_ID (if the model used it) — route to the
sibling `mul_mat_vec_q<...>` template, a separate kernel template
with a separate name in nsys. Per the captured kernel table, the
non-fused MMVQ kernel is **not in the top 15** — its calls dominate
neither prefill nor decode at this shape.

**Consequence.** Candidate A as originally scoped does not exist
because (a) there is no MoE in the model and (b) the kernel template
fires from one source. The ~3% wall reclaim ceiling for A as
originally framed (recover ~50% on the MoE-routed portion) was
based on a structural assumption that doesn't hold. Updating the
candidate description to reflect the corrected kernel-level scope
(see Candidate A above).

**Cost of the falsification.** ~30k tokens of code-reading,
methodology-matched against §1 (think before coding) and §8 (read
code before claiming behavior). Cheap relative to the alternative —
the original framing would have driven an instrument-rebuild-bench
cycle (estimated 60-100k tokens) that would have produced a single
histogram bin with all 8 194 calls in it, confirming the same
falsification at much higher cost. The §8 lesson lands: read code
before claiming behavior.

**Update to PHASE_TU102_SPECIALIZATION.** Section #4 of that phase
doc carries the inherited error. A follow-up edit to that phase
file (separate commit per §5 rule) should annotate the section with
the corrected understanding so the next session reading TU102_SPEC
doesn't re-inherit the framing.

---

## Recommended round-1 plan

```
PHASE_PERF_R2_NP1 — Step 1: AsyncReduce viability microbench  (Cand B prep)
  → Build a 100-line microbench: cudaMemcpyPeerAsync on stream A
    while a dummy compute kernel runs on stream B. nsys trace.
    Check whether timeline shows overlap on TU102.
  → Verify by: nsys timeline visually shows comm-stream and
    compute-stream regions intersecting; if not, mark Candidate B
    dead and pivot to Candidate A.

PHASE_PERF_R2_NP1 — Step 2: branch on the microbench
  → If overlap works → open PHASE_ASYNC_REDUCE_IMPL.md, drive
    PHASE_ASYNC_REDUCE.md to T1-T10 closure (~80-120k tokens).
  → If overlap fails on TU102 → pivot to Candidate A.

PHASE_PERF_R2_NP1 — Step 3 (Candidate A: F.4.1' kernel-level investigation)
  → Original "dispatch split" framing falsified by code-reading
    2026-05-24 (see "Candidate A: premise falsification" above).
    Reframed as a kernel-level investigation.
  → ncu probe of fused_mul_mat_vec_q<Q4_0,1,4,1> at this bench
    shape. Capture: registers/thread, theoretical+achieved occupancy,
    DRAM throughput %, achieved DP4A %, shmem/CTA.
  → Decision branch from ncu result:
     (a) Latency-bound + occupancy headroom → Lever-D-style rewrite
         (bump minBlocksPerSM from 1 to 2 on Turing).
     (b) Bandwidth-bound at peak → no architectural win; close as
         "kernel is hardware-limited at this shape".
     (c) Shmem-bound at <50% occupancy → smaller-fragment rewrite
         (Lever D analog).
  → Verify by: ncu evidence captured; if kernel rewrite ships,
    NPC matrix GREEN at NP={1,2,4,8} after change; tg128 gains
    ≥ 0.5 t/s at the np=1 shape.

PHASE_PERF_R2_NP1 — Step 4 (Candidate C: MMQ Q4_0 ncols=8 shape audit)
  → Instrument dispatcher to log (M, N, K) per call.
  → Histogram offline. Decision branch:
     (a) unimodal → close with evidence "kernel is near-optimal at
         this shape, variance is workload-intrinsic".
     (b) bimodal → ship shape-aware tile dispatcher.
  → Verify by: std-dev drops below 5 µs OR step closed with evidence.

PHASE_PERF_R2_NP1 — Step 5 (Candidate D: cast trio elision)
  → Identify the call site for the 53 621-instance cutlass+casts.
  → Mark surrounding tensors fp16-native in graph build to elide
    the f32↔f16 cast pair.
  → Verify by: convert_unary count drops ≥10× OR pair vanishes;
    NPC matrix GREEN; ~2% wall reclaim observed.
```

**Combined ceiling** if all four land: ~12-15% kernel-time reduction,
projected ~5-8% TG wall improvement (24.77 → ~26-27 t/s at this shape).

**Real-world expected** at ⅓ to ½ of ceiling: ~2-4% TG wall improvement
(24.77 → ~25.5-26 t/s). Honest framing per CLAUDE.md §8.

---

## Disciplines (carried from TU102_SPEC and CLAUDE.md)

- **NPC contract is sacrosanct.** Every change verified by
  `bash scripts/verify-production-determinism.sh` GREEN across
  NP={1,2,4,8} multi-GPU before merge. The six baked NPC fixes per
  `PHASE_NPC4_FIX_AUDIT.md` are unchanged by this work.
- **No env-gates** — bake the verified behaviour or revert. Per
  `feedback_bake_measurement_env_gates.md`.
- **ncu pass before kernel-design lock.** Per
  `feedback_kernel_replacements_must_be_sota_sm75.md`.
- **Spec lockdown before code on non-trivial changes.** AsyncReduce has
  Allium + TLA+ specs already. Candidates A/C/D are simpler — design
  doc + acceptance binding is sufficient.
- **Locked-clocks discipline** — TU102_SPEC ran at locked 1455 MHz; this
  re-rank ran at default clocks. Follow-up: lock clocks before any
  binding perf-delta measurement on this host. Documented as a
  prerequisite for Step 3+ verify-by gates.

---

## Open items

| Item | What |
|---|---|
| Lock GPU clocks on xeon | `nvidia-smi -lgc 1455,1455 -i 0,1`. Prerequisite for binding perf-delta gates on Steps 3+. Currently default (variable). |
| Relocate nsys captures | Move `/tmp/nsys-nvlink/nvlink-vanilla-np1-2026-05-24.nsys-rep` → `data/nsys-perf-2026-05-24-vanilla/`. Repo convention. |
| ~~Higher-c capture~~ | **CLOSED 2026-05-24.** Captured at `-c 262144` (`/tmp/nsys-nvlink/nvlink-vanilla-np1-ctx256k-2026-05-24.nsys-rep`). Kernel mix is functionally identical to `-c 4096` — every top-15 line within ±0.4 pp (largest shift: NCCL 9.1% → 9.5%, mul_mat_q ncols=112 7.3% → 7.4%). Working set / KV pre-alloc does not change the kernel-time distribution for this bench shape. The `-c 4096` ranking is the production-shape ranking. |
| ~~Production ctx 256k re-rank~~ | **CLOSED 2026-05-24** via the higher-c capture above. Same finding. |

---

## Companion files

- Predecessor ranking: `PHASE_TU102_SPECIALIZATION.md`
- AsyncReduce spec/plan: `PHASE_ASYNC_REDUCE.md` (planning, pre-impl)
- F.4.1' kernel: `PHASE_PERF_F4_1.md` (shipped 2026-05-17)
- MMQ Q4_0_AR16: `PHASE_MMQ_Q4_0_AR16.md` (Phases A+B+C shipped)
- T6 characterisation framing: `PHASE_T6_CHARACTERISATION.md`
- NPC closure (sacrosanct): `PHASE_NP_DETERMINISM_CLOSED.md`
- Acceptance wrapper: `scripts/verify-production-determinism.sh`
