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

### Candidate A — fused_mmvq F.4.1' dispatch split

**Cost.** 714 ms (15.7%), 8 194 calls @ 87.1 µs (tight, std 0.5 µs —
kernel is well-tuned, not a candidate for rewrite).

**Lever per TU102_SPEC #4.** F.4.1' is best-in-class under the NPC
contract for decode rows. It also serves **MoE expert-routing fan-out** at
the same call site. The expert fan-out path doesn't need the NPC
guarantee — it can run on a looser-NPC DP4A int8 dot with s32 accumulate.

**Recover target.** ~50% on the MoE-routed portion of the 8 194 calls.
If MoE is ~⅓ of qwen3.6's 27B-MoE layers per token, that's
~8 194 × 0.33 × 0.5 ≈ 1 350 call-equivalents @ 87 µs = ~118 ms ≈ **2.6% wall**.

**NPC contract.** Decode-row path stays on the current F.4.1' kernel
byte-identically. MoE path is new kernel with its own NPC contract
(byte-identical *per expert routing*, but ULP-divergent vs
F.4.1' is acceptable because routing isn't on the NPC-binding critical
path — verify against `scripts/verify-production-determinism.sh` before
default-on).

**Effort.** Medium. Dispatch-site triage (find where MoE expert routing
calls F.4.1' vs where decode rows call it) + new kernel TU + acceptance.

**Per CLAUDE.md §1**: dispatch-site triage is genuinely needed — I don't
yet know whether the 8 194 calls are split 50/50 or 90/10 between decode
rows and MoE routing. The 50% recovery projection assumes a ⅓ MoE share;
falsifiable upfront with NVTX annotations or a counter probe.

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

### Candidate D — convert_unary cast trio elision (TU102_SPEC #10)

**Cost.** 64 ms + 57 ms convert_unary pairs + 77 ms splitKreduce =
**198 ms (4.4%)** combined, all paired with 446 ms of cutlass.

**Lever.** Promote matched-dtype neighbours so the F16↔F32 casts elide.
Or, if the cast is around cuBLAS-Lt calls where the math layout is
fp16-internal, mark the surrounding tensors as fp16 in the graph build
and skip the cast pair.

**Effort.** Medium. Cross-kernel dispatch surgery. Per TU102_SPEC #10's
characterization — not architectural, purely op-graph cleanup.

**Ceiling.** ~120 ms / 4 540 ms = **~2.6% wall** if the full cast time
elides. Lower if only half the pairs are eligible.

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

PHASE_PERF_R2_NP1 — Step 3 (Candidate A: F.4.1' dispatch split)
  → Instrument F.4.1' call site with NVTX annotation distinguishing
    decode-row vs MoE-expert-routing paths.
  → Rerun nsys at this shape; count calls in each branch.
  → If MoE share ≥ 25% of calls → ship MoE-path looser-NPC kernel.
  → Verify by: NPC matrix GREEN at NP={1,2,4,8} after split;
    tg128 gains ≥ 0.5 t/s at the np=1 shape.

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
| Higher-c capture | This trace used `-c 4096` to match TU102_SPEC methodology. Capture at `-c 262144` (production KV pre-alloc) as a follow-up to verify memory-bandwidth kernels don't reshuffle the top-15. |
| Production ctx 256k re-rank | This trace is at `-c 4096` for methodology match. The real production runs at ctx 262144 — KV scratch is ~10 GiB not 78 MiB. Re-capture at production ctx if any candidate's projection depends on memory-bandwidth saturation. |

---

## Companion files

- Predecessor ranking: `PHASE_TU102_SPECIALIZATION.md`
- AsyncReduce spec/plan: `PHASE_ASYNC_REDUCE.md` (planning, pre-impl)
- F.4.1' kernel: `PHASE_PERF_F4_1.md` (shipped 2026-05-17)
- MMQ Q4_0_AR16: `PHASE_MMQ_Q4_0_AR16.md` (Phases A+B+C shipped)
- T6 characterisation framing: `PHASE_T6_CHARACTERISATION.md`
- NPC closure (sacrosanct): `PHASE_NP_DETERMINISM_CLOSED.md`
- Acceptance wrapper: `scripts/verify-production-determinism.sh`
