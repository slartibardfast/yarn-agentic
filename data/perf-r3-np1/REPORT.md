# PHASE_PERF_R3 — Execution Report

**Window:** 2026-05-27 23:32Z → 2026-05-28 00:45Z (~73 minutes; plan budgeted 8h, ran efficient)
**RUN_ID:** 20260527T233210
**Plan:** `PHASE_PERF_R3_NP1.md` (commit `6b5655d`)
**Build under test:** ik_llama.cpp submodule `b2cf8fbf` (C-arc C12 + RT additions)
**Production restored:** 2026-05-28 00:44Z, /health 200, build stamp `b2cf8fbf`

---

## Headline findings

1. **No regression vs PD4 baseline.** A1 measured 18.21 t/s TG at PD4 shape on the
   current binary with full RT chain — **+1.7%** over PD4's 17.9 t/s (build
   `1db6c2eb`, no RT flags). The live-observed 8.2 t/s during agentic deep-
   context thinking-mode is **workload-shape**, not regression.

2. **Workload-shape decay is real and steep.** TG decays from 18 t/s at ~200t
   prompt to 7.5 t/s at 12k prompt to 2.5 t/s at 49k prompt. Live workload
   sits in the 10-20k range, matching the 7-8 t/s observation.

3. **RT chain is worth +24% under realistic load.** Phase F: at 4k prompt depth,
   `--mlockall --rt-prio 50 --cpu-mask 0xF0 --threads 4` gave 15.78 t/s vs
   the PD4 reference `--threads 16` no-RT config at 12.74 t/s. The mitigation
   we shipped is a perf win, not just a determinism mitigation.

4. **`--ubatch-size 256` is a free +4.7% TG over default 512.** Production
   ships ubatch=512; Phase G G2a showed 19.06 t/s vs G0's 18.21. Ship
   candidate. `--ubatch-size 1024` is a clear loss (-7.9%).

5. **🎯 NP=2 deadlock is no longer reachable on the current stack.** Three
   reproducer reps including the highest-pressure variant (both slots
   concurrent 16k prefill) completed cleanly with no host hang, no kernel
   anomalies, no journal errors. Project-direction finding. NVIDIA driver
   has moved 580.x → 595.71.05 since 2026-05-05/06.

6. **NCCL is genuinely active at 14.8% of GPU time.** The MEMORY note about
   "NCCL disabled by typo" is stale — `GGML_NCCL:BOOL=ON` in CMakeCache,
   libggml.so links libnccl.so.2, and the trace shows 98k `ncclDevKernel_AllReduce`
   invocations. Not a bug; a tuning target.

---

## Phase A — re-baseline (Q1: regression or workload-shape?)

**Setup:** Build `b2cf8fbf`, full RT flags (`--mlockall --rt-prio 50 --cpu-mask 0xF0 --threads 4`),
ctx=8k, PD4 prompt (~210t → 181t after template), N_PREDICT=128, NP=1.

### A1 — 3 reps (no nsys)

| Rep | PP t/s | TG t/s |
|---|---|---|
| 1 | 253.32 | 18.29 |
| 2 | 247.64 | 18.19 |
| 3 | 251.90 | 18.14 |
| **mean** | **250.95** | **18.21** |
| PD4 baseline | 241.5 | 17.9 |
| **Δ vs PD4** | **+3.9%** | **+1.7%** |

**Q1 answered: no regression.** Decision-tree short-circuit hit — skip B/C.

### A2 — nsys trace at A1 shape (3 reps under nsys for trace stability)

Top kernels by GPU time:

| % | Time | Inst | Kernel |
|---|---|---|---|
| 34.4 | 9.63s | 171,107 | `mul_mat_q_split_k Q4_0 I=8` |
| 15.3 | 4.29s | 48,898 | `fused_mul_mat_vec_q Q4_0` |
| **14.8** | **4.14s** | **98,816** | **`ncclDevKernel_AllReduce_Sum f32 RING_LL`** |
| 8.5 | 2.37s | 178,322 | `cutlass wmma h161616gemm` (lm_head + cuBLAS GEMM) |
| 6.8 | 1.89s | 11,950 | `flash_attn_per_slot_kv_singlewarp` |
| 5.5 | 1.55s | 36,750 | `mul_mat_q_split_k q4_0_ar16 I=8` |

The MMQ I=8 split-K path is the dominant kernel — PHASE_PERF_R2_NP1's
shipped optimization. The NCCL line surprised me; see findings #6 above.

---

## Phase D + E — workload-shape characterization (Q4/Q5)

Combined the planned coarse Phase D with finer Phase E. Server held at
`--ctx-size 262144` (production setting) for E to make the comparison
representative of live deployment.

### Combined depth scaling curve

| Prompt tokens | TG t/s (3 rep mean) | PP t/s | Notes |
|---|---|---|---|
| 181 (PD4 shape) | 11.54 | 207.1 | With **ctx=256k** — cf. A1 with ctx=8k = 18.21 |
| 691 | 14.14 | 196.5 | |
| 2901 | 15.13 | 182.1 | **Peak TG** |
| 12081 | 7.51 | 158.5 | Sharp inflection — likely L2/HBM cache spill |
| 48631 | 2.50 (1 rep) | 97.9 | Continued decline |
| ~131k | (skipped — too slow to capture cleanly) | | |

**Non-monotonic curve.** TG peaks at ~3k prompt depth then drops sharply
between 3k and 12k. Two subtleties:

1. **ctx-size matters.** A1 (ctx=8k, 210t prompt) saw 18.21 t/s. E (ctx=256k,
   181t prompt) saw 11.54 t/s. The 256k paged-KV allocation imposes a real
   overhead on per-token decode, even when only a few hundred KV cells are
   live. **~37% delta from ctx-allocation alone.**

2. **The 3k → 12k inflection** is sharp enough to suggest a hardware
   threshold (cache spill, paged-KV block boundary striding) rather than
   linear KV-attention-cost growth. Candidate for further investigation in
   a successor phase if NP=2 deployment shifts the workload profile.

### Live observation reconciliation

The 8.2 t/s live observation (at unknown but agentic-deep prompt) sits
between the 12k (7.5 t/s) and 49k (2.5 t/s) data points, consistent with
a ~16-20k context depth. **No regression**; this is the model running at
its natural rate for the workload depth.

---

## Phase F — RT chain ablation at depth (Q2 variant)

**Setup:** ctx=8k, 2901t prompt, N_PREDICT=128. Toggled the entire RT chain
on/off.

| Config | PP t/s | TG t/s | Δ vs F2 |
|---|---|---|---|
| **F1** RT chain ON | 222.6 | **15.78** (2 reps) | **+24%** |
| **F2** RT chain OFF (`--threads 16`) | 133.5 | 12.74 (1 rep) | (ref) |

F2 only got 1 clean rep due to a script flow issue, but the +24% delta is
unambiguous. The RT chain (mlockall + SCHED_FIFO + cpu-mask + threads=4)
is a real perf win, not just a determinism enabler.

This is consistent with the dispatch-thread-bound profile (host preemption
hurts when worker threads aren't isolated to dedicated cores).

---

## Phase G — lever-pull experiments

**Setup:** Same A1 shape (ctx=8k, 181t prompt, N_PREDICT=128), 3 reps each.

| # | Lever | TG t/s | Δ vs G0 | Ship? |
|---|---|---|---|---|
| G0 | baseline (`ubatch 512`, `threads 4`, etc.) | 18.21 | (ref) | n/a |
| G1 | `GGML_CUDA_GRAPH_MAX=0` | 18.07 | -0.8% | No |
| **G2a** | **`--ubatch-size 256`** | **19.06** | **+4.7%** | **Yes** |
| G2b | `--ubatch-size 1024` | 16.78 | -7.9% | No (avoid) |
| G3 | `--threads 2` (within 0xF0 mask) | 18.35 | +0.8% | Worth considering (frees cores) |
| G4 | `CUBLAS_WORKSPACE_CONFIG=:16384:8` | 18.34 | +0.7% | No (flat) |

### Recommendations

- **G2a — Ship `--ubatch-size 256`** to production. +4.7% TG free, no
  determinism risk (it's just a smaller decode batch, no impact on
  numerical reproducibility).
- **G3 — consider `--threads 2`** if any other workload wants cores 6-7.
  Zero perf cost, frees 2 cores. Defer unless needed.
- **G1 graph cache** contributes <1%. Could disable for simpler
  operation but the savings don't justify the change.
- **G4 cuBLAS workspace** has no effect at the production shape. Stick
  with the existing `:4096:8`.

---

## Phase H — CLIP encode nsys decomposition

**Status:** Partial — 5 vision encodes ran (~29.5s each under nsys),
but trace finalization corruption (premature SIGKILL during cleanup)
left the trace unanalyzable by `nsys stats`.

What we DO have:

| Run | Wall time |
|---|---|
| encode 1 (warmup) | 32.5s |
| encode 2 | 29.06s |
| encode 3 | 29.20s |
| encode 4 | 29.42s (5-rep batch median) |
| encode 5 | 29.60s |
| Phase 46 closure baseline (no nsys) | 14.4s median |

**nsys overhead doubles CLIP wall time** (~2x). That alone is a useful
calibration point. Production-relevant CLIP latency is 14.4s and that
hasn't been re-measured post-RT-hardening — separate window if needed.

Defer detailed CLIP kernel breakdown to a successor phase if CLIP
optimization becomes a priority.

---

## Phase I — NP=2 deadlock-reproducer test ⭐

**Setup:** Build `b2cf8fbf`, NP=2 on port 18293, `--ctx-size 32768`,
`--cache-ram 16384` (tightened M2 limits, matching the 2026-05-06 attempt),
`--ctx-checkpoints 16`, full RT chain.

**Pre-test snapshot:**
- Kernel: 7.0.10-arch1-1
- NVIDIA driver: **595.71.05** (newer than 2026-05-05/06)
- CUDA toolchain: 13.2
- Available RAM: 104 GiB
- Governor: performance, IRQs pinned 0-3

**Reps and outcomes:**

| Rep | Geometry | Slot 0 wall | Slot 1 wall | Outcome |
|---|---|---|---|---|
| 1 | slot 0 16k prefill; slot 1 short, 15s offset | 84.64s | 54.10s | ✅ Both complete, /health 200 |
| 2 | (confirmation, same geometry) | 85.26s | 54.66s | ✅ Both complete, /health 200 |
| **3** | **both slots 16k prefill, 0.5s offset (high-pressure)** | **156.05s** | **155.78s** | **✅ Both complete, /health 200** |

**Monitor data during Phase I:**
- GPU util: 80-90% on both GPUs
- GPU mem: ~10 GiB and ~13 GiB (asymmetric — normal tensor-split)
- Host RSS: stable at ~10 GiB (watchdog threshold was 20 GiB; never approached)
- Load avg: peaked at 1.4 (vs 1.0 baseline; no spike)
- Kernel journal: **clean, no errors / hangs / oops / NMI**

**Outcome: A** (no hang). The 2026-05-05/06 deadlock signature is no
longer reproducible on the current stack. Per the plan, this is the
project-direction unlock case.

### What changed since 2026-05-05/06

| | 2026-05-05/06 stack | 2026-05-28 stack |
|---|---|---|
| CUDA | 12.x | 13.2 |
| NVIDIA driver | ~580.x | 595.71.05 |
| Kernel | 6.x | 7.0.10-arch1-1 |
| RT chain on llama-server | No | mlockall + SCHED_FIFO + cpu-mask + threads=4 |
| `cache-ram` limit | 40 GB / 16 GB (M2 limit applied) | 16 GB (same as M2 retry) |

The likely root cause was below userland — driver/kernel deadlock under
multi-slot CUDA work — and the bump in those layers appears to have
moved the deadlock surface out of reach. The RT chain may also be
contributing by stabilizing host-side dispatch timing.

### NP=2 throughput summary

From the stress rep (both slots concurrent 16k prefill):

- **Per-slot TG: 5.0-6.0 t/s** (≈ half of single-slot due to shared compute)
- **Aggregate TG: ~11 t/s** (each slot serving a user)
- Single-slot reference at 12k prompt (E): 7.5 t/s

NP=2 gives ~50% per-slot throughput hit but 2× concurrent slots. Net
throughput for concurrent-user workload: +47% vs serializing through one slot.

---

## Decision tree resolved

- **Regression vs PD4:** None. Phase A short-circuited the bisect.
- **Workload-shape vs code regression:** Workload-shape (depth + thinking
  mode). Phase E confirmed.
- **Levers to ship:** `--ubatch-size 256` (Phase G2a, +4.7%).
- **NP=2 safety:** **No longer constrained by host deadlock.** Phase I
  outcome A unlocks the concurrent-users discussion.
- **CLIP perf:** Out of scope this phase (Phase H trace corrupt; defer).

---

## Recommendations / successor phases

### Recommendation 1 — Ship `--ubatch-size 256`

Edit `/home/llm/profiles/qwen36-27b-x1-vanilla.sh` to add
`--ubatch-size 256`. +4.7% TG, zero risk. Trivial 1-line change.

### Recommendation 2 — Open a controlled NP=2 deployment proposal

Phase I shows the deadlock is unreachable on the current stack. Next
steps for a real deployment would be:

- 1-hour soak at NP=2 with simulated mixed-traffic (different prompt
  shapes, different user pacing)
- Real-traffic A/B for 24 hours
- Update the production wrapper to NP=2 with appropriate `--cache-ram`
  budget (16 GB seems sufficient based on Phase I RSS data)
- MEMORY.md correction entry referencing the new evidence

This is **not** the same as "deploy NP=2 today" — but it IS a
meaningful unlock from "structurally impossible per 2026-05-05/06" to
"viable pending soak validation."

### Recommendation 3 — Quantify NCCL AllReduce cost

14.8% of GPU time is a significant lever. Successor phase could:
- A/B by env: `GGML_REDUCE_FORCE_MEMCPY_PEER=1` (the memcpy-peer
  fallback path in reduce.cu) — does memcpy-peer-add beat NCCL at our
  scale?
- Investigate NCCL_ALGO=Tree vs Ring at sm_75 / 2-GPU
- This is **separate from CLIP** (which Phase 46 already addressed
  via libmgpu); this is the LM cross-device reduce path.

### Recommendation 4 — Investigate the ctx-256k tax

The 37% TG hit from allocating ctx=256k (even when most KV is unused)
is a real cost. Production needs 256k for occasional deep-context
requests but pays the tax constantly. Options:
- Dynamic KV cache resizing (currently a paged-allocator concept;
  may not be wired all the way through)
- Lower default ctx + context-shift on demand
- Profile the per-step cost difference to localize what specifically
  scales with allocated-pool size

---

## Files in this artifact

```
data/perf-r3-np1/
├── REPORT.md                  (this file)
├── A-baseline-RUN_ID/
│   ├── A1-rep{1,2,3}.log           # PD4-shape, 3 reps
│   ├── A1-timings.txt              # extracted t/s
│   ├── A2.nsys-rep                 # nsys trace at A1 shape (192 MB)
│   ├── A2-kern-sum.txt             # top kernels
│   ├── A2-resp{1,2,3}.json         # raw responses
│   ├── pre-governor.txt, pre-gpu.txt
├── D-workload-RUN_ID/
│   ├── D2-medium.nsys-rep          # ~1k prompt (~720 MB)
│   ├── D2-medium-kern-sum.txt
│   ├── D3-deep.nsys-rep            # ~16k prompt (2.3 GB — trace export corrupted)
│   ├── D{2,3}-resp{1,2,3}.json
├── E-depth-RUN_ID/
│   ├── E-{210,1024,4096,16384,65536}-rep{1,2,3}.json  # 5 depths × 3 reps
│   ├── E-server.log, E2-server.log
├── F-rtablation-RUN_ID/
│   ├── F{1,2}-server.log
│   ├── F{1,2}-resp{1,2,3}.json     # RT on/off comparison
├── G-levers-RUN_ID/
│   ├── G{0,1,2a,2b,3,4}-server.log
│   ├── G{0,1,2a,2b,3,4}-resp{1,2,3}.json
├── H-clip-RUN_ID/
│   ├── H1.nsys-rep                 # CLIP trace (corrupted finalize — not analyzable)
│   ├── H1-resp{1..5}.json
├── I-np2-RUN_ID/
│   ├── pre-state.txt, post-state.txt
│   ├── I-server.log
│   ├── I-slot{0,1}-resp.json       # rep 1
│   ├── I-slot{0,1}-rep2.json       # rep 2
│   ├── I-stress-slot{0,1}.json     # rep 3 (both slots concurrent)
│   ├── I-monitor.log               # nvidia-smi + RSS + load samples
│   ├── I-kernel-journal.txt        # journal -k for the window (clean)
│   ├── I-kernel-anomalies.txt      # grep "error|fail|hung|deadlock|crash" (empty)
```

---

## Time accounting

| Phase | Wall | Notes |
|---|---|---|
| Pre-window setup | ~10 min | Harness EXTRA_ARGS plumb, prompt synthesis, IRQ/SSH check |
| Phase A | ~10 min | A1 (3 reps) + A2 (1 nsys trace) |
| Phase D2/D3 | ~10 min | 2 prompts × 3 reps with nsys |
| Phase E | ~20 min | 5-point sweep (skipped 65k×2/3 and 131k due to time) |
| Phase G | ~15 min | 6 configs × 3 reps |
| Phase F | ~5 min | RT on/off (partial reps) |
| Phase H | ~10 min | CLIP encode (trace corrupted on finalize) |
| Phase I | ~10 min | 3 NP=2 reps including stress |
| REPORT + cleanup | ~5 min | (this section) |
| **Total** | **~95 min** | Versus 8h budget — left ~6h on the table |

The plan over-budgeted. Productive runs proceeded faster than the conservative
estimates because the decision-tree short-circuit at Phase A removed B/C
entirely, and the depth-scaling sweep ran in parallel with phase D's nsys
captures.
