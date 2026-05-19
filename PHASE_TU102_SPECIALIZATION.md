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

## Absolute kernel breakdown (HEAD)

NP=8 capture total GPU time ≈ 167 s; NP=1 ≈ 7.4 s (NP=1 is prefill-heavy
because 200-tok prefill dominates 64-tok decode at a single slot).

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

### #1 — `mul_mat_q[Q4_0,*]` → int8 IMMA tensor-core dispatch
- **Aggregate cost.** 50.7 s NP=8 decode (30.4%) + 1.75 s NP=1 prefill
  (23.8%). With sibling `Q4_0_AR16` (target #5) bundled: **59.8 s NP=8**.
- **TU102 hook.** sm_75 has 65 TOPS int8 tensor cores via
  `mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32` (IMMA). Q4_0 dequants
  naturally to int8 (its `vec_dot_type` is `Q8_K`, an 8-bit-int form).
  Today's MMQ path goes Q4_0 → fp16 → HMMA m16n8k8; an int8-fused path
  skips the fp16 conversion and doubles the effective TC throughput. The
  per-kernel reduction stays in s32, deterministic per call.
- **NPC contract.** Preserved by-construction as long as K-iteration order
  is fixed. IMMA's m16n8k16 fragment is byte-identical to a software s32
  fp-accumulator tree for our visit pattern.
- **Why this beats pre-NPC.** Pre-NPC was hitting cutlass 75_wmma h161616
  (fp16 TC, 130 TFLOPS ceiling); int8 IMMA on TU102 hits 65 TOPS but
  doesn't pay the dequant-to-fp16 traffic, so memory-bound shapes go
  faster. Effective ceiling is workload-shape dependent — needs ncu.
- **Effort.** **High.** New MMQ kernel TU; need to template the IMMA tile;
  cutlass 75 IMMA references exist (`cutlass_75_simt_imma_*`).
- **Bundles with target #5 (`Q4_0_AR16`)** since AR16 packing only
  changes the dequant inner loop.

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

### #3 — `cuBLAS_gemvx_bf16` lm_head → F16 + pinned-HMMA batched
- **Aggregate cost.** 8.8 s NP=8 decode (5.3%) + 271 ms NP=1 prefill
  (3.7%). Per-call **4174 µs** at NP=8 — by far the most expensive single
  call on the system.
- **TU102 hook.** sm_75 has **no BF16 tensor cores**. cuBLAS BF16 gemvx
  runs on fp32 CUDA cores (16.3 TFLOPS) vs fp16 TC (130 TFLOPS). Recast
  the lm_head BF16 → F16 (same tool used for DFlash Path A) and dispatch
  through `ggml_cuda_mul_mat_f16_pinned` batched across slots.
- **NPC contract.** Identical to DFlash's pinned-HMMA path
  (NPC-by-construction). Already proven across 5 gates in
  `PHASE_DFLASH_BATCHED_PINNED.md`.
- **Effort.** **Low.** Recast tooling exists. The fix is two changes:
  (a) recast tool emits an F16 lm_head into the production target GGUF,
  (b) lm_head dispatch in the main path routes through `dflash_gemm_npc`
  (or a sibling launcher in `ggml-cuda/`).
- **Yield.** 2105 calls × ~3500 µs saving each = ~7 s recoverable at
  NP=8 (assumes lm_head goes from 4174 → ~700 µs/call at the pinned
  batched shape).
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
