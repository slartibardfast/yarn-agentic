# DeltaNet np>1 determinism — production lockdown (staged)

> "DeltaNet" here is the user-named label for the full vanilla hybrid pipeline (DeltaNet recurrence + interleaved full-attention + projection GEMMs + RMSNorms). The drift surface spans all of it; treating one component as THE source proved wrong empirically (see D1/D2 below).

## Context

Vanilla decode on Qwen 3.6 27B (hybrid `linear_attn` + `full_attention` arch) is non-deterministic at np>1. Locked findings from prior work + this session:

- **PHASE45 D10.e** characterized the framework (cuBLAS heuristic shape-sensitivity, FA split-size variation, per-row reduction-order drift) but **D10.e.2** (the planned 3-kernel reduction-order rewrite) did not ship. See `docs/phases/60-llama-context-decompose/PHASE45.md`.
- **DFlash T9.1** added direct token-vs-NP=1 diff for vanilla at np ∈ {2,4,8} on 8 production prompts × 64 tokens:
  - NP=1 ≡ NP=2 byte-identical at TOKEN level (greedy argmax stable).
  - NP=4 ≡ NP=8 deterministic drift, identical first-divergence positions across both np values for 3 of 4 prompts (p0:tok3, p1:tok19, p3:tok0).
  - Binary boundary at NP=2 → NP=4 at the TOKEN level.
  - Data: `data/phase_dflash_t8/gate7-token-diff-summary.json`.
- **D1/D2 this session** added per-layer residual capture and direct per-prompt first-divergent-layer localization at NP=2 vs NP=4:

  | Prompt | first-divergent layer | layer type | max_abs_diff |
  |---|---|---|---|
  | p0 | 19 | FA (5th FA layer) | 0.006 |
  | p1 | 3  | FA (1st FA layer) | 0.003 |
  | p3 | 0  | **DeltaNet** | **0.99** |

  **The drift site is prompt-dependent and spans the whole hybrid pipeline.** No single kernel is THE source. Earlier "DeltaNet recurrence is fine" conclusion was p0-specific. Evidence: `data/deltanet/d2-first-divergent-layer.json` + `data/deltanet/d2-multi-prompt-update.json`.

- **NP=1 is its own special case**: residual stream is F32-stored at single-token decode (n_tokens=1), F16-stored at multi-token (n_tokens≥2). NP=1 vs NP≥2 comparisons mix a STORAGE-PATH delta with the COMPUTATIONAL delta. NP=2 is the cleanest deterministic baseline for diagnosis.

**Why this blocks everything downstream**: DFlash and MTP speculative overlays require a deterministic target forward at the np they ship at. Any spec-decode method built on a drifty target is structurally broken — accept-rate inflation, mid-sequence trajectory divergence, irreproducible bug reports. Production-quality np>1 ship paths cannot exist until this is fixed.

This plan replaces all prior DFlash multi-slot / DeltaNet optimization work. DFlash kernel optimization, MTP multi-slot, and DFlash overlay on np>1 are explicitly downstream of Stage 3 closure here.

## Direction

**Staged by NP transition.** NP=2 is the floor; address each break as np increases:

1. **Stage 1 — NP=2 self-determinism**. Confirm NP=2 produces byte-identical residuals across repeated decodes (same prompt × N reps). This is the deterministic floor every later stage builds on. CUDA kernels are typically rep-deterministic on identical inputs; this stage is a quick verification, not expected to need fixes.
2. **Stage 2 — NP=2 ↔ NP=4 byte-identity**. The primary fix surface. Multi-prompt D2 evidence shows drift origins across DeltaNet recurrence, FA path, and projection GEMMs depending on input. All in scope for Stage 2.
3. **Stage 3 — NP=4 ↔ NP=8 byte-identity**. Sub-F16-precision drift at layer 0 cumulates across the stack. Likely upstream of DeltaNet (RMSNorm / input GEMMs at varying n_tokens). Distinct from Stage 2's drift sources.
4. **Stage 0 (optional, deferred) — NP=1 ↔ NP=2 byte-identity**. Requires unifying the F32/F16 residual storage path. Lower priority; deferred until Stages 1–3 close.

The approach within each stage: **localize → characterize → replace per Thinking Machines Lab batch-invariance recipe** (per-row CTA, fixed compile-time tile, no Split-K, no cross-block `atomicAdd<float>`, warp-shuffle inner reductions, SMEM-tree block-level reductions).

**Performance is a co-equal objective with determinism, not a tradeoff to surface at the end.** Determinism-at-any-cost is rejected. Each stage's replacement kernels must hit the measured-baseline t/s at np=1 AND deliver np>1 aggregate uplift consistent with the TU102 + NVLINK envelope work named in `project_dflash_t8_closed`. Two reasons this is achievable, not aspirational:

1. **Heuristic dispatch IS a perf cost.** cuBLAS `cublasGemmEx` at small M (decode batch widths) routinely runs at <5% of TU102 peak because the heuristic picks general-purpose algorithms. Fixed-tile bespoke MMQ-style kernels with compile-time geometry and explicit tensor-core scheduling beat it at the shapes we care about. Removing heuristic dispatch removes a runtime branch AND opens the door to nvcc full-loop-unroll.
2. **The replaced kernels are the same ones DFlash T8 named as bound at ~1% of TU102 peak.** The `gemm_row_x_col_kernel` and `dflash_drafter_lm_head_kernel` analysis applies to vanilla projection GEMMs at decode shapes too — same Turing peak, same memory-bandwidth ceiling, same massive headroom. Determinism is not subtracting perf here; it is the lever that lets us claim the perf.

Each stage's design step must include a measured perf target per replacement kernel. Each stage's verification verifies that target, not merely "no regression."

## Stage 1 — NP=2 self-determinism (FLOOR)

### S1.1 — Multi-rep NP=2 capture

Run the D1 harness 3 times at NP=2 with offset=0. Same prompts (p0, p1), same context init, fresh KV cache each rep.

**Verify by**: for each layer L in [0, 63], slot K in {0, 1}, three reps produce byte-identical .bin contents. `data/deltanet/s1-1-rep-identity.json` records rep1, rep2, rep3 hashes per (L, K).

### S1.2 — Stage 1 closure or remediation

- All 3 reps byte-identical → Stage 1 closes immediately. CUDA kernel-level determinism on identical inputs is the expected behavior.
- Any (L, K) differs across reps → identify the non-deterministic op (likely atomicAdd or unguarded race). Fix and re-bind. **This would invalidate every downstream stage's measurement framework**, so this MUST close before Stage 2 starts.

## Stage 2 — NP=2 ↔ NP=4 byte-identity

### S2.1 — Multi-prompt first-divergent-layer map (CLOSED)

Captured in `data/deltanet/s2-1-per-prompt-first-div.json` (all 8 prompts):

| Prompt | first-divergent layer | layer type | max_abs_diff |
|---|---|---|---|
| p0 | 19 | FA (5th FA layer) | 0.006 |
| p1 | 3  | FA (1st FA layer) | 0.003 |
| p2 | 3  | FA (1st FA layer) | 0.007 |
| p3 | 0  | **DeltaNet** | **0.99** |
| p4 | 3  | FA (1st FA layer) | 0.003 |
| p5 | 3  | FA (1st FA layer) | 0.001 |
| p6 | — | no drift at first decoded token | — |
| p7 | — | no drift at first decoded token | — |

Distribution at first-decoded-token: FA = 5/8, DeltaNet = 1/8, no drift = 2/8.

### S2.2 — Kernel-dispatch characterization (cheap hypothesis test, before op-level capture)

**Hypothesis under test:** the drift originates at heuristic-dispatch flips — cuBLAS picks a different `cublasGemmAlgo_t` at M=2 vs M=4, or the FA kernel template varies at n_tokens=2 vs n_tokens=4. If TRUE, the fix is targeted (force fixed algorithm at the named call sites). If FALSE, drift is computational (split-K accumulators, atomic reductions, etc.) and S2.3 op-level capture is needed.

Instrumentation:
- **cuBLAS**: enable `CUBLAS_LOGINFO_DBG=1` (or equivalent); capture `cublasGemmAlgo_t` picked at every GEMM call site, tagged with the (M, N, K, op_name) tuple. Compare logs from NP=2 vs NP=4 runs.
- **FA**: instrument the FA dispatcher (`fattn-common.cuh` or equivalent) to printf the chosen kernel template parameters (KQ_per_iter, head_dim, n_seq_block, ncols) at every FA call. Compare logs.
- **DeltaNet recurrence**: confirm `threads_per_block` selection (256 vs 128) is identical across NP=2 and NP=4 — both have n_tokens ∈ {2, 4}, both ≤ 8, both should hit the 256-thread variant. If confirmed identical → DeltaNet kernel itself is dispatched the same way; p3's drift must originate upstream (its input RMSNorm or Q/K/V projection GEMM).
- **Projection GEMM dispatcher in ggml-cuda**: separately, instrument `ggml-cuda.cu`'s matmul entry point to log which backend (cuBLAS heuristic vs MMQ vs cuBLAS fixed) gets selected.

Cost: roughly 10–15k tokens (instrumentation + 2 runs + log diff). Cheaper than S2.3 op-level capture because no harness extension or cb_eval matching changes — just printf at the dispatch sites.

**Verify by**: `data/deltanet/s2-2-dispatch-trace.json` records the kernel/algo/template chosen at each call site at NP=2 and NP=4, with diffs flagged. If any diff is found at or near the first-divergent layer for any prompt, the hypothesis is confirmed and the named call site is the S2.4 design target.

**Decision branch on S2.2 outcome:**
- DISPATCH DIFFERS at the first-divergent layer for some prompt → skip S2.3; go directly to S2.4 design with the named call site(s) as the target.
- DISPATCH IDENTICAL across all call sites → drift is computational; run S2.3.

### S2.3 — Per-prompt op-level intermediate capture (CONDITIONAL on S2.2 not naming it)

Only run if S2.2's dispatcher trace doesn't name a clear culprit. Heavier diagnostic — extends the cb_eval extract API to match intermediate-op tensor names, not just `l_out-<il>`.

For each prompt's first-divergent layer (S2.1 result), capture per-op intermediates:
- `attn_norm-<il>`, `Q-<il>`, `K-<il>`, `V-<il>`, `KQV-<il>` (FA output), `attn_out-<il>` (O projection), `ffn_norm-<il>`, MLP gate/up/down intermediates, `l_out-<il>`

Find the FIRST op at the layer where NP=2 vs NP=4 residual differs.

**Verify by**: `data/deltanet/s2-3-per-prompt-op-localization.json` maps each prompt → (layer, first-divergent op, kernel name).

Expected breakdown (per S2.1):
- p3 layer 0: localize within DeltaNet's input pipeline (RMSNorm → Q/K/V projection → recurrence kernel)
- p1/p2/p4/p5 layer 3: localize within first FA layer (attn_norm → Q/K/V → FA → O)
- p0 layer 19: localize within mid-stack FA layer (same ops, different position)

Cost: 25–40k tokens (cb_eval matcher extension + 8 captures + analysis). Higher than S2.2 because of harness extension and per-op file proliferation.

### S2.4 — Design replacement kernels (SoTA sm_75 spec)

Per `feedback_kernel_replacements_must_be_sota_sm75`: any replacement kernel spec MUST be SoTA for sm_75 with full register allocations and profiling data committed. Not aspirational "TML recipe applied" prose — concrete numbers.

For each op needing replacement, document in `specs/deltanet/<kernel-name>.md`:

- **Kernel design**: state-of-the-art for sm_75, citing the specific Turing features used:
  - `mma.sync.m16n8k8` PTX (the only m16n8k8 tensor-core instruction on Turing)
  - `ldmatrix.sync` swizzle patterns for shared-memory loads
  - 64 KiB SMEM per SM budget
  - 65536-register file per SM budget
  - 32-thread warps, no warp-level subdivision
- **Grid + block geometry**: per-row CTA, fixed compile-time tile dims, explicit blockDim / gridDim for the target shape.
- **Per-thread register budget**: from `--ptxas-options=-v` output, committed to the spec. Target ≤ 64 regs/thread for high occupancy.
- **SMEM layout per CTA**: bytes per buffer, total per-CTA budget, expected blocks-per-SM via occupancy calculator.
- **Reduction strategy**: warp-shuffle inner; SMEM-tree across warps; no cross-block atomics on float.
- **Numerical contract**: scalar fp32 oracle behavior to bind byte-identity (mirror DFlash T3/T4 wmma-mimicking-oracle pattern).
- **Perf contract** (anchored to TU102 ceiling):
  - HBM bandwidth: 624 GB/s per-GPU, 1.25 TB/s aggregate with NVLINK
  - FP16 tensor-core peak: 130.5 TFLOPs per-GPU, 261 TFLOPs aggregate
  - NVLINK aggregate: ~100 GB/s
  - Target latency or throughput cited as % of relevant ceiling, with work amount (bytes streamed or FLOPs computed) shown explicitly.
- **Profiling-data binding**: committed nsys + ncu profile artifacts at the target shape in `data/deltanet/` showing the kernel hits its target. The DESIGN is incomplete without these.
- **Tensor-core usage mandatory** for any GEMM-shaped op at our shapes. Scalar fp32 deviations require explicit justification with perf-cost analysis in the spec.

Template: `specs/dflash/kernel-design.md §6.1, §6.2, §6.3, §6.6` — each has full register budgets, SMEM allocations, occupancy targets, and binding to architecture peaks. Mirror that template.

**Verify by**: spec file committed with all of the above; `--ptxas-options=-v` output captured; nsys/ncu profiles bound.

### S2.5 — Implement replacement kernels + scalar oracle + unit test

Per `feedback_test_first_discipline.md`: scalar oracle and RED unit test FIRST.

- `tests/test-hybrid-batch-invariance.cpp` — randomized inputs × (np ∈ {2,4}) × seeds × shape variations.
- Scalar oracle: header-only fp32 CPU reference per replaced kernel.
- Kernel: new file under `ggml/src/ggml-cuda/` if entirely new, or surgical edit if drop-in.

**Verify by**: byte-identity unit test PASS at np ∈ {2,4} on randomized inputs (or ≤ 1 fp16 ULP, matching DFlash T3 standard). Existing test-backend-ops sweep stays GREEN.

### S2.6 — Stage 2 integration: byte-identity binding on production model

Wire replacement kernels into build graph. Extend test-np-validity-vanilla.cpp (or fork a Stage 2 binding harness) with byte-identity assert at NP=2 vs NP=4.

**Verify by**:
- All 8 prompts × 3 reps each: byte-identical full 64-token sequence vs NP=2 reference at NP=4.
- Existing 5 T9.1 vanilla validity asserts GREEN at NP=2 AND NP=4.
- Per-layer per-slot residual byte-identity (the D1 harness's lens) GREEN at NP=2 vs NP=4 for all 8 prompts.
- `data/deltanet/s2-6-stage2-binding.json` records all hashes.

### S2.7 — Stage 2 perf binding

Measure np=1 single-token decode tg t/s, np=2 aggregate tg t/s, np=4 aggregate tg t/s. Compare against pre-Stage-2 baseline.

**Verify by**: `data/deltanet/s2-7-perf-binding.json`. Acceptance:
- np=1: ≥ baseline tg t/s.
- np=2: ≥ baseline aggregate tg t/s.
- np=4: ≥ baseline aggregate tg t/s, per-slot ≥ baseline-per-slot / 1.02.
- Each replaced kernel hits ≥ 30% of theoretical ceiling (HBM or tensor-core peak), measured by nsys.

Stage 2 closure requires S2.6 AND S2.7 both GREEN.

## Stage 3 — NP=4 ↔ NP=8 byte-identity

Sub-tasks mirror Stage 2 in shape. The drift signature (sub-F16 at layer 0) is distinct enough that op-level localization may name different ops than Stage 2; do not assume Stage 2 fixes carry to Stage 3.

### S3.1 — Per-prompt first-divergent-layer at NP=4 vs NP=8

8 prompts × 1 capture each (NP=4 vs NP=8 at slot K aligned by prompt). Hamming-diff per layer.

**Verify by**: `data/deltanet/s3-1-per-prompt-first-div.json`.

### S3.2 — Kernel-dispatch characterization (cheap hypothesis test)

Same approach as S2.2 — dispatcher trace at NP=4 vs NP=8. Compare to S2.2's NP=2 vs NP=4 dispatch logs; if Stage 3 introduces a fresh dispatch flip at a previously-stable call site, that's the new target.

### S3.3 — Op-level localization (CONDITIONAL)

Same approach as S2.3, only if S3.2 doesn't name the culprit precisely.

### S3.4 — Design replacement kernels

Same approach as S2.4. May reuse S2.4 work if same ops are implicated; do not assume so without evidence.

### S3.5 — Implement + unit test

Same approach as S2.5. Same harness, NP ∈ {4, 8}.

### S3.6 — Stage 3 binding

Byte-identity at NP=4 vs NP=8 on all 8 prompts × 3 reps. Stage 2 binding (NP=2 vs NP=4) must remain GREEN.

### S3.7 — Stage 3 perf binding

np=4 and np=8 aggregate. Same envelope-% gate as S2.7.

## Stage 0 — NP=1 ↔ NP=2 (deferred)

Closes the F32-vs-F16 storage-path delta at single-token decode. Options:
- Force F16 residual at np=1 (small overhead, no perf regression expected).
- Make np=1 use the same multi-token kernel path as np=2 (n_tokens=1 path becomes a special-case of the general path).

Deferred until Stages 1–3 close. May not need any fix beyond a configuration knob.

## Composite verification gate

All must hold before this PLAN.md archives:

1. **Stage 1 GREEN** — NP=2 self-deterministic.
2. **Stage 2 GREEN** — NP=2 ↔ NP=4 byte-identical per-prompt per-rep + perf binding.
3. **Stage 3 GREEN** — NP=4 ↔ NP=8 byte-identical per-prompt per-rep + perf binding.
4. **DFlash np=1 output preserved exactly** — pre-PLAN baseline retained.
5. **MTP np=1 output preserved exactly + MTP np>1 deterministic across reps**.
6. **Existing T9.1 5 validity asserts GREEN at all np** (no PPL band violation, no NaN, no decode failure, no vocab glitch).
7. **No regression in `test-backend-ops`**.
8. **Production health check (`bash healthcheck.sh`) GREEN.**

Stage 0 (NP=1 ↔ NP=2) is bonus, not gating. Determinism-without-perf is not a ship state.

## Critical files

**Read** (D1/D2 done; remaining stages reference these):
- `ik_llama.cpp/ggml/src/ggml-cuda/delta-net.cu`, `.cuh` — DeltaNet recurrence kernel
- `ik_llama.cpp/ggml/src/ggml-cuda/per-step-restore.cu` — DeltaNet state restore
- `ik_llama.cpp/ggml/src/ggml-cuda/fattn-*.cu`, `.cuh` — FA kernels
- `ik_llama.cpp/ggml/src/ggml-cuda/mmq.cu`, `mmvq.cu` — quantized matmul
- `ik_llama.cpp/src/llama-delta-net.cpp`, `.h` — DeltaNet C++ plumbing
- `ik_llama.cpp/src/graphs/build_qwen35.cpp` — layer-by-layer graph build (l_out emission site)
- `ik_llama.cpp/include/llama.h` — `llama_set_dflash_extract_layers` / `llama_get_dflash_extract_data` (cap 80 after D1 bump)

**Modify** (Stage 2/3 implementation):
- Kernel files named by S2.2/S2.3 / S3.2/S3.3 op-level localization
- `ik_llama.cpp/src/graphs/build_qwen35.cpp` — integration if new kernels need build-graph plumbing
- `ik_llama.cpp/tests/dflash-speculative/test-np-validity-vanilla.cpp` — extend with byte-identity asserts
- `ik_llama.cpp/ggml/CMakeLists.txt` — build flag if needed

**Create**:
- `ik_llama.cpp/tests/test-hybrid-batch-invariance.cpp` — Stage 2/3 unit test harness
- New `.cu` kernel files for replacements
- `specs/deltanet/batch-invariance.md` — Stage 2/3 spec
- `PHASE_DELTANET.md` (yarn-agentic top level) — once this PLAN.md archives
- Per-stage `data/deltanet/sN-*.json` evidence files

## Out of scope

- **Vulkan backend.** Production is CUDA on sm_75 dual TU102.
- **Mamba / other recurrent architectures.** Scope is Qwen 3.5/3.6 hybrid only.
- **Quantization-level changes** to the target model.
- **DFlash kernel optimization to TU102 envelope** — downstream of Stage 3 closure.
- **DFlash multi-slot libllama API extension** — downstream.
- **MTP perf optimization at np>1** — downstream.
- **Cross-host parity** — different problem; intra-host determinism only.

## Change discipline

- Plan + spec + MEMORY commit + push separately per CLAUDE.md §5/§6.
- No phase/stage/D-N nomenclature in source files per `feedback_no_host_concerns_in_code`. Feature names only (`batch_invariant`, `deterministic`, etc.).
- Test-first per `feedback_test_first_discipline`.
- Surface spec deviations before writing code per `feedback_surface_tradeoff_decisions`.
- Survey prior infrastructure per `feedback_survey_prior_phase_before_new_mechanism`. The D1 harness, the extract API, and the T9.1 validity bind are all reusable.

## Risk surface (each treated as a task, not a hedge — per `feedback_no_risks_only_tasks`)

- **Drift sites span multiple ops across the pipeline.** Multi-prompt D2 already showed this. Stage 2 must address all named ops, not just one. Treating any single op as THE source repeats the p0-only mistake.
- **DeltaNet recurrence DOES matter for some prompts.** p3 evidence binds. Fix carries the full TML recipe in DeltaNet's `delta-net.cu` cross-warp reduction (lines 137–147), Q/K/V projection paths, AND input RMSNorm.
- **FA kernel template at decode shapes is non-trivial.** Stage 2 may require a bespoke fixed-tile FA kernel for sm_75 — overlap with DFlash kernel-design `§6.3` (the bespoke verify-attn already scoped). Reuse or extract.
- **Stage 3 distinct from Stage 2.** Sub-F16 drift at layer 0 is a different mechanism than the visible-F16 drift Stage 2 fixes. Do not assume Stage 2 work auto-fixes Stage 3.

## Estimated cost (per CLAUDE.md §8 — tokens not days)

| Task | Estimate |
|---|---|
| S1.1 (multi-rep NP=2 capture) | 5–8k |
| S1.2 (closure or fix) | 0–20k (probably 0) |
| S2.1 (extend p2,p4–p7 D2) | 5–8k |
| S2.2 (op-level localization at first-div layers) | 15–25k |
| S2.3 (dispatch characterization) | 10–20k |
| S2.4 (design spec) | 15–25k |
| S2.5 (implement + oracle + unit) | 30–50k |
| S2.6 (Stage 2 byte-identity binding) | 20–35k |
| S2.7 (Stage 2 perf binding) | 10–15k |
| S3.1–S3.7 (mirror Stage 2) | 60–115k |
| Hygiene (PHASE, MEMORY) | 5–10k |
| **Total** | **~175–330k** |

Bold-on-design (commit to TML batch-invariance recipe + staged approach). Measured-on-diagnosis (multi-prompt D2 already overturned an early hypothesis; carry that discipline through op-level work).

## Pickup state

Stage 1, S2.1, S2.2, S2.3 (incl. n_kv-pad confirmation) all CLOSED 2026-05-14. Allium + TLA+ specs locked. Ready for **S2.4 design (SoTA sm_75 FA replacement kernel)**.

### Findings summary (2026-05-14)

**Mechanism definitively identified:** the FA kernel's K-loop iteration count is determined by the BATCH-MAX KV occupancy (`ne11 = K->ne[1]`), not the per-slot valid range. Mask-zero attention to extra K positions produces fp32-eps roundoff in the online softmax that amplifies through subsequent GEMMs to F16-visible drift.

n_kv-pad confirmation test (`data/deltanet/s2-3-nkvpad-confirmation.json`): when all slots load the SAME prompt (per-slot KV content identical), slot-0 chain is byte-identical from `kqv_wo` onward. Residual fp32-eps variation at `flash_attn-1003` (2.4e-7) is sub-F16; rounded away downstream.

Spec closures (`specs/deltanet/batch-invariance.allium`):
- `FA_NkvIsDominantBatchShapeEntryPoint` — n_kv is the dominant source
- `MMQ_FaithfulPropagation` — MMQ propagates noise faithfully; not a fresh source

### S2.4 (next): SoTA sm_75 FA replacement kernel spec

Per `feedback_kernel_replacements_must_be_sota_sm75` — the replacement spec MUST include:
1. State-of-the-art sm_75 design (Turing `mma.sync.m16n8k8`, `ldmatrix.sync`, 64KiB SMEM, 65536 regs)
2. Fully worked-out register allocations from `--ptxas-options=-v`
3. Profiling data baseline (nsys + ncu) committed to repo
4. Performance contract: % of TU102 ceiling (HBM 624 GB/s, fp16 tensor-core 130.5 TFLOPs per-GPU)

Template: `specs/dflash/kernel-design.md §6.1, §6.2, §6.3, §6.6`. Each has full register budgets, SMEM allocations, occupancy targets, and architecture-peak bindings. Mirror that structure.

**Specific FA fix:** the K-loop at `fattn-wmma-f16.cuh:180`:
```cpp
for (int k_VKQ_0 = ip*FATTN_KQ_STRIDE; k_VKQ_0 < ne11; k_VKQ_0 += parallel_blocks*FATTN_KQ_STRIDE)
```
Replace `ne11` with `ne11_for_slot_K = (slot K's actual KV occupancy, padded to FATTN_KQ_STRIDE)`. Combined with per-row CTA (TML recipe — eliminates fragment cross-column aggregation), this addresses the dominant entry point empirically.

MMQ replacement is OPTIONAL for batch-invariance (closed via MMQ_FaithfulPropagation). If MMQ is replaced for perf per the SoTA mandate, it must preserve byte-identity propagation.

### First reads for resuming session

The kernel work has progressed past spec-locking. Current state on 2026-05-14:

**Completed and committed:**

1. `specs/deltanet/fattn-per-slot-kv-sm75.md` — locked spec (all 8 OQs resolved)
2. `ik_llama.cpp/tests/dflash-speculative/fattn-per-slot-kv-sm75-reference.h` — scalar fp32 oracle (S2.5.a)
3. `ik_llama.cpp/tests/dflash-speculative/test-fattn-per-slot-kv-sm75.cpp` — RED unit test (S2.5.b) covering ~256 configs × 3 scenarios (oracle-vs-kernel, rep-determinism, batch-invariance)
4. `ik_llama.cpp/tests/dflash-speculative/fattn-per-slot-kv-sm75-stub.cpp` — temporary stub returning -1, replaced when the real launcher lands
5. `data/deltanet/perf/baseline/llama-bench-shapes.json` (and `SUMMARY.md`) — pre-merge wmma_f16 wall-clock at NP=1 across prefill 16/32/128/512/1024/2048 and decode 1/4/8 tokens. Speed-of-light numbers: tg1 ≈ 29.7 t/s, tg4 ≈ 30.6 t/s, tg8 ≈ 31.3 t/s, pp512+ saturates at ~385 t/s. **Caveat**: this baseline uses F16 KV cache; production uses Q4_0 + Hadamard rotation (see SUMMARY.md warning). Re-capture at production KV config is a pre-merge gate.
6. `examples/llama-bench/llama-bench.cpp` fix: `--tensor-split 1,1` now parses correctly (was sending all weight to GPU 0; fixed at commit c267962)
7. `data/deltanet/perf/baseline/enable-gpu-profiling.sh` — sudo script to enable non-root ncu profiling (NVreg_RestrictProfilingToAdminUsers=0). Applied 2026-05-14; survives reboot.

**Pre-implementation perf-floor captures (sequencing corrected 2026-05-14):**

Original G1→G2→G3 sequencing flipped: patching the measurement harness before
the measurement is methodologically backwards (patch becomes a suspect for any
anomaly in the measured numbers). Corrected sequence is two zero-/minimal-
source-touch captures FIRST, then llama-bench ergonomics patch lands as a
post-implementation regression-test surface (not a measurement prereq).

- **`P1` — nsys on production llama-server**: zero source touch. Wrapper
  `nsys profile` around the running `profiles/qwen36.sh` server; drive a fixed
  HTTP prompt; extract FA kernel wall-clock range. This is the literal
  production path — already at Q4_0 + Hadamard via server cparams. Output:
  `data/deltanet/perf/baseline/nsys-llama-server-prod.nsys-rep`.
- **`P2` — ncu on minimally-patched test-np-validity-vanilla**: 2-LOC patch
  (flip `cparams.k_cache_hadamard = cparams.v_cache_hadamard = true` — existing
  fields, production-supported, no new code). Q4_0 KV footprint (~4 GiB) +
  fixed-shape test binary fits in VRAM for ncu kernel-replay snapshot. Captures
  detailed FA kernel metrics: registers/thread, SMEM bytes/block, SM throughput
  %, memory bandwidth %. Output: `data/deltanet/perf/baseline/ncu-fa-vanilla-prod.ncu-rep`.
- **`G1` (post-S2.5 follow-up, NOT a measurement prereq)** — llama-bench
  Hadamard flag support: `-khad / -vhad` + env-var reads, mirror
  `common.cpp:1666-1672` flag-parse + `common.cpp:513-514` env-var. Lands AFTER
  the new kernel exists, as a regression-test ergonomic so future sweeps can be
  driven through llama-bench at production KV config.

P1 + P2 together establish the perf floor the replacement must match or beat.
S2.5 kernel implementation gates on these landing.

**S2.5 kernel implementation — Phase 2 progress (updated 2026-05-14):**

Spec correction landed (Dv 128→256 per GGUF metadata; KV_BLOCK_SIZE primary
flipped 32→16). Per Q&A (b/cosine/d/6c/3c/7b): raw PTX path, COSINE bind,
skeleton-first incremental, full closure 6c, Approach C multi-head packing,
3-stage incremental within Stage 2.2.

Phase 2 sub-stages completed (each 464/464 GREEN across all 3 test scenarios):

| Stage | Commit | Change |
|---|---|---|
| Phase 1 | `3b32170e` | naïve scalar device kernel + launcher (skeleton) |
| Stage 2.1 | `dc024801` | mma.sync.m16n8k8 inner dot product (1-warp) |
| Stage 2.2a | TBD | 4-warp CTA + partial-D SMEM reduction |
| Stage 2.2b | TBD | Approach C multi-head decode packing (H=gqa=6 rows) |
| Stage 2.2c | TBD | Q tile in SMEM (once per CTA) |
| Stage 2.2d.1 | TBD | K block in SMEM (per K-iter cooperative load) |
| Stage 2.2d.2 | TBD | V block in SMEM (per K-iter cooperative load) |
| Stage 2.2d.3 | TBD | ldmatrix.sync.aligned.m8n8.x2.b16 for B-fragment loads |

All variants accessible via `FATTN_KERNEL_VARIANT` env (phase1, stage21,
stage22a, default → latest with Approach C decode pack + ldmatrix).

Phase 2 remaining (Phase 2 closure 6c):

- **Stage 2.2d.4 (deferred)**: SMEM swizzle for bank-conflict-free ldmatrix —
  pure perf, no correctness blocker.
- **Stage 2.3**: parallel_blocks split-K + custom combine kernel — required
  for SoTA perf at decode NP=1 (current pb=1 gives only 4 grid CTAs at
  NP=1 vs wmma_f16's 96, massively underutilizing GPU). The existing
  `flash_attn_combine_results` doesn't appear to handle n_seqs > 1 (blockIdx.z
  unused in its body — would write multiple slots to slot-0's output).
  Custom combine needed.
- **Production dispatcher wiring** at `fattn.cu:140` — needs a NEW launcher
  accepting device-side ggml_tensor pointers (current launcher is
  HostHalf-based). Also needs `slot_seq_lens` plumbed as a new ggml input
  tensor per spec OQ-4 — build-graph change in `src/graphs/build_qwen35.cpp`.
- **`test-np-validity-vanilla` binding** at NP={2, 4, 8} — proves end-to-end
  determinism fix on production target.
- **nsys/ncu** vs `data/deltanet/perf/baseline-prod/` — SoTA perf binding.

**Original S2.5 spec deliverables status:**

Replace the stub `fattn_per_slot_kv_sm75_launch` with the real kernel in
`ik_llama.cpp/ggml/src/ggml-cuda/fattn-per-slot-kv-sm75.cu` per the spec.
Per-row CTA + multi-head packing, Turing `mma.sync.m16n8k8` PTX, fp32
accumulators, per-slot K-loop bound. 4 template instantiations:
KV_BLOCK_SIZE ∈ {32, 64} × USE_SOFTCAP ∈ {false, true}.

The kernel must accept **Q4_0** K and V tensors (with inline dequant to fp16
before mma — mirror `fattn-mma-f16.cu:102-105`'s `ggml_get_to_fp16_cuda` call).
Hadamard rotation is applied at the build-graph level via cparams; the kernel
receives post-rotation Q4_0 inputs and does not handle the rotation itself.

First reads for the kernel implementation session:

1. `specs/deltanet/fattn-per-slot-kv-sm75.md` — the spec to implement
2. `ik_llama.cpp/ggml/src/ggml-cuda/fattn-wmma-f16.cuh` — the kernel being replaced (reference layout)
3. `specs/dflash/kernel-design.md §6.3` (dflash_verify_attn) — same per-row-CTA pattern; structural ancestor
4. `data/deltanet/perf/baseline/SUMMARY.md` — perf floor the replacement must match or beat
5. `data/deltanet/perf/baseline/ncu-fa-decode.ncu-rep` (after ncu re-capture) — wmma_f16's measured registers/thread, SMEM, occupancy, SM/memory throughput pct-of-peak
6. The "ssiu/flash-attention-turing" public sm_75 FA reference — only public FA implementation tuned for sm_75 head_dim=128 at ~63% peak on T4. Layout reference for the per-row CTA replacement.

When the real launcher lands:
- Swap `fattn-per-slot-kv-sm75-stub.cpp` out of the `test-fattn-per-slot-kv-sm75` add_executable in `tests/CMakeLists.txt`
- Add `fattn-per-slot-kv-sm75.cu` to the ggml-cuda library build
- Wire dispatcher at `fattn.cu:140` to route (HEAD_DIM_Q=256, HEAD_DIM_V=128) tuples through the new kernel
- Tests transition RED → GREEN; perf measurement (S2.5.d) gates the merge

---

## Pickup state (revised 2026-05-15 — reconciliation)

### TL;DR for resuming session

S2.5 kernel implementation has been built and iterated through six designs without keeping the spec aligned. The current empirical state is documented in `specs/deltanet/fattn-per-slot-kv-sm75.md §15` (lifecycle reconciliation). **Production decode runs wmma_f16 by default**; the new op is opt-IN via `LLAMA_FATTN_PER_SLOT_KV_ENABLE=1`. **Determinism contract is NOT delivered in production**.

Three forward paths are open and need a decision before any more kernel code lands:

- **P1**: Accept the perf gap. Ship determinism via opt-in env. Requires Option A plumbing (per-row `slot_seq_lens` indexed by `Q->ne[1]`) to actually deliver the fix at NP>1. Production-default stays wmma_f16.
- **P2**: Abandon this kernel. Modify wmma_f16 in-place with `slot_seq_lens` arg + K-loop bound. ~5 LOC kernel + Option A plumbing for per-row bound. Preserves wmma_f16's perf tuning.
- **P3**: Defer determinism work indefinitely. Production stays as-is.

### What's in `production/2026-q2-next` right now

- `GGML_OP_FLASH_ATTN_EXT_PER_SLOT_KV` ggml op exists (commit `64d2a1f3`)
- Dispatcher hook at `fattn.cu:140` (commit `f5456863`)
- Build-graph emits the op for Qwen 3.5/3.6 shape, gated by `LLAMA_FATTN_PER_SLOT_KV_ENABLE=1` (commits `cef48724` + `780657b2`)
- Kernel (Approach A single-head split-K + SoTA SMEM redesign), commit `63842621`. 183 µs/call vs wmma_f16's 33.7 µs (5.4×).
- Unit test `test-fattn-per-slot-kv-sm75` 464/464 GREEN (algorithmic batch-invariance)
- NP={2,4,8} validity GREEN via `test-np-validity-vanilla` but at the legacy stage22a path (not the new SoTA kernel) and without functional per-slot K bound

### Empirical perf ladder

See `specs/deltanet/fattn-per-slot-kv-sm75.md §15.2`. Per-call wall-clock at long-ctx NP=1:
- Approach C (original spec): 408 µs
- Approach A single-head: 339 µs
- Decode-specific 1-row state: 296 µs
- SoTA SMEM redesign: 183 µs ← current default for the new op
- wmma_f16 reference: 33.7 µs

### Stashed work

git stash entry: "2-warp + MAX_PB=8 (post-SoTA-redesign, uncommitted, pre-reconcile)". Reduces NWARPS from 4 to 2 + caps MAX_PB at 8. Net effect on perf untested at the time of stash.

### What NOT to do without explicit user direction

Per `specs/deltanet/fattn-per-slot-kv-sm75.md §15.5`:
- No more kernel-design iteration without spec update first
- No path choice (P1/P2/P3) without explicit user decision
- No production-routing change without P-choice

---

## Pickup state (revised 2026-05-15 — Path P2 locked, executing NP>1 determinism)

### TL;DR

User picked **P2** (modify `wmma_f16` in-place to take a per-row K-loop bound). Spec change committed as `§15.6 Path P2 LOCKED`. Now executing the 7-step delivery plan. Performance work on `fattn-per-slot-kv-sm75.cu` kernels is paused / retired.

### Goal binding

Closure: at `LLAMA_FATTN_PER_SLOT_KV_ENABLE=1`, production server produces byte-identical slot-0 token sequences across NP ∈ {1, 2, 4, 8} given the same prompt. Empirically measured on Qwen 3.6 27B production target with Q4_0 + Hadamard KV. End-to-end production binding (not just a kernel unit-test argument).

### 7-step plan (per spec §15.6)

1. **ggml op signature** — `src[5]` length becomes `q->ne[1]` (n_tok), one per query row. Update ggml.c assertion + builder. Semantic rename in spec already done in §15.6.
2. **Build-graph** — `build_std_attention` emits `inp_slot_seq_lens` of length `q->ne[1]` (was `q->ne[3]`). Decoder struct field renamed `inp_per_row_k_bound`.
3. **llama.cpp populate** — for each token in batch: lookup seq_id → kv_self->seq_pos_max(seq_id) → write per_row_k_bound[i] = pos_max + 1.
4. **wmma_f16 kernel** — accept nullable `const int * per_row_k_bound`. K-loop: `n_kv_eff = bound ? min(ne11, bound[my_q_row]) : ne11`. Plumb through `launch_fattn`.
5. **Dispatcher** — when new ggml op is encountered, call wmma_f16 with the bound pointer. Default callers pass nullptr (no behavior change).
6. **Retire `fattn-per-slot-kv-sm75.cu`** — add DEPRECATED comment block; remove from production dispatch routing. Keep compiled for unit test backing.
7. **NP-cross byte-identity harness** — drive production server at NP={1,2,4,8} with identical prompt; compare slot-0 token sequences byte-for-byte across NP.

### Closure verification (composite)

- (a) Production smoke OK at default routing (no regression).
- (b) NP-cross byte-identity GREEN at NP={1,2,4,8} on production prompts under env-on.
- (c) 3-run intra-NP reproducibility GREEN at each NP.
- (d) Unit test 464/464 stays GREEN (kernel-level invariance preserved as reference).
- (e) wmma_f16 default-routing perf preserved (no measurable regression on baseline benchmarks).

### What this plan does NOT do

- No further perf work on the per-slot kernels (retired).
- No SoTA-from-scratch redesigns.
- No DFlash / MTP overlay work — strictly the np>1 determinism path on the production engine.

### What's in `production/2026-q2-next` right now (post-spec-commit)

- All §15.6 spec content committed and pushed.
- Tree state at the production submodule: unchanged from prior reconciliation pickup. The bespoke kernels still route under env-on; Step 5 below will repoint dispatch to `wmma_f16+bound`.

---

### Empirical results (2026-05-15 end of session)

Steps 1–6 + 6.5 all committed and pushed. Build clean. Step 7 status:

- **Kernel-level binding GREEN** — `test-fattn-per-slot-kv-dispatch-np-invariance` PASS (6144 fp32 floats byte-identical across K-cache strides {256, 512}).
- **Server-level binding RED** — `test-fattn-per-slot-kv-np-determinism.sh` fails: NP>1 same-prompt outputs diverge from NP=1 baseline.

Diagnostic probe (`scripts/probe-np-determinism-sources.sh`) findings:
- E1: NP=1 3-run reproducibility PASS.
- E2: NP=4 SEQUENTIAL 3 requests → 2/3 match NP=1 baseline byte-identical; 1/3 diverges ("foundation" vs "groundwork" at token ~15). Even without concurrent batching, NP=4 setting produces a per-request split.
- E3: NP=4 CONCURRENT → 3 batched slots produce byte-identical output to each other (≠ NP=1); 1 solo-scheduled slot matches NP=1. The intra-batch agreement validates FA kernel-level NP-invariance at the server.

Remaining divergence is upstream of FA + non-FA shape-dependent ops (Q/K/V matmuls, RoPE, RMSNorm) and possibly cgraph cache warm-up state at NP>1. See `specs/deltanet/fattn-per-slot-kv-sm75.md §15.8 / §15.9`.

**Step 7 stays OPEN per CLAUDE.md §4 "no follow-up cover"**. The FA piece of NP determinism is delivered; server-level byte-identity at NP>1 requires shape-independence in non-FA ops too — a separate workstream that needs explicit scoping before re-opening.

---

### Second iteration (2026-05-15 session continuation, /loop fired)

Diagnostic probes localized the FA-vs-non-FA divergence boundary:

- `probe-cgraph-effect.sh` — disabling CUDA graph capture (`GGML_CUDA_DISABLE_GRAPHS=1`) does NOT fix divergence. Not graph state.
- `probe-slot-pin.sh` — explicitly pinning slot IDs at request time PASSES (slot 0 + slot 1 both match NP=1, byte-identical) when each slot is the FIRST one used on a fresh server. Not slot-position dependence per se.
- `probe-cache-leakage.sh` — slot 1 AFTER slot 0 has run **diverges**; slot 0 AFTER slot 1 has run **matches NP=1**. Asymmetric. Established that "cache state from a prior request affects subsequent slot's output".
- `probe-chunk-alignment-hypothesis.sh` — slot 1 at different offsets (modulo FATTN_KQ_STRIDE) produces the SAME divergent output. Rules out fp16-warp-reduce-thread-position hypothesis at the simple level.

FA-side iterations attempted (all in `specs/deltanet/fattn-per-slot-kv-sm75.md §15.10`):
1. wmma_f16-pb1<256,256,8,half>: warm-slot-1 DIVERGES (fp16 warp_reduce_sum non-associative).
2. wmma_f16-pb1<256,256,8,float>: warm-slot-1 MATCHES; concurrent batched-decode DIVERGES (fp16 frag_c_VKQ inside mma).
3. multi_row_kernel (per-row CTA + fp32 throughout + full ne11 with mask): warm-slot-1 STILL DIVERGES.

The §15.10/3 attempt should have been bulletproof at FA correctness level. That it still diverges proves the gap is NOT in FA.

### Next iteration plan (for /loop pickup)

Per `specs/deltanet/fattn-per-slot-kv-sm75.md §15.11`: the residual divergence is in non-FA ops. To make further progress on "complete determinism":

1. Build a cb_eval-based probe that captures per-layer residual streams for slot 1 at warm vs fresh cache state. Use existing `examples/dflash-extract/` infrastructure as scaffolding.
2. Compare layer outputs byte-for-byte. The FIRST divergent layer names the upstream culprit.
3. Within that layer, capture op-level outputs (existing infra: cb_eval matches by tensor name suffix). Localize WHICH op (matmul / RoPE / RMSNorm / etc) first produces different output.
4. Fix THAT op the way we fixed FA: identify shape dependence → force NP-independent dispatch.
5. Repeat until cache-leakage probe + NP-cross harness PASS.

This is potentially a long sequence (each non-FA op may have its own determinism breaker). Per CLAUDE.md §8 estimate-in-tokens: each op fix ~30k–80k tokens. Could be 10+ ops × 50k = 500k+ tokens of work for full coverage.

The /loop is set to fire every 4 minutes; each iteration can target one op localization + fix.

---

### Third iteration (session continuation, /loop fire 2 — multi_row → wmma fp32)

Hypothesis test for multi_row: kernel claimed to be more NP-invariant than wmma but EMPIRICALLY failed cache-leakage probe. Has its own bugs.

Reverted production route back to `wmma_f16-pb1<256, 256, 8, float>`. Verified:
- `probe-cache-leakage.sh`: ALL 4 configs byte-identical.
- `probe-slot-pin.sh`: slot 0 = slot 1 = NP=1, byte-identical.
- `test-fattn-per-slot-kv-dispatch-np-invariance`: PASS.

Remaining gap: concurrent batched-decode (ne[1]>1 with multiple slots in one FA call). Argmax flips after a few decode steps. Sourced from fp16 frag_c_VKQ inside the wmma mma instruction.

Tried `--ubatch-size 1`: broke model output entirely (looped "the the the"). `--ubatch-size 16`: still diverged but more consistent (intra-batch agreement at lower NP). The ubatch tweak doesn't actually prevent concurrent batched-decode at runtime.

Step 7 marked `[~]` (partial) per CLAUDE.md §5: sequential NP-cross GREEN, concurrent batched-decode RED, gap named and tracked. See `specs/deltanet/fattn-per-slot-kv-sm75.md §15.13` for the full scope summary and the three possible next-workstream paths:
1. Add fp32-`frag_c_VKQ` wmma variant + parallel template family.
2. Find and fix the multi_row_kernel cache-leakage bug.
3. Server-config to actually disable concurrent batched-decode (deeper than `--ubatch-size`).

---

### Fourth iteration (session continuation, /loop fire 3 — --no-cont-batching probe)

Tested `--no-cont-batching` server flag. Result: partial improvement, not full closure.
- NP=2: slot 1 byte-identical to NP=1. Slot 0 diverges.
- NP=4: 3 distinct outputs across slots.
- NP=8: 4 distinct outputs.

Server state (CUDA graph cache, cuda_pool_alloc returns, scheduling order) varies per request order even with cont-batch disabled. The "first" slot processed has different state than "second", which leaks into FA inputs and decode outputs.

Captured in spec §15.14 / §15.15.

### Next iteration plan (for /loop fire 4 — cb_eval layer localization)

The current `[~]` closure is honest but the user-facing "complete determinism" goal isn't met. To make further progress, the highest-value next step is cb_eval-based per-layer diff:

1. Reuse `examples/dflash-extract/llama-dflash-extract` (uses cb_eval to capture residuals at named layers).
2. Run prefill for slot 0 (NP=1 config) → dump per-layer residuals at all 60+ layers.
3. Run prefill for slot 0 with NP=4 server config → dump per-layer residuals (same prompt, different server config).
4. Compare residuals at each layer. The FIRST layer with non-identical output names the upstream culprit.
5. Within that layer, capture op-level outputs (cb_eval matches by tensor-name suffix). Find WHICH op (matmul / RoPE / RMSNorm / attention output projection) first diverges.
6. Fix that op like we fixed FA: identify shape dependence → force NP-independent dispatch.

Cost estimate per CLAUDE.md §8: ~30k tokens for steps 1-5 (instrumentation + comparison). +30-80k per op fix. Could be 3-10 ops × 50k = 150-500k tokens for full coverage.

If cb_eval comparison reveals the divergence at layer 0 RoPE: that's the smallest local fix.
If at layer 0 K_proj matmul: changes spread across multiple matmul kernels.
If at layer 5+ deep: more cascading concerns.
