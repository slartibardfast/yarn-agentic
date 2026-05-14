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

### S2.4 — Design replacement kernels (with perf contract)

For each op needing replacement, document in `specs/deltanet/batch-invariance.md` (new spec file):

- New kernel grid + block geometry (per-row CTA, fixed compile-time tile dims).
- Reduction strategy (warp-shuffle inner; SMEM-tree across warps; no cross-block atomics on float).
- Numerical contract: scalar fp32 oracle behavior to bind byte-identity.
- **Perf contract**: per-call latency target derived from TU102 ceiling — HBM bandwidth (624 GB/s per GPU, 1.25 TB/s aggregate) for memory-bound ops, fp16 tensor-core peak (130.5 TFLOPs per GPU, 261 TFLOPs aggregate) for compute-bound ops. Target cited as % of relevant ceiling, work amount shown.
- **Tensor-core usage**: any GEMM-shaped op MUST use Turing `mma.sync.m16n8k8` PTX (or WMMA m16n16k16 fp16) at our shapes. Deviations require explicit justification with perf cost in the spec.

**Verify by**: spec file committed with perf contract section explicit.

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

D1 + S2.1 (3 of 8 prompts) closed 2026-05-14.

Next work: **complete S2.1 — capture p2, p4–p7** (5 more NP=2 captures at offset=2..6, then per-prompt first-divergent-layer analysis vs NP=4 offset=0). Cheap, ~5–8k tokens, finishes the multi-prompt D2 evidence.

Then **S2.2 — op-level localization at each prompt's first-divergent layer**. This is where the bulk of Stage 2's diagnostic value lives: which specific kernel call diverges first at p3 layer 0, at p1 layer 3, at p0 layer 19? The answer drives S2.4 design scope.

First reads for resuming session:
1. `data/deltanet/d2-first-divergent-layer.json` + `d2-multi-prompt-update.json` — current evidence.
2. `ik_llama.cpp/tests/dflash-speculative/test-deltanet-d1-capture.cpp` — the harness; S2.1 just re-runs with new offsets, S2.2 extends with intermediate-tensor matching.
3. `ik_llama.cpp/src/graphs/build_qwen35.cpp` — layer-by-layer ops named by `cb()` calls. The cb names are what S2.2's cb_eval matches on.
4. `ik_llama.cpp/src/llama.cpp:9608-9676` — the `llama_dflash_extract_cb_eval` matcher. Extension for S2.2: match more names than just `l_out-<il>`.
