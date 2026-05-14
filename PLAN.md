# DeltaNet np>1 determinism — production lockdown

## Context

Vanilla decode on Qwen 3.6 27B (hybrid `linear_attn` + `full_attention` arch) is non-deterministic at np>1. Locked findings from prior work:

- **PHASE45 D10.e** characterized the framework (cuBLAS heuristic shape-sensitivity, FA split-size variation, per-row reduction-order drift) but **D10.e.2** (the planned 3-kernel reduction-order rewrite) did not ship. See `docs/phases/60-llama-context-decompose/PHASE45.md`.
- **DFlash T9.1** added direct token-vs-NP=1 diff for vanilla at np ∈ {2,4,8} on 8 production prompts × 64 tokens. Signature is **specific, not general**:
  - NP=1 ≡ NP=2 byte-identical.
  - NP=4 ≡ NP=8 deterministic drift, identical first-divergence positions across both np values for 3 of 4 prompts (p0:tok3, p1:tok19, p3:tok0).
  - One outlier (p2) picks up an additional NP=8-only drift past token 50.
  - Binary boundary at NP=2 → NP=4, not accumulating with batch width.
  - Data: `data/phase_dflash_t8/gate7-token-diff-summary.json`, `gate7-validity-vanilla-np{1,2,4,8}-*.json`.
  - Harness: `ik_llama.cpp/tests/dflash-speculative/test-np-validity-vanilla.cpp`.

**Why this blocks everything downstream**: DFlash and MTP speculative overlays both require a deterministic target forward at the np they ship at. Any spec-decode method built on a drifty target is structurally broken — accept-rate inflation, mid-sequence trajectory divergence, irreproducible bug reports. Production-quality np>1 ship paths cannot exist until this is fixed.

This plan replaces all prior DFlash multi-slot / DeltaNet optimization work. DFlash kernel optimization, MTP multi-slot, and DFlash overlay on np>1 are explicitly **downstream** of D7 closure here.

## Direction

The drift surface in the hybrid arch spans four ops, in priority order of suspicion:

1. **Full-attention FA path** (61 full_attention layers). Turing `mma_f16` PTX tile is `m16n8k8`; first batch dim multiple is 8. NP=4 may trip a tile-quantization or split-K transition vs NP=2.
2. **DeltaNet recurrence kernel** (`ggml/src/ggml-cuda/delta-net.cu`). Cross-warp reduction order depends on `block_size/WARP_SIZE`; the dispatch branches between `threads_per_block=256` (n_tokens ≤ 8) and `=128` (otherwise). Per-batch compute is grid-independent, but if upstream ops feed batch-shape-sensitive state into DeltaNet, drift propagates through the recurrence.
3. **cuBLAS GEMM dispatch** for Q/K/V/O projections and FFN. `cublasGemmEx` picks different algorithms via heuristic at different M dimensions; M scales with effective batch width.
4. **DeltaNet Q/K/V projections** specifically. `linear_attn.in_proj_a/b` weights are AutoRound-preserved BF16 — possibly different dispatch path than IQ4_KS Q4_0 paths.

Approach is **localize → characterize → replace per Thinking Machines Lab batch-invariance recipe** (per-row CTA, fixed compile-time tile, no Split-K, no atomicAdd<float> cross-block, warp-shuffle inner reductions, SMEM-tree block-level reductions).

**Performance is a co-equal objective with determinism, not a tradeoff to surface at the end.** Determinism-at-any-cost is rejected. The replacement kernels must hit the measured-baseline t/s at np=1 AND deliver np>1 aggregate uplift consistent with the TU102 + NVLINK envelope work named in `project_dflash_t8_closed`. Two reasons this is achievable, not aspirational:

1. **Heuristic dispatch IS a perf cost.** cuBLAS `cublasGemmEx` at small M (decode batch widths) routinely runs at <5% of TU102 peak because the heuristic picks general-purpose algorithms. Fixed-tile bespoke MMQ-style kernels with compile-time geometry and explicit tensor-core scheduling beat it at the shapes we care about. Removing heuristic dispatch removes a runtime branch AND opens the door to nvcc full-loop-unroll.
2. **The replaced kernels are the same ones DFlash T8 named as bound at ~1% of TU102 peak.** The `gemm_row_x_col_kernel` and `dflash_drafter_lm_head_kernel` analysis applies to vanilla projection GEMMs at decode shapes too — same Turing peak, same memory-bandwidth ceiling, same massive headroom. Determinism is not subtracting perf here; it is the lever that lets us claim the perf.

D5 design must include a measured perf target per replacement kernel. D8 is verification of that target, not discovery of regression.

The plan does **not** assume DeltaNet kernel is the sole source. The user-named "DeltaNet non-determinism" is treated as covering the recurrent-attention subsystem as a whole — including its projection inputs and the FA layers that interleave with it in the hybrid stack. D1–D4 localize concretely; D5+ scopes the fix to the specific ops the localization names.

## Tasks

Per CLAUDE.md §4: each task has a strong verification check. Per §5: PLAN.md + PHASE_DELTANET.md + MEMORY.md commit separately from code and from each other.

### D1 — Per-layer per-slot residual capture at np ∈ {1,2,4,8}

Extend the existing T9.1 vanilla validity harness (`test-np-validity-vanilla.cpp`) to emit, per slot per layer, the residual stream output (`l_out-<il>` post-residual-add) for the first generated token. Same prompts (p0–p7), same n_gen, but capture instead of compare.

Reuse the existing `cb_eval`-based extract hook landed during DFlash T2 (`llama_set_dflash_extract_layers` / `llama_get_dflash_extract_data` in `include/llama.h`) — extended to capture all 65 layers, not the 5 DFlash source-layer indices. Branch on `t->type` per `feedback_cb_eval_dtype_split` (F32 vs F16 ubatches).

**Verify by**: for each (np, slot, layer) triple, `data/deltanet/d1-np{1,2,4,8}-slot{i}-layer{l}.bin` exists; per-prompt manifest written to `data/deltanet/d1-manifest.json`. No comparison logic yet — D1 is pure capture infrastructure.

### D2 — Localize first-divergent layer

Compare D1 residuals across np values at slot N (where slot N exists in all configs, e.g., slot 0). For each layer l, compute hamming distance between `np=1, slot=0, layer=l` and `np=4, slot=0, layer=l` and find the smallest l where hamming > 0.

Repeat for np=2 (expect all l: hamming=0, per the T9.1 signature). Repeat for np=8 (expect divergence starts at same l as np=4 per the binary-boundary signature).

**Verify by**: `data/deltanet/d2-first-divergent-layer.json` names a specific layer index `l*` and the layer type at that index (`linear_attn` or `full_attention`). The result narrows the search surface from "the whole stack" to one layer type.

If D2 names a `linear_attn` layer → DeltaNet recurrence path is implicated.
If D2 names a `full_attention` layer → FA path is implicated.
If divergence appears at layer 0 → upstream input embedding or pre-layer-norm; widen scope.

### D3 — Localize first-divergent op within layer `l*`

Within layer `l*`, capture per-op output via `cb_eval` matching on intermediate tensor names. Candidate ops:

- `attn_norm.weight*x` (RMSNorm output)
- `Q.weight @ x`, `K.weight @ x`, `V.weight @ x` (projection GEMMs)
- For full_attention layers: FA kernel output (`KQV-<il>`)
- For linear_attn layers: DeltaNet kernel output (`delta_net_out-<il>`)
- `O.weight @ ...` (output projection GEMM)
- `ffn_norm.weight*x`, MLP gate/up/down
- Final `l_out-<il>` (post-residual)

Hamming-diff each across np=1 vs np=4 at slot 0. Smallest in-layer divergence wins.

**Verify by**: `data/deltanet/d3-first-divergent-op.json` names a specific op + tensor name. Result narrows surface from "the whole layer" to one op (typically: one CUDA kernel).

### D4 — Characterize the drift-introducing kernel

For the kernel named by D3, dump its dispatch parameters at np=1, np=2, np=4, np=8:

- For cuBLAS GEMMs: enable `CUBLAS_LOGINFO_DBG`, capture which `cublasGemmAlgo_t` is picked at each M dimension.
- For FA: instrument the dispatcher (`fattn-common.cuh` or equivalent) to printf chosen kernel template parameters (KQ_per_iter, head_dim, n_seq_block) at each (n_tokens, n_seq) tuple. Same n_tokens=1 across decode at every np; what changes is grid geometry.
- For DeltaNet: confirm `threads_per_block` selection (256 vs 128) is identical across np values for the same n_tokens; if it is, the kernel itself is invariant per-batch and drift originates upstream.

**Verify by**: `data/deltanet/d4-dispatch-trace.json` shows exact kernel/algo/tile picked at each np. The np=2/4 boundary should correspond to a visible dispatch difference. If no difference visible → drift is from atomicAdd, scheduling, or some other non-obvious mechanism; widen D4.

### D5 — Design batch-invariant replacement WITH explicit perf target

Document in `specs/deltanet/batch-invariance.md` (new spec file):

- The specific op(s) requiring replacement (from D3+D4).
- The new kernel's grid + block geometry (per-row CTA: one CTA per output row, fixed compile-time tile dims).
- The reduction strategy (warp-shuffle inner; SMEM-tree across warps; no cross-block atomics on float).
- The fixed cuBLAS algorithm (if replacing a GEMM call) or the new bespoke kernel signature (if replacing a heuristic-dispatched op).
- Numerical contract: scalar fp32 oracle behavior to bind byte-identity against.
- **Perf contract**: per-call latency target derived from the relevant TU102 ceiling — HBM bandwidth (624 GB/s per GPU, 1.25 TB/s aggregate) for memory-bound ops, fp16 tensor-core peak (130.5 TFLOPs per GPU, 261 TFLOPs aggregate) for compute-bound ops, or NVLINK aggregate (~100 GB/s) for cross-GPU comms. The target must be cited as a percentage of the relevant ceiling, with the work amount (bytes streamed or FLOPs computed) shown.
- **Tensor-core usage**: any GEMM-shaped op MUST use Turing `mma.sync.m16n8k8` PTX (or WMMA m16n16k16 fp16) at the shapes we care about, not scalar fp32. Deviations from tensor-core usage require explicit justification per `feedback_surface_tradeoff_decisions.md` and must include the perf cost in the spec.

**Verify by**: spec file committed to repo with the perf contract section explicit. Per `feedback_surface_tradeoff_decisions.md`: surface any deviation from TML's recipe in the spec, not silently in code.

### D6 — Implement replacement kernel + scalar oracle + unit test

Per `feedback_test_first_discipline.md`: write the scalar oracle and RED unit test first.

- `tests/test-deltanet-batch-invariance.cpp` — randomized inputs across (np ∈ {1,2,4,8}) × (seeds) × (head_dim, head_count variations).
- Scalar oracle: header-only fp32 CPU reference (mirrors `dflash-combine-features-reference.h` pattern from DFlash T3).
- Kernel: new file under `ggml/src/ggml-cuda/` if entirely new, or surgical edits to the existing kernel if drop-in.
- Build-time guarded behind `GGML_BATCH_INVARIANT_DELTANET=ON` (default ON when CUDA on).

**Verify by**: unit test passes byte-identity (or ≤ 1 fp16 ULP, matching DFlash T3 standard) at all configs. Existing test-backend-ops sweep stays GREEN.

### D7 — Integration: byte-identity binding on production model

Wire the replacement kernel into the build graph (`src/graphs/build_qwen35.cpp` and/or `build_qwen3next.cpp` depending on D3 location).

Extend `test-np-validity-vanilla.cpp` with a 6th assert: **byte-identical full 64-token sequence vs np=1 reference** at np ∈ {1,2,4,8} on 8 prompts × 3 reps each.

**Verify by** (this is the gate that closes the workstream):
- 32 slot-runs × 3 reps = 96 total runs. All 96 produce byte-identical first-64-token sequences vs the corresponding np=1 reference.
- Existing 5 T9.1 asserts remain GREEN (no validity regression).
- `data/deltanet/d7-byte-identity-binding.json` records all 96 hashes and the binding result.

This is the **definition of done** for the workstream. Closing requires this evidence; no `[~]` partial.

### D8 — Perf binding (not regression check) + flag default

Measure np=1 single-token decode tg t/s, np=4 aggregate tg t/s, AND np=8 aggregate tg t/s on the production benchmark (8 prompts × 3 reps each). Compare against:

1. **Pre-D6 baseline** captured at D1 start (current production state).
2. **D5 perf contract** (the % of TU102 ceiling targeted in the spec).
3. **Per-kernel nsys profile** to confirm each replaced op now runs within its memory-bandwidth or tensor-core peak envelope as designed.

**Verify by**: `data/deltanet/d8-perf-binding.json`. Acceptance (this is a positive binding, not a regression check):
- **np=1**: ≥ baseline tg t/s. A win here is expected, not just a non-loss — heuristic dispatch removal and tensor-core enforcement should push np=1 up, not down. Acceptable floor: equal to baseline within measurement noise (≤ 2%).
- **np=4, np=8**: ≥ baseline aggregate tg t/s, with per-slot t/s ≥ baseline-per-slot / 1.02 (no per-slot regression). Aggregate uplift consistent with the D5 perf contract.
- **Per-kernel envelope**: each replaced kernel hits ≥ 30% of its theoretical ceiling (HBM bandwidth or tensor-core peak), measured by nsys + arithmetic on the work amount. The DFlash T8 ~1%-of-peak baseline that two named kernels were stuck at is rejected as an acceptable end state.

If any binding fails: return to D5 design with the nsys evidence, not to D8 as "acceptable regression". Per `feedback_no_skipping_lessening.md`: do NOT declare a perf miss as "structural cost of correctness". The TML recipe is a recipe AND a perf framework — it gives us tensor cores AND determinism, not one at the expense of the other. Per `feedback_zero_waste_mantra.md`: 100% CPU, 100% memory bandwidth utilization, not one wasted or repeated byte.

### D9 — DFlash overlay re-bind

Re-run T9.1 harness with DFlash spec method enabled on production prompts at np=1. Verify pre-D6 DFlash output preserved exactly (no behavior change at np=1).

If DFlash multi-slot API extension lands (separate workstream — see PHASE_DFLASH future-work pointers), bind same byte-identity criterion at np>1.

**Verify by**: `data/deltanet/d9-dflash-rebind.json` shows DFlash np=1 output unchanged.

### D10 — MTP overlay re-bind

Re-run production MTP `--draft 3` benchmark (the current production path) at np=1. Verify byte-identical output vs pre-D6 baseline. Run at np>1; verify **deterministic across 3 reps** (not necessarily matching np=1, since MTP draft batching has independent shape concerns).

**Verify by**: `data/deltanet/d10-mtp-rebind.json` shows MTP np=1 output unchanged + np>1 reproducible.

### D11 — Plan and memory hygiene

Per CLAUDE.md §5/§6. Each commit separate, pushed immediately:
- `PHASE_DELTANET.md` at yarn-agentic top level — task tracker mirroring D1..D10.
- `docs/SUMMARY.md` entry pointing at the future-archived path.
- `MEMORY.md` (yarn-agentic): entry on D7 closure with the binding evidence, the named ops, the specific D3 finding (which layer + op was the cause).
- Auto-memory entry on D7 outcome (in `~/.claude/projects/-home-llm-yarn-agentic/memory/`) so future sessions know vanilla np>1 is locked.

On D7 closure: move `PLAN.md` → `docs/phases/80-deltanet-determinism/PHASE_DELTANET.md` per the README convention; update `docs/SUMMARY.md`.

## Verification (composite end-of-phase gate)

All must hold before this PLAN.md archives:

1. **D7 byte-identity binding GREEN.** 8 prompts × np ∈ {1,2,4,8} × 3 reps = 96 runs, all byte-identical vs np=1 reference. Evidence in `data/deltanet/d7-byte-identity-binding.json`.
2. **D8 perf binding GREEN.** np=1 ≥ baseline; np ∈ {4,8} aggregate ≥ baseline aggregate with no per-slot regression; per-kernel envelope hits ≥ 30% of theoretical peak. Evidence in `data/deltanet/d8-perf-binding.json`.
3. **D9 DFlash np=1 output preserved exactly.**
4. **D10 MTP np=1 output preserved exactly + np>1 deterministic across reps.**
5. **Existing T9.1 5 validity asserts GREEN at all np** (no PPL band violation, no NaN, no decode failure, no vocab glitch).
6. **No regression in `test-backend-ops` for DeltaNet, attention, or matmul ops.**
7. **Production health check (`bash healthcheck.sh`) GREEN.**

Determinism without perf binding (gate 1 GREEN but gate 2 FAIL) does NOT close this workstream. Both are required.

## Critical files

**Read** (no edits initially, used for D1–D4 localization):
- `ik_llama.cpp/ggml/src/ggml-cuda/delta-net.cu`, `.cuh`
- `ik_llama.cpp/ggml/src/ggml-cuda/per-step-restore.cu`
- `ik_llama.cpp/ggml/src/ggml-cuda/fattn-*.cu`, `.cuh`
- `ik_llama.cpp/ggml/src/ggml-cuda/mmq.cu`, `mmvq.cu`
- `ik_llama.cpp/src/llama-delta-net.cpp`, `.h`
- `ik_llama.cpp/src/graphs/build_qwen35.cpp`, `build_qwen3next.cpp`
- `ik_llama.cpp/include/llama.h` (existing `llama_set_dflash_extract_layers` / `llama_get_dflash_extract_data` — reused for full-stack capture)

**Modify** (D5–D7, scope depends on D3 localization):
- One of the above `ggml-cuda/` kernel files (named after D3 closes)
- Possibly `ik_llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` (dispatch)
- `ik_llama.cpp/src/graphs/build_qwen35.cpp` (integration)
- `ik_llama.cpp/tests/dflash-speculative/test-np-validity-vanilla.cpp` (extend with byte-identity assert)
- `ik_llama.cpp/ggml/CMakeLists.txt` (build flag if needed)

**Create**:
- `ik_llama.cpp/tests/test-deltanet-batch-invariance.cpp` (D6 unit test)
- Possibly a new `.cu` kernel file (D6 implementation)
- `specs/deltanet/batch-invariance.md` (D5 spec)
- `PHASE_DELTANET.md` (D11)
- `data/deltanet/*.json`, `*.bin` (D1–D10 evidence)

## Out of scope (and why)

- **Vulkan backend batch-invariance.** Production is CUDA-only on sm_75 dual TU102. Vulkan path stays as-is.
- **Mamba / other recurrent architectures.** Scope is Qwen 3.5/3.6 hybrid (DeltaNet + full_attention only).
- **Quantization-level changes** to the target model. Target stays Q4_0 + Q4_0_AR16 + AutoRound-preserved BF16 layers.
- **DFlash kernel optimization to TU102 + NVLINK envelope.** Documented in `docs/phases/70-dflash/PHASE_DFLASH.md` future-work pointers; downstream of D7 closure here.
- **DFlash multi-slot libllama API extension.** Same downstream relation.
- **MTP perf optimization at np>1.** Same downstream relation; the most this plan does for MTP is verify np=1 byte-identity preservation + np>1 reproducibility.
- **Cross-host parity (a different host running production).** Determinism here is intra-host across reps at varying np; cross-host deterministic forward is a separate problem.

## Risk surface (per CLAUDE.md §4 "no risks only tasks" — each treated as a task, not a hedge)

The "risks" here are all conditional tasks. They are NOT cover for ducking the work.

- **If D3 names cuBLAS GEMM as the culprit**: D5 must replace heuristic dispatch with fixed-algorithm `cublasGemmEx` call OR a bespoke MMQ-style kernel. Replacement surface is every projection layer × every linear_attn + full_attention layer = ~250+ GEMM call sites. Mitigation: a single wrapper in `mmvq.cu` / `mmq.cu` is the dispatch point — one site, not 250.
- **If D3 names FA as the culprit**: D5 must produce a fixed-split-size FA kernel for Turing sm_75. Per `kernel-design.md §6.3` (DFlash spec), this kernel is bespoke from-scratch already designed; D5 may need to extract or generalize that work.
- **If D3 names DeltaNet recurrence kernel**: the cross-warp reduction order needs warp-shuffle replacement; lines 137–147 of `delta-net.cu` are the surgical site.
- **If D8 shows ANY np=1 regression vs baseline**: this is a D5 design failure, not an acceptable outcome. Return to D5 with the nsys evidence: which replaced kernel didn't hit its perf contract? Tensor-core peak not reached? Memory bandwidth ceiling not approached? Kernel-tuning work resumes until D8 binds. The plan rejects "deterministic but slower" as a ship state.
- **If D7 doesn't close at any np value tested**: do not declare structural per `feedback_no_skipping_lessening.md`. Add additional np values, additional prompts, additional layers to instrumentation. The drift is finite and traceable.

## Change discipline (carry-forward)

- Plan + spec + MEMORY commit + push separately per CLAUDE.md §5/§6. No bundling with code.
- No phase/D-N/PHASE nomenclature in source files, harness scripts, tests, dirs, or branch names per `feedback_no_host_concerns_in_code`. Feature names only: `batch_invariant`, `deterministic`, etc.
- Test-first per `feedback_test_first_discipline`. Scalar oracle → RED test → kernel implementation.
- Surface spec deviations before writing code per `feedback_surface_tradeoff_decisions`.
- Survey prior phase infrastructure before designing per `feedback_survey_prior_phase_before_new_mechanism`. PHASE45 D10.e, DFlash T9, and the existing `cb_eval`-based extract API are all directly reusable.
- Use existing test infrastructure per `feedback_use_existing_test_infra`. Extend `test-np-validity-vanilla.cpp`, do not replace it.

## Estimated cost (per CLAUDE.md §8 — tokens not days)

| Task | Estimate |
|---|---|
| D1 (residual capture infra) | 10–15k |
| D2 (first-divergent layer localize) | 5–10k |
| D3 (first-divergent op localize) | 15–25k |
| D4 (dispatch characterize) | 10–20k |
| D5 (spec replacement design) | 15–25k |
| D6 (implement + oracle + unit test) | 30–50k |
| D7 (integration + binding harness) | 20–35k |
| D8 (perf regression check) | 10–15k |
| D9 (DFlash re-bind) | 5–10k |
| D10 (MTP re-bind) | 5–10k |
| D11 (hygiene) | 5–10k |
| **Total if all pass** | **~130–225k** |

Bold-on-design (commit to TML batch-invariance recipe as the production framework). Measured-on-diagnosis (D2/D3/D4 instrument before declaring root cause — the T9 binary-boundary signature is a specific hint, not a complete diagnosis).

## Pickup state

D1 + D2 CLOSED 2026-05-14. Drift localized; primary hypothesis ("DeltaNet recurrence kernel is the source") refuted by direct evidence; PLAN.md proceeds with refined targets.

### D1 closure

- New harness: `ik_llama.cpp/tests/dflash-speculative/test-deltanet-d1-capture.cpp`
- libllama: extract-layers cap bumped 16 → 80 to fit Qwen 3.6 27B's 65-layer target
- Captures: 4 runs at np ∈ {1,2,4,8} offset=0, full transformer-stack per-slot residuals
- Per-slot per-layer .bin (n_embd=5120 fp32) + manifest JSON in `data/deltanet/d1/`

### D2 closure

First byte-identity break by NP pair at slot 0 (p0):

| NP pair | First-divergent layer | Layer type | Note |
|---|---|---|---|
| NP=1 ↔ NP≥2 | 0 | storage-path | F32 single-token vs F16 multi-token storage — not a clean computational delta |
| **NP=2 ↔ NP=4** | **19** | **FULL_ATTENTION** | **byte-identical layers 0-18, breaks at the 5th FA layer** |
| NP=2 ↔ NP=8 | 0 | sub-F16 | max_abs_diff ≈ 1e-6 (F32 epsilon, below F16 precision) |
| NP=4 ↔ NP=8 | 0 | sub-F16 | same |

`data/deltanet/d2-first-divergent-layer.json`

### Headline implication for D3+

**DeltaNet recurrence kernel is not the primary drift source.** The PLAN.md's working hypothesis was that DeltaNet's non-determinism at np>1 was the gating issue; D2 evidence overturns this. Layers 0-18 — which include DeltaNet recurrence + projections + interleaved earlier FA (indices 3, 7, 11, 15) — are byte-identical between NP=2 and NP=4 at slot 0 (p0). The drift first amplifies at LAYER 19 (FA), and earlier FA layers (3, 7, 11, 15) at the SAME NP transition do NOT introduce visible drift — which is itself a clue.

Refined targets:
- **D3 primary**: instrument layer 19 ops (Q/K/V/O GEMMs + FA kernel + RMSNorm) at NP=2 vs NP=4. The byte-identity break at exactly this layer suggests a kernel-template / tile-geometry pick at the layer's specific shape — possibly the FA kernel handles `n_tokens ∈ {1,2,3}` with one template and `n_tokens ≥ 4` with another (Turing `mma_f16` m16n8k8, 4 hits new tile boundary).
- **D3 secondary**: at NP=4 vs NP=8 layer 0, sub-F16-precision drift from upstream of DeltaNet (RMSNorm or input GEMMs). Bring to fp32 representation if relevant.

The user's "DeltaNet" framing is preserved as the workstream NAME (since the hybrid linear_attn + full_attention stack is the surface), but the empirical evidence reroutes D5+ design to the FA path as primary site.

### First reads for resuming session

1. `data/deltanet/d2-first-divergent-layer.json` — the D2 evidence + D3 target nomination.
2. `ik_llama.cpp/ggml/src/ggml-cuda/fattn-*.cuh` + dispatcher — layer 19 FA op localization site.
3. `ik_llama.cpp/src/graphs/build_qwen35.cpp` — for layer 19's per-op graph structure (Q/K/V → FA → O → ffn).
4. `ik_llama.cpp/tests/dflash-speculative/test-deltanet-d1-capture.cpp` — the harness to extend at D3.
5. T9.1 / PHASE45 D10.e context (already linked above).
