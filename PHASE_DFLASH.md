# PHASE_DFLASH — Qwen3.6-27B DFlash speculative decoding on sm_75

Tracking the DFlash workstream on `production/2026-q2-next`. Spec: `specs/dflash/`
(DESIGN.md, kernel-design.md, dflash.allium, DFlashCycle.tla, DFlashMultiSlot.tla,
allium-tla-binding.json).

Gate sequence per `specs/dflash/DESIGN.md §6` and `specs/dflash/kernel-design.md §10`.
Checkbox semantics per CLAUDE.md §5.

## Tasks

- [x] **T1 — Gate 1: converter binding** (closed)
  - `convert_hf_to_gguf.py::DFlashModel` ports `z-lab/Qwen3.6-27B-DFlash` to GGUF.
  - 6 metadata keys + 2 tensor names: `LLM_ARCH_DFLASH`, `DFLASH_FC`, `DFLASH_HIDDEN_NORM`.
  - Verified output: 58 tensors, 3.3 GiB GGUF, SHA256 `34390c8166f4f7…`
  - Closed at ik_llama.cpp `677943e`.

- [x] **T2 — Gate 2: extract-features hook** (closed)
  - cb_eval matched on `l_out-<il>` tensor names; per-layer residual snapshots dumped via `llama_get_dflash_extract_data`.
  - Cross-stack verification (ik_llama vs vLLM PR #40898): cosine ≥ 0.99988, NMSE ≤ 2.3e-4 across all 5 source layers.
  - vLLM-side dumper: `scripts/dflash-extract-vllm.py` with cloudpickle msgpack patch + fused residual hook (sums `output[0] + output[1]`).
  - Q-mix GGUF reframing: faithful repackaging of AutoRound INT4, not cross-quantization.

- [x] **T3 — Gate 3a: combine_features + inject_kv_fused kernels** (closed)

  Two CUDA kernels delivered byte-identity (within ≤ 1 ULP) sweep across
  (N_slots × MAL_anchors × seed) configurations:

  - `ggml/src/ggml-cuda/dflash/dflash-combine-features.{cuh,cu}` — anchor-level FC + hidden_norm. 8/8 sweep PASS.
  - `ggml/src/ggml-cuda/dflash/dflash-inject-kv.{cuh,cu}` — per-layer K_proj + V_proj + K_norm + RoPE + cache write. 8/8 sweep PASS (V perfectly byte-identical, validating @KAsymmetricallyNormedVNot).

  Spec deviations (committed to `specs/dflash/kernel-design.md §6.2, §6.6`):
  - WMMA m16n16k16 → scalar fp32 accumulators (required for byte-identity vs serial fp32 oracle).
  - Output in registers, not SMEM (avoids fp32→fp16→fp32 round-trip).
  - RoPE transcendentals (`pow`/`cos`/`sin`) evaluated in fp64, cast to fp32 at use (fp32 versions diverge ≤6 fp32 ULP between CUDA libdevice and CPU libm).

  Measured budgets:
  - combine_features: 64 regs/thread, 272 B SMEM, 2 blocks/SM occupancy.
  - inject_kv_fused: 74 regs/thread, 4368 B SMEM, 2 blocks/SM occupancy.

  Allium hygiene added in this gate:
  - 4 new `@invariants` in `dflash.allium`: `CombineOrderFCThenHiddenNorm`, `ContextStatesAnchorLevel`, `InjectPerLayerLaunches`, `KernelDeterminism`.
  - 10 invariants migrated to `bindings_external` with explicit `bound_by` pointing at the (then-unwritten) T3 test files — test-first contract.
  - `@witnesses:` test-side citation pattern added to `scripts/check-bindings.py` check #3; 9/58 invariants now have explicit witness bindings.
  - `kernel-design.md §7` binding-table drift check (5b) added; two pre-existing drift bugs (`InjectKV`, `VerifyOutputArbitratedByTarget`) fixed.
  - All 6 drift checks (forward, reverse, C++ citations, divergence, §7 table, external) green.

- [ ] **T4 — Gate 3b/4: drafter forward + argmax + plumbing** (IN FLIGHT)

  Allium hygiene done (test-first contract for kernel + plumbing):
  - 16 T4 invariants migrated to `bindings_external` in
    `allium-tla-binding.json` (pointed at not-yet-written test files).
    Drift check 6/6 GREEN.
  - Spec edits committed to `kernel-design.md §6.1`: signature
    clarifications (drop `target_features`, output is `out_hidden`
    not `output_logits`, `input_tokens_emb` pre-embedded F16).

  Test oracle infrastructure done:
  - `tests/dflash-speculative/wmma-mimicking-oracle.h` — CPU emulation
    of WMMA m16n16k16 Turing tensor-core MMA with binary-tree-within-
    tile fp32 reduction. Plus serial-fp32 sanity-check pair.
  - `tests/dflash-speculative/test-wmma-mimicking-oracle.cpp` — oracle
    self-test, sweeps M={16}, N={16,64,128,1024}, K={16…5120}, 4 seeds;
    confirms determinism + sub-ULP agreement vs serial fp32 at K≤128
    and bounded-rate drift at K=5120. PASS at /opt/llm/build-dflash.
  - `tests/dflash-speculative/dflash-drafter-forward-reference.h` — full
    scalar drafter reference composing WMMA oracle + serial fp32 RMSNorm
    + fp64-transcendental RoPE + scalar fp32 single-query SWA/full
    attention + silu + gate*up + residual at attention + MLP boundaries.
    Tiny-shape smoke test (L_d=2, D_emb=64, …) confirms 512/512 cells
    non-zero with mean_abs=0.126.
  - `tests/dflash-speculative/test-dflash-drafter-forward.cpp` — test
    driver. Phase 1 (always runs): reference smoke. Phase 2 (stub-
    mode): kernel-vs-reference at production shape. Currently STUB_
    KERNEL=true → exits 77 (CTest SKIP) until kernel body lands.

  Skeleton kernel done:
  - `ggml/src/ggml-cuda/dflash/dflash-drafter-forward.{cuh,cu}` —
    launcher signature locked per `kernel-design.md §6.1`. Stub body
    zeros the output buffer.

  Still to do for T4 closure:
  - Kernel body (the big work): per-layer pipeline implementation. Two
    valid implementation strategies under evaluation:
      A. Per-layer separate launches (simpler, ~50 launches, ~250 µs
         overhead) — correctness-first.
      B. Cooperative mega-kernel with `cg::this_grid().sync()` (spec
         literal, perf-tuned).
    Phased approach: start with A, validate byte-identity vs reference,
    then refactor to B if T8 perf gate demands it. Either approach
    requires SMEM chunking for intermediate=17408 MLP buffers (too big
    for one CTA's SMEM at full size on sm_75 = 64 KiB).
  - `dflash_drafter_lm_head` separate kernel — BF16 GEMV against target's
    `output.weight` [5120, 248320].
  - `dflash_argmax_match` kernel — per-slot accept-prefix + bonus.
  - DFlash arch dispatch + drafter loader plumbing.
  - vLLM reference logits dump for closure binding.

  Closure binding (per `kernel-design.md §10`):
  - Kernel-vs-reference byte-identity sweep across {N_slots×BLOCK_SIZE×
    seed} configurations at production shape, ≤ 1 fp16 ULP at ≤ 1% rate.
  - Drafter logits within 1e-5 NMSE vs vLLM PR #40898 reference at
    BLOCK_SIZE=4 on a fixed prompt.
  - Measured persistent kernel register count ≤ 64 regs/thread on
    sm_75 via `--ptxas-options=-v`.

- [ ] **T5 — Gate 4: full block-emit + accept loop on Qwen3.6-27B**
  - `common_speculative_dflash_*` wiring.
  - `examples/speculative-simple/` --dflash flag.
  - Closure: within 10 % of the Gate 0 vLLM oracle (24.46 tok/s spec=4 np=1 on Qwen3.6-27B INT4).

- [ ] **T6 — Gate 5: 27B np=1 determinism**
  - `dflash_state_checkpoint`/`dflash_state_restore` (DeltaNet recurrent state ping-pong).
  - `dflash_verify_attn` from scratch (sm_75 PTX `mma.sync.m16n8k8`, fixed-split-size).
  - BF16→FP16 cast for target's AutoRound-preserved linear_attn at server init.
  - Closure: 3-run byte-identical at np=1; `ne[1]=5` verify deterministic; state save/restore round-trip bit-identical.

- [ ] **T7 — Gate 5b: drafter np-invariance binding**
  - `tests/test-dflash-determinism-np-invariance.cpp` — SHA-256 match across np ∈ {1, 2, 4, 8}.
  - Closure: bit-identical drafter logits across np values. If fail, instrument per-kernel; do not bail.

- [ ] **T8 — Gate 6: Qwen3.6-27B speedup measurement**
  - Pre-Gate MTP `--draft 3` baseline measurement (mandatory anchor — see auto-memory `feedback_anchor_to_measured_baselines`).
  - DFlash speedup measurement, block_size sweep ∈ {4, 5, 6, 8}.
  - Ship outcome: PASS (≥ 1.5× MTP) → ship `profiles/qwen36-27b-x1-dflash.sh`; NEUTRAL (1.0–1.5×) → tunable option; FAIL (< 1.0×) → stay on MTP.

- [ ] **T9 — Gate 7 (conditional on T7 GREEN): batched verify at np > 1**
  - Aggregate vs vanilla batched at np=8.
  - Ship outcome: PASS (≥ 1.8× vanilla) → `profiles/qwen36-27b-x8-dflash.sh`.

## Verification (end-of-phase composite)

Builds with `-DGGML_CUDA_DFLASH=ON -DCMAKE_CUDA_ARCHITECTURES=75`:

```sh
cd ik_llama.cpp && cmake -B build -G Ninja \
  -DGGML_CUDA=ON -DGGML_CUDA_DFLASH=ON \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF \
  && cmake --build build -j 32
```

Unit tests:

- `test-dflash-combine-features` (T3, GREEN at sweep)
- `test-dflash-inject-fused` (T3, GREEN at sweep)
- `test-dflash-determinism-ne5` (T6)
- `test-dflash-determinism-np-invariance` (T7)
- `test-dflash-state-revert` (T6)

Allium ↔ TLA+ ↔ C++ drift check (must pass on every commit to spec dir):

```sh
python3 scripts/check-bindings.py
```
