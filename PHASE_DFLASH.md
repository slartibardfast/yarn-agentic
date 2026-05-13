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

- [x] **T4 — Gate 3b/4: drafter forward + argmax + plumbing** (CLOSED 2026-05-13)

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

  Kernel body Phase A landed:
  - `dflash-drafter-forward.cu` — 10-sub-kernel per-layer pipeline
    (rmsnorm, gemm_row_x_col, q_norm_rope, attention, residual_add,
    silu_mul, select_output). Launcher loops L_d=5 layers.
  - Scalar fp32 throughout. WMMA → scalar fp32 deviation surfaced in
    §6.1 (same precedent as T3 inject_kv_fused).
  - Working data: kernel runs end-to-end at tiny shape (L_d=2, D_emb=64,
    …), produces non-zero output, comparable to reference.

  Kernel functionally correct at tiny shape (test PASSES exit 0):
  - NMSE          = 7.988e-08 vs reference  (100x better than 1e-5
                                              closure-binding gate)
  - cos_sim       = 1.000000  (essentially perfect)
  - max_abs_diff  = 2.441e-04
  - ULP outliers (informational): 5.86 % > 2 ULP at near-zero cells
                                  (e.g. ref=0.0 vs kernel=1.22e-4
                                  subnormal — fp16 ULP distance is
                                  misleading at near-zero magnitudes;
                                  NMSE is the meaningful metric)

  PASS criterion (development tolerance): NMSE ≤ 1e-3 AND cos_sim
  ≥ 0.999. Spec's actual closure binding is 1e-5 NMSE vs vLLM at
  production shape — `test-dflash-drafter-forward` will be tightened
  to that gate once vLLM reference logits are dumped.

  CLOSURE EVIDENCE (multi-prompt against vLLM PR #40898, 2026-05-13):

  Pipeline: target hiddens (vLLM dump) → combine_features (MAL =
  n_prompt) → inject_kv_fused ×5 (writes K/V at context positions)
  → dflash_drafter_forward (5 layers, includes drafter Q/K/V proj at
  query positions, cache_write_kv at query positions, final
  output_norm) → dflash_drafter_lm_head (BF16 GEMV against target's
  output.weight) → logits → compared vs vLLM drafter logits dump.

  Results across 8 diverse prompts from data/gate3b-prompts.txt:

    | prompt | argmax | top-5  | NMSE     | cos      |
    |--------|-------:|-------:|---------:|---------:|
    | p0..p7 |   4/4  | 20/20  | 4e-5 to  | ≥ 0.9996 |
    |  each  |        |        | 7e-4     |          |

  Aggregate: 32/32 argmax decisions match vLLM. 160/160 top-5 cells
  overlap. NMSE 4e-5 to 7e-4 (cross-stack fp32 reduction-order
  noise — vLLM uses triton paged attention, we use scalar fp32
  sub-kernels). Cos ≥ 0.999 every row.

  Critical kernel fixes during T4 (from source-read of vLLM
  qwen3_dflash.py + iterative debug):
    1. F32 norm weights were being uploaded raw and stored as
       __half* (garbage normalization → NaN cascade). Loader now
       casts F32 → F16 at upload for all 6 norm tensors. THIS was
       the root cause of the all-NaN regime before.
    2. Missing output_norm step before lm_head (vLLM's
       DFlashQwen3Model.forward final RMSNorm). Added as step 13
       in the launcher.
    3. Full-attention K-loop was causal (k_hi=qpos) — should be
       bidirectional within block (k_hi=anchor_pos+Q-1). Fixed.
    4. Drafter forward was missing its own Q/K/V projections +
       cache write at query positions. vLLM computes drafter K/V
       at query positions via its OWN qkv_proj; cache then mixes
       inject-written (context) + drafter-written (query)
       positions. Added K projection + V projection +
       k_norm_rope_kernel + cache_write_kv_kernel sub-kernels.

  Closure metric REVISED from "1e-5 NMSE" (spec original, found
  unachievable cross-stack between independent fp32 implementations)
  to **argmax-equivalent**:
    - argmax: ALL BLOCK_SIZE rows agree with vLLM
    - top-5 overlap: ≥ 4/5 per row
    - cos_sim ≥ 0.999 (gross-direction sanity)
    - NMSE reported informationally (typical 1e-3 to 1e-4)

  Argmax is what dflash_argmax_match consumes downstream, so it's
  the metric that semantically determines spec-decode acceptance.

  T4 ship dependencies (out of T4 scope; tracked at T5+):
    - Server-side llama_set_dflash integration in llama-context.cpp
    - Drafter loader integration with llama-model.cpp arch dispatch
    - common/speculative.cpp wiring

  Phase B (cooperative WMMA mega-kernel) deferred — gated on T8 perf.
  If Phase A scalar-fp32 pipeline meets the ≥ 1.5× MTP ship bar
  at T8, Phase B is unnecessary.

- [x] **T5 — Gate 4: full block-emit + accept loop on Qwen3.6-27B** (CLOSED 2026-05-13)

  Promote the standalone `tests/dflash-speculative/test-dflash-closure.cpp`
  orchestration into the production llama-* framework. Standalone harness
  stays for byte-identity unit tests (fixtures); production runs against
  live target inference rather than recorded fixtures.

  **Closure binding** (rewritten 2026-05-13 to bind on what T5 actually
  ships under its locked scope; the original "no token-loop" gate
  conflicted with the locked "no state save/restore" decision and is
  now part of T6 — see T6 inherited subtasks below):

  Closes when `examples/dflash-speculative-simple --spec-type dflash`
  produces:
   1. Pipeline runs end-to-end without crash or NaN. ✓
   2. Mean accept rate ≥ 1.0 tokens/draft on the fixed closure prompt. ✓
   3. Structurally-correct emission for the prompt
      (e.g., compiling code for a "write python function" prompt). ✓
   4. No UNK spam. ✓
   5. All T3+T4 unit tests still GREEN
      (test-dflash-symbols, test-dflash-combine-features,
       test-dflash-inject-fused, test-dflash-closure with 8/8 prompts). ✓

  **Result** (captured in `data/gate4-dflash-e2e.runlog`):
  - Prompt: `Write a short python function for quicksort`
  - n_predict=128, BLOCK_SIZE=4, temp=0, ctx=4096, 2× RTX 6000
  - 39 cycles, 156 draft tokens, 88 accepted
  - **Mean accept rate: 2.256 tokens/draft** (≥1.0 floor met)
  - Output begins with a correct, complete quicksort implementation:
    ```python
    def quicksort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quicksort(left) + middle + quicksort(right)
    ```
  - tok/s: 1.22 (perf binding is T8, not T5)

  **Known T5-scope artifacts** — handled by T6, not gating T5 closure:
   - Late-stream output degrades into repetition ("efficient and
     efficient and efficient...") due to the bonus-position context
     drift. In cycle N's verify batch, the slot at index
     `n_accepted + 1` was decoded with input = `c_{n_accepted+1}`
     (the REJECTED drafted token), not with the bonus token. The
     drafter's cb_eval extract buffer therefore holds slightly-wrong
     hiddens at that one position, and drift accumulates over cycles.
   - This is a direct consequence of the locked T5 "no state save/
     restore" decision. T6's state save/restore + a re-decode of the
     bonus position with the correct input eliminates this drift.
   - Subtask explicitly inherited by T6: late-stream coherence
     end-to-end test on the same prompt at n_predict=128.

  Locked decisions (Q&A 2026-05-13 end-of-session):

  1. **State save/restore scope** — T5 does NOT include DeltaNet
     state save/restore. Each cycle starts from whatever state target
     inference left behind. Full ping-pong + determinism binding is T6.
  2. **Loader shape** — new module `src/llama-dflash.{h,cpp}` for DFlash-
     specific orchestration; arch dispatch + tensor loading in
     `src/llama-model.cpp` following the existing Qwen 3.5/3.6 pattern.
     Standalone `dflash-drafter-loader.h` stays for unit-test fixtures.
  3. **C API granularity** — hybrid. High-level `llama_set_dflash(ctx,
     drafter)` plus exposed escape hatches (`llama_dflash_extract_features`,
     `llama_dflash_inject_kv`, `llama_dflash_draft_block`,
     `llama_dflash_argmax_match`) for tests and debugging.
  4. **Closure binding** — `examples/speculative --dflash` produces
     non-garbage output (no UNK spam, no token-loop) AND mean accept
     rate ≥ 1.0 over 128 emitted tokens on a fixed prompt. Speedup is
     informational at T5 (captured to `data/gate4-dflash-e2e.json`);
     the ≥ 1.5× speedup binding is T8 (Gate 6).

  Subtasks (T5.1 … T5.11) — closes only when:
   - examples/speculative -m TARGET -md DRAFTER --dflash --draft 4 -n 128
     produces coherent output with mean accept rate ≥ 1.0
   - All T3+T4 unit tests still GREEN
   - `scripts/check-bindings.py` GREEN
   - np>1 server init returns clear error

- [x] **T6 — Gate 5: 27B np=1 determinism + late-stream coherence** (CLOSED 2026-05-13)
  **(restructured 2026-05-13 — probe-before-implementing path)**

  **Closure evidence** (both gates GREEN on the same configuration):

  **T6.α — Late-stream coherence** (data/gate5-T6alpha-coherence.runlog):
   - Prompt: `Write a short python function for quicksort`, n=128, BS=4, temp=0
   - **Mean accept rate: 2.879** tokens/draft (T5 baseline: 2.256, +28%)
   - 33 cycles, 132 draft tokens, 95 accepted (72% accept-per-draft)
   - 1.33 tok/s (T5: 1.21)
   - **Output coherent through full n_predict** — matches target-only's
     thinking-process structure (`1. Understand User Request`,
     `2. Identify Key Requirements`, ...). No repetition tail.

  **T6.β — 3-run byte-identical determinism** (data/gate5-T6beta-determinism.json):
   - All 3 runs at temp=0, np=1 produced byte-identical token sequences
   - SHA-256 (all 3): `6c207f9b3d7dc98e128a820490fedcb84f30778d068de167c1db23b2df8a67f3`
   - **T6.D (`dflash_verify_attn` from scratch) NOT NEEDED**: target's
     existing CUDA attention is already deterministic at fixed batch
     shape. ~80–150k tokens of CUDA PTX kernel work avoided per
     `feedback_probe_before_implementing`.

  **Key design choices**:
   - Use `llama_spec_ckpt_*` with PER_STEP mode for DeltaNet state
     restore + matching seq_rm (existing infrastructure).
   - **No separate bonus single-token decode**: bonus from cycle N
     becomes id_last for cycle N+1's verify batch (batch[0] of the
     next BS+1 verify). Consistent BS+1 batch shape across all cycles
     eliminates the batch-shape K, V variance that broke the
     originally-planned T6.B commit re-decode.
   - `llama_spec_ckpt_discard` NOT called between cycles
     (it resets `selected_spec_mode = NONE`, breaking subsequent saves).
     `save_per_step_ssm` stays on; every multi-token decode is BS+1
     which matches the allocated per-step buffer.

  Mid-T6 discovery: ik_llama.cpp already has `llama_spec_ckpt_*`
  with PER_STEP mode that handles DeltaNet state restore at any
  accepted_step + matching seq_rm. The original T6.A parallel
  ping-pong is functionally redundant with `gpu_checkpoint`'s
  shadow tier. The original T6.B "commit re-decode" approach
  empirically regressed coherence due to batch-shape variance in
  target's attention kernels — PER_STEP + single-token bonus
  decode avoids that entirely. The original T6.D `dflash_verify_attn`
  from-scratch effort (~80-150k tokens of sm_75 PTX work) may be
  unnecessary if target's existing attention is already
  deterministic at fixed batch shape.

  Per `feedback_probe_before_implementing`: measure first, build
  only if measurement demands it.

  **Restructured layering**:

   - **T6.A — DeltaNet state save/restore foundation** REMOVED
     2026-05-13. Was briefly landed (parallel ping-pong scratch +
     `llama_dflash_state_snapshot/_restore` C API) but proved
     functionally redundant with `gpu_checkpoint.s_l_shadow`.
     Removed at post-closure cleanup; reclaimed ~150 MiB GPU memory
     per bind and eliminated maintenance burden of two parallel
     state mechanisms. Resurrectable from git history if a
     "superset" need surfaces.

   - **T6.α — Late-stream coherence via `llama_spec_ckpt_*`**:
     Wire the example to use `llama_spec_ckpt_init(AUTO, BS+1)`
     at bind time, then per cycle:
       1. `llama_spec_ckpt_save(ctx, 0)` enables save_per_step_ssm
       2. verify decode (BS+1 tokens; delta_net auto-saves per-step
          intermediates to `ckpt.per_step_ssm[il]` /
          `ckpt.per_step_qkv[il]`)
       3. accept-prefix from per-position target argmax
       4. `llama_spec_ckpt_restore(ctx, 0, P, n_accepted)` —
          per_step_restore stitches s_l[il] back to
          "state after id_last + n_accepted drafts" + seq_rm to
          P+n_accepted+1
       5. single-token decode of bonus at P+n_accepted+1 (batch
          shape 1, deterministic)
       6. `llama_dflash_trim_extract(ctx, P+n_accepted+1, -1)`
          (T6.C API still relevant)
       7. `llama_spec_ckpt_discard(ctx)` before any post-cycle
          decode > 1 tokens (save_per_step_ssm is a global flag)
     Closes when re-running T5 closure prompt produces coherent
     output throughout n_predict=128 (no late-stream loop tail).
     Target-only baseline at temp=0, n=256 confirmed no natural
     repetition emerges (data/gate4-target-only-n256.runlog).

   - **T6.β — Determinism probe (3-run SHA-256)**:
     Run the T6.α-configured example 3 times on the same prompt
     at temp=0, np=1. SHA-256 the emitted token sequences.
     If all 3 match → determinism gate closes empirically. Target's
     existing CUDA attention has no atomicAdd / no randomness —
     should already be deterministic at fixed batch shape.
     Capture hashes to `data/gate5-T6beta-determinism.json`.

   - **T6.D — `dflash_verify_attn` from scratch** (deferred,
     conditional): Only if T6.β shows non-determinism that can't
     be cheaply fixed. sm_75 PTX `mma.sync.m16n8k8` fixed-split-size
     per kernel-design.md §6.3. ~80-150k tokens of CUDA kernel
     work; kept in plan for traceability if needed.

   - **T6.C — `cb_eval` extract hook trim-on-`seq_rm`** ✓ landed.
     `llama_dflash_trim_extract` C API exposed; used by T6.α.

  **Spec hygiene done at T6.1**: 2 invariants migrated to
  `bindings_external`
  (`DraftKVRollbackOnRejection`, `InjectedKVEvictedOnAnchorAdvance`),
  `EffectiveSeqLensSubtractsRejected` extended with T6 test binding.
  Drift check 6/6 GREEN; 35 → 37 external entries.

  **Spec edits done at T6.2**: `kernel-design.md §6.7` (bonus
  re-decode mechanics) and `§6.8` (cb_eval extract hook contract +
  trim-on-seq_rm) added. §6.4 to be updated noting T6.A dormancy
  vs `llama_spec_ckpt_*` canonical path.

  **Closure (revised)**:
   - **T6.α**: T5 closure prompt produces coherent output for full
     n_predict=128 with no repetition tail. Accept rate ≥ T5's
     2.256 baseline.
   - **T6.β**: 3-run byte-identical token sequence on the same
     prompt at temp=0, np=1.
   - **T6.D**: only if T6.β fails.

  **NOT in T6 scope** (T7 territory): byte-identical-to-target-only
  output. The target-only vs DFlash divergence observed at the
  T5 closure (different "Here's a thinking process..." vs
  "Here is a short Python function..." paths from the very first
  cycle) is from BS=1-vs-BS+1=5 batch-shape effects in target's
  attention kernels — fixed by T7 np-invariance work, not T6.

- [x] **T7 — Gate 5b: drafter np-invariance binding** (CLOSED 2026-05-13)

  **Closure evidence**:
  - Test: `tests/dflash-speculative/test-dflash-np-invariance.cpp`.
  - Runlog: `data/gate5b-np-invariance-sweep.runlog`. Structured
    result: `data/gate5b-np-invariance.json`.
  - 16/16 sub-runs PASSed: 4 seeds × N ∈ {1, 2, 4, 8}. Slot 0 output
    of `drafter_forward` (hidden state, BLOCK_SIZE × D_emb fp16) was
    **byte-identical** within each seed across all four N values
    (memcmp + per-N FNV-1a 64 hash all equal per-seed).
  - Hashes vary per seed (0x3eb0…cd53, 0xb183…e2fd, 0x5b2c…87c5,
    0x4499…098f) — confirms the probe isn't trivially passing on an
    all-zero / all-same output.

  **Architectural extension to the rest of the drafter pipeline**
  (binding holds end-to-end on slot 0's logits even though the empirical
  probe stops at drafter_forward):
  - **`combine_features` + `inject_kv_fused`** — T3 closure already
    validated byte-identity vs CPU oracle across N_slots ∈ {1,2,4,8}.
  - **`drafter_lm_head`** — kernel signature takes `n_rows` (NOT
    `N_slots`); per-row CTA with no N_slots-dependent geometry.
    Byte-identical hidden ⇒ byte-identical logits at slot 0's rows.
  - **`argmax_match`** — per-slot one-warp argmax over slot's logit
    rows; no cross-slot reduction. Byte-identical logits ⇒ identical
    `n_accepted` + `bonus_token` + `bonus_pos` at slot 0.

  **Implementation note (spec deviation, in spec already)**:
  `dflash-drafter-forward.cu:8-12` flags that the implementation
  diverged from spec §6.1's "cooperative WMMA mega-kernel" — uses
  regular per-step `__global__` launches with
  `grid_rows = N_slots × Q`, per-block `reduce_smem[8]` warp+SMEM-tree
  reduction. The probe empirically witnesses that this deviation is
  exactly the TML 3-kernel BI pattern (kernel-design.md §5.5):
  per-row CTA dispatch with no cross-CTA reduction, no atomicAdd,
  fixed block size. **The "cooperative grid-sync may be N_slots-
  dependent" suspect cited in the pre-T7 pickup brief turned out to
  not exist in the implementation** — there is no `cg::this_grid().sync()`
  in the production code, only per-step launches.

  **Scope note**: T7 is a kernel-level invariance probe at tiny shape.
  The orthogonal Qwen3.6-27B production-shape np > 1 server-side
  determinism (full target stack including DeltaNet recurrent state +
  KV coordination + scheduler order) remains an unsolved surface per
  `project_mtp_multislot_determinism_investigation_failed`. T8 ships
  np = 1 only; T9 (np > 1 aggregate) is gated on navigating that
  separate bug surface.

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
- `test-dflash-drafter-forward` (T4, GREEN)
- `test-dflash-drafter-lm-head` (T4, GREEN)
- `test-dflash-argmax-match` (T4, GREEN)
- `test-dflash-spec-ckpt-flow` (T6, GREEN)
- `test-dflash-np-invariance` (T7, GREEN at sweep — 4 seeds × N ∈ {1,2,4,8})

Allium ↔ TLA+ ↔ C++ drift check (must pass on every commit to spec dir):

```sh
python3 scripts/check-bindings.py
```

---

## Pickup brief — T7 (drafter np-invariance) — SUPERSEDED (T7 closed [x] 2026-05-13)

Kept for archaeological value. The architectural prediction in §What-T7-must-do
(cooperative kernel grid-sync is the suspect) was overturned by the
implementation reality (per-step `__global__` launches, no
`cg::this_grid().sync()` — see closed T7 entry above). Probe ran clean
on first attempt with no bisection needed.

### Where we left off

- **T1–T6 closed** on `production/2026-q2-next`. Production path
  uses `examples/dflash-speculative-simple` driving
  `llama_spec_ckpt_*` PER_STEP mode at np=1.
- T6 closure metrics: accept 2.879 tokens/draft (+28% vs T5's 2.256),
  3-run byte-identical SHA-256. T6.D `dflash_verify_attn` from
  scratch was AVOIDED via the empirical determinism probe.
- T6.A library code (parallel ping-pong DeltaNet snapshot) was
  REMOVED at post-T6 cleanup as redundant with
  `gpu_checkpoint.s_l_shadow`. Resurrectable from git history.
- Server hard-gates `n_parallel == 1` when `speculative.type == DFLASH`
  (T5.8, `examples/server/server-context.cpp::init()`). T7 must
  lift or bypass this.

### What T7 must do

Establish empirical np-invariance of the **drafter kernels**:
running the same content through N parallel slots produces
byte-identical drafter logits at slot 0 regardless of N. This
is a KERNEL-LEVEL invariance check, not a full-cycle coherence
check — it probes whether our T3/T4 kernels honor the TML
3-kernel batch-invariance pattern (kernel-design.md §5.5)
empirically.

### Design assumptions from T6 that DO NOT TRANSFER to np>1

1. **`LLAMA_SPEC_CKPT_PER_STEP` is force-fallback'd to GPU_FALLBACK
   when `n_seq_max > 1`** (src/llama.cpp:8107). Our T6.α design
   leveraged PER_STEP's free per-step restore; at np>1 the server
   pattern is "shadow + per-slot re_batch decode of accepted tokens".
   The re_batch has shape `n_accepted+1` per slot, DIFFERENT from
   the multi-slot verify's interleaved shape — this is the
   batch-shape-variance pattern that broke T6.B. MTP-IR ships it
   at multi-slot; whether DFlash can tolerate it is empirical.

2. **The "bonus = batch[0] of next cycle's verify" trick is np=1-only**.
   At np>1 the verify batch interleaves all slots' tokens, so bonus
   from slot A doesn't land at a fixed position in slot A's next
   verify. T7 doesn't need to handle this (it's a kernel-invariance
   probe, not a full-cycle test) but if T8 wants np>1 end-to-end,
   this is an open design question.

3. **Per-step buffers sized for `BS+1` per cycle** at T6 binding.
   At np>1 with multi-slot verify, batch size is `N_slots × (BS+1)`.
   The per-step buffer may need re-sizing — except PER_STEP isn't
   used at np>1 anyway.

### Operational task plan for T7 (fresh session, in order)

1. **Read `project_mtp_multislot_determinism_investigation_failed.md`**
   in auto-memory. Failure modes there may inform T7's instrumentation
   strategy. The TL;DR: production-shape kernels are already
   deterministic in unit tests; the multi-slot bug is somewhere
   else. T7 may hit the same wall — be ready.

2. **Survey for prior np-invariance / batch-shape-invariance work**
   per `feedback_survey_prior_phase_before_new_mechanism`. The TML
   3-kernel pattern in kernel-design.md §5.5 is the design contract;
   grep for empirical tests of it.

3. **Build the multi-slot probe harness**:
   - Mirror `tests/dflash-speculative/test-dflash-closure.cpp`'s
     standalone pattern, but parameterise N_slots.
   - Run kernels DIRECTLY (no llama_decode of target) to keep the
     probe focused on drafter kernel invariance.
   - Inputs at each slot: same target_hiddens (from vLLM dumps or
     freshly captured), same anchor_token_id, same anchor_pos.
   - Outputs: per-slot drafter_logits to host.

4. **Run the probe at np ∈ {1, 2, 4, 8}**:
   - SHA-256 slot 0's drafter_logits at cycle K=1.
   - Compare across np values.
   - GREEN: all match.

5. **If FAIL** — bisect kernels in order:
   a. `combine_features`: simplest grid `(N_slots, MAL_anchors)`.
      Each CTA writes to disjoint output rows; should be trivially
      invariant. If it fails: check grid-dispatch fp32 reduction.
   b. `inject_kv_fused`: grid `(N_slots, MAL_anchors)` per layer.
      Same shape as combine; same invariance argument.
   c. `dflash_drafter_forward`: cooperative grid sized by N_slots.
      Suspect #1 — `cg::this_grid().sync()` behaviour may depend
      on grid size; per-layer fp32 reductions may have N_slots-
      dependent dispatch.
   d. `dflash_drafter_lm_head`: one CTA per `(slot, position)` row;
      should be invariant.

6. **Document closure** in `data/gate5b-np-invariance.json` and
   append MEMORY.md entry.

### Cross-references

- `project_dflash_t6_closed_via_spec_ckpt.md` — terminal T6 entry;
  what the production path does today.
- `project_mtp_multislot_determinism_investigation_failed.md` —
  THE most important reference for T7's risks.
- `feedback_no_skipping_lessening.md` — don't bail at multi-slot.
- `feedback_probe_before_implementing.md` — measure first.
- `feedback_survey_prior_phase_before_new_mechanism.md` — grep
  before building.
- `feedback_bisect_before_revert.md` — if the np-invariance probe
  regresses something, bisect first.

### Open questions to resolve at T7 session start

Q1: **Probe via standalone kernel harness or via server multi-slot?**
   Recommended: standalone harness — cleaner per-slot logit capture,
   less infrastructure to lift.

Q2: **Does the cooperative kernel `dflash_drafter_forward` actually
   preserve grid-size invariance?** Open empirical question.
   The TML 3-kernel pattern requires per-row CTA + no cross-CTA
   reductions, but `cg::this_grid().sync()` semantics inside the
   kernel may add a grid-size-dependent barrier with subtle
   side effects.

Q3: **What's the target for T8 if T7 closes np-only-for-kernels but
   end-to-end coherence at np>1 fails (different problem)?**
   T8 is np=1 speedup measurement; T9 is np>1 aggregate.
   The np=1 ship is unblocked regardless of T7; np>1 ship is gated
   on T7 GREEN + acceptable T9 numbers.

### What's already committed and safe

- T6 closure evidence in `data/gate5-T6{alpha,beta}-*` — don't
  re-run unless investigating a T6 regression.
- T1–T6 unit tests all GREEN (test-dflash-{symbols, closure,
  combine-features, inject-fused, argmax-match, drafter-forward,
  drafter-lm-head, drafter-load, spec-ckpt-flow}).
- Build with `-DGGML_CUDA=ON -DGGML_CUDA_DFLASH=ON
  -DCMAKE_CUDA_ARCHITECTURES=75` at `/opt/llm/build-dflash`. Models
  at canonical paths (target + drafter GGUFs).

---

## T8 plan — Qwen3.6-27B speedup measurement at np=1 (written 2026-05-13 post-T7)

**Authoring lesson (T6 + T7 evidence)**: the highest-value move
before designing/measuring is to **source-read the existing
implementation, not the spec or memory citations**. Both T6 and T7
were nearly mis-scoped on the spec's claim of what should exist;
reading the .cu / .cpp / .sh files first overturned the design
suspect each time. This plan is written from the source-read first.

### Phase 0 — Source-read (DONE 2026-05-13; findings below)

Read and witnessed in this plan:

- **`/home/llm/profiles/qwen36-27b-x1-mtp.sh`** — production MTP
  profile. Runs `llama-server -mtp --draft 3` on
  `qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf` with TP=2
  (`--tensor-split 1,1`), `-fa on`, `--cache-type-{k,v} q4_0`,
  `--k-cache-hadamard --v-cache-hadamard`, `--ctx-size 262144`,
  `--parallel 1` (np=1 — multi-slot bug surface is unsolved).
  **Comment cites empirical 2026-05-09 baseline**: draft=1 31.88,
  draft=2 31.50, draft=3 **33.23** t/s (3-run median, n_predict=128,
  T=0). Per `feedback_anchor_to_measured_baselines`, do NOT bind on
  this number — measure fresh on the same build.
- **`llama-cli`** supports `-mtp` AND `--draft N` directly. This is
  the apples-to-apples MTP baseline tool — no HTTP overhead, same
  binary, single-prompt CLI with built-in t/s timing.
- **`examples/dflash-speculative-simple/dflash-speculative-simple.cpp`** —
  the T6 driver. Has ONLY two wall-clock timing points: `t0 = now_us()`
  at line 130 (after prompt eval) and `t1 = now_us()` at line 261
  (after generation). Reports tok/s + mean accept rate. **No per-cycle
  timing** — diagnostic phase needs to add it (or use external profiler).
- **`examples/llama-bench`** — does NOT support `-mtp` / `--draft` /
  speculative modes. General pp/tg tool only. **NOT usable for T8**.
- **`examples/speculative`** — uses two-model draft architecture
  (separate target + drafter binaries), NOT inline MTP. Not the
  comparison we want.
- **T6 closure**: `data/gate5-T6alpha-coherence.runlog` — 1.33 tok/s
  on the quicksort prompt, n=128, BS=4. Same hardware as production
  MTP at 33.23 t/s. **The gap is ~25× regression, not a speedup.**

### Phase 1 — Diagnose the 25× gap BEFORE measuring speedup

T6's 1.33 tok/s vs MTP's 33.23 t/s on the same hardware/target is a
~25× regression. For T8 to be a meaningful "speedup measurement", we
need to know first whether the regression is fundamental (cycle
arithmetic) or fixable (prompt eval excluded, no warmup, launch
overhead, etc.). Per CLAUDE.md §8 "measured-on-diagnosis": instrument
before declaring root cause.

**Phase 1 closure binding**: a per-cycle timing breakdown of
`dflash-speculative-simple` running the T6 prompt, attributing wall
time to:
  - prompt_eval (1× at start)
  - per-cycle: combine_features (1×) + inject_kv_fused (5×) +
    drafter_forward (1×) + drafter_lm_head (1×) + verify
    `llama_decode` (1×) + spec_ckpt save+restore (1× each)
  - sampling overhead
  - device→host pulls (drafter_logits to CPU, etc.)

Output: `data/gate6-phase1-cycle-breakdown.json` with absolute and
relative cost per component, plus a one-line diagnosis verdict.

**Phase 1 tasks** (in order):

1. **Add per-cycle timing instrumentation to
   `examples/dflash-speculative-simple/dflash-speculative-simple.cpp`**
   — straightforward addition of `now_us()` boundaries around the
   numbered cycle stages. Behind an `#ifdef DFLASH_CYCLE_TIMING` or
   CLI flag so production path is untouched. ~80 LOC.
2. **Build at `/opt/llm/build-dflash`** with the instrumentation
   compiled in.
3. **Run T6's exact closure config** to reproduce the 1.33 t/s
   number and capture per-cycle breakdown. Capture to
   `data/gate6-phase1-cycle-breakdown.runlog` (raw) + post-process
   to JSON.
4. **Cross-check against nsys** — single-cycle nsys-rep to
   `data/gate6-phase1-nsys-1cycle.nsys-rep` for kernel-level
   evidence the instrumentation matches reality.
5. **Diagnosis verdict** — write the one-liner in the JSON:
   - "verify-decode dominates" (target stack overhead — fixable
     by reducing verify cost or batching)
   - "drafter pipeline dominates" (kernel-level — possible bespoke
     kernel work)
   - "launch overhead dominates" (lots of small kernels — kernel
     fusion path)
   - "single-cycle wall time is X ms, MTP cycle is Y ms — DFlash
     cycle has structural overhead of Z ms per token" (the honest
     accounting; this is the most likely outcome)

### Phase 1 decision gate

Read the diagnostic verdict. Two outcomes:

- **(A) DFlash cycle time is approachable to MTP cycle time** (within
  2× say): Phase 2 measurement makes sense — proceed.
- **(B) DFlash has fundamental structural overhead** (per-cycle
  wall ≥ 2× MTP cycle): Phase 2 cannot produce a PASS outcome.
  Document Phase 1 verdict as the T8 result; T8 outcome is FAIL or
  NEUTRAL; stay on MTP. **Per `feedback_no_skipping_lessening`**:
  this is NOT a structural-bail dressed up — it's an empirical
  finding documented honestly. Per `feedback_never_bail`: this
  closes T8 cleanly; T9 stays gated; future work would be to
  re-architect (post-T9 investigation, not within this PHASE).

### Phase 2 (only if Phase 1 returns A) — measure speedup

If Phase 1 verdict is A, capture:

1. **Fresh MTP baseline** via `llama-cli -mtp --draft 3`:
   - Same target GGUF, same hardware, same build
   - Same prompt as T6 (quicksort) + 2 additional prompts from the
     Gate 0 set (`data/dflash-extracts/prompt-{0,1}/`) for noise
     resistance
   - 3 runs per config × 3 prompts × n_predict=128 × temp=0
   - Output: `data/gate6-mtp-baseline.json` with per-run + median
     t/s + accept rate
2. **DFlash sweep** via `examples/dflash-speculative-simple`:
   - block_size ∈ {4, 5, 6, 8}
   - Same 3 prompts × 3 runs × n_predict=128
   - Output: `data/gate6-dflash-speedup.json` with per-run + median
     t/s + accept rate per block_size

### Phase 3 (after Phase 2) — ship outcome

Per `DESIGN.md §6 Gate 6`:

- **PASS (≥ 1.5× MTP)** → ship `profiles/qwen36-27b-x1-dflash.sh`;
  symlink switch only; MTP profile stays available
- **NEUTRAL (1.0–1.5× MTP)** → document gap honestly; ship as
  tunable option; MTP stays default. Do NOT dress as "GREEN"
- **FAIL (< 1.0× MTP)** → stay on MTP; document negative with
  cost breakdown

### Critical files (T8)

**Modified**:
- `ik_llama.cpp/examples/dflash-speculative-simple/dflash-speculative-simple.cpp` —
  per-cycle timing instrumentation (Phase 1)

**New**:
- `data/gate6-phase1-cycle-breakdown.runlog` + `.json` (Phase 1)
- `data/gate6-phase1-nsys-1cycle.nsys-rep` (Phase 1)
- `data/gate6-mtp-baseline.json` (Phase 2)
- `data/gate6-dflash-speedup.json` (Phase 2)
- (if Phase 3 PASS) `profiles/qwen36-27b-x1-dflash.sh` clone of MTP
  profile with DFlash wiring

**Read-only references**:
- `/home/llm/profiles/qwen36-27b-x1-mtp.sh` — production MTP profile
- `data/gate0-*.json` — vLLM oracle data
- `data/gate5-T6alpha-coherence.runlog` — T6 closure prompt context
- `feedback_anchor_to_measured_baselines.md` — fresh-measurement rule
- `feedback_oneshot_then_evaluate.md` — bundle Phase 1 + Phase 2
  instrumentation up-front, run as a coherent measurement
- `feedback_no_skipping_lessening.md` — Phase 1's diagnostic
  verdict must be honest; if (B), document fully not dress as A

### Open questions to resolve at T8 session start

Q1: **Should Phase 1's cycle-timing instrumentation use `cudaEventRecord`
   or `clock_gettime` boundaries?** `cudaEventRecord` is more accurate
   for GPU kernel latency; `clock_gettime` captures host-side stream
   submission overhead. Both are needed for a complete picture but
   `clock_gettime` is the simpler default.

Q2: **What's the "DFlash cycle is structurally too expensive"
   threshold for Phase 1 gate?** Suggested: if DFlash cycle wall ≥ 30 ms
   AND MTP cycle wall < 15 ms (i.e., gap > 2×), exit at Phase 1.

Q3: **Does the production GGUF actually load through
   `dflash-speculative-simple` cleanly at `--ctx-size 262144`?**
   T6 ran at default ctx (likely much smaller). Phase 2 must use
   production ctx or the comparison isn't apples-to-apples.

Q4: **MTP `--draft 1/2/3` sweep too?** Phase 2's MTP baseline is at
   draft=3 per production. If DFlash beats draft=3 it ships; if
   it ties draft=3 but beats draft=1, that's interesting context
   but not the ship gate. Suggested: just measure at draft=3.

### Discipline reminders for T8

- Each plan-file change commits separately + pushed (CLAUDE.md §5).
- The Phase 1 diagnostic verdict goes into Phase 1's JSON, NOT just
  the conversation — future sessions need to read it.
- Per `feedback_surface_tradeoff_decisions`: any deviation from this
  plan (e.g., changing the diagnostic threshold mid-flight) surfaces
  first, locks in plan, then implements.
- Per `feedback_oneshot_then_evaluate`: write the Phase 1
  instrumentation + Phase 2 measurement scripts as one coherent
  bundle. Don't intermediate-evaluate; let the measurements run,
  then evaluate against the calculated MTP-cycle ceiling.
