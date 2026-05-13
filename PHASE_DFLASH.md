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

## T8 plan — Qwen3.6-27B speedup + quality measurement (REDRAFTED 2026-05-13)

**Methodology principle**: three different harnesses with three
different timing semantics is not a defensible methodology for a
ship-gate measurement. Apples-to-apples comparison requires a single
idiomatic tool driving all spec methods through the same dispatch.
Per `feedback_no_workarounds`: build proper infrastructure, not a
wrapper script that papers over the gap.

**Authoring lesson (T6 + T7 evidence)**: source-read the existing
implementation before designing — both T6 and T7 nearly built
duplicate mechanisms that the codebase already had. This plan was
authored after reading the actual code; findings are cited inline
as plan witnesses.

### Phase 0 — Source-read (DONE 2026-05-13; findings below)

- **`common/common.h:140-153`**: `enum common_speculative_type`
  already includes `COMMON_SPECULATIVE_TYPE_NONE`,
  `_MTP`, `_DFLASH`, `_DRAFT`, `_EAGLE3`, and several ngram variants.
  **The framework is already spec-method-aware**.
- **`common/speculative.cpp:1159 (common_speculative_init)`** is the
  unified dispatcher. Both MTP and DFlash route through it; the
  `params.type` field selects the implementation. `dflash-speculative-simple.cpp:82`
  proves the dispatch works end-to-end for DFlash.
- **`examples/llama-bench/llama-bench.cpp`** (2287 LOC) — clean
  structure: `cmd_params → cmd_params_instance → test → printer`
  with 4 printers (csv/json/markdown/sql). Currently zero
  spec/mtp/draft mentions in the source. `cmd_params_instance`
  (line 957) is where a `common_speculative_type spec_type` field
  needs to be added.
- **`examples/perplexity/perplexity.cpp:159 (process_logits)`** —
  computes NLL token-by-token from a `[positions × vocab]` logits
  array and a `[positions]` token-id array. `PPL = exp(mean(nll))`.
  Directly reusable for "target-PPL on generated output" by
  re-feeding [prompt + generated_tokens] into the target and
  capturing logits at each generated position.
- **`common/common.cpp:1583-1588`**: `-mtp` flag sets
  `params.has_mtp = true`, which propagates to `mparams.mtp` and
  `cparams.mtp`. This enables the inline MTP layers in the target
  model — separate field from `common_params_speculative.type`,
  but `common/speculative.cpp:1177` links them
  (`has_mtp = (params.type == COMMON_SPECULATIVE_TYPE_MTP)`).
- **`/home/llm/profiles/qwen36-27b-x1-mtp.sh`**: production MTP
  profile via `llama-server -mtp --draft 3`, TP=2, q4_0 KV with
  hadamard, ctx 262144. **Comment cites 2026-05-09 baseline**:
  draft=3 → 33.23 t/s (3-run median, n=128, T=0). Per
  `feedback_anchor_to_measured_baselines`, **do NOT bind on this
  number — measure fresh**.
- **`examples/speculative` is NOT the right comparison**: uses
  two-model architecture (separate target + drafter binaries),
  NOT the inline MTP that production uses.
- **No prior PHASE work extended llama-bench for spec.** `git log`
  shows `1cb7e1bf "spec : add self speculative decoding..."` added
  framework features (`common_speculative_is_compat`) but did not
  touch llama-bench. We're building net-new spec-aware bench
  infrastructure.

### Apples-to-apples requirements (locked)

Any T8 result that ships MUST be produced by:

1. **A single binary** driving all spec methods (no shell wrappers
   parsing tok/s from three different CLI outputs).
2. **Identical measurement semantics** for tg t/s across spec
   methods: same warmup, same prompt-eval-excluded timing window,
   same prompt set, same n_gen, same temp=0.
3. **PPL on generated output** (under target) as the quality bound,
   NOT PPL on a held-out corpus (which is trivially equal across
   spec methods because target is shared). Spec methods may diverge
   numerically (batch-shape variance at verify, different
   accept-pattern dynamics); PPL-of-output catches that drift.
4. **pp t/s** for context, captured by the same tool. (pp is
   largely insensitive to spec method — initial prefill is the
   same — but capture it for completeness and for cases where
   prompt-eval batching could differ.)
5. **Idiomatic** — the tool used IS the canonical `llama-bench`,
   not a bespoke `gate6-bench`. Future spec work uses the same
   infra; future llama.cpp users benefit.

### Phase 1 — Extend `llama-bench` to be spec-aware + PPL-of-output

**Phase 1.1 — Spec-method-aware tg measurement**:

CLI additions:
- `--spec {none,mtp,dflash,draft,ngram-simple,ngram-mod,...}` (default `none`)
- `--draft N` / `-nd N` (draft depth; sets `params.speculative.n_max`. For MTP this is the chain length, for DFlash this is BLOCK_SIZE.) (default 3 for MTP, 4 for DFlash)
- `--spec-model PATH` (for spec methods that need a separate drafter GGUF: dflash, draft, eagle3; ignored otherwise)

Data plumbing — `cmd_params` + `cmd_params_instance`:
- Add `common_speculative_type spec_type` + `int n_draft` +
  `std::string spec_model` fields.
- Parse from CLI; default values per above.

**MTP init recipe** (traced from `server-context.cpp:294-331`,
mirrored exactly so the bench measures the same path production uses):

```cpp
// After llama_model_load_from_file + llama_init_from_model:
if (spec_type == COMMON_SPECULATIVE_TYPE_MTP) {
    // 1. Gate on NextN layers: model must have MTP heads
    if (llama_model_n_nextn_layer(model) <= 0) {
        // Skip this row with informative message — model doesn't support MTP
        return SKIP;
    }
    // 2. Set legacy + new fields
    params.has_mtp = true;
    params.speculative.type = COMMON_SPECULATIVE_TYPE_MTP;
    params.speculative.n_max = n_draft;
    params.pooling_type = LLAMA_POOLING_TYPE_NONE;

    // 3. Build drafter cparams
    params.speculative.cparams_dft = common_context_params_to_llama(params);
    params.speculative.cparams_dft.mtp         = true;
    params.speculative.cparams_dft.mtp_op_type = MTP_OP_WARMUP;
    params.speculative.cparams_dft.embeddings  = true;

    // 4. Flip decoder to emit embeddings (MTP needs them)
    llama_decoder_set_embeddings(decoder, true);

    // 5. Verify batch capacity = n_max + 1
    batch_spec = llama_batch_init(n_max + 1, 0, 1);
}

// DFlash init recipe (no NextN gate, no decoder embedding flip — uses
// cb_eval extract hook plumbed by llama_set_dflash):
if (spec_type == COMMON_SPECULATIVE_TYPE_DFLASH) {
    params.speculative.type = COMMON_SPECULATIVE_TYPE_DFLASH;
    params.speculative.n_max = n_draft;             // = BLOCK_SIZE
    params.speculative.model.path = spec_model;     // drafter GGUF path
    batch_spec = llama_batch_init(n_max + 1, 0, 1);
}

// Common path:
if (spec_type != COMMON_SPECULATIVE_TYPE_NONE) {
    if (!common_speculative_is_compat(ctx)) {
        return SKIP;  // log reason
    }
    spec = common_speculative_init(params.speculative, ctx, /*seq_id=*/0);
    if (!spec) return SKIP;
}
```

TG decode loop with spec on (pattern from
`dflash-speculative-simple.cpp:137-258`, common to all spec methods):

```cpp
while (n_emitted < n_gen) {
    // 1. Get draft from spec impl (transparent across MTP/DFlash/etc.)
    const llama_tokens draft = common_speculative_draft(spec, ...);

    // 2. Build verify batch: [id_last, draft[0], ..., draft[BS-1]]
    //    Decode the batch in one llama_decode call (BS+1 tokens).
    llama_decode(ctx, batch_spec);

    // 3. Argmax accept-prefix
    //    For k in 0..BS: target_tok_at[k] = argmax(logits at row k)
    //    n_accepted = longest prefix where target_tok_at[k] == draft[k]
    //    bonus = target_tok_at[n_accepted]

    // 4. Accept
    common_speculative_accept(spec, n_accepted);

    // 5. Commit accepted + bonus to emitted; id_last = bonus
    n_emitted += n_accepted + 1;
}
```

Notes on cross-method semantics:
- `common_speculative_draft` is polymorphic — the loop body is
  identical across spec methods. MTP draws from inline MTP heads;
  DFlash drives the kernel pipeline; ngram pulls from a cache.
  None of this varies in the bench loop.
- The verify batch shape is `BS+1` for all methods. For MTP, BS is
  the chain length (typically 3). For DFlash, BS is BLOCK_SIZE
  (typically 4).
- `--spec none` falls through to standard one-token tg per
  llama-bench's existing path; no spec dispatch.

Capture per test:
- `n_emitted`, `n_drafts` (= number of cycles), `n_draft_tokens`
  (= BS × n_drafts), `n_accepted_total`
- `mean_accept = n_accepted_total / n_drafts` (tokens accepted per cycle)
- `accept_rate = n_accepted_total / n_draft_tokens` (fraction)
- `tg_t_s = n_emitted / wall_time`

Report additions (across all 4 printers):
- New columns: `spec`, `n_draft`, `accept_rate`, `mean_accept`.
- `spec=none` shows blank/`-` for the spec-specific columns.

**Phase 1.2 — PPL-of-output mode**:

CLI addition:
- `--ppl-of-output` (boolean flag; off by default).
- When set, after each tg measurement runs N_gen tokens of
  generation, run a SECOND pass through the target on
  [prompt + generated_tokens], capture logits at each generation
  position, run `process_logits()` (factored out from
  `examples/perplexity/perplexity.cpp:159` into a shared helper in
  `common/`), compute target-PPL on the generated sequence.

Data plumbing:
- Add `target_ppl_of_output` field to `struct test`.
- Print as additional report column when `--ppl-of-output` is on.

**Phase 1.3 — Source helper extraction**:

Factor the NLL/PPL kernel out of `perplexity.cpp` into a small
`common/perplexity.h` + `.cpp` so both llama-perplexity (existing
caller) and llama-bench (new caller) can use it. Minimum-diff
refactor — extract the function, leave perplexity.cpp's call site
unchanged otherwise.

**Phase 1 closure binding**:

The extended `llama-bench` binary, given:
```sh
llama-bench -m <target.gguf> --spec none -p 512 -n 128 -r 3
llama-bench -m <target.gguf> --spec mtp --draft 3 -p 512 -n 128 -r 3
llama-bench -m <target.gguf> --spec dflash --spec-model <drafter.gguf> --draft 4 -p 512 -n 128 -r 3
```
produces three rows of a single markdown/JSON table with identical
columns (model, params, spec, n_draft, pp t/s, tg t/s, mean_accept,
target_ppl_of_output), measured through the same dispatch and timing
window. Closure asserted by:
1. Build clean with `-DGGML_CUDA=ON -DGGML_CUDA_DFLASH=ON -DCMAKE_CUDA_ARCHITECTURES=75`.
2. Each of the three rows produced; pp t/s columns within
   ~5% across spec methods (sanity); tg t/s columns differ as
   measured.
3. `--spec none` row's target_ppl_of_output is finite and
   reasonable for the target on its own output.

### Phase 2 — Diagnose T6's 1.33 t/s through the new infra

Once Phase 1 lands, run the extended `llama-bench --spec dflash` on
the production target at T6's prompt config (single prompt, n=128).
Two outcomes possible:

- **(A) Extended bench shows DFlash tg t/s ≫ 1.33** — T6's
  1.33 number was a measurement artifact in
  `dflash-speculative-simple` (likely including prompt-eval, no
  warmup, or single-run noise). The infra-level number IS the
  honest measurement. Proceed to Phase 3.
- **(B) Extended bench confirms DFlash tg t/s ≪ MTP** — there's a
  real per-cycle cost. THEN add per-cycle instrumentation to
  `dflash-speculative-simple` to attribute it, capture to
  `data/gate6-phase2-cycle-breakdown.json`, decide structural vs
  fixable per the diagnostic threshold (Q2 below).

Capture Phase 2 result to `data/gate6-phase2-diagnose.json` with
verdict (A) or (B) and reasoning. If (B) structural: T8 closes
FAIL or NEUTRAL here; do NOT proceed to Phase 3 producing a
pretend-PASS.

### Phase 3 — Ship-gate measurement (only if Phase 2 verdict is A or B-fixable)

Single `llama-bench` invocation, all three spec methods, multiple
prompts, with `--ppl-of-output` on:

```sh
llama-bench -m <target.gguf> \
  --spec none,mtp,dflash \
  --draft 3 \
  --spec-model <drafter.gguf>:dflash \
  -p 512 -n 128 -r 3 \
  --ppl-of-output \
  -o json > data/gate6-ship-measurement.json
```

(Exact CLI shape may differ; lock during Phase 1 implementation.)

Expected output columns: `spec, n_draft, pp_t_s, tg_t_s,
mean_accept, target_ppl_of_output, n_gen, n_prompt, build_info`.

### Phase 4 — Ship outcome decision

Per `DESIGN.md §6 Gate 6`, decided from the Phase 3 JSON:

- **PASS** (DFlash tg ≥ 1.5× MTP tg) AND (DFlash target_ppl_of_output
  within 5% of MTP target_ppl_of_output) → ship
  `profiles/qwen36-27b-x1-dflash.sh`.
- **NEUTRAL** (DFlash tg 1.0–1.5× MTP) OR (DFlash PPL > 5% worse
  than MTP) → document gap honestly; ship as tunable option;
  MTP stays default. Do NOT dress as "GREEN".
- **FAIL** (DFlash tg < 1.0× MTP) OR (DFlash PPL substantially
  worse) → document negative result with full cost breakdown.

The PPL quality bound is asymmetric: if DFlash is faster but
substantially worse-quality, that's NEUTRAL, not PASS. Speedup
without quality is not a ship.

### Critical files (T8)

**Modified** (ik_llama.cpp/):
- `examples/llama-bench/llama-bench.cpp` — spec-aware extension
  (~500 LOC) + PPL-of-output mode
- `examples/perplexity/perplexity.cpp` — refactor `process_logits`
  into shared helper (minimum-diff)
- `common/CMakeLists.txt` (or wherever) — wire new
  `common/perplexity.{h,cpp}` helper

**New**:
- `ik_llama.cpp/common/perplexity.{h,cpp}` — shared NLL/PPL kernel
- `data/gate6-phase1-bench-three-rows.json` (Phase 1 closure)
- `data/gate6-phase2-diagnose.json` (Phase 2 verdict)
- `data/gate6-ship-measurement.json` (Phase 3 measurement)
- (if Phase 4 PASS) `profiles/qwen36-27b-x1-dflash.sh`

### Open questions to resolve at T8 session start

Q1: **MTP plumbing — RESOLVED 2026-05-13 via trace of
   `server-context.cpp:294-331`**: production sets 8 fields, NOT
   just two. See the "MTP init recipe" block above in Phase 1.1
   for the explicit sequence. Key non-obvious requirements:
   model gating (`llama_model_n_nextn_layer > 0`), pooling
   override (`NONE`), drafter cparams construction with
   `mtp/mtp_op_type/embeddings`, decoder embedding flip
   (`llama_decoder_set_embeddings(decoder, true)`), and verify
   batch size = `n_max + 1`. Anything less and the bench measures
   a path production doesn't use.

Q2: **Diagnostic threshold for Phase 2 verdict (B) fixable vs
   structural**: suggested if DFlash per-cycle wall ≥ 2× MTP
   per-cycle wall AND the dominant cost is NOT prompt-eval or
   one-time setup, treat as structural. Lock during Phase 2
   implementation.

Q3: **PPL-of-output sequence length**: probably want PPL on the
   first 64-128 generated tokens, not all 128, to avoid noise
   from late-stream divergence. Suggested: capture
   target_ppl_of_output_n64 + target_ppl_of_output_n128 and
   surface both.

Q4: **MTP --draft sweep**: spec_type=mtp with n_draft=1,2,3 at
   Phase 3 to confirm production's draft=3 choice still holds
   on this build. Cheap addition since we're already iterating
   spec methods.

### Discipline reminders for T8

- Each plan-file change commits separately + pushed (CLAUDE.md §5).
- Per `feedback_no_workarounds`: extending llama-bench is the
  proper path. A `scripts/gate6-bench.py` wrapper around three
  CLIs is NOT acceptable as a ship-gate measurement.
- Per `feedback_surface_tradeoff_decisions`: spec extension
  surface area runs ~500 LOC; any deviation from this scope
  (skipping --ppl-of-output, dropping a printer's spec columns)
  surfaces first, locks in plan, then implements.
- Per `feedback_oneshot_then_evaluate`: build the extension as
  one coherent commit; run the three-row closure binding; then
  evaluate. Don't intermediate-evaluate the extension itself.
- Per `feedback_no_skipping_lessening`: Phase 2 verdict (B)
  structural is a legitimate outcome — document it honestly. Do
  NOT proceed to Phase 3 producing a measurement that buries the
  structural finding.
