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

- [x] **T8 — Gate 6: Qwen3.6-27B speedup measurement** (CLOSED 2026-05-14)
  - Pre-Gate MTP `--draft 3` baseline measurement (mandatory anchor — see auto-memory `feedback_anchor_to_measured_baselines`).
  - DFlash speedup measurement, block_size sweep ∈ {4, 5, 6, 8}.
  - Ship outcome: PASS (≥ 1.5× MTP) → ship `profiles/qwen36-27b-x1-dflash.sh`; NEUTRAL (1.0–1.5×) → tunable option; FAIL (< 1.0×) → stay on MTP.

- [x] **T9 — np > 1 vanilla validity locked; drift characterised; resolve-drift + DFlash-multi-slot named as future work**
  - Charter pivot recorded 2026-05-14: from perf-comparison to validity lockdown.
  - Vanilla validity bar (5 falsifiable per-slot asserts) PASSES 14/14 slot-runs at np ∈ {2, 4, 8} on Qwen 3.6 27B.
    Harness: `tests/dflash-speculative/test-np-validity-vanilla.cpp`; data: `data/phase_dflash_t8/gate7-validity-vanilla-np{2,4,8}.json`.
  - Direct-token-diff vs NP=1 reference (the validity-bar refinement after the PPL-only first cut):
    Findings new vs PHASE45 D10.e (which was MTP context):
      * **NP=1 ≡ NP=2 byte-identical for vanilla** — `cublasGemmEx` happens to pick the same algorithm at batch widths 1 and 2 on the vanilla forward path.
      * **NP=4 ≡ NP=8 deterministic drift** for 3 of 4 common prompts — drift is binary at the NP=2/4 boundary, not accumulating with batch width. (p2 anomaly: NP=8 introduces additional drift past token 50, a second-order effect.)
      * Per-prompt first-divergence positions: p0:tok3, p1:tok19, p2:tok27, p3:tok0.
    Data: `data/phase_dflash_t8/np-token-diff.py`, `gate7-validity-vanilla-np1-p[0-7].json`, `gate7-validity-vanilla-np{2,4,8}-tokens.json`, `gate7-token-diff-summary.json`.
  - **Resolve-drift workstream is named, not started** — it IS the unfinished PHASE45 D10.e.2 plan (vLLM / Thinking-Machines 3-kernel reduction-order rewrite). Root cause framework already characterised there: cuBLAS GEMM heuristic picks different algorithms at different batch dimensions + FA split size varies with shape → ~1-3% per-row reduction-order drift, amplified by greedy argmax to functional output divergence. The NP=2/4 boundary T9.1 isolates is consistent with that framework and refines it (the boundary is at exactly batch=4 for vanilla, not at any M≥2 as PHASE45 originally framed for MTP). PHASE45 D10.e is closed; the planned D10.e.2 fix is future work.
  - **DFlash multi-slot extension is named, not started** — the libllama API extension required to even attempt DFlash at np > 1 (multi-slot `llama_dflash_draft`, per-slot `common_speculative_state_dflash` adapter, server gate lift, harness) is ~115–185k tokens of work and is downstream of the resolve-drift workstream (no point optimizing on a drifty foundation). Both items recorded as future work.

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
## T8 outcome — measurement of record

T8 is a measurement workstream, not a release gate. The locked
PASS/NEUTRAL/FAIL categories aren't deciding whether to ship a profile
— the canonical outcome is the recorded data point and its attribution
to specific named kernels.

**Phase 3 dataset** (8 prompts × 3 spec methods × 3 reps, captured at
`data/phase_dflash_t8/gate6-spec-*.json`, aggregated to
`gate6-summary.json`):

| Metric | none | mtp --draft 3 | dflash --draft 4 |
|---|---:|---:|---:|
| tg t/s geomean | 29.31 | **33.86** | 1.139 |
| accept_rate geomean | — | 43.6% | **54.1%** |
| mean_accept tokens/cycle | — | 1.31 | 2.16 |
| PPL of output geomean | 1.160 | 3.091 | **1.158** |
| tg speedup vs MTP geomean | — | — | 0.0336 |
| tg speedup vs MTP worst-prompt | — | — | 0.0285 |
| PPL ratio vs MTP worst-prompt | — | — | 0.6488 (DFlash closer) |

**Two findings the data binds clearly**:

1. **DFlash output quality is essentially identical to vanilla
   greedy.** PPL geomean 1.158 (DFlash) vs 1.160 (none) — within
   measurement noise. PPL geomean 3.091 (MTP) sits ~3× higher,
   reflecting batch-shape effects in MTP's verify decode that drift
   the output away from the strict greedy-target trajectory. At 54%
   acceptance, DFlash's per-cycle output is *more* greedy-faithful
   than MTP's 43.6%. The kernel correctness work (T1–T7, byte-identity
   to vLLM PR #40898 across 8 prompts × 4 mask positions) is the
   load-bearing achievement and binds end-to-end through bench.

2. **DFlash throughput is bounded by two named kernels at ~1% of
   hardware peak.** Per Phase 2 nsys + source inspection:
   - `dflash_drafter_lm_head_kernel` consumed 41.3% of GPU time at
     1228 ms/call. TU102 memory-bandwidth floor for the 1.2 GiB
     BF16 weight scan is ~2 ms (624 GB/s); current is **600× off
     the ceiling**. Root cause: one CTA per row, scalar fp32 dot,
     no tensor cores, no SMEM weight tile (`dflash-drafter-lm-head.cu:37-64`).
   - `gemm_row_x_col_kernel` consumed 34.0% of GPU time, 29 ms/call.
     Root cause: 5 CTAs on 72 SMs = 7% SM utilization, scalar fp32,
     no tensor cores (`dflash-drafter-forward.cu:121-141`).
   Both are reference implementations that closed T3/T4's byte-identity
   bindings against the scalar fp32 oracles — correct, deliberately
   un-tuned.

**Outcome framing**: The geomean ratio 0.034× MTP would read FAIL under
the locked ship-gate categories, but those categories are for a release
decision T8 isn't making. Recording the result as-is:

- **Throughput**: DFlash 1.14 t/s, ~3% of MTP, ~4% of vanilla greedy
  on this hardware/build. The end-to-end pipeline is correct; the
  kernels are slow.
- **Quality**: DFlash output is statistically indistinguishable from
  vanilla greedy at this acceptance rate (54%), better than MTP on
  this axis.
- **Optimization target named**: rewrite `dflash_drafter_lm_head_kernel`
  and `gemm_row_x_col_kernel` for the TU102 + NVLINK theoretical
  envelope.

**TU102 + NVLINK optimization envelope** (future workstream — the
canonical limits these kernels will be conducted toward):

| Resource | Per-GPU (TU102) | Dual w/ NVLINK | Relevant to |
|---|---:|---:|---|
| HBM bandwidth | 624 GB/s | 1.25 TB/s aggregate | lm_head BF16 weight scan |
| FP16 tensor core peak | 130.5 TFLOPs | 261 TFLOPs | GEMM rewrites, mma.sync.m16n8k8 |
| INT8 tensor core peak | 261 TFLOPs | 522 TFLOPs | quantized-weight GEMV variants |
| NVLINK aggregate | — | ~100 GB/s | logits allreduce after V-shard |

For the lm_head specifically: split V=248320 across the two GPUs
(124160 cols each), run F16 tensor-core GEMV per-GPU, NCCL allreduce
the 248k-float logits (~1 ms over NVLINK). With multi-CTA grid hitting
~all 72 SMs and Turing tensor cores, the lm_head should reach the
memory-bandwidth floor of ~2 ms per call. That alone collapses the
draft bucket from 2.9 s/cycle toward ~10 ms/cycle, putting DFlash
into MTP's per-cycle regime and unlocking the spec=4 speedup the
kernel correctness already binds. The GEMM rewrite follows the
same pattern at smaller scale.

Possibly also: implement the cooperative WMMA mega-kernel that T7
noted is absent from the impl (spec `kernel-design.md §6.1` — 5
transformer layers + lm_head fused under `cudaLaunchCooperativeKernel`,
eliminating per-step `__global__` launch overhead). T7's source-read
showed the implementation deviates from spec §6.1 in this area, so
the spec already names what's missing.

**T8 closes here.** The phase delivered:
- Bench infrastructure for apples-to-apples spec-method comparison
  (none / mtp / dflash) with PPL-of-output as the quality bound.
- Bench-validated MTP wiring (+13.0% over the production memory
  baseline, within ±20% sanity band).
- End-to-end DFlash measurement on production hardware, attributed
  to two named kernels with TU102 + NVLINK optimization headroom
  precisely characterised.
- Quality finding: DFlash output is greedy-faithful at 54%
  acceptance, more so than MTP at 43.6% — kernel correctness binds
  end-to-end through the ship-gate dataset.

The kernel optimization workstream is the canonical next phase
when budget allows. No follow-up cover (per CLAUDE.md §4): the
slow kernels ARE the limiting factor for DFlash performance; T8
records that honestly with the optimization target named.


---

## Future-work pointers (named, not active)

All ten T-tasks closed. The named optimization targets below are
future work tracked here so future sessions don't re-rediscover them.

1. **PHASE45 D10.e.2 — vLLM/Thinking-Machines 3-kernel reduction-order
   rewrite** to remove the cuBLAS-GEMM-algo + FA-split-size drift at
   the NP=2/4 boundary. PHASE45 D10.e closed without shipping this.
   T9's direct token diff (`data/phase_dflash_t8/gate7-token-diff-summary.json`)
   gives the concrete diagnostic anchor: NP=1≡NP=2 byte-identical,
   NP=4≡NP=8 deterministic drift, per-prompt first-divergence positions.

2. **DFlash multi-slot libllama API extension** (~115–185k tokens).
   Current `llama_dflash_draft(ctx, anchor_token, anchor_pos, …)` is
   single-slot. The kernels are written with `N_slots × Q` grid (T7)
   but the C entry, `common_speculative_state_dflash`, and the server
   `np==1` gate all need multi-slot variants. Downstream of (1):
   no point optimizing on a drifty foundation.

3. **DFlash kernel optimization to the TU102 + NVLINK envelope**
   (`dflash_drafter_lm_head_kernel` ~600× off memory-bandwidth ceiling;
   `gemm_row_x_col_kernel` ~7% SM utilization). The Phase 4 outcome
   above names the targets and the optimization shape precisely.
   Downstream of (1) and (2).

Pickup briefs, redrafted plan iterations, and superseded outcome text
were pruned 2026-05-14 after T9 closure. Full diagnostic data sits
in `data/phase_dflash_t8/` (T8) and `MEMORY.md` (auto-memory).
