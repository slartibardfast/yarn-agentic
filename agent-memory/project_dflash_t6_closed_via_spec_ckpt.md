---
name: DFlash T6 CLOSED via llama_spec_ckpt_* PER_STEP — T6.D avoided
description: T6 (Gate 5: 27B np=1 determinism + late-stream coherence) closed 2026-05-13 both gates GREEN on production/2026-q2-next. Wired the example to llama_spec_ckpt_* (PER_STEP) + bonus-as-next-cycle-batch[0] (no separate single-token bonus decode). T6.D dflash_verify_attn from scratch deferred and ultimately not needed.
type: project
originSessionId: 9b52ee2b-d969-442c-83ba-98396525f6fa
---
T6 closed on `production/2026-q2-next` (2026-05-13) on the
probe-before-implementing path. Both gates GREEN without building
the planned dflash_verify_attn kernel.

### Closure evidence

- **T6.α coherence** (data/gate5-T6alpha-coherence.runlog):
  Prompt `Write a short python function for quicksort`, n=128, BS=4,
  temp=0. Mean accept rate **2.879** tokens/draft (T5 baseline:
  2.256, +28%). 33 cycles, 132 drafts, 95 accepted (72%). Output
  coherent through full n_predict — matches target-only's
  thinking-process structure exactly. No late-stream repetition tail.

- **T6.β determinism** (data/gate5-T6beta-determinism.json):
  Three runs at temp=0, np=1 → SHA-256 byte-identical:
  `6c207f9b3d7dc98e128a820490fedcb84f30778d068de167c1db23b2df8a67f3`.
  Empirical: target's existing CUDA attention is deterministic at
  fixed batch shape. `dflash_verify_attn` from scratch NOT NEEDED.

### Key design

Per-cycle protocol in `examples/dflash-speculative-simple`:
1. `llama_spec_ckpt_save(ctx, 0)` — shadow + arm `save_per_step_ssm`
2. `llama_decode(verify_batch)` — BS+1 tokens; delta_net auto-populates
   `per_step_ssm[il]` and `per_step_qkv[il]` (delta_net.cpp:152, 627)
3. accept-prefix from per-position target argmax
4. `llama_spec_ckpt_restore(ctx, 0, n_past=P, accepted_step=n_accepted)`
   — `per_step_restore` stitches `s_l[il]` to "after id_last +
   n_accepted drafts" + matching `seq_rm` to P+n_accepted+1
5. `llama_dflash_trim_extract(ctx, P+n_accepted+1, -1)` — sync cb_eval
   buffer to seq_rm
6. emit accepted drafts + bonus; **id_last ← bonus** (NO separate
   single-token bonus decode)

The bonus becomes `batch[0]` of the NEXT cycle's verify batch.
Consistent BS+1 batch shape every cycle eliminates the batch-shape
K, V variance that broke the original T6.B commit re-decode.

**`llama_spec_ckpt_discard` is NOT called between cycles** — discard
resets `selected_spec_mode = NONE` and breaks the next save.
`save_per_step_ssm` stays on; every multi-token decode in the loop is
exactly BS+1=5 which matches the allocated per-step buffer.

### Restructured T6 layout (final)

- **T6.A REMOVED** post-closure. Parallel ping-pong DeltaNet
  snapshot (~150 MiB GPU memory) was functionally redundant with
  `gpu_checkpoint.s_l_shadow`. The C API + scratch + smoke test
  were deleted; resurrectable from git history.

- **T6.α / T6.β** — closure gates above. New witness test
  `tests/dflash-speculative/test-dflash-spec-ckpt-flow.cpp`
  drives the spec_ckpt cycle protocol on a synthetic forward and
  asserts `kv_cache_seq_pos_max(0) == n_past + accepted_step`.

- **T6.C kept**: `llama_dflash_trim_extract(ctx, p_start, p_end)`
  syncs the cb_eval extract buffer to seq_rm.

- **T6.D DEFERRED CONDITIONAL → not needed**. dflash_verify_attn
  from scratch (~80-150k tokens of sm_75 PTX work) was the original
  plan for the determinism gate; T6.β closed empirically without it.

### Mid-T6 lessons that became feedback memories

- `feedback_survey_prior_phase_before_new_mechanism` — T6.A nearly
  rebuilt `llama_spec_ckpt_*` (our own prior PHASE41/45 work).
- `feedback_bisect_before_revert` — T6.B regression was bisected
  only after user pushback ("you must get to the bottom of the
  issue"). Bisect-1 (snap+restore alone, 1.909) revealed snapshot
  itself was perturbing state, not just the re-decode.
- `feedback_probe_before_implementing` — strongest empirical
  validation yet: the 5-min 3-run SHA probe gated 100k+ tokens of
  avoidable CUDA PTX work.

### Authoritative artifacts

- Branch: `production/2026-q2-next` on both yarn-agentic and
  ik_llama.cpp submodule
- Closure data: `data/gate5-T6alpha-coherence.runlog`,
  `data/gate5-T6beta-determinism.json`,
  `data/gate5-T6alpha-postremoval.runlog`
- Spec: `specs/dflash/kernel-design.md §6.4` (notes T6.A removed,
  llama_spec_ckpt_* canonical), `specs/dflash/dflash.allium` +
  `specs/dflash/allium-tla-binding.json` (3 invariants repointed at
  test-dflash-spec-ckpt-flow.cpp)
- Tracker: `PHASE_DFLASH.md` (T6 `[x]`)
- Public log: `MEMORY.md` — T6 closure entry committed 2026-05-13

### Next gates

- **T7**: drafter np-invariance binding at np ∈ {1, 2, 4, 8}.
  Multi-slot may force GPU_FALLBACK (PER_STEP only at n_seq_max=1).
  Worth profiling whether non-PER_STEP path still closes coherence.
- **T8**: Qwen3.6-27B speedup measurement vs MTP baseline. Pre-T8
  fresh MTP --draft 3 measurement required per
  `feedback_anchor_to_measured_baselines`.

### Related memories

- `project_dflash_t1_t4_kernel_layer_closed.md` — kernel-layer
  closure (terminal). Not superseded by T6.
- `reference_vllm_v1_subprocess_patches.md` — vLLM oracle setup.
- `feedback_validate_gguf_dtype_at_load.md` — root NaN bug from T4.
- `feedback_source_read_reference_before_instrumenting.md` —
  pattern that generalises to "source-read before code change",
  cousin to feedback_survey_prior_phase_before_new_mechanism.
