---
name: DFlash T7 CLOSED — drafter np-invariance GREEN, "cooperative kernel suspect" didn't exist in code
description: T7 (Gate 5b: drafter kernel np-invariance) closed 2026-05-13 GREEN on production/2026-q2-next via a kernel-level probe (4 seeds × N ∈ {1,2,4,8}, all byte-identical slot 0). Pre-T7 pickup brief flagged `cg::this_grid().sync()` as suspect #1, but reading the .cu source showed the implementation already deviated from spec — uses per-step `__global__` launches with no grid-sync. Probe ran clean on first attempt.
type: project
originSessionId: 9b52ee2b-d969-442c-83ba-98396525f6fa
---
T7 closed on `production/2026-q2-next` (2026-05-13) on the
probe-before-implementing path. Drafter kernels are np-invariant
by direct measurement.

### Closure evidence

- **Test**: `ik_llama.cpp/tests/dflash-speculative/test-dflash-np-invariance.cpp`
- **Runlog**: `data/gate5b-np-invariance-sweep.runlog`
- **Structured**: `data/gate5b-np-invariance.json`
- **Result**: 16/16 sub-runs PASSed (4 seeds × N ∈ {1, 2, 4, 8})
  with **byte-identical** slot 0 outputs within each seed across
  all N values. Hashes vary per seed (0x3eb0…cd53, 0xb183…e2fd,
  0x5b2c…87c5, 0x4499…098f) — confirms not a trivial all-zero pass.

### Key finding — the "suspect" didn't exist in code

The pre-T7 pickup brief (and PHASE_DFLASH bisect order) flagged
`dflash_drafter_forward` cooperative kernel as suspect #1 because
spec `kernel-design.md §6.1` called for `cg::this_grid().sync()`
with potentially grid-size-dependent barrier semantics. **Reading
`ik_llama.cpp/ggml/src/ggml-cuda/dflash/dflash-drafter-forward.cu:8-12`
revealed an explicit spec deviation in source comments**:
implementation uses regular per-step `__global__` launches with
`grid_rows = N_slots × Q` and per-block `reduce_smem[8]` warp+
SMEM-tree reduction. There is no `cg::this_grid().sync()` in the
production code. The deviation is exactly the TML 3-kernel BI
pattern (kernel-design.md §5.5).

The empirical probe witnessed the prediction: byte-identity across
N. Probe ran clean on the first attempt; no bisection was needed.

### Architectural extension chain

The probe stops at `drafter_forward` (hidden state output). The
binding extends to drafter_logits + argmax outcomes by construction:

- `combine_features` + `inject_kv_fused`: T3 sweep already validated
  byte-identity vs CPU oracle at N ∈ {1,2,4,8} (per-anchor CTAs).
- `drafter_lm_head`: takes `n_rows` (not `N_slots`); per-row CTA.
  Byte-identical hidden ⇒ byte-identical logits at slot 0's rows.
- `argmax_match`: per-slot one-warp argmax; no cross-slot reduction.
  Byte-identical logits ⇒ identical `n_accepted` + `bonus_token` +
  `bonus_pos` at slot 0.

### Out of scope for T7

The orthogonal Qwen3.6-27B production-shape np > 1 server-side
determinism bug (see `project_mtp_multislot_determinism_investigation_failed`)
remains an unsolved surface — KV coordination / RoPE / SSM
intermediate state / scheduler / CUDA Graphs interaction. T7
intentionally bypassed the target stack to focus on drafter kernel
invariance. T8 (np=1 speedup) is unblocked; T9 (np > 1 aggregate)
needs that separate bug surface to be navigated.

### Cross-cutting lessons (already in feedback memory)

- `feedback_probe_before_implementing` — yet another empirical
  win: a kernel-level probe at tiny shape closed the gate without
  needing real GGUF weights / production-shape infrastructure.
  Cost: ~15k tokens of probe writing + 0.5 min wall to run.
- `feedback_survey_prior_phase_before_new_mechanism` — the source-
  read revealed T3 had already validated combine + inject's
  invariance; only drafter_forward needed an empirical probe.
- `feedback_source_read_reference_before_instrumenting` — applies
  equally to reading our own implementation. Reading the .cu file
  before launching the bisect plan revealed the suspect didn't
  exist; saved the cost of bisecting a non-bug.

### Related memories

- `project_dflash_t6_closed_via_spec_ckpt.md` — predecessor (T6
  closure via probe-before-implementing on the determinism question)
- `project_dflash_t1_t4_kernel_layer_closed.md` — kernel-layer
  closure including T3 multi-N byte-identity sweep
- `project_mtp_multislot_determinism_investigation_failed.md` —
  the parallel unsolved problem at the target stack (T9 surface)
- `reference_qwen36_27b_mtp_upstream.md` — primary references
