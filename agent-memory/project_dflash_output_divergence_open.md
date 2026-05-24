---
name: dflash-output-divergence-root-cause-reopened
description: "P0.A.3 RESOLVED 2026-05-21. Root cause: MMQ I=8 split-K kernel (mul_mat_q_split_k_i8 with mma_int_C_I8J8 fragment) is byte-shape-invariant for OUTPUT column 0 only — columns ≥1 in a multi-token same-slot batch produce different fp32 bits than same input at col 0 of single-token dispatch. Drift compounds through ~64 transformer layers, flipping argmaxes. Fix: i8_shape_supported = false at ggml/src/ggml-cuda/mmq.cuh:4986. Eight regression tests landed on production/2026-q2-next (submodule HEAD ~8e233e9b), all PASS. DFlash CLI now produces coherent output ('The capital of Germany is Berlin. The capital of Italy is Rome...') with mean accept 2.30/4. Cost: some decode TG perf hit. Re-enabling I=8 requires fixing the col-j>0 FMA accumulation in the mma_int_C_I8J8 fragment to match col-0's order."
metadata:
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

DFlash CLI on production/2026-q2-next post-fold submodule produces
**degenerate output** ("user user wants wants a a quick quick quick")
that diverges from spec-none baseline at the first multi-token
verify-batch decode. Reproducible with same prompt + seed + temp.

**Variables ruled out** (tested 2026-05-20):
- Q4_0 quantization (also fails with f16 KV)
- Hadamard rotation (also fails without it)
- temperature (fails at both temp=0 and temp=0.6)
- GEMV-vs-GEMM accumulation ULP (would be statistically dampened by
  rejection sampling at temp>0)
- **cb_eval install + scheduler slow-path + l_out-<il> capture
  (FALSIFIED 2026-05-20 by binding observational test — see below)**

**Falsification of cb_eval-as-cause (2026-05-20):**
Wrote a libllama-level binding test that decodes the same prompt
twice with `llama_set_dflash_extract_layers` armed on the
production layer set {1, 16, 31, 46, 61} vs disarmed, and compares
per-row argmax. Three shapes — single 12-token prefill, prefill
+ 64 single-token autoregressive decodes, prefill + 32 verify-style
5-wide multi-token decodes (160 generated tokens) — all show
**byte-identical argmax** with extract armed. The extract buffer
populates correctly (61440 floats = 12 rows × 5120 dim, confirming
cb_eval fires on every layer). The cb_eval install alone does NOT
produce the divergence seen in DFlash CLI output.

The bundle from 2026-05-20 (Allium + TLA+ + binding test) is held on
disk uncommitted at:
- `specs/dflash/cb_eval_residual_capture.allium`
- `specs/dflash/CbEvalObservational.tla`
- `specs/dflash/CbEvalObservationalMC.{tla,cfg}`
- `specs/dflash/CbEvalObservationalMC_callback.cfg`
- `ik_llama.cpp/tests/dflash-speculative/test-dflash-extract-observational.cpp`

The Allium spec parses clean. The TLA+ positive config verifies; the
negative config (Mechanism = SCHEDULER_CALLBACK) produces the
expected counterexample on SchedulerStaysFastPath. The libllama
test passes on HEAD. They encode a legitimate contract
(extraction is observational on cb_eval armed/disarmed) and bind on
that contract — but the contract is NOT the P0.A.3 fix path; cb_eval
is observational on the surface the test exercises. Revive when the
real mechanism is named so the spec can be retargeted.

**2026-05-21 CLOSURE — fix landed.**
- L4 (`tests/dflash-speculative/test-dflash-per-layer-batch-shape-diff`):
  per-layer divergence localiser. Captures l_out at every 4th layer
  for path A (two 1-token decodes) vs path B (one 2-token batch).
  Row 0 byte-identical at every layer; row 1 diverges at layer 0
  (|Δ|=1.0) compounding to |Δ|=154 at layer 63. Bug is INSIDE layer 0.
- L5 (`tests/dflash-speculative/test-mulmat-batch-shape-invariance`):
  kernel-direct ggml_mul_mat(Q4_0, F32) at production K=5120 N=8192.
  Col 0 byte-invariant across ne11 ∈ {1, 2, 8}. Col-1 input data
  through ne11=1's col 0 differs from same data through ne11=2's col 1
  by 8192/8192 floats, max |Δ|=0.363. Confirmed MMQ kernel root cause.
- Diagnostic disable of MMQ I=8 path (i8_shape_supported = false at
  ggml/src/ggml-cuda/mmq.cuh:4986) → L5 immediately PASSES, all
  binding tests PASS, dflash-speculative-simple produces coherent
  output. Fix committed as permanent (with detailed comment).
- The MMQ I=8 path was added in PHASE 71-74 for sm_75 decode TG perf.
  Its NPC verification compared col 0 across concurrent multi-slot
  single-token — did NOT exercise col j>0 in multi-token same-slot.
  Real test-coverage gap: NP-determinism gates verify cross-slot at
  n_tokens=1 per slot, NOT cross-shape (n_tokens=1 vs n_tokens=N) for
  same slot. Both must be tested; the latter wasn't.

**2026-05-21 update — L3 BINDS the actual mechanism: batch-shape variance.**
- L3 (`tests/dflash-speculative/test-dflash-multi-cycle-restore-drift`):
  rewritten as a clean batch-shape invariance probe after the first
  multi-cycle run showed cycle-0 already produced mismatched argmaxes.
  Pure libllama, no spec_ckpt, no DFlash pipeline. Compares
  verify-batch row-k argmax to autoregressive at the same effective
  context. **FAIL on HEAD: 10/25 rows mismatched across 5 windows × 5
  tokens.** Test stays FAIL until the variance is fixed.
- Implication: the DFlash CLI's degenerate output is at least partly
  caused by verify-batch producing sequences that diverge from
  autoregressive. The drafter supplies tokens; verify-batch accepts/
  rejects by its own argmax (different from autoregressive); the
  bonus sequence is what verify-batch decodes. If that's degenerate,
  the CLI is degenerate.
- K1' said the DeltaNet kernel is byte-equivalent across n_tokens.
  So the variance enters at a layer K1' doesn't reach — most likely
  the FA per-slot KV singlewarp dispatch at n_tokens=5 same-slot
  (uncovered by production NPC gates which test concurrent
  single-token-per-slot, not same-slot multi-token), OR the
  graph-build conditional at delta-net.cpp:380-389 (permute/L2-norm
  ordering branches on n_seq_tokens > 1).
- Production NP-determinism gates DO NOT cover this case. They
  verify cross-NP byte-identity at n_tokens=1 per slot; they do
  NOT verify n_tokens=N same-slot vs n_tokens=1 same-slot.
- Open question: how does MTP-IR work at production np=1 if
  verify-batch ≠ autoregressive? L3' (sweep verify_bs) would
  identify whether MTP's shape avoids the variance.

**2026-05-21 update — K1' + L2 PASS, Suspect 4 falsified.**
- K1' (`tests/dflash-speculative/test-deltanet-save-all-steps-intermediate`):
  per_step_ssm[k] from an n=5 save_all_steps=true run byte-equals the
  final state of a fresh n=k+1 save_all_steps=false run, for all
  k ∈ {0..4} at production DeltaNet geometry (262144 fp32 floats per
  comparison). The CUDA kernel writes correct intermediate per-step
  state at every iteration — the source data per_step_restore reads
  is bit-perfect.
- L2 (`tests/dflash-speculative/test-dflash-per-step-restore-byte-identity`):
  Bonus-decode logits at pos P+3 are byte-identical between a fresh
  3-token decode path (no spec_ckpt) and a 5-token verify-batch path
  with save_per_step_ssm=true + llama_spec_ckpt_restore(accepted_step=2)
  on Qwen 3.6 27B q4_0 Hadamard dual-GPU. 248320 fp32 logits match,
  argmax A=B=13. The full save→verify→restore→bonus chain is
  observationally equivalent to a clean (accepted_step+1)-token
  decode at the libllama API surface.
- K1 layout caveat discovered while writing K1': K1's beta/g tensors
  were ggml-contiguous (t-fast) rather than production's h-fast
  layout post-permute. K1's PASS still binds "save_all_steps branch
  is neutral" because both modes read the same non-production layout
  symmetrically. K1' fixes the layout via slice_first_n_tokens_h_fast.
- Implication: the CLI failure cannot be the save→restore chain in
  isolation. Must involve drafter pipeline (combine_features,
  inject_kv_fused, drafter_forward, drafter_lm_head) OR multi-cycle
  drift across many save→restore cycles. Next tests: L3 (multi-cycle
  loop, identity drafter); L4 (full drafter pipeline with deterministic
  draft tokens vs spec-none).

**2026-05-21 update — L1 + K1 PASS, Suspect 2 falsified.**
- L1 (`tests/dflash-speculative/test-dflash-save-per-step-ssm-observational`):
  toggling `save_per_step_ssm = true` via
  `llama_spec_ckpt_init(PER_STEP) + llama_spec_ckpt_save(0)` before a
  verify-style 5-row decode produces BYTE-IDENTICAL per-row argmax to
  the disarmed control on Qwen 3.6 27B q4_0 Hadamard dual-GPU. The
  per-step buffer (364 MiB) and shadow buffer (74 MiB) allocate and
  fire correctly during the armed run.
- K1 (`tests/dflash-speculative/test-deltanet-save-all-steps-last-state`):
  `ggml_delta_net` at production geometry (HEAD_DIM=128, H_V=16,
  H_K=2, n_tokens=5, n_seqs=1) produces byte-identical output rows
  AND last per-step state across `save_all_steps ∈ {true, false}`.
- Implication: the CLI Run E bisect ("LLAMA_NO_SPEC_CKPT_SAVE=1
  produces different degenerate output") was correlation, not
  causation. The flag flip changed downstream behavior — most likely
  via `per_step_restore()` reading state differently when per_step
  buffers exist vs don't — but the save side itself is byte-clean.

**Status:** P0.A.3 root cause is unknown again. Candidate theories
that need fresh investigation:
- `combine_features` cuBLAS pinned-HMMA GEMM dispatch ordering
  (PHASE 67-69 batched-pinned rewrite landed but may interact with
  post-fold 4D KV)
- `inject_kv_fused` async sync (Phase 69's batched-pinned variant —
  does it serialise correctly against subsequent target decodes?)
- drafter_forward kernel state leakage into target context
- `common_speculative_draft` sample-and-accept loop position math
  (verify-batch position drift relative to target's committed pos)
- post-fold 4D KV interaction with DFlash drafter's own KV
  (drafter cache lives in same context — possible alias)

**Next experiment**: do not retry libllama-level isolation; the
actual mechanism is in the DFlash speculative pipeline. Either
- A/B on the dflash-speculative-simple binary (existing target at
  examples/dflash-speculative-simple/) with cb_eval install force-disabled
  vs intact — confirms or further refutes cb_eval at the CLI level.
- Instrument per-stage tensor outputs (target prefill logits,
  combine_features output, inject_kv_fused output, drafter_forward
  output, target verify logits) and bisect first divergent stage.

**Pre-requisite work already landed** (production/2026-q2-next):
- MAL cap fix (P0.A.1): drafter K/V cache 21.5 GB → 85 MB.
- stage_target_hiddens end-trim restore (P0.A.2): acceptance 8 % → 54 %.

**Do not retry P0.A.3 root-cause from inside an existing conversation**
unless that conversation is fresh / dedicated to this. The 2026-05-20
push-through reached a kernel-level diagnostic boundary; the
follow-on cb_eval falsification was done in the same session and is
recorded above. Further work needs the dflash-speculative-simple
A/B or pipeline-stage tensor capture infrastructure that is its own
session's worth of context.

**Related:** [[project_phase_nstream_kv_perf_scoped]] (parent phase
status), [[project_dflash_multislot_phase4_landed]] (libllama tests
that DO pass), [[feedback_verify_test_mechanism_before_trusting]] (the
discipline that found the falsification — write the binding test
even when the diagnostic looks confident), [[feedback_bisect_before_revert]]
(don't ship a fix on a falsified diagnostic).
