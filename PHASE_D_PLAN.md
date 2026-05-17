# Phase D — Multi-GPU peer-access deterministic ordering

> **SUPERSEDED 2026-05-17.** Phase D was based on a misdiagnosis — the race
> exists identically on single-GPU, so multi-GPU peer-access isn't the source.
> The actual work is now under [`PLAN_DETERMINISM_AUDIT.md`](PLAN_DETERMINISM_AUDIT.md).
> See [`STATUS.md`](STATUS.md) for the current narrative. This file is kept
> for traceability (the D.1 audit table content remains useful).

**Branch**: `production/2026-q2-next`

**Pre-state (2026-05-17)**:
- Phase CY closed on its unit-test (10/10 NP=2 byte-identical) + one-shot harness (14/14 at NP={1,2,4,8}) bindings.
- D.1 peer-write site audit done; table in `PHASE_MMQ_Q4_0_AR16.md §7`.
- Cross-process NP=1 baseline drifts SHA across server restarts (5 unique SHAs / 10 runs observed). This is the residual race.

## What Phase D owns

CX.7's closure ("single-GPU NP={1,2,4,8} 5/5 stable") and D.4's closure ("multi-GPU NP={1,2,4,8} byte-identical to single-GPU baseline across 3 restarts") are the same class of test, gated on the same evidence. Both fold into Phase D.

## Evidence-gathering matrix (running)

Probes in `scripts/phase-d-evidence-probes.sh`, output `data/phase-d-evidence/probe-*-summary.txt`:

| Probe | Hypothesis | Config | Outcome | Phase impact |
|---|---|---|---|---|
| **P1** | Single-GPU is stable; race is multi-GPU specific | `DEVICE=CUDA0`, NP={1,2,4,8}, 5 runs | PASS expected → closes CX.7 → narrows D to multi-GPU only | binding |
| **P2** | NP=1 alone drifts cross-process (NOT slot interaction) | Multi-GPU, NP=1 only, 5 server cycles | FAIL expected → drift is in NP=1 cross-process path | binding |
| **P3** | Q4_0 KV cache quant round-trip introduces drift | Multi-GPU, NP=1, F16 cache | PASS = Q4_0 cache is the source; FAIL = unrelated | source-isolating |
| **P4** | Hadamard rotation introduces drift | Multi-GPU, NP=1, no Hadamard | PASS = Hadamard is the source; FAIL = unrelated | source-isolating |
| **P5** | Re-bind multi-GPU full sweep on new code (sanity) | Multi-GPU, NP={1,2,4,8}, 5 runs | establishes current failure rate | baseline |

## Decision tree (after probes complete)

```
P1 PASS → CX.7 CLOSED. Race is purely multi-GPU.
  ├── P2 FAIL → drift is in NP=1 multi-GPU cross-process (process init / peer-access setup)
  │     ├── P3 PASS → Q4_0 cache is the culprit. Fix: pin Q4_0 quant determinism at server boundary OR ship F16 cache.
  │     ├── P4 PASS → Hadamard is the culprit. Fix: pin Hadamard rotation phase / fp32 it.
  │     └── P3 + P4 FAIL → bug is in CUDA peer-access init or memory layout. Audit ggml_cuda_set_peer_access.
  └── P2 PASS → today's observation was variance. Re-run with larger N.

P1 FAIL → CX.7 NOT CLOSED. Single-GPU has its own non-determinism. New investigation: which op?
  └── Independent of multi-GPU. Phase D narrows but CX expands.
```

## Fix candidates by decision-tree branch

### Branch: Q4_0 cache is the source (P3 PASS)
- Server init re-quantizes the cache at startup with some non-deterministic input. Audit `llama_kv_cache_init` for ordering.
- OR: cache-element-zero is uninitialized memory. Add `cudaMemset(0)` post-alloc.
- Cost: ~10-30 LOC.

### Branch: Hadamard is the source (P4 PASS)
- Hadamard rotation has a static state initialized per-process. Verify it's deterministic.
- Possibly fix by pinning rotation matrix construction order.

### Branch: peer-access init order (both P3/P4 FAIL)
- `ggml_cuda_set_peer_access` (ggml-cuda.cu:1987) iterates over devices. Order may depend on internal state. Pin order.
- May need a `cudaDeviceReset` at process start (heavy hammer but deterministic).

### Branch: multi-GPU specific only (P1 PASS, P2 FAIL)
- Combined with P3/P4 outcomes above.

## D.4 closure binding

Once a fix lands, `data/phase-d-closure-NNNN/`:
- Single-GPU NP=1 SHA, 3 server restarts, identical.
- Multi-GPU NP=1 SHA, 3 server restarts, identical AND matching single-GPU baseline.
- Multi-GPU NP={2,4,8} × 3 server restarts, all slots byte-identical to single-GPU baseline.

## Risks

- The fix may require a `cudaDeviceReset` at process init (heavy hammer; serializes all multi-process work that shares the GPU).
- The drift might depend on GPU memory layout that varies per process (UVA address randomization). No clean fix; would need pinned allocator.
- More than one source may exist; staged fixes required.

## Token budget (per CLAUDE.md §8)

| Step | Tokens |
|---|---|
| Probe matrix analysis (read summaries) | ~10k |
| Source-isolating audit per decision branch | ~15-30k |
| Fix iteration + rebuild + verify | ~30-60k |
| D.4 closure binding script + run | ~10k |
| **Total per branch** | **~65-110k** |

If multiple sources: multiply by branches.

## Pickup discipline

- Probe outputs land in `data/phase-d-evidence/`. Read those first.
- The decision tree is the navigator; don't speculate beyond what the probes say.
- Per `feedback_verify_test_mechanism_before_trusting`: any "PASS" must repeat across 5+ runs before reporting as a fix.
- Plan-file edits commit separately from code per CLAUDE.md §5.

## Out of scope for Phase D

- The actual fix design depends on the decision-tree branch. This plan stops at the disambiguation; the fix scope writes when probes complete.
- Performance tuning (Phase E) is gated on D.4 binding.
- Phase F (overall closure) is gated on D + E.
