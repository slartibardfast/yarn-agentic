# PHASE_CLIP_CAPTURE_SYNC — make the CLIP encoder capture-legal so MAX_COPIES=2 deploys

**Opened**: 2026-05-29
**Status**: OPEN — root cause localized; design pass needed before implementation
**Triggered by**: `data/cuda-native-dispatch/post-merge-maxcopies2-20260529T104724/report.md` — the MAX_COPIES=2 verification window crashed the CLIP encoder on encode 1.
**Predecessor**: `PHASE_CUDA_NATIVE_DISPATCH` (determinism arc complete + verified; this phase unblocks its captured-graph perf path for the CLIP/libmgpu consumer).
**Production impact**: none. Production stays on Phase-46 closure `1db6c2eb`; this phase is what would let a MAX_COPIES=2 build deploy.

---

## §1 — The collision

`MAX_COPIES=2` enables PHASE_CUDA_NATIVE_DISPATCH's outer-capture path
(C4/C5/C7) — on `MAX_COPIES=1` it was dead code, gated off by the
2026-05-27 `&& sched->n_copies > 1` condition. With capture active, the
multi-GPU CLIP encoder aborts on the first encode:

```
processing image...
CUDA error: operation not permitted when stream is capturing
  current device: 0, in function ggml_backend_cuda_synchronize at
  ggml/src/ggml-cuda.cu:4516
  cudaDeviceSynchronize()
```

`ggml_backend_cuda_synchronize` (ggml-cuda.cu:4502-4519) unconditionally
calls `cudaDeviceSynchronize()`. That call is the **Phase-46 B.5e Phase-C
fix**: a full-device drain required because libmgpu's multi-device CLIP
encode lands peer-access writes on copy/event streams that a stream-only
sync would miss. Run inside an active `cudaStreamBeginCapture` region it is
illegal, and the process aborts.

The 2026-05-27 gate masked this by keeping capture disabled at
`MAX_COPIES=1`; it did not fix it. The LM dispatch path does not call
`ggml_backend_cuda_synchronize` inside its captured region, so LM
determinism is unaffected (NP={1,2,4,8} ×3 reps + SERIALIZE A/B all PASS
on the MAX_COPIES=2 binary).

## §2 — Why it matters

PHASE_CUDA_NATIVE_DISPATCH is "for libmgpu" — the tensor-parallel CLIP
encoder is the primary intended consumer of the captured-graph path. The
captured-graph perf benefit (one `cudaGraphLaunch` per encode instead of
271 graph-split dispatches) is exactly what this consumer should get. As
long as the CLIP path crashes under capture, MAX_COPIES=2 cannot deploy
and the perf path stays unrealized.

## §3 — Approaches

### §3.a — Gate outer capture OFF for the CLIP/libmgpu path (stopgap)

Skip outer capture whenever the dispatch is the CLIP encode, leaving it on
the C1 eager path (Phase-46 closure behavior, B.5e sync legal). LM keeps
capture if it benefits.

- Pro: low risk; CLIP reverts to proven-correct eager dispatch.
- Con: the primary consumer gets NO captured-graph benefit — undercuts the
  phase's reason for existing. Also unverified whether the LM even benefits
  (CAPTURE_EVIDENCE was silent in the 05-29 logs — capture firing for LM
  was never confirmed). Likely a non-win dressed as a fix.

### §3.b — Re-express the B.5e drain as in-graph event nodes (real fix) — RECOMMENDED

Replace the coarse `cudaDeviceSynchronize()` with capture-legal
`cudaStreamWaitEvent` chains that (a) are legal during capture and (b) get
captured INTO the graph so replay enforces the same ordering. This is
precisely what `specs/cuda-native-dispatch/cross_device_event_chain.allium`
(C4 fan-in) already specifies — the B.5e coarse sync is a host-side
shortcut that predates the in-graph event mechanism.

- Pro: the principled fix, aligned with the C4 architecture; CLIP gets the
  captured-graph perf while B.5e determinism is preserved by the event
  chain rather than the device drain.
- Con: real engineering. Must identify exactly which peer-write streams the
  B.5e drain covers and prove the event-node replacement drains the same
  set. B.5e-regression risk is the central hazard.

## §4 — Verification plan

Reuse the MAX_COPIES=2 window harness (`/tmp/maxcopies2-window.sh`):

1. CLIP encode ×10 completes (no crash) AND 10/10 byte-identical
   `reasoning_content` sha256 — the B.5e determinism contract must hold
   under the event-node drain.
2. CLIP encode median ≤ Phase-46 baseline (14421 ms) — capture should not
   regress, ideally improves.
3. LM NP={1,2,4,8} ×3 + SERIALIZE A/B still PASS (regression guard — these
   already pass on MAX_COPIES=2; the fix must not break them).
4. Confirm the capture path actually fires (add a log line / counter; the
   05-29 run could not confirm capture from logs).

## §5 — Risk

- **B.5e-regression** (the dominant hazard): the device drain exists to
  close the cross-encode state-leak race ([[project-phase46-b5e-closed]]).
  Any replacement must be proven bit-deterministic across ≥10 encodes
  before deploy. Do NOT no-op the sync under capture as a shortcut.
- Production runs `--parallel 1` and stays on Phase-46 closure throughout;
  no production risk until an explicit deploy gate after §4 passes.

## §6 — Evidence

- `data/cuda-native-dispatch/post-merge-maxcopies2-20260529T104724/report.md`
  — the verification window that localized this (LM determinism PASS, CLIP crash).
- Crash: `/tmp/phase46-multigpu-clip/run-20260529T105243/server.stderr`.
