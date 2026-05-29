# PHASE_CUDA_NATIVE_DISPATCH — MAX_COPIES=2 live-verification

**RUN_ID:** maxcopies2-20260529T104724
**Date:** 2026-05-29
**Host:** xeon (2× Quadro RTX 6000, clocks locked 1455 MHz, governor=performance)
**Test binary:** build-tree `llama-server`, `GGML_SCHED_MAX_COPIES=2` (confirmed: 263 `=2` lines in build.ninja; `ggml-backend.cpp.o` recompiled 10:11)
**Production binary (untouched):** Phase-46 closure at `/opt/llm-server/` — restarted clean, `/health=200`

## Verdict: MAX_COPIES=2 is NOT deployable as-is

Determinism gates all PASS; the **CLIP encoder crashes on the first encode**.
The captured-graph perf path that MAX_COPIES=2 unlocks collides with the
Phase-46 B.5e mandatory device sync. Production stays on Phase-46 closure.

## Results

| Gate | Result | Detail |
|---|---|---|
| §5.1 NP determinism {1,2,4,8} ×3 | ✅ PASS ×3 | All slots byte-identical to NP=1; cross-NP slot-0 identical, all 3 reps |
| §5.2 SERIALIZE A/B (NP=8) | ✅ PASS | NP=8 byte-identical to NP=1 under `GGML_SCHED_EVAL_SERIALIZE=1` — no latent race the capture path hides |
| §5.1.3 CLIP encode ×10 | ❌ CRASH | Server died on encode 1 after 10040 ms; encodes 2-10 connection-refused |
| Production restore | ✅ | trap restarted Phase-46 closure binary; `/health=200` |

The NP-determinism PASS confirms what `PHASE_NP8_FLAKE` already localized:
with governor=performance the NP=8 single-slot flake does not reproduce.
3/3 clean reps on the MAX_COPIES=2 binary. The LM dispatch path is correct.

## The crash

```
processing image...
CUDA error: operation not permitted when stream is capturing
  current device: 0, in function ggml_backend_cuda_synchronize at
  ggml/src/ggml-cuda.cu:4516
  cudaDeviceSynchronize()
```

`ggml_backend_cuda_synchronize` (ggml-cuda.cu:4502-4519) unconditionally
calls `cudaDeviceSynchronize()`. That call is the **Phase-46 B.5e Phase-C
fix**: a full device drain required because libmgpu's multi-device CLIP
encoder lands peer-access writes on copy/event streams that a stream-only
sync would miss (its own inline comment says so).

On `MAX_COPIES=1` this was harmless because the 2026-05-27 gate fix
(`&& sched->n_copies > 1`) kept outer capture **disabled** — the CLIP
encode ran on the C1 eager path, no capture active, the device sync legal.

On `MAX_COPIES=2` outer capture **fires** on the CLIP encode (events are
now allocated, the gate opens). The B.5e `cudaDeviceSynchronize()` then
runs *inside* an active `cudaStreamBeginCapture` region — which CUDA
forbids — and the process aborts on the first encode.

The LM determinism battery does not hit this: the LM dispatch does not
call `ggml_backend_cuda_synchronize` inside its captured region. The
collision is specific to libmgpu's CLIP encoder, which is the phase's
primary consumer.

## Why this is not a window-hack fix

The obvious patch — gate the sync to a no-op when
`ggml_cuda_outer_capture_active()` is true (the guard exists at
ggml-cuda.cu:5925) — risks **reintroducing the B.5e nondeterminism race**.
That device drain exists precisely to make the multi-device CLIP encode
bit-deterministic; removing it under capture, without first re-expressing
the peer-write drain as in-graph event nodes (C4's fan-in mechanism),
could bring back the cross-encode state leak Phase 46 closed. This needs
a design pass, not a one-line gate.

## Decision surface

1. **Stay on Phase-46 closure** (deployable now). The C-arc's determinism
   merits are proven (3/3 NP + SERIALIZE). The captured-graph perf path
   stays unrealized — same outcome as the 2026-05-27 window, now with the
   crash root-caused rather than masked.
2. **Resolve the sync-vs-capture collision** (follow-on work item). Either
   (a) gate outer capture OFF for the CLIP/libmgpu path until the B.5e
   drain is re-expressed as graph event nodes, or (b) replace the coarse
   `cudaDeviceSynchronize` with capture-legal per-stream event waits that
   preserve B.5e determinism. Both need their own verification window.

## Files

- `pre-state.txt` — health/VRAM/governor/binaries before the window
- `np-rep{1,2,3}.log` + `np-rep{1,2,3}/` — determinism battery (3 PASS)
- `np-serialize.log` — SERIALIZE A/B control (PASS)
- `clip.log` — CLIP harness (crash)
- server crash: `/tmp/phase46-multigpu-clip/run-20260529T105243/server.stderr`
- `window-summary.txt` — one-line-per-gate summary
