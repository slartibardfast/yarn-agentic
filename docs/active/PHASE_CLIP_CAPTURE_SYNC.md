# PHASE_CLIP_CAPTURE_SYNC — make the CLIP encoder capture-legal so MAX_COPIES=2 deploys

**Opened**: 2026-05-29
**Status**: RESOLVED, awaiting closure move — capture question answered (negative). Captured-graph replay now **functional + byte-identical** (`ik_llama.cpp@ceb534ae`) but **NOT a perf win for CLIP** (~3% slower; encode is compute-bound), so it stays default-OFF behind `GGML_SCHED_OUTER_CAPTURE`. The eager decoupled-events baseline (`ik_llama.cpp@7a43ef87`, ~28% faster, byte-identical) is the deploy candidate; that deploy is a **separate user decision** (production still on Phase-46 closure `1db6c2eb`). Archive on deploy-decision.
**Triggered by**: `data/cuda-native-dispatch/post-merge-maxcopies2-20260529T104724/report.md` — the MAX_COPIES=2 verification window crashed the CLIP encoder on encode 1.
**Predecessor**: `PHASE_CUDA_NATIVE_DISPATCH` (determinism arc complete + verified).
**Production impact**: none yet (not deployed). Production stays on Phase-46 closure `1db6c2eb`. The new baseline is committed and verified, ready to deploy when chosen.

## Outcome so far (2026-05-29)

The chase to make CLIP capture-legal under `MAX_COPIES=2` peeled four
capture-illegal layers (device sync, reduce post-fence, user-input host
staging, mid-capture allocation) and one calloc/`std::unordered_set` SIGFPE.
The decisive reframe: `MAX_COPIES=2` was only ever wanted to allocate the
cross-sync **events** the C4 outer capture needs — but its `n_copies=2`
**buffer doubling** pushed CLIP's CUDA0 (12.5 GB compute meta) over its 24 GB
ceiling (recoverable-OOM → IMA). **Decoupling the events from `n_copies`**
(allocate `events[b][0]` at `n_copies=1`) gives the event-based ordering
without the doubling.

That decouple is a **win on its own, without capture**: `copy_inputs` now uses
fine-grained `cudaStreamWaitEvent`/`event_synchronize` at `n_copies=1` instead
of the coarse full-device `ggml_backend_synchronize` the old `events==NULL`
path took. Verified (RUN_ID `clipsync-20260529T154119`):

- CLIP encode **10/10 byte-identical** to the Phase-46 closure (sha `fb5167dbc1e7f95b`).
- Median **10423 ms vs 14421 ms baseline — ~28% faster**.
- LM NP-determinism unregressed (NP={1,2,4,8}×3 + SERIALIZE PASS).

Committed `ik_llama.cpp@7a43ef87`, default-safe (non-capture path unchanged).

## Captured-graph replay — now functional, but not a CLIP perf win (2026-05-29)

Outer capture is OFF by default. The warm-first scheme (eager encode 1 to grow
allocators, capture from encode 2, cache-hit replays thereafter) now runs
end-to-end under `GGML_SCHED_OUTER_CAPTURE=1`: **10/10 byte-identical** to the
eager baseline (sha `fb5167dbc1e7f95b`), capture fires (count=9 over a 10-encode
run = 1 capture + 8 replays), no crash. Verified RUN_ID `clipcap-20260529T161531`.

Four blockers were peeled in order (each hidden behind the prior), committed
`ik_llama.cpp@ceb534ae`:

1. **WAR-guard cross-stream wait** (`copy_inputs`, ggml-backend.cpp). The
   consumer `event_wait` targets the consumer's OWN sched event, last recorded
   in the prior (uncaptured) encode → illegal dependency on uncaptured work
   inside capture (`ggml-cuda.cu:5686`). Skipped on the in-capture intermediate
   pass; the real producer→consumer RAW edge is carried by
   `cpy_tensor_async`'s `copy_event`, and the warm encode fully drains before
   `BeginCapture`.
2. **`cudaMemcpyPeerAsync` is capture-prohibited** (`ggml-cuda.cu:4464`). With
   peer access enabled (UVA), a 1D linear DtoD `cudaMemcpyAsync` on the src
   stream is the identical, capturable transfer. Used only while capturing.
3. **calloc'd `std::unordered_map` SIGFPE** (`sched->outer_graphs`). The
   calloc'd map lands with `bucket_count==0`; `find()` survives empty
   (libstdc++ small-size linear scan), but the first insert computes
   `hash % 0` → SIGFPE in `compute_splits` on the capture pass. Replaced with
   calloc-safe parallel `std::vector` keys+execs (same fix already applied to
   `outer_topo_seen`). A latent landmine for anyone enabling the flag.
4. **Poisoned-event host sync** (`copy_inputs` USER_ONLY pre-stage). A host
   `cudaEventSynchronize` on a sched event recorded INSIDE the prior capture →
   "invalid argument" (`ggml-cuda.cu:5713`) on the next cache-hit dispatch. The
   pre-stage now drains the consumer device directly (legal outside capture;
   primary stream + C4 fan-in cover every secondary).

**Perf finding — the reason this stays default-OFF.** Captured-graph median
**10734 ms** vs eager decoupled baseline **10423 ms** — **~3% slower, not
faster.** A CLIP encode is **compute-bound** (10+ s of matmul/attention), not
dispatch-bound, so collapsing 271 split dispatches into one `cudaGraphLaunch`
saves negligible overhead while adding a per-replay device-sync. The phase's
original premise — "CLIP is the consumer that benefits from captured graphs" —
was wrong for the latency that matters. The eager decoupled-events path already
banked the real win (coarse-sync removal). **Capture is correct, now a closed
question, and not a deploy candidate for CLIP.** (Whether the LM token-gen path
— many tiny latency-bound ops — benefits from capture is a separate, genuinely
open question under `PHASE_CUDA_NATIVE_DISPATCH`, not pursued here.)

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
