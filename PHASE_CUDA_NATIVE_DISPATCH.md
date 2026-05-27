# PHASE_CUDA_NATIVE_DISPATCH — ground-up CUDA-idiomatic replacement for ggml-backend's openmp-parallel multi-backend dispatch

**Status**: Design document. Not yet active. Prerequisite: this design + verification + performance plan reviewed and approved.
**Predecessor**: `PHASE_NSTREAM_KV_PERF.md` (open). `PHASE46` (closed).
**Triggered by**: NP=8 single-slot determinism flake localized 2026-05-27 to host-side CUDA driver state racing under openmp-parallel multi-backend dispatch — verified by `GGML_SCHED_EVAL_SERIALIZE=1` PASS (Test 2 of the discriminator window).

---

## §1 — The diagnostic that motivates this redesign

The NP=8 single-slot determinism flake (`tests/spec/test-production-np-determinism.sh` failing one slot per run, slot rotating between 6 and 7) reproduces under:

| Knob | NP=8 result |
|---|---|
| Default (post-Phase-46) | FAIL — rotating slot |
| `GGML_CUDA_STREAM_SYNC=1` (reverts Phase 46 sync default) | FAIL — rotating slot |
| `LLAMA_FATTN_STRICT_SEQUENTIAL_DECODE=1` (one decode per tick) | FAIL — slot 7 byte-identical divergence |
| **`GGML_SCHED_EVAL_SERIALIZE=1`** (omp-critical around eval) | **PASS — all 8 slots byte-identical** |

Conclusion: the race lives at or below `ggml_backend_sched_eval` (`ggml/src/ggml-backend.cpp:2126`), called concurrently from multiple openmp threads. It is **not** in the FA kernel (per-CTA design with no shared state, no atomics), **not** in concurrent batched-decode (STRICT_SEQUENTIAL didn't fix it), and **not** in cross-stream peer-copy completion (Phase 46's per-split drain + cudaDeviceSynchronize default are both already in force).

The race is between openmp threads racing on host-side CUDA driver state during eval-dispatch — a fundamental **openmp ↔ CUDA conceptual impedance mismatch**:

- openmp: thread-parallel; correctness depends on annotated critical sections or thread-local data.
- CUDA: per-thread `cudaSetDevice` / current device, but per-context shared state (graph cache map, pool allocator, lazy event creation, cublas handles).

When N openmp threads call into the CUDA backend concurrently, the per-context shared state is accessed without locks. The race surface is identified but the specific racing field is not yet narrowed; candidates include:

- `cuda_ctx->cuda_graphs` (`ggml-cuda/common.cuh:858-969`) — `std::unordered_map<uint64_t, ggml_cuda_graph>`, per-context. Insertions/eviction not lock-protected.
- `cuda_ctx->copy_event` — lazy-created at `ggml-cuda.cu:4472-4473` with single-checked `if (!copy_event)`. Two threads can both see `nullptr`, both `cudaEventCreate`, both store (one leaked).
- `cuda_ctx->pools[]` — per-device memory allocators. Concurrent host-side alloc from different openmp threads to the same pool.
- cuBLAS handle workspace + stream binding state.

The per-context state was originally designed for single-threaded host dispatch. Phase 46's per-split drain (`ggml-backend.cpp:2300-2310`) protects **cross-split** ordering but doesn't lock **within-split** concurrent context access. Phase 46 itself was a tactical patch for the CLIP encoder's worst-symptom path; this design replaces the underlying dispatch model.

---

## §2 — Design principles for the replacement

The replacement is **CUDA-idiomatic single-threaded host + multi-stream device async**. Three principles:

1. **One host thread.** All `ggml_backend_sched_compute_splits` invocations run on one host thread. No `#pragma omp parallel`, no `std::barrier`, no per-backend worker threads.

2. **All parallelism on the device, via streams + events.** Each backend keeps its existing per-context stream set (`streams[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS]`, already `cudaStreamNonBlocking`). Cross-backend dependencies expressed as `cudaEventRecord(stream_src, event)` + `cudaStreamWaitEvent(stream_dst, event, 0)`. The host enqueues all work asynchronously and the device-side stream scheduler handles concurrency.

3. **Capture the whole forward pass into one CUDA Graph.** Per-token replay reduces to a single `cudaGraphLaunch` plus input-pointer patching. The CUDA Graph already exists per-backend (`ggml_cuda_graph` cache at `ggml-cuda/common.cuh:858-969`) — the redesign extends capture across backends so the whole multi-GPU graph becomes one graph object, eliminating the cross-backend host stitching that's currently the race surface.

### Why host parallelism doesn't buy anything

Today's openmp parallel dispatch runs `num_threads(sched->n_backends)` (3 threads for {CUDA0, CUDA1, CPU_Host}). Each thread's host-side work is **kernel-launch overhead only** — tens of microseconds per kernel. Real compute is on the device, async. Parallelizing the host launches doesn't increase device throughput; the streams already overlap on the device once kernels are submitted. The host parallelism only adds: (a) a race surface, (b) omp synchronization overhead, (c) per-split drain serialization.

Single-threaded dispatch with proper event chains achieves the same device-side concurrency without the race, omp overhead, or per-split drain.

### Why CUDA Graph capture across the whole forward pass

Current CUDA Graphs are captured per-backend. Each backend's `ggml_backend_cuda_graph_compute` (`ggml-cuda.cu:5150-5250`) does its own `cudaStreamBeginCapture` over its split, then `cudaGraphLaunch`. Cross-backend data flow is handled outside the graph (via host-side `cpy_tensor_async` + event wait). That cross-backend stitching is the openmp barrier dance — exactly the part this redesign replaces.

A **single multi-device CUDA Graph** captures the entire forward pass including cross-device peer copies. Per-token replay is then ONE `cudaGraphLaunch` regardless of backend count. CUDA Graph supports cross-device dependencies via `cudaGraphAddEventRecordNode` / `cudaGraphAddEventWaitNode` (CUDA 11.4+, all our hardware supports it). Inside the graph, the CUDA driver handles cross-stream synchronization at submission time, not at execution time — best-possible latency.

---

## §3 — The new architecture

### 3.1 — Stage-1 (race-fix only, perf-neutral)

**Goal**: eliminate the openmp race without changing the per-backend graph capture. Smallest viable change. Validates the design's race-fix before extending to cross-backend graph capture.

Replace the openmp parallel block (`ggml-backend.cpp:2215-2350`) with single-threaded iteration:

```
for each split in topological order:
    if split has inputs from other backends:
        // wait for those producer-backends' streams via event chain
        for each cross-backend input:
            cudaStreamWaitEvent(this_split's_stream, producer_event, 0)
        // copy_inputs as currently, but on the producer's stream
        copy_inputs(split)
    // dispatch this split's eval — runs async on this backend's stream
    sched_eval(split)
    // record an event marking this split's completion
    cudaEventRecord(this_split_done_event, this_split's_stream)
// at end-of-graph: wait for all backends' final streams once
ggml_backend_synchronize(end_backend)
```

Properties:
- Single host thread → no host-side concurrent context access → race surface eliminated.
- All splits enqueued async on their backend streams → device-side concurrency preserved.
- Cross-backend deps expressed via events → no per-split drain needed.
- Per-backend CUDA Graph capture **unchanged**.
- Phase 46's per-split drain at `ggml-backend.cpp:2300-2310` is **deleted** (its purpose is subsumed by the event chain).
- `cudaDeviceSynchronize` default flip from Phase 46 remains as a defensive default for CLIP encode but is no longer the load-bearing fix.

Code surface modified:
- `ggml/src/ggml-backend.cpp` — `ggml_backend_sched_compute_splits` (lines 2177-2430) rewritten.
- New helpers in the same file for the event-chain bookkeeping.
- `ggml/src/ggml-cuda.cu` — minor: `cpy_tensor_async` already does event record/wait. The lazy-create race at line 4472-4473 fixed (`std::call_once` or pre-creation at context init).
- Phase 46's openmp-conditional drain at 2300-2310: deleted.

The `GGML_SCHED_EVAL_SERIALIZE`, `GGML_SCHED_NO_DRAIN`, and `GGML_CUDA_STREAM_SYNC` env knobs become diagnostic-only (preserved for one release for rollback, then deleted).

### 3.2 — Stage-2 (cross-backend CUDA Graph capture)

**Goal**: best-possible TG perf by capturing the entire multi-device forward pass into one CUDA Graph.

The Stage-1 design's per-split event-chain is the EXACT primitive that CUDA Graphs use internally (`cudaGraphAddEventRecordNode` / `cudaGraphAddEventWaitNode`). To extend graph capture across backends:

1. The outer driver issues `cudaStreamBeginCapture` on ONE designated stream (e.g., the main backend's stream) with mode `cudaStreamCaptureModeRelaxed` (allows cross-device dependencies during capture).
2. Each backend's split enqueue, including its own `cudaStreamBeginCapture`/`cudaStreamEndCapture` pair, becomes a sub-capture that's stitched into the outer graph via captured events. Modern CUDA's "graph capture forks" handles this.
3. At capture end, the driver returns ONE `cudaGraph_t` containing all backend splits + cross-device sync nodes.
4. The cache key extends from `topology_hash` to `(topology_hash, multi_device_layout)`; per-token replay is a single `cudaGraphLaunch`.

Code surface modified:
- `ggml/src/ggml-cuda.cu` — extend `ggml_backend_cuda_graph_compute` (lines 5150-5250) to handle external capture (caller already in capture mode).
- `ggml/src/ggml-backend.cpp` — `ggml_backend_sched_compute_splits` wraps the Stage-1 single-threaded loop in `cudaStreamBeginCapture` / `cudaStreamEndCapture` when all backends are CUDA.
- New cache structure for multi-device graphs in `ggml/src/ggml-cuda/common.cuh`.

Stage-2 lights up only when all backends in the sched are CUDA (mixed CPU/CUDA scheds fall back to Stage-1). For the production LM at `--device CUDA0,CUDA1 --split-mode graph` the practical sched is CUDA0 + CUDA1 + CPU_Host. CPU_Host is mostly the input-staging backend; its splits can be hoisted out of the captured region (run once before `cudaGraphLaunch`, never inside).

### 3.3 — Migration order

```
Stage 0 (already done) — characterize race + verify SCHED_EVAL_SERIALIZE fixes it.
Stage 1 — single-threaded multi-backend dispatch. Race fix. Perf-neutral target.
Stage 2 — cross-backend graph capture. Perf improvement target.
Stage 3 — clean up: remove openmp dispatch code path, std::barrier fallback, Phase 46 per-split drain, GGML_CUDA_STREAM_SYNC env knob, GGML_SCHED_NO_DRAIN env knob, GGML_SCHED_EVAL_SERIALIZE env knob.
```

Each stage commits and ships independently. Stage 1 is the load-bearing race fix and must close before Stage 2 begins. Stage 3 is cleanup; deferrable.

---

## §4 — Best-possible performance plan

### 4.1 — Baseline measurement (before Stage 1)

Capture today's perf with the openmp dispatch:

- `scripts/bench-multislot.sh` with `N_PARALLEL ∈ {1, 2, 4, 8}` on production LM. Measure aggregate tg t/s, per-slot tg t/s, PP t/s.
- `scripts/verify-multigpu-clip.sh LATENCY_N=10` (CLIP encode median + p95).
- `llama-bench` PP+TG at NP=1 and NP=8 with `--device CUDA0,CUDA1 --split-mode graph`.
- Counter: `dispatch counter: total=N multi_seq=M` ratio at each NP (from server-context.cpp:4811-4815). Records the multi-seq dispatch fraction baseline.

Outputs land in `data/cuda-native-dispatch/baseline-<RUN_ID>/`.

### 4.2 — Stage-1 perf target

Stage-1 keeps per-backend graph capture so device-side concurrency is preserved. Host-side parallelism removed but host work is ~microseconds. **Expected: ±2% on TG/PP at NP=8.** Loss budget: -5%. Anything worse than -5% blocks Stage-1 merge.

Risk: the omp barriers that exist today might be doing useful host-side reordering that single-threaded loses. Mitigation: profile with `nsys` to confirm device-side overlap is preserved.

### 4.3 — Stage-2 perf target

Stage-2 amortizes host overhead across N tokens of the same topology. Per-decode-token host cost drops from N kernel launches × 3 backends to ONE `cudaGraphLaunch`. **Expected gain: +5 to +15% TG at NP=8** (LM is host-launch-bottlenecked at the matmul-heavy decode steps).

vLLM achieves 154.77 t/s aggregate at NP=8 on this hardware (the `PHASE_NSTREAM_KV_PERF.md` reference). Current ik_llama.cpp is at ~26 t/s aggregate at NP=8 — 5.84× below. Most of that gap is dispatch-layer (per `PHASE_NSTREAM_KV_PERF` Tier 3 analysis: per-stream dispatch is 8 graph rebuilds per tick at NP=8). Cross-backend graph capture is the single biggest lever to close that gap inside ik_llama.cpp.

Risk: CUDA Graph capture across devices is less mature than single-device capture. Some ops may not support capture (e.g., dynamic shape paths). Mitigation: per-topology cache + capture-failure fallback to Stage-1 path.

### 4.4 — Specific optimizations to fold in during Stage-2

1. **Reuse the existing `ggml_cuda_graph` cache**. Don't build a parallel cache. Extend its key to include cross-device layout.
2. **Hoist CPU_Host input staging out of the captured region.** Capture only the GPU compute graph; CPU staging runs once per token before launch.
3. **Pre-allocate all cross-device event objects at context init.** Eliminate the lazy-create race surface in `cpy_tensor_async`.
4. **Use `cudaStreamCaptureModeRelaxed`** to allow cross-device event dependencies during capture (the strict mode disallows them).
5. **Multiple compute streams per device for ILP.** The context already has `streams[device][n_streams]` infrastructure. Use 2 streams per device for the two halves of split-mode-graph matmuls, allowing overlap of the matmul + the post-reduce. Today only `stream(device, 0)` is used.
6. **NCCL for the cross-device REDUCE** (gated for regular graphs today). At NP=8 the REDUCE ops happen N_layers × N_decode times — using NCCL collective primitives (already linked) saves ~30 µs per reduce vs the current cudaMemcpyPeer + element-wise add. Per `ggml-cuda.cu:4449-4461` the NCCL path exists but is `#ifdef GGML_USE_NCCL__` (note the trailing underscore — it's effectively disabled). Re-enable for the REDUCE op specifically.

### 4.5 — Performance target ceiling

vLLM's 154.77 t/s at NP=8 is the hardware-attainable ceiling on the same 2× Quadro RTX 6000. After Stage-2:

- **Conservative target**: 50 t/s aggregate NP=8 (2× current).
- **Stretch target**: 100 t/s aggregate NP=8 (4× current; ~65% of vLLM).
- **Aspirational target**: 130 t/s aggregate NP=8 (5× current; ~85% of vLLM).

The remaining gap to vLLM after Stage-2 is in non-dispatch areas (kernel-level: persistent kernel for decode, FA kernel specialization, MMQ Q4_0 throughput). Those are outside this phase's scope.

---

## §5 — Verification

### 5.1 — Determinism gates (must PASS before each stage's merge)

1. **`scripts/test-production-np-determinism.sh`** at NP ∈ {1, 2, 4, 8}, 3 reps each (12 invocations). Byte-identical NP=1 baseline; byte-identical across NPs; cross-NP slot-0 matrix all identical. **The closing gate.**
2. **`scripts/r5-probe-c4.sh`** at 20 iters NP=2 single-GPU. Rate=0%.
3. **`scripts/verify-multigpu-clip.sh LATENCY_N=10`** — CLIP encode determinism (10/10 byte-identical `reasoning_content` sha256) AND median ≤ 1.3× CPU-vision baseline.
4. **All spec unit tests** (paged-aware + n-stream battery): `test-n-stream-kv-layout`, `test-paged-allocator-determinism`, `test-paged-kshift-byte-identity`, `test-kv-defrag-per-stream`, `test-kv-shift-per-stream`, `test-mtp-x-n-stream`, `test-unified-stream-dispatch`, `test-multi-seq-decode-byte-identity`. All exit 0 (or 77 SKIP).

### 5.2 — Stage-1 verification

- All gates §5.1 PASS.
- Perf within ±5% of baseline §4.1.
- `GGML_SCHED_EVAL_SERIALIZE` no longer has any effect (race surface gone). Document this in a §3.1 commit.

### 5.3 — Stage-2 verification

- All gates §5.1 PASS (still).
- Perf at conservative target §4.5 minimum (50 t/s aggregate NP=8). Stretch target a bonus.
- `nsys` profile shows single `cudaGraphLaunch` per decode token, no host stalls between backends.
- Multi-device graph cache hit rate >95% in steady-state decode (cache miss only on topology changes).

### 5.4 — Production deploy gate

- All gates §5.2 PASS for the to-deploy build.
- 24-hour soak at production NP=1 traffic with the new dispatch path. No /health drops, no slot-state leaks, RSS stable.
- Rollback drill (matching Phase 46 B.8): deploy a known-good pre-Stage-1 build via `scripts/deploy-llama-server.sh --allow-no-mmproj-mgpu`; confirm clean recovery; redeploy forward.

---

## §6 — Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Stage-1 perf regression worse than -5% | Low | Profile with nsys before merging; if regression, identify specific overlap loss and patch before merge |
| Stage-2 cross-device capture fails on some op | Medium | Per-topology capture-failure fallback to Stage-1; log captured-vs-fallback ratio |
| CUDA Graph capture mode change breaks libmgpu | Medium | libmgpu's per-device sub-graphs already use capture (Phase 46 work); extend to relaxed mode jointly |
| NCCL re-enablement reintroduces past races | Low-Medium | NCCL was disabled for regular graphs for a reason (see project_phase46_npc4_six_tests.md); re-enable behind an env knob first, verify across the full determinism battery, then bake |
| Multi-stream-per-device ILP (§4.4.5) reintroduces openmp-era thread-context confusion | Low | Streams are per-CONTEXT (one host thread), so concurrent stream submissions from one thread are race-free at the driver level |
| Rollback path complexity | Low | Phase 46's `--allow-no-mmproj-mgpu` deploy script flag generalizes — add an `--openmp-dispatch` fallback flag if Stage-1 needs to be rolled back |

---

## §7 — Out of scope

- Kernel-level optimizations (FA kernel, MMQ, Q4_0 throughput).
- Allocator changes (paged KV pool sizing, kv_pool_blocks tuning).
- Sampler changes.
- Reverting T5.9 paged BACKING.
- Production deployment of NP>1 (separate decision; this phase only restores correctness at NP>1).

---

## §8 — Acceptance summary

This phase closes when all of:

1. Stage-1 merged with §5.2 PASS.
2. Stage-2 merged with §5.3 PASS.
3. Stage-3 cleanup merged (openmp dispatch code path deleted; obsolete env knobs removed).
4. `MEMORY.md` records the closure with the perf delta vs §4.1 baseline.
5. Production deployed via §5.4 gate.

Estimated cost: **3 stages × 1-2 days dev + 1 maintenance window each + 1 soak week**. Tokens: ~250-400k across the three stages. **Phase 46's per-split drain is the precedent — that landed in one combined window; this phase is a structural replacement of the same surface, so expect ~3-5× the engineering effort but commensurate determinism gain + perf headroom toward the vLLM ceiling.**

---

## §9 — Forward to implementation

Stage-1 is the immediate next step. The branch should be cut off submodule HEAD `1db6c2eb` (current production). The first commit is the new single-threaded `compute_splits`; the second is the cleanup of the openmp/std::barrier paths. The third is the `cpy_tensor_async` `copy_event` pre-creation.

After Stage-1 merges, Stage-2 begins with the cross-backend capture experiment in isolation (no production deploy until §5.3 binds).
