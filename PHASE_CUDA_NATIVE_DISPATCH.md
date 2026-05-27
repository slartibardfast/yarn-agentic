# PHASE_CUDA_NATIVE_DISPATCH — ground-up CUDA-idiomatic dispatch redesign for libmgpu (and LM as beneficiary)

**Status**: Design locked, pre-design measurements in progress. Implementation gated on PD1-PD6 outputs.
**Predecessor**: `PHASE_NSTREAM_KV_PERF.md` (open). `PHASE46` (closed — was tactical patch on the same race surface this phase replaces structurally).
**Triggered by**: NP=8 single-slot determinism flake localized 2026-05-27 to host-side CUDA driver state racing under openmp-parallel multi-backend dispatch — verified by `GGML_SCHED_EVAL_SERIALIZE=1` PASS (Test 2 of the discriminator window).

**Reframing 2026-05-27**: this work is FOR libmgpu, with LM as a beneficiary. libmgpu's tensor-parallel CLIP encoder is the primary consumer of multi-backend dispatch; Phase 46 was a tactical patch on libmgpu's openmp-parallel path, this is the structural replacement. The LM NP=8 flake was the symptom-carrier that surfaced the underlying race.

---

## §1 — Motivation: the diagnostic + the architectural goal

### §1.1 — Diagnostic chain

Three-discriminator window (RUN_ID=20260527T113550, production stopped, GPUs locked 1455 MHz):

| Test | Env | Result |
|---|---|---|
| NP=8 single-GPU | `DEVICE=CUDA0` | infeasible (`GGML_SCHED_MAX_SPLIT_INPUTS` ASSERT) |
| NP=8 + `GGML_CUDA_STREAM_SYNC=1` | multi-GPU | FAIL — reverts Phase 46 sync default, flake unchanged |
| NP=8 + `LLAMA_FATTN_STRICT_SEQUENTIAL_DECODE=1` | multi-GPU | FAIL slot 7 (byte-identical fingerprint to prior reps) |
| **NP=8 + `GGML_SCHED_EVAL_SERIALIZE=1`** | multi-GPU | **PASS — all 8 slots byte-identical to NP=1** |

`GGML_SCHED_EVAL_SERIALIZE=1` wraps each backend's `ggml_backend_sched_eval` in `#pragma omp critical(sched_eval)` (`ggml/src/ggml-backend.cpp:2257-2262`), serializing across openmp threads. The race is therefore at-or-below sched_eval, in host-side CUDA driver state racing under openmp-parallel concurrent dispatch. Not in: concurrent batched-decode (Test 3 ruled out), the FA kernel (per-CTA design with no shared state), cross-stream peer-copy completion (Phase 46's drain + cudaDeviceSynchronize already in force).

### §1.2 — The architectural goal: libmgpu

libmgpu builds per-device subgraphs for tensor-parallel CLIP encoding. The current dispatch model (openmp parallel multi-backend at `ggml-backend.cpp:2215`) was originally introduced because libmgpu needs concurrent per-device subgraph evaluation. Phase 46 made that path deterministic via three default-on patches, but did not address the underlying impedance mismatch:

- openmp: thread-parallel; correctness via critical sections or thread-local data
- CUDA: per-thread current device, but per-context shared state (graph cache, pool, lazy events, cuBLAS handles)

When N openmp threads call into the CUDA backend concurrently, per-context shared state is accessed without locks. The race is fundamental to combining openmp + CUDA at the dispatch layer. Patching the symptoms (drains, sync defaults) doesn't fix it; replacing the dispatch model does.

The new model is CUDA-idiomatic: single host thread enqueues all work async; cross-device dependencies via `cudaEventRecord` / `cudaStreamWaitEvent`; per-token replay via single `cudaGraphLaunch` over a multi-device captured graph.

---

## §2 — Design principles

1. **One host thread.** All `compute_splits` runs on one host thread. No `#pragma omp parallel`, no `std::barrier`, no per-backend worker threads.
2. **All parallelism on the device, via streams + events.** Each context's existing `streams[device][n]` infrastructure provides per-stream queues; cross-stream and cross-device deps expressed as events.
3. **Multi-device CUDA Graph capture.** Per-token replay = one `cudaGraphLaunch`. Cross-device sync nodes (`cudaGraphAddEventRecordNode` / `cudaGraphAddEventWaitNode`) embed dependencies INTO the graph, not host-side.
4. **libmgpu's per-device subgraphs become captured subgraphs.** No special-case path; the same dispatch handles LM and CLIP.

### Why host parallelism doesn't buy anything

Host-side work in CUDA dispatch is kernel-launch overhead — tens of microseconds per launch. Real compute is on the device, async. Parallelizing host launches doesn't increase device throughput; the streams already overlap on the device. Host parallelism contributes only: race surface, omp synchronization overhead, per-split drain serialization.

Single-threaded dispatch with proper event chains achieves identical device-side concurrency without the race or overhead.

---

## §3 — Architecture (single arc — no staging)

The user has chosen big-bang: one branch, one merge, no `GGML_DISPATCH_OPENMP` escape hatch. The openmp dispatch path is deleted, not gated.

### §3.1 — Implementation arc (12 commits, ~12-15 days focused engineering)

```
C1.  compute_splits() rewritten as single-threaded multi-backend
     iteration. Cross-backend deps via cudaEventRecord/cudaStreamWaitEvent.
     The openmp parallel block at ggml-backend.cpp:2215-2350 deleted in
     this commit.

C2.  cpy_tensor_async: copy_event pre-allocated at context init (not
     lazily). Kills the lazy-create race surface at ggml-cuda.cu:4472-4473.

C3.  Per-backend graph_compute extended with "in-capture" awareness:
     when called from inside an outer cudaStreamBeginCapture, the
     callee skips its own BeginCapture/EndCapture and emits nodes
     into the outer graph instead.

C4.  Outer cudaStreamBeginCapture(cudaStreamCaptureModeRelaxed) wraps
     the new compute_splits. Captures the entire multi-device forward
     pass into one cudaGraph_t.

C5.  Multi-device graph cache: key extended from topology_hash to
     (topology_hash, device_layout, n_seq). Eviction policy preserves
     the existing "too many updates → disable graph" guard.

C6.  CPU split handling — decision binds on PD3 enumeration:
     - If all CPU splits are upstream of GPU work → hoist-out (CPU
       splits run sync on host thread BEFORE cudaGraphLaunch)
     - If any CPU split is mid-graph → cudaLaunchHostFunc as a graph
       node (constraints: no CUDA API calls inside the host fn)

C7.  libmgpu port: per-device subgraphs become captured subgraphs in
     the outer graph. Delete libmgpu's openmp-dispatch usage. The
     mgpu split builders (Phase 46 graph-mode work) keep producing
     the same node sequence; only the eval driver changes.

C8.  Multi-stream-per-device ILP for libmgpu matmul (gated on PD6
     evidence). Use streams[device][0] and streams[device][1] for
     two halves of the matmul; reduce overlaps with compute. Same
     determinism contract: reduction order pinned per replay.

C9.  NCCL re-enable for cross-device REDUCE in libmgpu (gated on PD5
     evidence). The #ifdef GGML_USE_NCCL__ typo at ggml-cuda.cu:4449
     fixed. Limited to libmgpu's REDUCE shape; LM regular graphs
     continue without NCCL.

C10. Delete obsolete env knobs:
     - GGML_SCHED_EVAL_SERIALIZE (race surface gone)
     - GGML_SCHED_NO_DRAIN (drain itself deleted with C1)
     - GGML_CUDA_STREAM_SYNC (sync default no longer load-bearing)
     - GGML_SCHED_NO_ZERO_ACTIVATIONS (gallocr zeroing no longer needed)
     - GGML_CPY_POST_DEVICE_SYNC (Phase 46 NPC.4 diagnostic; obsolete)
     Delete the dead-code Phase 46 per-split drain at ggml-backend.cpp:2300-2310.

C11. Delete std::barrier non-openmp fallback path
     (ggml-backend.cpp:2353-2429). Production builds with openmp; the
     fallback was a portability claim that carried the same race
     surface. Verify all build configs work without openmp removed —
     if any build needs a non-openmp path, it's the new single-threaded
     dispatch (which works without openmp by construction).

C12. Verification commit: runs the full determinism battery + perf
     comparison vs PD4 baseline + A/B controls (against SERIALIZE on
     the same branch — both must be byte-identical). Commits the
     measurement artifacts to data/cuda-native-dispatch/.
```

Each commit must compile + pass `test-paged-allocator-determinism` (cheap). The branch lives as a unit until C12; bisection lands on a specific commit if a regression appears.

### §3.2 — What's NOT in this phase

- Kernel-level optimizations (FA, MMQ, Q4_0 throughput)
- Allocator changes (paged pool sizing, kv_pool_blocks)
- Sampler changes
- Reverting T5.9 paged BACKING

---

## §4 — Performance plan

All perf claims are conditional on PD4-PD6 measurement. The targets below become outputs of pre-design, not inputs to design.

### §4.1 — Baseline measurement (PD4)

Capture today's perf with the openmp dispatch:

- `scripts/bench-multislot.sh` at `N_PARALLEL ∈ {1, 2, 4, 8}` on production LM. Aggregate tg t/s, per-slot tg t/s, PP t/s, peak RSS.
- `scripts/verify-multigpu-clip.sh LATENCY_N=10` (CLIP encode median + p95). This is the libmgpu baseline.
- `llama-bench` PP+TG at NP=1 and NP=8 with `--device CUDA0,CUDA1 --split-mode graph`.
- Dispatch counter `total=N multi_seq=M` ratio at each NP (server-context.cpp:4811-4815).

Outputs land in `data/cuda-native-dispatch/baseline-<RUN_ID>/`.

### §4.2 — Pre-design go/no-go gates

**PD5 (NCCL on libmgpu reduce):** profile NCCL `ncclAllReduce` vs the current `cudaMemcpyPeer + element-wise add` on libmgpu's REDUCE shape (CLIP encoder layer output, F16, ~vision-token-count × hidden-dim). If NCCL is ≥10% faster, C9 ships. If neutral or worse, C9 is dropped from this phase and the libmgpu reduce stays on memcpy-peer.

**PD6 (multi-stream-per-device ILP for libmgpu matmul):** profile a single CLIP-layer matmul split across 2 streams per device (4 total compute streams) vs 1 stream per device (2 total). The matmul kernels are typically full-occupancy on RTX 6000 (Turing TU102, 72 SMs); the question is whether the reduce-axis split lets the partial-reduce-add overlap with the matmul tail. If overlap gives ≥5% on CLIP encode, C8 ships. If neutral or worse, C8 drops.

### §4.3 — Post-implementation perf targets (bound after PD4)

- LM TG at NP=1: ±5% of baseline (host parallelism never benefited NP=1)
- LM TG at NP=8 aggregate:
  - Conservative: 1.5× baseline (single cudaGraphLaunch amortizes host overhead across N decode tokens of same topology)
  - Stretch: 3× baseline (if cross-backend capture eliminates the openmp barrier overhead the dispatch counter analysis implies)
  - Ceiling: vLLM 154.77 t/s aggregate on same hardware (the `PHASE_NSTREAM_KV_PERF.md` reference; ~5.84× current)
- CLIP encode median: ≤ baseline (Phase 46's 14421 ms). Stretch: with C8+C9 active, target 50% reduction (vLLM-style perf on CLIP is the libmgpu motivation).
- Phase 46's 1.3× CPU-vision regression ceiling (42000 ms) remains in force as a hard gate.

---

## §5 — Verification

### §5.1 — Determinism gates (must PASS, no exceptions)

1. **`scripts/test-production-np-determinism.sh`** at NP ∈ {1, 2, 4, 8}, 3 reps each (12 invocations). Byte-identical NP=1 baseline; all slots byte-identical at each NP; cross-NP slot-0 matrix all identical. **Closing gate.**
2. **`scripts/r5-probe-c4.sh`** at 20 iters NP=2 single-GPU. Rate=0%.
3. **`scripts/verify-multigpu-clip.sh LATENCY_N=10`** — CLIP encode determinism (10/10 byte-identical `reasoning_content` sha256) AND median ≤ §4.3 target.
4. **All spec unit tests** at n_parallel ∈ {2, 8}: `test-n-stream-kv-layout`, `test-paged-allocator-determinism`, `test-paged-kshift-byte-identity`, `test-kv-defrag-per-stream`, `test-kv-shift-per-stream`, `test-mtp-x-n-stream`, `test-unified-stream-dispatch`, `test-multi-seq-decode-byte-identity`, `test-clip-encode-equivalence`, `test-clip-multi-backend-init`, `test-clip-weight-split`. All exit 0 or 77 SKIP.

### §5.2 — A/B controls (binding)

- **vs SERIALIZE on the same branch**: new dispatch path NP=8 byte-identical to new dispatch + `GGML_SCHED_EVAL_SERIALIZE=1` (which proves the new path has no latent race that SERIALIZE was hiding).
- **vs current main HEAD with SERIALIZE**: new dispatch path output byte-identical to main + `GGML_SCHED_EVAL_SERIALIZE=1` (which proves we haven't drifted numerically — the new path produces the SAME bits as the known-good serialized openmp path).
- **vs current main HEAD without SERIALIZE**: NP=1/2/4 byte-identical (current main is correct at NP=1/2/4; we just made NP=8 also correct without regressing the rest).

### §5.3 — New tests added by this phase

| Test | What it binds | Commit |
|---|---|---|
| `test-multi-device-graph-capture` (new) | Outer cudaStreamBeginCapture with multi-device deps produces a launchable `cudaGraph_t` with byte-identical output to non-captured eval. Cache hit rate >95% steady-state. | C4 |
| `test-ngpu-matmul-stream-ilp` (new) | libmgpu matmul with 2 streams per device produces byte-identical output to 1 stream per device (determinism contract preserved) AND ≥ ILP target latency (perf contract). | C8 |
| `test-ngpu-reduce-nccl-equiv` (new) | NCCL `ncclAllReduce` numerically equivalent (byte-identical or within fp16 ULP) to `cudaMemcpyPeer + add` on the libmgpu reduce shape. | C9 |

Tests land in the SAME commit as the feature. Not bolted on later.

### §5.4 — Production deploy gate

- All §5.1 + §5.2 PASS for the merge candidate.
- 24-hour soak at production NP=1 with the new dispatch. No /health drops, no slot-state leaks, RSS stable, no SEGV.
- Rollback drill: deploy the pre-phase build via existing `scripts/deploy-llama-server.sh` (no flag — pre-phase deploy uses the standard path). Confirm clean recovery. Redeploy forward.
- **There is no `--allow-no-mmproj-mgpu` analogue for this phase.** The build either runs or doesn't; if it doesn't, rollback is to the pre-phase build.

---

## §6 — Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| PD1 doesn't bind a specific racing field | Low | Single-threaded dispatch eliminates ALL host-concurrent CUDA access by construction; binding the specific field is for understanding, not gating implementation |
| PD2 shows multi-device capture doesn't work in CUDA 13.2 | Medium | Design pivots to per-device subgraph capture + `cudaGraphAddDependencies` stitching at parent-graph level. Same perf class, more code |
| C8/C9 land but don't deliver expected perf | Medium | PD5/PD6 gate them BEFORE landing; only land if profiling justifies |
| CPU split mid-graph requires cudaLaunchHostFunc with constraints | Low-Medium | PD3 enumeration determines whether this is needed; if it is, the host fn is constrained to read-only of pre-staged data |
| libmgpu port (C7) breaks existing CLIP determinism | Medium | New libmgpu tests in C7's commit; A/B against Phase 46's closure baseline sha `fb5167dbc1e7f95b` |
| std::barrier deletion (C11) breaks a build config | Low | Verify all build configs in CI; the non-openmp path becomes the new single-threaded dispatch (works without openmp by construction) |
| 24-hour soak surfaces a regression | Low | Rollback to pre-phase build via standard deploy script; the new path has no `--openmp-fallback` flag, so rollback is full-rebuild |
| Multi-stream ILP (C8) changes determinism bits | Medium | New test `test-ngpu-matmul-stream-ilp` binds byte-identity; if test fails, C8's design needs revision (e.g., deterministic reduction order across streams) |
| NCCL re-introduction surfaces past-incident hazards | Low-Medium | Limited to libmgpu's reduce shape only; LM regular graphs continue without NCCL; new test `test-ngpu-reduce-nccl-equiv` binds numerical equivalence |

---

## §7 — Out of scope

- Kernel-level optimizations beyond ILP (FA, MMQ, Q4_0 throughput).
- Allocator changes (paged pool sizing, kv_pool_blocks tuning).
- Sampler changes.
- Reverting T5.9 paged BACKING.
- LM-side NCCL re-enable (this phase enables NCCL only for libmgpu's reduce).
- Adding multi-stream ILP outside libmgpu's matmul (LM matmuls keep current single-stream-per-device).

---

## §8 — Acceptance

This phase closes when:

1. All commits C1-C12 merged.
2. All §5.1 determinism gates PASS.
3. All §5.2 A/B controls PASS.
4. PD4 perf targets met (LM NP=1 ±5%, LM NP=8 conservative target, CLIP encode ≤ baseline).
5. 24-hour soak completes with no incidents.
6. Production deployed; the build's commit stamp on the production stack matches the merge HEAD.
7. `MEMORY.md` records closure with perf delta vs PD4 baseline.
8. Auto-memory `project_cuda_native_dispatch_closed.md` written.

---

## §9 — Pre-design tasks (PD1-PD6)

These run BEFORE C1 begins. Each has a binding output that the implementation depends on.

### PD1 — Bind the specific racing field(s)

**Method**: read `ggml-backend.cpp:2215-2350` (openmp parallel region) and list every shared-state field accessed inside the `#pragma omp parallel` block. Cross-reference with `ggml-cuda.cu` for each field's thread-safety properties. Optionally instrument: add `__sync_fetch_and_add` counters to each candidate field's access points, rebuild, run G3.a NP=8 once with default openmp dispatch — the field with the highest cross-thread access count under the failing slot is the likely race.

**Output**: list of specific shared fields with their access patterns. Becomes part of the implementation's design notes for C1-C5.

**Effort**: 0.5-1 day. Read-only unless instrumentation is needed; instrumentation needs a maintenance window.

**Status**: not started.

### PD2 — Verify multi-device `cudaStreamCaptureModeRelaxed`

**Method**: write a ~50-line standalone CUDA test that:
1. Begins capture on stream A (CUDA0) with mode=Relaxed
2. Records event on stream A
3. Switches to CUDA1 via `cudaSetDevice`
4. Allocates a tensor on CUDA1
5. Has stream B (CUDA1) wait on the event
6. Enqueues a kernel on stream B
7. Records event on stream B
8. Switches back to CUDA0
9. Has stream A wait on B's event
10. End capture; launch the resulting graph; verify the output

**Output**: yes/no on multi-device capture. If yes, C3-C5 use single-graph capture. If no, C3-C5 use per-device subgraph capture + parent-graph stitching.

**Effort**: 0.5 day. Standalone test, no production impact.

**Status**: not started.

### PD3 — Enumerate CPU splits in production LM and libmgpu graphs

**Method**: enable sched logging (search for an existing log knob in ggml-backend.cpp or add one temporarily), run one LM decode + one CLIP encode at production config, capture the split list. Tag each CPU split with: (a) what op, (b) what data flows into it, (c) what data flows out, (d) whether it has GPU dependencies upstream.

**Output**: list of CPU splits; decision binds on C6.

**Effort**: 0.5-1 day. Read-only if existing log knob; small instrumentation if not.

**Status**: not started.

### PD4 — Baseline perf measurement

**Method**: maintenance window, production stopped, GPUs locked. Run:
- `scripts/bench-multislot.sh` at N_PARALLEL ∈ {1, 2, 4, 8}
- `scripts/verify-multigpu-clip.sh LATENCY_N=10`
- `llama-bench` PP+TG at NP=1, NP=8 with split-mode graph
- Dispatch counter `total/multi_seq` at each NP

**Output**: baseline numbers; perf targets in §4.3 become bound to these numbers.

**Effort**: 1 maintenance window (~60-90 min).

**Status**: not started.

### PD5 — NCCL vs cudaMemcpyPeer profile on libmgpu reduce

**Method**: write a microbenchmark that:
- Allocates the same input shape as libmgpu's REDUCE (CLIP encoder layer output, F16, vision-token-count × hidden-dim)
- Runs N=1000 iterations of `cudaMemcpyPeer + element-wise add` (current path)
- Runs N=1000 iterations of `ncclAllReduce` (target path)
- Measures p50, p95, p99 latency for both
- Verifies numerical equivalence (byte-identical or within fp16 ULP)

**Output**: go/no-go on C9. Threshold: NCCL ≥10% faster → ship. Otherwise drop C9.

**Effort**: 1 day (need to write the microbenchmark, then run during a maintenance window).

**Status**: not started.

### PD6 — Multi-stream-per-device ILP profile on libmgpu matmul

**Method**: write a microbenchmark that:
- Allocates the same matmul shape as libmgpu uses for a CLIP layer
- Runs N=1000 iterations with 1 compute stream per device (current)
- Runs N=1000 iterations with 2 compute streams per device (split matmul row-wise, partial-reduce overlapped)
- Measures total wall time per iteration
- Verifies numerical equivalence

**Output**: go/no-go on C8. Threshold: ≥5% latency improvement → ship. Otherwise drop C8.

**Effort**: 1 day.

**Status**: not started.

### Pre-design summary

| Task | Effort | Maintenance window? | Gates |
|---|---|---|---|
| PD1 | 0.5-1 day | optional | informs C1-C5 |
| PD2 | 0.5 day | no | gates capture-mode decision |
| PD3 | 0.5-1 day | maybe | gates CPU split handling (C6) |
| PD4 | maintenance window | YES (~60-90 min) | binds §4.3 perf targets |
| PD5 | 1 day + window | YES (folded with PD4 + PD6) | go/no-go on C9 |
| PD6 | 1 day + window | YES (folded with PD4 + PD5) | go/no-go on C8 |

**Total pre-design**: ~5-7 days engineering + one combined maintenance window for PD4 + PD5 + PD6. After pre-design, implementation begins with full visibility into perf targets and gated features.

---

## §10 — Pre-design results (PD1, PD2, PD3 — completed 2026-05-27)

### PD1 result — racing fields enumerated

Inside `ggml-backend.cpp:2215-2350` (openmp parallel block), the `if (ith == split_backend_id) ggml_backend_sched_eval(...)` descends into `ggml_backend_cuda_graph_compute`. Four CUDA-context shared fields are accessed without locks under concurrent multi-thread entry on the SAME backend context:

| Racing field | Site | Mechanism |
|---|---|---|
| `cuda_ctx->cuda_graphs` (unordered_map) | `ggml-cuda.cu:5176` `ggml_cuda_get_graph()` | unprotected map find/insert/erase under concurrent threads on same context |
| `cuda_ctx->pools[device]` | `ggml-cuda/common.cuh:959-964` | lazy-create `if (pools[device] == nullptr) new_pool_for_device(...)` |
| `cuda_ctx->cublas_handles[device]` | `ggml-cuda/common.cuh:931-948` | lazy-create `if (cublas_handles[device] == nullptr) cublasCreate(...)` |
| `cuda_ctx->streams[device][n]` | `ggml-cuda/common.cuh:903-909` | lazy-create `if (streams[device][n] == nullptr) cudaStreamCreateWithFlags(...)` |

`copy_event` lazy-create at `ggml-cuda.cu:4472-4473` is single-thread accessed via `copy_thread` (only one thread calls `cpy_tensor_async` per split per audit), so it's not in the racing set BUT still a latent race surface if any future change makes cpy_tensor_async concurrent.

`sched->statuses[ith]`, `sched->needs_sync[]`, `sched->events[*][*]` are all guarded by openmp barriers between writes and reads. Not racing.

**C1 design binding**: single-threaded dispatch eliminates ALL four racing fields by construction. **C2 generalized**: pre-allocate `copy_event`, `pools`, `cublas_handles`, AND `streams` at context init — kills every lazy-create race surface, not just `copy_event`. Adjusts the implementation arc: rename C2 to "pre-allocate all CUDA-context lazy fields at init."

### PD2 result — multi-device single-graph capture works in CUDA 13.2

Standalone test `data/cuda-native-dispatch/predesign/pd2_multi_device_capture.cu` compiled with `nvcc -ccbin /usr/bin/g++-15` and run on production hardware (2× Quadro RTX 6000, CUDA 13.2):

- `cudaStreamBeginCapture(stream_CUDA0, cudaStreamCaptureModeRelaxed)`
- Kernel on CUDA0 stream
- `cudaEventRecord(e0, stream_CUDA0)`
- `cudaSetDevice(1); cudaStreamWaitEvent(stream_CUDA1, e0); kernel on stream_CUDA1; cudaEventRecord(e1, stream_CUDA1)`
- `cudaSetDevice(0); cudaStreamWaitEvent(stream_CUDA0, e1); kernel on stream_CUDA0`
- `cudaStreamEndCapture(stream_CUDA0, &graph)` → single `cudaGraph_t`
- `cudaGraphInstantiate` + `cudaGraphLaunch(graph, stream_CUDA0)` → correct output across both devices

Captured graph has 3 nodes (one per kernel); events became edge dependencies, not separate nodes. Graph size scales with kernel count.

**C3-C5 design binding**: single-graph capture. NOT per-device subgraph stitching. Simpler than the contingency. The outer `cudaStreamBeginCapture` in C4 wraps the entire `compute_splits` loop; cross-device dependencies live inside the captured graph.

### PD3 result — CPU splits are upstream-only

Static analysis of `ggml_backend_sched_split_graph` at `ggml-backend.cpp:1412-1850`:

- 5-pass assignment algorithm; CPU is lowest-priority backend (index `n_backends - 1`).
- Pass 2 (lines 1517-1599) **prevents** CPU nodes from propagating between GPU zones — explicit skip at line 1532.
- Mainstream LM forward-pass ops (GET_ROWS, ROPE, SOFT_MAX, FLASH_ATTN_EXT, MUL_MAT, RMS_NORM, REDUCE, SUM_ROWS) all have CUDA implementations and assign to GPU when weights are GPU-resident.
- Production graph inputs (token IDs, positions, attention masks) are marked `GGML_TENSOR_FLAG_INPUT` and default to CPU at line 1308, then `tensor_id_copy()` creates GPU mirrors at lines 1811-1842 — pure upstream.

`/tmp/production-np-determinism/run-20260527T091357/server-np8.log` and `/tmp/phase46-multigpu-clip/run-20260527T092327/` show 387 total splits for CLIP at NP=8 with no "CPU" backend named in any split. Production uses `--no-mmproj-offload` (mmproj on GPU).

`GGML_SCHED_DEBUG=1` (`ggml-backend.cpp:2509`) prints the split list for verification — useful during implementation to confirm hoist-out covers every CPU split.

**C6 design binding**: hoist-out pattern. CPU input staging runs on the host thread BEFORE `cudaGraphLaunch`. No `cudaLaunchHostFunc` plumbing needed. No mid-graph CPU dependencies. The implementation simplifies materially.

### PD4 / PD5 / PD6 — still pending

Combined maintenance window required (~3-4 hours):
- PD4: baseline perf measurement (`bench-multislot.sh` NP={1,2,4,8}, `verify-multigpu-clip.sh LATENCY_N=10`, `llama-bench`)
- PD5: NCCL `ncclAllReduce` vs `cudaMemcpyPeer + add` microbenchmark on libmgpu reduce shape
- PD6: 1-stream vs 2-stream-per-device microbenchmark on libmgpu matmul shape

PD5/PD6 microbenchmarks not yet written — that's the next slug of work, before the maintenance window.

---

## §11 — Formal specifications (Allium + TLA+) — content DEFERRED until PD4-PD6 integrated

User directive 2026-05-27: "we need allium and TLA+ content for this document, add once we have data and integrated it."

This section is filled AFTER PD4-PD6 complete and after the design's parameter values (multi-stream count, NCCL message shape, capture cache key structure) are bound by measurement. Pattern follows `PHASE_NSTREAM_KV_PERF.md` §"P0.B — Radical Allium / TLA+ / test surface expansion" (5 Allium specs + 3 TLA+ specs + 5 property tests + trace-harness extension).

### §11.1 — Allium specs (planned, content TBD)

Five specs covering the new dispatch's correctness invariants. Locations: `/home/dconnolly/yarn-agentic/specs/cuda-native-dispatch/`. Working titles:

1. **`single_threaded_dispatch.allium`** — ALL host-side CUDA driver state mutations happen on exactly one host thread per process lifetime. No concurrent entry into `ggml_backend_sched_compute_splits` or below. Binds C1.
2. **`cross_device_event_chain.allium`** — every cross-device data dependency in the captured graph has a `cudaEventRecord` on the producer + `cudaStreamWaitEvent` on the consumer. No data read on device D before its producing write is event-synchronized. Binds C4 + C7.
3. **`multi_device_graph_cache.allium`** — graph cache key `(topology_hash, device_layout, n_seq)` produces identical `cudaGraph_t` for matching keys. Cache hit replays bit-identically to capture. Eviction FIFO bounded by `GGML_CUDA_GRAPH_MAX`. Binds C5.
4. **`libmgpu_subgraph_capture.allium`** — libmgpu per-device subgraphs capture cleanly into the outer multi-device graph. Cross-device REDUCE produces byte-identical output to the pre-phase memcpy-peer+add (or NCCL ncclAllReduce when C9 is active). Binds C7 + C9.
5. **`multi_stream_ilp_determinism.allium`** — per-device matmul split across 2 streams produces byte-identical output to 1-stream. Reduction order pinned per replay (same captured graph always sequences partial-reduce identically). Binds C8.

### §11.2 — TLA+ specs (planned, content TBD)

Three models covering dispatch safety + liveness. Locations: `/home/dconnolly/yarn-agentic/specs/cuda-native-dispatch/`.

1. **`CUDANativeDispatch.tla`** — single-threaded `compute_splits` as a state machine `{IDLE, ENQUEUING, CAPTURING, LAUNCHED, COMPLETE}`. Verifies absence of host-side races, event-record/wait ordering, every split's eval completes before the graph end.
2. **`CUDAGraphCacheConsistency.tla`** — multi-device graph cache. Insertions, evictions, lookups. Verifies cache key uniqueness (no hash collisions) and that evicted-then-re-captured graphs are bit-identical.
3. **`CrossDeviceReduceOrdering.tla`** — libmgpu's REDUCE under both NCCL and memcpy-peer+add paths. Verifies the two paths produce equivalent numerical output under the same input sequence (fp16 ULP bound).

### §11.3 — Property tests + trace-harness extensions (planned)

Five property tests in `tests/spec/` bound by the Allium specs:

1. `test-single-threaded-dispatch.cpp` — counts threads entering `compute_splits` over N graph computes; asserts always == 1.
2. `test-cross-device-event-chain.cpp` — instruments event-record/event-wait pairs; asserts every consumer wait has a matching producer record before it.
3. `test-multi-device-graph-cache.cpp` — runs N captures of the same topology; asserts cache hit rate steady-state, bit-identical replay output.
4. `test-libmgpu-subgraph-capture.cpp` — captures libmgpu's per-device subgraphs; asserts byte-identity to the pre-phase non-captured output.
5. `test-multi-stream-ilp-determinism.cpp` — same matmul under 1-stream vs 2-stream; asserts byte-identical (or fp16 ULP bound if reduction order differs).

Trace-harness extension: `cudaGraphTraceCapture` + `cudaGraphTraceLaunch` hooks (CUDA 12+ `cudaGraphAddEventRecordNode` introspection) → JSON trace of captured nodes for property tests to parse.

### §11.4 — Integration order

Once PD4/PD5/PD6 land:
1. Update §4.3 perf targets with measured numbers.
2. Update §3.1 commit list with C8/C9 finalized go/no-go from PD5/PD6.
3. Write §11.1 + §11.2 Allium + TLA+ spec stubs at `specs/cuda-native-dispatch/`.
4. Each spec gets a placeholder commit BEFORE its implementation commit (C1 ships with `single_threaded_dispatch.allium` already in tree).
5. Property tests in §11.3 land with their feature commits (per §5.3 "tests land in the SAME commit as the feature").
6. TLA+ model-checking runs in CI before merge via the existing `.github/workflows/spec-tla-gate.yml` gate.
