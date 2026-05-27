# PHASE_CUDA_NATIVE_DISPATCH — ground-up CUDA-idiomatic dispatch redesign for libmgpu (and LM as beneficiary)

**Status**: Design locked, pre-design (PD1-PD6) complete (see §10). Implementation arc C0-C14 ready to start (~15-18 days focused engineering).
**Branch base**: parent repo off `origin/main@628aaae`; submodule off `ik_llama.cpp@e6cb4c1b` on `production/2026-q2-next` (includes the 2026-05-27 `test-n-stream-kv-layout` refresh for T5.9 paged invariants — required for the verification battery in §5.1.4 to assert post-T5.9 shapes, NOT pre-T5.9).
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

### §3.0 — Calibrated dispatch as the default pattern

User directive 2026-05-27: "Do size adaptive for everything possible, make it the default pattern."

The PD5 NCCL-vs-memcpy sweep showed the right answer depends on hardware (NVLink topology, GPU SM count, PCIe gen) AND payload size. Hardcoding a threshold for xeon's specific config (2× Quadro RTX 6000 + NV2) is fragile — the same binary on different hardware would pick the wrong strategy. So instead of hardcoding, the design uses **calibrated dispatch** as the architectural primitive for every multi-strategy decision.

#### Framework

```cpp
// ggml/src/ggml-cuda/calibration.{h,cu}

enum ggml_cuda_calibrated_op {
    GGML_CAL_REDUCE_CROSS_DEVICE,   // memcpy-peer+add  vs  ncclAllReduce
    GGML_CAL_MATMUL_STREAM_SPLIT,   // 1-stream cuBLAS  vs  2-stream split
    GGML_CAL_PEER_COPY,             // direct cudaMemcpyPeer  vs  staged-via-pinned
    GGML_CAL_GRAPH_CAPTURE,         // capture vs eager (small-graph case)
    // future entries land here
};

struct ggml_cuda_calibration_table {
    std::unordered_map<int, size_t> thresholds;  // op_id → byte threshold
    bool                            loaded_from_cache;
    std::string                     cache_key;   // (gpu_uuid, cuda_version, ggml_version)
};

// Run at context init. Probes each registered op at representative
// payload sizes; records the smallest size where alt-strategy.p95 <
// default-strategy.p50 ("conservative crossover"). Stores SIZE_MAX
// if no crossover in probe range.
void ggml_cuda_calibrate(ggml_backend_cuda_context * ctx);

// Per-dispatch lookup. Each calibrated op site reads this and routes.
size_t ggml_cuda_threshold_for(ggml_backend_cuda_context * ctx,
                               ggml_cuda_calibrated_op op);
```

#### Contract every calibrated op must satisfy

1. **Strategies are numerically equivalent** — byte-identical, or fp16/fp32 ULP-bound, documented per op. Without this contract, calibration variance becomes a determinism gap.
2. **One probe function per op**, returning the chosen strategy's `(p50, p95)` latency at the requested payload size.
3. **Crossover criterion is uniform**: smallest payload size where `alt.p95 < default.p50`. No exceptions; same predicate everywhere.
4. **Threshold is quantized** to the nearest of **`{0, 1 MB, 10 MB, 100 MB, 1 GB, SIZE_MAX}`** (LOCKED 2026-05-27 — five buckets, log10-spaced, plus the SIZE_MAX sentinel). Probes run at the same sizes; the calibration result is the smallest bucket where `alt.p95 < default.p50`. Five probes × four ops × ~100 ms per probe ≈ ~2 s total at first launch. The bucket set can be refined in a future commit if production data shows insufficient resolution near the crossover; for the big-bang, 5 buckets balance calibration cost against threshold granularity.
5. **No fallback at runtime** — once the threshold is set (from probe or cache), dispatch picks strategy deterministically per call.

#### Cache

```
~/.cache/ggml/cuda-calibration-{gpu_uuid_hash}.json:
{
  "schema_version": 1,
  "cache_key": "<gpu_uuid_hash>-cuda<version>-ggml<commit>",
  "calibrated_at": "<iso8601>",
  "thresholds": {
    "REDUCE_CROSS_DEVICE": 1073741824,  // 1 GB bucket on xeon (true crossover 750 MB; quantized up per §3.0 rule 4)
    "MATMUL_STREAM_SPLIT": null,        // SIZE_MAX → never split on xeon (PD6)
    "PEER_COPY":           null,        // SIZE_MAX → direct path always wins
    "GRAPH_CAPTURE":       0            // always capture above 0 nodes
  }
}
```

- First launch on a host: ~1.5–2 s total (4 ops × 4 probe sizes × ~100 ms per probe), then write cache.
- Subsequent launches: cache hit → ~10 ms to load + validate JSON.
- A `cache_key` mismatch (gpu/cuda/ggml change) triggers re-calibrate.
- Env overrides:
  - `GGML_CALIBRATION_DISABLE=1` — skip cache I/O, run every time
  - `GGML_CALIBRATION_FORCE_RECALIBRATE=1` — ignore cache, re-probe
  - `GGML_CAL_<OP>_THRESHOLD_BYTES=N` — per-op manual override

#### Determinism story

- **First cold launch**: calibration runs; threshold reflects measured hardware behavior. Quantization (rule 4) prevents tiny variance from flipping the chosen strategy.
- **Subsequent launches**: cache hit → same threshold → same dispatch decisions → same output bits.
- **Cross-host portability**: each host has its own cache; same binary on different hardware adapts.
- **Equivalence contract (rule 1)** is the safety net — even if threshold changed across runs, output bits don't.

#### What's calibrated (in scope for this phase)

| Op | Default strategy | Alt strategy | Expected xeon result |
|---|---|---|---|
| `REDUCE_CROSS_DEVICE` | memcpy-peer + add | ncclAllReduce | True crossover ~750 MB (PD5); quantizes to **1 GB bucket** per §3.0 rule 4. libmgpu production reduce ~2-3 MB → memcpy always |
| `MATMUL_STREAM_SPLIT` | 1-stream cuBLAS | 2-stream column-split | SIZE_MAX on xeon (PD6); finite on hardware where 2-stream wins |
| `PEER_COPY` | direct cudaMemcpyPeer | staged via pinned host | SIZE_MAX on NVLink fabric; finite on poor PCIe topology |
| `GRAPH_CAPTURE` | capture | eager-launch | 0 on modern CUDA (always capture); finite if capture-overhead-dominated graphs exist |

Each calibrated op gets its own commit; the framework provides the common abstraction.

#### What's NOT calibrated

- **Kernel block-size selection** — lives below dispatch layer; per-kernel auto-tune is a different phase.
- **cuBLAS algorithm picking** — breaks the determinism contract (different algos → different reduction trees → different bits). The codebase pins ALGO0_TENSOR_OP and that stays.
- **Dynamic re-calibration during runtime** — hardware doesn't change mid-process; one calibration at init is sufficient.
- **LM op calibration** — this phase scopes to libmgpu's cross-device ops + the cross-cutting matmul/copy/capture primitives. LM kernel tuning is a future phase.

### §3.1 — Implementation arc (14 commits, ~15-18 days focused engineering)

```
C0.  Calibration framework (NEW per §3.0). Implements
     ggml/src/ggml-cuda/calibration.{h,cu}: probe abstraction, JSON
     cache I/O at ~/.cache/ggml/cuda-calibration-{key}.json, context
     init hook. Initial calibrated op registry is empty; ops register
     themselves in C8/C9/C10/C11. Adds env knobs
     GGML_CALIBRATION_DISABLE, GGML_CALIBRATION_FORCE_RECALIBRATE.
     New test: test-cuda-calibration-framework.cpp.

C1.  compute_splits() rewritten as single-threaded multi-backend
     iteration. Cross-backend deps via cudaEventRecord/cudaStreamWaitEvent.
     The openmp parallel block at ggml-backend.cpp:2215-2350 deleted in
     this commit.

C2.  Eager pre-allocation of CUDA-context lazy fields at context init:
     copy_event, pools[device], cublas_handles[device], and
     streams[device][n] for each (device, stream_idx) the dispatch
     will use. Kills all four race surfaces enumerated in PD1
     (ggml-cuda.cu:4472-4473; common.cuh:903-948,959-964) by removing
     the lazy-create branches entirely.

C3.  Per-backend graph_compute extended with "in-capture" awareness:
     when called from inside an outer cudaStreamBeginCapture, the
     callee skips its own BeginCapture/EndCapture and emits nodes
     into the outer graph instead.

C4.  Outer cudaStreamBeginCapture(cudaStreamCaptureModeRelaxed) wraps
     the new compute_splits. Captures the entire multi-device forward
     pass into one cudaGraph_t. (PD2 verified this works on production
     hardware in CUDA 13.2.)

C5.  Multi-device graph cache: key extended from topology_hash to
     (topology_hash, device_layout, n_seq). Eviction policy preserves
     the existing "too many updates → disable graph" guard.

C6.  CPU split hoist-out (per PD3: production LM + libmgpu CLIP have
     no mid-graph CPU splits; all CPU work is upstream of GPU). CPU
     splits run sync on host thread BEFORE cudaGraphLaunch. No
     cudaLaunchHostFunc plumbing needed.

C7.  libmgpu port: per-device subgraphs become captured subgraphs in
     the outer graph. Delete libmgpu's openmp-dispatch usage. The
     mgpu split builders (Phase 46 graph-mode work) keep producing
     the same node sequence; only the eval driver changes.

C8.  Register GGML_CAL_MATMUL_STREAM_SPLIT with the calibration
     framework. Implementation: 1-stream cuBLAS hgemm (default) +
     2-stream column-split alt path. Probe at 4 representative
     matmul shapes; calibration table populated. On xeon PD6 expects
     SIZE_MAX (1-stream always); on future hardware the threshold
     auto-discovers. New test:
     test-calibration-equivalence-matmul.cpp.

C9.  Register GGML_CAL_REDUCE_CROSS_DEVICE. Default strategy:
     existing memcpy-peer + add. Alt strategy: ncclAllReduce behind
     GGML_USE_NCCL build flag (which is fixed from the ggml-cuda.cu:4449
     typo `#ifdef GGML_USE_NCCL__` → `GGML_USE_NCCL`). Eager
     ncclComm_t init at context creation. Probe at 4 sizes; on xeon
     PD5 expects ~750 MB threshold. New test:
     test-calibration-equivalence-reduce.cpp.

C10. Register GGML_CAL_PEER_COPY. Default strategy: direct
     cudaMemcpyPeerAsync. Alt strategy: staged via pinned-host
     intermediate. Probe at 4 sizes; on xeon NVLink fabric likely
     SIZE_MAX (direct always wins). New test:
     test-calibration-equivalence-peer-copy.cpp.

C11. Register GGML_CAL_GRAPH_CAPTURE. Default strategy:
     cudaGraphLaunch (capture path). Alt strategy: eager kernel
     launches with no capture. Probe at 4 graph sizes (small/med/
     large/huge node counts). On modern CUDA expects 0 threshold
     (always capture). New test:
     test-calibration-equivalence-graph.cpp.

C12. Delete obsolete env knobs (race surfaces gone):
     - GGML_SCHED_EVAL_SERIALIZE
     - GGML_SCHED_NO_DRAIN
     - GGML_CUDA_STREAM_SYNC
     - GGML_SCHED_NO_ZERO_ACTIVATIONS
     - GGML_CPY_POST_DEVICE_SYNC (Phase 46 NPC.4 diagnostic)
     Delete the dead-code Phase 46 per-split drain at ggml-backend.cpp:2300-2310.

C13. Delete std::barrier non-openmp fallback path
     (ggml-backend.cpp:2353-2429). The new single-threaded dispatch
     works without openmp by construction.

C14. Verification commit: runs full determinism battery + perf
     comparison vs PD4 baseline + A/B controls (against SERIALIZE on
     same branch — both must be byte-identical) + calibration cache
     sanity (cache hit reproduces same thresholds; force-recalibrate
     produces matching results within quantization tolerance).
     Commits measurement artifacts to data/cuda-native-dispatch/.
```

Each commit must compile + pass `test-paged-allocator-determinism` (cheap). The branch lives as a unit until C14; bisection lands on a specific commit if a regression appears.

**Spec/test pairing per commit** (mirrors §11.4):
- C0 ships `calibrated_dispatch_framework.allium` (live) + `CalibrationFramework.tla` (model-checked in CI) + `test-cuda-calibration-framework.cpp`.
- C1 ships `single_threaded_dispatch.allium` + `CUDANativeDispatch.tla` + `test-single-threaded-dispatch.cpp`.
- C4 ships `cross_device_event_chain.allium` + `test-cross-device-event-chain.cpp` + `test-multi-device-graph-capture.cpp`.
- C5 ships `multi_device_graph_cache.allium` + `CUDAGraphCacheConsistency.tla` + `test-multi-device-graph-cache.cpp`.
- C7 ships `libmgpu_subgraph_capture.allium` + `test-libmgpu-subgraph-capture.cpp`.
- C8/C9/C10/C11 each ship one `test-calibration-equivalence-{matmul,reduce,peer-copy,graph}.cpp` + incremental updates to the shared `calibrated_op_equivalence.allium`. `CalibratedOpEquivalence.tla` model-check runs in C11 once all four ops are registered.
- C12/C13 ship no new specs (cleanup commits — they DELETE knobs and dead code).
- C14 ships `data/cuda-native-dispatch/post-merge-<RUN_ID>/` evidence + the closure `MEMORY.md` entry (separate commit per CLAUDE.md §6).

### §3.2 — What's NOT in this phase

- Kernel-level optimizations (FA, MMQ, Q4_0 throughput)
- Allocator changes (paged pool sizing, kv_pool_blocks)
- Sampler changes
- Reverting T5.9 paged BACKING

---

## §4 — Performance plan

All perf claims are bound to PD4 measured baselines (see §10). The PD4-PD6 measurements are integrated; perf targets in §4.3 are now binding numbers, not placeholders.

### §4.1 — Baseline measurement (PD4 — completed)

Capture today's perf with the openmp dispatch:

- `scripts/bench-multislot.sh` at `N_PARALLEL ∈ {1, 2, 4, 8}` on production LM. Aggregate tg t/s, per-slot tg t/s, PP t/s, peak RSS.
- `scripts/verify-multigpu-clip.sh LATENCY_N=10` (CLIP encode median + p95). This is the libmgpu baseline.
- `llama-bench` PP+TG at NP=1 and NP=8 with `--device CUDA0,CUDA1 --split-mode graph`.
- Dispatch counter `total=N multi_seq=M` ratio at each NP (server-context.cpp:4811-4815).

Baselines landed at `data/cuda-native-dispatch/predesign/pd4-baseline-20260527T121951/`. Full results in §10.

### §4.2 — Calibration framework reframes the original PD5/PD6 go/no-go

The original design treated PD5 and PD6 as binary go/no-go gates for C8/C9. Under the calibrated-dispatch framework (§3.0), both ops always register — what changes is the threshold value calibration returns on each hardware:

- **PD5** (NCCL vs memcpy-peer+add) gave the bucket where NCCL's `p95 < memcpy's p50`. On xeon (NV2 NVLink + PCIe Gen3), the conservative crossover is between 500 MB and 1 GB → calibration quantizes to **1 GB bucket** → NCCL never fires on libmgpu's production reduce shape (~2-3 MB), but does fire if any future workload exceeds 1 GB. Equivalence is byte-identical (PD5 §10).

- **PD6** (1-stream vs 2-stream matmul) gave no crossover in tested shapes. Calibration returns `SIZE_MAX` on xeon → 1-stream always fires. The framework records this as the expected outcome; on different hardware (more SMs, smaller per-matmul tile) calibration would find a finite threshold and the alt-strategy would auto-activate.

The framework decoupling means C8 and C9 ship UNCONDITIONALLY (vs the original "drop on no perf benefit" gate). Threshold values are hardware-portable.

### §4.3 — Post-implementation perf targets (bound by PD4 baseline)

- **LM TG at NP=1**: ±5% of **17.9 t/s** baseline → 17.0–18.8 t/s (host parallelism never benefited NP=1).
- **LM TG at NP=8 aggregate**:
  - Conservative: 1.5× **30.8 t/s** → **46 t/s** (single cudaGraphLaunch amortizes host overhead across decode tokens of same topology).
  - Stretch: 2× → **60 t/s** (cross-backend capture + per-context state pre-allocation eliminates all openmp dispatch overhead and lazy-create races).
  - Aspirational: 3× → **92 t/s** (if calibration enables C8/C9 alt-paths on libmgpu's tensor-parallel matmul/reduce flow at production shape — unlikely on xeon per PD5/PD6, but framework is hardware-portable).
  - Ceiling: vLLM 154.77 t/s aggregate on same hardware (PHASE_NSTREAM_KV_PERF.md reference; ~5.0× current). Closing more of that gap requires kernel-level work outside this phase.
- **CLIP encode median**: ≤ **14450 ms** baseline (PD4). Conservative: 10-20% reduction from graph-launch amortization → 11500–13000 ms. Stretch: calibration finds NCCL beneficial at some libmgpu reduce shape → further reduction.
- **Phase 46's 1.3× CPU-vision regression ceiling** (42000 ms / 1.3 = 54600 ms) remains in force as a hard gate.

---

## §5 — Verification

### §5.1 — Determinism gates (must PASS, no exceptions)

> Note: each invocation triggers a calibration on first context creation (~1.5-2 s cold; ~10 ms on cache hit). G3.a's 12 invocations cold add ~24 s to the battery; warm (cache populated) adds ~120 ms. Negligible. The calibration cache file (`~/.cache/ggml/cuda-calibration-{key}.json`) for the running user persists across invocations within a battery run.

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
| `test-cuda-calibration-framework` (new) | Calibration produces deterministic thresholds; cache load reproduces; quantization absorbs noise. | C0 |
| `test-single-threaded-dispatch` (new) | Counts threads entering `compute_splits` over N graph computes; always == 1. | C1 |
| `test-cross-device-event-chain` (new) | Every consumer `cudaStreamWaitEvent` has a matching producer `cudaEventRecord` before it. | C4 |
| `test-multi-device-graph-capture` (new) | Outer cudaStreamBeginCapture with multi-device deps produces a launchable `cudaGraph_t` with byte-identical output to non-captured eval. | C4 |
| `test-multi-device-graph-cache` (new) | N captures of same topology → cache hit rate >95% steady-state; bit-identical replay output across cache hits. | C5 |
| `test-libmgpu-subgraph-capture` (new) | libmgpu per-device subgraphs capture into outer graph; output byte-identical to Phase 46 closure baseline sha `fb5167dbc1e7f95b`. | C7 |
| `test-calibration-equivalence-matmul` (new) | 1-stream vs 2-stream matmul: fp16 ULP-bound equivalence at all probed sizes. | C8 |
| `test-calibration-equivalence-reduce` (new) | memcpy-peer+add vs ncclAllReduce: byte-identity at all probed sizes (PD5 confirmed). | C9 |
| `test-calibration-equivalence-peer-copy` (new) | direct vs staged-via-pinned peer copy: byte-identity. | C10 |
| `test-calibration-equivalence-graph` (new) | captured vs eager-launch: byte-identity. | C11 |

10 new tests total. Each lands in the SAME commit as the feature it binds. Not bolted on later.

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
| Multi-stream ILP (C8) changes determinism bits | Medium | Calibration's equivalence contract (§3.0 rule 1) binds fp16 ULP-bound output across strategies; `test-calibration-equivalence-matmul.cpp` enforces. If contract fails, C8 cannot register against the framework |
| NCCL re-introduction surfaces past-incident hazards | Low-Medium | NCCL only activates above calibrated threshold (xeon: ~750 MB; LM regular graphs always below). Equivalence test binds byte-identity (PD5 confirmed at all measured shapes) |
| Calibration variance produces different thresholds across runs → different dispatch decisions → different output bits | Low-Medium | Two-layer mitigation: (a) threshold quantization (§3.0 rule 4) absorbs measurement noise; (b) equivalence contract (rule 1) ensures output is bit-equivalent regardless of which strategy fires. Even if threshold flips, output doesn't |
| Calibration probe OOMs on small-VRAM hardware | Low | Probe sizes auto-shrink to free VRAM at calibration time; if even minimum probe doesn't fit, threshold = SIZE_MAX (memcpy/default only) |
| Calibration cache I/O failure (disk full, permissions) | Low | Cache failure is non-fatal; falls back to in-memory calibration each launch |
| Submodule push gate — every C0-C14 commit lives in `ik_llama.cpp/` and bumps the parent pointer; the user has a standing "never push submodule fork without explicit authorization" rule | Operational | Each commit batched and authorized in groups (typically 2-3 commits per authorization window). The parent-pointer bump in `yarn-agentic` waits for the submodule push to succeed first. Branch lives local + on fork only; no force-push to upstream `ikawrakow/ik_llama.cpp.git` |
| Production rollback Recovery Time Objective (RTO) | Operational | Rollback path is `scripts/deploy-llama-server.sh` against the pre-phase build tree (preserved at the branch base commit `1db6c2eb`). End-to-end RTO target: 5 minutes — `systemctl stop` (1 min) + deploy script (2 min) + `/health` confirm (2 min). 24-hour soak is the test window; once soak passes, rollback drills become quarterly |

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

1. All commits **C0-C14** merged (15 commits total: framework + 11 implementation + 3 cleanup/verification).
2. All §5.1 determinism gates PASS (12 G3.a invocations + G3.c + CLIP encode + 11 spec unit tests).
3. All §5.2 A/B controls PASS (vs SERIALIZE on same branch; vs main HEAD with SERIALIZE; vs main HEAD without SERIALIZE at NP≤4).
4. §4.3 perf targets met (LM NP=1 within ±5% of 17.9 t/s; LM NP=8 aggregate ≥ 46 t/s conservative target; CLIP encode median ≤ 14450 ms).
5. 24-hour soak completes at production NP=1 with no /health drops, no slot-state leaks, RSS stable, no SEGV.
6. Production deployed via existing `scripts/deploy-llama-server.sh`; build's commit stamp on the production stack matches the merge HEAD.
7. Rollback drill PASS: pre-phase build (`ik_llama.cpp@1db6c2eb`) deploys cleanly via the same deploy script; recovery within §6's 5-minute RTO target.
8. `MEMORY.md` records closure (separate commit per CLAUDE.md §6) with perf delta vs PD4 baseline.
9. Auto-memory `project_cuda_native_dispatch_closed.md` written.
10. `SUMMARY.md` (mdBook nav) lists this phase doc under closed phases.

---

## §9 — Pre-design tasks (PD1-PD6)

These run BEFORE C1 begins. Each has a binding output that the implementation depends on.

### PD1 — Bind the specific racing field(s)

**Method**: read `ggml-backend.cpp:2215-2350` (openmp parallel region) and list every shared-state field accessed inside the `#pragma omp parallel` block. Cross-reference with `ggml-cuda.cu` for each field's thread-safety properties. Optionally instrument: add `__sync_fetch_and_add` counters to each candidate field's access points, rebuild, run G3.a NP=8 once with default openmp dispatch — the field with the highest cross-thread access count under the failing slot is the likely race.

**Output**: list of specific shared fields with their access patterns. Becomes part of the implementation's design notes for C1-C5.

**Effort**: 0.5-1 day. Read-only unless instrumentation is needed; instrumentation needs a maintenance window.

**Status**: DONE (see §10 for results).

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

**Status**: DONE (see §10 for results).

### PD3 — Enumerate CPU splits in production LM and libmgpu graphs

**Method**: enable sched logging (search for an existing log knob in ggml-backend.cpp or add one temporarily), run one LM decode + one CLIP encode at production config, capture the split list. Tag each CPU split with: (a) what op, (b) what data flows into it, (c) what data flows out, (d) whether it has GPU dependencies upstream.

**Output**: list of CPU splits; decision binds on C6.

**Effort**: 0.5-1 day. Read-only if existing log knob; small instrumentation if not.

**Status**: DONE (see §10 for results).

### PD4 — Baseline perf measurement

**Method**: maintenance window, production stopped, GPUs locked. Run:
- `scripts/bench-multislot.sh` at N_PARALLEL ∈ {1, 2, 4, 8}
- `scripts/verify-multigpu-clip.sh LATENCY_N=10`
- `llama-bench` PP+TG at NP=1, NP=8 with split-mode graph
- Dispatch counter `total/multi_seq` at each NP

**Output**: baseline numbers; perf targets in §4.3 become bound to these numbers.

**Effort**: 1 maintenance window (~60-90 min).

**Status**: DONE (see §10 for results).

### PD5 — NCCL vs cudaMemcpyPeer profile on libmgpu reduce

**Method**: write a microbenchmark that:
- Allocates the same input shape as libmgpu's REDUCE (CLIP encoder layer output, F16, vision-token-count × hidden-dim)
- Runs N=1000 iterations of `cudaMemcpyPeer + element-wise add` (current path)
- Runs N=1000 iterations of `ncclAllReduce` (target path)
- Measures p50, p95, p99 latency for both
- Verifies numerical equivalence (byte-identical or within fp16 ULP)

**Output**: go/no-go on C9. Threshold: NCCL ≥10% faster → ship. Otherwise drop C9.

**Effort**: 1 day (need to write the microbenchmark, then run during a maintenance window).

**Status**: DONE (see §10 for results).

### PD6 — Multi-stream-per-device ILP profile on libmgpu matmul

**Method**: write a microbenchmark that:
- Allocates the same matmul shape as libmgpu uses for a CLIP layer
- Runs N=1000 iterations with 1 compute stream per device (current)
- Runs N=1000 iterations with 2 compute streams per device (split matmul row-wise, partial-reduce overlapped)
- Measures total wall time per iteration
- Verifies numerical equivalence

**Output**: go/no-go on C8. Threshold: ≥5% latency improvement → ship. Otherwise drop C8.

**Effort**: 1 day.

**Status**: DONE (see §10 for results).

### Pre-design summary

| Task | Effort actual | Maintenance window used? | Bound |
|---|---|---|---|
| PD1 | ~1 hour | none | named 4 race surfaces |
| PD2 | ~30 min | none | single-graph capture confirmed |
| PD3 | ~30 min | none | hoist-out pattern chosen |
| PD4 | ~30 min | YES (60 min window 12:19Z-12:35Z) | perf baseline bound |
| PD5 | ~60 min | YES (folded into PD5/PD6 window 12:34Z-12:54Z) | xeon threshold ~750 MB → quantizes to 1 GB bucket |
| PD6 | ~30 min | YES (same window) | xeon → SIZE_MAX |

**Actual pre-design**: ~4 hours total + two maintenance windows (~45 min each). All 6 tasks complete 2026-05-27.

---

## §10 — Pre-design results (PD1-PD6 all completed 2026-05-27)

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

### PD4 result — baseline perf (RUN_ID=20260527T121951)

Production build 1db6c2eb, GPU clocks locked 1455 MHz.

**LM perf** (`test-production-np-determinism.sh` with N_PREDICT=128, prompt ~210 tokens, multi-GPU `--split-mode graph --tensor-split 1,1`):

| NP | Aggregate TG | Per-slot TG | PP per-slot | Scaling |
|---|---|---|---|---|
| 1 | 17.9 t/s | 17.9 t/s | 241.5 t/s | 1.00× |
| 2 | 25.2 t/s | 12.6 t/s | 121.5 t/s | 1.41× |
| 4 | 28.9 t/s | 7.2 t/s | 101.5 t/s | 1.61× |
| 8 | 30.8 t/s | 3.8 t/s | 64.0 t/s | 1.72× |

vLLM reference at NP=8 on same hardware: 154.77 t/s aggregate → **5.0× headroom** to the dispatch-layer ceiling.

**Dispatch counter** (first 64 dispatches):

| NP | total | multi_seq | ratio |
|---|---|---|---|
| 1 | 64 | 0 | 0% |
| 2 | 64 | 64 | 100% |
| 4 | 64 | 55 | 86% |
| 8 | 64 | 38 | 59% |

Multi-seq atomic dispatch fires more at low NP (slot ordering tighter), less at high NP. The atomic 4D-uniform graph captures wins from this branch.

**CLIP encode** (`verify-multigpu-clip.sh LATENCY_N=10`):

- median **14450 ms** (Phase 46 closure was 14421 — within noise)
- p95 14746 ms
- warm-up sample 1: 15900 ms (graph-cache cold)
- steady-state variance 330 ms across samples 2-10
- 73.6% headroom under 1.3× CPU-vision ceiling (54600 ms)

**Bound perf targets for §4.3** updated below.

### PD5 result — NCCL vs cudaMemcpyPeer+add shape sweep

Two runs total: initial 6-point coverage (RUN_ID=20260527T123458) + refined sweep at 5 intermediate sizes to narrow crossover (RUN_ID=20260527T1234xx). Full result JSONs at `data/cuda-native-dispatch/predesign/pd5pd6-run-20260527T123458/`. Hardware: 2× Quadro RTX 6000, **NV2 bonded NVLink active (2 × 25.781 GB/s = ~50 GB/s)**, CUDA 13.2, NCCL 2.30.4.

| Size F16 | memcpy-peer+add p50 | NCCL ncclAllReduce p50 | Speedup | Notes |
|---|---|---|---|---|
| 2 MB | 0.089 ms | 0.129 ms | 0.69× | NCCL 31% slower (protocol overhead) |
| 5 MB | 0.177 ms | 0.221 ms | 0.80× | |
| 10 MB | 0.322 ms | 0.374 ms | 0.86× | |
| 20 MB | 0.601 ms | 0.662 ms | 0.91× | |
| 50 MB | 1.442 ms | 1.520 ms | 0.95× | |
| 100 MB | 2.848 ms | 2.895 ms | 0.98× | |
| **150 MB** | **4.236 ms** | **4.237 ms** | **1.00× — parity** | within ~0.02% |
| 200 MB | 5.645 ms | 5.535 ms | 1.02× | first crossover (within noise) |
| 300 MB | 8.435 ms | 8.091 ms | 1.04× | |
| 500 MB | 14.041 ms | 13.081 ms | 1.07× | |
| **750 MB** | **20.996 ms** | **19.143 ms** | **1.10× — first conservative point** | NCCL p95 < memcpy p50 |
| 1 GB | 28.055 ms | 25.023 ms | 1.12× | |
| 1.5 GB | 41.954 ms | 36.909 ms | 1.14× | |
| 2 GB | 56.014 ms | 48.627 ms | 1.15× | asymptote — NCCL approaches ~1.15-1.20× steady-state speedup as message size grows; diminishing returns past 1 GB |

**Equivalence**: byte-identical across all measured shapes (`n_diff=0`, max_abs_diff=0). Memcpy + element-wise-add and NCCL ncclAllReduce produce the SAME bits at the libmgpu reduce shape category.

**Decision**: not a single hardcoded threshold but the **calibration framework binding** (§3.0). The true conservative crossover on xeon is ~750 MB (p50 hits 1.10× AND NCCL p95 < memcpy p50). With the locked quantization bucket set `{0, 1MB, 10MB, 100MB, 1GB, SIZE_MAX}`, the calibration result on xeon for `GGML_CAL_REDUCE_CROSS_DEVICE` quantizes to the **1 GB bucket** (conservative: keep memcpy through 1 GB even though true crossover is 750 MB).

libmgpu's production per-CLIP-layer reduce shape is ~2-3 MB — well below threshold → memcpy path always fires in production. Calibration auto-discovers the right bucket on other hardware without code changes; an NVLink-rich box with smaller protocol overhead might pick the 100 MB or 10 MB bucket; a PCIe-only box might pick SIZE_MAX.

**Why NCCL is slow at small messages on NVLink**: NCCL's per-call protocol overhead is ~40-50 µs (group setup, channel/algo selection, ring topology). Below ~150 MB this overhead exceeds the bandwidth benefit. Direct `cudaMemcpyPeerAsync` is the hardware optimum at small sizes because the CUDA driver picks the NVLink path automatically with zero protocol cost.

### PD6 result — 1-stream vs 2-stream matmul

Hardware: single Quadro RTX 6000 (72 SMs, TU102). cuBLAS default math mode (TF32 disabled).

| Shape M,K,N | Output bytes | 1-stream p50 | 2-stream p50 | Speedup | Equiv |
|---|---|---|---|---|---|
| 1024, 1280, 2560 | 5 MB | 0.119 ms | 0.146 ms | 0.82× (−18%) | ULP-diff at K-tile boundary |
| 1024, 2560, 7680 | 15 MB | 0.542 ms | 0.535 ms | 1.01× (+1%) | byte-identical |
| 1024, 4096, 4096 | 8 MB | 0.408 ms | 0.419 ms | 0.97× (−3%) | byte-identical |

**Decision**: on this hardware at these shapes, 2-stream does not help — matmul kernels fully occupy SMs at these output sizes. Splitting adds launch overhead without extracting concurrency.

**Framework binding**: `GGML_CAL_MATMUL_STREAM_SPLIT` registers with the calibration framework (§3.0). On xeon, calibration is expected to return `SIZE_MAX` (1-stream always). On hardware with larger SM counts or smaller matmul tiles, the threshold may light up — the code is uniform either way. This is the canonical example of "ship the framework, let the hardware decide" vs the original "drop C8" plan.

### Status

All 6 PD tasks complete:

| PD | Status | Output |
|---|---|---|
| PD1 | ✅ done | 4 race surfaces named (cuda_graphs map, pools, cublas_handles, streams lazy-creates) |
| PD2 | ✅ done | Multi-device single-graph capture works in CUDA 13.2 |
| PD3 | ✅ done | CPU splits upstream-only; hoist-out pattern |
| PD4 | ✅ done | Baseline perf bound for §4.3 |
| PD5 | ✅ done | Crossover sweep complete; calibrated dispatch binding (xeon ≈ 750 MB) |
| PD6 | ✅ done | 1-stream wins on xeon; calibration framework returns SIZE_MAX |

---

## §11 — Formal specifications (Allium + TLA+) — scope LOCKED 2026-05-27

User directive 2026-05-27: "we need allium and TLA+ content for this document, add once we have data and integrated it."

PD4-PD6 are complete (§10); design parameters (calibration bucket set §3.0, NCCL/memcpy crossover ~750 MB on xeon, multi-device single-graph capture confirmed) are bound. The Allium/TLA+ specs listed below are scoped against those bound values; spec FILE content writes during pre-implementation (§11.4 step 1). Pattern follows `PHASE_NSTREAM_KV_PERF.md` §"P0.B" with 6 Allium + 4 TLA+ + 10 property tests.

### §11.1 — Allium specs (planned, content TBD)

Six specs covering the new dispatch + calibration framework's correctness invariants. Locations: `/home/dconnolly/yarn-agentic/specs/cuda-native-dispatch/`. Working titles:

1. **`single_threaded_dispatch.allium`** — ALL host-side CUDA driver state mutations happen on exactly one host thread per process lifetime. No concurrent entry into `ggml_backend_sched_compute_splits` or below. Binds C1.
2. **`cross_device_event_chain.allium`** — every cross-device data dependency in the captured graph has a `cudaEventRecord` on the producer + `cudaStreamWaitEvent` on the consumer. No data read on device D before its producing write is event-synchronized. Binds C4 + C7.
3. **`multi_device_graph_cache.allium`** — graph cache key `(topology_hash, device_layout, n_seq)` produces identical `cudaGraph_t` for matching keys. Cache hit replays bit-identically to capture. Eviction FIFO bounded by `GGML_CUDA_GRAPH_MAX`. Binds C5.
4. **`libmgpu_subgraph_capture.allium`** — libmgpu per-device subgraphs capture cleanly into the outer multi-device graph. Binds C7.
5. **`calibrated_dispatch_framework.allium`** — for every registered op `op_id`, every probe call returns a deterministic threshold for a fixed `(gpu_uuid, cuda_version, ggml_version)` tuple within a single process (modulo quantization). Cache load produces identical thresholds to a fresh probe (within quantization). Threshold is quantized to one of `{0, 1MB, 10MB, 100MB, 1GB, SIZE_MAX}` (LOCKED — see §3.0 rule 4). Binds C0.
6. **`calibrated_op_equivalence.allium`** — for every registered op, the default and alt strategies produce numerically equivalent output (byte-identical OR fp16/fp32 ULP-bound, per the op's contract). The dispatch decision (which strategy fires) may vary across runs/hardware via calibration, but the OUTPUT bits MUST NOT vary across strategy choices for the same input. Binds C8 + C9 + C10 + C11.

### §11.2 — TLA+ specs (planned, content TBD)

Four models covering dispatch safety + liveness + calibration. Locations: `/home/dconnolly/yarn-agentic/specs/cuda-native-dispatch/`.

1. **`CUDANativeDispatch.tla`** — single-threaded `compute_splits` as a state machine `{IDLE, ENQUEUING, CAPTURING, LAUNCHED, COMPLETE}`. Verifies absence of host-side races, event-record/wait ordering, every split's eval completes before the graph end.
2. **`CUDAGraphCacheConsistency.tla`** — multi-device graph cache. Insertions, evictions, lookups. Verifies cache key uniqueness (no hash collisions) and that evicted-then-re-captured graphs are bit-identical.
3. **`CalibrationFramework.tla`** — calibration as a state machine `{INIT, PROBING, CACHED, READY}`. Probe completes for each registered op; threshold is quantized; cache persists; cache hit on second context produces identical thresholds. Determinism: subsequent dispatch decisions are a pure function of `(op_id, payload_bytes, threshold[op_id])`.
4. **`CalibratedOpEquivalence.tla`** — for each calibrated op (REDUCE, MATMUL, PEER_COPY, GRAPH_CAPTURE), models the default and alt strategies as two computation chains; verifies they produce equivalent output (under each op's documented ULP/byte criterion). Replaces the old `CrossDeviceReduceOrdering.tla` plan with a generic equivalence model that covers all calibrated ops.

### §11.3 — Property tests + trace-harness extensions (planned)

Nine property tests in `tests/spec/` bound by the Allium + TLA+ specs:

**Dispatch tests (bind §11.1.1-4 / §11.2.1-2):**
1. `test-single-threaded-dispatch.cpp` — counts threads entering `compute_splits` over N graph computes; asserts always == 1.
2. `test-cross-device-event-chain.cpp` — instruments event-record/event-wait pairs; asserts every consumer wait has a matching producer record before it.
3. `test-multi-device-graph-cache.cpp` — runs N captures of the same topology; asserts cache hit rate steady-state, bit-identical replay output.
4. `test-libmgpu-subgraph-capture.cpp` — captures libmgpu's per-device subgraphs; asserts byte-identity to the pre-phase non-captured output.

**Calibration framework tests (bind §11.1.5 / §11.2.3):**
5. `test-cuda-calibration-framework.cpp` — calibrate twice in same process, assert identical thresholds; write cache, reload from cache, assert identical thresholds; force-recalibrate, assert quantized thresholds match within bucket tolerance.

**Per-op equivalence tests (bind §11.1.6 / §11.2.4):**
6. `test-calibration-equivalence-reduce.cpp` — memcpy-peer+add vs ncclAllReduce at threshold/2 (memcpy fires), threshold*2 (NCCL fires), and threshold (boundary). Both byte-identical.
7. `test-calibration-equivalence-matmul.cpp` — 1-stream vs 2-stream cuBLAS hgemm; fp16 ULP-bound equivalence (cublas tile config can shift K-reduction tree shape, so byte-identity not guaranteed).
8. `test-calibration-equivalence-peer-copy.cpp` — direct vs staged-via-pinned cudaMemcpyPeer; byte-identical.
9. `test-calibration-equivalence-graph.cpp` — captured vs eager-launch of the same kernel sequence; byte-identical.

**Trace-harness extension**: `cudaGraphTraceCapture` + `cudaGraphTraceLaunch` hooks (CUDA 12+ `cudaGraphAddEventRecordNode` introspection) → JSON trace of captured nodes for property tests to parse. Also adds per-op-dispatch counters (which strategy fired) so tests #6-9 can assert the right path was taken.

### §11.4 — Integration order

PD4/PD5/PD6 are complete (see §10). The remaining order:

1. **Pre-commit work**: write all six Allium specs + four TLA+ models at `specs/cuda-native-dispatch/`. Each spec lands as a placeholder file in tree BEFORE its corresponding implementation commit. This makes the spec the binding contract that implementation must satisfy.
2. **Commit pairing**: each implementation commit (C0-C11) ships with its bound spec(s) updated to "live" status + the property test that enforces them:
   - C0 ships `calibrated_dispatch_framework.allium` (live) + `CalibrationFramework.tla` (model-checked in CI) + `test-cuda-calibration-framework.cpp`
   - C1 ships `single_threaded_dispatch.allium` + `CUDANativeDispatch.tla` + `test-single-threaded-dispatch.cpp`
   - C4 ships `cross_device_event_chain.allium` + `test-cross-device-event-chain.cpp`
   - C5 ships `multi_device_graph_cache.allium` + `CUDAGraphCacheConsistency.tla` + `test-multi-device-graph-cache.cpp`
   - C7 ships `libmgpu_subgraph_capture.allium` + `test-libmgpu-subgraph-capture.cpp`
   - C8-C11 each ship one `calibration-equivalence-*.cpp` test + the shared `calibrated_op_equivalence.allium` (incremental updates) + `CalibratedOpEquivalence.tla` (model-checked once after all ops register)
3. **CI gate**: TLA+ model-checking runs via the existing `.github/workflows/spec-tla-gate.yml` workflow. The gate must pass for each commit, not just the final one.
4. **Verification commit (C14)**: re-runs the full determinism battery + perf comparison + calibration cache sanity (cache hit reproduces same thresholds; force-recalibrate produces thresholds within quantization tolerance of cache).
