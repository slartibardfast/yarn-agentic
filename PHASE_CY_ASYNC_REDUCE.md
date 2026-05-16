# PHASE CY-Async-Reduce — Option B implementation plan

**Status**: planning. Pre-implementation. Triggered by user direction "B will be a great success, worthy of a full allium / tla+ spec and plan."

**Branch**: `production/2026-q2-next` (extending Phase CY.F.16 Option A which is already shipped).

## Predecessor (Phase CY.F.16 Option A — shipped)

Arch-init force `cparams.reduce_type = GGML_TYPE_F32` for Qwen 3.5 / 3.6 split-mode-graph. Closes NP={1,4,8} byte-determinism. Cost: ~2× cross-device reduce bandwidth at prefill.

## This phase (Option B)

Replace the synchronous F32 cross-device reduce with an **async F32 reduce on a dedicated comm CUDA stream + explicit event-based consumer wait**. Achieves the same determinism, hides the bandwidth cost behind compute via stream overlap.

Per `specs/cy-async-reduce/DESIGN.md` §Architecture, the per-layer transfer of 1.96 MiB at NP=8 prefill takes ~160 µs at PCIe 3.0 x16 peer bandwidth. Per-layer compute is 10–20 ms. **Transfer fits in <2% of compute window** — easily hidden via stream overlap.

## Specs (committed, locked before implementation)

- [ ] **Allium spec**: `specs/cy-async-reduce/cy-async-reduce.allium`. Defines 12 invariants (correctness, ordering, isolation, liveness, perf, fallback). Locks the public contract.
- [ ] **TLA+ spec**: `specs/cy-async-reduce/CYAsyncReduce.tla`. Models 2-device per-layer compute + comm streams with CUDA events. Verifies safety (no read-before-write, event ordering), liveness (all layers complete, all events signal), deadlock-freedom, and overlap-possibility.
- [ ] **Design doc**: `specs/cy-async-reduce/DESIGN.md`. Architecture diagram, perf budget, fallback path, open questions.

## Tasks

### T1 — Spec lockdown

- [ ] Author `specs/cy-async-reduce/cy-async-reduce.allium` (DONE — initial draft).
- [ ] Author `specs/cy-async-reduce/CYAsyncReduce.tla` (DONE — initial draft).
- [ ] Run `allium check specs/cy-async-reduce/`.
- [ ] Run TLC on `CYAsyncReduce.tla` with N_LAYERS=4 (small but exhaustive). Verify all properties (SafetyConsumeAfterCompute, SafetyEventOrdering, LivenessAllLayersComplete, LivenessAllReducesSignal, DeadlockFreedom, OverlapPossible).
- [ ] Lock invariants. Any later code that violates an invariant requires the spec to change FIRST in a separate commit.

**Closure**: Allium drift checks GREEN; TLC verifies all properties on the 4-layer model with no violations.

### T2 — API surface

- [ ] Decide: extend `ggml_reduce(ctx, srcs, n, OP_ADD)` with a new mode flag `ASYNC`, OR add a new op `ggml_reduce_async(ctx, srcs, n, OP_ADD, comm_stream_id)`. **Recommended**: new op for clarity; old op stays as sync fallback.
- [ ] Add `ggml/include/ggml.h` declarations:
  - `enum ggml_reduce_mode { GGML_REDUCE_SYNC, GGML_REDUCE_ASYNC };`
  - `ggml_tensor * ggml_reduce_async(ctx, srcs, n, op, mode);`
- [ ] Add CUDA backend type `ggml_cuda_comm_context_t` holding per-device comm streams + events.

**Closure**: header compiles; new op enumerable.

### T3 — CUDA backend

- [ ] Allocate per-device comm stream at backend init (`ggml_backend_cuda_init`).
- [ ] Implement `ggml_cuda_op_reduce_async` in `ggml/src/ggml-cuda/reduce.cu`:
  - Wait on `evt_input_ready[d]` for each device
  - Issue `cudaMemcpyPeerAsync` on comm stream
  - Launch F32 add kernel on comm stream
  - Record `evt_reduce_done[d]` on comm stream
- [ ] At scheduler level: `cudaStreamWaitEvent(compute_stream, evt_reduce_done)` inserted at each consumer.

**Closure**: unit test `tests/test-ggml-reduce-async-correctness.cpp` PASS — output bit-identical to sync F32 reduce on random F32 inputs at varying shapes (M ∈ {1, 4, 8, 12, 16, 32, 96}).

### T4 — Build-graph wiring (FFN)

- [ ] Switch `ggml_reduce` to `ggml_reduce_async` in `src/llama-build-context.cpp:814` (the FFN cross-device sum) for split_mode=GRAPH paths.
- [ ] Validate: `test-cy-seq-id-batched.cpp` PASS at batched=8 with async reduce.

**Closure**: serial vs batched logits bit-identical.

### T5 — Build-graph wiring (DeltaNet)

- [ ] Switch `ggml_reduce` to `ggml_reduce_async` in `src/llama-delta-net.cpp:592` (the DeltaNet output sum).
- [ ] Validate.

**Closure**: V4 production NP-determinism harness PASS at NP={1, 2, 4, 8} for 10 consecutive runs.

### T6 — Overlap verification

- [ ] nsys-capture a prefill at NP=8.
- [ ] Verify: comm-stream activity for layer N overlaps compute-stream activity for layer N+1 in the timeline.
- [ ] Measure: per-layer wall-clock from "FFN done on dev 0" to "reduce result read by RMSNorm" — should be dominated by compute time, not transfer.

**Closure**: nsys timeline shows overlap; wall-clock ratio (transfer/total) <5%.

### T7 — Perf measurement

- [ ] Benchmark prefill PP at NP=8 with:
  - Option A (current production, force F32 sync reduce)
  - Option B (async F32 reduce with overlap)
  - Pre-Option-A (F16-cast, batch-shape non-deterministic — historical baseline)
- [ ] Target: Option B PP ≥ Option A PP, and Option B PP within 5% of pre-Option-A F16-cast PP.

**Closure**: tokens/sec measurement table; Option B matches or beats Option A; documented vs F16-cast baseline.

### T8 — Default flip

- [ ] Change default behavior: when split_mode=GRAPH and >1 device, use async F32 reduce.
- [ ] Sync F32 reduce remains as fallback for single-device or when comm stream allocation fails.
- [ ] `--graph-reduce-async=0` env override to revert to sync if needed.

**Closure**: production server defaults to async; runtime log confirms `reduce_mode = async-f32`.

### T9 — Decode perf check

- [ ] Benchmark decode tokens/sec at NP=1 and NP=8 with Option B vs Option A.
- [ ] Expected: unchanged (cast didn't fire at decode at ne[1] ≤ 8 either way).

**Closure**: decode tokens/sec parity.

### T10 — Option A retirement

- [ ] Remove the arch-init force for QWEN35 / QWEN35MOE in `src/llama.cpp` (added in Option A).
- [ ] Replace with a runtime check: if split_mode=GRAPH and reduce_mode=async, no warning; if reduce_mode=sync, emit one-line info "using sync F32 reduce — consider async for perf".
- [ ] Keep `--graph-reduce-type` user override available.

**Closure**: Option A code removed; tests still GREEN; production NP-determinism maintained.

## Composite closure (ship gate)

- [ ] Allium drift checks GREEN.
- [ ] TLC verifies all TLA+ properties.
- [ ] T1–T10 GREEN.
- [ ] V4 NP-determinism harness PASS at NP={1, 2, 4, 8} for 10 consecutive runs.
- [ ] Perf: Option B ≥ Option A PP, within 5% of pre-Option-A F16-cast PP.
- [ ] NP=2 residual (CY.F.17) explicitly checked under Option B — may or may not be fixed; if not, separate workstream.

## Risks

| Risk | Mitigation |
|---|---|
| `cudaMemcpyPeerAsync` doesn't actually overlap with kernel launches on TU102 | T6 nsys probe confirms or refutes; if no overlap, Option B degrades to Option A bandwidth cost (still byte-deterministic, just no perf win). Decision point: keep Option A, abandon B. |
| Graph reuse (`cparams.graph_reuse`) breaks with new stream/event semantics | T2 API surface includes explicit reset; graph-reuse-aware tests in T4/T5. |
| Scheduler deadlocks due to subtle event-dependency cycle | TLA+ DeadlockFreedom property verified in T1; runtime cudaStreamWaitEvent insertion goes through one canonical path. |
| Async path adds latency at decode (event signal overhead) | T9 measures; if regression, gate async-mode on `ne[1] > threshold` (analog of original 32 cutoff but only for the SCHEDULING choice, not for precision). |
| TLC state-space explosion at N_LAYERS > 4 | Run incremental TLC: N_LAYERS=2, 3, 4. If small models verify, larger follow by induction over the layer structure. |

## Open questions (from DESIGN.md, resurfaced for spec lockdown)

1. **Does ggml's CUDA backend already use multiple streams?** If yes, hook into existing; if no, add stream class. (Investigate before T2.)
2. **Is `cudaMemcpyPeerAsync` overlap-able with compute on TU102?** Empirical question — answer via nsys microbenchmark BEFORE T3 to de-risk.
3. **Should `evt_input_ready` and `evt_reduce_done` be merged?** TLA+ models them separately for clarity. Implementation may merge if no consumer needs the intermediate signal.
4. **How does this interact with graph reuse?** Reused graphs cache kernel launch lists. New stream/event semantics may require launch list re-recording — investigate in T2.

## References

- Phase CY.F.16 Option A: `MEMORY.md` 2026-05-16 entry; `src/llama.cpp:7169` arch-init force.
- TML batch-invariance pattern: Thinking Machines Lab blog (already cited in Phase CX work).
- NVIDIA CUDA Programming Guide §3.2.6 Streams + §3.2.8 Events.
- ggml backend stream infrastructure: `ggml/src/ggml-backend.cpp`, `ggml/src/ggml-cuda.cu`.
