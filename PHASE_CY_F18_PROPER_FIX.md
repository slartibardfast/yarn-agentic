# Phase CY.F.18 — proper fix for scheduler needs_sync lifecycle race

**Branch**: `production/2026-q2-next`

## State at this checkpoint

NP=2 cross-NP byte-determinism is **achieved** with two env-gates:

```bash
GGML_CUDA_MMQ_DISABLE_STREAM_K=1   # CY.F.17 — MMQ stream_K shape-dep
GGML_SCHED_FORCE_SYNC_INPUTS=1     # CY.F.18 — scheduler sync lifecycle
```

Both defaults are ON in `scripts/test-production-np-determinism.sh`.

Test result (test-cy-np2-multi-step-decode, 5 runs × 20 decode steps):
- slot 0 matches NP=1: 5/5 ✓
- slot 1 matches NP=1: 5/5 ✓
- slot 0 == slot 1: 5/5 ✓

## CY.F.17 — root cause and fix (DONE)

**Root cause**: MMQ's stream_K dispatch (`ggml/src/ggml-cuda/mmq.cuh`) does
cooperative split-K across `nsm` CTAs + a fixup kernel to combine partial
results. The fixup accumulation order depends on `ntiles_x = ceil(M / mmq_x)`.
Different M values within the prefill regime (e.g., NP=1 prefill at M=215 vs
NP=2 batched prefill at M=430) produce DIFFERENT float outputs.

**Fix**: `GGML_CUDA_MMQ_DISABLE_STREAM_K=1` falls through to vanilla blocked
GEMM — one CTA per output tile, no cross-CTA fixup, deterministic across M.

**Cost**: ~5-15% prefill throughput on long-context (decode unaffected).

Files touched: `ik_llama.cpp/ggml/src/ggml-cuda/mmq.cuh`,
`ik_llama.cpp/tests/dflash-speculative/test-mmq-q4-0-ar16-shape-invariance-prod-dim.cpp`.

## CY.F.18 — root cause (found) and stopgap fix (DONE)

**Root cause**: `ggml_backend_sched`'s `needs_sync[backend_id]` flag is cleared
to `false` after one synchronization (assumes "sync covers all remaining reads
from this backend in this pass"). But the reduce op's `k_reduce_add_T` kernel
does direct peer P2P writes from one device to another's memory. These writes
are queued on the source device's stream — `ggml_backend_cuda_synchronize` on
the destination only calls `cudaStreamSynchronize(own_stream)`, which does
NOT drain incoming peer writes from the source device's stream.

Sequence that races:
1. Reduce broadcasts slot 1's data to device 0's memory via peer write
   (initiated by device 1's stream, queued there).
2. Scheduler clears `needs_sync[0] = false` after device 0 syncs its own stream.
3. Next split reads slot 1's region in device 0's memory.
4. `needs_sync[0] = false` ⇒ no sync triggered.
5. The peer write from device 1's stream hasn't fully landed ⇒ stale read.

Slot 0 is deterministic because it's in device 0's local chunk of the reduce
(written by device 0's OWN kernel — properly stream-ordered). Only the
peer-written region (slot 1) races.

**Stopgap fix**: `GGML_SCHED_FORCE_SYNC_INPUTS=1` keeps `needs_sync = true`
after sync, forcing every input read to re-sync.

**Cost**: heavy. Every cross-backend tensor read pays a `cudaStreamSynchronize`
even when redundant. Decode perf cost estimate ~5-15% on multi-GPU configs;
not yet characterized.

Files touched:
- `ik_llama.cpp/ggml/src/ggml-backend.cpp` (3 spots — see below)
- `ik_llama.cpp/ggml/src/ggml-cuda.cu` (diagnostic env, can be removed)

## Investigation history (eliminated suspects)

These were probed and ruled out:
- Reduce P2P kernel path (forced fallback `cudaMemcpyPeerAsync`, still raced).
- `graph_reuse` (disabled, still raced).
- Prefill→decode async leak (`llama_synchronize` between them, still raced).
- Post-reduce `needs_sync` clearing (kept `true` for reduce sources, still raced).
- `__threadfence_system()` in `k_reduce_add_T` (still raced).
- `cudaDeviceSynchronize` in `ggml_backend_cuda_synchronize` (still raced).

The race is at the **scheduler-level needs_sync flag**, not the kernel-level
sync implementation.

## Why the stopgap is wrong as a permanent fix

`GGML_SCHED_FORCE_SYNC_INPUTS=1` re-syncs every input read. For a 64-layer
decode at NP=2, each layer has multiple cross-backend splits. The unnecessary
syncs serialize device streams that could otherwise overlap, costing throughput.

The correct fix is **precise tracking**: mark `needs_sync[X] = true` after
ANY async peer write targeting backend X's memory, regardless of which
backend initiated it. The current code only sets `needs_sync[X] = true`
when backend X itself completes a split with inputs.

## Proper fix — design space

### Option A — Mark needs_sync on the reduce broadcast path

In `ggml-cuda/reduce.cu`, after the reduce kernel launches that do peer
broadcasts, set a flag on each TARGET backend (not just the source) saying
"your memory was just async-written from another device".

Concretely: after `cudaMemcpyPeerAsync` or peer kernel writes in `reduce.cu`,
trigger `sched->needs_sync[target_device] = true` for each device that
received a peer write.

Challenge: the reduce kernel is BELOW the scheduler. It doesn't have a
handle to the scheduler. Would need a callback/hook.

### Option B — Treat reduce src tensors as cross-device inputs

The scheduler already has logic for "this input came from a different
backend, need to sync." The reduce result is technically MULTIPLE outputs
(one per device, all containing the same reduced data). When the next op
reads one of these per-device copies, the scheduler should recognize it
came from cross-device communication and sync accordingly.

This requires the scheduler to track that the reduce broadcast touched
multiple backends, not just one.

### Option C — Drain peer writes via cudaDeviceSynchronize on the destination

Replace `cudaStreamSynchronize(own_stream)` with `cudaDeviceSynchronize()`
in `ggml_backend_cuda_synchronize` when the previous op was a reduce
broadcast. The cudaDeviceSynchronize drains ALL streams on the destination
device, including incoming peer write receivers.

Heavier than per-stream sync but lighter than force-sync-all-inputs because
it only fires when a reduce broadcast happened.

Caveat: we tested `cudaDeviceSynchronize` in the backend_synchronize and
it didn't fix the race. That suggests `needs_sync` is being cleared before
the sync is even called, OR the sync timing is different.

### Option D — Add explicit cross-device sync barrier after reduce

In the build context where `ggml_reduce` is built, immediately insert a
synthetic "cross-device barrier" node that's wired through the scheduler
to force sync on all participating backends.

Cleanest semantically. Requires a new op type or special-case in the scheduler.

### Option E — Replace kernel-direct peer writes with explicit memcpy

Refactor `k_reduce_add_T` so it does the broadcast via `cudaMemcpyPeerAsync`
calls (the fallback path's pattern) instead of kernel-side peer stores.
The ring path (`dst->ne[1] >= 32`) already does this. Make the small-shape
path follow the same pattern.

This is the most invasive but fundamentally changes the sync semantics so
peer writes ARE stream-ordered.

## Recommended approach

**Option B (treat reduce src as cross-device input)** is the cleanest design.

The scheduler already understands "input from backend Y is being read on
backend X — sync Y first." After a reduce, the output is logically on N
backends. When backend X subsequently reads its per-device copy, the
scheduler should treat it as "data was just touched by N-1 other backends,
sync all of them."

Implementation sketch:
1. Add a per-tensor `last_writer_set` (a bitmask of backends that wrote to it).
2. On reduce: set last_writer_set[reduce_output] = all participating backends.
3. On any read of that tensor: sync all backends in last_writer_set, not just
   the "primary" backend.
4. Decay: after the sync, clear last_writer_set entries that have been
   synced.

This is targeted: only the reduce outputs (which have peer writes) trigger
the broader sync. Other tensors with single-writer backends use the existing
needs_sync mechanism.

## Tasks for the proper-fix session

1. **Audit** `ggml-backend.cpp` for all places `needs_sync` is consulted and
   set. Map the lifecycle.
2. **Audit** `ggml-cuda/reduce.cu` for which paths do peer writes (P2P kernel
   path, ring path, fallback path) — each has different sync semantics.
3. **Design** the per-tensor last_writer_set tracking (Option B). Write up
   the data structure + lifecycle in this file before coding.
4. **Implement** the fix in a new branch. Test with
   `test-cy-f18-layer-bisect` + `test-cy-np2-multi-step-decode`.
5. **Measure perf cost**: compare decode throughput with stopgap (FORCE_SYNC)
   vs proper fix vs no fix (race-prone) using `llama-bench`.
6. **Remove stopgap env-gates**: if proper fix lands, the env-gates become
   redundant. Remove from the codebase + production scripts.

## Test infrastructure already in place

- `tests/dflash-speculative/test-cy-f18-layer-bisect.cpp` — captures
  decode-step-1 per-slot logits + per-layer residuals; race-localization probe.
  Env: `LLAMA_TEST_NO_EXTRACT=1` (pure decode), `LLAMA_TEST_NO_GRAPH_REUSE=1`,
  `LLAMA_TEST_BISECT_LAYER=N`.
- `tests/dflash-speculative/test-cy-np2-multi-step-decode.cpp` — full
  multi-step decode test, NP=1 baseline + NP=2 comparison. Env:
  `LLAMA_TEST_RACE_LOCALIZE=1` for cross-run logit comparison,
  `LLAMA_TEST_SERIAL_PREFILL=1`, etc.
- `tests/dflash-speculative/test-mmq-q4-0-ar16-shape-invariance-prod-dim.cpp`
  — MMQ shape-invariance, extended to M ∈ {1..1720}.

## Critical files

**Modified for stopgap fix**:
- `ik_llama.cpp/ggml/src/ggml-backend.cpp` — lines ~1976 (`k_set_sync` in
  copy_inputs), ~2258 & ~2333 (post-reduce needs_sync clearing).
- `ik_llama.cpp/ggml/src/ggml-cuda/mmq.cuh` — lines ~4389 & ~4439
  (stream_K dispatch env-gate).
- `ik_llama.cpp/ggml/src/ggml-cuda.cu` — line ~4470 (backend_synchronize
  device-sync probe).
- `ik_llama.cpp/ggml/src/ggml-cuda/reduce.cu` — lines ~438 (REDUCE_NO_P2P
  env-gate).

**Read-only references**:
- `ik_llama.cpp/ggml/src/ggml-backend.cpp` lines 1948-2007
  (`ggml_backend_sched_copy_inputs` — the sync entry point).
- `ik_llama.cpp/ggml/src/ggml-backend.cpp` lines 2174-2275 (OMP scheduler).
- `ik_llama.cpp/ggml/src/ggml-backend.cpp` lines 2278-2358 (std::barrier
  scheduler).

## Estimated token cost

Rough estimate per `CLAUDE.md §8` (estimate in tokens, not days):

| Step | Tokens |
|---|---|
| 1. Audit scheduler + reduce sync paths | 15-25k |
| 2. Design last_writer_set tracking | 20-30k |
| 3. Implement + iterate | 30-50k |
| 4. Test + measure perf | 15-25k |
| 5. Remove stopgap env-gates | 5-10k |
| **Total** | **~85-140k** |

Single-session work if scope holds.

## Open questions

1. Does the race also fire at NP=4 and NP=8 cross-NP? Or only at NP=2?
   (CY.F.16 Option A note said NP=4/8 are intra-run deterministic but
   never tested cross-NP byte-identity against NP=1.)
2. Is the `__threadfence_system` failure at the reduce kernel a Turing-specific
   limitation, or a fundamental P2P-write semantics gap?
3. Does the proper fix need to handle non-reduce peer writes too? Audit
   `ggml-cuda` for other places that initiate peer P2P writes.
4. Perf cost: is the stopgap actually 5-15%, or wider? Run benchmark.

## Pickup discipline

- This file edits commit + push immediately (per `CLAUDE.md §5`).
- Read `MEMORY.md` 2026-05-16 entries (3 of them — Phase A.16 / CY.F.17 /
  CY.F.18) for full context.
- Test the stopgap is still working before changing anything: run
  `test-cy-np2-multi-step-decode` with both env-gates and verify 5/5 pass.
