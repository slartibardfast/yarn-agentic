# PHASE_NSTREAM_KV_PERF — recover the -6.2 % TG-NP=8 regression

**Branch**: `production/2026-q2-next` (off submodule HEAD `16b608d1`).
**Predecessor**: `PHASE_NSTREAM_KV_4D.md` — closed 2026-05-20. Bug C structurally closed; decode-side prefill gate removed; 6 correctness gates green; perf regression carried over.
**Status**: Open; scope and starting hypothesis below. No design decisions locked yet.

## Why this phase exists

PHASE_NSTREAM_KV_4D's G3.h gate failed by **-6.2 %** on `llama-batched-bench` TG NP=8 (26.00 t/s vs 27.73 t/s baseline), outside the ±1 % bound. User selected "Override locked policy — merge with -6.2 % regression" on the merge of N2+N3, with the understanding that the perf work would land in a follow-on phase.

The same overhead also makes `test-pp-serialization.sh`'s wall time recover only ~1.3 % vs pre-port (15.7 s vs 15.9 s) — far below the ~38 % the cached plan had estimated for TG-overlap recovery.

## Root cause (already diagnosed at G3.h close)

Graph reuse is disabled at `n_stream > 1` in `llama-context::can_reuse_graph`:

```cpp
// PHASE_NSTREAM_KV_4D N2.b: read views in llm_build_kqv bake a
// stream_id-derived offset; reusing a graph across stream
// boundaries would read from the wrong stream's slice. Disable
// reuse under n_stream > 1 until a stream-aware reuse check
// lands.
if (transformer_kv.n_stream > 1) { g_can_reuse_last_miss_reason = 6;  return false; }
```

Each per-stream sub-batch rebuilds the graph (~2–3 ms each). At NP=8 steady-state TG, that's 8 rebuilds per tick. The bench's 2112 single-token sub-calls amortise to ≈ 5–6 s of pure rebuild cost in an 80 s run — the source of the 6 % delta.

The same overhead is paid by the production server's steady-state TG at NP=8, just less visible (no single benchmark number ties to it).

## Starting hypothesis — per-stream graph cache

Cache one `prev->graph` per `stream_id` instead of one for the whole context. At NP=8 that's 8 cached graphs; each per-stream sub-batch reuses its own stream's graph.

Design sketch (not locked):

- Replace `lctx.prev` (single `unique_ptr<Prev>`) with `std::array<unique_ptr<Prev>, MAX_N_STREAM>` or a small map keyed by stream_id.
- `can_reuse_graph`: derive `stream_id` from `u_batch.seq_id[0][0]`, look up `prev[stream_id]`, run the existing reuse-check predicates against that entry.
- `cache_prev` write: after a successful build at stream_id S, store the graph in `prev[S]`.
- Tear-down: clear all entries on context reset.

Open questions (don't pre-decide):

- Does the existing `cache_prev_eligible` check (single-token or fused MTP) need refining at multi-stream?
- Are the per-stream graphs structurally identical apart from the baked offset, or do they differ in node count? (If identical, a single graph with a runtime-patchable offset is even cheaper. Worth measuring before committing to the per-stream cache.)
- Interaction with `update_cache_copies` / graph-exec-update path?

## Adjacent open follow-ups carried from PHASE_NSTREAM_KV_4D

These are not part of the perf recovery itself but are scope-adjacent and need lifting off `GGML_ASSERT(n_stream == 1)` for full multi-slot coverage:

- `build_k_shift` per-stream — needed for `ctx_shift` at multi-slot.
- `build_defrag` per-stream — needed for cache compaction under multi-slot.
- `v_trans` non-FA V path per-stream — production runs FA-on, so currently a guard; lift when adding non-FA paths.
- MLA (DeepSeek) — out of scope.

## Binding gates (proposed; not locked)

- **GP3.a** — `llama-batched-bench` TG NP=8 on the prior baseline GGUF: within **±1 %** of 27.73 t/s. Hard binding.
- **GP3.b** — `scripts/test-pp-serialization.sh` wall ≤ 11 s (recovers the bulk of the TG-overlap window).
- **GP3.c** — `scripts/test-production-np-determinism.sh` byte-identity preserved, single + multi-GPU. Cache change must not perturb determinism.
- **GP3.d** — `bin/test-dflash-np-multislot` GREEN unchanged.
- **GP3.e** — `scripts/r5-probe-c4.sh ITERS=20` 0/20, single + multi-GPU. Bug C absence preserved.

If a measurement reveals that the per-stream cache hypothesis doesn't recover the gap (e.g. graphs are rebuilt anyway due to a different invalidation), the diagnostic surfaces honestly per `feedback_negative_results_land_cheap_when_honest` — no narrative cover.

## What's NOT in scope

- New correctness gates (PHASE_NSTREAM_KV_4D's are sufficient — this phase reuses them as preservation checks).
- N-stream layout changes (the 4D port itself is closed).
- DFlash perf (separate workstream).
- PSKV singlewarp FA kernel (separate ralph perf loop).

## Token estimate

Per CLAUDE.md §8: decision-locking the per-stream cache design first (~20 k), implementation of cache + reuse-check refactor (~40 k), gate verification rounds (~30 k). Total ≈ **90 k tokens** if the per-stream cache hypothesis holds. If measurement says it doesn't, the phase shifts into diagnosis-mode — budget for 2–3 rounds of nsys / ncu (~30 k each).
