# PHASE_NSTREAM_KV_4D — full 4D per-stream KV port

**Branch**: `production/2026-q2-next`
**Predecessors**: `PHASE_NSTREAM_KV.md` (a)–(g) for design history. **(g) is the load-bearing finding**: there is no kernel-level patch for the mixed-batch GEMM-vs-GEMV divergence.
**Status**: Specced. Foundation slice landed (`n_stream`, `v_heads` fields on the struct, init only, no allocator use). **Spec layer S1–S5 landed (2026-05-20) — see [S-phase closure status](#s-phase-closure-status-2026-05-20).**

## S-phase closure status (2026-05-20)

Pre-N1 spec layer per `/home/llm/.claude/plans/cached-crunching-tiger.md`:

- **S1.a — `specs/scheduler/batch_composition.allium`**: ✓ landed. Contracts: `PrefillSerialisationGate`, `DecodeHoldGate`, `BatchCompositionInvariant`, `MixedBatchProhibition`, `AtMostOnePrefillSlotPerBatch`. `allium check` clean.
- **S1.b — `specs/kv-cache/n_stream_layer.allium`**: ✓ landed. Contracts: `PerStreamAllocator`, `MaskPerStream`, `PerStreamDispatch`, `BugCAbsenceByConstruction`, `DFlashCompatibility`. `allium check` clean.
- **S1.c — `mtp_fused_draft.allium` tend**: ✓ landed. Added `FusedDraftRoundsRunOnPureDecodeBatches` + `FusedDraftRespectsStreamPartition` cross-cutting invariants linking S1.a / S1.b.
- **S2.a — `specs/multislot/BatchComposition.tla`**: ✓ landed. State machine + WF fairness. SANY-clean.
- **S2.b — `specs/multislot/StreamIsolation.tla`**: ✓ landed. SANY-clean.
- **S3 — TLC model-check**: ✓ landed. Configs `BatchCompositionMC.cfg` (gate ON, 541 distinct states, PASS), `BatchCompositionMC_no_gate.cfg` (gate OFF, `BatchCompositionInvariant` violated — spec binds), `StreamIsolationMC.cfg` (PerStreamDispatch ON, 78 distinct states, PASS), `StreamIsolationMC_legacy.cfg` (PerStreamDispatch OFF, `StreamPartition` violated — spec binds).
- **S4 — Property tests via `allium plan` obligations**: ✓ landed. `tests/spec/test-batch-composition-gates.cpp` (1296 slot-config sweep, PASS on HEAD). `tests/spec/test-n-stream-kv-layout.cpp` (foundation PASS; `KVTensorIsFourD` FAIL on HEAD with `k_l[3]->ne[3]=1 vs n_stream=2` — the binding RED test for N1).
- **S5 — NDJSON trace harness + live validation**: ✓ landed. `examples/server/server-trace-ndjson.h` emit helper, single emit site at the dispatch boundary, gated on `LLAMA_TRACE_NDJSON_DIR`. Validator at `scripts/validate-batch-composition-trace.py`. Live-verified on Qwen3.5-0.8B BF16: 2 concurrent completion requests, 2-record trace, validator PASS — spec models HEAD reality.

N1+N2+N3 work proceeds against this binding spec set: `test-n-stream-kv-layout` flips from RED→GREEN as the 4D port lands, and the headline closure gate (G3.a–G3.h) remains the bundle binding.

## N1 closure status (2026-05-20)

**N1 — 4D structural foundation: ✓ LANDED** (submodule commits 52d845e9 → c1beb104 → 38ea4127). What landed:

- `struct llama_kv_cache` extended with `n_stream`, `kv_size_per_stream`, `v_heads` fields.
- `llama_kv_cache_init`: K/V allocated as 4D `[head_dim, n_head_kv, kv_size_per_stream, n_stream]`. `kv_size` rounded UP to a multiple of `n_stream`.
- `_clear / _seq_rm / _seq_keep / _seq_add` carry per-stream `v_heads` tracking (benign bookkeeping; not consumed by the legacy find_slot path).
- `tests/spec/test-n-stream-kv-layout.cpp` binds GREEN at `n_parallel ∈ {1, 2}`.

**The axis-order load-bearing decision.** The chosen 4D order `[head_dim, n_head_kv, kvps, n_stream]` (positions outer, heads inner per stream) makes the byte layout IDENTICAL to the legacy 2D K `[head_dim, n_head_kv*kv_size]` at all `n_stream` values — just partitioned by `ne[3]`. This was deliberate: it lets the existing ~30–40 K/V view/copy sites in `llama-build-context.cpp` and per-arch graph builders keep their legacy offset math unchanged. **The cost:** the byte-compatible layout is INCOMPATIBLE with a per-stream allocator. Legacy graph builders treat the cells[] array as a flat single arena; allocating slot s's metadata at flat range `[s*kvps, (s+1)*kvps)` puts the K-store / K-read at the right bytes (because of byte-compatibility), BUT each `llama_decode` still has to operate over the full flat cell range, mask-filtering by seq_id. No structural Bug-C closure follows from N1 alone; the decode-side prefill gate stays load-bearing.

**N2 and N3 are OPEN.** The structural Bug C closure (per-stream dispatch with no mixed mul_mat shapes, and removal of the decode-side gate to recover v1's TG-overlap) needs the FULL upstream-aligned non-byte-compatible 4D layout — i.e. axis order `[head_dim, kvps, n_head_kv, n_stream]` (heads outer, positions inner per stream) — paired with rewriting every K/V view/copy site to use stream-aware base offsets and per-stream `n_kv` bounds. Empirical bisect during this session (single-request to slot 1 with per-stream `find_slot` returned garbage tokens; concurrent NP=2 with per-stream `process_batch_tokens` dispatch returned garbage on R2 after the first few tokens) confirmed that the byte-compatible shortcut precludes per-stream semantics under unchanged graph builders.

## TL;DR

Port upstream's per-stream KV layout: K/V tensors gain a 4th axis `n_stream = max(1, n_seq_max)`. Each session owns its own contiguous KV slice. The graph builder dispatches one `llama_decode` per stream (or per-stream graph nodes within one decode) so **the mixed prefill+decode batch geometry never exists in mul_mat input shapes**. Slot 0's decode token always runs through a single-row mul_mat on its own stream; slot 1's prefill runs through a multi-row mul_mat on its own stream. No mixed batch → no GEMM-vs-GEMV divergence → no Bug C → **the decode-side prefill gate can be removed**, recovering v1's TG-overlap perf goal.

## Goal

After this phase, the production server:

1. Removes the decode-side prefill gate in `add_sampled_tokens` (no longer load-bearing).
2. Recovers v1's TG-overlap behaviour — decode tokens continue producing while one slot prefills on its own stream.
3. Closes Bug C structurally — by construction, no mixed batches ever form, so the GEMM-vs-GEMV divergence has no opportunity to misalign slot output.
4. Keeps all existing NPC closures: `r5-probe-c4 ITERS=20` = 0/20, multi-GPU `test-production-np-determinism.sh` matrix byte-identical, DFlash multi-slot tests green.
5. TG NP=8 burst throughput stays within ±1 % of current `27.73 t/s`. PP NP=2 wall-time drops below the current 15.9 s (true TG overlap during PP).

## Why this is now the right next step (vs deferring)

- The decode-side gate's TG-stall cost is real for any continuous-arrival workload. The PP-serialization bench wall (~15.9 s vs theoretical ~10 s with overlap) measures the tax directly.
- The kernel-level path to recover overlap is **closed** (PHASE_NSTREAM_KV update (g) — GEMM-vs-GEMV bit divergence is not patchable without rewriting all mul_mat kernels at major perf cost; the user has ruled this out).
- The 4D port is the only structurally clean path to TG-overlap.
- DFlash multi-slot's `N_slots`/`n_slots_cap` distinction gains a type-level guarantee.
- Upstream-alignment: future port work narrows in scope.

## Architecture target

```
                 K, V per layer
  current (3D):  [head_dim, kv_size, n_head_kv]
                 single contiguous arena, allocator scans whole tensor
                 cells[0..kv_size) flat metadata array
                 cache.head: one global next-free pointer

  target (4D):   [head_dim, kv_size_per_stream, n_head_kv, n_stream]
                 each stream's [0..kv_size_per_stream) slice is its own arena
                 v_cells[n_stream][kv_size_per_stream]: per-stream metadata
                 v_heads[n_stream]: one next-free per stream
```

Where `n_stream = max(1, cparams.n_seq_max)` and `kv_size_per_stream = cparams.n_ctx`.

Graph dispatch (the part that actually closes Bug C):

```
  current decode flow                  target decode flow

  one llama_decode per tick            one llama_decode per tick
  one batch contains all active        batch is split per-stream BEFORE the
  slots' tokens (mixed possible)       graph build; one graph per stream OR
                                       one graph with per-stream nb13 offsets
  mul_mat shapes [d, total_tokens]     mul_mat shapes [d, stream_tokens] per
  → bad GEMV/GEMM mixing               stream — pure GEMV for 1-token decodes
```

This is the upstream `llama-kv-cache.{cpp,h}` architecture. We are not porting the `llama_memory_i` virtual interface — only the layout + allocator + graph-builder strides.

## Work packages (bundle landing per `feedback_oneshot_then_evaluate`)

### N1 — Per-stream allocator + tensor reshape

**Files:**
- `src/llama-context.h` lines 37–170 — struct `llama_kv_cache` extension.
- `src/llama.cpp:790–1155` — `llama_kv_cache_init`: allocate K/V as 4D tensors with `ne[3] = n_stream`. Allocate `v_cells[n_stream]` per-stream metadata arrays.
- `src/llama.cpp:1156–1254` — `llama_kv_cache_find_slot`: take `seq_id` as input, scope scan to that stream's `v_cells[stream_id]` + `v_heads[stream_id]`. Return a slot_info-style struct.
- `src/llama.cpp:1942–2204` — `_clear`, `_seq_rm`, `_seq_cp`, `_seq_keep`, `_seq_add`, `_seq_pos_min/max`, `_defrag`: rewrite per-stream awareness.

**Scope:**
- Foundation slice already in place: `n_stream` + `v_heads` fields exist; not yet referenced by allocator.
- Recurrent state (`s_l`) already per-stream via `qnext_state_slots` per (a). No touch needed.
- Attention K/V (`k_l`/`v_l`) — actual structural change.

**Gate N1:**
- `bin/test-backend-ops` GREEN.
- Build clean with zero warnings on sm_75 nvcc.
- Existing DFlash unit tests GREEN (`test-dflash-closure`, `test-dflash-np-invariance`).

### N2 — Graph builder per-stream strides

**Files:**
- `src/llama-build-context.cpp` — ~30–40 sites that view/copy into `kv_self.k_l[il]` / `kv_self.v_l[il]`. Update each to use `nb13 = stream_size * n_head_kv * head_dim * sizeof(elem)` and stream-aware base offset.
- `src/graphs/build_qwen35.cpp` — Qwen 3.6 27B builder, mirror the build-context pattern at its attention K/V sites.
- `src/llama-dflash.cpp` — stride-accounting verification (~5 LoC changes; this file consumes pointers through the build path rather than constructing K/V views).

**Per-stream dispatch decision (the load-bearing choice):**
- Option A: one `llama_decode` per stream (server-side split). Simpler. Slightly higher CPU overhead per call.
- Option B: one graph per `llama_decode` with per-stream sub-graphs (mirrors current per-block split in `build_layer_attn_linear`, but for the whole attention pipeline). Lower CPU overhead but bigger graph.

Default: **Option A** — server-side split, one `llama_decode` per active stream per tick. Cleanest closure for Bug C semantics. Reassess if CPU overhead bites.

**Gate N2:**
- Build clean.
- `bin/test-dflash-closure` GREEN at np=1.
- `bin/test-dflash-np-invariance` GREEN at np ∈ {1, 2, 4, 8}.
- Single-GPU `scripts/r5-probe-c4.sh ITERS=20` PASSes WITHOUT the decode-side gate (the test the gate was previously load-bearing for).

### N3 — Server-side cleanup + gate removal

**Files:**
- `examples/server/server-context.cpp:2225, 4576–4687` — slot↔seq_id binding: with per-stream KV, `slot.id` IS `stream_id`. Simplify or drop the `batch.seq_id[i][0] == slot.id` check.
- `examples/server/server-context.cpp` `add_sampled_tokens()` — **remove** the decode-side prefill gate (the early-return when any slot is LOAD_PROMPT). This is the gate that recovers TG-overlap.
- Keep v1's PP-serialisation gate in `batch_pending_prompt` — that's a perf choice (parallel prefill is 4.8× slower per-seq on this geometry), independent of Bug C correctness.

**Headline closure gate (this is the binding test for the whole phase):**
- `scripts/test-production-np-determinism.sh` PASSes the full NP={1,2,4,8} matrix byte-identical at slot 0 on **single-GPU AND multi-GPU**.
- `scripts/r5-probe-c4.sh ITERS=20` = 0/20 fails single-GPU AND multi-GPU.
- `bin/test-dflash-np-multislot` GREEN unchanged.
- `scripts/test-pp-serialization.sh`: PP ≥ 60 t/s per request (unchanged from current), wall ≤ **10 s** (improved — recovers the ~6 s TG-overlap window we currently lose).
- `llama-batched-bench` TG NP=8 burst: within ±1 % of current `27.73 t/s` baseline.

### N4 — Production bake

- Submodule pointer bump.
- `MEMORY.md` entry covering Bug C structural closure + TG-overlap recovery measurement.
- `profiles/active.sh` review — note that the existing dflash-256K profile separately OOMs on startup (orthogonal to this phase; needs its own fix).
- Production server restart + 10-min mixed-workload soak.

## Critical files (reference card)

| Path | Lines | Phase |
|---|---|---|
| `src/llama-context.h` | 37–170 | N1 — struct extension |
| `src/llama.cpp` | 790–1155 | N1 — init + 4D K/V tensor alloc |
| `src/llama.cpp` | 1156–1254 | N1 — find_slot per-stream |
| `src/llama.cpp` | 1942–2204 | N1 — _seq_* per-stream |
| `src/llama-build-context.cpp` | ~30 view/cpy sites | N2 — per-stream nb13 |
| `src/graphs/build_qwen35.cpp` | search K/V views | N2 — same |
| `src/llama-dflash.cpp` | pointer-pass audit | N2 — stride accounting |
| `examples/server/server-context.cpp` | 2225, 4576–4687, `add_sampled_tokens` | N3 — gate removal + cleanup |
| `/home/llm/yarn-agentic/llama.cpp/src/llama-kv-cache.{cpp,h}` | reference only | — |

## Sequencing

Per `feedback_oneshot_then_evaluate`: write N1 + N2 + N3 as one coherent bundle, then evaluate against the headline gates. Do not commit at intermediate "build green but not yet correct" points.

Per `feedback_no_workarounds`: implement properly. The hybrid range-partition variant (b) attempted in PHASE_NSTREAM_KV slice 2 broke (empirically confirmed). Do not retry it.

Per `feedback_no_host_concerns_in_code`: code identifiers use `n_stream` (the upstream name) and content-descriptive function names. No `phase`/`slice`/`N1`/`N2` nomenclature in source.

## Open subtasks (named per `feedback_no_risks_only_tasks`)

- **NS4.OPEN.1 — `_seq_cp` cross-stream semantics.** Speculative draft acceptance copies content across slots. With per-stream KV, this becomes a cross-slice memcpy. Confirm correctness via `bin/test-dflash-closure`.

- **NS4.OPEN.2 — `--ctx-size` semantics review.** Today `--ctx-size N --parallel K` means total `N` cells across all slots. Per-stream changes this to `N` per stream (so total `N * K`). Document in `examples/server/README.md` and surface in the release note. Memory budget guidance: at production geometry (Q4_0 KV, 64 layers, head_dim=128), per-stream `n_ctx=8192` × 8 streams ≈ 2.3 GiB total — fits dual-24-GiB easily.

- **NS4.OPEN.3 — graph cache invalidation.** ggml may cache built graphs by tensor shape. The K/V shape change invalidates every cached graph. Trivial flush at startup; verify no stale graph survives a `llama_kv_cache_clear`.

- **NS4.OPEN.4 — Option A CPU overhead.** One `llama_decode` per active stream per tick increases per-tick call overhead. At NP=8 with one decode token each per tick, this is 8 calls vs 1 today. Profile after N2 lands; if overhead is material, pivot N2 dispatch to Option B (per-stream sub-graphs within one decode).

- **NS4.OPEN.5 — Cross-stream `_seq_keep` / `_clear` ordering.** When a slot finishes, its stream is released back to a free pool. Confirm the existing slot-lifecycle code in `server-context.cpp` releases cleanly without leaking cells from prior sessions into a re-allocated stream.

## Out of scope

- `llama_memory_i` virtual interface port — keep ik's free-function `llama_kv_cache_*` shape; the internal layout is the only change.
- CUDA kernel changes — kernels are already wire-compatible with per-stream strides (verified in PHASE_NSTREAM_KV (a)–(b)). The current Bug C closure happened in the scheduler; this phase moves it into the KV layout, leaving kernels alone.
- Production profile OOM (`qwen36-27b-x2-dflash @256K`) — orthogonal pre-existing issue; not a Bug C regression.
- D9.8 `transformer_kv` migration to `llama_session` — proceeds on its own timeline.

## Token cost estimate (per CLAUDE.md §8)

- N1 (struct + allocator + per-stream `_seq_*`): 40–60 k
- N2 (graph builder strides + per-stream dispatch): 60–100 k
- N3 (server cleanup + gate removal + bundle gate): 25–40 k
- N4 (production bake + MEMORY): 10–15 k
- PHASE + MEMORY commits per §5 push-per-edit: 5–10 k
- **Total: 140–225 k** for a clean closure cycle.
