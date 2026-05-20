# PHASE_NSTREAM_KV — port upstream's per-stream KV axis into ik_llama.cpp

**Branch**: `production/2026-q2-next`
**Predecessors**: `PHASE_NP_CLOSURE.md` (Bug C open after R5), `PHASE_NPC4_FIX_AUDIT.md` (kernel-layer audit closed).
**Status**: Specced, not yet implemented.

## TL;DR

ik_llama.cpp's KV cache is a single shared region, sliced by a linear
`head + n_tokens` allocator, with seq_id→cell mapping hand-rolled by
the server. Upstream llama.cpp's KV cache carries a third tensor axis
`n_stream` (`[head_dim, kv_size, n_stream]`) with a per-stream
`slot_info` allocator — each session owns its own stream by construction,
and the server's seq_id→stream binding is trivial.

This phase ports **only** that per-stream axis and `slot_info` allocator
into ik. It does **not** port upstream's `llama_memory_i` virtual
interface, does **not** touch CUDA kernels (they already accept per-slot
strides), and does **not** rewrite ik's `transformer_kv`,
`delta-net.cpp`, or `dflash.cpp` machinery.

Closes by construction:
- **Bug C** (PHASE_NP_CLOSURE): NP=2 stochastic ~10% NPC failure caused
  by slot-KV cross-contamination at concurrent prefill.
- **v1 scheduler reland** (PHASE_PERF_F4_1): the "shared pool pretends
  to be per-session" friction goes away; v1's PP-serialised + TG-overlapped
  model becomes the natural mapping onto streams.
- **DFlash multi-slot's `N_slots` vs `n_slots_cap` class of bug**
  (PHASE_DFLASH_MULTISLOT Phase 4): the bug exists because the surrounding
  KV layer is shared. Per-stream KV makes the distinction enforced by the
  type system.

## Goal

The production server's KV cache layer presents per-session ownership at
the tensor and allocator level, with no shared region that can be
contaminated by a concurrent peer. After this phase:

- `scripts/test-production-np-determinism.sh` PASSes the full NP=
  {1, 2, 4, 8} matrix byte-identically at slot 0.
- The harness reproducer for Bug C (10× iterations of NP=2 fire-2,
  `scripts/r5-probe-c4.sh`) shows 0 / 10 failures.
- v1 scheduler relands (cherry-pick of commit `67878813` onto the
  post-phase base) PASSes NPC + recovers its 3.5× wall-time win on
  `scripts/test-pp-serialization.sh`.
- `bin/test-dflash-np-multislot` PASSes unchanged.

## Architectural context

### Current shared-pool model (ik_llama.cpp)

- KV cache is a `struct llama_kv_cache` defined inline in `src/llama.cpp`,
  one instance per `llama_context`. K/V tensors are 3D:
  `[head_dim, kv_size, n_head_kv]`.
- Allocation by `llama_kv_cache_find_slot` at `src/llama.cpp:1156–1256`:
  linear scan over `cache.cells[]` finding a contiguous run of free
  cells of length `n_tokens`, then assigns them to one or more seq_ids.
- Seq_id ↔ slot binding is hand-rolled in
  `examples/server/server-context.cpp:2225, 4576–4687`. The server
  enforces `batch.seq_id[i][0] == slot.id` but there is no type-level
  guarantee that two slots' prefill writes don't land in overlapping
  cell ranges under concurrent fire.
- **This is the surface where Bug C lives.** Concurrent prefill from
  two requests submits two `llama_batch`es that race through
  `find_slot`; the bug signature (slot 0 attempts to re-emit slot 1's
  prompt) is consistent with `cache.head` advancing during slot 1's
  allocator call between slot 0's `find_slot` and the kernel write.

### Target n_stream-axis model (port from upstream)

- K/V tensors gain a third axis: `[head_dim, kv_size, n_head_kv, n_stream]`
  where `n_stream == n_seq_max` (one stream per session). DeltaNet's
  recurrent state already has this shape; KV cache catches up.
- `cells[]` becomes `v_cells[n_stream]` and `head` becomes `v_heads[n_stream]`.
  Each stream allocates independently inside its own region.
- `find_slot` returns a `slot_info` carrying per-stream cell indices.
  `prepare()` iterates ubatches; within a ubatch, each token's
  `seq_id[0]` selects its stream.
- Kernels receive K/V pointers with `nb13 == sizeof(K_per_stream)` (the
  stride between streams). They are **already wire-compatible**:
  `fattn-per-slot-kv-singlewarp-sm75.cu:68–94` reads
  `K + nb13*seq + nb12*head_kv` today — the bug is upstream, in the
  allocator that produces the strides.

### What this phase explicitly does NOT do

Per CLAUDE.md §2 (Simplicity First) and §1 (push back where simpler
exists):

- Do **not** port `llama_memory_i` / `llama_memory_t` virtual
  interface. The runtime polymorphism upstream uses (recurrent vs
  hybrid vs standard) is not load-bearing here — ik has one cache
  shape per model and dispatches statically.
- Do **not** touch CUDA kernels. They are already per-slot-pointer
  compatible (research finding from `Per-session KV abstraction
  survey`, 2026-05-19).
- Do **not** rewrite ik-only machinery: `transformer_kv`,
  `delta-net.cpp`, `dflash.cpp`, `llama-build-context.cpp` (125k LoC).
  These compose with the new allocator by stride.
- Do **not** introduce a new C API surface (`llama_session_*`,
  `llama_memory_*`). Keep ik's existing `llama_kv_cache_*` free-function
  shape; the change is internal.

## Work packages

Four packages, each with a binding verification gate. Per §4 (Goal-Driven
Execution), each step's gate names the exact harness call that closes it.

### N1 — Allocator + tensor reshape

Add the `n_stream` axis to `struct llama_kv_cache`. K/V tensors become
4D with the new axis = `cparams.n_seq_max`. Replace the single
`cache.head` / `cache.cells[]` with `v_heads[n_stream]` /
`v_cells[n_stream]`. Rewrite `llama_kv_cache_find_slot` to allocate
within `v_cells[stream_id]` where `stream_id = batch.seq_id[i][0]`.

Touched files:
- `src/llama.cpp:790` (`llama_kv_cache_init`)
- `src/llama.cpp:1156–1256` (`llama_kv_cache_find_slot` and supporting)
- `src/llama.cpp:1942–2143` (`_clear / _seq_rm / _seq_cp / _seq_keep /
  _seq_add` — all need per-stream awareness)
- `src/llama.cpp:2177–2202` (`_seq_pos_min/max / _defrag`)

Verification gate N1:
- `bin/test-backend-ops` GREEN — unit tests for KV cache ops still pass.
- A new test `tests/dflash-speculative/test-kv-cache-stream-isolation.cpp`
  asserts that writes to stream `s1` do not modify any cell of stream
  `s2 ≠ s1`. **Build this test before the implementation lands** and
  fail it against the current shared-pool code to prove the test
  mechanism works (per `feedback_verify_test_mechanism_before_trusting`).
- ik builds clean with zero warnings on sm_75 nvcc.

### N2 — Graph-builder strides

Update every graph builder site that constructs K/V references to pass
`nb13 = n_kv * n_head_kv * head_dim_size * sizeof(elem)` (the per-stream
stride) instead of the current shared-pool stride. Likely sites:

- `src/llama-build-context.cpp` — the bulk; ~125k LoC, only the K/V
  view / copy / select sites change.
- `src/graphs/build_qwen35.cpp:127` and adjacent — Qwen 3.6 27B graph
  builder.
- DFlash drafter graph in `src/llama-dflash.cpp` — has its own
  per-slot scratch; verify the stride accounting matches.

Touched files:
- `src/llama-build-context.cpp` (search for `ggml_view_*` and `ggml_cpy`
  on `kv.k_l[il]` / `kv.v_l[il]`).
- `src/graphs/build_qwen35.cpp`.
- `src/llama-dflash.cpp`.

Verification gate N2:
- `bin/test-dflash-closure` GREEN at np=1 (DeltaNet/DFlash drafter
  composes with the new strides).
- `bin/test-dflash-np-invariance` GREEN at np ∈ {1, 2, 4, 8}.
- A `scripts/test-production-np-determinism.sh` run at np=1 produces
  byte-identical output vs the pre-phase np=1 baseline. (np>1 not yet
  expected to pass — that's gated on N3.)

### N3 — Server-side seq_id → stream cleanup

The server's `server-context.cpp:4576–4687` hand-rolls slot↔seq_id
binding. With per-stream KV, slot.id naturally maps to stream_id. The
work here is removing the bookkeeping that existed to defend the
shared-pool model against the cross-contamination Bug C, and
simplifying the per-slot mask construction now that the streams are
truly independent.

Touched files:
- `examples/server/server-context.cpp:2225, 4576–4687`.
- Any server-side code that uses `cache.head` as a global value.

Verification gate N3:
- `scripts/test-production-np-determinism.sh` PASSes the full NP=
  {1, 2, 4, 8} matrix byte-identically at slot 0. This is the headline
  gate — it binds on **Bug C closure**.
- `scripts/r5-probe-c4.sh ITERS=10` shows 0 / 10 failures on
  single-GPU NP=2.
- The same probe run multi-GPU (`--device CUDA0,CUDA1 --split-mode
  graph --tensor-split 1,1`) also shows 0 / 10.

### N4 — v1 scheduler reland + DFlash regression + perf

With per-stream KV in place, cherry-pick `67878813` (the v1 scheduler)
onto the post-N3 base. Per the upstream survey, v1 was reverted because
it failed NPC under the shared-pool model — with per-stream KV its
"PP-serialised, TG-overlapped" model fits the architecture cleanly.

Touched files:
- `examples/server/server-context.cpp` (cherry-pick of v1).
- Re-bench against the recorded baselines.

Verification gate N4:
- `scripts/test-pp-serialization.sh`: PP ≥ 60 t/s per request, wall
  ≤ 20s for 2× 710-tok concurrent requests (the v1 win recovered).
- `scripts/test-production-np-determinism.sh`: full byte-identity
  matrix still PASSes (v1 didn't reintroduce Bug C).
- `llama-batched-bench` at production shape: TG NP=8 regression vs
  pre-phase baseline ≤ 1%. (Per `feedback_determinism_must_co_optimize_perf` —
  this phase must not trade perf for determinism.)
- `bin/test-dflash-np-multislot` GREEN unchanged.

## Critical files

| Path | Role | Touch |
|---|---|---|
| `src/llama.cpp:790` | `llama_kv_cache_init` | **N1** — add `n_stream` axis |
| `src/llama.cpp:1156–1256` | `llama_kv_cache_find_slot` allocator | **N1** — per-stream cell allocation |
| `src/llama.cpp:1942–2202` | `_seq_*` ops + `_defrag` | **N1** — per-stream awareness |
| `src/llama-build-context.cpp` | Graph builder K/V views | **N2** — pass per-stream stride |
| `src/graphs/build_qwen35.cpp:127` | Qwen 3.6 27B builder | **N2** — same |
| `src/llama-dflash.cpp` | DFlash drafter graph | **N2** — stride accounting |
| `examples/server/server-context.cpp:2225, 4576–4687` | Slot↔seq_id binding | **N3** — simplify |
| Upstream `src/llama-kv-cache.{cpp,h}` | Port targets | **read-only reference** |
| `tests/dflash-speculative/test-kv-cache-stream-isolation.cpp` | New isolation unit test | **N1** — write before fix |
| `scripts/test-production-np-determinism.sh` | Binding harness | reuse, do not modify |
| `scripts/r5-probe-c4.sh` | Bug C stochastic reproducer | reuse for N3 gate |

## Sequencing notes

Per `feedback_oneshot_then_evaluate`: this is a structural rewrite, not
an experimental sweep. Write N1+N2+N3 coherently as one bundle, then
evaluate measured results against the gates — including failures. Do
not stop at intermediate "build green" points and call it progress.

Per `feedback_no_tmp_for_large_artifacts`: harness outputs land in
`data/nstream-kv/run-YYYYMMDDTHHMMSS/`, not `/tmp`.

Per `feedback_no_host_concerns_in_code`: NSTREAM / N1 / N2 nomenclature
is for this doc only — code identifiers use `n_stream` (the upstream
name) and content-descriptive function names.

## Out of scope

- `llama_memory_i` virtual interface port — see "What this phase
  explicitly does NOT do".
- CUDA kernel changes — they are wire-compatible.
- `transformer_kv` / DFlash / DeltaNet machinery — composes with the
  new allocator.
- D9.8 (`transformer_kv` migration to `llama_session`) — orthogonal;
  proceeds on top of this phase or on its own timeline.
- TU102 + NVLINK perf envelope — orthogonal.
- MMQ I=8 further perf work — landed; out of scope here.

## Open subtasks (named, per `feedback_no_risks_only_tasks`)

These are not "risks with mitigations" — they are known gaps that
become concrete sub-tasks if hit:

- **NS.OPEN.1 — DeltaNet recurrent state stride.** DeltaNet's `[head_dim,
  head_dim*n_heads, 1, n_seqs]` tensor already has a seq axis but is
  allocated in a single contiguous block. N1 must verify that the
  per-stream stride convention matches the existing recurrent-state
  layout, OR change the recurrent-state layout to match. Decided at
  N1-implementation time by reading `src/llama-delta-net.cpp`.

- **NS.OPEN.2 — `_seq_cp` semantics under per-stream.** Sequence-copy
  ops (used by speculative draft acceptance) currently copy within the
  shared pool. Per-stream, they may need to copy across streams. Could
  be a pointer-swap rather than a memcpy if streams are equal-sized.

- **NS.OPEN.3 — `_defrag` cost.** Per-stream, defrag runs per stream
  independently — total work is unchanged but the inner loop body
  doesn't have to coalesce across streams. Confirm this is a
  simplification, not a regression.

- **NS.OPEN.4 — KV memory budget.** Per-stream KV at `n_seq_max=8` and
  `n_ctx=8192` requires the same total bytes as the current shared
  pool at `n_ctx=8192*8`. Confirm the existing `--ctx-size` flag
  semantics still hold (it has historically meant "total context across
  all parallel requests"). If users have been treating it as
  per-request, this phase changes that — document in
  `examples/server/README.md` if so.

- **NS.OPEN.5 — Graph cache invalidation.** If ggml's graph cache keys
  on tensor shapes, the n_stream axis change invalidates every cached
  graph. Trivial flush at startup, but verify no stale graph references
  survive a `llama_kv_cache_clear`.

Each subtask becomes a checkbox under its parent work package's
verification gate. None of them are "deferred to a follow-up phase".

## Estimated token cost (per CLAUDE.md §8)

- N1 allocator + tensor reshape: 30–50k
- N2 graph-builder strides: 20–30k
- N3 server-side cleanup + N3 verification: 15–25k
- N4 v1 reland + DFlash regression + perf bench: 20–30k
- MEMORY entries + PHASE updates + commits (per §5 push-per-edit): 5–10k
- **Total: 90–145k** for a clean closure cycle.

Compares to the prior estimate of 70–175k to close Bug C locally
without touching the wider goals. The structural fix has a higher
floor but a lower ceiling, and closes three workitems in one effort.
