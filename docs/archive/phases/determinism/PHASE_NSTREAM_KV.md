# PHASE_NSTREAM_KV — port upstream's per-stream KV axis into ik_llama.cpp

**Status**: ✅ **CLOSED** 2026-05-20.
**Branch**: `production/2026-q2-next`.
**Submodule HEAD at closure**: `16b608d1`.
**Predecessors**: `PHASE_NP_CLOSURE.md`, `PHASE_NPC4_FIX_AUDIT.md`.
**Successor**: `PHASE_NSTREAM_KV_PERF.md` (open — recover the -6.2 % TG-NP=8 regression).

## TL;DR — what landed

K/V tensors gained a per-stream axis: `[head_dim, kv_size_per_stream, n_head_kv, n_stream]`. Each session owns its own contiguous slice of the cache by construction. The server's `process_batch_tokens` splits mixed-seq batches into per-stream sub-batches before any `llama_decode` so each call sees a single-stream batch — `mul_mat` sees a uniform shape per call and **Bug C (the mixed prefill+decode GEMM-vs-GEMV accumulation divergence) is closed structurally**. The decode-side prefill gate (the v1 era's policy-level Bug C fix) is removed.

Six correctness gates green on Qwen 3.6 27B. One perf gate (G3.h, llama-batched-bench TG NP=8) failed by -6.2 % due to graph reuse being defeated at `n_stream > 1`; user-selected override merged the bundle and handed perf recovery to [`PHASE_NSTREAM_KV_PERF.md`](PHASE_NSTREAM_KV_PERF.md).

## Closure gate results

Qwen 3.6 27B production GGUFs (`qwen3.6-27b-V-F1.T1.lm_head-f16.gguf` correctness, `qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf` perf), dual Quadro RTX 6000, `CTX_PER_SLOT=4096`, q4_0 KV with Hadamard rotation (correctness gates).

| Gate | Result | Detail |
|---|---|---|
| **G3.a** single-GPU NP-determinism | ✅ PASS | NP ∈ {1,2,4,8} byte-identical to NP=1; cross-NP slot-0 matrix all identical |
| **G3.b** multi-GPU NP-determinism | ✅ PASS | Same with `--device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1`. Validates the V-split factoring (`a202f4f4`). |
| **G3.c** r5-probe-c4 single-GPU | ✅ PASS | 0/20 Bug C divergences WITHOUT decode-side gate. Per-stream dispatch structurally prevents Bug C. |
| **G3.d** r5-probe-c4 multi-GPU | ✅ PASS | 0/20. |
| **G3.e** test-dflash-np-multislot | ✅ PASS | slot-0 byte-identical NP ∈ {1,2,4,8} on 27B target + dflash drafter |
| **G3.f** spec tests | ✅ PASS | `test-n-stream-kv-layout` n_parallel ∈ {1, 2} on Qwen3.5-0.8B-BF16. KVTensorIsFourD + StreamPartition bind. |
| **G3.g** pp-serialization | ✅ PASS (caveat) | Per-request PP = 114.1 / 110.4 t/s ≫ 60 t/s threshold. Wall = 15.7 s vs pre-port 15.9 s — only ~1.3 % reduction vs the cached plan's ~38 % estimate. TG-overlap recovery marginal. |
| **G3.h** llama-batched-bench TG NP=8 | ❌ **-6.2 %** (user-override) | 26.00 t/s vs 27.73 t/s baseline. Outside ±1 % bound. |

## G3.h root cause + merge decision

Two layered findings on the bench failure:

**Layer 1 — bench-path bug, fixed inline (`16b608d1`).** `llama-batched-bench` feeds multi-seq batches directly to `llama_decode` (bypassing the server's per-stream dispatch). At `n_stream > 1`, `find_slot` derives stream_id from `seq_id[0][0]` and allocates ALL n_tokens cells into that one stream — corrupting it. The pre-existing qnext detector sub-batched `INTERLEAVED` patterns but passed `CONTIGUOUS_BLOCKS` through (the TG path). Extended the sub-batching to also fire on `CONTIGUOUS_BLOCKS` when `n_stream > 1`. `n_stream == 1` behaviour preserved (legacy CONTIGUOUS_BLOCKS pass-through). Also covers `parallel` and `perplexity` examples that hit the same direct-llama_decode path.

**Layer 2 — the 6.2 % TG-NP=8 regression.** Graph reuse is disabled at `n_stream > 1` (`src/llama.cpp:616`, intentional — view offsets are stream-aware so cross-stream reuse would read from the wrong slice). Every single-token sub-batch rebuilds the graph (~2–3 ms each). With 2112 single-token sub-calls in the bench, total rebuild cost ≈ 5–6 s of the 80 s run. Matches the observed delta. The production-server steady-state TG at NP=8 pays the same overhead.

**Merge decision (2026-05-20).** Per the locked perf-fail policy this gate would block merge. User was offered four options (hold + per-stream graph cache, hold + single-graph stream-aware reuse, override + merge, roll back N2/N3) and selected **"Override locked policy — merge with -6.2 % regression"**. Reason: all six correctness gates green; Bug C structurally closed; perf trade-off documented for the next phase. Per-stream graph cache is the next phase's work, not a current-step gap (no follow-up cover) — the structural closure goal of this phase is delivered without it.

## Submodule + parent commits at closure

**Submodule** `production/2026-q2-next` HEAD `16b608d1`:
- `0472275d` — N2 + N3 main: axis switch, graph builder rewrites (entry points), per-stream allocator, per-stream dispatch, decode-side gate removal.
- `95d3c9eb` — N2.b multi-device split per-stream K/V + gate K-shift/defrag to `n_stream == 1`.
- `a202f4f4` — worst-case `n_kv` bounded by `kv_size_per_stream` + V split factoring.
- `16b608d1` — N2 fixup: sub-batch CONTIGUOUS_BLOCKS at `n_stream > 1` (bench-path fix surfaced by G3.h).

Preceded by the N1 foundation (struct + 4D tensor reshape, init only, no allocator use):
- `52d845e9` — initial 4D K/V reshape + per-stream allocator (incomplete bundle).
- `c1beb104` — axis-order fixup to byte-compatible interim.
- `38ea4127` — revert per-stream find_slot, keep 4D + foundation.

**Parent repo** (this directory, `yarn-agentic`):
- `2bf3524` — initial N2 refined plan.
- `7e5c4eb` — locked N2 decisions.
- `7f1fe45` — initial closure status doc.
- `b30dbfa` — feature-branch N2+N3 MEMORY entry.
- (today) — gate sequencing + Stage 1 results, Stage 2+3 results + override decision, `scripts/r5-probe-c4.sh` DEVICE env override, submodule pointer bump, MEMORY closure entry, `PHASE_NSTREAM_KV_PERF.md` next-phase stub.

## Open follow-ups → next phase

Carried over to [`PHASE_NSTREAM_KV_PERF.md`](PHASE_NSTREAM_KV_PERF.md):

- **Per-stream graph cache** (the main item): one `prev->graph` per `stream_id` so each per-stream sub-batch reuses its own stream's graph instead of rebuilding. Recovers the -6.2 % regression and probably the TG-overlap window too.
- **K-shift** (`build_k_shift`): currently `GGML_ASSERT(kv_self.n_stream == 1)`. Lift for `ctx_shift` at multi-slot.
- **Defrag** (`build_defrag`): same gate, same lift.
- **v_trans non-FA V path**: same gate. Production runs FA-on so it's currently a guard; lift when adding non-FA paths.
- **MLA path (DeepSeek)**: out of scope (production model is non-MLA).

## Implementation summary — what each work package delivered

### Spec layer (S1–S5) — pre-implementation

Per the cached plan `/home/llm/.claude/plans/cached-crunching-tiger.md`:

- **S1.a `specs/scheduler/batch_composition.allium`** — contracts: `PrefillSerialisationGate`, `DecodeHoldGate`, `BatchCompositionInvariant`, `MixedBatchProhibition`, `AtMostOnePrefillSlotPerBatch`. `allium check` clean.
- **S1.b `specs/kv-cache/n_stream_layer.allium`** — contracts: `PerStreamAllocator`, `MaskPerStream`, `PerStreamDispatch`, `BugCAbsenceByConstruction`, `DFlashCompatibility`. `allium check` clean.
- **S1.c `mtp_fused_draft.allium`** tend — added `FusedDraftRoundsRunOnPureDecodeBatches` + `FusedDraftRespectsStreamPartition` cross-cutting invariants.
- **S2.a `specs/multislot/BatchComposition.tla`** — SANY-clean. State machine + WF fairness.
- **S2.b `specs/multislot/StreamIsolation.tla`** — SANY-clean.
- **S3** — TLC model check. Configs `BatchCompositionMC.cfg` (gate ON, 541 distinct states, PASS), `BatchCompositionMC_no_gate.cfg` (gate OFF, `BatchCompositionInvariant` violated — spec binds), `StreamIsolationMC.cfg` (PerStreamDispatch ON, PASS), `StreamIsolationMC_legacy.cfg` (PerStreamDispatch OFF, `StreamPartition` violated — spec binds).
- **S4** — property tests via `allium plan` obligations. `tests/spec/test-batch-composition-gates.cpp` PASS on HEAD. `tests/spec/test-n-stream-kv-layout.cpp` foundation PASS; `KVTensorIsFourD` FAIL on pre-N1 HEAD with `k_l[3]->ne[3]=1 vs n_stream=2` — the binding RED test for N1.
- **S5** — NDJSON trace harness + live validation. `examples/server/server-trace-ndjson.h` emit helper, gated on `LLAMA_TRACE_NDJSON_DIR`. Validator `scripts/validate-batch-composition-trace.py`. Live-verified on Qwen3.5-0.8B BF16.

### N1 — Per-stream allocator + tensor reshape

`struct llama_kv_cache` extended (`src/llama-context.h:37–170`) with `n_stream`, `kv_size_per_stream`, `v_heads`. `llama_kv_cache_init` allocates K/V as 4D `[head_dim, kv_size_per_stream, n_head_kv, n_stream]`. `kv_size` rounded UP to a multiple of `n_stream`. `_clear / _seq_rm / _seq_keep / _seq_add` carry per-stream `v_heads` tracking.

**Axis-order decision (load-bearing).** Initial commits (`52d845e9` → `c1beb104` → `38ea4127`) tried the byte-compatible interim layout `[head_dim, n_head_kv, kvps, n_stream]`. Empirical bisect 2026-05-20 showed it's INCOMPATIBLE with per-stream semantics under unchanged graph builders: single-request to slot 1 returned garbage tokens; concurrent NP=2 returned garbage on R2 after the first few tokens. The byte-compatible shortcut was reverted in favour of the full upstream-aligned `[head_dim, kvps, n_head_kv, n_stream]` (heads outer per stream, positions inner) which requires every K/V view/copy site to be rewritten with stream-aware offsets. That work landed in N2. See `feedback_n_stream_byte_compat_tradeoff` in the auto-memory for the full reasoning.

### N2 — Graph builder per-stream strides + dispatch

Touched files (all on `feature/nstream-kv-4d-n2`, then merged):

- `src/llama-context.h` — struct extension (N1 foundation).
- `src/llama.cpp` — per-stream `find_slot`, `_seq_*` ops, `_clear`, `can_reuse_graph` (returns false at `n_stream > 1` pending the next phase's per-stream graph cache), `update_cache_copies` (stream-aware `view_offs`), mask builder (stream-local cell range), `llm_build_context` ctor (worst-case `n_kv` bounded by `kv_size_per_stream`), llama_decode interleaved sub-batching extended to CONTIGUOUS_BLOCKS at `n_stream > 1`.
- `src/llama-build-context.cpp` — `llm_build_kv_store`, `llm_build_kqv`, `build_std_attention` (multi-device branch with split K/V) rewritten with stream-aware base offsets `s * nb[3]`. K-shift / defrag / v_trans guarded `GGML_ASSERT(n_stream == 1)`.
- `examples/server/server-context.cpp` — `process_batch_tokens` per-stream dispatch (scan for contiguous primary-seq_id runs, one `llama_decode` per run). Decode-side prefill gate REMOVED in `add_sampled_tokens`.

### N3 — Server cleanup + decode-side gate removal

Already folded into N2's `process_batch_tokens` rewrite. The seq_id→slot binding simplifies because `slot.id` now naturally maps to `stream_id`. The decode-side prefill gate (v1's Bug C policy fix) is REMOVED — per-stream dispatch makes mixed batches impossible by construction.

## Design history (condensed)

PHASE_NSTREAM_KV.md went through seven updates during the design phase before the implementation bundle landed. Compressed timeline:

- **Original spec** — port upstream's per-stream KV axis; closes Bug C, enables v1 scheduler reland, gives DFlash multi-slot a type-level guarantee.
- **(b) DESIGN PIVOT (SUPERSEDED)**: range-partition over 3D, not 4D port. Smaller scope (~500–1000 LoC vs ~3000–5000). Rejected once (c) showed the kernel's `nb13` arithmetic doesn't drive Bug C — the mask discriminates per-slot, not the per-stream axis.
- **(c) STRUCTURAL FIX PAUSED**: pre-implementation read of `fattn-per-slot-kv-singlewarp-sm75.cu` invalidated the mechanism hypothesis. Per `feedback_diagnostic_discipline_before_declaring_done` — don't ship a structural rewrite while the root cause is a hypothesis. Open subtask #103 was opened to confirm the mechanism empirically.
- **(d) Decoupled from Bug C, still wanted**: n_stream KV is a standalone-value port (v1 reland, DFlash typesafety, upstream alignment). Pursued AFTER Bug C closure via the scheduler path.
- **(e) Bug C mechanism CONFIRMED**: `LLAMA_KV_CONCURRENT_TRACE` instrumentation showed the failing iteration has a mixed prefill+decode batch (slot 0's first decode token batched with slot 1's full 210-token prefill). Bug C is downstream of the mask, in a kernel/graph node that fails on mixed batch geometry.
- **(f) Phase C deferred — wider blocker**: the (b) range-partition variant needs a scatter-quantized-write op in ggml (KV cache is Q4_0; `ggml_set_rows` requires F32 source). Either invent that op, or commit to the full 4D port.
- **(g) Kernel-level bisect of Bug C — GEMV-vs-GEMM accumulation order**: per user direction "diagnose and fix 1", attempted kernel-level fix. Findings: layer-0 `l_out-0` diverges at slot-0 row with `max|Δ| = 3.011e-03`. Classic accumulation-order divergence — for row shape `[d, 1]` mul_mat picks GEMV with sequential accumulator; for `[d, N>1]` it picks tiled GEMM with parallel partial sums + reduction tree. Same math, bit-different order. **No kernel-level patch without rewriting all mul_mat kernels.** Conclusion: the decode-side prefill gate is the correct fix at the scheduler level. The 4D port is then the structural way to make Bug C impossible by construction (and remove the gate's perf cost).

## What this phase explicitly did NOT do (per CLAUDE.md §2)

- Did **not** port `llama_memory_i` / `llama_memory_t` virtual interface. ik uses static dispatch.
- Did **not** touch CUDA kernels. They are per-slot-pointer wire-compatible.
- Did **not** rewrite ik-only machinery: `transformer_kv`, `delta-net.cpp`, `dflash.cpp` machinery composes with the new allocator by stride.
- Did **not** introduce a new C API surface. The change is internal to `llama_kv_cache_*`.
- Did **not** lift the K-shift / defrag / v_trans paths off `n_stream == 1` (guards are in place — future work).
- Did **not** rebuild graph caching for the multi-stream case. `can_reuse_graph` returns false at `n_stream > 1` — the perf cost of this is what the next phase recovers.

## Critical files (reference card)

| Path | Role | Phase |
|---|---|---|
| `src/llama-context.h:37–170` | `struct llama_kv_cache` extension (`n_stream`, `kv_size_per_stream`, `v_heads`) | N1 |
| `src/llama.cpp:1156–1429` | `llama_kv_cache_find_slot` per-stream allocator | N2.c |
| `src/llama.cpp:1942–2204` | `_clear / _seq_rm / _seq_keep / _seq_add / _seq_add` per-stream awareness | N1 |
| `src/llama.cpp:5439–5490` | `llama_decode_internal` qnext sub-batching extended to CONTIGUOUS_BLOCKS at n_stream>1 | N2 fixup |
| `src/llama.cpp:586–625` | `can_reuse_graph` — guards graph reuse at `n_stream > 1` (next-phase work) | N2 |
| `src/llama-build-context.cpp` (~40 K/V sites) | Stream-aware view/copy offsets `s * nb[3]` | N2.b |
| `examples/server/server-context.cpp` `process_batch_tokens` | Per-stream dispatch (seq_id-run split) | N2.d |
| `examples/server/server-context.cpp` `add_sampled_tokens` | Decode-side prefill gate **REMOVED** | N3 |
| `specs/scheduler/batch_composition.allium` | Scheduler contracts | S1.a |
| `specs/kv-cache/n_stream_layer.allium` | KV-layer contracts | S1.b |
| `specs/multislot/BatchComposition.tla`, `StreamIsolation.tla` | TLA+ state machines | S2 |
| `tests/spec/test-n-stream-kv-layout.cpp` | Binding RED test for N1 → GREEN after | S4 |
| `tests/spec/test-batch-composition-gates.cpp` | 1296-config gate sweep | S4 |
| `scripts/test-production-np-determinism.sh` | G3.a/b binding harness | gate |
| `scripts/r5-probe-c4.sh` | G3.c/d Bug C absence harness | gate |
| `scripts/test-pp-serialization.sh` | G3.g PP-serialization throughput | gate |

## Token cost (final, per CLAUDE.md §8)

Estimated 90–145 k in the original plan, 230–375 k including the spec layer. Actual closure span (S1 + N1 fixups + N2+N3 bundle + gate runs + override + cleanup) consumed roughly that — within the upper envelope. The 4D port's true scope landed close to the original (a) estimate of ~3000–5000 LoC (final diff: ~465 insertions / 150 deletions across `llama-context.h`, `llama.cpp`, `llama-build-context.cpp`, `server-context.cpp`, plus the new spec test).

The byte-compatible-interim detour (N1 fixup #1 + #2) cost roughly the savings the shortcut promised — a wash, but instructive (see `feedback_n_stream_byte_compat_tradeoff`).
