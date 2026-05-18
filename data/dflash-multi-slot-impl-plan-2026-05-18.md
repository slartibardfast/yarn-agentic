# DFlash multi-slot libllama API — implementation plan (2026-05-18)

Companion to `data/dflash-multi-slot-api-brief-2026-05-18.md`. The brief
laid out scope before reading code; this plan reflects what the
source-read revealed.

## State verified at session start

- `scripts/verify-production-determinism.sh` PASSES at NP={1,2,4,8} on
  current `production/2026-q2-next` (re-run 2026-05-18). PSKV 4-way ILP
  preserves the NPC contract.
- DFlash build flag is still `-DGGML_CUDA_DFLASH=ON` (not in current
  production build; will need a separate build dir for harness work).

## Source-read findings

### 1. The `LLAMA_DFLASH_NP_GT_1` enum is a no-op placeholder

Declared at `include/llama.h:1758`, **never returned from any call**
(0 grep matches in `src/`, `common/`, `examples/`, `include/` outside
its declaration). The implicit np>1 gate is the hardcode
`const int N_slots = 1;` at `src/llama-dflash.cpp:591`, plus all the
scratch allocations being sized for 1 slot at lines 355-374.

So there is no rejection site to "lift" — instead we need to plumb a
real `N_slots` value end-to-end and remove the constant.

### 2. Kernel side is already multi-slot-shaped

`dflash_drafter_forward_launch` at
`ggml/src/ggml-cuda/dflash/dflash-drafter-forward.cu:512` takes
`N_slots` as a parameter; `n_rows = N_slots * Q` is used throughout.
All kernels (`combine_features`, `inject_kv_fused`, `attention_kernel`,
`drafter_lm_head_kernel`, etc.) accept `N_slots`. T7 already validated
np-invariance for `N_slots × Q` grid at the kernel level.

The kernel layer needs no new work. The drafter forward kernels also
provide the in-flight parallelism we'd want anyway.

### 3. Dispatcher already supplies seq_id per slot

`common_speculative_init(params, ctx_tgt, seq_id)` at
`common/speculative.cpp:1162` takes seq_id and is called once per NP
slot. The MTP branch (line 1262) passes seq_id into
`common_speculative_state_mtp`. The DFlash branch (line 1281) does
NOT pass seq_id; it discards it.

This means the per-slot wiring already exists at the server orchestrator
level. DFlash just needs to use it.

### 4. One-drafter-per-context invariant forces a shared dispatch

`llama_set_dflash(ctx_tgt, drafter)` errors on second call to the same
context (line 407: "called twice on same context"). So we cannot
instantiate one `common_speculative_state_dflash` per slot each with
its own drafter. The drafter binds once; multi-slot must fan out
inside that one binding.

This makes DFlash structurally different from MTP, where each slot's
state is independent. DFlash needs:
- A single "owner" `common_speculative_state_dflash` (one binding to
  ctx_tgt), OR
- N adapter instances that all reference one shared drafter handle
  with serialized access to its scratch buffers.

### 5. cb_eval extract buffer has no seq_id awareness

This is the deepest architectural gotcha. The cb_eval hook in
`src/llama.cpp:9661-9725` (`llama_dflash_extract_cb_eval`) fires once
per ubatch and appends ALL rows of `l_out-<il>` to a single host
buffer indexed by source-layer slot.

Critical lines (9707-9716):
```cpp
buf.resize(old_n_floats + (size_t) n_elements);
ggml_backend_tensor_get(t, buf.data() + old_n_floats, 0, nbytes);
```

At np>1 with multiple seq_ids in the ubatch, the rows from different
slots get interleaved into the same per-layer buffer with no way to
demux at extract time. The current MAL accounting at
`llama-dflash.cpp:595` (`MAL = anchor_pos`) implicitly assumes one
sequence per context.

**The cb_eval hook needs a row→seq_id map** for multi-slot to be
correct.

### 6. Scratch buffer sizing is single-slot-shaped

`alloc_ctx_scratch` at `src/llama-dflash.cpp:340-378` sizes:
- `d_input_emb` = `Q * D_emb` (no N_slots factor)
- `d_drafter_hidden` = `BS * D_emb`
- `d_drafter_logits` = `BS * V`
- `d_anchor_pos` = `mal_cap`
- `d_slot_positions` = `sizeof(int)` (literally 4 bytes)
- `d_ctx_states` = `mal_cap * D_emb`
- `d_target_hiddens` = `mal_cap * L_src * D_emb`
- `h_logits` = `BS * V`
- K/V cache = `L_d * SeqLen * H_kv * D_h` (per-layer; needs N_slots
  factor per the cuh comment that says
  `[L_d, N_slots, SeqLen, H_kv, D_h]`)

All of these need an `N_slots` factor for multi-slot.

## Implementation phases

Cheapest, most reversible first.

### Phase 1 — C API surface (declaration + stub)

Goal: surface the multi-slot entry point as a no-op at np=1 to confirm
the API shape doesn't break anything.

Touches:
- `include/llama.h` — add `llama_dflash_draft_batch` declaration and
  the `LLAMA_DFLASH_MAX_SLOTS` constant (or just document that the
  caller is bounded by `cparams.n_seq_max`).
- `src/llama-dflash.cpp` — stub `llama_dflash_draft_batch` that
  asserts `n_slots == 1` and calls existing `llama_dflash_draft` once.

Verify: builds, single-slot existing tests still pass, NPC
verification still PASSES.

### Phase 2 — Per-slot scratch sizing

Goal: scale all scratch buffers by `cparams.n_seq_max` so multi-slot
allocations have room. Still pass `N_slots = 1` to kernels — only
the allocation surface changes.

Touches:
- `src/llama-dflash.cpp` `alloc_ctx_scratch` — multiply slot-major
  dims by `n_slots_cap` (taken from `ctx_tgt->cparams.n_seq_max` at
  bind time).
- `llama_dflash_ctx_state` — store `n_slots_cap` for later use.
- K/V cache layout — confirm kernels write `[L_d, N_slots, ...]`
  ordering matches our allocation.

Verify: builds, single-slot tests still pass, NPC still PASSES,
memory usage at `n_seq_max=1` unchanged.

### Phase 3 — cb_eval per-slot demux

The deepest change. The cb_eval hook needs to know which rows belong
to which seq_id.

Touches:
- `src/llama.cpp:9661` cb_eval implementation — track a parallel
  `row→seq_id` map populated by the decode driver before scheduling.
- `src/llama-decoder-internal.h` — add `dflash_extract_buf[]` as
  `vector<vector<float>>` (per [slot][seq_id]) or maintain a
  row-seq_id index buffer.
- `src/llama-dflash.cpp` `stage_target_hiddens` — demux rows per
  seq_id into N parallel staging arrays.
- The build_qwen35 graph node where the residual is named — confirm
  whether the F16 multi-token ubatch path needs any change.

Verify: cb_eval logs show per-slot row counts at np>1, byte-identical
when N=1 to current behavior.

This phase has the highest implementation risk because the ubatch→
seq_id mapping has to be wired through layers that don't currently
expose it.

### Phase 4 — Multi-slot dispatch in C entry

Touches:
- `src/llama-dflash.cpp` `llama_dflash_draft_batch` — replace
  `N_slots = 1` with the `n_slots` arg, stage all slots' target
  hiddens, run combine/inject/forward/lm_head with N_slots > 1,
  argmax per slot.
- `src/llama-dflash.cpp` `llama_dflash_draft` — implement as a
  trampoline calling `_batch` with `n_slots = 1`.

Verify: `llama_dflash_draft_batch` produces byte-identical output to
single-slot at N=1, and at N>1 each slot's BS candidates match what
serial single-slot calls would have produced.

### Phase 5 — Adapter glue

Touches:
- `common/speculative.cpp` `common_speculative_state_dflash` — accept
  seq_id at construction (line 974), store it.
- `common/speculative.cpp` dispatcher line 1281 — pass seq_id to the
  constructor.
- `common/speculative.cpp` `common_speculative_draft_batched` — add
  a DFlash branch mirroring the all_mtp path (line 1452+). Detects
  "all DFlash + shared ctx_tgt" and dispatches via
  `llama_dflash_draft_batch`. Falls back to per-slot serial otherwise.

Verify: server at np=2 then np=4 then np=8 runs DFlash without
crashing; spec accept rate measurable.

### Phase 6 — Multi-slot DFlash test harness

Touches:
- New file `tests/dflash-speculative/test-dflash-np-multislot.cpp`
  modeled on `test-dflash-np-invariance.cpp` (T7) for the kernel
  layer + `test-cy-np2-multi-step-decode.cpp` for the C-API
  driver layer.

Verify gates (binding):
1. NPC harness PASSES at NP={1,2,4,8} with DFlash bound.
2. New multi-slot harness shows np=1 → np=8 byte-identical per slot.
3. Bench at npp=200 ntg=64 npl=8 with `--spec dflash --draft 4`
   produces non-degenerate t/s against
   `data/phase_dflash_t8/bench-spec-{none,mtp}.json`.

## Key risks

1. **cb_eval seq_id demux (Phase 3)** is structurally the hardest
   change. If the row→seq_id information isn't easily plumbed through
   the existing decode path, this phase grows significantly.

2. **n_seq_max at bind time vs decode time.** `llama_set_dflash` is
   called once when the spec adapter is constructed. If the server
   later changes n_parallel, the scratch allocation can't grow. The
   bind needs to size for max-anticipated n_slots.

3. **MAL semantics at multi-slot.** Currently `MAL = anchor_pos =
   prompt length`. At np>1 each slot has its own anchor_pos, so MAL
   becomes a per-slot quantity. The combine_features and
   inject_kv_fused kernels need to be checked for per-slot MAL
   handling vs. shared.

## Files NOT in scope

- DFlash kernel optimization (lm_head + GEMM rewrites). Future work
  per PHASE_DFLASH item 3.
- Server profile flip. Production stays on current deterministic
  profile until multi-slot DFlash validates.
- NPC framework or `scripts/verify-production-determinism.sh`.
