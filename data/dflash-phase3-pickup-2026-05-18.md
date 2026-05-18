# DFlash Phase 3 — cb_eval per-slot demux pickup brief (2026-05-18)

Continuation of `data/dflash-multi-slot-api-brief-2026-05-18.md` and
`data/dflash-multi-slot-impl-plan-2026-05-18.md`. This brief covers
Phase 3 specifically, which the previous session marked as the
architectural risk.

## State at handover

Phases 1 and 2 are landed and verified:
- Submodule commit `fa3e50c7` — DFlash Phase 2 scratch sizing.
- Submodule commit `8008feaf` — DFlash Phase 1 API surface.
- Parent commits `7d997a5`, `d708299` (submodule bumps).

`test-dflash-closure` still PASSES 8/8 prompts argmax-equivalent after
both phases (single-slot byte-identical to pre-Phase-1).
`scripts/verify-production-determinism.sh` PASSES at NP={1,2,4,8}.

## The Phase 3 problem in one paragraph

At np>1, multiple seq_ids share the target ubatch. The cb_eval hook
`llama_dflash_extract_cb_eval` at `src/llama.cpp:9661` fires once per
ubatch with a tensor `t` containing `n_rows = ubatch.n_tokens` rows of
residual hiddens. The hook appends ALL rows to a single per-source-layer
buffer (`ctx->default_decoder.dflash_extract_buf[slot]`) with no
seq_id information. The downstream consumer
(`stage_target_hiddens` at `src/llama-dflash.cpp:498`) treats the buffer
as one continuous slot's worth of MAL anchors. At np>1, rows from
different seq_ids interleave with no demux info — so each "slot"'s view
is corrupt.

Phase 3 must make the cb_eval row→seq_id mapping recoverable so the
staging step can demux per slot.

## Files to read first (sub-5k tokens)

1. **`src/llama.cpp:9655-9755`** — the cb_eval hook in full. Pay
   attention to:
   - Lines 9707-9716: the F32/F16 append path. Note that the appended
     count is `n_elements = D_emb * n_rows`, where `n_rows` is
     `t->ne[1]` for the 2D tensor case.
   - Line 9723: `dflash_extract_n[slot]` tracks total floats appended.
   - Lines 9727-9754: `llama_set_dflash_extract_layers` initializes
     `dflash_extract_buf[]` per source-layer slot. This is where
     per-slot/seq_id storage would be sized.

2. **`src/llama-decoder-internal.h:140-180`** (approx — exact line of
   the `default_decoder` struct) — the per-context decoder state.
   Currently holds `dflash_extract_buf[80]` and `dflash_extract_n[80]`.
   Phase 3 may need to add a parallel structure indexed by
   [source_layer][seq_id] or add a row→seq_id index buffer.

3. **`src/llama-dflash.cpp:495-555`** — `stage_target_hiddens`. This is
   the consumer of the extract buffer. The signature would need to
   change to know which seq_id slot to stage, or to stage all slots in
   one call.

4. **`src/llama-decoder*.cpp`** (find the file that builds ubatches and
   invokes the scheduler) — locate where `ubatch.seq_id[j]` is
   established per row. The cb_eval hook lives in `llama.cpp` but the
   ubatch is set up by the decoder driver.

   Grep: `grep -rn "ubatch.seq_id\|ubatch\.seq_id\[" src/ | head -20`

5. **`src/llama-dflash.cpp:720-755`** — `llama_dflash_trim_extract`
   (T6.C). Resizes the per-slot buffer based on a seq_rm rollback.
   At multi-slot, trim is called per-slot; the trim semantics need to
   match the new per-slot extract storage.

## Architectural design choices

### Option A — Per-seq_id parallel buffers (recommended)

```cpp
// In llama-decoder-internal.h
std::vector<float> dflash_extract_buf[80][LLAMA_MAX_PARALLEL_SEQUENCES];
size_t            dflash_extract_n  [80][LLAMA_MAX_PARALLEL_SEQUENCES];
```

cb_eval iterates rows of `t`, reading the parallel "current ubatch
seq_id per row" vector. Each row gets appended to
`dflash_extract_buf[source_layer][seq_id]`.

Pros:
- Clean separation; consumer reads per-seq_id buffer directly.
- Trim semantics (`llama_dflash_trim_extract`) naturally extend per
  seq_id.
- Matches MAL-per-slot accounting needed in Phase 4.

Cons:
- Memory blow-up scales with 80 × n_seq_max × max_context. At
  np=8 + 80-layer drafter + 4096 context × 5120 D_emb × 4B fp32 =
  ~6.7 GB. Mitigation: only the FIRST `dflash_extract_count` source-
  layer slots are actually populated (typically 5), so realistic
  footprint is 5 × 8 × 4096 × 5120 × 4 = ~3.4 GB. Still significant.
- Per-row indirection in cb_eval: needs row→seq_id map plumbed in.

### Option B — Single buffer + parallel row→seq_id index

```cpp
std::vector<float> dflash_extract_buf[80];     // unchanged
std::vector<int>   dflash_extract_row_seq[80]; // one int per APPENDED row
```

cb_eval appends all rows, plus appends `n_rows` seq_id values to the
index. Consumer (`stage_target_hiddens`) reads the index and demuxes
on the fly.

Pros:
- Lower memory cost (one int per row vs O(seq_id) buffers).
- Smaller change to cb_eval (single append vs branching by seq_id).

Cons:
- Demux happens at every stage call.
- Trim semantics get awkward — partial seq_id rm has to walk the
  index, not just shrink one buffer.

### Option C — One buffer per (source_layer, seq_id) but allocated lazily

Hybrid. Use a `std::unordered_map<seq_id, std::vector<float>>` per
source-layer slot. Only allocate when a seq_id is first seen.

Pros: avoids memory blow-up when only a few seq_ids active.

Cons: heap fragmentation, harder to reason about.

**Recommendation:** Option A. The memory cost is bounded by
`cparams.n_seq_max` (capped) and the consumer code stays clean.

## Plumbing the row→seq_id map

The cb_eval hook doesn't receive ubatch directly; only the tensor and
the user_data pointer (the llama_context). So we need the decoder to
write the current ubatch's per-row seq_ids to a context-visible field
before invoking ggml_backend_sched_compute_async, and clear it after.

Sketch:
```cpp
// llama-decoder-internal.h (default_decoder)
std::vector<llama_seq_id> dflash_ubatch_row_seq;  // length = n_tokens of current ubatch

// llama-decoder*.cpp (where decoder calls scheduler)
ctx->default_decoder.dflash_ubatch_row_seq.resize(ubatch.n_tokens);
for (uint32_t j = 0; j < ubatch.n_tokens; ++j) {
    // ubatch.seq_id is uint8_t * per row? or int *?  CHECK STRUCTURE.
    ctx->default_decoder.dflash_ubatch_row_seq[j] = ubatch.seq_id[j][0];
}
// ... schedule + compute ...
ctx->default_decoder.dflash_ubatch_row_seq.clear();

// llama.cpp cb_eval
for (int64_t row = 0; row < n_rows; ++row) {
    llama_seq_id sid = ctx->default_decoder.dflash_ubatch_row_seq[row];
    auto & buf = ctx->default_decoder.dflash_extract_buf[slot][sid];
    // append D_emb floats from t starting at row * D_emb stride
}
```

**Critical unknown:** the exact shape and field name of
`ubatch.seq_id`. In llama.cpp upstream it's `llama_ubatch::seq_id`
which is `llama_seq_id ** seq_id` (per-row pointer to a list of
seq_ids — supports a row belonging to multiple sequences in tree-spec
batching). For DFlash purposes, taking seq_id[row][0] is the right
choice (each row's primary owning sequence). Verify in
`src/llama-ubatch.h` or wherever the ubatch struct lives.

## Verification gates for Phase 3

1. **Build clean** — same set of targets as Phase 2.

2. **test-dflash-closure still PASSES 8/8** — single-slot path
   byte-identical (the closure test runs n_seq_max=1, so only seq_id=0
   buffers are touched; output must be identical to pre-Phase-3).

3. **NPC harness still PASSES** at NP={1,2,4,8} (the NPC path doesn't
   use DFlash but the binary changed; ensure no regression).

4. **New diag test** — a small test that:
   - Builds an llama_context with n_seq_max=4.
   - Calls llama_set_dflash to install the cb_eval hook.
   - Decodes a batch where rows come from different seq_ids (e.g.,
     2 seq_ids × 4 tokens each in one ubatch).
   - Inspects the extract buffers per source-layer per seq_id and
     verifies each seq_id's buffer has the correct row count.

   No need for kernel verification yet — Phase 4 handles that.

## Out-of-scope reminders for Phase 3

- Multi-slot kernel dispatch is Phase 4. Phase 3 keeps the existing
  `llama_dflash_draft` single-slot, just makes its inputs (the cb_eval
  buffer) per-seq_id-clean.
- `stage_target_hiddens` semantics: at Phase 3 it should accept a
  seq_id parameter (defaulting to 0 for the existing single-slot
  call sites) and read from
  `dflash_extract_buf[source_layer][seq_id]`. Phase 4 loops over
  seq_ids when staging multi-slot.

## How to actually start Phase 3

1. Read all 5 files listed above.
2. Confirm the ubatch `seq_id` field shape (single int per row vs
   pointer-to-list).
3. Sketch the per-row append in cb_eval against Option A's
   `dflash_extract_buf[source_layer][seq_id]` storage.
4. Add the plumbing field to `default_decoder`.
5. Find the one or two decoder call sites that drive ubatches and
   populate the per-row seq_id vector there.
6. Run test-dflash-closure to confirm no single-slot regression.
7. Write the multi-seq_id diag test described in §4 above.
8. Commit + push + parent submodule bump.
