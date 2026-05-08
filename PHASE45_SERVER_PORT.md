# PHASE45 Server Port Plan (D10)

**Target:** Complete migration of `examples/server/` from `llama_context` to the new `(session, decoder, spec_loop)` architecture.

**Status:** Planning phase. This document serves as a callsite-by-callsite checklist for D10's migration work.

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total callsites in server/** | 127+ API calls across 4 files |
| **Heaviest user** | `server-context.cpp` (140 calls, ~50% of server work) |
| **Architecture mapping** | One shared session per tenant; one verify + one draft decoder per session |
| **Estimated LoC delta** | +150–250 lines (new API calls, decoder/session lifecycle) |
| **Risk level** | High (multi-slot state, spec checkpoint integration, slot lifecycle) |

---

## 1. Per-File Callsite Inventory

### 1.1 `server-context.cpp` (140 callsites — HEAVY MIGRATION)

This is the main server context management file. All callsites below are critical for the port.

#### Session / KV Management (36 calls → `llama_session_*`)

| Line | Call | Purpose | New API Replacement |
|---|---|---|---|
| 122 | `llama_n_ctx(ctx)` | Get context size | `llama_session_n_ctx(session)` |
| 324 | `llama_n_batch(ctx)` | Get batch size | `llama_session_n_batch(session)` |
| 327 | `llama_batch_init(...)` | Allocate batch | **Same API**, borrow session ctx/n_batch |
| 393 | `llama_state_seq_get_size(ctx, id, 0)` | Measure seq state | `llama_session_state_seq_get_size(session, id)` |
| 403 | `llama_state_seq_get_data(ctx, ...)` | Load seq state | `llama_session_state_seq_get_data(session, ...)` |
| 547 | `llama_decode_reset()` | Reset decoder | `llama_decoder_synchronize(decoder)` (context); no "reset" in new API |
| 1083 | `llama_n_ctx(ctx)` | Penalty last-N init | `llama_session_n_ctx(session)` |
| 1087 | `llama_n_ctx(ctx)` | Dry penalty last-N | `llama_session_n_ctx(session)` |
| 1693 | `llama_kv_cache_clear(ctx)` | Clear KV | `llama_session_kv_clear(session)` |
| 1708 | `llama_n_batch(ctx)` | Decode batch size | `llama_session_n_batch(session)` |
| 1728 | `llama_kv_cache_seq_cp(ctx, 0, i, ...)` | Copy seq KV | `llama_session_kv_seq_cp(session, 0, i, ...)` |
| 2581 | `llama_get_kv_cache_token_count(ctx)` | KV usage stats | `llama_session_get_kv_cache_token_count(session)` (**NEW**) |
| 2582 | `llama_get_kv_cache_used_cells(ctx)` | KV cell usage | `llama_session_get_kv_cache_used_cells(session)` (**NEW**) |
| 2615 | `llama_state_seq_save_file(ctx, ...)` | Save seq to disk | `llama_session_state_seq_save_file(session, ...)` |
| 2656 | `llama_state_seq_load_file(ctx, ...)` | Load seq from disk | `llama_session_state_seq_load_file(session, ...)` |
| 2699 | `llama_kv_cache_seq_rm(ctx, slot->id, ...)` | Remove seq KV | `llama_session_kv_seq_rm(session, slot->id, ...)` |
| 2716 | `llama_lora_adapters_apply(ctx, ...)` | Apply LoRA | `llama_session_lora_adapter_set(session, ...)` (**multi-call**) |
| 2730 | `llama_n_layer(model)` | Layer count | Accessor on model (unchanged) |
| 2758 | `llama_control_vector_load(...)` | Load CV | Keep as-is (returns CV data struct) |
| 2770 | `llama_model_n_embd(model)` | Embedding dim | Unchanged (model accessor) |
| 2877 | `llama_control_vector_apply(ctx, ...)` | Apply CV (clear) | `llama_session_control_vector_apply(session, ...)` |
| 2911 | `llama_control_vector_apply(ctx, ...)` | Apply CV (set) | `llama_session_control_vector_apply(session, ...)` |
| 2957 | `llama_kv_cache_seq_pos_min(slot.ctx, ...)` | Min seq pos | `llama_session_kv_seq_pos_min(session, ...)` (**NEW**) |
| 2958 | `llama_kv_cache_seq_pos_max(slot.ctx, ...)` | Max seq pos | `llama_session_kv_seq_pos_max(session, ...)` |
| 2959 | `llama_kv_cache_seq_rm(ctx, ...)` | Remove slot KV | `llama_session_kv_seq_rm(session, ...)` |
| 2960 | `llama_kv_cache_seq_add(ctx, ...)` | Shift seq KV | `llama_session_kv_seq_add(session, ...)` |
| 3825 | `llama_kv_cache_seq_div(...)` | Div seq KV | `llama_session_kv_seq_div(session, ...)` |

#### Decoder / Forward Pass (24 calls → `llama_decoder_*`)

| Line | Call | Purpose | New API Replacement |
|---|---|---|---|
| 278 | `llama_set_embeddings(ctx, true)` | Enable embedding mode | `llama_decoder_set_embeddings(decoder, true)` |
| 547 | `llama_decode_reset()` | Reset decode state (**BLOCKER**) | No direct replacement; handled in spec_loop or per-decoder |
| 1720 | `llama_decode(ctx, batch)` | Main forward pass | `llama_decoder_decode(decoder, batch)` |
| 1721 | (error log) | Error handling | Same (no API change) |
| 2188 | `llama_get_embeddings_ith(ctx, i)` | Extract embedding | `llama_decoder_get_embeddings_ith(decoder, i)` |
| 2191 | `llama_get_embeddings_seq(ctx, seq_id)` | Extract seq embedding | `llama_decoder_get_embeddings_seq(decoder, seq_id)` |
| 3185 | `llama_set_draft_input_hidden_state(hs_ctx, ...)` | MTP hidden state | `llama_decoder_set_draft_input_hidden_state(draft_decoder, ...)` (**decoder-local**) |
| 3188 | `llama_get_embeddings_ith(ctx, -1)` | Get last embedding | `llama_decoder_get_embeddings_ith(decoder, -1)` |

**Additional decoder calls (from grep output):**

| Line | Call | Purpose | New API Replacement |
|---|---|---|---|
| 1891 | `llama_n_vocab(llama_get_model(ctx))` | Vocab size accessor | `llama_decoder_model(decoder)` then `llama_n_vocab()` |
| 3380 | `llama_set_fast_argmax_for_verify(ctx, ...)` | MTP verify tuning | `llama_decoder_set_fast_argmax_for_verify(verify_decoder, ...)` (**NEW**) |
| 3632 | `llama_set_draft_input_chain_residual(ctx, ...)` | MTP chain residual | `llama_decoder_set_draft_input_chain_residual(draft_decoder, ...)` |

#### Speculative Checkpoint (6 calls → `llama_spec_loop_*` / decoder lifecycle)

| Line | Call | Purpose | New API Replacement |
|---|---|---|---|
| 39 | `llama_spec_ckpt_discard(ctx)` | Discard spec checkpoint | **FOLD into decoder/spec_loop.** Checkpoints are managed by spec_loop on accept/reject. |
| 48 | `llama_spec_ckpt_init(ctx, ckpt_mode, ...)` | Init spec checkpoint | **FOLD into spec_loop.** Spec_loop handles this internally. |
| 54 | `llama_spec_ckpt_save(ctx, slot.id)` | Save checkpoint | **FOLD into spec_loop.** Managed by `llama_spec_loop_step()` internally. |
| 56 | `llama_spec_ckpt_discard(ctx)` | Discard (error path) | **FOLD into decoder lifecycle.** |
| 3804 | `llama_spec_ckpt_restore(ctx, ...)` | Restore checkpoint | **FOLD into spec_loop.** Called after `llama_spec_loop_step()` on mismatch. |

**BLOCKER:** `llama_spec_ckpt_*` is a legacy per-context checkpoint API. The new design moves checkpointing into `llama_spec_loop`'s internal state machine. **Decision needed:** Either keep checkpoint stubs for compatibility during D10, or rewrite the spec_loop integration in server-context.cpp to use spec_loop's accept/reject directly.

#### Model & Vocab (47 calls → `llama_model_*` + **pass-through**)

| Line | Call | Purpose | New API Replacement |
|---|---|---|---|
| 77 | `llama_free_model(model)` | Free model | Same API (model is model) |
| 88 | `llama_free_model(model_draft)` | Free draft model | Same API |
| 124 | `llama_should_add_bos_token(model)` | Check BOS | Same (model accessor) |
| 125 | `llama_add_eos_token(model)` | Check EOS | Same (model accessor) |
| 192 | `llama_model_load_from_file(...)` | Load model | Same API |
| 204 | `llama_model_n_nextn_layer(model)` | Check MTP layer | Same (model accessor) |
| 260 | `llama_model_n_nextn_layer(model)` | Check MTP layer (again) | Same (model accessor) |
| 970 | `llama_vocab_n_tokens(vocab)` | Token count | Same (vocab accessor) |
| 989 | `llama_model_get_vocab(model)` | Get vocab | Same (model accessor) |
| 1202 | `llama_n_vocab(model)` | Vocab size | Same (model accessor) |
| ... | (many similar) | Model/vocab accessors | **All pass-through; no changes** |

**All model APIs remain unchanged.** They are read-only accessors on `llama_model`.

#### Batch Management (6 calls → **pass-through**)

| Line | Call | Purpose | New API Replacement |
|---|---|---|---|
| 102 | `llama_batch_free(slot.batch_spec)` | Free batch | Same API |
| 105 | `llama_batch_free(batch)` | Free batch | Same API |
| 274 | `llama_batch_init(...)` | Alloc batch | Same API (batch is bag of tokens) |
| 327 | `llama_batch_init(...)` | Alloc batch | Same API |
| 3835 | `llama_batch_init(...)` | Alloc batch | Same API |
| 3836 | `llama_batch_free(...)` | Free batch | Same API |

**Batch APIs remain unchanged.** Batches are token input structures, not context-dependent.

#### Token Utilities (9 calls → **pass-through**)

| Line | Call | Purpose | New API Replacement |
|---|---|---|---|
| 1397 | `llama_token_eos(model)` | Get EOS token | Same (model accessor) |
| 1762 | `llama_token_is_eog(model, tok)` | Check EOG | Same (model-dependent) |
| 1805 | `llama_token_is_eog(model, tok)` | Check EOG | Same (model-dependent) |
| 1844 | `llama_token_is_eog(model, tok)` | Check EOG | Same (model-dependent) |
| 1934 | `llama_token_eos(model)` | Get EOS token | Same (model accessor) |
| ... | (similar) | Token utility calls | **All pass-through; no changes** |

**All token APIs remain unchanged.** They are model-dependent utilities.

#### Sampling (3 calls → **pass-through**)

| Line | Call | Purpose | New API Replacement |
|---|---|---|---|
| 1647 | `llama_sampling_types_from_names(...)` | Parse sampler config | Same (utility, not context-dependent) |
| 1650 | `llama_sampling_types_from_chars(...)` | Parse sampler config | Same (utility) |
| 1940 | `llama_sampling_type_to_str(...)` | Format sampler type | Same (utility) |

**Sampling utilities remain unchanged.** PHASE45 explicitly does not move sampling state into session/decoder.

#### Lifecycle / Misc (9 calls → **refactor**)

| Line | Call | Purpose | New API Replacement |
|---|---|---|---|
| 72 | `llama_free(ctx)` | Free main context | `llama_session_free(session)` + `llama_decoder_free(decoder)` |
| 84 | `llama_free(ctx_draft)` | Free draft context | `llama_session_free(draft_session)` (**if dual-model**) or **FOLD into shared session** |
| 99 | `llama_free(slot.ctx_dft)` | Free slot draft | **DELETE** (slot-local draft removed in new arch) |
| 111 | `llama_init_from_gpt_params(...)` | Init context | Split: `llama_session_create()` + `llama_decoder_create()` |
| 2187 | `llama_pooling_type(slot.ctx)` | Check pooling mode | `llama_decoder_get_pooling_type(decoder)` (**NEW?**) or direct model accessor |
| 2202 | `llama_pooling_type(slot.ctx)` | Check pooling mode | Same |
| 3183 | `llama_get_model(ctx)` | Extract model | `llama_session_model(session)` or `llama_decoder_model(decoder)` |
| 3190 | `llama_get_model(ctx)` | Extract model | Same |
| 4364 | `llama_get_model(ctx)` | Extract model | Same |

---

### 1.2 `server.cpp` (11 callsites — LIGHT MIGRATION)

| Line | Call | Purpose | New API Replacement |
|---|---|---|---|
| ~50 | `llama_backend_init()` | Backend init | Same API (global; stays at program start) |
| ~N | `llama_backend_free()` | Backend cleanup | Same API |
| ~N | `llama_numa_init(...)` | NUMA init | Same API (global configuration) |
| ~N | `llama_print_system_info()` | Print system info | Same API (diagnostic utility) |
| ~N | `llama_decode_stop()` | Stop decode threads | **DELETE** or **NOP** (new arch doesn't have per-context threads) |
| ~N | `llama_token_bos(model)` | Get BOS token | Same (model accessor) |
| ~N | `llama_token_eos(model)` | Get EOS token | Same (model accessor) |
| ~N | `llama_get_vocab(model)` | Get vocab | Same (model accessor) |
| ~N | `llama_model_meta_val_str(model, ...)` | Get model metadata | Same (model accessor) |
| ~N | `llama_pooling_type(ctx)` | Check pooling (**REFACTOR**) | Extract model from decoder: `llama_decoder_model()` |

**Risk:** `llama_decode_stop()` is a per-context thread-management API. In the new architecture, threads are per-decoder (owned by `llama_decoder_params.n_threads`). **Decision:** Likely DELETE this call; decoders manage their own threads.

---

### 1.3 `server-common.cpp` (8 callsites — TRIVIAL)

| Line | Call | Purpose | New API Replacement |
|---|---|---|---|
| ~N | `llama_get_logits_ith(ctx, i)` | Extract logit | `llama_decoder_get_logits_ith(decoder, i)` |
| ~N | `llama_get_model(ctx)` | Extract model | `llama_decoder_model(decoder)` or `llama_session_model(session)` |
| ~N | `llama_model_get_vocab(model)` | Get vocab | Same (model accessor) |
| ~N | `llama_n_vocab(model)` | Vocab size | Same (model accessor) |
| ~N | `llama_tokenize(model, ...)` | Tokenize | Same API (model operation) |
| ~N | `llama_token_to_piece(vocab, tok)` | Decode token | Same API (vocab operation) |
| ~N | `llama_n_batch(ctx)` | Batch size | `llama_session_n_batch(session)` |
| ~N | `llama_vocab_n_tokens(vocab)` | Token count | Same (vocab accessor) |

**All trivial:** Most are model/vocab accessors (unchanged). Only `llama_get_logits_ith` and `llama_n_batch` need decoder/session substitutions.

---

### 1.4 `server-task.cpp` (1 callsite — TRIVIAL)

| Line | Call | Purpose | New API Replacement |
|---|---|---|---|
| ~N | `llama_state_seq_set_data(ctx, ...)` | Restore seq state | `llama_session_state_seq_set_data(session, ...)` |

**Trivial:** Single call. Part of `server_task` state persistence.

---

## 2. Slot ↔ Session ↔ Decoder Mapping

### Today's Architecture (PHASE44)

```
┌─────────────────────┐
│   server_context    │
│   - ctx (master)    │
│   - model           │
│   - batch           │
│   - slots[]         │
└─────────────────────┘
         │
         ▼
    ┌─────────────────────┐
    │  slot[0..N-1]       │
    │  - ctx (alias)      │
    │  - cache_tokens[]   │
    │  - n_past           │
    │  - batch_spec       │
    │  - ctx_dft (draft)  │
    └─────────────────────┘
```

### PHASE45 Architecture (D10 Target)

Per PHASE45.md's locked decision:

```
┌──────────────────────────────────────┐
│      server_context                  │
│      - session (shared)               │  <- ONE session for all slots
│      - model                          │
│      - verify_decoder                 │  <- ONE verify decoder
│      - draft_decoder                  │  <- ONE draft decoder
│      - batch (primary)                │
│      - spec_loop                      │  <- Orchestrates verify+draft
│      - slots[]                        │
└──────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│   slot[0..N-1]                       │
│   - seq_id (0..N-1)                  │ <- seq_id partition in shared KV
│   - cache_tokens[]                   │ <- still per-slot (seq_id-scoped)
│   - n_past                           │ <- per-slot sequence position
│   - batch_spec                       │ <- per-slot spec batch
├──────────────────────────────────────┤
│   (NO slot.ctx, slot.ctx_dft)        │ <- DELETED
└──────────────────────────────────────┘
```

### Critical Mapping Details

#### Session Sharing

- **Old:** Each slot has `slot.ctx` (one context per slot).
- **New:** One `server_context.session` is shared by all slots.
- **Why:** KV cache is expensive (~90% of VRAM). PHASE45 exists to avoid per-slot duplication.
- **Implication:** `llama_session.n_seq_max = slots.size()`. Each slot is a seq_id in the shared session.

#### Decoder Pair

- **Old:** Master context has a full forward; draft contexts (`ctx_dft`, `ctx_draft`) run separate forward passes.
- **New:**
  - `verify_decoder` (role=VERIFY) — full forward on shared session; batches all slots' seq_ids.
  - `draft_decoder` (role=DRAFT_MTP) — MTP head forward on shared session; batches all slots' seq_ids.
  - Both use `llama_decoder_decode(decoder, batch)` with multi-seq_id batches.

#### Slot State Transformation

| Slot Field | Old | New | Notes |
|---|---|---|---|
| `slot.ctx` | `llama_context *` | **DELETED** | Use `server_context.session` + decoder pair |
| `slot.ctx_dft` | Draft context (per-slot) | **DELETED** | Use shared `server_context.draft_decoder` + seq_id batching |
| `slot.seq_id` | **Not tracked** | `llama_seq_id` (0..N-1) | New: each slot is a seq partition in the session |
| `slot.cache_tokens[]` | Per-slot KV metadata | **Unchanged** | Still tracks which tokens are in KV for this seq_id |
| `slot.n_past` | Per-slot position | **Unchanged** | Still sequence position (within this seq_id's KV range) |
| `slot.batch_spec` | Per-slot spec batch | **Unchanged** | Still needed for per-slot speculative decoding |

#### Initialization Changes

**Old:**
```cpp
for (auto& slot : slots) {
    // slot.ctx = llama_new_context_with_model(model, params);  // per-slot
}
```

**New:**
```cpp
// ONE session for all slots
session = llama_session_create(model, {
    .n_ctx = params.n_ctx,
    .n_seq_max = slots.size(),      // <-- # of slots becomes session capacity
    .n_batch = params.n_batch,
    ...
});

// ONE verify decoder, ONE draft decoder
verify_decoder = llama_decoder_create(session, {
    .role = LLAMA_DECODER_VERIFY,
    .n_threads = params.n_threads,
    ...
});

draft_decoder = llama_decoder_create(session, {
    .role = LLAMA_DECODER_DRAFT_MTP,
    .n_threads = params.n_threads,
    ...
});

// NO per-slot context allocation
for (auto& slot : slots) {
    slot.seq_id = slot_index;  // <-- slot is now a seq_id partition
}
```

#### KV Operation Changes

| Operation | Old | New |
|---|---|---|
| Clear all KV | `llama_kv_cache_clear(ctx)` | `llama_session_kv_clear(session)` |
| Copy slot KV | `llama_kv_cache_seq_cp(ctx, src, dst, ...)` | `llama_session_kv_seq_cp(session, src, dst, ...)` |
| Remove slot KV | `llama_kv_cache_seq_rm(ctx, slot.id, ...)` | `llama_session_kv_seq_rm(session, slot.id, ...)` |
| Shift slot KV | `llama_kv_cache_seq_add(ctx, ...)` | `llama_session_kv_seq_add(session, ...)` |
| Save slot state | `llama_state_seq_save_file(ctx, slot.id, ...)` | `llama_session_state_seq_save_file(session, slot.id, ...)` |
| Load slot state | `llama_state_seq_load_file(ctx, slot.id, ...)` | `llama_session_state_seq_load_file(session, slot.id, ...)` |

---

## 3. Speculative Decoding & Spec-Loop Integration

### Current Spec Infrastructure (server-context.cpp)

The server uses **per-step checkpointing** (PHASE36 era) to implement speculative acceptance:

```cpp
save_speculative_checkpoint(slot, model, ctx, mode);  // lines 42–68
  → llama_spec_ckpt_init(ctx, mode, max_tokens)
  → llama_spec_ckpt_save(ctx, slot.id)
  → slot.spec_ckpt.sampler = common_sampler_init(...)

// Later, on spec loop:
for (int step = 0; step < num_speculative_steps; ++step) {
    llama_decode(ctx, batch);  // draft forward
    llama_decode(ctx, batch);  // verify forward
    if (verify_agrees) {
        // accept tokens
    } else {
        discard_speculative_checkpoint(slot, ctx);  // lines 37–40
        llama_spec_ckpt_restore(ctx, slot.id, ...);  // line 3804
    }
}
```

### PHASE45's Spec Loop (D10 Target)

The **functional body** of speculative orchestration moves into `llama_spec_loop`:

```cpp
// NEW: Create spec_loop with verify + draft decoders
spec_loop = llama_spec_loop_create(verify_decoder, &draft_decoder, 1, {
    .sampler = sampling_chain,
    .min_chain_prob = 0.5,
    .max_draft_depth = mtp_steps,
});

// Main loop: one step = forward for all slots + accept/reject
for (each slot in slots) {
    // Prepare batch with slot.seq_id
    int n_accepted = llama_spec_loop_step(spec_loop, batch);
    
    // Accepted tokens are returned by:
    const llama_token * tokens;
    llama_spec_loop_last_accepted(spec_loop, &n_accepted);
    slot.drafted.assign(tokens, tokens + n_accepted);
}

// Cleanup
llama_spec_loop_free(spec_loop);
```

### Migration Path for Checkpoint Code

**Current state (lines 37–68, 3804):**
- `save_speculative_checkpoint()` — initializes per-step checkpoint, saves sampler.
- `discard_speculative_checkpoint()` — cleans up on mismatch.
- `llama_spec_ckpt_restore()` — rolls back KV on rejection.

**Problem:** `llama_spec_ckpt_*` is a legacy per-context API. The new `llama_spec_loop` handles checkpointing internally via `llama_kv_txn` (KV transactions).

**Solution (pick one):**

1. **Rewrite to use `llama_spec_loop` directly:**
   - Delete `save_speculative_checkpoint()` / `discard_speculative_checkpoint()`.
   - Use `llama_spec_loop_step()` which returns accepted tokens and handles rollback.
   - Cost: ~100 LoC rewrite of the main spec loop in `server_context::process_token_by_slot_step()`.

2. **Keep checkpoint stubs for D10, remove in D11:**
   - Implement dummy `llama_spec_ckpt_*` wrappers that NOP or wrap spec_loop.
   - Allows phased migration; riskier (adds shim code).

**Recommendation:** **Rewrite to spec_loop.** The spec loop integration is architecturally cleaner and `spec_ckpt` is legacy code that doesn't survive D11 anyway.

### Spec-Loop Callsites to Rewrite

| Current Code | New Code | File:Line |
|---|---|---|
| `llama_spec_ckpt_init()` | Delete; move to spec_loop init | 48 |
| `llama_spec_ckpt_save()` | Delete; handled by spec_loop | 54 |
| `llama_spec_ckpt_discard()` | Delete | 39, 56 |
| `llama_spec_ckpt_restore()` | Delete; rollback handled by spec_loop accept/reject | 3804 |
| `save_speculative_checkpoint()` | Delete entire function (lines 42–68) | 42–68 |
| `discard_speculative_checkpoint()` | Delete entire function (lines 37–40) | 37–40 |

---

## 4. MTP-Specific Callsites

The server implements **inline MTP** (PHASE39) with per-step checkpointing. D10 maps this to the new `LLAMA_DECODER_DRAFT_MTP` role.

### MTP Decoder Setup (NEW in D10)

```cpp
// Lines 204, 260: Check for MTP layer
if (llama_model_n_nextn_layer(model) > 0) {
    // Create DRAFT_MTP decoder
    draft_decoder = llama_decoder_create(session, {
        .role = LLAMA_DECODER_DRAFT_MTP,
        .mtp_fused_n_steps = params.speculative.n_tokens,
        .mtp_inline_kv_hook = true,  // PHASE36 Step 3
        ...
    });
} else {
    // Single-model speculative (tree drafting, etc.)
    draft_decoder = llama_decoder_create(session, {
        .role = LLAMA_DECODER_DRAFT_MTP,
        ...
    });
}
```

### MTP Callsites in server-context.cpp

| Line | Call | Purpose | New API / Decoder Parameter |
|---|---|---|---|
| 204 | `llama_model_n_nextn_layer(model)` | Detect MTP layer | Same (model accessor) |
| 260 | `llama_model_n_nextn_layer(model)` | Conditional on MTP | Same (model accessor) |
| 274 | `llama_batch_init(..., n_max + 1)` | Allocate spec batch | Same API (batch structure) |
| 3185 | `llama_set_draft_input_hidden_state(ctx, ...)` | Set draft hidden state | `llama_decoder_set_draft_input_hidden_state(draft_decoder, ...)` |
| 3380 | `llama_set_fast_argmax_for_verify(ctx, ...)` | Verify optimization (**NEW in headers**) | `llama_decoder_set_fast_argmax_for_verify(verify_decoder, bool)` |
| 3632 | `llama_set_draft_input_chain_residual(ctx, ...)` | Draft chain residual | `llama_decoder_set_draft_input_chain_residual(draft_decoder, ...)` |

### PHASE39 Inline MTP Wrapping

Per PHASE45.md §Architectural decisions:

- **DRAFT_MTP decoder** builds the inline MTP head (same path as PHASE39 ported upstream).
- **VERIFY decoder** builds layers 0..N-2 (full transformer, excluding MTP).
- **Layer N-1 (MTP head K/V)** is written exclusively by draft decoder.
- **No INLINE_KV hook needed** in the new architecture (single-writer per layer).

**Action:** Set `llama_decoder_params.mtp_inline_kv_hook = true` (PHASE36 mode) when draft_decoder role is DRAFT_MTP, if the model has MTP.

---

## 5. Risks & Open Questions

### HIGH-RISK BLOCKERS

#### 1. **Spec Checkpoint API (PHASE36 legacy)**
   - **Issue:** `llama_spec_ckpt_*` is a per-context state machine for rollback. The new architecture moves this into `llama_spec_loop`'s internal KV transactions.
   - **Impact:** Lines 37–68, 3804 need full rewrite OR compatibility shims.
   - **Mitigation:** Rewrite to use `llama_spec_loop_step()` (recommended) or provide stub implementations.
   - **Effort:** ~100 LoC refactor + testing.

#### 2. **`llama_decode_reset()` → no direct replacement**
   - **Issue:** Line 547 calls `llama_decode_reset()` to reset decoder state between sequences. In the new architecture, decoders hold recurrent state per-step, not globally.
   - **Impact:** Spec-loop manages stepping; may need to replace with `llama_spec_loop_step()` or omit if spec_loop resets internally.
   - **Mitigation:** Check `llama_spec_loop` implementation; likely DELETE this call.
   - **Effort:** Investigate behavior in D6 implementation.

#### 3. **Slot-local draft context (`slot.ctx_dft`)**
   - **Issue:** Today, each slot can have a per-slot draft context for independent spec loops. New arch shares one draft decoder.
   - **Impact:** Lines 99 (free), 3166 (TODO comment) — multi-slot spec decoding orchestration changes.
   - **Mitigation:** Use seq_id batching: one draft_decoder processes all slots' seq_ids in one batch.
   - **Effort:** Refactor main decode loop to batch multi-slot specs.

#### 4. **`llama_decode_stop()` in server.cpp**
   - **Issue:** Thread management API. New decoders own threads via `llama_decoder_params.n_threads`.
   - **Impact:** server.cpp cleanup code may hang if not removed.
   - **Mitigation:** DELETE or NOP this call.
   - **Effort:** 1 line.

### MEDIUM-RISK GAPS

#### 5. **Pooling type accessor**
   - **Issue:** Lines 2187, 2202 call `llama_pooling_type(ctx)` to check embedding pooling mode. This is a per-context setting in the old API.
   - **Impact:** New API: pooling is a model property, not a session/decoder override.
   - **Mitigation:** Check if `llama_pooling_type()` accepts model directly, or add `llama_decoder_get_pooling_type()`.
   - **Effort:** Investigate headers; likely trivial.

#### 6. **`llama_get_kv_cache_token_count()` / `llama_get_kv_cache_used_cells()`**
   - **Issue:** Lines 2581–2582 extract KV usage stats for monitoring. New API must expose these on session.
   - **Impact:** Stats are needed for `/slots` endpoint response.
   - **Mitigation:** Add `llama_session_get_kv_cache_token_count()` and `llama_session_get_kv_cache_used_cells()` if missing.
   - **Effort:** ~2 function stubs in llama-session.h + impl.

#### 7. **`llama_kv_cache_seq_pos_min()` (NEW function)**
   - **Issue:** Line 2957 calls `llama_kv_cache_seq_pos_min()` which may not exist in the new API.
   - **Impact:** Context-shifting logic needs min position to decide how much KV to shift.
   - **Mitigation:** Add `llama_session_kv_seq_pos_min()` if missing.
   - **Effort:** ~1 function stub.

### LOW-RISK CLEANUP

#### 8. **`llama_init_from_gpt_params()` decomposition**
   - **Issue:** Line 111 uses the monolithic init function. New API requires separate session + decoder creation.
   - **Impact:** ~20-line refactor at server initialization.
   - **Mitigation:** Refactor `server_context::load_model()` to call `llama_session_create()` + `llama_decoder_create()`.
   - **Effort:** ~30 LoC.

#### 9. **Draft model support**
   - **Issue:** Lines 192, 204–260 handle optional separate draft model. New arch must create separate session for draft if model differs.
   - **Impact:** Dual-model spec decoding needs two sessions (one for verify, one for draft).
   - **Mitigation:** Check if PHASE45 allows dual sessions; likely yes, but requires careful lifetime management.
   - **Effort:** Moderate (~50 LoC).

#### 10. **Control vector per-slot tracking**
   - **Issue:** Lines 2758–2877, 2911 load/apply control vectors. New API: control vectors apply to session (shared across slots).
   - **Impact:** Per-slot control vector application is no longer possible. All slots use the same CV or none.
   - **Mitigation:** Move CV state to `server_context` level (not per-slot). Document as behavior change.
   - **Effort:** ~20 LoC refactor + documentation.

---

## 6. Estimated LoC Delta

| Phase | Additions | Deletions | Net | Notes |
|---|---|---|---|---|
| **Initialization** | +50 | -20 | +30 | Session + decoder creation; delete per-slot context alloc |
| **Decode loop** | +40 | -30 | +10 | Spec_loop integration; remove per-slot spec_ckpt calls |
| **KV ops** | +15 | -15 | 0 | API name changes (ctx→session); logic unchanged |
| **Decoder output** | +10 | -10 | 0 | API name changes (ctx→decoder) |
| **Slot cleanup** | +5 | -40 | -35 | Remove slot.ctx, slot.ctx_dft, related fields |
| **Spec checkpoint removal** | 0 | -100 | -100 | Delete save_speculative_checkpoint(), discard_speculative_checkpoint() |
| **SUBTOTAL** | **+120** | **-215** | **-95** | Net code reduction (spec_ckpt removal is large) |

**Overall estimate: -95 to +50 LoC (depending on spec_loop implementation choices).**

If spec_loop integration is tight, the server code shrinks. If compatibility shims are added, code grows.

---

## 7. Port Checklist (D10 Execution)

### Phase 1: Setup & Headers (1 session)
- [ ] Verify all four headers (`llama-session.h`, `llama-decoder.h`, `llama-kv-txn.h`, `llama-spec-loop.h`) are in `/include/` and compile.
- [ ] Check for missing accessors:
  - [ ] `llama_session_get_kv_cache_token_count()` (line 2581)
  - [ ] `llama_session_get_kv_cache_used_cells()` (line 2582)
  - [ ] `llama_session_kv_seq_pos_min()` (line 2957)
  - [ ] `llama_decoder_set_fast_argmax_for_verify()` (line 3380)
- [ ] Add missing accessors to headers if needed.

### Phase 2: Refactor `server_context` Initialization (1 session)
- [ ] Replace `llama_init_from_gpt_params()` with:
  - [ ] `llama_session_create()` (shared across all slots)
  - [ ] `llama_decoder_create(..., VERIFY)` (verify decoder)
  - [ ] `llama_decoder_create(..., DRAFT_MTP)` (draft decoder, if MTP model)
- [ ] Delete per-slot `llama_new_context_with_model()` loops.
- [ ] Add `slot.seq_id` initialization (slot index → seq_id mapping).
- [ ] Test single-slot forward pass.

### Phase 3: Refactor KV & Session Operations (1 session)
- [ ] Map all `llama_kv_cache_*` calls to `llama_session_kv_*`.
- [ ] Map all `llama_state_seq_*` calls to `llama_session_state_seq_*`.
- [ ] Map all `llama_n_ctx/batch/ubatch(ctx)` to `llama_session_*`.
- [ ] Map `llama_lora_adapters_apply(ctx, ...)` to per-adapter `llama_session_lora_adapter_set()` calls.
- [ ] Map `llama_control_vector_apply(ctx, ...)` to `llama_session_control_vector_apply()`.
- [ ] Test slot save/load and KV operations.

### Phase 4: Refactor Decoder & Output (1 session)
- [ ] Replace `llama_decode(ctx, batch)` with `llama_decoder_decode(verify_decoder, batch)` (main path).
- [ ] Replace `llama_set_embeddings(ctx, ...)` with `llama_decoder_set_embeddings(verify_decoder, ...)`.
- [ ] Replace `llama_get_embeddings_ith(ctx, ...)` with `llama_decoder_get_embeddings_ith(decoder, ...)`.
- [ ] Replace `llama_get_embeddings_seq(ctx, ...)` with `llama_decoder_get_embeddings_seq(decoder, ...)`.
- [ ] Add MTP decoder calls:
  - [ ] `llama_decoder_set_draft_input_hidden_state(draft_decoder, ...)`
  - [ ] `llama_decoder_set_draft_input_chain_residual(draft_decoder, ...)`
  - [ ] `llama_decoder_set_fast_argmax_for_verify(verify_decoder, ...)`
- [ ] Test embedding extraction and output buffers.

### Phase 5: Spec-Loop & Checkpoint Rewrite (1 session — **HIGHEST RISK**)
- [ ] Delete `save_speculative_checkpoint()` and `discard_speculative_checkpoint()` functions.
- [ ] Delete all `llama_spec_ckpt_*` calls (lines 39, 48, 54, 56, 3804).
- [ ] Implement `llama_spec_loop_create()` in `server_context::load_model()`.
- [ ] Rewrite main spec loop in `process_token_by_slot_step()` to use `llama_spec_loop_step()`.
- [ ] Replace spec checkpoint restoration with spec_loop's built-in accept/reject.
- [ ] Delete `llama_decode_reset()` call or replace with decoder synchronization.
- [ ] Test single-slot spec decoding (single draft depth).
- [ ] Test multi-slot batched decoding via seq_id partitions.

### Phase 6: Cleanup & Removal (1 session)
- [ ] Delete all slot-local context fields: `slot.ctx`, `slot.ctx_dft`.
- [ ] Delete `server_context.ctx_draft`, `server_context.model_draft` (or refactor for dual-model case).
- [ ] Remove `llama_decode_stop()` call from `server.cpp`.
- [ ] Update `server-context.h` struct definitions.
- [ ] Test full multi-slot server startup and serving.

### Phase 7: Integration & E2E Testing (1 session)
- [ ] Full server startup with multi-slot config (e.g., `--parallel 3`).
- [ ] Single-turn completion: verify output byte-identical to old API.
- [ ] Multi-turn conversation: verify slot state persistence (save/load).
- [ ] Spec decoding: verify acceptance rate ≥ PHASE44 baseline.
- [ ] Multi-slot under load: verify no OOM, no race conditions.
- [ ] Performance profiling: verify no regression vs. PHASE44.

---

## 8. Summary Table

| Metric | Count | Status |
|--------|-------|--------|
| **Total callsites** | 127 | Inventoried |
| **Session/KV API calls** | 36 | Mapped to `llama_session_*` |
| **Decoder API calls** | 24 | Mapped to `llama_decoder_*` |
| **Spec-loop API calls** | 6 | Candidate for rewrite / deletion |
| **Model/vocab API calls** | 47 | Pass-through (unchanged) |
| **Batch API calls** | 6 | Pass-through (unchanged) |
| **Token utility calls** | 9 | Pass-through (unchanged) |
| **Lifecycle/misc calls** | 9 | Refactor / NOP |
| **Missing from headers** | ~3 | TBD (token_count, used_cells, seq_pos_min, fast_argmax) |
| **High-risk blockers** | 4 | Spec checkpoint, decode_reset, slot_ctx_dft, decode_stop |
| **Est. LoC delta** | -95 to +50 | Spec_ckpt removal dominates |

---

## 9. References

- `PHASE45.md` — Architecture locked; D6-D10 binding decisions.
- `PHASE45_CALLSITES.md` — D4 external callsite audit (high-level).
- `llama-session.h`, `llama-decoder.h`, `llama-spec-loop.h`, `llama-kv-txn.h` — D5 header stubs.
- `server-context.cpp` — Main target (140 callsites).
- `server.cpp`, `server-common.cpp`, `server-task.cpp` — Secondary targets (11 + 8 + 1 callsites).

---

**Author:** Callsite inventory for PHASE45 D10 server port.  
**Date:** 2026-05-08  
**Status:** Planning document; ready for execution phase.

