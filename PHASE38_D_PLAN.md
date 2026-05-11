# PHASE 38 D — KV unification + multi-slot unblock

## Goal

Unblock `np=3 × 256K parallel` MTP by aliasing `ctx_mtp`'s KV cache to
`ctx_tgt`'s parent. Without this, KV doubles per slot (~13 GB extra at
np=3 × 256K) and the multi-slot configuration won't fit in 48 GB total
VRAM.

## Bench target

| Config | KV memory | Fit in 48 GB? |
|---|---:|:---:|
| np=1 × 256K, MTP off (today's production `qwen36-27b-x1.sh`) | ~4.3 GB | ✓ |
| np=3 × 256K, MTP on, no D | ~52 GB | ✗ (won't fit) |
| **np=3 × 256K, MTP on, D landed** | **~26 GB** | **✓** |

## Closure spec (from PHASE38.md, framing 3)

> Final framing (full unification + drop inline-hook on Full #2 path):
> Layer n_layer-1 (MTP layer) has a single canonical writer
> (UPDATE_ACCEPTED, mtp-shortcut-derived). Verify forward iterates layers
> 0..n_layer-2; without inline-KV-hook, verify never writes layer
> n_layer-1. Disjoint layers between concurrent dispatchers. No race
> anywhere. Single semantic class per layer.

Trade-off: re-enables UPDATE_ACCEPTED's separate decode (~5ms/cycle on
production-context), erodes -3 to -5% single-slot tg. Multi-slot wins
because doubled-KV scenario doesn't fit at all.

## Already-landed scaffolding (this session)

- `llama_cparams::parent_ctx` field (`src/llama-cparams.h:74-83`): nullable
  pointer set BEFORE `llama_init_from_model` runs. Default nullptr, so
  no behavior change.

## Implementation roadmap

### D1. Public API: `llama_link_shared_tensors`

Add to `include/llama.h` near the other init APIs (around line 614):

```c
LLAMA_API struct llama_context * llama_init_from_model_linked(
    struct llama_model           * model,
    struct llama_context_params    params,
    struct llama_context         * parent_ctx);

// Alias for the above with the user-facing name. Same semantics:
// child_ctx will be created with KV aliased to parent_ctx for
// transformer layers; parent_ctx's mtp_inline_kv_hook is forced off.
LLAMA_API struct llama_context * llama_link_shared_tensors(
    struct llama_model           * model,
    struct llama_context_params    params,
    struct llama_context         * parent_ctx);
```

In `src/llama.cpp`, near `llama_init_from_model` (line 6958):

```c
struct llama_context * llama_init_from_model_linked(
    struct llama_model * model,
    struct llama_context_params params,
    struct llama_context * parent_ctx) {

    // Stash parent_ctx so kv_cache_init can read it. We don't add a
    // public field on llama_context_params — that breaks ABI. Instead
    // pass via a thread-local or set on the freshly-constructed ctx
    // BEFORE kv_cache_init runs (see D3).

    if (!parent_ctx) {
        return llama_init_from_model(model, params);
    }

    if (&parent_ctx->model != model) {
        LLAMA_LOG_ERROR("%s: parent_ctx and model must match\n", __func__);
        return nullptr;
    }

    // Force parent's inline-KV-hook off (single-writer rule for layer N-1)
    parent_ctx->cparams.mtp_inline_kv_hook = false;

    // Fork llama_init_from_model: same body, but stash parent_ctx in
    // the new ctx's cparams BEFORE kv_cache_init. Implementation
    // option A: copy llama_init_from_model body and inject parent_ctx
    // assignment after `llama_context * ctx = new llama_context(*model);`
    // (line 7002) and before `llama_kv_cache_init(...)` call (line 7354).
    //
    // Implementation option B: refactor llama_init_from_model into
    // an internal helper that takes parent_ctx, with the existing
    // public entry as a wrapper that passes nullptr.
    // PREFERRED: option B (avoids code duplication).

    return llama_init_from_model_internal(model, params, parent_ctx);
}

// Public alias.
struct llama_context * llama_link_shared_tensors(
    struct llama_model * model,
    struct llama_context_params params,
    struct llama_context * parent_ctx) {
    return llama_init_from_model_linked(model, params, parent_ctx);
}
```

### D2. kv_cache aliased flag

Add to `src/llama-context.h` `struct llama_kv_cache` (around line 65, near
`save_per_step_ssm`):

```cpp
    // PHASE38 D: per-layer alias flag. When set, k_l[il] / v_l[il]
    // (and split_k_l/split_v_l for split-mode) point to the parent
    // context's tensors — destructor MUST NOT free them. Only
    // transformer layers can be aliased; recurrent layers (s_l) stay
    // per-context.
    std::vector<bool> aliased_layer;
```

Update destructor (line 157-164):

```cpp
    ~llama_kv_cache() {
        // Free only contexts/buffers we OWN. Aliased layers' tensors
        // live in parent's bufs/ctxs and will be freed by parent.
        // Note: ctxs/bufs vectors only contain entries WE allocated
        // (see kv_cache_init: aliased layers skip the alloc loop).
        for (struct ggml_context * ctx : ctxs) {
            ggml_free(ctx);
        }
        for (ggml_backend_buffer_t buf : bufs) {
            ggml_backend_buffer_free(buf);
        }
    }
```

(No code change needed in the destructor itself — the contract is
that aliased layers don't add to `ctxs`/`bufs`. See D3.)

### D3. kv_cache_init aliasing logic

`src/llama.cpp:777` — modify `llama_kv_cache_init` signature OR read
`ctx->cparams.parent_ctx` from inside. Preferred: read from cparams
since the ctx pointer is already passed in.

Around line 900 (transformer layer alloc loop), add:

```cpp
const llama_context * parent = ctx->cparams.parent_ctx;
auto layer_has_attention_kv = [&](int il) {
    return !model.hparams.is_recurrent(il);
};

// Pre-resize aliased_layer
cache.aliased_layer.assign(n_layer, false);

for (int i = 0; i < n_layer; ++i) {
    // Recurrent layers (DeltaNet s_l): always per-context, never aliased
    if (!layer_has_attention_kv(i)) {
        // ...existing recurrent-layer alloc path...
        continue;
    }

    // Transformer layer: alias when parent is set
    if (parent && (int)parent->kv_self.k_l.size() > i && parent->kv_self.k_l[i] != nullptr) {
        cache.k_l.push_back(parent->kv_self.k_l[i]);
        if (needs_v_cache) {
            cache.v_l.push_back(parent->kv_self.v_l[i]);
        }
        cache.aliased_layer[i] = true;

        if (split_cache && i < (int)parent->kv_self.split_k_l.size()) {
            cache.split_k_l.push_back(parent->kv_self.split_k_l[i]);
            cache.split_v_l.push_back(parent->kv_self.split_v_l[i]);
        }
        continue;
    }

    // ...existing transformer alloc path (creates fresh tensors in
    // cache.ctxs[buft])...
}
```

The key invariant: aliased layers DO NOT add to `cache.ctxs` /
`cache.bufs`. Parent owns those allocations.

### D4. Wire common/speculative.cpp

`common/speculative.cpp:173`:

```cpp
// Before:
ctx_mtp = llama_init_from_model(const_cast<llama_model *>(model), mtp_cparams);

// After:
ctx_mtp = llama_init_from_model_linked(
    const_cast<llama_model *>(model),
    mtp_cparams,
    parent_ctx);  // = ctx_tgt, passed to common_speculative_state_mtp ctor
```

`common_speculative_state_mtp` constructor signature gets `parent_ctx`
parameter; caller `common_speculative_state_init` passes the verify ctx.

### D5. Verify INLINE_KV behavior

When parent_ctx is set, `parent->cparams.mtp_inline_kv_hook` is forced
false (in D1's `llama_init_from_model_linked`). The existing
`mtp_accept_tokens` early-return already keys off
`getenv("LLAMA_MTP_INLINE_KV")` (`common/speculative.cpp:1708`) — when
the env is unset, `_hook_on=false` and UPDATE_ACCEPTED dispatch
runs. We need the cparams flag, not just the env, to be the source of
truth here. Either:

- (a) Update `mtp_accept_tokens` early-return to read
  `parent->cparams.mtp_inline_kv_hook` instead of the env, OR
- (b) Have `llama_init_from_model_linked` `unsetenv("LLAMA_MTP_INLINE_KV")`
  for parent (hacky, prefer (a)).

### D6. Smoke test sequence

1. **VRAM sanity**: build green; start single-slot server with MTP on
   and a parent_ctx test path; log `kv_self.bufs` total size for
   parent vs child. Child should be <500 MiB (recurrent state only).
2. **Correctness**: `--fast` harness GREEN — output text matches a
   reference run (greedy decoding, `temp=0`).
3. **Multi-slot fit**: write profile `qwen36-27b-x3-mtp.sh` with
   `np=3`, `ctx-size=262144`, `-mtp --draft 3`,
   `-no-mtp-inline-kv-hook` (or whatever flag flips it), and
   `link_shared_tensors` semantically active. Server starts, all 3
   slots warm without OOM.
4. **Throughput**: multi-turn bench across slots; binding metric is
   per-slot tg vs the np=1 nomtp baseline (29.62 t/s).

## Anti-patterns to avoid (per session memory)

- ❌ Add a new env knob to gate D — make it default-on when
  parent_ctx is set
- ❌ Bench partial implementations to decide whether to continue
  (`feedback_circular_diagnostic_bench`)
- ❌ Multiple coexisting MTP code paths — D should obsolete the
  ctx_mtp-with-its-own-KV path entirely

## Out of scope (separate later phases)

- **PHASE38 E** (seed-source fix): wire `llama_main_graph_h_pre_norm()`
  + indexed access into the chain-residual seed path. After D lands.
- **PHASE38 F** (Hadamard absorption into weights): multiply W_Q/W_K/W_V
  by H at model load when `k_cache_hadamard` is set; drop runtime
  `ggml_hadamard` ops in `llama-build-context.cpp:1782-1792`. After D+E.

## Pickup brief for the session that lands D

1. Read this file + PHASE38.md sections D1-D6.
2. Verify scaffolding is intact (`grep parent_ctx src/llama-cparams.h`).
3. Implement D1 (public API + internal helper).
4. Implement D2 (aliased_layer field).
5. Implement D3 (alias logic in kv_cache_init).
6. Implement D4 (speculative.cpp wiring).
7. Implement D5 (UPDATE_ACCEPTED gate).
8. Run smoke tests in order: build, single-slot correctness,
   single-slot VRAM sanity, multi-slot fit, multi-slot bench.
9. Document results inline below this file's "Bench target" table.

Estimated cost: ~30-50k tokens, focused single-feature branch.
