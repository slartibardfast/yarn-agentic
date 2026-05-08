# PHASE 45 D4 — External `llama_context` Callsites

Inventory of all `llama_context` usages outside `src/`. Drives D10 (delete `llama_context`) and the migration plan for server / common / examples.

## Summary

| Category | Count | Migration cost |
|---|---:|---|
| **Live callsites** (in our build / production) | 12 | high — full migration required |
| **Dead callsites** (legacy examples we don't ship) | 18 | zero — delete with the type |
| **Total API methods touched** | ~90 | classified by destination below |

## Live callsites (must migrate)

### Core (`common/`)

| File | Usage | Migration notes |
|---|---|---|
| `common/common.cpp` | Context lifecycle, KV cache management | Refactor `common_init_from_params` to return `(session, decoder)` pair |
| `common/sampling.cpp` | Logit extraction | Operates on decoder output — straightforward |
| `common/speculative.cpp` | **Heavy MTP user** — 14+ `mtp_*` functions, dual-context spec | Becomes the body of `llama_spec_loop`; ctx_tgt/ctx_mtp collapse to verify/draft decoders |
| `common/train.cpp` | Weak dependency (model accessor) | Trivial — only needs `llama_session_model()` |

### Examples (shipped with build)

| File | Usage | Migration notes |
|---|---|---|
| `examples/server/server-context.cpp` | **Heaviest user** — 50+ API calls, full lifecycle, slot management | Multi-stage migration; each slot becomes (session, decoder) |
| `examples/server/server-common.cpp` | Batch response helpers | Simple refactor |
| `examples/server/server-task.cpp` | Per-task state persistence | Touches kv_seq_state — moves to session API |
| `examples/main/main.cpp` | Standard CLI inference | Trivial: init → session, decode → decoder |
| `examples/speculative/speculative.cpp` | Dual-context spec | Becomes spec_loop user |
| `examples/simple/simple.cpp` | Minimal example (3 calls) | Trivial |
| `examples/batched/batched.cpp` | Multi-sequence batching | Session = shared, decoder = primary |
| `examples/embedding/embedding.cpp` | Embedding extraction | Decoder.role = embedding-output |

## Dead callsites (delete with `llama_context`)

18 files — none in production path:

- Test suites (`tests/test-*.cpp`) — tests get rewritten or deleted per PHASE45 D11
- Benchmark tools (`batched-bench`, `llama-bench`, `sweep-bench`) — reimplement against new API only if still useful
- Demos (`lookahead`, `passkey`, `perplexity`, `retrieval`, etc.) — not in `profiles/`, not shipped, delete

**Action in D10:** `git rm` these files alongside the `llama_context` deletion. No migration cost.

## API surface (~90 methods classified)

### → `llama_session_*` (31 methods)

KV cache and sequence management (state lives in session):
```
llama_kv_self_clear / _seq_cp / _seq_rm / _seq_add / _seq_div / _seq_keep
llama_kv_self_update / _defrag
llama_state_get_size / _get_data / _set_data
llama_state_seq_get_size / _get_data / _set_data
llama_state_save_file / _load_file
llama_state_seq_save_file / _seq_load_file
llama_set_n_threads (session-side because n_threads affects kv ops)
llama_lora_adapter_set / _remove / _clear
llama_control_vector_apply
llama_n_ctx / _n_seq_max / _n_batch / _n_ubatch
llama_get_model
```

### → `llama_decoder_*` (14 methods)

Forward computation (per-execution-role):
```
llama_decode / llama_encode
llama_get_logits / _get_logits_ith
llama_get_embeddings / _get_embeddings_ith / _get_embeddings_seq
llama_synchronize
llama_perf_context / _perf_context_reset / _perf_context_print
llama_set_causal_attn / _set_embeddings (decoder-mode flags)
llama_set_warmup
```

### → `llama_spec_loop_*` (18 methods)

Speculative orchestration (today scattered across `common/speculative.cpp`):
```
llama_mtp_fused_dispatch / _fused_dispatch_async / _fused_extract_results
llama_mtp_fused_draft_invoke / _draft_get_results
llama_mtp_set_inline_kv_hook / _set_persist_from_host
llama_mtp_accept_tokens / _shortcut_residual
llama_mtp_speculative_gen_draft (?)
common_speculative_state_init / _free / _gen_draft / _accept (the existing API surface around speculative.cpp moves here)
```

### Deleted (no replacement)

PHASE38 E async dispatch APIs that landed but were never used (per D3):
```
mtp_fused_skip_extraction (flag — delete)
mtp_fused_pending_gf / _pending_n_steps / _async_guess (state — delete)
```

## Migration ordering for D6-D10

1. **D6** (CPU forward): `common/common.cpp` simplest path — `main.cpp` is the proving ground.
2. **D7** (CUDA single-slot): same path, CUDA backend.
3. **D8** (spec decoding): `common/speculative.cpp` rewrite. The functional body becomes `llama_spec_loop` impl. `examples/speculative/speculative.cpp` validates.
4. **D9** (multi-slot): `examples/server/server-context.cpp` rewrite. The slot abstraction maps onto (session, decoder-pair). This is the biggest single migration.
5. **D10** (`llama_context` deletion): `git grep llama_context src/` → 0; delete dead callsites in `examples/`.

## Risk assessment

- **Highest-risk migration:** `server-context.cpp` (50+ callsites, multi-slot, async io, KV save/load, slot state machine). Suggest dedicating a full session to this alone.
- **Highest-leverage migration:** `common/speculative.cpp` → `llama_spec_loop`. Done well, this becomes the cleanest part of the codebase. Done poorly, it's tech debt that infects everything.
- **Trivial migrations:** `main.cpp`, `simple.cpp`, `embedding.cpp`, `batched.cpp` — each is < 100 LoC of changes.
