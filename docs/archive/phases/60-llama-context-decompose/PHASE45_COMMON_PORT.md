# PHASE45 D6/D10 Callsite Inventory & D6 Edit Plan

**Status:** Planning document for PHASE45 D6 (CPU greedy-decode 50 tokens) and D10 (delete llama_context) port.

**Date:** 2026-05-08

---

## 1. `examples/main/main.cpp` Callsite Inventory

All `llama_*` and `llama_kv_*` API calls in main.cpp, with replacements for D6 (PRIMARY decoder role, single-slot session).

| Line | Call | Context | D6 Replacement |
|------|------|---------|---|
| 33 | `static llama_context ** g_ctx;` | Global context pointer | `static llama_session ** g_session; static llama_decoder ** g_decoder;` |
| 55 | `const llama_context * ctx` param | write_logfile() signature | `const llama_session * session` |
| 82 | `llama_model_desc(model, ...)` | Log model info | ✓ No change (model-only) |
| 94 | `llama_dump_timing_info_yaml(logfile, ctx)` | Dump timing to YAML | `llama_decoder_perf_print(decoder)` (or expose timing accessor) |
| 107 | `llama_print_timings(*g_ctx)` | Print to stdout | `llama_decoder_perf_print(*g_decoder)` |
| 144 | `llama_log_set(...)` | Set log callback | ✓ No change (global) |
| 196 | `llama_backend_init()` | Backend setup | ✓ No change |
| 197 | `llama_numa_init(params.numa)` | NUMA affinity | ✓ No change |
| 208 | `llama_init_result llama_init = llama_init_from_gpt_params(params)` | Load model + ctx | `llama_model * model = llama_model_load_from_file(...); llama_session * session = llama_session_create(model, sparams); llama_decoder * decoder = llama_decoder_create(session, dparams);` |
| 213 | `struct llama_context_params lparams = common_context_params_to_llama(params)` | Guidance ctx params | `struct llama_session_params sparams_g = common_session_params_to_llama(params)` (new helper) |
| 214 | `ctx_guidance = llama_init_from_model(model, lparams)` | Create guidance ctx | `llama_session * session_guidance = llama_session_create(model, sparams_g); llama_decoder * decoder_guidance = llama_decoder_create(session_guidance, dparams_g);` |
| 223 | `const int n_ctx_train = llama_n_ctx_train(model)` | Model ctx train size | ✓ No change (model-only) |
| 224 | `const int n_ctx = llama_n_ctx(ctx)` | Query ctx size | `const int n_ctx = llama_session_n_ctx(session)` |
| 260 | `llama_state_load_file(ctx, path, tokens.data(), ...)` | Load session state | `llama_session_state_get_data(session, ...)` (session owns state) |
| 269 | `llama_should_add_bos_token(model)` | Model property | ✓ No change (model-only) |
| 270 | `llama_model_has_encoder(model)` | Model property | ✓ No change (model-only) |
| 271 | `llama_add_eos_token(model)` | Model property | ✓ No change (model-only) |
| 301 | `::common_tokenize(ctx, prompt, ...)` | Tokenize with vocab | `::common_tokenize(model, prompt, ...)` (change to take model, not ctx) |
| 308 | `LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp)` | Log tokens | `LOG_TOKENS_TOSTR_PRETTY(model, embd_inp)` (change to take model) |
| 314 | `llama_token_bos(model)` | Get BOS token | ✓ No change (model-only) |
| 329 | `::common_tokenize(ctx_guidance, ...)` | Guidance tokenize | `::common_tokenize(model, ...)` (guidance uses same model) |
| 332 | `::common_tokenize(ctx, ...)` | Original prompt tokenize | `::common_tokenize(model, ...)` |
| 368 | `llama_kv_cache_seq_rm(ctx, -1, n_matching, -1)` | KV cache seq removal | `llama_session_kv_seq_rm(session, -1, n_matching, -1)` |
| 404 | `common_token_to_piece(ctx, token)` | Detokenize | `common_token_to_piece(model, token)` (change to take model) |
| 412 | `common_token_to_piece(ctx, token)` | Detokenize | `common_token_to_piece(model, token)` |
| 419 | `common_token_to_piece(ctx, token)` | Detokenize | `common_token_to_piece(model, token)` |
| 449 | `::common_tokenize(ctx, antiprompt, ...)` | Tokenize antiprompt | `::common_tokenize(model, antiprompt, ...)` |
| 464 | `::common_tokenize(ctx, input_prefix, ...)` | Tokenize prefix | `::common_tokenize(model, input_prefix, ...)` |
| 474 | `::common_tokenize(ctx, input_suffix, ...)` | Tokenize suffix | `::common_tokenize(model, input_suffix, ...)` |
| 548 | `::common_tokenize(ctx, antiprompt, ...)` | Tokenize antiprompt | `::common_tokenize(model, antiprompt, ...)` |
| 561 | `llama_encode(ctx, llama_batch_get_one(...))` | Encoder-decoder path | `llama_decoder_encode(decoder, llama_batch_get_one(...))` |
| 566 | `llama_model_decoder_start_token(model)` | Decoder start token | ✓ No change (model-only) |
| 568 | `llama_token_bos(model)` | BOS fallback | ✓ No change (model-only) |
| 610 | `llama_kv_cache_seq_rm(ctx, 0, n_keep, n_keep+n_discard)` | KV cache discard | `llama_session_kv_seq_rm(session, 0, n_keep, n_keep+n_discard)` |
| 611 | `llama_kv_cache_seq_add(ctx, 0, n_keep+n_discard, n_past, -n_discard)` | KV cache shift | `llama_session_kv_seq_add(session, 0, n_keep+n_discard, n_past, -n_discard)` |
| 638 | `llama_kv_cache_seq_add(ctx, 0, ga_i, n_past, ib*bd)` | Self-Extend shift 1 | `llama_session_kv_seq_add(session, 0, ga_i, n_past, ib*bd)` |
| 639 | `llama_kv_cache_seq_div(ctx, 0, ga_i+ib*bd, ga_i+ib*bd+ga_w, ga_n)` | Self-Extend div | `llama_session_kv_seq_div(session, 0, ga_i+ib*bd, ga_i+ib*bd+ga_w, ga_n)` |
| 640 | `llama_kv_cache_seq_add(ctx, 0, ga_i+ib*bd+ga_w, n_past+ib*bd, dd)` | Self-Extend shift 2 | `llama_session_kv_seq_add(session, 0, ga_i+ib*bd+ga_w, n_past+ib*bd, dd)` |
| 703 | `llama_decode(ctx_guidance, llama_batch_get_one(...))` | Guidance decode | `llama_decoder_decode(decoder_guidance, llama_batch_get_one(...))` |
| 720 | `llama_decode(ctx, llama_batch_get_one(&embd[i], ...))` | **Main greedy decode** | `llama_decoder_decode(decoder, llama_batch_get_one(&embd[i], ...))` |
| 747 | `llama_state_save_file(ctx, path, tokens.data(), ...)` | Save session state | `llama_session_state_get_data(session, dst, sz)` (session owns state) |
| 752 | `common_sampler_sample_legacy(ctx_sampling, ctx, ctx_guidance)` | Sample from logits | **See §2: Sampler passes decoder to logit getter** |
| 787 | `common_token_to_piece(ctx, id, params.special)` | Detokenize output | `common_token_to_piece(model, id, params.special)` |
| 818 | `llama_sampling_prev_str(ctx_sampling, ctx, n_prev)` | Get sampled tokens | **Sampler signature change: pass model instead** |
| 863 | `::common_tokenize(ctx, antiprompt.front(), ...)` | Tokenize antiprompt | `::common_tokenize(model, antiprompt.front(), ...)` |
| 879 | `common_token_to_piece(ctx, id, false)` | Detokenize assist msg | `common_token_to_piece(model, id, false)` |
| 891 | `llama_token_bos(model)` | BOS for input | ✓ No change (model-only) |
| 937 | `::common_tokenize(ctx, params.input_prefix, ...)` | Tokenize prefix | `::common_tokenize(model, params.input_prefix, ...)` |
| 938 | `::common_tokenize(ctx, user_inp, ...)` | Tokenize user input | `::common_tokenize(model, user_inp, ...)` |
| 939 | `::common_tokenize(ctx, params.input_suffix, ...)` | Tokenize suffix | `::common_tokenize(model, params.input_suffix, ...)` |
| 945 | `llama_token_eot(model)` | EOT token | ✓ No change (model-only) |
| 946 | `llama_token_eos(model)` | EOS token | ✓ No change (model-only) |
| 957 | `common_token_to_piece(ctx, token)` | Detokenize user token | `common_token_to_piece(model, token)` |
| 998 | `llama_state_save_file(ctx, path, tokens.data(), ...)` | Save final session | `llama_session_state_get_data(session, dst, sz)` |
| 1001 | `llama_print_timings(ctx)` | Print final timing | `llama_decoder_perf_print(decoder)` |
| 1002 | `write_logfile(ctx, params, model, ...)` | Write logfile with ctx | `write_logfile(session, params, model, ...)` |
| 1004 | `llama_free(ctx_guidance)` | Free guidance ctx | `llama_decoder_free(decoder_guidance); llama_session_free(session_guidance);` |
| 1005 | `llama_free(ctx)` | **Free main ctx** | `llama_decoder_free(decoder); llama_session_free(session);` |
| 1006 | `llama_free_model(model)` | Free model | ✓ No change |
| 1008 | `common_sampler_free(ctx_sampling)` | Free sampler | ✓ No change (sampler is independent) |
| 1009 | `llama_backend_free()` | Backend cleanup | ✓ No change |

**Total main.cpp callsites:** 60+ `llama_context`/`llama_kv_*` references.

---

## 2. Hot-path Subset for D6 (Greedy-decode 50 tokens)

The following calls are on the critical path for the D6 binding test. These must be ported first:

1. **Context initialization (line 208):**
   - `llama_init_from_gpt_params(params)` → Split into:
     - `llama_model_load_from_file(...)` (unchanged)
     - `llama_session_create(model, session_params)` (new)
     - `llama_decoder_create(session, decoder_params)` (new)
   
2. **Main decode loop (lines 720, 703):**
   - `llama_decode(ctx, batch)` → `llama_decoder_decode(decoder, batch)` (2 callsites: main + guidance)
   
3. **Logits access for sampling (implicit in line 752):**
   - Sampling code calls `llama_get_logits_ith(ctx, i)` internally
   - New: `common_sampler_sample_legacy(ctx_sampling, decoder, decoder_guidance)` (pass decoders, not contexts)
   - Decoder provides: `llama_decoder_get_logits_ith(decoder, i)` → `float*`
   
4. **KV cache operations (lines 610, 611, 368):**
   - `llama_kv_cache_seq_rm(ctx, ...)` → `llama_session_kv_seq_rm(session, ...)`
   - `llama_kv_cache_seq_add(ctx, ...)` → `llama_session_kv_seq_add(session, ...)`
   - `llama_kv_cache_seq_div(ctx, ...)` → `llama_session_kv_seq_div(session, ...)`
   
5. **State persistence (lines 260, 747, 998):**
   - `llama_state_load_file(ctx, ...)` → `llama_session_state_set_data(session, ...)`
   - `llama_state_save_file(ctx, ...)` → `llama_session_state_get_data(session, ...)`
   
6. **Cleanup (line 1004-1005):**
   - `llama_free(ctx)` → `llama_decoder_free(decoder); llama_session_free(session);`
   
7. **Query APIs (line 224):**
   - `llama_n_ctx(ctx)` → `llama_session_n_ctx(session)`

**Non-hot-path (no change for D6):**
- Tokenization: `::common_tokenize(ctx, ...)` → kept as-is for now; D10 can refactor to take `model` directly
- Detokenization: `common_token_to_piece(ctx, ...)` → same, D10 refactor
- Timing/logging: `llama_print_timings(ctx)`, `yaml_dump_non_result_info(ctx, ...)` → D10 refactor
- Logging infrastructure: model properties, log callbacks, backend init

---

## 3. Draft D6 Edit to `examples/main/main.cpp`

**Minimal diff showing context → session + decoder migration for greedy decode path.**

### Global declarations (lines 33-34)

```cpp
// BEFORE:
static llama_context           ** g_ctx;
static llama_model             ** g_model;

// AFTER:
static llama_session           ** g_session;
static llama_decoder           ** g_decoder;
static llama_model             ** g_model;
```

### Context initialization (lines 208-214)

```cpp
// BEFORE:
llama_model * model;
llama_context * ctx;
llama_context * ctx_guidance = NULL;

llama_init_result llama_init = llama_init_from_gpt_params(params);
model = llama_init.model;
ctx = llama_init.context;
if (sparams.cfg_scale > 1.f) {
    struct llama_context_params lparams = common_context_params_to_llama(params);
    ctx_guidance = llama_init_from_model(model, lparams);
}

// AFTER:
llama_model * model = nullptr;
llama_session * session = nullptr;
llama_decoder * decoder = nullptr;
llama_session * session_guidance = nullptr;
llama_decoder * decoder_guidance = nullptr;

// Load model
struct llama_model_params mparams = common_model_params_to_llama(params);
model = llama_model_load_from_file(params.model.c_str(), mparams);

if (model == NULL) {
    LOG_TEE("%s: error: unable to load model\n", __func__);
    return 1;
}

// Create main session and decoder (PRIMARY role for greedy decode)
struct llama_session_params sparams = llama_session_default_params();
sparams.n_ctx = params.n_ctx;
sparams.n_seq_max = 1;  // D6: single-slot
sparams.n_batch = params.n_batch;
sparams.n_ubatch = params.n_ubatch;
sparams.type_k = params.type_k;
sparams.type_v = params.type_v;
sparams.rope_freq_base = params.rope_freq_base;
sparams.rope_freq_scale = params.rope_freq_scale;
sparams.n_ctx_orig_yarn = params.n_ctx_orig;
sparams.yarn_ext_factor = params.yarn_ext_factor;
sparams.yarn_attn_factor = params.yarn_attn_factor;
sparams.yarn_beta_fast = params.yarn_beta_fast;
sparams.yarn_beta_slow = params.yarn_beta_slow;
sparams.flash_attn = params.flash_attn;

session = llama_session_create(model, sparams);

struct llama_decoder_params dparams = llama_decoder_default_params(LLAMA_DECODER_PRIMARY);
dparams.n_threads = params.n_threads;
dparams.n_threads_batch = params.n_threads_batch;
dparams.causal_attn = true;
dparams.embeddings = false;
dparams.rope_cache = params.cache_type_k != GGML_TYPE_F32;

decoder = llama_decoder_create(session, dparams);

// Create guidance session and decoder if needed
if (sparams.cfg_scale > 1.f) {
    session_guidance = llama_session_create(model, sparams);
    decoder_guidance = llama_decoder_create(session_guidance, dparams);
}
```

### Query context size (line 224)

```cpp
// BEFORE:
const int n_ctx = llama_n_ctx(ctx);

// AFTER:
const int n_ctx = llama_session_n_ctx(session);
```

### KV cache operations (e.g., line 368, 610-611, 638-640)

```cpp
// BEFORE:
llama_kv_cache_seq_rm(ctx, -1, n_matching_session_tokens, -1);

// AFTER:
llama_session_kv_seq_rm(session, -1, n_matching_session_tokens, -1);
```

```cpp
// BEFORE:
llama_kv_cache_seq_rm (ctx, 0, params.n_keep, params.n_keep + n_discard);
llama_kv_cache_seq_add(ctx, 0, params.n_keep + n_discard, n_past, -n_discard);

// AFTER:
llama_session_kv_seq_rm (session, 0, params.n_keep, params.n_keep + n_discard);
llama_session_kv_seq_add(session, 0, params.n_keep + n_discard, n_past, -n_discard);
```

```cpp
// BEFORE (Self-Extend):
llama_kv_cache_seq_add(ctx, 0, ga_i, n_past, ib*bd);
llama_kv_cache_seq_div(ctx, 0, ga_i + ib*bd, ga_i + ib*bd + ga_w, ga_n);
llama_kv_cache_seq_add(ctx, 0, ga_i + ib*bd + ga_w, n_past + ib*bd, dd);

// AFTER:
llama_session_kv_seq_add(session, 0, ga_i, n_past, ib*bd);
llama_session_kv_seq_div(session, 0, ga_i + ib*bd, ga_i + ib*bd + ga_w, ga_n);
llama_session_kv_seq_add(session, 0, ga_i + ib*bd + ga_w, n_past + ib*bd, dd);
```

### Main decode loop (lines 703, 720)

```cpp
// BEFORE:
if (llama_decode(ctx_guidance, llama_batch_get_one(input_buf + i, n_eval, n_past_guidance, 0))) {
    LOG_TEE("%s : failed to eval\n", __func__);
    return 1;
}

// AFTER:
if (llama_decoder_decode(decoder_guidance, llama_batch_get_one(input_buf + i, n_eval, n_past_guidance, 0))) {
    LOG_TEE("%s : failed to eval\n", __func__);
    return 1;
}
```

```cpp
// BEFORE (Main hot-path decode):
if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval, n_past, 0))) {
    LOG_TEE("%s : failed to eval\n", __func__);
    return 1;
}

// AFTER:
if (llama_decoder_decode(decoder, llama_batch_get_one(&embd[i], n_eval, n_past, 0))) {
    LOG_TEE("%s : failed to eval\n", __func__);
    return 1;
}
```

### State persistence (lines 260, 747, 998)

```cpp
// BEFORE:
if (!llama_state_load_file(ctx, path_session.c_str(), session_tokens.data(), ...)) {
    // error handling
}

// AFTER:
size_t state_size = llama_session_state_seq_get_size(session, 0);
if (state_size > 0 && !llama_session_state_seq_set_data(session, session_tokens_data, state_size, 0)) {
    // error handling
}
```

```cpp
// BEFORE (Save):
llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());

// AFTER:
uint8_t * state_buf = new uint8_t[session_tokens.capacity()];
size_t state_sz = llama_session_state_seq_get_data(session, state_buf, session_tokens.capacity(), 0);
// write state_buf[0..state_sz] to file
delete[] state_buf;
```

### Encoder-decoder path (line 561)

```cpp
// BEFORE:
if (llama_encode(ctx, llama_batch_get_one(enc_input_buf, enc_input_size, 0, 0))) {
    LOG_TEE("%s : failed to eval\n", __func__);
    return 1;
}

// AFTER:
if (llama_decoder_encode(decoder, llama_batch_get_one(enc_input_buf, enc_input_size, 0, 0))) {
    LOG_TEE("%s : failed to eval\n", __func__);
    return 1;
}
```

### Sampling integration (line 752)

```cpp
// BEFORE:
const llama_token id = common_sampler_sample_legacy(ctx_sampling, ctx, ctx_guidance);

// AFTER:
const llama_token id = common_sampler_sample_legacy(ctx_sampling, decoder, decoder_guidance);
```

**Note:** `common_sampler_sample_legacy()` signature changes from `(sampler, ctx_main, ctx_cfg)` to `(sampler, decoder_main, decoder_cfg)`. The sampler internally calls `llama_decoder_get_logits_ith(decoder, idx)` instead of `llama_get_logits_ith(ctx, idx)`. **Decoder takes ownership of logit access; sampler still independent.** (See §5 and PHASE45.md lock: sampling is fully external, sampler borrows decoder pointer.)

### Timing and diagnostics (lines 107, 1001)

```cpp
// BEFORE:
llama_print_timings(*g_ctx);

// AFTER:
llama_decoder_perf_print(*g_decoder);
```

```cpp
// BEFORE:
llama_print_timings(ctx);

// AFTER:
llama_decoder_perf_print(decoder);
```

### Cleanup (lines 1004-1005)

```cpp
// BEFORE:
if (ctx_guidance) { llama_free(ctx_guidance); }
llama_free(ctx);
llama_free_model(model);

// AFTER:
if (decoder_guidance) { llama_decoder_free(decoder_guidance); }
if (session_guidance) { llama_session_free(session_guidance); }
llama_decoder_free(decoder);
llama_session_free(session);
llama_free_model(model);
```

---

## 4. `common/` Callsite Inventory (D10 scope)

All `llama_context`-dependent callsites in `common/` (files: `common.h`, `common.cpp`, `sampling.h`, `sampling.cpp`, `speculative.cpp`, etc.).

| File | Line | Call | Type | D10 Action |
|------|------|------|------|---|
| **common.h** | 200 | `llama_model * model_dft` | Field | ✓ Rename to `draft_model` (no change, comment: PHASE45 uses separate decoder_draft instead) |
| **common.h** | 202 | `llama_context_params cparams_dft` | Field | ✓ Rename to `decoder_params_dft` (or remove if D10 refactors dft path) |
| **common.h** | 619-621 | `struct llama_init_result { llama_model*, llama_context* }` | Type def | **Split into:** `llama_init_result { llama_model*, llama_session*, llama_decoder* }` |
| **common.h** | 625 | `llama_init_result llama_init_from_gpt_params(...)` | Function sig | **Port to:** `llama_init_result llama_init_from_gpt_params(...) { create session + decoder, return both; }` |
| **common.h** | 627 | `llama_context_params common_context_params_to_llama(...)` | Helper | **Remove or split:** `llama_session_params common_session_params_to_llama(...)` + `llama_decoder_params common_decoder_params_to_llama(...)` |
| **common.h** | 654 | `const struct llama_context * ctx` in common_batch_add | Param | **Change to:** `const struct llama_session * session` (or `const struct llama_model * model` if only vocab access needed) |
| **common.h** | 653-659 | `std::vector<llama_token> common_tokenize(const llama_context *, ...)` | Function sig (3 overloads) | **Change all to:** `common_tokenize(const llama_model *, ...)` (vocab access only) |
| **common.cpp** | 3355-3462 | `llama_init_from_gpt_params()` function body | Helper | **Major refactor:** split into `llama_init_session()` + `llama_init_decoder()`, call llama_session_create + llama_decoder_create |
| **common.cpp** | 3376 | `llama_context * lctx = llama_init_from_model(model, cparams)` | Create ctx | Change to session+decoder creation |
| **common.cpp** | 3384 | `llama_set_offload_policy(lctx, ...)` | API call | **Remove or port:** offload_policy → decoder_params field (read at decoder construction, not per-call) |
| **common.cpp** | 3391 | `llama_control_vector_load(params.control_vectors)` | Load CV | ✓ No change (returns adapter object, not ctx-dependent) |
| **common.cpp** | 3434 | `llama_n_ctx(lctx)` | Query | Change to `llama_session_n_ctx(session)` |
| **common.cpp** | 3453 | `llama_encode(lctx, batch)` | Encode | Change to `llama_decoder_encode(decoder, batch)` |
| **common.cpp** | 3462 | `llama_decode(lctx, batch)` | Decode | Change to `llama_decoder_decode(decoder, batch)` |
| **common.cpp** | 4073-4131 | `common_tokenize(const llama_context *, ...)` overloads | Tokenize | Change all to take `const llama_model *` instead (3 functions) |
| **common.cpp** | 4132 | `common_token_to_piece(const llama_context *, ...)` | Detokenize | Change to take `const llama_model *` or `const llama_vocab *` |
| **sampling.h** | 246 | Comment: "sample from llama_get_logits_ith(ctx, idx)" | Doc | Update to "llama_decoder_get_logits_ith(decoder, idx)" |
| **sampling.cpp** | 51 | `common_sampler_init(const llama_model *, params)` | Init | ✓ Already model-only (no change) |
| **sampling.cpp** | 213 | `llama_grammar_free(ctx->grammar)` | Free | ✓ No change (internal to common_sampler state) |
| **sampling.cpp** | 535 | `float * logits = llama_get_logits_ith(ctx_main, idx)` | **Hot path** | Change to `llama_decoder_get_logits_ith(decoder_main, idx)` |
| **sampling.cpp** | 581 | `float * logits = llama_get_logits_ith(ctx_main, idx)` | **Hot path** | Change to `llama_decoder_get_logits_ith(decoder_main, idx)` |
| **sampling.cpp** | 630 | `float * logits = llama_get_logits_ith(ctx_main, idx)` | **Hot path** | Change to `llama_decoder_get_logits_ith(decoder_main, idx)` |
| **sampling.cpp** | 644 | `float * logits_guidance = llama_get_logits_ith(ctx_cfg, idx)` | **Hot path** | Change to `llama_decoder_get_logits_ith(decoder_cfg, idx)` |
| **sampling.cpp** | 862 | `float * logits = llama_get_logits_ith(ctx, idx)` | **Hot path** | Change to `llama_decoder_get_logits_ith(decoder, idx)` |
| **speculative.cpp** | 203 | `llama_kv_cache_seq_rm(ctx_mtp, ...)` | KV op | Change to `llama_session_kv_seq_rm(session_mtp, ...)` |
| **speculative.cpp** | 372-373 | `llama_kv_cache_seq_rm(ctx_dft, ...); llama_kv_cache_seq_add(ctx_dft, ...)` | KV ops | Change to session equivalents |
| **speculative.cpp** | 379 | `llama_kv_cache_seq_rm(ctx_dft, ...)` | KV op | Change to `llama_session_kv_seq_rm(session_dft, ...)` |
| **speculative.cpp** | 398, 412, 448, 1582 | `llama_decode(ctx_dft, batch)` | Decode (4x) | Change all to `llama_decoder_decode(decoder_dft, batch)` |
| **speculative.cpp** | 996 | `llama_decode(ctx_tgt, batch)` | Decode | Change to `llama_decoder_decode(decoder_tgt, batch)` |
| **speculative.cpp** | 1004 | `llama_kv_cache_seq_rm(ctx_tgt, 0, 1, -1)` | KV op | Change to `llama_session_kv_seq_rm(session_tgt, 0, 1, -1)` |
| **speculative.cpp** | 1353-1354 | `llama_kv_cache_seq_rm/add(ctx_mtp, ...)` | KV ops | Change to session equivalents |
| **speculative.cpp** | 1656, 1672, 1688 | `llama_kv_cache_seq_rm(ctx, ...); llama_decode(ctx, ...)` | Mixed KV+decode | Change to session + decoder |

**Total common/ callsites:** ~40+ `llama_context`/`llama_kv_*` references.

---

## 5. `common.cpp::llama_init_from_gpt_params` Refactor

This is the canonical "initialize a context from gpt_params" helper. Currently returns `llama_init_result { model, context }`.

### Current (old) implementation outline (lines 3355-3462):

```cpp
struct llama_init_result llama_init_from_gpt_params(gpt_params & params) {
    llama_init_result iparams;
    
    // Load model
    llama_model * model = llama_model_load_from_file(...);
    
    // Apply LoRA, control vectors
    // ...
    
    // Create context from model
    llama_context * lctx = llama_init_from_model(model, cparams);
    
    // Offload policy, dry_penalty tuning
    llama_set_offload_policy(lctx, ...);
    llama_n_ctx(lctx); // query
    
    iparams.model = model;
    iparams.context = lctx;
    return iparams;
}
```

### New (D10) implementation outline:

```cpp
struct llama_init_result llama_init_from_gpt_params(gpt_params & params) {
    llama_init_result iparams;
    
    // Load model (unchanged)
    struct llama_model_params mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.c_str(), mparams);
    
    // Apply LoRA, control vectors (unchanged, model-level)
    // ...
    
    // NEW: Create session
    struct llama_session_params sparams = common_session_params_to_llama(params);
    llama_session * session = llama_session_create(model, sparams);
    
    // NEW: Create decoder (PRIMARY role for non-speculative)
    struct llama_decoder_params dparams = common_decoder_params_to_llama(params);
    llama_decoder * decoder = llama_decoder_create(session, dparams);
    
    // MOVED: offload_policy, dry_penalty etc. → decoder_params (set before create)
    // Example: dparams.n_threads = params.n_threads (already done in common_decoder_params_to_llama)
    
    iparams.model = model;
    iparams.session = session;      // NEW field
    iparams.decoder = decoder;      // NEW field
    // iparams.context = nullptr;   // REMOVED or deprecated
    
    return iparams;
}
```

### New helper signatures:

```cpp
// D10: Split llama_context_params into two
struct llama_session_params common_session_params_to_llama(const gpt_params & params) {
    struct llama_session_params sparams = llama_session_default_params();
    sparams.n_ctx = params.n_ctx;
    sparams.n_seq_max = params.mtp ? params.n_seq_max : 1;
    sparams.n_batch = params.n_batch;
    sparams.n_ubatch = params.n_ubatch;
    sparams.type_k = params.type_k;
    sparams.type_v = params.type_v;
    sparams.rope_freq_base = params.rope_freq_base;
    sparams.rope_freq_scale = params.rope_freq_scale;
    sparams.flash_attn = params.flash_attn;
    // ... rest of session-specific fields
    return sparams;
}

struct llama_decoder_params common_decoder_params_to_llama(const gpt_params & params) {
    enum llama_decoder_role role = params.mtp ? LLAMA_DECODER_DRAFT_MTP : LLAMA_DECODER_PRIMARY;
    struct llama_decoder_params dparams = llama_decoder_default_params(role);
    dparams.n_threads = params.n_threads;
    dparams.n_threads_batch = params.n_threads_batch;
    dparams.causal_attn = !params.is_encoder_only;
    dparams.embeddings = params.embedding;
    dparams.rope_cache = params.cache_type_k != GGML_TYPE_F32;
    // ... rest of decoder-specific fields
    return dparams;
}
```

### Updated llama_init_result struct (common.h):

```cpp
struct llama_init_result {
    struct llama_model   * model   = nullptr;
    struct llama_session * session = nullptr;     // NEW
    struct llama_decoder * decoder = nullptr;     // NEW
    // struct llama_context * context = nullptr;  // DEPRECATED (can remove in D10 or keep as nullptr alias)
};
```

---

## 6. Risks & Open Questions

### A. Logits Access Path (PHASE45 lock)

**Issue:** Sampling code calls `llama_get_logits_ith(ctx, idx)` to fetch logits for candidate tokens. This is currently context-centric.

**D6/D10 Resolution:**
- Logits are **decoder output**, not context state.
- New API: `llama_decoder_get_logits_ith(decoder, idx)` returns `float*` to logit vector.
- `common_sampler_sample_legacy()` signature changes: takes `(decoder_main, decoder_cfg)` instead of `(ctx_main, ctx_cfg)`.
- **Sampler is independent** of context/session/decoder per PHASE45 lock; it just borrows the decoder pointer to call `llama_decoder_get_logits_ith()`.

**Blocker risk:** If sampling code is deeply tangled with context fields (e.g., grammar state, temperature history) beyond logits access, refactoring may be larger than anticipated. **ACTION:** Audit `sampling.cpp` for non-logits context dependencies (answer: none observed in initial scan; grammar is self-contained, sampler state is in `common_sampler` struct).

---

### B. State Save/Load (llama_state_*)

**Issue:** `llama_state_load_file(ctx, ...)` and `llama_state_save_file(ctx, ...)` are context methods, but state is really session state (K/V cache, positions) + decoder state (recurrent cells for MTP).

**D6/D10 Resolution (locked in PHASE45.md):**
- **Session state** (transformer K/V, positions) ← `llama_session_state_*()` APIs
- **Decoder state** (recurrent, scheduler) ← `llama_decoder_state_*()` APIs (future; not exposed in D5 stubs)
- D6 **only uses session state** (no recurrent/MTP). Simpler: file format stores session state only.

**Action for D6:**
- Replace `llama_state_load_file(ctx, path, tokens, ...)` with:
  ```cpp
  size_t state_size = llama_session_state_get_size(session);
  uint8_t * buf = new uint8_t[state_size];
  llama_session_state_set_data(session, file_bytes, file_size);
  ```
- This assumes file format is compatible with new session_state APIs (should be, as session owns K/V).

**Blocker risk:** If old checkpoint format is incompatible with new session_state format (unlikely, but possible if K/V layout changed). **ACTION:** Verify session.transformer_kv layout matches old llama_context.kv_self layout.

---

### C. Helper Functions Taking llama_context

**Tight coupling identified:**

1. **`common_tokenize(const llama_context *, ...)`** (3 overloads, common.cpp:4073-4131)
   - Uses: `llama_model_get_vocab(llama_context_get_model(ctx))`
   - **Fix:** Change signature to `common_tokenize(const llama_model *, ...)`, pass model directly.
   - **Impact:** Callers in main.cpp, sampling.cpp must change 20+ callsites, but mechanical.

2. **`common_token_to_piece(const llama_context *, ...)`** (common.cpp:4132)
   - Uses: vocab access only.
   - **Fix:** Change to `common_token_to_piece(const llama_model *, ...)` or `common_token_to_piece(const llama_vocab *, ...)`.
   - **Impact:** ~10 callers in main.cpp.

3. **`common_batch_add(llama_batch &, const llama_context *, ...)`** (common.h:638-648)
   - Uses: ???
   - **Check in D10:** Does not appear to use context fields; if only validation, may be removable.

4. **`yaml_dump_non_result_info(..., ctx, ...)`** (used in main.cpp:83)
   - Uses: Context for timing, model info.
   - **Fix:** Change to take `(session, decoder, model)` instead; extract timing from decoder via `llama_decoder_timings()`.

**Blocker risk:** Low; these are helpers, not hot-path. Can be refactored mechanically. **ACTION:** Refactor in D10 as part of common cleanup.

---

### D. Sampling Code (common/sampling.cpp)

**Lock (PHASE45.md):** "Sampling is fully external. Existing `llama_sampler_*` API handles chains; PHASE45 does not add sampling state to session/decoder."

**D6/D10 findings:**
- `common_sampler_init(const llama_model *, params)` — **already model-only, no change needed**.
- `common_sampler_sample_legacy(ctx_sampling, ctx_main, ctx_cfg)` — **signature changes to pass decoders**:
  ```cpp
  // OLD:
  llama_token common_sampler_sample_legacy(
      struct common_sampler * ctx_sampling,
      const struct llama_context * ctx_main,
      const struct llama_context * ctx_cfg
  )
  
  // NEW:
  llama_token common_sampler_sample_legacy(
      struct common_sampler * ctx_sampling,
      const struct llama_decoder * decoder_main,
      const struct llama_decoder * decoder_cfg
  )
  ```
- Inside the function, all `llama_get_logits_ith(ctx_main, idx)` → `llama_decoder_get_logits_ith(decoder_main, idx)`.

**Blocker risk:** None identified. Sampling state is fully self-contained in `common_sampler`; no context dependency. **Confidence:** High.

---

### E. LoRA / Control Vector Adapters

**Issue:** Current code stores LoRA/CV state in `llama_context`. After PHASE45, where do adapters live?

**Resolution (out of scope for D6, decision in D10):**
- Option A: Adapters live in `llama_session` (shared by all decoders using that session).
- Option B: Adapters live in `llama_model` (read-only, shared globally).
- **D6:** Defer; main.cpp doesn't use LoRA in the binding test. D10 must decide.

**Action:** D10 audit of llama_session_lora_adapter_* and llama_session_control_vector_* signatures in llama-session.h (lines 103-113). Already exposed in D5 stubs; no risk.

---

### F. Encoder-Decoder Models (e.g., T5, encoder path)

**Issue:** main.cpp has encoder support (line 557-573). Uses `llama_encode(ctx, batch)`.

**D6/D10 Resolution:**
- `llama_encode(ctx, batch)` → `llama_decoder_encode(decoder, batch)` (symmetric with decode).
- Decoder role is still PRIMARY; encoder is just a different graph builder.

**Blocker risk:** None if `llama_decoder_encode()` is in D5 stubs. **Check:** llama-decoder.h line 99 ✓ exists.

---

### G. KV Cache Defragmentation & Updates

**Issue:** `llama_kv_cache_seq_rm()`, `llama_kv_cache_seq_add()`, `llama_kv_cache_seq_div()` are called on context. After PHASE45, these are session methods.

**D6/D10 Resolution:**
- All three have new equivalents in `llama_session`: `llama_session_kv_seq_rm()`, etc. (llama-session.h lines 81-86).
- D6 uses all three (context shifting, Self-Extend, KV cache compaction).
- **No blocker:** APIs are already exposed.

---

### H. Performance Counters (Timing)

**Issue:** `llama_print_timings(ctx)` and `llama_dump_timing_info_yaml(logfile, ctx)` extract timing from context.

**D6/D10 Resolution:**
- Timing is **per-decoder** (verify vs draft have separate counters).
- New API: `llama_decoder_timings(decoder)` returns `llama_timings` struct.
- `llama_print_timings(ctx)` → `llama_decoder_perf_print(decoder)` (for human output).
- `yaml_dump_non_result_info()` → refactored to take decoders, not context.

**Blocker risk:** Low; already in D5 stubs (llama-decoder.h lines 112-114). **Action:** Verify D6 edit uses correct accessor.

---

## 7. Summary: D6 Edit LoC Impact

| Category | Lines | Notes |
|----------|-------|-------|
| Global declarations (lines 33-34) | 2 | Replace ctx with session + decoder |
| Context initialization (lines 208-214) | ~25 | Replace llama_init_from_model with session+decoder creation |
| Query context size (line 224) | 1 | llama_n_ctx(ctx) → llama_session_n_ctx(session) |
| KV cache operations (6 callsites) | 6 | llama_kv_cache_* → llama_session_kv_* |
| Guidance context cleanup (line 1004-1005) | 4 | llama_free(ctx) → decoder_free + session_free |
| Main decode loop (2 callsites: lines 703, 720) | 2 | llama_decode(ctx, ...) → llama_decoder_decode(decoder, ...) |
| Sampling call (line 752) | 1 | Signature change: pass decoders |
| Timing (lines 107, 1001) | 2 | llama_print_timings(ctx) → llama_decoder_perf_print(decoder) |
| **Subtotal (D6 scope: hot path only)** | **~43** | Minimal; greedy decode path only |
| **Tokenization/detokenization helpers (D10 refactor)** | ~20 | Common calls; defer to D10 |
| **State persistence (lines 260, 747, 998)** | ~9 | Change llama_state_* to llama_session_state_* |
| **Encoder path (line 561)** | 1 | llama_encode → llama_decoder_encode |
| **TOTAL (full main.cpp port)** | **~73** | If D10 completes full common refactor |

**D6 binary change expectation:** ~60 LoC changed or added (initialization setup longer; cleanup longer; hot decode loop shorter). Net: +10–15 LoC.

---

## 8. Blockers

**None identified at planning stage.** All new APIs are exposed in D5 stubs.

**Highest-risk areas for D6 validation:**
1. **Logits access via decoders:** Verify `llama_decoder_get_logits_ith(decoder, idx)` is wired correctly.
2. **State persistence:** Verify `llama_session_state_get_data/set_data` format matches old `llama_state_*` (likely yes; same K/V layout).
3. **Timing extraction:** Ensure `llama_decoder_perf_print()` or `llama_decoder_timings()` expose the right counters (e.g., token/sec, batch ms).

**D10 highest-risk areas:**
1. **Sampling signature change:** Ensure all 5 `llama_get_logits_ith(ctx_main/ctx_cfg, idx)` callsites in sampling.cpp are updated.
2. **Speculative decoding:** Audit ~8 `llama_decode()` + ~3 `llama_kv_cache_*()` callsites in speculative.cpp for compatibility.
3. **Common helpers:** Ensure tokenization/detokenization refactor to model-centric signatures doesn't break downstream users (server, profiles).

**No architectural blockers.** Proceed with confidence.

---

## 9. Next Steps

1. **D6 implementation:**
   - Port main.cpp per edit plan (§3).
   - Implement `common_session_params_to_llama()` and `common_decoder_params_to_llama()` helpers in common.cpp.
   - Update `llama_init_result` struct to include session + decoder.
   - Binding test: greedy-decode 50 tokens on Qwen 3.6 27B CPU; verify byte-identical output vs old API.

2. **D10 implementation:**
   - Refactor `common_tokenize()` and `common_token_to_piece()` to take model instead of context.
   - Update `sampling.cpp` logits accessors: `llama_get_logits_ith(ctx, idx)` → `llama_decoder_get_logits_ith(decoder, idx)`.
   - Audit speculative.cpp for `llama_decode()` + `llama_kv_cache_*()` replacements.
   - Delete llama_context from headers; ensure `git grep -l llama_context src/ common/ examples/server/` returns 0.

3. **D11 (honest renames, orthogonal):**
   - Rename internal fields: `llama_context.kv_self` → `llama_session.transformer_kv`, etc.
   - Code readability pass; no functional change.

---

**End of planning document.**
