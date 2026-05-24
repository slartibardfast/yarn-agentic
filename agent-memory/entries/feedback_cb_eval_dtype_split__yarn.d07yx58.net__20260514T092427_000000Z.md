---
name: libllama cb_eval residual-stream capture must branch on t->type (F16 vs F32)
description: l_out-<il> nodes are F32 for single-token decodes but F16 for multi-token (prefill, verify) ubatches. cb_eval hooks that resize a host float buffer by nbytes/sizeof(float) without checking t->type silently halve the row count for F16 ubatches and produce a constant offset gap downstream.
type: feedback
originSessionId: 9b52ee2b-d969-442c-83ba-98396525f6fa
---
Rule: any cb_eval-style host-side residual capture in libllama must branch on `t->type` and convert (e.g., `ggml_fp16_to_fp32_row(...)` for F16) before computing `nbytes / sizeof(float)`-based row counts. Assuming F32 silently halves the row count for F16 ubatches.

**Why:** The `l_out-<il>` node in ik_llama.cpp's Qwen 3.5/3.6 build graph is F32 for single-token decodes (per-cycle anchor, fallback) but F16 for multi-token ubatches (prefill, verify) — the build graph keeps batched residuals in F16 for performance. The original `llama_dflash_extract_cb_eval` in `src/llama.cpp` did:

```cpp
const size_t nbytes = ggml_nbytes(t);
buf.resize(old_n_floats + nbytes / sizeof(float));
ggml_backend_tensor_get(t, buf.data() + old_n_floats, 0, nbytes);
```

For F16 t with shape [5120, 103] (103 tokens × 5120 embd), `nbytes = 103 * 5120 * 2 = 1054720`, so `nbytes / sizeof(float) = 263680` floats. But the data is 103 rows of 5120 *half-precision* values, not 51.5 rows of 5120 *float* values. The buffer ended up with 51 rows of F32 data + 53 rows of garbage tail. Every subsequent `llama_dflash_draft` cycle saw `buf.size() / D_emb = 51 < mal_anchors = 103` and returned `rc = -5` ("extract buffer too short"). The constant 52-row gap stayed constant because per-cycle growth (5 verify rows of F32) and per-cycle trim (P + n_acc + 1) cancelled, never closing the original-prefill deficit.

**How to apply:** For any new cb_eval / host-side ggml_tensor capture hook in libllama:

```cpp
if (t->type == GGML_TYPE_F32) {
    buf.resize(old + (size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, buf.data() + old, 0, ggml_nbytes(t));
} else if (t->type == GGML_TYPE_F16) {
    std::vector<ggml_fp16_t> h_stage(ggml_nelements(t));
    ggml_backend_tensor_get(t, h_stage.data(), 0, ggml_nbytes(t));
    buf.resize(old + (size_t) ggml_nelements(t));
    ggml_fp16_to_fp32_row(h_stage.data(), buf.data() + old, ggml_nelements(t));
} else {
    // BF16 needs ggml_bf16_to_fp32_row; quant types don't apply to residuals.
    return true; // skip rather than corrupt buf
}
```

**Diagnostic signature** (when present): a constant-offset gap between "have" and "need" rows in any per-position downstream consumer, where the gap = (n_prompt_tokens / 2) and stays constant across cycles. If you see "have N, need N+K" with K stable across many failed cycles and K ≈ n_prompt_tokens / 2, this is the bug.

Fix landed in `src/llama.cpp llama_dflash_extract_cb_eval` 2026-05-13 (PHASE_DFLASH T8 Phase 1 closure). Env-gated `DFLASH_DIAG=1` stderr tracing in `stage_target_hiddens` and `llama_dflash_trim_extract` preserved for future regression triage of the same bug class.
