# PHASE 45 D1 — `llama_context` Field Audit

Source: `ik_llama.cpp/src/llama-context.h` (and nested `llama_kv_cache`).
Total fields: **61** in `llama_context`.

## Summary by destination

| Destination | Count | Notes |
|---|---:|---|
| **session** | 15 | model ref, cparams, kv_self (transformer K/V + cells), cvec, scale_data, lora_adapters, backend handles |
| **decoder** | 38 | inp_* tensors (19), MTP state (18), scheduler, output buffers, execution state |
| **ambiguous** | 5 | needs architectural decision before D5 |
| **delete** | 0 | nothing explicitly dead, but candidates flagged below |

## Critical finding: `s_l` is misplaced

`llama_kv_cache::s_l` (per-layer recurrent state for Qwen 3 Next / DeltaNet) is currently nested inside `kv_self` (a session-level concept), but logically **belongs per-decoder**. Draft and verify run different recurrent trajectories — they cannot share `s_l` without corruption. The `gpu_checkpoint` nested struct (`s_l_shadow`, `per_step_ssm`, `per_step_qkv`) amplifies the misclassification: it's speculative/executor-local state.

**PHASE45 implication:** session owns `cells[]`, `head`, `used`, `k_l[]`, `v_l[]`, `split_k_l[]`, `split_v_l[]`, defrag state. Decoder owns `s_l[slots][layers][hidden]`, `gpu_checkpoint`, recurrent rollback machinery.

## Ambiguous fields requiring decision

| Field | Today's home | Question | Tentative |
|---|---|---|---|
| `qnext_slot_alloc` | llama_context | Allocator state persistent across batches or reset per-batch? | decoder if persistent |
| `prev` (Prev*) | llama_context | Graph-reuse cache — per-executor or shared? | decoder (read llama-impl.cpp to confirm) |
| `cache_copies` | llama_context | Per-step CPY snapshots — global coordination or per-decoder independent? | decoder per-decoder (see PHASE38 graphcache lessons) |
| `gpu_checkpoint` (in kv_self) | session | Per-decoder snapshots or single sync point? | decoder |
| `sampling` | llama_context | One sampler across operations or per-decoder? | decoder per-decoder (cleaner ownership) |

## Definite assignments

### To `llama_session`

```
model              const llama_model &
cparams            llama_cparams           (some fields likely move to decoder; sub-audit in D5)
kv_self            llama_kv_cache          (transformer K/V — but s_l moves out)
cvec               llama_control_vector
scale_data         llama_scale_data
lora_adapters      vector<llama_lora_adapter*>
backend_*          backend handles (read-only access to model buffers)
```

### To `llama_decoder`

19 input tensors (decoder graph builds these):
```
inp_tokens, inp_embd, inp_pos, inp_out_ids,
inp_KQ_mask, inp_KQ_mask_swa, inp_KQ_mask_cross,
inp_K_shift, inp_mean, inp_cls,
inp_s_copy, inp_s_mask, inp_s_seq, inp_s_seq_qnext,
inp_pos_bucket, inp_embd_enc, inp_scale,
inp_mtp_states
```

18 MTP-specific fields (per-execution-role):
```
mtp_fused_compute_count
mtp_fused_results_argmax / _logprob / _logprobs
mtp_fused_offset_argmax / _logprob / _logprobs
mtp_fused_chain_residuals
mtp_fused_skip_extraction
mtp_fused_pending_gf / _pending_n_steps
mtp_fused_async_guess
mtp_persist_ctx / _buf / _array / _n
pending_chain_residual_step
mtp_cycle_counter
mtp_hook_fire_count
mtp_inline_decode_count
```

Other decoder-side: scheduler (`sched`), `ctx_compute`, output buffers (`logits`, `embeddings`, `output_ids`), batch tracking (`n_outputs`, `n_queued_tokens`), perf counters, `worst_graph_tokens`.

## Diagnostic / instrumentation cluster

Counters that bloat decoder noise (consider extracting to a `decoder_perf` substruct):
- `mtp_hook_fire_count`
- `mtp_inline_decode_count`
- `qnext_mixed_seq_fallback_count`
- `mtp_cycle_counter`

Not architectural; cosmetic for D11 honest-renames step.

## Questions for the human (resolve before D5)

1. **Per-decoder sampling** — is the verify decoder's sampler ever reused for draft? If yes, sampling stays context-global (or moves to spec_loop). If no, per-decoder is correct.
2. **`prev` cache scope** — does graph reuse benefit from cross-decoder sharing, or does verify always rebuild for its own graph and draft always rebuild for its own? Per-decoder is safer; cross-decoder needs a clear win to justify.
3. **Whether `cparams` itself needs splitting** — fields like `n_ctx` are session-global; fields like `mtp_fused_n_steps` are decoder-local. Worth a sub-audit in D5 when sketching headers.

These don't block D2/D3/D4 from completing; they shape D5's header design.
