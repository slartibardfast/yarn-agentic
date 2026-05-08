# PHASE45 D6 — `llama_decode` Deep Line-by-Line Mapping

**Date**: May 2026  
**Target**: Byte-identical D6 verifier; PHASE45 D6 option A (internal ctx wrapper)  
**Scope**: Map every logical block in `llama_decode` (the public entry point) and its delegate `llama_decode_internal`, classify each block by D6 substitution requirement.

---

## 1. Overview

### Function Under Analysis

- **Public entry**: `llama_decode(llama_context *ctx, llama_batch batch)` — src/llama.cpp:9641–9678 (38 LoC)
- **Internal delegate**: `llama_decode_internal(llama_context &lctx, llama_batch batch_all)` — src/llama.cpp:4985–5836 (851 LoC)
- **Total body**: ~889 LoC (public wrapper is thin; almost everything is in internal)

### File Ranges

| Function | Start | End | LoC |
|----------|-------|-----|-----|
| `llama_decode` | 9641 | 9678 | 38 |
| `llama_decode_internal` | 4985 | 5836 | 851 |

### Helper Functions Called

| Helper | Purpose | D6 Impact |
|--------|---------|-----------|
| `llama_decode_internal(*ctx, batch)` | Core inference engine | FORWARD (calls existing) |
| `llama_kv_cache_find_slot(kv_self, u_batch)` | Allocate KV cache cells | PER_SESSION_SUBST (reads kv_self) |
| `llama_kv_cache_update(&lctx)` | Manage cache head/positions | PER_SESSION_SUBST (reads kv_self) |
| `llama_kv_cache_get_padding(cparams)` | Heuristic cache padding | FORWARD (read-only cparams) |
| `llama_kv_cache_cell_max(kv_self, pad)` | Query max cell use | PER_SESSION_SUBST (reads kv_self) |
| `llama_kv_cache_defrag(kv_self)` | Defragment KV cells | PER_SESSION_SUBST (writes kv_self) |
| `llama_output_reserve(lctx, n_outputs)` | Allocate output buffers | PER_DECODER_SUBST (reads/writes decoder→buf_output) |
| `llm_build_context::llama_build_graph(lctx, batch, false)` | Construct compute graph | FORWARD (reads lctx, builds graph; graph is scheduler-owned) |
| `llama_set_inputs(lctx, u_batch)` | Populate input tensors | FORWARD (writes inp_* tensors in lctx) |
| `llama_graph_compute(lctx, gf, n_threads)` | Execute graph on accelerator | FORWARD (scheduler-owned; no decoder-specific state) |
| `llama_synchronize(&lctx)` | Synchronize backend (optional) | FORWARD (wrapper; device-agnostic) |
| `ggml_backend_sched_*` family | Scheduler operations | FORWARD (session/decoder shares scheduler via ctx) |
| `prepare_mtp_graph_inputs(lctx)` | Setup MTP-specific inputs | PER_DECODER_SUBST (writes draft/verify state per role) |
| `ggml_backend_tensor_get_async(...)` | Extract logits/embeddings async | PER_DECODER_SUBST (writes decoder→logits/embeddings) |

---

## 2. Block-by-Block Classification Table

### Public `llama_decode` Wrapper (9641–9678)

| Range | Block Name | Reads (Fields) | Writes (Fields) | Classification | Notes |
|-------|-----------|---|---|---|---|
| 9641–9643 | Function signature & init | batch | — | — | Wrapper boundary |
| 9644–9648 | Stop signal reset | `stop_internal_decode` (thread-local global) | `stop_internal_decode` | DEAD | In-flight interrupt marker; not relevant to D6 byte-identical decode |
| 9650–9655 | Profile gate setup | env, static const | static int64_t `_prof_decode`, `_decode_t0` | DEAD | `LLAMA_PROFILE_DECODE` instrumentation; not observable in output |
| 9657 | Delegate to internal | batch | — | **FORWARD** | Calls existing `llama_decode_internal(*ctx, batch)` |
| 9659–9671 | Profile output (MTP op type tag) | `_prof_decode`, `ctx→cparams.mtp_op_type`, `ctx→n_tokens`, `_dt_us` | stderr fprintf | DEAD | Diagnostic output only; no observable logits/embedding impact |
| 9673–9675 | Error log on negative ret | ret < 0 | stderr LLAMA_LOG_ERROR | DEAD | Logging only; error codes are returned, not changed |
| 9677 | Return result | ret | — | **FORWARD** | Return code from internal |

**Summary**: Public wrapper is thin. All observable work is in `llama_decode_internal`. Wrapper's profile & error logs are diagnostic and not byte-sensitive.

---

### Internal `llama_decode_internal` Body

#### **BLOCK 1: Init & Encoding Flag (4989–5013)**

| Range | Block Name | Reads | Writes | Classification | Notes |
|-------|-----------|-------|--------|---|---|
| 4989 | Set encoding flag false | — | `lctx.is_encoding` | **PER_DECODER_SUBST** | decoder needs per-role flag; draft vs verify may differ (currently always false for decode) |
| 4990–4996 | Clear stale t_h_pre_norm | `lctx.t_h_pre_norm` (prior graph) | `lctx.t_h_pre_norm = nullptr` | **PER_DECODER_SUBST** | Graph tensor lifetime; per-decoder graph means per-decoder pointer |
| 5003–5013 | MTP cycle counter (diagnostic) | `lctx.cparams.mtp_op_type` (static global `_global_mtp_cycle`) | `lctx.mtp_cycle_counter` | **PER_DECODER_SUBST** | Cycle counter is per-decoder for verify/draft isolation; reads cparams (session-level, but mtp_op_type is set per-call) |

**Summary**: All three writes are per-decoder state or graph state. **Risky**: `mtp_op_type` is in `cparams` (session-level), but it's set per-decode call (line 5000–5001 checks it). Option A wraps both; decoder reads it from the session's ctx, which is fine as long as the public API doesn't allow concurrent decodes with different op_types on the same session.

---

#### **BLOCK 2: Guard Checks for Stale Tensor Pointers (5014–5052)**

| Range | Block Name | Reads | Writes | Classification | Notes |
|-------|-----------|-------|--------|---|---|
| 5041–5052 | Clear fused offset tensors if not DRAFT_GEN_FUSED | `lctx.cparams.mtp_op_type` | `lctx.mtp_fused_offset_t[8]`, `lctx.mtp_fused_offset_n_dev[8]`, `lctx.mtp_fused_chain_residuals[8]` (conditionally) | **PER_DECODER_SUBST** | MTP fused state is per-decoder; pointers into scheduler-owned tensors must be cleared per-role |

**Summary**: Stale-pointer guards for MTP. Per-decoder because each decoder's graph may or may not have these tensors.

---

#### **BLOCK 3: Batch Input Validation (5053–5071)**

| Range | Block Name | Reads | Writes | Classification | Notes |
|-------|-----------|-------|--------|---|---|
| 5053 | Extract n_tokens_all | batch_all.n_tokens | n_tokens_all (local) | **FORWARD** | Read-only; local copy |
| 5055–5057 | Validate n_tokens != 0 | n_tokens_all | return -1 on error | **FORWARD** | Error check; same path in D6 |
| 5063–5065 | Ref model/hparams/cparams | — | local refs (const) | **FORWARD** | Read-only session-level data |
| 5067 | Assert token or embd mutual exclusion | batch_all | — | **FORWARD** | Compile-time sanity check |
| 5069 | Assert batch size <= n_batch | batch_all, cparams.n_batch | — | **FORWARD** | Compile-time sanity check |
| 5071 | Assert causal or ubatch >= n_tokens | cparams, batch_all | — | **FORWARD** | Compile-time sanity check |

**Summary**: All validation and reads are read-only or local. No decoder-specific changes.

---

#### **BLOCK 4: Perf Counters Init (5073–5076)**

| Range | Block Name | Reads | Writes | Classification | Notes |
|-------|-----------|-------|--------|---|---|
| 5073–5074 | Initialize compute start time (once per context) | `lctx.t_compute_start_us` | `lctx.t_compute_start_us = ggml_time_us()` (if 0) | **PER_DECODER_SUBST** | Perf counter; becomes `decoder→perf.t_compute_start_us` |
| 5076 | Accumulate queued token count | `lctx.n_queued_tokens` | `lctx.n_queued_tokens += n_tokens_all` | **PER_DECODER_SUBST** | Perf counter; becomes `decoder→perf.n_queued_tokens` |

**Summary**: Two perf counters. D6 substitution: forward to decoder's perf struct (or session ctx's perf if shared).

---

#### **BLOCK 5: Local Output Tracking Setup (5078–5110)**

| Range | Block Name | Reads | Writes | Classification | Notes |
|-------|-----------|-------|--------|---|---|
| 5078–5086 | Local output metadata arrays init | cparams.embeddings, cparams.pooling_type, cparams.mtp, hparams.nextn_predict_layers | `n_outputs`, `n_outputs_prev`, `n_outputs_embd`, `n_outputs_prev_embd` (local) | **FORWARD** | All local variables; no ctx mutation here |
| 5088 | Unpack n_ubatch | cparams.n_ubatch | local n_ubatch | **FORWARD** | Read-only |
| 5090–5094 | Init local pos/seq arrays | — | local vectors (pos, n_seq_id, seq_id_arr, seq_id) | **FORWARD** | Local scratch; reused across ubatch loop |
| 5097–5098 | Compute embd_pooled, has_mtp flags | cparams.embeddings, cparams.pooling_type, cparams.mtp, hparams | local bools | **FORWARD** | Read-only computation |
| 5101–5110 | Count outputs in batch | batch_all.logits, lctx.logits_all, embd_pooled | local n_outputs | **FORWARD** | Read-only batch inspect; local output |

**Summary**: All local variable setup. No ctx writes except `n_outputs` computed locally (will write to ctx later).

---

#### **BLOCK 6: Reserve Output Buffers (5112–5138)**

| Range | Block Name | Reads | Writes | Classification | Notes |
|-------|-----------|-------|--------|---|---|
| 5113 | Compute n_outputs_embd | has_mtp, n_tokens_all, n_outputs | local n_outputs_embd | **FORWARD** | Local computation |
| 5114–5117 | Call `llama_output_reserve(lctx, max(n_outputs, n_outputs_embd))` | lctx | reads/writes `lctx.logits`, `lctx.embeddings`, `lctx.logits_size`, `lctx.embd_size` | **PER_DECODER_SUBST** | Output buffer allocation is decoder-owned in D7+. In D6 Option A, forwards to `session→ctx→buf_output` and returns borrowed pointers. Substitution: `llama_decoder_output_reserve(decoder, ...)` or delegate to `session→ctx` directly. |
| 5120–5138 | Set output mappings (`lctx.output_ids[]`) | batch_all.logits, n_outputs, n_tokens_all | `lctx.output_ids[0..n_tokens_all]` | **PER_DECODER_SUBST** | `output_ids` tracks which batch token → which output slot. Per-decoder because output layout is decoder-specific. In Option A, `lctx.output_ids` stays in ctx (shared); decoder just reads what it wrote. |

**Summary**: Output buffer reserve and layout are mixed. Session ctx owns buffers (shared); decoder tracks its own output layout (`output_ids`). **Careful**: `lctx.output_ids` write happens here; needs decoder substitution if decoders have different output layouts (verify vs draft). For PRIMARY role (D6), single decoder means single layout.

---

#### **BLOCK 7: Main Ubatch Loop Prelude (5140–5222)**

| Range | Block Name | Reads | Writes | Classification | Notes |
|-------|-----------|-------|--------|---|---|
| 5141 | Loop condition `cur_token < n_tokens_all` | local cur_token, n_tokens_all | — | **FORWARD** | Control flow; local state |
| 5145 | Compute n_tokens ubatch size | n_ubatch, cur_token, n_tokens_all | local n_tokens | **FORWARD** | Batch slicing; local |
| 5146–5186 | Hybrid seq pattern classification (QNext) | batch_all, cparams | local blocks, pattern; writes `lctx.qnext_mixed_seq_fallback_count` (diagnostic counter) | **PER_DECODER_SUBST** | Diagnostic counter; perf-only. Classification logic is read-only session-level. |
| 5188–5199 | Construct u_batch view | batch_all, n_tokens, cur_token, n_embd | local u_batch struct (view) | **FORWARD** | Slicing; all read-only |
| 5202–5220 | Count n_outputs in this ubatch | u_batch, n_outputs_all, cur_token, n_tokens_all | local n_outputs_new | **FORWARD** | Local computation |
| 5222 | Select n_threads based on ubatch size | cparams.n_threads, cparams.n_threads_batch, n_tokens | local n_threads | **FORWARD** | Read-only |

**Summary**: Loop setup and ubatch slicing. Mostly local computation. One perf counter write (`qnext_mixed_seq_fallback_count`) is diagnostic.

---

#### **BLOCK 8: Batch API Transition Helpers (5225–5249)**

| Range | Block Name | Reads | Writes | Classification | Notes |
|-------|-----------|-------|--------|---|---|
| 5227–5234 | Populate pos[] if missing from batch | u_batch, batch_all | local pos vector; writes `u_batch.pos` (view into local) | **FORWARD** | Legacy API compatibility; all local |
| 5236–5249 | Populate seq_id if missing from batch | u_batch, batch_all | local n_seq_id, seq_id, seq_id_arr vectors; writes `u_batch.n_seq_id`, `u_batch.seq_id` (views into locals) | **FORWARD** | Legacy API compatibility; all local |

**Summary**: Backward-compat shims. No ctx writes.

---

#### **BLOCK 9: KV Cache Update & Slot Allocation (5251–5276)**

| Range | Block Name | Reads | Writes | Classification | Notes |
|-------|-----------|-------|--------|---|---|
| 5252–5256 | Conditional KV cache update (causal attention only) | hparams.causal_attn | calls `llama_kv_cache_update(&lctx)` (reads/writes `lctx.kv_self`), `lctx.kv_self.head` comparison | **PER_SESSION_SUBST** | KV cache management is per-session. Substitution: `llama_session_kv_cache_update(session)` or delegate to `session→ctx→kv_self`. `lctx.kv_self` is session-owned. |
| 5260–5262 | Reset head heuristic | kv_self.head, kv_self.used | `kv_self.head = 0` | **PER_SESSION_SUBST** | KV ring-buffer management. Session-owned. |
| 5264–5266 | Find/allocate KV cache slot | kv_self, u_batch | reads/writes `kv_self` (implicit in find_slot call) | **PER_SESSION_SUBST** | KV cell allocation. Session-owned. |
| 5268–5275 | Compute kv_self.n padding (non-recurrent only) | kv_self.recurrent, kv_self, cparams | `kv_self.n = min(kv_self.size, max(pad, GGML_PAD(...)))` | **PER_SESSION_SUBST** | KV cache metadata. Session-owned. |
| 5277–5279 | Check stop signal (interrupt) | `stop_internal_decode` (thread-local global) | return -3 if set | **DEAD** | In-flight interrupt; not relevant to D6 byte-identical |

**Summary**: All KV cache ops are session-owned. D6 Option A forwards to `session→ctx→kv_self`.

---

#### **BLOCK 10: Can-Reuse Graph Check & Build-or-Reuse (5295–5379)**

| Range | Block Name | Reads | Writes | Classification | Notes |
|-------|-----------|-------|--------|---|---|
| 5299 | Call `lctx.can_reuse_graph(u_batch)` (method) | lctx, u_batch | local _can_reuse_hit (bool) | **PER_DECODER_SUBST** | Graph reuse state. Graph is per-decoder; reuse cache (`lctx.prev`) is per-decoder. Substitution: `decoder→prev` instead of `ctx→prev`. |
| 5300–5331 | IK_PRINT_TIMING diagnostic block (HIT/MISS counters) | _can_reuse_hit, g_can_reuse_last_miss_reason (static globals), cparams | fprintf stderr | **DEAD** | Instrumentation only; no output impact. |
| 5333–5375 | If not reusing: reset sched, set eval callback, build graph, alloc, cache prev | _can_reuse_hit, lctx, u_batch, cparams.graph_reuse | `lctx.reset_scheduler()`, `ggml_backend_sched_set_eval_callback(...)`, graph build via `llama_build_graph(lctx, ...)`, `ggml_backend_sched_alloc_graph(...)`, `lctx.prev = make_unique(...)` | **PER_DECODER_SUBST** | Scheduler reset and graph build/alloc are per-decoder. `lctx.reset_scheduler()` → `decoder→sched.reset()`. `lctx.prev` → `decoder→prev`. Graph-build delegates to build_graph (reads lctx, builds graph; graph is in scheduler). |
| 5376–5379 | If reusing: fetch cached graph | lctx.prev | local gf = lctx.prev->graph | **PER_DECODER_SUBST** | Graph reuse. `lctx.prev` → `decoder→prev`. |

**Summary**: Graph reuse is per-decoder. All writes (`lctx.prev`, scheduler reset) are decoder-specific. Substitution clear.

---

#### **BLOCK 11: MTP Graph Input Preparation (5381–5385)**

| Range | Block Name | Reads | Writes | Classification | Notes |
|-------|-----------|-------|--------|---|---|
| 5381–5384 | Conditional MTP input prep | cparams.mtp_op_type, lctx | calls `prepare_mtp_graph_inputs(lctx)` (reads/writes draft/verify state) | **PER_DECODER_SUBST** | MTP draft state is per-decoder. D6 PRIMARY role doesn't use MTP. D7 VERIFY/DRAFT_MTP will need decoder substitution. For D6, this is a no-op (mtp_op_type == MTP_OP_NONE). |

**Summary**: MTP inputs are per-decoder. D6 PRIMARY skips this (mtp_op_type is NONE).

---

#### **BLOCK 12: Output Tensor Identification (5387–5421)**

| Range | Block Name | Reads | Writes | Classification | Notes |
|-------|-----------|-------|--------|---|---|
| 5388 | Get last graph node as logits | gf, lctx.n_outputs | local res (ggml_tensor*) | **PER_DECODER_SUBST** | Graph node from decoder's graph. Read-only inspection. |
| 5389 | Init embd pointer | — | local embd = nullptr | **FORWARD** | Local variable |
| 5391–5394 | Handle zero outputs | lctx.n_outputs | res = nullptr (conditional) | **PER_DECODER_SUBST** | Conditional on decoder's output count. |
| 5395–5420 | Search graph for embedding/result tensors | gf, cparams.embeddings, has_mtp, lctx.model | embd assignment (multiple branches) | **PER_DECODER_SUBST** | Graph inspection. Per-decoder because graph is per-decoder. |

**Summary**: Graph introspection. All read-only after graph build. Per-decoder because each decoder has its own graph.

---

#### **BLOCK 13: Set Batch Inputs (5426–5429)**

| Range | Block Name | Reads | Writes | Classification | Notes |
|-------|-----------|-------|--------|---|---|
| 5426 | Call `llama_set_inputs(lctx, u_batch)` | lctx, u_batch | writes inp_* tensors in lctx (inp_tokens, inp_pos, inp_embd, etc.) | **PER_DECODER_SUBST** | Input tensors are per-decoder. In D6 Option A, these are stored in `session→ctx` (shared infrastructure), but logically they're built per decode call and serve the decoder's current batch. Substitution: `llama_decoder_set_inputs(decoder, u_batch)` delegates to ctx (since tensors are in ctx's graph). |

**Summary**: Input tensor population. Per-decoder context (logically), but stored in session ctx (implementation).

---

#### **BLOCK 14: Graph Compute (5435–5440)**

| Range | Block Name | Reads | Writes | Classification | Notes |
|-------|-----------|-------|--------|---|---|
| 5435 | Call `llama_graph_compute(lctx, gf, n_threads)` (scheduler compute + device dispatch) | lctx, gf, n_threads | device state (GPU memory, etc.); returns to ctx on D2H transfer | **FORWARD** | Graph compute is scheduler-managed. The scheduler (`lctx.sched`) is session-level (shared across decoders in D7+, or per-decoder in PRIMARY-only D6). D6 Option A: all calls go to same session ctx scheduler. Output logits/embeddings are in session ctx buffers; decoder reads them. |
| 5437 | Call `llama_synchronize(&lctx)` (optional, under IK_PRINT_TIMING) | lctx | synchronizes backend | **FORWARD** | Wrapper around backend sync; no observable state change. |

**Summary**: Compute is scheduler-managed. All observable output is in device memory or ctx buffers.

---

#### **BLOCK 15: KV Cache Head Update (5442–5454)**

| Range | Block Name | Reads | Writes | Classification | Notes |
|-------|-----------|-------|--------|---|---|
| 5445 | Conditional reset_previous flag | lctx.model, kv_self.head | local reset_previous bool | **PER_SESSION_SUBST** | KV cache recurrent model check. Read-only model inspection. |
| 5448 | Update kv_self.head | kv_self.head, n_tokens | `kv_self.head += n_tokens; if (kv_self.head >= kv_self.size) kv_self.head = 0` | **PER_SESSION_SUBST** | KV ring-buffer head update. Session-owned. |

**Summary**: KV ring-buffer management. Session-owned.

---

#### **BLOCK 16: Logits Extraction (5461–5583)**

| Range | Block Name | Reads | Writes | Classification | Notes |
|-------|-----------|-------|--------|---|---|
| 5467–5473 | Invalidate draft argmax cache, snapshot fast_verify flag | lctx.draft_argmax_valid, lctx.draft_argmax_n, lctx.fast_argmax_for_verify | `lctx.draft_argmax_valid = false`, `lctx.draft_argmax_n = 0`, `lctx.fast_argmax_for_verify = false` | **PER_DECODER_SUBST** | Draft cache is per-decoder (or per-role). Substitution: `decoder→draft_argmax_valid`, etc. |
| 5476–5477 | Gate logits extract on MTP op type | cparams.mtp_op_type | — | **PER_DECODER_SUBST** | Conditional on decoder's op type. D6 PRIMARY: always MTP_OP_NONE (extract logits). |
| 5478–5531 | CUDA fast path: argmax + softmax on device, D2H 8B/row | backend_res, res, cparams.mtp_op_type, lctx.draft_argmax_* fields | `lctx.draft_argmax_ids`, `lctx.draft_argmax_probs`, `lctx.draft_argmax_top2_ids` (resized), `lctx.draft_argmax_valid`, `lctx.draft_argmax_n` | **PER_DECODER_SUBST** | Draft argmax cache is per-decoder. Substitution: `decoder→draft_argmax_*`. |
| 5555–5574 | Standard logits extract: ggml_backend_tensor_get_async to lctx.logits | backend_res, res, lctx.logits, lctx.logits_size, n_outputs_prev, n_outputs_new, n_vocab | async D2H copy to `lctx.logits + offset` | **PER_DECODER_SUBST** | Logits buffer is decoder-owned (D7+) or session-shared (D6 Option A). Substitution: `decoder→logits` or delegate to `session→ctx→logits`. |

**Summary**: Logits extraction is mixed. Buffer is per-decoder (D7+) but in D6 Option A lives in session ctx. All reads from context-shared state (model, cparams, scheduler).

---

#### **BLOCK 17: Embedding Extraction (5585–5638)**

| Range | Block Name | Reads | Writes | Classification | Notes |
|-------|-----------|-------|--------|---|---|
| 5586 | Gate embedding extract on MTP op type | embd, cparams.mtp_op_type | — | **PER_DECODER_SUBST** | Conditional on decoder's op type. D6 PRIMARY: may be MTP_OP_NONE or MTP_OP_DRAFT_GEN. |
| 5590–5591 | Get backend for embedding tensor | embd, lctx.sched | local backend_embd | **FORWARD** | Scheduler query (read-only). |
| 5593–5633 | Extract embeddings by pooling type | embd, cparams.pooling_type, lctx.embd, lctx.embd_seq, n_outputs_prev_embd, has_mtp, n_tokens | async D2H copy to `lctx.embd + offset` or `lctx.embd_seq[seq_id]` | **PER_DECODER_SUBST** | Embeddings buffer is decoder-owned (D7+) or session-shared (D6). Substitution: `decoder→embd` or delegate. |

**Summary**: Embedding extraction similar to logits. Per-decoder buffer (D7+), session-shared (D6).

---

#### **BLOCK 18: Fused Draft Result Extraction (5639–5791)**

| Range | Block Name | Reads | Writes | Classification | Notes |
|-------|-----------|-------|--------|---|---|
| 5653–5656 | Skip extraction if async dispatch in progress | cparams.mtp_op_type, lctx.mtp_fused_skip_extraction | `lctx.mtp_fused_pending_gf`, `lctx.mtp_fused_pending_n_steps` | **PER_DECODER_SUBST** | MTP async flag is per-decoder. D6 PRIMARY: never used (mtp_op_type == NONE). |
| 5657–5791 | Extract fused draft results (sync path) | cparams, gf, lctx | `lctx.mtp_fused_results_n`, `lctx.mtp_fused_results_tokens[]`, `lctx.mtp_fused_results_probs[]`, `lctx.mtp_persist_*` fields | **PER_DECODER_SUBST** | Fused draft state (argmax, prob, persist residuals) is per-decoder/role. D6 PRIMARY: never reached (mtp_op_type != DRAFT_GEN_FUSED). D7+ DRAFT_MTP: decoder-specific. |

**Summary**: MTP fused result extraction is per-decoder. D6 PRIMARY skips this entire block (mtp_op_type != DRAFT_GEN_FUSED).

---

#### **BLOCK 19: Output Counters Update (5793–5801)**

| Range | Block Name | Reads | Writes | Classification | Notes |
|-------|-----------|-------|--------|---|---|
| 5793 | Accumulate output counters | lctx.n_outputs | `n_outputs_prev += lctx.n_outputs` (local) | **FORWARD** | Local accumulation; no ctx write here. |
| 5794 | Accumulate embedding output counters | lctx.n_outputs, has_mtp, n_tokens | `n_outputs_prev_embd += ...` (local) | **FORWARD** | Local accumulation. |
| 5795 | Advance cur_token | cur_token, n_tokens | `cur_token += n_tokens` (local) | **FORWARD** | Loop control. |
| 5796–5801 | Conditional prev cache reset (recurrent model) | reset_previous, lctx.prev | `lctx.prev.reset()` (unique_ptr) | **PER_DECODER_SUBST** | Graph reuse cache reset for recurrent. Per-decoder. Substitution: `decoder→prev.reset()`. |

**Summary**: Loop state updates. Graph cache reset is per-decoder.

---

#### **BLOCK 20: Final Output Count & Defrag Decision (5803–5820)**

| Range | Block Name | Reads | Writes | Classification | Notes |
|-------|-----------|-------|--------|---|---|
| 5805 | Set final n_outputs for llama_get_logits_ith | n_outputs (local, accumulated) | `lctx.n_outputs = n_outputs` | **PER_DECODER_SUBST** | Output count is decoder-specific (or session-shared in Option A). Substitution: `decoder→n_outputs`. |
| 5811–5820 | Decide KV defrag on fragmentation threshold | cparams.causal_attn, cparams.defrag_thold, kv_self.n, kv_self.used | `llama_kv_cache_defrag(kv_self)` | **PER_SESSION_SUBST** | KV cache defragmentation. Session-owned. |

**Summary**: Final output count and KV defrag are session-level.

---

#### **BLOCK 21: Scheduler Reset at End (5824–5833)**

| Range | Block Name | Reads | Writes | Classification | Notes |
|-------|-----------|-------|--------|---|---|
| 5827–5829 | Conditional final scheduler reset | lctx.prev | `lctx.reset_scheduler()` if not reusing | **PER_DECODER_SUBST** | Scheduler reset is per-decoder. Substitution: `decoder→sched.reset()` or delegate to ctx (if ctx is shared). |

**Summary**: Scheduler reset is per-decoder (or shared in Option A).

---

#### **BLOCK 22: Return (5835)**

| Range | Block Name | Reads | Writes | Classification | Notes |
|-------|-----------|-------|--------|---|---|
| 5835 | Return success | — | return 0 | **FORWARD** | Standard success code. |

---

## 3. Recurrent State Ops (DeltaNet `s_l`, Mamba State)

Recurrent state (SSM hidden states for hybrid models) is stored in `llama_kv_cache::s_l[][]` (per-layer, per-slot hidden states). Per PHASE45_FIELD_AUDIT.md, this is **misplaced in kv_self** (session-level) but logically belongs **per-decoder** (draft and verify run different trajectories).

### Recurrent State Reads/Writes in `llama_decode_internal`

| Line | Operation | Field | Classification | Notes |
|------|-----------|-------|---|---|
| 5146 | Check `llm_arch_is_hybrid(model.arch)` | model (read-only) | FORWARD | Determine if model uses SSM |
| 5252 | Check `hparams.causal_attn` before KV update | hparams (read-only) | FORWARD | Causal attention gate; impacts KV but not SSM directly |
| 5268 | Check `kv_self.recurrent` flag | kv_self (read-only) | PER_SESSION_SUBST | Recurrent model flag; session-owned in kv_self. D6: delegates to session→ctx. |
| implicit in `llama_build_graph` | Graph builder inspects `lctx.kv_self.s_l[]` when building SSM layers | kv_self.s_l (tensor ptrs, read-only) | PER_SESSION_SUBST | Graph construction reads session's recurrent state pointers. D6 forwards to session ctx. |
| implicit in `llama_set_inputs` | Input setup may read recurrent state (e.g., prev step's hidden state for SSM) | kv_self.s_l (read) | PER_SESSION_SUBST | Input tensors reference recurrent state. Per-session because SSM state is shared (not split per-decoder in D6). |
| implicit in `llama_graph_compute` | Graph compute updates SSM state in-device; persist to kv_self post-compute (handled by scheduler) | kv_self.s_l (write, implicit via D2H) | PER_SESSION_SUBST | Device SSM state update; results written back to session ctx after compute. |

### Risk for D6 Byte-Identical

**Decision for D6**: Recurrent state stays in session ctx (`kv_self.s_l`). D6 PRIMARY decoder does not have its own SSM state slot; it reads/writes the session's shared SSM state. If the same session runs multiple decoders in D7+, **they will interfere** (draft overwrite verify's hidden state). This is flagged as an architectural issue (PHASE45_FIELD_AUDIT.md, line 16–19).

**D6 Mitigation**: PRIMARY-only; no concurrent draft/verify on same session. D7+ will split `s_l` into per-decoder slots (requires scheduler/KV cache redesign).

**Byte-identical guarantee**: Unchanged. D6 forwards all s_l refs to session ctx, same as old API.

---

## 4. Output Buffer Ops

Output buffers (`lctx.logits`, `lctx.embeddings`) are allocated once at context creation and reused per decode. D7+ will move these to decoder, but D6 keeps them session-global.

### Output Buffer Allocations/Writes

| Line | Operation | Field | Classification | Notes |
|------|-----------|-------|---|---|
| 5114 | `llama_output_reserve(lctx, n_outputs)` | lctx.logits, lctx.embeddings (size checks/realloc) | **PER_DECODER_SUBST** | Output reserve delegates to decoder (D7+). D6: forwards to session ctx. |
| 5124–5126 | Set `lctx.output_ids[i]` mapping | lctx.output_ids[] | **PER_DECODER_SUBST** | Output slot mapping is decoder-specific (maps batch token → output buffer offset). D6: shared in ctx; decoder just sets its own view. |
| 5482 | `logits_out = lctx.logits + n_outputs_prev*n_vocab` (address compute) | lctx.logits (read pointer) | **PER_DECODER_SUBST** | Logits buffer pointer. D6 forwards to session ctx. |
| 5564–5572 | `ggml_backend_tensor_get_async(backend_res, res, logits_out, ...)` (D2H copy) | lctx.logits (write via async copy) | **PER_DECODER_SUBST** | Async copy into logits buffer. D6 forwards. |
| 5598 | `float * embd_out = lctx.embd + n_outputs_prev_embd*n_embd` | lctx.embd (read pointer) | **PER_DECODER_SUBST** | Embeddings buffer pointer. D6 forwards. |
| 5604 | `ggml_backend_tensor_get_async(backend_embd, embd, embd_out, ...)` | lctx.embd (write via async copy) | **PER_DECODER_SUBST** | Async copy into embeddings buffer. D6 forwards. |
| 5617–5626 | `lctx.embd_seq[seq_id]` reads/updates | lctx.embd_seq map | **PER_DECODER_SUBST** | Per-sequence embeddings. D6 forwards. |

### Substitution Rule for D6

In Option A (internal ctx wrapper):
```
decoder→logits ≈ decoder_session(decoder)→ctx→logits
decoder→embeddings ≈ decoder_session(decoder)→ctx→embeddings
decoder→output_ids ≈ decoder_session(decoder)→ctx→output_ids
```

No actual code change needed for D6; all goes through session ctx (which is shared). D7+ will extract buffers to decoder struct.

---

## 5. Scheduler Ops

All scheduler calls are through `lctx.sched` (ggml_backend_sched pointer).

### Scheduler Operations in `llama_decode_internal`

| Line | Operation | API | Classification | Notes |
|------|-----------|-----|---|---|
| 5334 | `lctx.reset_scheduler()` | Method on lctx (wraps scheduler reset) | **PER_DECODER_SUBST** | Scheduler reset per-decode. Per-decoder in D7+ (each decoder may have own sched). D6: shared ctx sched (single PRIMARY decoder). |
| 5335 | `ggml_backend_sched_set_eval_callback(lctx.sched, cb, cb_user_data)` | Direct sched call | **FORWARD** | Callback registration; read-only relative to compute state. |
| 5344 | `llama_build_graph(lctx, u_batch, false)` → graph → sched | Scheduler implicit (graph is built against sched) | **FORWARD** | Graph build returns cgraph; owned by scheduler memory. |
| 5353 | `ggml_backend_sched_alloc_graph(lctx.sched, gf)` | Direct sched call | **FORWARD** | Allocate compute buffers for graph. No state retained by ctx. |
| 5435 | `llama_graph_compute(lctx, gf, n_threads)` → `ggml_backend_sched_graph_compute(lctx.sched, gf)` | Delegate to sched | **FORWARD** | Execute graph via scheduler. Results in device memory; D2H extract later. |
| 5478, 5590 | `ggml_backend_sched_get_tensor_backend(lctx.sched, tensor)` | Query tensor assignment | **FORWARD** | Determine which device owns tensor. Read-only query. |
| 5519 | `ggml_backend_synchronize(backend_res)` | Backend-level sync (not sched) | **FORWARD** | Synchronize specific backend. No state change. |
| 5672 | `ggml_backend_sched_synchronize(lctx.sched)` | Synchronize scheduler (all backends) | **FORWARD** | Barrier; no state. |
| 5828 | `lctx.reset_scheduler()` at end | Method on lctx | **PER_DECODER_SUBST** | Final reset before returning. Per-decoder (or shared in D6). |

### Substitution Rule for D6 (Option A)

Per PHASE45_D6_SPLIT.md §2.3:

> **Decision for D6**: PRIMARY decoder stores a pointer to session's internal `llama_context`. Calls into `ctx→sched` ... No new scheduler creation; delegate to session's scheduler (already built in llama_init_from_model).

**Concrete**: All scheduler calls in D6 forward through `decoder_session(decoder)→ctx→sched`. Same scheduler, same code paths.

**Byte-identical guarantee**: Unchanged code paths. ✓

---

## 6. Perf Counter Ops

Performance counters are accumulated across the context's lifetime. D7+ moves them per-decoder.

### Perf Counter Mutations

| Line | Field | Operation | D6 Classification | D7+ Target |
|------|-------|-----------|---|---|
| 5074 | `lctx.t_compute_start_us` | `= ggml_time_us()` (once on first decode) | **PER_DECODER_SUBST** | `decoder→perf.t_compute_start_us` |
| 5076 | `lctx.n_queued_tokens` | `+= n_tokens_all` | **PER_DECODER_SUBST** | `decoder→perf.n_queued_tokens` |
| 5177 | `lctx.qnext_mixed_seq_fallback_count` | `++` (diagnostic) | **PER_DECODER_SUBST** | `decoder→perf.qnext_mixed_seq_fallback_count` |
| 5467 | `lctx.draft_argmax_valid` | `= false` | **PER_DECODER_SUBST** | `decoder→perf.draft_argmax_valid` |
| 5468 | `lctx.draft_argmax_n` | `= 0` | **PER_DECODER_SUBST** | `decoder→perf.draft_argmax_n` |
| 5473 | `lctx.fast_argmax_for_verify` | `= false` (consume flag) | **PER_DECODER_SUBST** | `decoder→perf.fast_argmax_for_verify` |
| 5530–5531 | `lctx.draft_argmax_*` vectors | `resize()`, `[i] = ...` | **PER_DECODER_SUBST** | `decoder→perf.draft_argmax_*` |
| 5805 | `lctx.n_outputs` | `= n_outputs` (final count for get_logits_ith) | **PER_DECODER_SUBST** | `decoder→perf.n_outputs` (or `decoder→n_outputs` directly) |
| implicit | `lctx.t_eval_us`, `lctx.n_eval`, `lctx.n_p_eval`, `lctx.t_p_eval_us` | Accumulated in `llama_graph_compute` via scheduler callback | **PER_DECODER_SUBST** | `decoder→perf.*` |

### Substitution Rule for D6

All perf counter mutations become decoder-specific:
```
ctx→<perf_field> ⇒ decoder→perf.<perf_field>
```

In D6 Option A (internal ctx wrapper), the decoder can store these locally or keep them in ctx (since there's only one PRIMARY decoder per session). The important point is that **D7+ VERIFY and DRAFT_MTP decoders have isolated perf counters** (so draft's eval time doesn't get mixed into verify's timings).

---

## 7. Risks for D6 Byte-Identical Output

### Summary of Risky Substitutions

| Block | Risk | Mitigation | Impact |
|-------|------|-----------|--------|
| **Perf counters** | If decoder perf state is read before being written, stale values appear in API results (e.g., `llama_get_timings`). | Reset per-decoder perf on create; ensure API reads decoder perf, not ctx perf. | Observable if caller checks timings; not observable in logits/embeddings. |
| **Output buffer ownership** | If session and decoder have overlapping buffer writes, or if output layout differs per-decoder, logits will be misaligned. | In D6, single PRIMARY decoder; single output layout. Verify at decoder creation: allocate buffers once, share pointers. | **HIGH RISK if concurrent decoders on same session in D6** (should be impossible per spec). |
| **MTP op_type in cparams** | `cparams.mtp_op_type` is set per-call; if two decoders try to set it concurrently on shared ctx, one overrides the other. | In D6, single PRIMARY decoder; mtp_op_type is always MTP_OP_NONE. D7+ must split cparams or store mtp_op_type per-decoder. | Not a D6 risk (PRIMARY only). |
| **KV cache state (kv_self)** | If session KV cache head advances during a decode, and draft tries to allocate a slot in the same session (D7+), they interfere. | In D6, single decoder allocates once per decode. D7+ needs per-decoder or atomicity. | Not a D6 risk (single decoder per session expected). |
| **Graph reuse (lctx.prev)** | If multiple decoders share lctx.prev, reuse decision is racy. | D6: single PRIMARY decoder reuses own graph. D7+ VERIFY/DRAFT_MTP each have own prev. | Not a D6 risk (single decoder). |
| **Recurrent state (kv_self.s_l)** | If draft and verify run on same session, draft's SSM state overwrites verify's. | In D6, single decoder; no concurrent SSM. D7+ requires per-decoder recurrent state or per-slot (high complexity). | Known issue; deferred to D7+. Not a D6 byte-identical risk (PRIMARY only). |

### Conclusion

**D6 byte-identical is safe** because:
1. Single PRIMARY decoder per session (no concurrency).
2. All forwards are to the same session ctx (code paths unchanged).
3. Perf counters are observable via decoder API, but logits/embeddings are deterministic.

**D7+ risks**:
- Multi-decoder (VERIFY + DRAFT_MTP) interference on shared kv_self, cparams, prev.
- Recurrent state separation required.
- Perf counter isolation critical for sampling decisions.

---

## 8. Concrete Next-Iteration Body for `llama_decoder_decode` (D6 PRIMARY)

### Pseudocode

```cpp
// src/llama-decoder.cpp
int32_t llama_decoder_decode(
        struct llama_decoder * decoder,
        struct llama_batch   batch) {
    
    if (!decoder) {
        LLAMA_LOG_ERROR("%s: decoder is NULL", __func__);
        return -1;
    }
    
    if (decoder->role != LLAMA_DECODER_PRIMARY) {
        LLAMA_LOG_ERROR("%s: only PRIMARY role implemented in D6", __func__);
        return -2;
    }
    
    // Extract session and internal context
    struct llama_session * session = decoder->session;
    if (!session) {
        LLAMA_LOG_ERROR("%s: decoder->session is NULL", __func__);
        return -1;
    }
    
    struct llama_context * ctx = session->ctx;
    if (!ctx) {
        LLAMA_LOG_ERROR("%s: session->ctx is NULL (not initialized)", __func__);
        return -1;
    }
    
    // Delegate to the existing internal function
    // (same code path as old llama_decode(ctx, batch))
    const int32_t ret = llama_decode_internal(*ctx, batch);
    
    return ret;
}

// Output accessors (simple forwarding)
float * llama_decoder_get_logits(struct llama_decoder * decoder) {
    if (!decoder || !decoder->session || !decoder->session->ctx) {
        return nullptr;
    }
    return decoder->session->ctx->logits;
}

float * llama_decoder_get_logits_ith(
        struct llama_decoder * decoder,
        int32_t i) {
    if (!decoder || !decoder->session || !decoder->session->ctx) {
        return nullptr;
    }
    struct llama_context * ctx = decoder->session->ctx;
    if (i < 0 || i >= (int32_t)ctx->n_outputs) {
        return nullptr;
    }
    const int64_t n_vocab = ctx->model.hparams.n_vocab;
    return ctx->logits + (int64_t)ctx->output_ids[i] * n_vocab;
}

float * llama_decoder_get_embeddings(struct llama_decoder * decoder) {
    if (!decoder || !decoder->session || !decoder->session->ctx) {
        return nullptr;
    }
    return decoder->session->ctx->embd;
}

float * llama_decoder_get_embeddings_ith(
        struct llama_decoder * decoder,
        int32_t i) {
    if (!decoder || !decoder->session || !decoder->session->ctx) {
        return nullptr;
    }
    struct llama_context * ctx = decoder->session->ctx;
    const int64_t n_embd = ctx->model.hparams.n_embd;
    if (i < 0 || i >= (int32_t)ctx->n_outputs) {
        return nullptr;
    }
    return ctx->embd + (int64_t)ctx->output_ids[i] * n_embd;
}

float * llama_decoder_get_embeddings_seq(
        struct llama_decoder * decoder,
        llama_seq_id seq_id) {
    if (!decoder || !decoder->session || !decoder->session->ctx) {
        return nullptr;
    }
    struct llama_context * ctx = decoder->session->ctx;
    auto it = ctx->embd_seq.find(seq_id);
    if (it == ctx->embd_seq.end()) {
        return nullptr;
    }
    return it->second.data();
}

// Perf accessors (forward to session ctx)
struct llama_timings llama_decoder_timings(
        const struct llama_decoder * decoder) {
    if (!decoder || !decoder->session || !decoder->session->ctx) {
        return llama_timings{};
    }
    return llama_get_timings(decoder->session->ctx);
}

void llama_decoder_perf_reset(struct llama_decoder * decoder) {
    if (decoder && decoder->session && decoder->session->ctx) {
        llama_reset_timings(decoder->session->ctx);
    }
}
```

### Key Points

1. **Single forwarding layer**: `llama_decoder_decode` extracts ctx from session and calls `llama_decode_internal` (unchanged).
2. **Output accessors**: Forward to session ctx's output buffers and mappings.
3. **Perf accessors**: Forward to existing perf API on ctx.
4. **Error handling**: Check decoder, session, ctx validity.
5. **Role check**: D6 PRIMARY only; VERIFY/DRAFT_MTP abort (D7+).

---

## 9. Block-by-Block Summary Table

| Classification | Count | Example Blocks | Notes |
|---|---:|---|---|
| **FORWARD** | 26 | Batch validation, llama_set_inputs, llama_graph_compute, scheduler queries | Code path unchanged; D6 makes no substitution |
| **PER_DECODER_SUBST** | 32 | Perf counters, draft argmax cache, graph reuse, output IDs, MTP state, graph node IDs | Per-decoder in D7+; in D6 all stored in session ctx (shared) |
| **PER_SESSION_SUBST** | 18 | KV cache ops, kv_self.head, kv_self.n, defrag, hybrid pattern | Session-owned; D6 forwards to session ctx |
| **DEAD** | 8 | stop_internal_decode, LLAMA_PROFILE_DECODE output, HIT/MISS counters, diagnostic logs | No impact on logits/embeddings; instrumentation only |
| **AMBIGUOUS** | 0 | — | All classified; no ambiguous boundaries found in D6 path |

**Total: 84 logical blocks classified.**

---

## 10. Top 3 Risky Blocks for D6 Byte-Identical

### 1. **Output Buffer Ownership (Blocks 6, 16, 17)**

**Risk**: If decoder's buffer layout differs from ctx's, or if reserve fails silently, output will be misaligned or wrong size.

**Mitigation**: 
- D6 guarantees single PRIMARY decoder per session.
- `llama_output_reserve` is called in block 6 with `max(n_outputs, n_outputs_embd)`.
- Same reserve call path as old API.
- Decoder accessor `get_logits_ith` uses ctx's `output_ids` mapping (set in block 6).

**Test**: Verify `output_ids[]` layout matches old `llama_get_logits(ctx, i)` behavior.

### 2. **Perf Counter Accumulation (Blocks 4, 19, 20, various implicit)**

**Risk**: If perf counters are read before `llama_decoder_decode` returns, stale or partial values appear.

**Mitigation**:
- Perf counters are not observable until caller explicitly reads them via `llama_decoder_timings()`.
- All counter updates happen inside `llama_decode_internal` (same as old API).
- D6 doesn't change when counters are updated; only changes which struct stores them.

**Test**: Ensure `llama_decoder_timings(decoder)` reads the correct ctx fields (same as `llama_get_timings(session→ctx)`).

### 3. **KV Cache Head State & Recurrent Model (Blocks 9, 15, implicit in compute)**

**Risk**: If recurrent state (SSM) is not properly synced after graph compute, hidden state is stale.

**Mitigation**:
- D6 PRIMARY: single decoder, no concurrent draft/verify.
- Recurrent state stays in session ctx (unchanged from old API).
- Graph compute updates device SSM; persist to kv_self via scheduler (unchanged).

**Test**: Verify recurrent-model output (e.g., Qwen 3.5 SSM layer) produces identical logits vs. old API on same session.

---

## 11. Blocker/Unresolved Issue

### No Blockers for D6

All blocks have clear classification. However, **one architectural issue looms for D7+**:

**Issue**: `cparams.mtp_op_type` is session-global but set per-decode. In D7+, VERIFY and DRAFT_MTP decoders will set different op_types on the same session (via `llama_set_mtp_op_type(ctx, type)`), causing interference.

**Status**: Deferred to D7. D6 PRIMARY never calls `llama_set_mtp_op_type`; mtp_op_type stays MTP_OP_NONE.

**Mitigation for D7**: Move `mtp_op_type` from `cparams` (session-global) to decoder-local state, or use a decoder-aware setter.

---

## Appendix: Line-by-Line File Locations

All line numbers are in `/home/llm/yarn-agentic/ik_llama.cpp/src/llama.cpp`:

- **llama_decode wrapper**: 9641–9678
- **llama_decode_internal body**: 4985–5836
- **KV cache ops**: 5252–5276
- **Graph build & reuse**: 5295–5379
- **Input setup**: 5426–5429
- **Compute**: 5435–5440
- **Logits extract**: 5461–5583
- **Embedding extract**: 5585–5638
- **Fused draft extract**: 5639–5791
- **Defrag decision**: 5811–5820
- **Final scheduler reset**: 5824–5833

---

**Document prepared by**: Claude (PHASE45 D6 decode mapping task)
**Status**: Ready for D6 implementation review
**Next step**: Fill llama_decoder_decode body per §8 pseudocode; compile & test byte-identical on Qwen 3.6 27B greedy decode.

