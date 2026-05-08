# PHASE45 Accretions Inventory: ik_llama_context vs upstream llama.cpp

## Summary
ik_llama.cpp's llama_context has accreted ~70+ fields across 8 major subsystems since the fork diverged. These support:
- **MTP (Multi-Token Prediction)**: speculative decoding, fused graph, chain residuals
- **Recurrent models**: qnext (Qwen3Next), SSM state checkpointing, per-step recovery
- **Speculative execution**: device-side caches, async dispatch, verification paths
- **Graph reuse**: scheduler state, persistent buffers, cache tracking
- **Instrumentation**: cycle counters, hook profiling

## Accretions by Subsystem

| Subsystem | Fields Added | Phase Landed | Destination | Notes |
|-----------|--------------|--------------|-------------|-------|
| **MTP Fused Graph** | `mtp_fused_last_compute_count`, `mtp_fused_results_n`, `mtp_fused_results_tokens[8]`, `mtp_fused_results_probs[8]` | Phase 36 | decoder | Per-step extraction from fused cgraph; obsoletes per-step variant results struct in server |
| MTP Fused Graph | `mtp_fused_offset_buf[8*16]`, `mtp_fused_offset_t[8]`, `mtp_fused_offset_n_dev[8]` | Phase 36 | decoder | Per-device argmax offset constants for multi-GPU reduce path |
| MTP Fused Graph | `mtp_fused_chain_residuals[8]`, `mtp_fused_chain_residuals_valid` | Phase 37#2 | decoder | Sched-owned h_pre_norm outputs per fused step; seed for next chain |
| MTP Fused Graph | `mtp_fused_skip_extraction`, `mtp_fused_pending_gf`, `mtp_fused_pending_n_steps` | Phase 38E | spec_loop | Async dispatch: deferred extraction block + cached cgraph for post-verify recovery |
| MTP Fused Graph | `mtp_fused_async_guess` | Phase 38E | spec_loop | Cached guess of chain_residual_step from prior async dispatch; compared post-verify for correctness |
| **MTP Persistent Buffers** | `mtp_persist_ctx`, `mtp_persist_buf`, `mtp_persist[8]`, `mtp_persist_n` | Phase 38B | session | Outlives sched_reset; holds chain residuals D2D across verify boundaries |
| **MTP Chain Residual Seed** | `pending_chain_residual_step` | Phase 37#2 | decoder | Arm flag: next fused decode pulls from `mtp_fused_chain_residuals[N]` instead of host |
| MTP Chain Residual Seed | `draft_input_hidden_state_buf`, `draft_input_hidden_state` | Phase 36 | decoder | Copy-on-set buffer immune to llama_output_reserve repointing (vs old pointer-on-set) |
| **MTP Draft Sampling** | `draft_residual_dev`, `draft_residual_dev_nbytes`, `draft_residual_dev_capacity`, `draft_residual_dev_device`, `draft_residual_dev_valid` | (pre-Phase36) | decoder | Device-resident DRAFT_GEN residual cache; eliminates D2H+H2D bounce for inp_mtp_states refill |
| MTP Draft Sampling | `draft_argmax_valid`, `draft_argmax_n`, `draft_argmax_ids`, `draft_argmax_probs` | (before Phase32) | decoder | On-device CUDA argmax cache for logits_extract; skips per-draft D2H |
| MTP Draft Sampling | `draft_top2_armed`, `draft_argmax_top2_ids` | (before Phase32) | decoder | Optional top-2 variant for tree-K=2 paths (LLAMA_PROBE_TOP2) |
| MTP Draft Sampling | `fast_argmax_for_verify` | (before Phase32) | decoder | Caller-arm flag: verify-step skips full logits D2H when sampler is trivial |
| **MTP Main-Graph Hook** | `t_h_pre_norm` | Phase 36 | decoder | Pre-final-norm residual from main cgraph; tagged by qwen35/qwen35moe builders; reset each decode |
| MTP Main-Graph Hook | `mtp_hook_fire_count`, `mtp_inline_decode_count` | Phase 36 | decoder | Observability counters for per-ubatch KV hook (test/mtp-ubatch-hook/) |
| **MTP Draft Logits** | `mtp_logits_buf`, `mtp_logits_valid`, `mtp_n_vocab`, `mtp_n_drafts`, `t_mtp_logits` | Phase 39B | decoder | MTP draft logits extracted from inline build_mtp_head; shaped [mtp_n_vocab][mtp_n_drafts] |
| **MTP State Input** | `inp_mtp_states` | Phase 36 | decoder | Tensor for MTP draft chain seed plumbing (from prior residual or host buffer) |
| **MTP Diagnostics** | `mtp_cycle_counter` | Phase 36 | decoder | Increments on each verify decode; threaded into fused/per-step stats for cycle-aligned comparisons |
| **QNext (Qwen3Next)** | `inp_s_seq_qnext` | Phase 32 | decoder | I32 [1, n_batch] for linear-attn recurrent state indexing |
| QNext State Management | `qnext_slot_alloc` | Phase 32 | session | Per-seq slot allocator for s_l[il] (linear-attn recurrent state buffer); maps llama_seq_id → slot |
| QNext Instrumentation | `qnext_mixed_seq_fallback_count` | Phase 32 | decoder | Count mixed-seq chunking fallbacks (should stay 0 on continuous-batch traffic) |
| **KV Cache Checkpointing** | `kv_cache.gpu_checkpoint` (nested struct) | Phase 38 | session | GPU-resident SSM state snapshots + per-step allocations; supports speculative verification |
| KV Cache Checkpointing | `kv_cache.save_per_step_ssm` | Phase 37 | session | Flag: enable per-step SSM state saves during delta_net graph build |
| **Graph Reuse State** | `Prev` (opaque struct) | (legacy) | session | Scheduler state for graph reuse detection; held in `prev` unique_ptr |
| Graph Reuse Detection | `can_reuse_graph()`, `reset_scheduler()` | (legacy) | session | Methods on context for graph caching logic |
| **Cache Copy Tracking** | `CacheCopy` (nested struct), `cache_copies`, `update_cache_copies()` | (legacy) | session | Tracks KV cache copies during graph build for multi-step execution |
| **Speculative Exec Plumbing** | `prepare_mtp_graph_inputs()`, `set_mtp_op_type()` | Phase 36 | decoder | APIs for seed plumbing and op-type gating |

## Summary by Destination

| Destination | Field Count | Primary Purpose | Deletion Candidate |
|-------------|-------------|-----------------|-------------------|
| **session** | 11 | Transformer K/V, model-aligned state, shareable across roles | No — load-bearing |
| **decoder** | 52 | Per-execution speculative paths, sampling caches, diagnostics | Some optional (top2, hooks) |
| **spec_loop** | 4 | Async dispatch orchestration, guess validation | Maybe — Phase 38E failed on hardware |
| **kv_txn** | 0 | (no transactional speculative writes yet) | N/A |
| **delete** | 0 | (none identified as dead) | N/A |

## Load-Bearing Accretions (Must Keep for Phase45)

1. **MTP Fused Graph** (`mtp_fused_results_*`, `mtp_fused_offset_*`, `mtp_fused_chain_residuals_*`)
   - **Why**: MTP is the core speculative decoding strategy. Fused path is the production fast path since Phase 36.
   - **Destination**: `decoder` (per-execution-role state; can be reset per decode)

2. **MTP Persistent Buffers** (`mtp_persist_*`)
   - **Why**: Eliminates host bounce for chain-residual seeds across verify boundaries (Phase 38B perf critical).
   - **Destination**: `session` (outlives sched_reset; tied to backend buffer lifetime)

3. **Draft Caches** (`draft_residual_dev`, `draft_argmax_*`)
   - **Why**: Eliminates D2H bounces on every draft event (~1.5 ms per draft per Phase 38B comment).
   - **Destination**: `decoder` (reset per-decode, but backend-owned)

4. **QNext State** (`inp_s_seq_qnext`, `qnext_slot_alloc`)
   - **Why**: Required for Qwen3Next (recurrent linear-attention) correctness; multi-slot allocation.
   - **Destination**: `session` (allocator is model-aligned) + `decoder` (tensor per-ubatch)

5. **KV Cache Checkpointing** (`gpu_checkpoint` struct inside `kv_cache`)
   - **Why**: Speculative verification for recurrent models (Qwen3Next, hybrid Mamba-Transformer).
   - **Destination**: `session` (checkpoint lifetime tied to cache)

## Candidates for Deletion or Deferral

1. **MTP Async Dispatch fields** (`mtp_fused_skip_extraction`, `mtp_fused_pending_gf`, `mtp_fused_async_guess`)
   - **Status**: Phase 38E **FAILED on hardware** (commit d8f2581e message: "FAIL on hardware")
   - **Action**: Delete. The async path is dead; restore sequential post-compute extraction.
   - **Impact**: Simplifies Phase45 decoder struct; no performance loss (Phase 38E didn't ship).

2. **Draft Top2** (`draft_top2_armed`, `draft_argmax_top2_ids`)
   - **Status**: Optional path for tree-K=2 speculative (LLAMA_PROBE_TOP2)
   - **Action**: Keep for now (low cost: 1 bool + 1 vector). No prod deployment, but opt-in.
   - **Impact**: Minimal footprint; deferred to future cleanup.

3. **MTP Hook Counters** (`mtp_hook_fire_count`, `mtp_inline_decode_count`)
   - **Status**: Test-only instrumentation (test/mtp-ubatch-hook/)
   - **Action**: Move to separate diagnostic struct or mark test-only. Not on critical path.
   - **Impact**: Cleans up decoder struct; no correctness impact.

## Migration Path for Phase45

| Accretion | Current Location | New Type | Notes |
|-----------|------------------|----------|-------|
| MTP fused results + offsets | `llama_context` | `decoder` (new) | Stays per-decode; reset on entry |
| Chain residuals (scheduler-owned) | `llama_context` | `decoder` | Move with fused results |
| Persistent chain residuals | `llama_context` | `session` | Keep; outlives sched_reset |
| Draft caches (residual, argmax) | `llama_context` | `decoder` | Can move to backend-owned or decoder |
| QNext allocator | `llama_context` | `session` | Model-aligned; goes with K/V |
| QNext tensor | `llama_context` | `decoder` | Per-ubatch input tensor |
| KV checkpoint struct | `llama_kv_cache.gpu_checkpoint` | `session` (nested) | Keep inside kv_cache |
| Async fields (Phase 38E) | `llama_context` | **DELETE** | Failed code; remove entirely |
| Graph reuse state (`Prev`, `CacheCopy`) | `llama_context` | `session` | Scheduler state; goes with cache |

## Key Decisions for Refactoring

1. **Chain Residuals Split**: Sched-owned (ephemeral, per-execute) vs. persistent (cross-verify). Both are real; keep distinction in code.
2. **Draft Caches Ownership**: Currently on context but semantically backend-owned. Consider a `spec_state` struct inside decoder.
3. **KV Checkpoint**: Nesting inside llama_kv_cache is correct; includes multi-GPU per-step allocations.
4. **Qnext**: Allocator (session) + tensor (decoder) are correctly split already.

## Test Coverage Notes

- Phase 36 Step 3 hook has dedicated test: `tests/mtp-ubatch-hook/`
- Phase 39B MTP logits extraction tested in `common_mtp_read_drafts` (server integration)
- QNext multi-slot: Phase 32 foundation test (continuous-batch correctness)
- Checkpoint: Phase 38 speculative verification tests (Qwen3Next integration)

---

**Total fields inventoried**: ~70 across llama_context + llama_kv_cache
**Load-bearing subsystems**: MTP (fused + persistent), Draft caches, QNext, KV checkpoint
**Safely deletable**: Phase 38E async fields (1 cgraph ptr, 1 bool, 2 int32s — <50 bytes)
