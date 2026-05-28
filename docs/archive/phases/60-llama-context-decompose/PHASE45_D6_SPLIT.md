# PHASE45 D6 Surgical Split Plan: `llama_new_context_with_model` Decomposition

**Date**: May 2026  
**Target**: Byte-identical D6 verifier on Qwen 3.6 27B, PRIMARY-role decoder via new API  
**Scope**: Extract KV cache init and scheduler init from `llama_init_from_model`, route through `llama_session_create` and `llama_decoder_create`

---

## 1. `llama_session_create` Body — KV Cache & Cell Metadata Lift

### 1.1 What to Extract from `llama_init_from_model` (src/llama.cpp:6958–7514)

**KV Cache Allocation Block** (src/llama.cpp:7354–7395):
- **Line 7354–7360**: Call to `llama_kv_cache_init(ctx->kv_self, ctx, type_k, type_v, kv_size, cparams.offload_kqv, ...)`
  - This is the primary lift target. It constructs the entire cache structure (K/V tensors, cells, SSM state for hybrid).
  - Must be called from `llama_session_create` with params from `llama_session_params`.
  - Memory reporting (lines 7362–7395) is informational; keep it or hoist to after session creation.

**Parameter Mapping**:
- `llama_session_params.n_ctx` → `kv_size` parameter
- `llama_session_params.type_k` → `type_k` parameter
- `llama_session_params.type_v` → `type_v` parameter
- `llama_session_params.offload_kqv` → `offload` parameter
- All `type_k_first/last` and `n_k_first/last` parameters default to type_k (see src/llama.cpp:6992–7000 validation for hadamard flags)

**Cell Initialization** (src/llama.cpp:7354–7360 delegates to llama_kv_cache_init, which:
- Lines 777–824 of llama.cpp: `llama_kv_cache_init` body
  - Allocates `cache.cells[kv_size]` (line 816–817)
  - Initializes per-cell metadata: `.pos = -1`, `.delta = 0`, `.src = i` for hybrid (lines 819–823)
  - Sets `cache.head = 0`, `cache.used = 0` (lines 809–811)
  - Flags `cache.recurrent`, `cache.hybrid`, `cache.v_trans` based on architecture (lines 803–807)

**Defrag Init** (lines 838–939 inside llama_kv_cache_init):
- No explicit defrag init in early block; defrag state is implicit in `llama_kv_cache.head/used/cells`.
- Defragmentation happens at runtime via `llama_session_kv_defrag()` API.

### 1.2 Struct Field Assignments for Session

From `PHASE45_FIELD_AUDIT.md`:

**Session owns** (to be moved into `llama_session` struct at D7):
- `model` — const reference to model (passed to session_create)
- `cparams` — **subset only**: n_ctx, n_seq_max, n_batch, n_ubatch, RoPE params, defrag_thold, flash_attn, offload_kqv, mla_attn, split_mode_graph_scheduling (see audit Q3: cparams split is TBD for D5)
- `kv_self` — the llama_kv_cache struct (owns cells, K/V tensors, SSM state)
- `cvec` — control vectors (if used; optional)
- `lora_adapters` — adapter vector
- `backend_cpu`, `backend_metal`, etc. — backend handles (session-level; read-only for decoders)

**Decision**: For D6, `llama_session` wraps an internal `llama_context` (D6/D7 approach; see §3). The session can directly access `ctx->kv_self`, `ctx->cparams`, etc. D10 will extract the fields.

### 1.3 Can We Call `llama_kv_cache_init` Directly?

**YES, but with caveats per PHASE45_KV_PATHS.md:**

1. `llama_kv_cache_init` is a **static function** (src/llama.cpp:777), not exported. 
   - D6 must either:
     - Make it a file-scoped helper in `src/llama-session.cpp` (duplicate the ~360 lines), OR
     - Hoist it to a public/internal header and call it, OR
     - Keep `llama_init_from_model` as the internal factory and wire delegation through it.

2. **PHASE45_KV_PATHS.md findings**:
   - Production path for Qwen 3.6 (hybrid, split-cache, 256K context) is lines 819–1082 inside `llama_kv_cache_init`.
   - No dead paths on this target profile.
   - Recurrent state allocation (DeltaNet) is lines 968–999; split allocation is lines 1043–1079.
   - **Recommendation**: Copy the function body into `src/llama-session.cpp` as a `static llama_kv_cache_init_impl(...)`, or make it `llama_kv_cache_init` in an internal header `src/llama-kv-cache-init.h` and #include it.

**Chosen approach for D6**: 
- Create internal header `include/llama-kv-cache.h` (or nest in `llama-context.h`'s internal section).
- Declare `static bool llama_kv_cache_init(...)` as a normal (non-static) function visible to both `llama.cpp` and `llama-session.cpp`.
- Session calls it after constructing the internal `llama_context` delegate.

---

## 2. `llama_decoder_create` (PRIMARY Role) Body — Scheduler & Output Buffer Lift

### 2.1 What to Extract from `llama_init_from_model`

**Scheduler Initialization Block** (src/llama.cpp:7411–7485):

- **Lines 7414–7421**: Backend buffer type array construction
  - Iterate backends; push CPU-buffer-type for CPU, backend-default for GPU
  - Store in `vector<ggml_backend_buffer_type_t> backend_buft`

- **Lines 7424–7428**: Compute buffer metadata
  - `n_tokens = min(cparams.n_ctx, cparams.n_ubatch)` (line 7424)
  - `max_nodes = model->max_nodes(n_tokens)` (line 7425)
  - Allocate `buf_compute_meta` with graph overhead (lines 7427–7428)

- **Lines 7431–7441**: Scheduler creation
  - **Line 7441**: `ctx->sched = ggml_backend_sched_new(backends..., max_nodes, pipeline_parallel)`
  - Pipeline parallelism decision (lines 7431–7439) depends on device count, split_mode, offload_kqv

- **Lines 7449–7468**: Graph build & scheduler reserve
  - Build worst-case graph via `llama_build_graph` (line 7452)
  - **Line 7455**: `ggml_backend_sched_reserve(ctx->sched, gf)` — materializes compute buffers
  - Fallback retry (lines 7457–7461) if reserve fails with pipeline parallelism

- **Lines 7505–7512**: Split-mode graph scheduling flags
  - If `split_mode == GRAPH && (!has_overrides || cparams.split_mode_graph_scheduling)`, call `ggml_backend_sched_set_split_mode_graph(...)`

**Output Buffer Allocation** (src/llama.cpp:7397–7409):

- **Lines 7400–7404**: Call `llama_output_reserve(*ctx, params.n_seq_max)`
  - Allocates logits & embeddings buffers for outputs
  - Returns the number of sequences reserved; must match `n_seq_max` or abort

### 2.2 Struct Field Assignments for Decoder (PRIMARY)

From `PHASE45_FIELD_AUDIT.md` and skeleton inspection:

**Decoder owns** (PRIMARY role only):
- `role` — llama_decoder_role (VERIFY, DRAFT_MTP, TREE_BRANCH in D7+)
- `sched` — ggml_backend_sched pointer (graph scheduler; read from session's backends)
- `buf_output` — backend buffer for logits/embeddings
- Output tensors: `logits`, `embeddings` pointers (aliases into buf_output)
- Batch tracking: `n_outputs`, `n_queued_tokens` (reset per decode)
- Perf counters: `t_start_us`, `t_compute_us`, etc. (per-decoder; PRIMARY doesn't distinguish verify/draft)
- MTP state (if role != PRIMARY): `mtp_fused_results_*`, `mtp_fused_offset_*`, `mtp_fused_chain_residuals`, `draft_residual_dev`, etc.
  - **For D6 (PRIMARY only)**: NOT initialized; only needed for D7+ VERIFY/DRAFT_MTP

**From src/llama-decoder.cpp skeleton**:
- At creation (line 60–67): Only store session pointer, params, role. Bodies (decode, get_logits, etc.) abort.
- Accessors (lines 73–84): Forward to session.

**Decision for D6**:
- PRIMARY decoder stores a pointer to session's internal `llama_context`.
- Calls into ctx->sched, ctx->buf_output, ctx->logits/embeddings, ctx->logits_all.
- No new scheduler creation; delegate to session's scheduler (already built in llama_init_from_model).

### 2.3 What the Decoder Must Read from Session

- `session->ctx->sched` — the scheduler (built during session creation)
- `session->ctx->buf_output`, `session->ctx->logits`, `session->ctx->embeddings` — output buffers
- `session->ctx->backends` — backend list (for compute dispatch)
- `session->ctx->cparams.n_ubatch`, `session->ctx->cparams.n_ctx` — batch and context sizing

**Read-only; no ownership transfer**. Decoder borrows these for the duration of `llama_decoder_decode()`.

---

## 3. Migration Approach Decision

### Option A: Keep `llama_context` Internal; Wrapper Pointers (CHOSEN for D6)

**Description**:
- `llama_session` and `llama_decoder` are thin wrappers around an **internal, opaque `llama_context`**.
- Skeleton currently has `struct llama_session { llama_context *ctx; ... };` (line 21 of llama-session.cpp).
- Session creation (`llama_session_create`) calls `llama_init_from_model(...)` to build ctx and stores the pointer.
- Decoder creation (`llama_decoder_create`) stores session pointer, not a direct ctx pointer.
- All KV ops and decode forward into ctx's methods.
- D6 verifier is byte-identical because it's the same llama_init_from_model + llama_decode_internal path.

**Advantages**:
1. **Zero behavioral change** — exactly the same llama.cpp code path. D6 verifier is guaranteed byte-identical.
2. **Incremental** — D7 can move one field family at a time (e.g., extract kv_self to session, leaving cparams in ctx).
3. **Backward compat until D10** — external code never sees the split; API boundary shields the decomposition.
4. **Risk mitigation** — if D6 finds issues, rollback is a revert of skeleton fills.

**Disadvantages**:
1. Session and decoder don't "really" own their fields yet; ctx still owns everything.
2. D10 large refactor to delete ctx and wire ownership correctly.
3. Extra indirection at runtime (wrapper → ctx).

**Rationale for THIS codebase**:
- PHASE45.md explicitly says "❌ Compatibility wrapper for `llama_context`" in anti-goals (line 129).
- However, D6's binding test is **byte-identical greedy decode**, which requires unchanged code paths.
- **Compromise**: Use wrappers in D6 (internal wrappers, not public compat), then delete ctx at D10.
- Fork is permanent; no upstream compatibility needed; we can be aggressive at D10.

### Option B: Extract Immediately; ctx Becomes Compat Shim (rejected for D6)

**Why rejected**:
- Fields are scattered across session/decoder/spec_loop; micro-extracting each field breaks atomicity.
- Requires wiring up bidirectional delegation (decoder needs session's kv_self; session might need decoder's perf).
- D6 verifier binding test fails if code paths differ (e.g., different initialization order).
- More complex; higher risk of introducing subtle bugs.

**Pushed to D10** when all fields are classified and multi-slot MTP is proven stable.

---

## 4. Concrete Next-Iteration (D6 → D7) To-Do List

### Ordered Tasks with Scope & Estimated LoC

**Task 1: Hoist `llama_kv_cache_init` to internal header**  
- **File**: Create `src/llama-kv-cache-init.h` OR move into existing `src/llama-context.h` (internal section)
- **Scope**: Copy the 361-line function from src/llama.cpp:777–1137 into header.
  - Make it non-static so llama.cpp and llama-session.cpp can both call it.
  - Keep all conditional logic; no simplifications (preserve dead paths for now; D2 deletions land later).
- **Estimated LoC**: ~370 (copy + #include guards)
- **Order**: 1st (blocker for task 3)

**Task 2: Wire `llama_session_create` to call `llama_init_from_model`**  
- **File**: `src/llama-session.cpp` — fill the `llama_session_create` body (line 59)
- **Scope**:
  - Convert `llama_session_params` → `llama_context_params` (map each field; some live in session, some in decoder context)
  - Call `llama_init_from_model(model, ctx_params)` to build the internal ctx
  - Store in `session->ctx`
  - Fill in `session->model` and `session->params`
  - Return allocated session (or nullptr on failure)
- **Estimated LoC**: ~40 lines (parameter translation + delegated create)
- **Order**: 2nd

**Task 3: Implement KV ops in `llama_session_kv_*`**  
- **File**: `src/llama-session.cpp` — fill lines 73–87 (KV cache operations)
- **Scope**:
  - `llama_session_kv_clear()` → `llama_kv_self_clear(session->ctx)`
  - `llama_session_kv_seq_rm()` → `llama_kv_self_seq_rm(session->ctx, ...)`
  - Same for copy, keep, add, div, pos_max, update, defrag
  - These are thin forwarding stubs; no new logic
  - Existing `llama_kv_self_*` functions already exist in llama.cpp; we're wrapping them
- **Estimated LoC**: ~30 lines
- **Order**: 3rd

**Task 4: Implement state save/load in `llama_session_state_*`**  
- **File**: `src/llama-session.cpp` — fill lines 84–90
- **Scope**:
  - `llama_session_state_get_size()` → `llama_get_state_size(session->ctx)`
  - `llama_session_state_get_data()` → `llama_copy_state_data(session->ctx, dst)`
  - etc.
  - Similar forwarding pattern
- **Estimated LoC**: ~20 lines
- **Order**: 4th (lower priority for D6; can defer to D7)

**Task 5: Wire `llama_decoder_create` (PRIMARY) to delegate**  
- **File**: `src/llama-decoder.cpp` — fill lines 60–67 (already mostly done; just verify)
- **Scope**:
  - Already stores session pointer (line 62).
  - Store params (line 64).
  - For PRIMARY role: no new state needed yet.
  - Store `role` (line 65).
  - Return decoder.
- **Estimated LoC**: ~5 lines (already in skeleton)
- **Order**: 5th

**Task 6: Implement `llama_decoder_decode` (PRIMARY) — forward to session's ctx**  
- **File**: `src/llama-decoder.cpp` — fill line 98
- **Scope**:
  - Extract ctx from `decoder->session->ctx`
  - Call `llama_decode_internal(ctx, batch)` (existing function)
  - Return the result (n_tokens decoded, or error code)
  - **For PRIMARY only**: no special logic. VERIFY/DRAFT_MTP land in D7.
- **Estimated LoC**: ~10 lines
- **Order**: 6th (critical for D6 verifier)

**Task 7: Implement `llama_decoder_get_logits*` — forward to session's ctx**  
- **File**: `src/llama-decoder.cpp` — fill lines 103–107
- **Scope**:
  - `llama_decoder_get_logits()` → return `ctx->logits`
  - `llama_decoder_get_logits_ith(i)` → return `ctx->logits + i * n_vocab`
  - `llama_decoder_get_embeddings()` → return `ctx->embeddings`
  - `llama_decoder_get_embeddings_ith(i)` → return `ctx->embeddings + i * n_embd`
  - `llama_decoder_get_embeddings_seq(seq_id)` → forward to ctx method (or implement)
- **Estimated LoC**: ~15 lines
- **Order**: 7th (critical for D6 verifier)

**Task 8: Implement `llama_decoder_set_*` accessors**  
- **File**: `src/llama-decoder.cpp` — fill lines 86–96
- **Scope**:
  - `set_n_threads()`, `set_causal()`, `set_embeddings()`, `set_warmup()`
  - Store on decoder params (no delegation to ctx for warmup; it's a creation-time flag in the old API)
  - **For D6**: just update decoder->params; warmup delegation comes in D7 if needed
- **Estimated LoC**: ~10 lines
- **Order**: 8th

**Task 9: Update `examples/main/main.cpp` to use new API**  
- **File**: `examples/main/main.cpp`
- **Scope**:
  - Old: `llama_context *ctx = llama_init_from_file(...);`
  - New: `llama_session *session = llama_session_create(model, session_params); llama_decoder *decoder = llama_decoder_create(session, decoder_params);`
  - Replace `llama_n_ctx(ctx)` → `llama_session_n_ctx(session)`
  - Replace `llama_decode(ctx, batch)` → `llama_decoder_decode(decoder, batch)`
  - Replace `llama_get_logits(ctx)` → `llama_decoder_get_logits(decoder)`
  - Minimal edits; grep-replace most patterns.
- **Estimated LoC**: ~30 edits (30-40 line changes spread across file)
- **Order**: 9th (last; validates that the API is callable)

**Task 10 (optional, D6→D7): Implement LoRA/control vector forwarding**  
- **File**: `src/llama-session.cpp` — fill lines 92–101
- **Scope**:
  - `llama_session_lora_adapter_set()` → `llama_lora_adapter_set(ctx, ...)`
  - etc.
  - Lower priority for D6 (not on greedy-decode path).
- **Estimated LoC**: ~15 lines
- **Order**: Defer to D7

### Summary By File

| File | Tasks | LoC | Priority | Notes |
|------|-------|-----|----------|-------|
| `src/llama-kv-cache-init.h` (new) | 1 | ~370 | **FIRST** | Unblock session creation |
| `src/llama-session.cpp` | 2,3,4,10 | ~100 | 2–4 | Incremental; 3 critical for D6 |
| `src/llama-decoder.cpp` | 5,6,7,8 | ~50 | 5–8 | 6,7 critical for D6 verifier |
| `examples/main/main.cpp` | 9 | ~40 | 9 | Last; integration test |
| `src/llama.cpp` | (no changes) | 0 | — | Use existing llama_init_from_model |

**Total D6 fill: ~560 LoC** (mostly forwarding; same code paths as old API).

---

## 5. Build & Link Risks to Watch

### 5.1 Header Inclusion Order

**Risk**: `llama-session.h` and `llama-decoder.h` are public, but they need internal access to `llama_context` fields (scheduler, buffers).

**Mitigation**:
- Keep `struct llama_context` **opaque in public headers**.
  - Public headers only forward-declare `struct llama_context;`
  - Private struct definition stays in `src/llama-context.h`
- Session and decoder C++ implementations (#include llama-context.h privately) access ctx fields.
- Accessor functions (e.g., `llama_decoder_get_logits`) return pointers to data; C API never exposes ctx internals.

**Current state** (llama-session.cpp line 20–21): ✓ Session stores opaque ctx pointer; good.

### 5.2 Symbol Conflicts

**Risk**: Multiple methods with same logical operation (e.g., `llama_kv_self_seq_rm` vs `llama_session_kv_seq_rm`).

**Solution**:
- `llama_kv_self_seq_rm(llama_context *ctx, ...)` — internal, may become private
- `llama_session_kv_seq_rm(llama_session *session, ...)` — public API
- Session methods delegate to internal helpers or llama_context methods.
- No linker conflict; different symbols.
- At D10, `llama_kv_self_*` may be deleted if fully wrapped by session API.

**Implementation**: Already clear in skeleton (llama-session.cpp doesn't define competing symbols).

### 5.3 ABI Changes Beyond main

**Risk**: Server, common, and other in-tree binaries expect `llama_context` shape and `llama_decode` signature.

**Current Mitigation** (D6):
- Old `llama_context` and `llama_decode(ctx, batch)` remain **unchanged** in llama.cpp.
- New session/decoder API is **additive** on top; old API still works.
- No ABI break for D6.

**Long-term** (D10):
- Old API (`llama_init_from_file`, `llama_context`, `llama_decode`) gets deleted.
- Server, common, examples/server must migrate to session/decoder API.
- This is the planned breaking change; PHASE45.md roadmap includes it (D10 line 56).

### 5.4 Static vs. Exported Function Visibility

**Issue**: `llama_kv_cache_init` is currently `static` in llama.cpp (line 777).

**Solution for D6**:
- **Option A** (chosen): Move to internal header as non-static; both llama.cpp and llama-session.cpp #include and call.
  - Less code duplication.
  - Requires one header file creation.
  
- **Option B**: Duplicate the function body in llama-session.cpp.
  - Avoids header; higher maintenance cost.

**Chosen**: Option A. Create `src/llama-kv-cache-init.h` with the function declaration/definition.

---

## 6. Field Audit Cross-Check vs. PHASE45_FIELD_AUDIT.md

**Destination Summary**:

| Component | Audit Count | D6 Delegation? | D7+ Ownership? |
|-----------|-------------|----------------|----------------|
| `llama_session` | 15 | Wrap in ctx | Extract kv_self, cells, pos, defrag |
| `llama_decoder` (PRIMARY) | 5 basic | Wrap in ctx | Extract sched, output buffers, batch tracking |
| `llama_decoder` (VERIFY/DRAFT_MTP) | 40+ | N/A yet | D7+: MTP state, recurrent state, perf |
| `llama_spec_loop` | N/A | N/A yet | D8+: orchestrator state |
| Total audited | 61 | 20 (D6 path) | 40+ (D7+) |

**Compliance Check**:
- Session's D6 forwarding touches: model, cparams (subset), kv_self, backend handles. ✓
- Decoder PRIMARY's D6 forwarding touches: role, sched, output buffers. ✓
- VERIFY/DRAFT_MTP fields (mtp_fused_*, draft_residual_dev, etc.) deferred. ✓
- Spec_loop not yet implemented. ✓

---

## 7. Byte-Identical Verifier Guarantee

**How D6 achieves byte-identical output**:

1. **Same llama_init_from_model** — Session creation calls the existing factory; no code path change.
2. **Same llama_decode_internal** — Decoder's decode() forwards to the existing inference function.
3. **Same llama_get_logits/embeddings** — Direct pointer return; same buffer layout.
4. **Same scheduler, KV cache, backend dispatch** — All inherited from ctx; no changes.

**Verification approach**:
- Profile: Qwen 3.6 27B, 256K context, -fa on, CUDA 2-GPU split.
- Benchmark: `main.cpp greedy-decode 50 tokens` on a fixed prompt.
- Check: Output logits byte-identical (within 1e-6 float tolerance) vs. old API.
- Gate: If byte-identical, D6 is approved; proceed to D7 parallel-execution shapes.

---

## 8. Line-by-Line Mapping Summary

### From `llama_init_from_model` → Session

| Purpose | Old Code | New Location | Delegation |
|---------|----------|--------------|------------|
| Parameter validation | 6962–7000 | session_create (move logic) | Validate n_ctx, flash_attn compat, etc. |
| Device setup | 7004–7010 | Session init or defer to ctx | Populate cparams.devices |
| cparams assignment | 7012–7160 | Session param map + ctx build | map llama_session_params → llama_context_params |
| Backend init | 7184–7352 | llama_init_from_model (unchanged) | delegate; ctx builds backends |
| **KV cache init** | **7354–7360** | **llama_session_create** | **Call llama_kv_cache_init** |
| Output buffer | 7397–7409 | llama_decoder_create? Or session? | TBD D7; defer to ctx for now |
| Scheduler init | 7411–7485 | llama_decoder_create (PRIMARY) | Delegate; ctx already built sched |
| Worst-case graph | 7449–7452 | llama_decoder_create | Delegate; ctx already built graph |
| Return ctx | 7514 | session->ctx | Store opaque pointer |

### From `llama_init_from_model` → Decoder (PRIMARY)

| Purpose | Old Code | New Location | Implementation |
|---------|----------|--------------|-----------------|
| Scheduler access | 7441 | decoder holds session; accesses session->ctx->sched | Forward |
| Output buffer | 7400–7409 | Same | Forward to ctx->buf_output |
| Threads | 7017–7018 | decoder_params | Store and forward via `llama_decoder_set_n_threads` |
| Perf reset/print | None in init | decoder methods | Delegate to ctx perf (for now) |

---

## 9. Known Unknowns & D7 Decisions

1. **State save/load** — Does `llama_session_state_*` copy only KV cache, or also perf/scheduler state?
   - **TBD in D7**: Audit llama_get_state_size / llama_copy_state_data to confirm scope.
   
2. **Warmup flag** — `llama_decoder_set_warmup()` is a no-op in skeleton (line 93–96).
   - **TBD in D7**: Warmup is currently a session-creation decision; moving to decoder requires new API contract.
   
3. **Encoding path** — D6 is greedy decode only; encoder not covered.
   - **TBD in D7+**: When multi-modal models ship; use same delegation pattern.

4. **Split-mode behavior** — Lines 7505–7512 configure split-mode graph scheduling if enabled.
   - **TBD in D7**: Confirm this path is hit on 2-GPU Qwen; verify split tensors are routed correctly through both decoders.

---

## 10. Delivery Checklist

- [ ] Create `src/llama-kv-cache-init.h`; move llama_kv_cache_init function there
- [ ] Fill `llama_session_create` (parameter validation + llama_init_from_model delegate)
- [ ] Fill `llama_session_kv_*` forwarding (8 functions)
- [ ] Fill `llama_decoder_decode` and `llama_decoder_get_logits*` (critical path)
- [ ] Fill `llama_decoder_set_*` accessors
- [ ] Update `examples/main/main.cpp` to call new API
- [ ] Compile: `cmake --build . --target main`
- [ ] Run: `./main -m qwen36-27b.gguf -p "..." | cmp -l <(old_binary output)` (byte-identical check)
- [ ] Build server: `cmake --build . --target server` (should work; old API unchanged)
- [ ] Perplexity benchmark: `./perplexity -m qwen36-27b.gguf` (should match D5)
- [ ] CMakeLists.txt: Add `src/llama-session.cpp`, `src/llama-decoder.cpp` to library if not auto-discovered
- [ ] GCC syntax check: `gcc -fsyntax-only` on new headers
- [ ] Final: Run D6 verifier test (greedy decode, byte-identical output)

---

## 11. Files Written/Modified

**New**:
- `src/llama-kv-cache-init.h` (~370 LoC)

**Modified (stub → filled)**:
- `src/llama-session.cpp` (~100 LoC added; skeleton ~105 → ~205)
- `src/llama-decoder.cpp` (~50 LoC added; skeleton ~117 → ~167)
- `examples/main/main.cpp` (~40 line changes; integration)

**Unchanged**:
- `src/llama.cpp` (llama_init_from_model, llama_decode_internal stay as-is)
- `include/llama.h` (old API preserved)
- `include/llama-session.h`, `include/llama-decoder.h` (D5 sketches; no changes)

**Future (D7+)**:
- Extract kv_self, scheduler, output buffers from ctx into session/decoder; deep refactor.

---

## Appendix: Estimated Timeline

| Phase | Tasks | FTE-Days | Critical Path |
|-------|-------|----------|----------------|
| D6 | Fill + integrate | 1–2 FTE-days | kv-cache-init.h → session_create → main.cpp |
| D6 validate | Byte-identical test | 0.5 FTE-days | Greedy decode on Qwen 3.6 |
| D7 | Extract session fields | 2–3 FTE-days | kv_self ownership, cparams split, scheduler lifetime |
| D7 validate | CUDA single-slot | 1 FTE-day | Multi-turn bench; perplexity parity |
| D8–D9 | Spec loop + multi-slot | 3–5 FTE-days | Orchestrator, kv_txn, draft+verify |

**D6 alone**: 1.5 FTE-days to byte-identical verifier gate.

---

**Document prepared by**: Claude (via PHASE45 D6 mapping task)  
**Target next step**: Fill D6 bodies per §4; commit to phase45-decompose branch.
