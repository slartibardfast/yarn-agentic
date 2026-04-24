# Phase 28: TURBO_KV_4B — Residual Window Implementation

## Status

**In progress.** Autonomous loop self-paces against a ~1h budget per iteration cycle. This phase executes the design locked in PHASE27 (decisions Q1–Q4 resolved) and closes the 18 SKIPs in `test-turbo-kv-residual-window-pbt.cpp`.

## Scope (inherited from PHASE27)

K-side residual window on `TURBO_KV_4B`. CPU + Vulkan FA dispatch. `--cache-residual-window N` CLI flag, default 128 (Q1). Clamp to `n_ctx` with a warning (Q2). Full buffer at init (Q3). Bytes-for-bytes GGUF persistence (Q4).

## Step checklist

Each step mirrors PHASE27's numbered implementation plan plus a Step 0 harness folded in at user request. Loop iterations update this list.

- [x] **Step 0.** Build `test-turbo-kv-residual-window-harness` — minimal C++ integration test that loads a model, constructs a context with varied `residual_window` / `type_k` combinations, emits structured output on stderr/stdout, exits cleanly. Used to retro-verify Steps 1–2 and to forward-verify every subsequent step without relying on noisy llama-cli stderr.
- [x] **Step 1.** Add `residual_window` to cparams + CLI flag + clamp-to-n_ctx warning.
- [x] **Step 2.** Allocate fp16 side-buffer in `llama-kv-cache.cpp` at context init.
- [x] **Step 3.** Rolling-write path on KV-cache append. Quantise-on-eviction.
- [ ] **Step 4.** Eviction correctness — quantised slot matches `quantize_row_turbo_kv_4b_ref`.
- [ ] **Step 5.** Two-pass CPU FA dispatch. Online-softmax merge across passes.
- [ ] **Step 6.** Two-pass Vulkan FA dispatch. Re-run `test-turbo-kv-attention-pbt` at `rw=128`.
- [ ] **Step 7.** GGUF state save/load with fp16 window (Q4 decision).
- [ ] **Step 8.** 9B IQ3_XXS PPL gate: ≤ +0.05 above F16 baseline.
- [ ] **Step 9.** Convert 18 SKIPs in `test-turbo-kv-residual-window-pbt.cpp` to `rc::check`.

## Out-of-plan items landed

Small self-contained work adjacent to residual window, done during idle time in the loop:

- [x] T3.2 — stale `@guidance` prose fix in `turbo_kv_4b_attention.allium` (PHASE26 Tier 3.2).

## Loop log

Each iteration appends a single line noting what landed.

- Iteration 1: loop started; PHASE28.md stub created; T3.2 prose fix in `turbo_kv_4b_attention.allium` (`allium check`: 0 errors).
- Iteration 2: Step 1 landed on llama.cpp master (`b746a733a`) — `residual_window` wired through `llama_cparams`, `llama_context_params`, `common_params`, `--cache-residual-window` flag, clamp-to-n_ctx warning. `--help` smoke confirms flag + default 128. Runtime clamp smoke deferred (llama-cli stderr is noisy during prompt eval; will verify once Step 2 gives a cleaner integration point). Also: cleanup commit (`afa8baf9c`) strips host-planning references from the four turbo_kv PBT skip-stubs per `feedback_no_host_concerns_in_code`.
- Iteration 3: Step 2 landed on llama.cpp master (`1bcd5b179`) — fp16 rolling-tail tensor per layer allocated via `ggml_new_tensor_3d(GGML_TYPE_F16, n_embd_k_gqa, residual_window, n_stream)` when `residual_window > 0`, nullptr otherwise. Constructor signature on `llama_kv_cache` gains `residual_window` before the filter/reuse callbacks; callers in `llama-model.cpp`, `llama-memory-hybrid.cpp`, and `llama-kv-cache-iswa.cpp` updated. iSWA and hybrid paths pass 0 (follow-on work flagged in inline comments — those paths don't overlap with current TURBO_KV target configs). Smoke: `llama-cli --cache-residual-window 0` and `... --cache-residual-window 128 --cache-type-k turbo_kv_4b` both load and begin decoding without crash. Loop stopped at 36/60 min to avoid landing a partial Step 3.
- Iteration 4: Step 0 harness landed on llama.cpp master (`57c4c2254`) — `test-turbo-kv-residual-window-harness` provides a decode-free llama_context init test with grep-friendly output. Running the harness against Qwen3.5 0.8B immediately exposed a Step-2 gap: the 0.8B is a hybrid DeltaNet+attention model that routes through `llama_memory_hybrid`, where the prior Step 2 hardcoded `residual_window=0`. Same commit extends the param wire through all 4 cache classes (`llama_memory_hybrid`, `llama_memory_hybrid_iswa`, `llama_kv_cache_iswa`, plus the already-wired `llama_kv_cache`). Harness confirms allocation now scales linearly on the hybrid path: `rw=0` → 4.48 MiB, `rw=128` → 5.36 MiB (+0.88 MiB for fp16 tail), `rw=9999` → 7.98 MiB clamped with warning. Steps 1 + 2 now fully verified against a real model init path.
- Iteration 5: Step 3 landed on llama.cpp master (`c57cff006`). Graph-build wiring for fp16 overlay writes on K cache append: new `cpy_k_window` method on `llama_kv_cache` (+ context wrapper) emits `ggml_set_rows` from the current K projection into a flat view of `layers[ikv].k_window_fp16`, using I64 slot indices `s*residual_window + (pos % residual_window)`. `llm_graph_input_attn_kv` gains a `self_k_window_idxs` field populated every ubatch via the existing set_input callback. The main K cache is unchanged — the fp16 buffer is a pure overlay. Harness extended with `--append N` to drive real decodes; `rw=128 ctx=512 type_k=turbo_kv_4b --append 200` on Qwen3.5 0.8B Q8_0 decodes all 200 tokens with KV buffer growing from 3.84 MiB reported to 4.59 MiB allocated (0.75 MiB fp16 window delta). No read path yet — that arrives with Step 5's two-pass FA dispatch.
- Iteration 6: mid-step design fix for native-weight awareness. The Step-2 overlay allocation was hardcoded `GGML_TYPE_F16`, which silently risks overflow on BF16-trained models (Qwen3.5 et al.) whose K activations can exceed fp16's ±65504 on wide-range heads. Fix: new `residual_window_type_k` cparam (public `llama_context_params` + internal `llama_cparams`) accepting F16, BF16, or COUNT=auto. Auto-resolution at context init reads `model.layers[i].wk->type`: any BF16 → BF16 overlay; all F16/F32 → F16 overlay; any quantised + no F16 → BF16 overlay (safe default, covers IQ3_XXS/Q8_0/... of BF16-trained sources). CLI: `--cache-residual-window-type {auto|f16|bf16}`. Harness: `--rw-type-k` added. Four-way verification on Qwen3.5 0.8B (`7abbf6215`): BF16 GGUF auto → bf16, Q8_0 GGUF auto → bf16 (quantised-no-f16 branch), BF16 + explicit f16 → f16, BF16 + explicit bf16 → bf16. All 200-decode runs succeed. Memory cost identical (both types are 2 bytes). Step 3 tick retained; the design-fix is scoped to Step 2's allocation choice rather than Step 3's wiring.
