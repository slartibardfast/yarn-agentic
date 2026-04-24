# Phase 28: TURBO_KV_4B — Residual Window Implementation

## Status

**In progress.** Autonomous loop self-paces against a ~1h budget per iteration cycle. This phase executes the design locked in PHASE27 (decisions Q1–Q4 resolved) and closes the 18 SKIPs in `test-turbo-kv-residual-window-pbt.cpp`.

## Scope (inherited from PHASE27)

K-side residual window on `TURBO_KV_4B`. CPU + Vulkan FA dispatch. `--cache-residual-window N` CLI flag, default 128 (Q1). Clamp to `n_ctx` with a warning (Q2). Full buffer at init (Q3). Bytes-for-bytes GGUF persistence (Q4).

## Step checklist

Each step mirrors PHASE27's numbered implementation plan. Loop iterations update this list.

- [x] **Step 1.** Add `residual_window` to cparams + CLI flag + clamp-to-n_ctx warning.
- [ ] **Step 2.** Allocate fp16 side-buffer in `llama-kv-cache.cpp` at context init.
- [ ] **Step 3.** Rolling-write path on KV-cache append. Quantise-on-eviction.
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
