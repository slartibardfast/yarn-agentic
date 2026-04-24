# PHASE28 Step 5 — Handoff Brief

Handoff for the fresh session that will wire the residual-window attention
read path into `build_attn`. The foundation (FA LSE primitive, peek API,
overlay write path, spec, PBTs) is all landed. This is the integration
pass that makes the overlay actually influence model output.

## Read this first

1. `PHASE28.md` — the loop log. Iterations 1-11 are the history.
2. `turbo_kv_residual_window.allium` — the spec. Overlay-model rules,
   with `ReadKVRecentFromOverlay` and `ReadKVTailFromMainCache` marked
   aspirational. `@guidance` on `ReadKVRecentFromOverlay` captures the
   approach and merge formula — read it before coding.
3. `llama.cpp/tests/test-turbo-kv-residual-window-pbt.cpp` — 23 obligations,
   17 PASS today, 6 SKIP. The six SKIPs are the test surface this work
   flips to PASS.
4. `llama.cpp/tests/test-flash-attn-lse.cpp` — proves FA_LSE returns
   `(VKQ, M, S)` that agrees with FA after manual `VKQ/S` scaling.
   Useful reference for how the merge math should look end-to-end.

## What is already in place

### Public API and cparam plumbing (iterations 1–2, 6, 8)

- `--cache-residual-window N` (default 128). Clamped to `n_ctx` at
  context init with a warning.
- `--cache-residual-window-type {auto|f16|bf16}` (default auto).
- `--cache-type-k auto` / `--cache-type-v auto` accepted; all three
  KV float types auto-resolve to the model's native float from
  `model.layers[i].wk->type` (BF16 if any wk is BF16, F16 if all
  wk are F16/F32, BF16 as a safe upgrade when all wk are quantised).
- Resolution happens in `llama_init_from_model` at
  `llama.cpp/src/llama-context.cpp:3055-3089`.

### Per-layer overlay allocation (iteration 3 / Step 2)

- Each layer with a K cache slot gets `k_window_fp16` sized
  `[n_embd_k_gqa, residual_window, n_stream]` in the resolved dtype.
- Not allocated when `residual_window == 0` or the layer has no K slot
  (recurrent layers in hybrid caches).
- Plumbed through `llama_kv_cache`, `llama_kv_cache_iswa`,
  `llama_memory_hybrid`, `llama_memory_hybrid_iswa` constructors.
- SWA side of iSWA caches always gets `residual_window = 0`.

### Graph-build overlay write (iteration 5 / Step 3)

- New method `llama_kv_cache::cpy_k_window` emits `ggml_set_rows` from
  the current K projection into a flat view of `k_window_fp16`, using
  I64 slot indices `s*residual_window + (pos % residual_window)`.
- `llm_graph_input_attn_kv::self_k_window_idxs` — new input tensor
  populated per ubatch via the existing `set_input` callback.
- Dispatched from `build_attn` at `llama.cpp/src/llama-graph.cpp:2208`
  after the standard `cpy_k`. No-op when `residual_window == 0` or the
  layer has no K slot.
- **Hybrid dispatch fix:** `llm_graph_input_mem_hybrid::set_input`
  calls individual `set_input_*` methods directly rather than
  delegating to `llm_graph_input_attn_kv::set_input`. Fixed at
  `llama-graph.cpp:597-599` to also populate `k_window_idxs`. Critical
  — without this, every hybrid-model decode (all Qwen3.5 sizes) wrote
  to slot 0 regardless of position.

### Peek API (iteration 7 / Step 4)

- `llama_memory_residual_window_peek(mem, il, stream, slot, dst, dst_size)`
- `llama_memory_residual_window_slot_nbytes(mem, il)`
- Public C API at `llama.cpp/include/llama.h:815-826`.
- Virtual on `llama_memory_i` with default-0 returns; concrete on
  `llama_kv_cache`; forwarders on iSWA and both hybrid memory classes.
- Harness `--check-window` flag verifies writes land at expected slots.

### FA LSE primitive (iteration 11 / Step 5.1)

- **`ggml_flash_attn_ext_lse`** in `llama.cpp/ggml/include/ggml.h`:
  same signature as `ggml_flash_attn_ext` but returns a tensor of
  shape `[n_embd_v + 2, n_head, n_batch, ne3]` where the extra two
  rows per `(head, query)` hold `(M, S)` — the unscaled
  online-softmax max and sum.
- Shares the `GGML_OP_FLASH_ATTN_EXT` op type; flag in `op_params`
  slot 4.
- `ggml_flash_attn_ext_is_lse(tensor)` query.
- CPU kernel modified at `llama.cpp/ggml/src/ggml-cpu/ops.cpp` to
  detect the flag and write raw `[VKQ, M, S]` instead of the
  normalised output. Dispatcher forces single-chunk non-tiled path
  on thread 0 when the flag is set (other threads idle at a barrier).
- Non-CPU backends not yet plumbed. Using this op with backends
  offloaded will fail the output-shape assert; needs a separate port
  alongside the Vulkan read-path work.

### Spec + tests

- `turbo_kv_residual_window.allium` rewritten to overlay model
  (iteration 9, commit `f13f500`). `allium check` passes.
- 23 PBT obligations propagated (iteration 10, commit `3e9f9b5bf`).
  17 PASS / 6 SKIP. The 6 SKIPs are the read-path rules this work
  flips.
- `test-flash-attn-lse` verifies FA_LSE numerics against FA
  (iteration 11). Pass.

## Step 5 remaining work

### Goal

Make attention actually READ from the overlay for the last
`residual_window` positions. Currently `build_attn` reads only the
main K cache (quantised); the overlay is written but ignored.

### Approach (recorded as `@guidance` in spec)

Two-pass flash-attention with online-softmax merge.

- **Pass A**: `ggml_flash_attn_ext_lse` over positions `[0, seq_len - rw)`.
  K = main quantised cache (turbo_kv_4b or whatever's configured),
  V = V cache. Mask = causal mask AND `-inf` for positions
  `>= seq_len - rw`. Returns `(VKQ_a, M_a, S_a)`.
- **Pass B**: `ggml_flash_attn_ext_lse` over positions `[seq_len - rw, seq_len)`.
  K = overlay tensor reshaped to `[n_embd_k_gqa, rw, n_stream]` (or the
  active-slot subview), V = V cache sliced to the same positions.
  Mask = causal mask restricted to that range. Returns
  `(VKQ_b, M_b, S_b)`.
- **Merge** using existing ggml primitives:
  ```
  M_new  = max(M_a, M_b)                        -- elementwise max
  sa     = exp(M_a - M_new)                     -- rescale factor A
  sb     = exp(M_b - M_new)                     -- rescale factor B
  S_new  = sa*S_a + sb*S_b                      -- merged sum
  VKQ    = sa*VKQ_a + sb*VKQ_b                  -- merged numerator
                                                   (broadcast sa,sb over DV)
  output = VKQ / S_new                          -- final scaled output
  ```
- **Edge cases**:
  - `seq_len <= rw`: Pass A is empty. Skip pass A, use pass B only
    (equivalent to running full FA over the overlay).
  - `residual_window == 0`: Skip both passes; fall through to the
    existing single-FA path.

### Implementation tasks

1. **Elementwise max op.** Ggml has no elementwise max. Implement via
   `max(a, b) = 0.5 * (a + b + |a - b|)` using `ggml_add`, `ggml_sub`,
   `ggml_abs`, `ggml_scale`. Or add a new op. Either fine; pick the
   cheaper path for the commit. This is for merging `M_a` and `M_b`
   per (head, query) — small tensor (1, n_head, n_batch).

2. **Pass A mask.** The current FA mask `inp->self_kq_mask` covers all
   cached positions with a causal restriction. For pass A, additionally
   zero out the last `rw` positions. Construct a modified mask — view
   of the existing mask with the tail region set to `-inf`. Look at
   how `set_input_kq_mask` populates the existing mask and add an
   analogous "pass-A mask" or derive it via ggml ops at graph-build
   time.

3. **Pass B K tensor.** The overlay is
   `[n_embd_k_gqa, rw, n_stream]`. FA expects K as
   `[DK, K_len, H, ne3]` (post-permute). Need to reshape the overlay
   into this layout. Since the overlay's rw dim corresponds to "K
   positions" for pass B, and the overlay is shared across heads
   within a stream (it stores K projections which FA broadcasts over
   query heads), the reshape should give `[DK, rw, n_head_kv, n_stream]`.
   Verify against how the main K cache is shaped in `get_k`.

4. **Pass B mask.** Causal restriction across the `rw` positions in
   the overlay, for each query position. For single-token decode
   (query at `pos`), all rw overlay positions are causally valid (they
   represent positions `[pos-rw+1, pos]`, all `<= pos`). For multi-token
   ubatch (PP), each query's visible set differs. Construct
   `[rw, n_batch]` mask tensor. Look at `build_attn_inp_kq_mask` and
   mirror the pattern.

5. **Order-of-overlay-slots.** The overlay is written at
   `(pos % rw)` — a ring buffer. For pass B to attend correctly, the
   mask's column ordering must match the logical position ordering
   each query expects. For a query at position `pos`, logical position
   `p` is in overlay slot `p % rw`. The slot-to-position map rotates
   as `pos` advances. Two options:
   - Always use overlay slots 0..rw-1 in physical order, construct
     the mask with `-inf` for slots whose stored position is outside
     the visible range per query. Simpler but requires a per-query
     slot-to-position lookup.
   - Rewrite the overlay into linear order before pass B via
     `ggml_get_rows` with an index tensor. More compute, simpler mask.
   Recommend option 1 for decode (rw is small, mask is cheap). For
   PP, option 2 might be clearer.

6. **FA kernel path selection.** `build_attn_mha` at
   `llama-graph.cpp:1883` currently calls `ggml_flash_attn_ext`. For
   the overlay-active case, replace with two `ggml_flash_attn_ext_lse`
   calls + merge, conditionally. When `residual_window == 0` OR the
   layer has no overlay, use the existing single-FA path. Consider
   wrapping the two-pass logic in a helper function in
   `llm_graph_context` that takes the same args as `build_attn_mha`.

7. **Flip the 6 PBT SKIPs.** After the implementation lands,
   `tests/test-turbo-kv-residual-window-pbt.cpp` SKIPs for
   `ReadKVRecentFromOverlay` (4) and `ReadKVTailFromMainCache` (2)
   should become real assertions. They can be structural (the
   attention path produces a result when the rule's requires are met)
   rather than numerical — the harness and PPL check give the
   numerics.

8. **PPL sanity check.** Add a 0.8B PPL comparison:
   - `turbo_kv_4b, rw=0` — baseline (loss everywhere)
   - `turbo_kv_4b, rw=128` — with the new read path
   Expected: rw=128 PPL should be LOWER than rw=0 (recent positions
   now lossless). If they're within noise, the read path might not
   be firing; if rw=128 is worse, check mask construction.

9. **Harness verification.** `test-turbo-kv-residual-window-harness
   --append N --check-window` already verifies writes. Consider adding
   a read-path verification mode that compares two-pass output against
   single-pass fp16 FA output when K is f16 throughout (should be
   identical — the overlay is lossless).

## Files you will probably edit

- `llama.cpp/src/llama-graph.cpp` — `build_attn`, `build_attn_mha`.
  Two-pass wiring + merge.
- `llama.cpp/src/llama-graph.h` — any new fields on
  `llm_graph_input_attn_kv` for per-pass masks.
- `llama.cpp/src/llama-kv-cache.{h,cpp}` — new helper methods if
  extracting overlay as a K-shaped tensor needs plumbing. New
  `build_input_*` / `set_input_*` for per-pass masks.
- `llama.cpp/src/llama-kv-cache-iswa.{h,cpp}`,
  `llama.cpp/src/llama-memory-hybrid{,-iswa}.{h,cpp}` — forwarders
  for any new helpers.
- `llama.cpp/tests/test-turbo-kv-residual-window-pbt.cpp` — flip 6 SKIPs.
- `llama.cpp/tests/test-turbo-kv-residual-window-harness.cpp` — maybe
  add a read-path verification mode.
- `PHASE28.md` — iteration log entries.

## Implementation bridge (spec ↔ code)

| Spec construct | Code | File:line |
|---|---|---|
| `KVector` entity | `k_cur` tensor in `cpy_k`/`cpy_k_window` | `llama.cpp/src/llama-graph.cpp:2144` |
| `Context` entity | `llama_context_params` / `llama_cparams` | `llama.cpp/include/llama.h:307`, `llama.cpp/src/llama-cparams.h:9` |
| `OverlayDtype` enum | `ggml_type` restricted to F16/BF16 | `llama.cpp/src/llama-context.cpp:182-188` |
| `config.residual_window` | `llama_context_params::residual_window`, `cparams.residual_window` | `llama.cpp/include/llama.h:365`, `llama.cpp/src/llama-cparams.h:23` |
| `config.overlay_dtype` | `cparams.residual_window_type_k` (auto = COUNT, resolved to F16/BF16) | `llama.cpp/src/llama-cparams.h:33` |
| rule `ComputeOverlaySlot` | `set_input_k_window_idxs` | `llama.cpp/src/llama-kv-cache.cpp:1656-1685` |
| rule `AppendOverlayRow` | `cpy_k_window` + graph dispatch | `llama.cpp/src/llama-kv-cache.cpp:1431-1466`, `llama.cpp/src/llama-graph.cpp:2208-2213` |
| rule `ResolveOverlayDtypeAuto` | `resolve_native_float` / `resolve_cache_type` lambdas | `llama.cpp/src/llama-context.cpp:3055-3089` |
| **rule `ReadKVRecentFromOverlay`** | **NOT IMPLEMENTED — target of this work** | (new code in build_attn) |
| **rule `ReadKVTailFromMainCache`** | **NOT IMPLEMENTED — same** | (new code in build_attn) |
| invariant `ResidualWindowWithinContext` | clamp in ctx init | `llama.cpp/src/llama-context.cpp:172-177` |
| invariant `OverlayDtypeIsFloat` | validation fallthrough | `llama.cpp/src/llama-context.cpp:182-188` |
| surface `OverlayPeek` | `llama_memory_residual_window_peek`/`_slot_nbytes` | `llama.cpp/include/llama.h:815-826`, `llama.cpp/src/llama-context.cpp:3453-3473` |

## Key design decisions already made

- **Overlay, not eviction.** Every position is quantised into the main
  cache; overlay is a parallel fp16/bf16 ring buffer. Recent positions
  exist in both. See `turbo_kv_residual_window.allium` header.
- **BF16-native awareness.** All three KV float types auto-resolve
  from `model.layers[i].wk->type`. On Qwen3.5 (BF16), defaults are
  BF16 throughout. Explicit user choices (including quantised types
  like `turbo_kv_4b`) are respected verbatim.
- **Overlay stores whatever `cpy_k` gets.** Post-RoPE on standard K
  caches, pre-RoPE on split-K / turbo_kv_4b caches. Flagged as an
  open question in the spec; for Step 5 read path, the mirroring
  behaviour means you can feed the overlay through the same RoPE
  application path as `get_k` does for the main cache.
- **Two-pass FA + online-softmax merge.** Approach selected because
  the single-pass assembled-K alternative requires full-cache
  dequant per decode (5+ GFLOP and 500+ MB scratch at 35B-A3B /
  32K ctx), defeating turbo_kv_4b's compression. `@guidance` on
  `ReadKVRecentFromOverlay` has the details.
- **Single-chunk FA_LSE.** Current `ggml_flash_attn_ext_lse` forces
  single-chunk non-tiled CPU execution. Parallelism is an
  optimisation once correctness lands.

## Gotchas

- **Clangd compile-commands staleness.** Every time you add a new
  `.cpp` file, clangd's diagnostics will show "file not found" /
  undeclared identifiers for 10+ minutes until it rebuilds its index.
  The actual `cmake --build` runs fine. Ignore these.
- **No phase/step/tier references in code.** Feedback memory
  `feedback_no_host_concerns_in_code`: "no Phase N in comments,
  commits, or branch names — use technical descriptive names."
  Applies to submodule code. PHASE28.md at top level is fine; that's
  the plan-file location per `feedback_no_time_references` and §5 of
  `yarn-agentic/CLAUDE.md`.
- **Co-Authored-By footer.** Every commit must end with
  `Co-Authored-By: Qwen 3.6 35B-A3B via Claude Code <noreply@anthropic.com>`
  — overriding Claude Code's default footer per the dedup section of
  `yarn-agentic/CLAUDE.md`.
- **Submodule workflow.** Commits in `llama.cpp/` submodule, then a
  separate top-level "bump" commit on `main`, then a separate
  PHASE28.md update commit. Three commits per meaningful iteration.
- **Never reuse PHASE28.md with batched code changes.** Per §5,
  plan-file edits are their own commits.
- **PBT `feedback_dont_skip_tests`.** Write every planned test step
  even when a diagnostic already points at the root cause; reusable
  test infra compounds.
- **PBT `feedback_test_first_no_defer`.** On feature ports, implement
  everything all-in with test-first approach.
- **GPU coordination.** See `reference_gpu_sharing_protocol` in
  memory — multi-agent GPU sharing uses flock + COORD.md. Single
  session is fine; if another agent is running, follow the protocol.

## Recommended first actions in fresh session

1. `cat PHASE28.md PHASE28_STEP5_HANDOFF.md` to load context.
2. Read the spec's `@guidance` on `ReadKVRecentFromOverlay` for the
   merge formula.
3. Scan `llama.cpp/src/llama-graph.cpp` lines 1883-2100 (build_attn_mha)
   and 2139-2285 (build_attn with kv).
4. Start with the elementwise-max op decision — fewest dependencies.
5. Build the merge graph as a standalone helper first, verify it with
   a small unit test (similar to `test-flash-attn-lse.cpp`) before
   wiring into `build_attn`.
6. Wire the two-pass logic once the merge helper is trusted.

## Current state (as of handoff)

- **Branch:** `main`
- **Main HEAD:** `8f4c264` (PHASE28 iteration 11 log)
- **llama.cpp HEAD:** `8cafefe2b` (ggml_flash_attn_ext_lse foundation)
- **Working tree:** clean except for pre-existing unrelated state on
  PHASE20.md, PHASE21.md, ik_llama.cpp (not from this session —
  carried over from before).
- **All residual-window work is committed and pushed** to both
  `slartibardfast/yarn-agentic` main and `slartibardfast/llama.cpp`
  master.

## Spec file locations (top-level repo)

- `turbo-kv-4b.allium` — core 4-bit algorithm
- `turbo-kv-3b.allium` — 3-bit variant
- `turbo_kv_residual_window.allium` — **this work's spec**
- `turbo_kv_asymmetric.allium`, `turbo_kv_layer_adaptive.allium`,
  `turbo_kv_outliers.allium` — sister community-improvement specs
- `turbo_kv_4b_backend.allium` — cross-backend equivalence
- `turbo_kv_4b_attention.allium` — attention-side equivalence
- `mul_mat_cpu.allium`, `nearest_centroid.allium` — foundational

## Skill sequence used this session (reference)

1. `/allium:weed` check-mode — surfaced eviction-vs-overlay divergence.
2. `/allium:tend` — rewrote spec to overlay model, added 6 open questions.
3. Inline tend (`@guidance` on `ReadKVRecentFromOverlay`) — recorded
   approach A vs B decision.
4. `/allium:propagate` — regenerated 23 PBTs from new spec.

Next session's skills:
- Implementation (no skill, just Edit/Write/Bash/Read).
- `/allium:weed` when complete to verify spec-code alignment.
- Possibly `/allium:tend` if new design decisions surface during
  implementation.
