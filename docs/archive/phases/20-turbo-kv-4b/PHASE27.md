# Phase 27: TURBO_KV_4B — Residual Window Implementation Design

## Status

**Design in progress.** No code is changed in this phase. PHASE27 captures the architectural decisions needed to close PHASE26 Tier 1.1 — the unimplemented residual window feature that `turbo_kv_residual_window.allium` describes and that `test-turbo-kv-residual-window-pbt.cpp` skip-stubs 18 obligations against.

A subsequent PHASE28 will implement the design, convert the skips into real `rc::check` properties, and re-measure 9B IQ3_XXS PPL.

## Motivation

Measurement chain from PHASE26 Tier 1.1:

- 9B Qwen3.5 IQ3_XXS: `turbo_kv_4b` on K (V=F16) is **+0.23 PPL** above the F16 KV baseline.
- `turbo_kv_4b_attention.allium` PBT confirms CPU FA ≡ Vulkan FA at head_dim=256 within fp32 tolerance — the gap is algorithmic, not a kernel bug.
- [TONBI-V3] measurement (`tonbistudio/turboquant-pytorch compressors_v3.py` README, Llama 3.2 needle test at 4K context):
  - `K4/V4 rw=128` → **EXACT** needle retrieval.
  - `K4/V4 rw=0` → **MISS** (garbage output).
  - `K4/V2 rw=0` → **MISS**.
- [TONBI-V3]'s explanation: "3-4 bit compression without a residual window produces garbage, same as V2."

The port runs exactly the failing `rw=0` configuration. Closing the gap starts with giving it a non-zero rolling fp16 tail.

## Scope

**In scope.**

- K-side residual window on the existing `TURBO_KV_4B` type. V is still F16 in current Qwen3.5 production, so there's no V-side work yet — V changes land with Tier 1.2 (asymmetric K/V).
- CPU dispatch path through `ggml_compute_forward_flash_attn_ext_f16_one_chunk` (`ops.cpp:8190`).
- Vulkan dispatch path for `GGML_OP_FLASH_ATTN_EXT` with mixed F16/TURBO_KV_4B K rows.
- Config surface: `--cache-residual-window N` on `llama-server` and `llama-cli`, default 0 (disabled) to preserve current behaviour until the feature is validated, but spec default 128 documented as the recommended production value.
- Roundtrip measurement gate: 9B IQ3_XXS PPL must drop back toward F16 baseline (gate: regression below +0.05 PPL).

**Out of scope.**

- V-side residual window (composes with Tier 1.2 asymmetric).
- Multi-seq / beam-search window accounting. First cut is single-seq.
- GGUF serialisation of the fp16 window across save/load. The window is rebuilt at load time from the uncompressed head of the stored cache.
- Per-layer residual-window overrides. Global config for PHASE28; per-layer tuning is a sibling improvement (Tier 2.1 layer-adaptive composes here).
- CUDA / ROCm / Metal dispatch. Only CPU + Vulkan land in PHASE28. HIPIFY gives ROCm when CUDA arrives.
- Attention-gated adaptive (TheTom/turboquant_plus): runtime decision to dequantise tokens on the fly. Out of scope per the spec's open question.

## Prior art in the port: SWA

The KV cache already supports Sliding Window Attention (`src/llama-kv-cache.h:257-276`):

```
const uint32_t n_swa = 0;
const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;
```

SWA discards tokens outside the window entirely. Residual window is its lossy cousin: it **retains** all tokens but compresses positions outside the fp16 tail. The window-bookkeeping code paths in the cache are a useful reference — attention masking against `n_swa` is structurally the same as masking quantised-vs-fp16 rows. PHASE28 should read `llama-kv-cache.cpp` and `llama-kv-cache-iswa.cpp` before touching any new file.

## Design decisions

### D1 — Dedicated fp16 side-buffer vs interleaved into the existing K tensor

**Decision: dedicated fp16 side-buffer.**

**Options considered.**

- (a) Interleave: store fp16 and quantised rows in the same buffer, tagged by per-row metadata. Attention dispatch branches per row.
- (b) Dedicated: a second ggml tensor per layer, fp16 type, sized `[head_dim, n_heads_kv, residual_window]`. Attention dispatch does two passes (quantised tail + fp16 window) then concatenates.

**Rationale for (b).**

- (a) forces the attention kernel to branch per row on a precision flag — bad for SIMD / shader throughput. Single-pass attention would need scalar fallback.
- (b) matches [TONBI-V3]'s Python implementation (`compress_kv` returns a dict with separate `compressed` and `fp16` keys).
- (b) lets the kernel remain type-homogeneous within each pass — the quantised-tail path is exactly today's FA with `TURBO_KV_4B` K, the fp16-window path is exactly today's FA with F16 K.
- Memory overhead: `residual_window * head_dim * n_heads_kv * n_layers * 2 bytes`. At rw=128, head_dim=256, n_heads_kv=4, 36 layers → 9.4 MB per sequence. Negligible.

### D2 — Quantisation at eviction time

**Decision: quantise synchronously on the append path, reusing `quantize_row_turbo_kv_4b_ref`.**

When `seq_len > residual_window`:

1. The row at position `seq_len - residual_window - 1` in the fp16 side-buffer is the oldest tracked row.
2. Run `quantize_row_turbo_kv_4b_ref` on that row's data, writing into the corresponding slot in the quantised K tensor.
3. Shift the fp16 side-buffer (or advance a ring-buffer head pointer — prefer the ring-buffer to avoid memmove).
4. Write the new row into the freed slot.

**Rationale.** Synchronous eviction keeps the cache consistent after every append, which matches the spec's `AppendRowWithEviction` postcondition. An async or deferred variant would complicate attention reads (which row is currently fp16? a race). The quantisation cost per append at head_dim=256 is ~311 ns (PHASE25 AVX2) × 2 blocks/head × n_heads_kv rows × eviction_rate — amortised over a full forward pass it is < 1 % of per-token latency.

### D3 — Ring buffer vs linear buffer for the fp16 window

**Decision: ring buffer with a head-position index.**

Per layer, the fp16 tensor shape is `[head_dim, n_heads_kv, residual_window]` plus a scalar `window_head` counter tracking which slot is position 0 of the window. Avoids `memmove(N-1)` at every append.

**Cost.** Attention read needs to unwrap the ring — two slices `[window_head, residual_window)` and `[0, window_head)` concatenated. This is one extra conditional per read path, negligible.

### D4 — Config surface

- `--cache-residual-window N` CLI flag on `llama-cli`, `llama-server`, `llama-bench`.
- `llama_context_params.residual_window` (uint32_t). Default 0 = disabled.
- Propagated through `llama-cparams.h` alongside the existing `n_swa` field.
- Validated at context init: must be `0` or in `[1, n_ctx)`.
- When > 0 AND either K or V type is a TURBO_KV type, allocate the side-buffer(s). When K and V are both non-TURBO types, ignore the flag (issue a log warning, no error).

### D5 — Attention dispatch

**Decision: two-pass concat.**

In `ggml_compute_forward_flash_attn_ext_f16_one_chunk` for each query:

1. If `residual_window > 0` and `seq_len > residual_window`:
   - Pass 1: run existing `use_turbo_kv_4b` path over the quantised tail `[0, seq_len - rw)`.
   - Pass 2: run a vanilla F16 KV path over the fp16 window `[seq_len - rw, seq_len)`.
   - Online softmax merges per-position scores from both passes using the standard max-normalised reduction (Rabe & Staats, the `S`, `M` running state already in the function).
2. If `residual_window == 0` or `seq_len <= residual_window`:
   - Single pass. When `seq_len <= rw`, everything is fp16 → fall through to the existing F16 KV path. When `rw == 0`, fall through to the existing `use_turbo_kv_4b` path.

The online-softmax merge is the existing code. Each pass already accumulates `S` (sum of softmax numerators) and `M` (running max). Running two passes in sequence against the same `S` / `M` accumulator produces the same result as a single pass over the concatenation — softmax is associative across the stream.

On Vulkan the same split applies. The FA shader dispatches over a K-range; running it twice with different K slices and the same output buffer is already supported by the `use_tiled` / split-K machinery.

### D6 — Per-layer config deferred

The spec allows global `residual_window`. Per-layer overrides (e.g. "early layers use rw=256, middle layers rw=128") are a composition with `turbo_kv_layer_adaptive.allium`. That's a separate phase. PHASE28 ships the single global config.

## Implementation plan (PHASE28)

Each step has an explicit verification check per CLAUDE.md §4.

1. **Add `residual_window` to cparams** → verify by
   - `llama_context_default_params()` returns `residual_window = 128` (Q1 decision)
   - CLI flag parsing accepts explicit `--cache-residual-window 0` to disable
   - CLI flag parsing clamps any value > `n_ctx` to `n_ctx` with a warning (Q2 decision)
   - Test: `llama-cli --cache-residual-window 128 ...` echoes the value in startup log
   - Test: `llama-cli --cache-residual-window 999999 -c 2048 ...` logs the clamp and proceeds
2. **Allocate fp16 side-buffer in `llama-kv-cache.cpp` at init** (Q3 decision) → verify by
   - `llama_kv_cache_init` allocates the full `residual_window * head_dim * n_heads_kv * 2 bytes` per layer up front when `residual_window > 0`
   - Assert no allocation when `residual_window == 0` (byte-exact regression check vs today's binary)
   - No reallocation path exists in the append hot loop (grep check)
3. **Rolling write path on `kv_cache_seq_add` / equivalent append** → verify by
   - Append 200 tokens at rw=128; assert positions [0, 72) are quantised, positions [72, 200) are fp16
   - Assert fp16 slot values match the original fp32 data within fp16 precision
4. **Quantise-on-eviction** → verify by
   - Append token 129 (first eviction); assert the quantised slot for position 0 matches `quantize_row_turbo_kv_4b_ref(original_position_0)` byte-for-byte
5. **Two-pass attention read (CPU)** → verify by
   - `test-turbo-kv-residual-window-pbt.cpp`'s `rule-success.ReadKVLongSequence` obligation converts from SKIP to rc::check PASS
   - `test-turbo-kv-residual-window-pbt.cpp`'s `invariant.CoverageCompleteNoOverlap` converts from SKIP to PASS
6. **Two-pass attention read (Vulkan)** → verify by
   - `test-turbo-kv-attention-pbt` re-run with `residual_window=128` shows CPU ≡ Vulkan within tolerance
7. **GGUF state save/load with fp16 window** (Q4 decision) → verify by
   - Save state with `--cache-residual-window 128` at `seq_len = 500`; load into a fresh context; assert the first 128 positions of attention output at the next token match byte-for-byte against a no-save-load control run.
   - Assert `kv.residual_window` metadata key is present and equals 128 on save.
   - Assert per-layer window buffer section is written and its byte count matches `residual_window * head_dim * n_heads_kv * 2`.
   - Cross-config reject: save with `rw=128`, load with `rw=64` → context init returns an error citing the mismatch.
   - Backward compat: load a pre-PHASE28 saved state (no `kv.residual_window` key) → proceeds with an empty window, logs an info message.
8. **9B IQ3_XXS PPL gate** → verify by
   - `reference/ppl/compare_kv_quants.sh` run against Qwen3.5-9B-UD-IQ3_XXS with `--cache-type-k turbo_kv_4b --cache-residual-window 128`
   - Gate: PPL above F16 baseline must be ≤ +0.05 (down from today's +0.23). If > +0.05, the per-token-weighted-magnitude evidence says something else is also broken — do not ship, open a follow-on investigation.
9. **Skip-to-rc::check conversion** → verify by
   - `test-turbo-kv-residual-window-pbt`: SKIP count drops from 18 to 0 (or to a smaller set explicitly deferred to later phases — that set must be named)

## Test strategy

`test-turbo-kv-residual-window-pbt.cpp` (skip-stub from commit `f01eab79b`) is the ground truth. Each obligation's SKIP has a comment describing what the `rc::check` should do when the API lands. PHASE28 converts them one at a time, never deleting an obligation — if a rule becomes trivial after implementation, that's fine, the check can be a compile-time assert.

Additional coverage to add in PHASE28 (not in the skip-stub today):

- **Cross-pass softmax merge equivalence.** Property: for a query against 200 positions with rw=128, the output of two-pass FA equals the output of single-pass FA over the concatenated dequantised tail + fp16 window, within `turbo_kv_4b_attention.allium`'s `output_rel_tol = 1e-5`.
- **Ring-buffer invariant.** Property: after any sequence of appends, `window_head + residual_window - 1 ≡ seq_len - 1  (mod residual_window)`.
- **Eviction consistency.** Property: after an eviction, the quantised-and-dequantised value of the evicted row matches the original fp32 within `reconstruction_rel_error = 0.1`.

## Open questions — resolved 2026-04-24

**Q1 — Default value of `--cache-residual-window`: 128 (SOTA by default).** Any user running `--cache-type-k turbo_kv_4b` gets the quality win on upgrade. Memory cost (~9 MB/seq at head_dim=256) is accepted as part of the correct configuration. Rationale: the port today implicitly runs the quality-broken `rw=0` configuration; making the correct configuration the default is the fix.

**Q2 — Behaviour when `residual_window > n_ctx`: clamp to `n_ctx` and log a warning.** The effective window becomes the whole sequence (no quantisation at all — equivalent to running without `turbo_kv_4b`). User sees `[warn] --cache-residual-window=N exceeds n_ctx=M; clamping to n_ctx (no KV quantisation will occur)`.

**Q3 — Memory allocation timing: allocate full size at context init.** Deterministic memory footprint, no reallocation during inference. The small short-session saving from grow-on-demand is not worth the hot-path reallocation branch.

**Q4 — GGUF cache persistence: persist the window bytes-for-bytes.** Full reload fidelity. Requires PHASE28 to add a new saved-state section for the fp16 window plus metadata keys. Scope implications:
- New GGUF metadata key: `kv.residual_window = N` (uint32_t). Readers that don't understand the key ignore it (standard GGUF behaviour).
- Per-layer state section: `kv.layer.L.residual_window = bytes` (raw fp16 buffer + a `window_head` uint32_t ring-buffer pointer).
- Saved-state format is additive. Loading a pre-PHASE28 saved state: no key present → assume window is empty, attention falls back to all-quantised for positions outside the new window until it refills naturally.
- Loading a PHASE28 saved state at a later configuration (e.g. different `residual_window` or `--cache-type-k` switched off): reject load with a clear error citing the mismatch. Same behaviour as today's cache-type mismatch.

## Files expected to change (PHASE28)

- `src/llama-cparams.h` — add `residual_window` field
- `src/llama-cparams.cpp` — default initialisation
- `src/llama-context.cpp` — accept the new `llama_context_params` field
- `src/llama-kv-cache.h` — add fp16 side-buffer members, window-head counter
- `src/llama-kv-cache.cpp` — allocate side-buffer, rolling-write + eviction in append, two-pass read helpers
- `ggml/src/ggml-cpu/ops.cpp` — two-pass dispatch in `ggml_compute_forward_flash_attn_ext_f16_one_chunk`
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp` — same split on the Vulkan FA path
- `common/common.h` + `common/arg.cpp` — CLI flag
- `tools/server/*` — server param plumbing
- `tests/test-turbo-kv-residual-window-pbt.cpp` — SKIP → rc::check conversions
- `reference/ppl/compare_kv_quants.sh` — add an rw=128 row to the sweep
- `src/llama-kv-cache.cpp` state write/read paths — Q4 GGUF window persistence
- `tests/test-save-load-state.cpp` or equivalent — save/load round-trip with rw > 0

## Risk register

Every entry here is a task — if it needs mitigation, the mitigation is an implementation sub-step in PHASE28, not advisory text.

- **R1. Online softmax merge across two passes.** If the two-pass dispatch does not preserve bit-exact numerical behaviour with today's single-pass, FA PBT will falsify. Mitigation: study the existing `ggml_flash_attn_ext_reduce_partials` path (`ops.cpp:8877`) — that's already the same cross-pass merge for the split-K case.
- **R2. Ring-buffer wrap bug.** Attention dispatch must unwrap the ring into the correct global position order. Mitigation: `test-turbo-kv-residual-window-pbt.cpp`'s `invariant.CoverageCompleteNoOverlap` covers this exactly once the skip is converted.
- **R3. Per-sequence accounting at batch > 1.** Current port is single-sequence in production; multi-seq server workloads hit a different code path. Mitigation: Q1 default 0 with an explicit opt-in means multi-seq users don't get blocked by this phase. Multi-seq residual window is a separate phase.
- **R4. GPU FA shader fallback on multi-block head_dim.** Vulkan already handles `head_dim=256` today; the two-pass split reuses the same shader at different K-range slices. Mitigation: re-run `test-turbo-kv-attention-pbt` after PHASE28 step 6.

## Next actions after PHASE28 lands

- PHASE29: asymmetric K/V (Tier 1.2). Probably simpler after the side-buffer machinery exists — V-side residual window composes directly.
- PHASE30: re-measure 9B IQ3_XXS + 35B-A3B with full Tier-1 (residual window + asymmetric K4/V3).
- PHASE31: layer-adaptive (Tier 2.1).
- Outlier handling (Tier 2.2) remains deferred unless PHASE30 shows insufficient quality closure.
