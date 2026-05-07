# Phase 39 — Adopt upstream's collapsed-context chained-rollout MTP

## Hypothesis

Upstream llama.cpp's architecture for Qwen 3.5/3.6 MTP delivers a measured +2.5× over baseline at production context (per `MEMORY.md` entry "Qwen 3.6 27B MTP upstream PR #22673 + froggeric GGUFs"). Their design is structurally simpler than ours and does not have the verify→fused seed dependency that bounded Phase 38 E. **Phase 39 ports the upstream architecture into ik_llama.cpp**, collapsing ctx_mtp into ctx_tgt, folding MTP rollout into the main forward graph, and obsoleting the Phase 38 chain_residual seed plumbing.

Phase 39 is NOT tree drafting. The earlier "tree drafting" framing was wrong — upstream uses **linear chained rollout** (similar in shape to our fused chain) but **inside a single forward pass on the verify context**. The +2.5× lift comes from eliminating the separate ctx_mtp dispatch overhead and from FastMTP vocabulary trimming (lm_head reduced to top-32K tokens), not from tree branching.

## Primary-source references (read these first)

All paths are in `/home/llm/yarn-agentic/llama.cpp/` (upstream master at HEAD `db231ec0d`):

| File | Lines | What it shows |
|---|---|---|
| `src/models/qwen35moe.cpp` | 449–625 | **`llm_build_qwen35moe::build_mtp_head`** — the chained rollout loop, FastMTP trimming, stacked logits emission. Loop reuses MTP layer weights `n_draft_rollout` times, each iteration feeding prior argmax forward. Final stacked tensor is `[n_vocab_mtp, k]` returned via `res->t_logits_mtp`. |
| `src/models/qwen35moe.cpp` | 449–504 | Greedy-token plumbing (uses main forward's argmax as the seed for chain step 0); RMSNorm on hidden state + token embedding; concat + eh_proj projection. |
| `src/models/qwen35moe.cpp` | 540–567 | Per-iteration MTP layer body: build_layer_attn (writes K/V at the MTP layer's slot in ctx_tgt's KV cache), MoE FFN, post-FFN residual. |
| `src/models/qwen35moe.cpp` | 568–610 | Per-iteration FastMTP vocab trim + lm_head + argmax for next iteration's input. |
| `src/models/qwen35moe.cpp` | 612–624 | Stacking via `ggml_concat` on dim 1 (n_tokens axis) into `[n_vocab_mtp, k]`. |
| `src/models/qwen35moe.cpp` | 108 | Call site: `build_mtp_head(inp, inp_pos, sections)` — invoked AT THE END of main forward, before final norm/output. |
| `src/llama-context.cpp` | 1864–1904 | **Post-compute extraction**: pulls stacked MTP logits from device into `mtp_logits_buf`, sets `mtp_n_drafts = n_rollout`. Lazy host-side cache. |
| `src/llama-context.cpp` | 803–807 | `get_mtp_logits` accessor. |
| `src/llama-context.cpp` | 3262–3273 | Public APIs `llama_get_mtp_logits`, `llama_get_mtp_n_drafts`, `llama_get_mtp_n_vocab`. |
| `src/llama-context.h` | 67–69, 261–271 | Context fields: `mtp_logits_buf` (host vector), `mtp_n_drafts`, `mtp_logits_valid`, `mtp_n_vocab`. |
| `include/llama.h` | 1037–1039 | API declarations. |
| `common/speculative.cpp` | 1158–1206 | **`common_mtp_read_drafts`** — host-side argmax per row of mtp_logits, returns token list. Drops EOG tokens to terminate chain. |
| `common/speculative.cpp` | 587–634 | `common_speculative_state_mtp` — speculation impl that calls `common_mtp_read_drafts`. Has `cooldown` flag to skip proposal after rejection. |
| `common/speculative.cpp` | 1007–1090 | `common_speculative_init` registers the MTP impl when `params.type == COMMON_SPECULATIVE_TYPE_MTP`. |
| `src/models/qwen35.cpp` | 398–625 | Same architecture for **dense** Qwen 3.5 (non-MoE). The MoE variant is structurally identical except for the FFN layer. |

## Architectural comparison (ours vs upstream)

| Aspect | ik_llama.cpp (Phase 36/37/38 stack) | llama.cpp upstream (target for Phase 39) |
|---|---|---|
| MTP context | Separate `ctx_mtp` allocated via `llama_init_from_model` (common/speculative.cpp:172–173) | Shared with `ctx_tgt`; no `ctx_mtp` |
| MTP graph | `build_qwen35_mtp_fused` runs as a separate cgraph dispatched via `llama_decode(ctx_mtp, ...)` | `build_mtp_head` invoked inline at end of main forward; lives in same cgraph |
| MTP KV cache | Separate buffer; layer 64 K/V written by fused dispatch | Same buffer as main; layer 64 K/V written during main forward (since the MTP layer is "layer 64" of ctx_tgt's KV) |
| MTP output | `fr.tokens[k]` from `llama_mtp_fused_draft_invoke` | `mtp_logits_buf` accessed via `llama_get_mtp_logits`; argmax done host-side in `common_mtp_read_drafts` |
| Chain mechanism | Loop in `build_qwen35_mtp_fused` (one cgraph, n_steps) | Loop in `build_mtp_head` (one cgraph, n_draft_rollout, weights reused) |
| Dispatch | Two `llama_decode` calls per cycle (verify + fused) | One `llama_decode` (verify with MTP folded in) |
| FastMTP trim | NO — lm_head full vocabulary | YES — `min(lm_head->ne[1], 32768)` reduces matmul ~8× |
| Inline-KV-hook | Phase 36 Step 3: separate `build_qwen35_mtp_kv_only` call inside main forward; writes K/V only | Equivalent semantic but extended through full MTP-layer compute (FFN + norm + lm_head); writes K/V AND produces drafts |
| Chain residual seed | Phase 37 #4 plumbing for cross-cycle seed (broken on this model — see PHASE38.md) | NOT NEEDED — chain reuses argmax-of-main-logits as input each iteration; no seed passing |
| Async dispatch APIs | Phase 38 E `llama_mtp_fused_dispatch_async` etc. | NOT NEEDED — single dispatch, no overlap question |
| Per-cycle GPU time | ~20 ms fused + ~50 ms verify ≈ 70 ms | ~55 ms (single forward, MTP folded) |
| Effective output ratio | Parity (1.05–1.10) with all Phase 38 work | +2.5× per upstream measurements |

**Key insight: upstream's design is what Phase 38 D (cache unification) was trying to enable, plus a different draft-generation pattern.** Phase 38 deferred unification because the seed-prediction problem dominated. Upstream skips the seed-prediction problem entirely by keeping MTP inside the main forward — the chain runs argmax-of-main-logits as its seed token, not h_pre_norm of a prior cycle.

## What Phase 39 ports + what it OBSOLETES

**Ports from upstream** (see file/line refs above):

1. **`build_mtp_head` graph builder** — chained rollout, FastMTP vocab trim, stacked logits.
2. **Context fields** — `mtp_logits_buf`, `mtp_n_drafts`, `mtp_n_vocab`, `mtp_logits_valid`.
3. **Post-compute extraction** — pull stacked MTP logits to host (lazy-cached).
4. **Public APIs** — `llama_get_mtp_logits`, `llama_get_mtp_n_drafts`, `llama_get_mtp_n_vocab`.
5. **Speculation glue** — `common_mtp_read_drafts` (host-side argmax + EOG drop).
6. **Speculative state class** — `common_speculative_state_mtp` with cooldown flag.
7. **Env knob** — `LLAMA_MTP_ROLLOUT=N` to control rollout depth.

**Obsoletes from Phase 36/37/38** (delete or env-gate OFF):

| Component | Disposition |
|---|---|
| `ctx_mtp` (`common_speculative_state_mtp::ctx_mtp` in our code) | DELETE the separate context allocation. Reuse `ctx_tgt`. |
| `build_qwen35_mtp_fused` (Phase 36 Step 1) | Replace with upstream's `build_mtp_head`. |
| `build_qwen35_mtp_kv_only` (Phase 36 Step 3 inline-KV-hook) | DELETE. The new `build_mtp_head` runs full MTP compute (not just KV-only) and integrates at the same point in main forward. |
| `build_qwen35_mtp_chain_residual` (Phase 37 #3 shared primitive) | DELETE. No longer needed — upstream uses inline rollout, not chain primitives. |
| `mtp_persist[]` and related (Phase 38 B) | DELETE. No cross-cycle h_pre_norm transfer needed. |
| `mtp_fused_chain_residuals[]`, `mtp_fused_n_extend`, `pending_chain_residual_step`, `mtp_fused_async_guess`, `mtp_fused_pending_gf` (Phase 38 C/E) | DELETE. |
| `llama_mtp_fused_draft_invoke`, `llama_mtp_fused_dispatch_async`, `llama_mtp_fused_extract_results`, `llama_mtp_fused_extract_results`, `llama_set_draft_input_chain_residual`, `llama_mtp_set_persist_from_host` | DELETE the APIs. Replace caller with `common_mtp_read_drafts`. |
| Diagnostic env knobs `LLAMA_MTP_FUSED`, `LLAMA_MTP_INLINE_KV`, `LLAMA_MTP_FUSED_EXTEND`, `LLAMA_MTP_FULL_2`, `LLAMA_MTP_CHAIN_MIN_PROB`, `LLAMA_MTP_CHAIN_RESIDUAL_SEED` | DELETE. Replace with single `LLAMA_MTP_ROLLOUT=N`. |
| Server slot fields `slot.mtp_hidden_state`, `slot.mtp_next_chain_residual_step`, `slot.has_mtp` (last one may stay) | Delete the host-bounce buffers. `has_mtp` stays. |

**Preserves from Phase 36/37/38** (no obsolescence):

- Phase 37 graph reuse (Phase 37 #5) — applies to the new MTP graph just as it did to the old fused graph. Generic infrastructure.
- Recalibrated harness gate (`tests/mtp-fused/gate.yaml`, `effective_output_ratio` floor) — same metric, different magnitude.
- CLAUDE.md §8 (estimate in tokens, not days) — generic.
- The diagnostic finding about chain_residual ≠ h_pre_norm distribution stays in PHASE38.md as historical record.

## Schedule

Each step has a token-cost estimate, an explicit verification step, and a commit boundary. Per CLAUDE.md §5 every PHASE39.md edit is committed and pushed immediately.

### Pre-flight: read upstream code

| Step | What | Verify |
|---|---|---|
| 39.0a | Read all primary-source refs above (file/line table). Take notes on tensor shapes, attention mask semantics, KV layer slot. | Notes captured in this session's working memory. |
| 39.0b | Diff upstream's `build_qwen35_moe` (whole file) against ours (`graphs/build_qwen35.cpp` `build_qwen35moe`). Identify what we have that upstream doesn't (Phase 36/37/38 additions) and what upstream has that we don't (FastMTP, mtp_inp_hidden filtering for prompt processing). | Diff reviewed. |

**Token cost estimate**: 30–50k.

### Phase 39.A: collapse ctx_mtp into ctx_tgt

This is Phase 38 D (KV unification) but done by REMOVING the separate context entirely, not by sharing buffers across two contexts. Cleaner.

| Step | File(s) | What | Verify |
|---|---|---|---|
| 39.A1 | `common/speculative.cpp` ~155–187 | Modify `common_speculative_state_mtp` constructor to NOT allocate `ctx_mtp`. Use `ctx_tgt` directly. Remove `ctx_mtp` member. Update `common_speculative_get_mtp_ctx` to return `ctx_tgt` (or remove the function entirely). | Builds clean. |
| 39.A2 | `examples/server/server-context.cpp` (multiple sites) | Replace all `mtp_target = mtp_ctx ? mtp_ctx : ctx;` with just `ctx`. Remove `mtp_ctx = common_speculative_get_mtp_ctx(slot.spec);` lookups. | Builds clean. Server boots. |
| 39.A3 | `common/speculative.cpp` `mtp_speculative_gen_draft` | Replace fused-dispatch path with `llama_tokens drafts = common_mtp_read_drafts(ctx, n_draft);` (after porting the function — comes in 39.B). For now, stub returning empty drafts. | Builds clean. |
| 39.A4 | `common/speculative.cpp` `mtp_update_kv_cache`, `mtp_accept_tokens` | DELETE these helpers. Their semantic (writing MTP layer K/V via UPDATE_ACCEPTED) is replaced by the MTP layer's K/V being written DURING MAIN FORWARD via the new `build_mtp_head`. | Builds clean. Remove all callers. |

**Token cost estimate**: 40–60k. Risk: `ctx_tgt` config may need adjustment (`embeddings = true`, `mtp = true`) — currently set on `cparams_dft` only. Move them to `params_base.cparams` (the main ctx's config).

### Phase 39.B: port MTP graph builder + context fields + APIs

| Step | File(s) | What | Verify |
|---|---|---|---|
| 39.B1 | `src/llama-context.h` | Add fields `mtp_logits_buf`, `mtp_n_drafts`, `mtp_n_vocab`, `mtp_logits_valid` (analogous to upstream lines 261–271). | Compiles. |
| 39.B2 | `src/llama-context.cpp` (or `llama.cpp`) | Add `get_mtp_logits()` and `get_mtp_n_drafts()` accessors. Public APIs `llama_get_mtp_logits`, `llama_get_mtp_n_drafts`, `llama_get_mtp_n_vocab`. | Linker resolves. |
| 39.B3 | `include/llama.h` | Declare the public APIs. | Header compiles. |
| 39.B4 | `src/graphs/build_qwen35.cpp` (or new `src/graphs/build_qwen35_mtp_head.cpp`) | Port `build_mtp_head` from upstream. Adapt to ik_llama.cpp's helpers — `build_norm` is `llm_build_norm`, `build_lora_mm` is `llm_build_lora_mm`, `build_layer_attn` maps to `build_std_attention`, `build_layer_ffn` maps to `llm_build_std_moe_ffn` (for MoE) or `llm_build_ffn` (for dense). | `build_qwen35moe()` calls it; cgraph builds without crashes. |
| 39.B5 | `src/graphs/build_qwen35.cpp` lines 95–111 | DELETE the inline-KV-hook block. Replace with `build_mtp_head(...)` call (matches upstream's call site at line 108 of `qwen35moe.cpp`). | Cgraph topology change committed. |
| 39.B6 | `src/llama.cpp` `llama_decode_internal` post-compute | Add post-compute extraction analogous to upstream lines 1864–1904: pull stacked MTP logits to host buffer, set `mtp_n_drafts`. Gate on `cparams.mtp` and presence of `t_logits_mtp` tag. | First decode produces non-zero `mtp_logits_buf`. |

**Token cost estimate**: 80–120k. This is the bulk of the work. Risk: `build_layer_attn` semantics may differ — upstream uses `inp->get_attn()` for the input wrapper while we use `KQ_mask + inp_pos` directly. Port carefully; may need to construct an analogous input wrapper.

### Phase 39.C: speculative.cpp glue + server integration

| Step | File(s) | What | Verify |
|---|---|---|---|
| 39.C1 | `common/speculative.cpp` after `mtp_update_kv_cache` deletion | Add `common_mtp_read_drafts(llama_context * ctx, int k_max)` (analogous to upstream lines 1158–1206). | Builds clean. |
| 39.C2 | `common/speculative.cpp` `mtp_speculative_gen_draft` | Wire `common_mtp_read_drafts(ctx, n_draft)` as the replacement for fused dispatch. Remove the LLAMA_MTP_FUSED/FUSED_EXTEND/FULL_2/CHAIN_MIN_PROB/CHAIN_RESIDUAL_SEED env knobs. Add a single `LLAMA_MTP_ROLLOUT=N` env knob (read in `build_mtp_head`). | mtp drafts returned to caller. |
| 39.C3 | `examples/server/server-context.cpp` ~3179–3196 (`add_sampled_tokens` MTP path) | Remove the `slot.mtp_hidden_state` host-bounce setup. The MTP rollout produces drafts directly from main forward — no seed plumbing needed. | Server boots, generates text. |
| 39.C4 | `examples/server/server-context.cpp` Phase B of `speculative_decoding_accept` | Remove all `slot.mtp_hidden_state` updates and the `slot.mtp_next_chain_residual_step` arming. UPDATE_ACCEPTED path is gone. | Phase B simplifies; server still works. |
| 39.C5 | `examples/server/server-context.h` | Remove `slot.mtp_hidden_state`, `slot.mtp_next_chain_residual_step` fields. Keep `slot.has_mtp`. | Header compiles. |

**Token cost estimate**: 30–50k.

### Phase 39.D: cleanup of obsolete Phase 36/37/38 code

Run-through of the obsolete-table above to remove dead code. Each item below maps to a deletion:

| Step | What to delete |
|---|---|
| 39.D1 | `build_qwen35_mtp_fused`, `build_qwen35_mtp_kv_only`, `build_qwen35_mtp_chain_residual` from `src/llama-build-context.h` and `src/graphs/build_qwen35.cpp` |
| 39.D2 | `cparams.mtp_fused_n_steps`, `cparams.mtp_fused_n_extend`, `cparams.mtp_inline_kv_hook` from `src/llama-cparams.h` and all references |
| 39.D3 | `lctx.mtp_fused_chain_residuals[]`, `lctx.mtp_persist*`, `lctx.mtp_fused_skip_extraction`, `lctx.mtp_fused_pending_gf`, `lctx.mtp_fused_async_guess`, `lctx.mtp_fused_offset_*`, `lctx.t_h_pre_norm`, `lctx.inp_mtp_states`, `lctx.draft_input_hidden_state*`, `lctx.draft_residual_dev*` from `src/llama-context.h` |
| 39.D4 | All Phase 38 APIs in `include/llama.h` (the dispatch_async, extract_results, set_persist_from_host, set_draft_input_chain_residual, get_async_guess, has_pending_async, etc.) |
| 39.D5 | `Prev::mtp_op_type`, `Prev::mtp_fused_n_steps`, `Prev::mtp_fused_n_extend` in `src/llama.cpp` Prev struct (the graph-reuse comparator); generic `n_tokens` stays |
| 39.D6 | Server slot fields per 39.C5 |
| 39.D7 | `mtp_update_kv_cache`, `mtp_accept_tokens` from `common/speculative.cpp` and callers |
| 39.D8 | All env knobs except `LLAMA_MTP_ROLLOUT` (and `LLAMA_MTP_INPUT_CHECKSUM` may stay as a generic debug aid) |

**Token cost estimate**: 40–60k. Mostly mechanical. Catch-all for anything left after 39.A/B/C compile/test cycles.

### Phase 39.E: harness update + first measurement

| Step | What | Verify |
|---|---|---|
| 39.E1 | Update `scripts/test-fused-harness.sh`: replace fused env block with `LLAMA_MTP_ROLLOUT=3` (or whatever depth). Drop the old env stack. | Harness runs. |
| 39.E2 | Update `tests/mtp-fused/gate.yaml`: keep the structure, plan to ratchet thresholds after measurement. Don't pre-set new thresholds — measure first, then ratchet. | Harness exits cleanly. |
| 39.E3 | Run `--fast` harness. Record `effective_output_ratio`, `accept_d3_ratio`, `tg_d3_ratio`. | Numbers logged. |
| 39.E4 | Run `--slow` harness. Record. Compare to upstream's published 2.5× on production-context. | Numbers logged. |
| 39.E5 | Ratchet `gate.yaml`: set `effective_output_ratio` thresholds at (measured - 5% noise margin). The new gate documents Phase 39's measured ceiling, not the original parity floor. | gate.yaml committed with new thresholds. |

**Token cost estimate**: 30–50k (harness runs + iteration if measurements are surprising).

### Phase 39.F: production swap + closure

| Step | What | Verify |
|---|---|---|
| 39.F1 | Update `/home/llm/profiles/qwen36-27b-x1.sh` (or `active.sh`) to set `LLAMA_MTP_ROLLOUT=3` and re-enable MTP via cparams. Restart `systemctl --user restart llama-server`. Smoke-test a chat completion. | Production runs cleanly with measured throughput. |
| 39.F2 | Append closure section to PHASE39.md with measurements, retain table of upstream-vs-final-ours throughput. | PHASE39.md complete. |
| 39.F3 | Append correction note to PHASE38.md pointing at PHASE39.md. ("Phase 38 E's +18% projection is moot — Phase 39's collapsed-context architecture delivers +Y% via different mechanism.") | PHASE38.md updated, no rewrites of historical text. |
| 39.F4 | MEMORY.md entry summarising Phase 39 lesson: collapsed-context MTP rollout + FastMTP vocabulary trim + single dispatch is the architectural win on this model class. | MEMORY.md committed. |
| 39.F5 | SUMMARY.md updated for mdBook. | SUMMARY.md committed. |

**Token cost estimate**: 20–30k.

## Total token budget

| Phase | Range |
|---|---|
| 39.0 (read upstream) | 30–50k |
| 39.A (collapse ctx_mtp) | 40–60k |
| 39.B (port MTP graph + APIs) | 80–120k |
| 39.C (speculative.cpp + server glue) | 30–50k |
| 39.D (cleanup) | 40–60k |
| 39.E (harness + measurement) | 30–50k |
| 39.F (production + closure) | 20–30k |
| **Total** | **270–420k** |

This fits in a single 1M context session with margin (under 50% target, leaving room for measurement-driven iteration).

## Binding closure criteria for Phase 39

Phase 39 closes (`[x]`) when, and only when, ALL hold:

1. All schedule items 39.A–39.F implemented and committed.
2. No reference to ctx_mtp, build_qwen35_mtp_fused, mtp_persist, chain_residual_seed, or LLAMA_MTP_FUSED env knobs remains in the source tree (grep returns empty).
3. `--fast` harness GREEN at recalibrated thresholds (no regression).
4. `--slow` harness shows `effective_output_ratio ≥ 2.0` at production context. The +2.5× upstream measurement is the binding ratchet target; 2.0 is the floor allowing for noise + hardware differences. Below 2.0 means the port lost something; reopen.
5. Production swapped to the new MTP path; smoke-tested; old GGUF/profile retained for 7-day rollback.

## Anti-goals (won't do without explicit user approval)

- Keep ctx_mtp "for safety" while porting the new path. Either commit to collapse or don't start.
- Preserve any of the broken Phase 38 plumbing (chain_residual seed, async dispatch, persist buffer) "in case it's useful later". It's not. Phase 39's architecture moots it.
- Tree drafting. Not in upstream. Not the +2.5× source. Don't conflate.
- Re-derive upstream's algorithm from scratch. Read it, port it. Primary-source references are above.
- Run --slow on the assumption fast GREEN → slow GREEN. Always measure both.

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| Upstream's `build_layer_attn` API differs from our `build_std_attention` | Carefully port the input-tensor construction (input wrapper). Compare graph nodes between upstream's expected and ours. Test with `--fast` early in 39.B. |
| FastMTP vocab trim (top-32K) may produce wrong tokens for our specific GGUF | Make the trim size an env knob (`LLAMA_MTP_VOCAB_TRIM=N`, default 32768). Disable for first --fast run; enable after baseline confirmed. |
| `cparams.mtp` and `cparams.embeddings` settings need different lifecycle (currently set on cparams_dft) | Move to params_base.cparams during 39.A2. Verify with cold-start server. |
| Graph reuse (Phase 37 #5) cache key may collide with the new graph topology | Update `Prev` struct to track only the parameters that affect topology. Verify hit rate via existing `g_can_reuse_*` counters. |
| Production model is built without nextn weights or with different layer count | Verify `hparams.nextn_predict_layers > 0` and `model.layers[il_mtp].nextn.eh_proj != nullptr` before attempting Phase 39 port. The production GGUF is `qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf` — confirm it has nextn weights via `gguf-dump` or `llama_model_n_nextn_layer()`. |
| Cross-context API removals may break tests | Run full test suite (`tests/mtp-*`, `tests/mtp-fused/*`, `tests/mtp-ubatch-hook/*`) after 39.D. Some tests may need to be retired (those that tested the now-deleted ctx_mtp behavior). |

## Picking up after compaction

If a fresh session starts here, the entry points are:

1. **Read this file** (PHASE39.md) end-to-end.
2. **Read PHASE38.md "Path 3 chosen"** section + "Phase 38 outcome" — gives the full diagnostic context for why Phase 39 looks the way it does.
3. **Read CLAUDE.md §8** — the time-vs-tokens reframe that informs Phase 39's execution mode.
4. **Pull up the primary-source references table** at the top of this file. Read upstream's code at those line numbers. Take notes on tensor shapes, masking, KV layer indexing.
5. **Begin at 39.A1**. Work the schedule top-down. Commit per CLAUDE.md §5 (every PHASE39.md edit, every meaningful code chunk).
6. **Don't get clever**. Upstream's code is the spec. Port it. Adapt where ik_llama.cpp helpers differ. The architecture is proven; the work is mechanical translation.

## Why Phase 39 is more promising than continuing Phase 38

- Phase 38 hit a fundamental dependency (verify→fused seed) that we couldn't break with our toolset on this model.
- Phase 39's design DOESN'T HAVE that dependency. MTP rollout uses argmax-of-main-logits as its seed token — which IS computed during main forward. No cross-cycle seed transfer. No prediction needed.
- Upstream measures +2.5× empirically on this model class. Not a projection. Direct portability evidence.
- The token budget for Phase 39 (270–420k) fits a single 1M-context session. The work is well-scoped because upstream is the spec.
- Phase 39 obsoletes most of Phase 36/37/38's complexity. The codebase becomes SMALLER, not larger. Architectural simplification is durable.

## What this session leaves for the next

- This document, complete and exhaustive.
- PHASE38.md with the diagnostic record (chain_residual ≠ h_pre_norm distribution, dependency analysis, measurement evidence).
- CLAUDE.md §8 (estimate in tokens).
- Tree state at parity (effective 1.067 PASS on --fast).
- Diagnostic tooling in place (`LLAMA_MTP_FULL_2_DIAG`, `LLAMA_MTP_INPUT_CHECKSUM`) — will be removed during 39.D cleanup.
- All Phase 38 commits on `phase36-mtp-throughput` (ik_llama.cpp) and `phase32-q4_0_ar16-integration` (yarn-agentic). Pushed.

The next session has a clean foundation, a precise plan, primary-source references, and a budget that fits the work. The bet for the next 1M context is whether 39.A–39.F land cleanly with the upstream architecture as the spec.
