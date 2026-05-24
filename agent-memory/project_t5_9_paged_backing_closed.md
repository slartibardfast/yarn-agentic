---
name: project-t5-9-paged-backing-closed
description: "production/2026-q2-next 2026-05-23 — T5.9 paged BACKING CLOSED. All seven binding gates GREEN. Block-major K/V tensor [head_dim, BLOCK, n_head_kv, total_pool_blocks]; n_kv cap at BLOCK × total_pool_blocks / n_stream; find_slot pool-exhaustion → GGML_STATUS_ALLOC_FAILED → server 503 + Retry-After. Production AUTO mode byte-identical to T5.8 (seed_identity_per_stream). User override --kv-pool-blocks N delivers ctx ≥ 1M feasibility with clean admission. State-save under override is a NEW deferral. Tier 5 CLOSED."
metadata:
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

T5.9 (paged BACKING + admission gate) landed and CLOSED Tier 5 same day as the T5.8 reopen on `production/2026-q2-next` 2026-05-23. Supersedes `[[project-t5-9-b-landed-npc-pass]]` (the mid-implementation pickup memory).

**HEADS.**
- Submodule HEAD `e8ab38be` (T5.9.B' fix-forward: n_kv cap + ALLOC_FAILED wiring).
- Parent HEAD `15b52a5` (MEMORY commit; immediately preceded by `3b87046` PHASE closure, `f78abf65` submodule bump, `d647879c` T5.9.C harness+ledger).
- Branch `production/2026-q2-next`.

**Status grid (all GREEN):**
- T5.9.A — spec (`paged_kv_pool_sizing.allium`, 3 contracts) + TLA (PagedKVAllocator extension + sibling MC) + trace validator (PoolBoundsRespected check + header parsing) + 2 spec tests. ✓
- T5.9.B — block-major layout flip + `seed_identity_per_stream` + `--kv-pool-blocks N` CLI + find_slot admission pre-check + build_defrag rework. ✓
- T5.9.B' (same-day fix-forward) — n_kv cap at construction + pool-exhaustion → GGML_STATUS_ALLOC_FAILED via `last_find_slot_fail_reason` enum on `llama_kv_cache`. ✓
- T5.9.C — `scripts/bench-ctx-feasibility.sh` + `profiles/qwen36-27b-x2-dflash-bigctx.sh` (sibling profile, host config at `/home/llm/profiles/`) + `data/t5.9-perf-gate-ledger.md`. ✓
- T5.9.D — seven binding gates ALL PASS. ✓
- T5.9.E — PHASE doc closure section + MEMORY append + this auto-memory. ✓

**Binding evidence at T5.9 closure:**

| Gate | Result | Notes |
|---|---|---|
| GP5.9.regression | PASS | 26.65 t/s mean (CV 0.18%, +0.76% vs T5.8 baseline 26.45) over 3 runs of `bash scripts/bench-t3.8-m3.sh` at 1455 MHz |
| GP5.9.NPC | PASS | `verify-production-determinism.sh NP_LIST="1 2 4 8" CTX_CHECKPOINTS=3` ACCEPTANCE PASS post-T5.9.B'; batch-shape invariance 4/4 |
| GP5.9.feasibility (HARD) | PASS | ctx 1M NP=8 `--kv-pool-blocks 24` oversubscribed 16 prompts: 2 × 200, 8 × **503 + Retry-After: 5**, /health 200. data/t5.9-admission6-20260523-142414 |
| GP5.9.DFlash | PASS | test-dflash-{np-invariance, np-multislot, closure} all GREEN |
| GP5.9.K-shift | PASS | test-paged-kshift-byte-identity + test-kv-shift-per-stream both (LAYER + GRAPH) on Qwen3.5-0.8B BF16 |
| GP5.9.defrag | PASS | test-paged-defrag-preserves-contents + test-kv-defrag-per-stream both |
| GP5.9.exhaustion | PASS | test-pool-bounds-respected + test-paged-backing-feasibility + bench-ctx-feasibility shell |
| GP5.9.spec | PASS | LLAMA_T5_TRACE_BUILD=1 llama-cli session → 129 NDJSON events; validator reports `OK: 129 events validated; all allocator invariants hold` — 5 invariants bind |

**Two T5.9.B' same-day gap surfaces (both fix-forward, no follow-up cover):**

1. **K/V view nbytes overflow.** `ggml_view_4d`'s assertion (ggml.c:5393) uses contiguous-stride bytes computed from `ne[]` alone — manual nb1/nb2/nb3 args don't reduce byte budget. Under user override the K/V view's `ne[1]=n_kv × ne[3]=n_seq_in_batch × bytes_per_pos` overshot the source's `BLOCK × n_head_per_dev × total_pool_blocks × bytes_per_pos`. **Fix:** cap `n_kv` at build_context construction to `min(kvps_or_size, BLOCK × total_pool_blocks / n_stream)`. At auto-size: no-op. At user override: per-slot effective ctx becomes `BLOCK × total_pool_blocks / n_stream`. Per-row K-bound (src[5] of FA per-slot-kv) still masks beyond active position.

2. **Pool exhaustion → wrong HTTP code.** find_slot's `return false` propagated as ret=1 → server's "Input prompt is too big" → 500. User-locked T5.9 spec was 503 + Retry-After. **Fix:** new `last_find_slot_fail_reason` enum on `llama_kv_cache` (NONE / KV_CACHE_FULL / POOL_EXHAUSTED); find_slot sets POOL_EXHAUSTED on admission failure; `llama_decode_internal` maps to `GGML_STATUS_ALLOC_FAILED` which the server's existing 503+Retry-After path consumes. Legacy KV_CACHE_FULL failures keep the historical 500 path.

**New deferral introduced at T5.9.B':**

State-save under user-override paged BACKING (ctx-checkpoints, cache-reuse, prompt-cache restore) — out-of-scope per user-locked T5.9 angles. The K/V-tensor-bytes reader assumes per-stream linear stride; user-override BACKING is block-major and smaller. Production AUTO mode unaffected (auto-sized buffer is byte-equivalent to T5.8). Bigctx sibling profile disables these features via `--ctx-checkpoints 0 --cache-ram 0`. A future T5.9.X or T6 iteration can rework the state-save reader through `block_table` indirection.

**Lessons (saveable):**
- `ggml_view_4d`'s nbytes assertion uses contiguous-stride bytes from `ne[]`, ignoring manual nb1/nb2/nb3 args. Caps must go on `ne`, not `nb`. → `[[feedback-ggml-view-4d-byte-budget]]` (worth its own feedback memory).
- Per-row K-bound (src[5] of `ggml_flash_attn_ext_per_slot_kv`) is the safety net under paged BACKING — it masks K positions ≥ bound[row] inside the K-loop. A tight `ne[1]` cap on the view is safe because positions beyond active are masked anyway.
- find_slot's bool-return-only semantic conflated two failure modes (cells-full vs paged-pool-exhausted). Adding a fail-reason enum was the surgical disambiguation.
- `cmake --target X` only rebuilds X's direct dependencies, not transitive dependents of changed PUBLIC headers (`include/llama.h`, `common/common.h`). Stale-build trap: always `cmake --build build -j 32` (no `--target` flag) after public-header changes.

**Operational constraints in force (carried verbatim):**
- No yarn-agentic nomenclature in code, scripts, tests, directories, branch names
- Production profile `qwen36-27b-x2-dflash.sh` UNCHANGED at T5.9 (auto-size, no `--kv-pool-blocks`)
- Bigctx sibling profile `qwen36-27b-x2-dflash-bigctx.sh` (host config at `/home/llm/profiles/`, untracked)
- Locked clocks 1455 MHz required for verify-production-determinism + perf benches
- coord/gpu BUSY/IDLE state machine before/after benchmarks
- Never run concurrent verify-production-determinism runs
- Never run concurrent inference benchmarks
- Don't write large artifacts to /tmp (tmpfs)
- All PLAN/PHASE/MEMORY edits commit+push immediately (CLAUDE.md §5, §6)
- Never skip hooks, force-push to main/master, or amend commits

Related: [[project-t5-9-b-landed-npc-pass]] (superseded; mid-implementation pickup), [[project-t5-9-paged-backing-reopen]] (the reopen decision; T5.9 closes it), [[project-t5-8-tier-5-closed]] (preserved audit record of T5.8 closure that was reopened), [[project-t5-probe-falsified-path-c-override]] (the user override that anchored T5.9 as forward-looking feasibility infra), [[feedback-no-followup]] (applied directly to drive both the T5.9 reopen and T5.9.B' fix-forward), [[feedback-no-host-concerns-in-code]] (--kv-pool-blocks name discipline; no phase nomenclature), [[feedback-oneshot-then-evaluate]] (T5.9.B coherent flip; gate sweep evaluated measured), [[feedback-bake-measurement-env-gates]] (no LLAMA_T5_9_* knob left in tree post-T5.9).
