---
name: project-t5-9-paged-backing-reopen
description: "production/2026-q2-next 2026-05-23 — T5 REOPENED same day as T5.8 closure. Paged BACKING is T5.9 subtask, not 'forward-looking deferral': it is load-bearing for the user override's high-ctx feasibility goal (without it, GP5.b is honestly only half-delivered — addressing live but KV buffer still contig-sized at llama_kv_cache_init, ctx 8M NP=8 OOMs). T5.1–T5.8 infra preserved as foundation. T5.9 spec stub: scope, 5 open decisions, mechanism sketch, 8 binding gates including GP5.9.feasibility (ctx 8M NP=8 allocates + runs)."
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

T5 reopen on `production/2026-q2-next` 2026-05-23, same day as T5.8 closure. Parent commit lands the PHASE doc edit (reopen note at top of T5 closure section + T5.9 stub) and the MEMORY correction entry. Tier 5 status: **CLOSED → REOPENED at T5.9**.

**Why reopened.** T5.8 closure section ended with three "forward-looking deferrals":
1. Paged BACKING — block-pool peak-concurrent buffer sizing
2. Kernel `block_table == nullptr` legacy branch removal
3. Graph-level defrag integration into `llama_kv_cache_defrag_internal`

User correction: #1 is **not** forward-looking work — it is the load-bearing piece of the user override that re-anchored Tier 5. The override (see `[[project_t5_probe_falsified_path_c_override]]`) re-scoped T5 as "forward-looking infra for ctx ≥ 1M workloads where contiguous can't allocate" after T5.0-probe falsified the original numeric-uplift premise. That goal is BACKING-bound, not ADDRESSING-bound. Without paged BACKING, GP5.b feasibility is honestly only half-delivered: addressing layer is end-to-end live, but the KV buffer is still sized `n_ctx_per_stream × n_stream × n_layer × n_head_kv × head_dim × Q4_0` at `llama_kv_cache_init`. Ctx 8M NP=8 OOMs at the buffer alloc site exactly as documented in `data/t5-probe-findings.md`. That is the gap the override scoped T5 to close; calling it "deferral" was misnamed.

Items #2 and #3 stay in the deferrals list — they are genuinely independent of T5's scope (dead-code cleanup in #2; trigger-path integration in #3 with `defrag_thold = -1.0f` default in production).

**Framing rule that came out of this** (added to `[[feedback_claudemd_no_followup_and_checkbox_semantics_provisional]]`-adjacent practice): audit-grade closure documentation (T5.C A/S/M/E/C) does NOT by itself satisfy CLAUDE.md §4. The framing test is *placement*: a deferral that, if it never lands, falsifies the step's stated goal is a **subtask in disguise**, not a deferral. Forward-looking list is for genuinely independent next steps. Apply this test to any future audit-grade closure section before declaring `[x]`.

**T5.9 scope (sketch from the PHASE doc stub):**
- Replace contig KV buffer allocation in `llama_kv_cache_init` with a block-pool sized to peak-concurrent blocks
- Paged ADDRESSING (`block_table` lookups) is unchanged — already operates on logical blocks
- Allocator's logical bid → physical-bid identity unchanged at NP=1, identity-at-block-table-init at NP>1
- T5.9 changes the *physical sizing*: `kv_size = total_pool_blocks × BLOCK_SIZE` instead of `n_ctx × n_stream`

**Open decisions to lock at T5.9 scoping (plan-mode):**
1. Sizing policy (static-at-init w/ CLI override vs dynamic high-water grow-on-demand)
2. Pool exhaustion semantics (admission rejection / eviction / error to client)
3. K-shift composition under reduced pool size (T5.7a paged K-shift already operates on blocks; re-verify)
4. Defrag composition under reduced pool (T5.7c allocator defrag is logical; likely no change but verify)
5. DFlash multi-slot per-slot scratch sizing composition

**Binding gates (T5.9 close):**
- GP5.9.feasibility (HARD): ctx 8M NP=8 allocates + decodes to finite t/s — the override's anchor gate
- GP5.9.regression: production NP=2 + Hadamard + DFlash byte-identical to T5.8
- GP5.9.NPC: verify-production-determinism NP={1,2,4,8} ACCEPTANCE PASS
- GP5.9.DFlash: test-dflash-{np-invariance, np-multislot, closure} GREEN
- GP5.9.K-shift: test-kv-shift-per-stream {LAYER, GRAPH} GREEN under reduced pool
- GP5.9.defrag: test-kv-defrag-per-stream {LAYER, GRAPH} + test-paged-defrag-preserves-contents GREEN
- GP5.9.exhaustion: new test — drive pool to exhaustion via concurrent prompts at high per-prompt ctx; assert clean admission rejection
- GP5.9.spec: new `paged_kv_pool_sizing.allium` + trace validator extension for `PoolBoundsRespected`

**How to apply.** When planning the next step on this branch: T5 is REOPENED, T5.9 is the only open binding subtask, T5.1–T5.8 infrastructure is preserved as-is. Recommend a plan-mode pass to lock the five open decisions before implementation. T5.9 implementation lands as its own bundle with the binding gates above; T5 closes again only when T5.9 is GREEN. Production profile is **not** touched by T5.9 (no profile edits) — production runs at NP=2 ctx 16K × 2 = 32K total, which doesn't exercise the BACKING constraint; the feasibility gate is at ctx 8M NP=8 which requires explicit bench harness config.

Related: [[project-t5-8-tier-5-closed]] (audit-grade record of what landed at T5.8 — closure-verb overridden by this reopen), [[project-t5-probe-falsified-path-c-override]] (user override that re-anchored Tier 5 as high-ctx feasibility infra), [[project-t5-7-bundle-b-complete]], [[project-t5-6-paged-write-read-end-to-end]], [[project-t5-bundle-a-closed]], [[project-t5-1-paged-allocator-landed-dormant]], [[feedback-no-followup]], [[feedback-no-workarounds]], [[feedback-claudemd-no-followup-and-checkbox-semantics-provisional]], [[feedback-bake-measurement-env-gates]].
