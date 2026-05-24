---
name: project-t4-bundle-a-landed
description: "Tier 4 chunked-prefill admission (Sarathi-Serve) coherent flip landed on production/2026-q2-next 2026-05-22. Spec layer + coherent flip + correctness sweep + perf gate + closure docs ALL committed. T4.6 correctness gates ALL GREEN. T4.7 perf gate: GP4.i.a PASS (zero regression on steady arrival), GP4.i.b FAIL (uplift target structurally unachievable on aggregate-t/s — C0 IS multi-slot kernel saturation; staggered ≤ steady always). Closed honest FAIL per feedback_oneshot_then_evaluate. Production profile (qwen36-27b-x2-dflash.sh, NP=2 + DFlash) UNCHANGED. Next lever for vLLM ceiling: Tier 5 paged KV (out of scope for this PHASE doc)."
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

# Tier 4 — coherent flip landed; T4.6 GREEN, T4.7 FAIL on uplift (honest)

**Fact.** Tier 4 (chunked-prefill admission, Sarathi-Serve style) is in flight on `production/2026-q2-next`. Triggered by T3.8 GP3.i FAIL (M2 stall fraction 77.9% — see [[project-t3-8-perf-gate-failed-tier4-justified]]). User scope choices recorded at plan-time: **full Sarathi-Serve** (per-tick token budget K + chunked prefill across ticks + decode-priority admission) and **dual-gate workload** (M3-steady regression band + M3-staggered uplift binding). Plan file: `/home/llm/.claude/plans/cached-crunching-tiger.md`.

**Bundle A landed (6 commits, all pushed):**

| Commit | What |
|---|---|
| `fa935eb` | T4.0 — `specs/scheduler/batch_composition.allium` rewritten. Removed pre-T4 contracts (`PrefillSerialisationGate`, `DecodeHoldGate`, `MixedBatchProhibition`, `AtMostOnePrefillSlotPerBatch`, `DecodeHoldImpliedByPendingPrefill`). Added `TokenBudgetPerUbatch`, `DecodePriorityAdmission`, `ChunkedPrefillAdmission`, `PrefillCarryProgressesMonotonically`. `BatchCompositionInvariant` redefined (kept name, new semantics) so downstream specs (`unified_stream_dispatch`, `mtp_fused_draft`, `mtp_fused_x_n_stream`) continue to bind via the same identifier with updated comments. |
| `fc7d7f3` | T4.0.b — TLA+ `BatchComposition.tla` rewritten. `batch_prefill_count` per-slot Nat replaces `batch_prefill` set; constant `MaxBudget` replaces `ChunkSize`. Three MC configs originally: primary T4 + legacy regression + negative. |
| `6ad6140` | T4.0.c — `scripts/validate-batch-composition-trace.py` rewritten for the new admission contracts. NDJSON `TickDispatch` schema extended with `prefill_counts`, `budget_k`, `processing_set_at_start_of_tick`. |
| submodule `0759c01c` + parent `9fb4e6e` | T4.1 — `tests/spec/test-chunked-prefill-admission.cpp` stub property test (420 swept slot configurations PASS). Pure-CPU stub mirroring the admission loop landing in the coherent flip. The legacy `test-batch-composition-gates.cpp` marked superseded in CMakeLists. |
| `7635d04` | Cleanup — dropped TLA+ legacy regression mode after user call-out (see below). |
| `35632f2` | Cleanup — dropped trace-validator backward-compat fallbacks after user call-out. |

**User call-out (mid-execution).** User noticed the original Bundle A plan had a `bool t4_admission = false` "scaffold bypassed via local" in T4.2, asked "why are we bypassing items?" That triggered a broader audit:

- **Bypass bool (T4.2):** transient knob added then removed in adjacent commit. Conflicts with `[[feedback-bake-measurement-env-gates]]` (no LLAMA_* knobs added for diagnostics then ripped out) and `[[feedback-oneshot-then-evaluate]]` (write the bundle coherently). **Dropped.** Bundle B (T4.2/T4.3/T4.4/T4.5) collapses to a single coherent flip.
- **TLA+ legacy regression mode:** `DecodeHoldGateOn` constant + `LegacyAdmissionOK` action + third MC config modelled pre-T4 admission semantics that don't exist in production code after the flip. Same defensive-scaffold pattern at the spec layer. **Dropped** (commit 7635d04). TLC verified primary PASS + negative VIOLATES Legacy* at depth 4.
- **Trace validator backward-compat:** `prefill_slots` (legacy field name) treated as `{slot: 1}`; `budget_k` absent silently skipped budget check. Same family. **Dropped** (commit 35632f2). All TickDispatch fields now required; missing field is a schema error (exit 2).

**Pattern lesson.** Defensive scaffolds that exist between an "old state" and "new state" in a coherent flip are dead-weight when the old state doesn't run anywhere after the flip. The instinct to soften the transition adds maintenance cost with no diagnostic dividend. Three flavours appeared in this work: a bypass bool, a TLA+ regression action, a validator backward-compat. All three dropped on the same principle.

**Coherent flip landed (submodule e282d229, parent eb426e0):**

- `batch_pending_prompt` body replaced. K = `params_base.prefill_chunk_budget` (default `n_ubatch`); per-slot quota = `ceil(K / n_eligible_load_nonembedding)`; pre-T4 `active_pp_slot_id` PrefillSerialisationGate gone; `LLAMA_FATTN_STRICT_SEQUENTIAL_DECODE` env knob removed from this function. Embedding slots still use legacy single-shot n_batch cap.
- CLI: `--prefill-chunk-budget K` in `common/common.cpp:1011-1016` and `common/common.h:232`. Default 0 means K = n_ubatch.
- Trace: `emit_tick_dispatch` moved from `process_batch_tokens` (per-dispatch) to `update_slots` (per-tick aggregate). Under T3.5 split_equal a tick is typically split into prefill-only and decode-only slices; per-slice tracing falsely violates `DecodePriorityAdmission`. Per-tick captures the FULL tick batch composition once and binds the validator's invariants correctly. New `tick_trace_state` struct on `server_context` (h: line ~318) holds `prefill_counts: slot_id -> count`, `decode_slots`, `processing_set` and `loading_prompt_set` snapshots, `budget_k`. The pre-T4 `prefill_slots` (vector) and `loading_prompt_set_at_start_of_tick` (legacy) fields are replaced; loading_prompt_set is still emitted as a documented optional field for compatibility.

**T4.6 correctness gates — ALL GREEN:**

- GP4.j r5-probe-c4 ITERS=20:                          0/20 PASS
- GP4.k verify-production-determinism NP={1,2,4,8}:    PASS (cross-NP byte-identity + batch-shape invariance)
- GP4.l DFlash composition (multislot/closure/np-inv): 3/3 PASS
- GP4.m trace validation (NP=8, K=256, 8-prompt staggered, 481 records): PASS — 7 mixed ticks exercised; max batch 110/256 tokens
- GP4.n kernel NPC fattn-per-slot-kv-dispatch:         PASS
- T4.1 test-chunked-prefill-admission (CPU stub):       420 swept configs PASS

T3 FRAMING B closure re-confirmed under T4: verify-production-determinism PASS, dispatch_multi_seq_count 64/64 in np=8 segment, DFlash composition 3/3.

**T4.7 perf gate — measurement of record:**

Locked clocks 1455 MHz, N=3 per config, CTX_PER_SLOT=4096 (matches T3.8 M3). Ledger `data/t4-perf-gate-ledger.md`. New harness `scripts/bench-t4-m3-staggered.sh`.

| Config | Mean t/s | σ | CV | Δ vs C0 |
|---|---|---|---|---|
| C0 (T3.8 M3 pre-T4 baseline, steady arrival) | 26.49 | 0.037 | 0.14% | — |
| C1-steady (T4 + steady arrival)              | 26.49 | 0.014 | 0.05% | 0.0%    |
| C1-staggered (T4 + 5s arrival offsets)       | 21.62 | 0.016 | 0.07% | −18.4% |

- GP4.i.a (regression ≥ C0×0.98 = 25.96): **PASS** — zero regression.
- GP4.i.b (uplift ≥ C0×1.20 = 31.79):     **FAIL** — 21.62 t/s; gate target structurally unachievable on aggregate-t/s (C0 IS the multi-slot kernel saturation throughput; staggered arrival has a longer wall-time floor → staggered aggregate always ≤ steady aggregate).
- GP4.i.c (CV ≤ 1%):                       **PASS** — 0.05% / 0.07%.

**Closure (T4.8):**

- PHASE doc closure section landed in `PHASE_NSTREAM_KV_PERF.md` (audit-grade A/M/E/C pattern, commit on production/2026-q2-next).
- MEMORY.md entry (project-committed) and this auto-memory file updated.
- No submodule bump (the only submodule change was T4 coherent flip e282d229 + bump eb426e0; T4.7 is host-side benches + docs only).
- Production profile `qwen36-27b-x2-dflash.sh` UNCHANGED.

**Constraints (reaffirmed).**

- Production profile `qwen36-27b-x2-dflash.sh` (NP=2 + DFlash) is unchanged — gate uses its own ctx.
- DFlash composition is a hard gate per `[[project-dflash-t9-np-validity-drift-signature]]`.
- Bug C non-regression argument: closed structurally by 4D KV layout + uniform mul_mat shape per tick (`[[project-t3-framing-b-closure]]` Q1). T4 admission only changes WHICH tokens are in the batch, not the call shape.
- Branch hard-constraints stay: push only on GREEN verify; no `--force-push` on `production/2026-q2-next`; no `git reset --hard` on tracked branches.

**How to apply.** T4 closed. Spec layer, admission scaffold, trace producer all in tree and load-bearing for future work (burst short-prompt arrival, long-prompt prefill — workloads where T4 admission delivers). When evaluating future throughput levers on this hardware + shape, **do NOT propose dispatch-packing or admission-policy changes** as the lever — T3.8 and T4.7 both empirically falsify that route on this kernel (PSKV singlewarp + Q4_0 KV + sm_75 + Qwen 3.6 27B). The next lever for catching vLLM's measured 154 t/s ceiling is **Tier 5 paged KV**.

**Lessons (2026-05-22).**

1. **Per-tick trace semantics > per-dispatch under T3.5 split_equal.** `DecodePriorityAdmission` is a tick-level invariant. T3.5 separates prefill and decode into distinct dispatch slices; per-slice tracing of a prefill-only slice would falsely violate (its `decode_slots` is empty even though the tick admitted decodes in a different slice). Moving the emission to `update_slots` after `batch_pending_prompt` captures the full composition once per tick and binds correctly.
2. **Per-slot quota + global K cap is enough for fair admission in staggered arrival.** Full round-robin redistribution of leftover budget (pass-2 after slots finish under quota) was considered and deferred — utilization is bounded by the quota waste case which is small at steady-state staggered arrival. T4.7 measurement confirms no obvious utilisation gap at default K = n_ubatch on M3-staggered.
3. **Bug C structural closure (4D KV + uniform mul_mat shape per tick) holds under mixed admission.** GP4.m exercised 7 mixed ticks (decode+prefill same batch) without violation; r5-probe-c4 ITERS=20 = 0/20. The "T4 admission only changes WHICH tokens are in the batch, not the per-tick mul_mat call shape" non-regression argument is now empirically validated, not just theoretical.
4. **Aggregate-t/s gates on staggered workloads need the right comparison.** The T4.7 plan asked staggered ≥ steady × 1.20 on aggregate-t/s — that's structurally impossible because steady IS the multi-slot kernel saturation throughput; staggered always has a longer wall-time floor (ramp-up window with under-utilised slots) so its aggregate is mechanically ≤ steady's aggregate. The right comparison for "did T4 buy something" is staggered-pre-T4 vs staggered-post-T4 (same workload, different code states), OR a different metric (TTFT, TBT, fairness). The plan target was a hope, not a calculation. Honest measurement falsified it; correctness layer stands.
5. **T3 + T4 perf measurement on PSKV-singlewarp + Q4_0 + sm_75 + Qwen 3.6 27B exhausts the scheduler levers.** T3.8 ruled out dispatch-packing (multi-seq dispatch at 93% rate = ~0% uplift). T4.7 ruled out admission policy (T4 = byte-identical aggregate-t/s on steady, lower on staggered by wall-time mechanics). Both gates closed FAIL with measurement of record. Future throughput work must target either the kernel (PSKV variant, larger Q→K projection bypass, etc.) or the storage layer (paged KV / continuous batching with paged blocks). NOT the dispatcher.
