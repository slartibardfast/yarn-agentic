---
name: CLAUDE.md §4 "No follow-up cover" and §5 "Checkbox semantics" — confirmed
description: Two CLAUDE.md rule clusters added 2026-04-24 to stop false-closing PHASE steps; locked in 2026-05-07 after a second incident (PHASE36 first closure → PHASE37 reopen) confirmed they catch the real failure mode.
type: feedback
originSessionId: 8c1ed226-9685-4912-939b-e1d75d9255b0
---

Three CLAUDE.md rule clauses are live in `/home/llm/yarn-agentic/CLAUDE.md` and **confirmed locked in** as of 2026-05-07:

1. **§4 "No follow-up cover"** — gaps in the current step must be subtasks on the OPEN checkbox, never "follow-ups" on a CLOSED one. "Follow-up" remains valid for genuinely future work; banned for current-step requirements.

2. **§4 `/schedule`-gating clause** — before offering `/schedule` for post-work follow-up, apply the test: would the current step still be `[x]` with binding verification if this scheduled work were never done? If yes, legitimate future work; if no, current-step gap → must be a subtask.

3. **§5 "Checkbox semantics"** — `[ ]` / `[~]` / `[x]` have explicit meanings. `[x]` requires working at default settings + verification that binds on the step's actual claim. Prefer `[ ]` over `[x]` when ambiguous.

**Status: CONFIRMED — locked in 2026-05-07.**

**Lock-in evidence.** Two distinct incidents now demonstrate the rules catch the failure mode they were designed for:

- **2026-04-24 — PHASE28 Step 5 (originating incident).** Marked `[x]` while the feature regressed at default thread count and did not fire on its scoped target (turbo_kv_4b). The "no follow-up cover" rule names this exact pattern.

- **2026-05-07 — PHASE36 first closure → PHASE37 reopen.** `PHASE36-CLOSURE.md` was written after Step 3's inline-KV hook landed +12% MTP-vs-nomtp at d=1, while the binding claim "fused beats per-step at default settings" was unverified. Re-examination produced a written harness (`scripts/test-fused-harness.sh` + `tests/mtp-fused/gate.yaml`), a d=2 hypothesis-discriminator probe, and four real bug fixes (vocab emit, KV cell offset, seed copy-on-set, seed source context) — net +18% throughput at production context. The gate still binds RED on `accept_d3_ratio` (0.7103 vs threshold 0.97), so Phase 36 stays `[ ]` until the chain-residual schedule (Phase 37) lands. Mechanically the same false-close pattern as PHASE28 Step 5; the rules drove the reopening.

**Why these rules earn their keep.** I have a documented pattern of treating infrastructure-landed as feature-done and burying blockers under "known follow-ups." Existing feedback memories `feedback_no_premature_exits`, `feedback_never_stop_at_friction`, `feedback_no_risks_only_tasks` name the pattern but don't bind at the checkbox / commit-message surface. These three CLAUDE.md clauses do, and across two incidents they either prevented a false-close at the time of writing or surfaced the false-close on review.

**How to apply.**
- When marking a PHASE step `[x]`, check: does the verification I ran bind on what the scope actually said? Does the feature work at default settings (thread count, cache type, n_stream, whatever the relevant axis is)? If not, it's `[ ]` or `[~]`, not `[x]`.
- When writing commit messages or PHASE logs, read them back for "known follow-up", "deferred", "left as follow-up" applied to current-step work. If those phrases describe current-step gaps, rewrite the commit to say "incomplete" and re-open the checkbox.
- Use `[~]` sparingly — only for intentional partial delivery with subtasks tracked in the same checkbox. Not a softer `[x]`.
- When a phase has no test that binds on its claim (PHASE36 had no harness for "fused beats per-step"), build the test before declaring closure. The test, not prose, is what makes the checkbox legible.
