---
name: Exhaust 0.8B work before moving to 35B-A3B
description: Workflow ordering — complete all feasible research on the 0.8B yardstick before spending 35B-A3B compute; do not interleave
type: feedback
originSessionId: c1440a07-71f9-4a7a-a10b-10db7f6c5c11
---
Do all the research that can be done on qwen35-0.8b first. Only move
to qwen35-35B-A3B after 0.8B work is exhausted. Do not interleave 0.8B
and 35B compute by default.

**Why:** User said "as discussed. we will do everything we can with
0.8B then move on" (2026-04-19), framing a deliberate two-phase
workflow: 0.8B is the cheap iteration regime, 35B-A3B is the gated
production evaluation. Mixing the two wastes 35B compute on questions
that 0.8B can answer, and fragments the 35B sweep schedule.

**How to apply:**
- Default to 0.8B for any new T1/T2/T3 cell unless there's a specific
  reason only 35B can answer it (e.g. cold-expert imatrix behaviour,
  MoE gate interaction).
- When a user says a line of work is "done on 0.8B", don't
  automatically propose a 35B follow-up — that's the next phase, not
  the closing bracket of this one.
- Parked items waiting for 35B-A3B (TURBO_2B evaluation, HARP_2B_S
  T4, HARP_2B T4, MoE per-expert lattice Path J) should stay parked
  as a batch until 0.8B research is declared exhausted by the user.
- Recognise the transition when the user explicitly moves the
  cursor — e.g. "let's start 35B work", "time to evaluate on 35B" —
  then unpark everything and sweep together to amortise model-load
  and imatrix-calibration cost.
- Existing memory [0.8B is signal phasing toward 35B-A3B] is the
  quality-measurement companion to this workflow rule: don't discard
  within-stderr 0.8B deltas because they're expected to flip on MoE
  weight-mass at scale. This rule governs *when* to run 35B; that
  one governs *how to read* 0.8B signal.
