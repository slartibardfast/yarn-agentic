---
name: feedback-phase-step-may-be-an-experiment
description: "User-flagged behavior — when the user insists on doing a step I've framed as \"optional / completeness\", treat the result as the experimental answer they're after, not as compliance. Surface the resulting comparison before they ask."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 9b52ee2b-d969-442c-83ba-98396525f6fa
---

When a plan-step looks "optional" from the closure-binding perspective and the user insists on doing it anyway, the step is probably answering an open experimental question, not adding completeness coverage.

Concrete case (2026-05-15, Phase B of PHASE_MMQ_Q4_0_AR16):

- I framed Phase B (Q4_0_AR16 MMVQ kernel) as optional because the production env stack pins `LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH=1`, which forces MMQ for all batch sizes; MMVQ would never fire on the production path. Strict closure-binding logic said "skip".
- User: "We will not skip it. there is not point in skipping it."
- After Phase B closed: the MMVQ row-0 was BYTE-IDENTICAL to the MMQ row-0 (`+0.865397 +1.219897 +0.636013 ...` matched bitwise). I described this as a "bonus finding".
- User: "This bonus finding is why i insisted we do Phase B. it was my question."

The user was running an experiment: *do MMQ and MMVQ paths produce identical results on AR16?* That question only gets answered by building Phase B. Framing it as a perf-completeness optional missed the point entirely.

**Why this matters:**

1. The bonus finding has cross-NP determinism implications far beyond MMVQ-vs-MMQ. If a request batch could cross the MMVQ↔MMQ boundary depending on batching (which happens at default dispatch transitions), byte-identity only holds if both paths agree. Two independent code paths converging bitwise is a stronger correctness witness than either matching a scalar reference.
2. The user thinks of plan steps as **experiments**, not as deliverables. The output of the step is the *answer*, not the *artifact*.
3. When I dismiss a step as "lower priority on the critical path", I may be killing the experiment without realizing it.

**How to apply:**

- When I'm tempted to skip a plan step on critical-path grounds, ASK what the step is supposed to reveal — what question does its closure answer? "Optional for closure" is a property of the binding, not of the step's information value.
- When a step produces an unexpected result that compares two paths / configurations / numbers, FRAME the comparison explicitly before declaring done. Don't bury it as "bonus." If the user designed the step to evoke this comparison, they want to see it called out.
- When the user pushes back on a "skip" recommendation with high conviction, the right response is "let me check what closing it tells us first" — not re-litigating the priority.

Related: [[feedback_probe_before_implementing]], [[feedback_dont_anchor_on_current_production]], [[feedback_no_skipping_lessening]].
