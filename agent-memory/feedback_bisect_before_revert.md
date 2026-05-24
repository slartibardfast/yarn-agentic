---
name: Bisect on regression before reverting — confirm the diagnosis empirically
description: When a change causes a measured regression, bisect to identify WHICH component caused it BEFORE reverting any of it. Hypothesis-only diagnosis leaves a wrong record AND may revert correct work alongside the bug.
type: feedback
originSessionId: 9b52ee2b-d969-442c-83ba-98396525f6fa
---
When a change causes a measured regression, bisect the change to
identify the load-bearing source of the regression BEFORE reverting
any of it. Hypothesizing without bisection leaves a wrong diagnosis
on the record AND may revert correct work alongside the bug.

**Why:** T6.B regressed accept rate 2.256 → 1.667. I diagnosed
"batch-shape variance in target attention" without bisecting and
proposed reverting. User pushed back: "you must get to the bottom of
the issue." Bisect-1 (snap+restore alone, NO re-decode) showed
accept rate ALREADY at 1.909 — the snapshot/restore mechanism
itself was perturbing state, not solely the re-decode. The
"batch-shape variance" diagnosis was partially wrong: restoring
DeltaNet to "before verify" put state in a WORSE position than T5's
"after verify" (state-too-far-behind vs state-too-far-ahead).
Without bisection, that finding would have stayed hidden behind the
revert and the wrong diagnosis would have shaped subsequent design
decisions.

**How to apply:**

- When a measured regression appears on a multi-component change,
  the FIRST move is bisection — not revert, not hypothesizing.
  Hypotheses are starting points for bisection, not stopping points.

- Each bisection step is one targeted run: typically 5 min wall +
  ~5k tokens of conversation. Cheap on the budget.

- Bisection protocol for spec-decode-style cycles:
  - Disable component A only → measure
  - Disable component B only → measure
  - Compare deltas to isolate which component(s) carry the
    regression
  - For our T6.B case: would have been (1) snap+restore alone,
    (2) re-decode alone, (3) both — three runs.

- Reverting before bisection is two costs:
  1. Loses the regression evidence (the very thing that would have
     forced the right diagnosis).
  2. Potentially reverts correct work alongside the bug.

- Even if revert IS the right next step, do bisection first so the
  revert is informed: drop only what's load-bearing for the bug,
  keep the rest.

- This pairs naturally with `feedback_probe_before_implementing`:
  measure before building, AND bisect before reverting. Both are
  about empirical discipline over hypothesis-only conclusions.

**Pattern signal:** if you find yourself writing "the most likely
explanation is X" in a commit message or status update, that's the
moment to bisect rather than commit to X.
