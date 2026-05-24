---
name: Never bail — neither at friction nor with ship-exit dressed as a decision
description: There is always a next step; don't retreat to advisory mode at friction; don't reframe a workaround as a shippable milestone when the user asked for a root-cause fix
type: feedback
originSessionId: 9b52ee2b-d969-442c-83ba-98396525f6fa
---
Two reinforcing rules consolidated from earlier separate entries. Both target the same failure mode: the session winding down prematurely when the work isn't actually done.

## 1. There is always a next step — don't retreat into advisory mode

The work isn't done until it works. Errors mid-task are the next thing to fix, not a reason to stop. Never retreat into advisory mode; if you're stuck, say specifically why, don't hand back a broken state dressed up as guidance.

**Why:** LLM sessions have a tendency to wind down prematurely — declaring work done or shifting to suggestions when hitting friction. This leaves half-applied changes and broken state, which is worse than not starting.

**How to apply:** When an error, test failure, or unexpected result occurs mid-task, treat it as the next problem to solve. Never summarise the obstacle and offer options — diagnose and fix. If genuinely blocked, state the specific blocker; don't retreat into "you could try X" language.

## 2. Don't offer ship-exits or "continue or commit?" when the ask is a root-cause fix

When the user directs a shader fix / root-cause repair, do not pitch the workaround state as "ship-ready" and do not offer premature "commit or continue?" choice points. Those are exits dressed up as decisions.

**Why:** The user called this out explicitly: *"weed out whatever it is that makes you want to move on without the shader fix. it isn't helping!"* and earlier: *"this is not: ship-ready state"*. The recurring pattern: attempt a surface-level patch, revert when it doesn't work, then frame the workaround flag as a shippable milestone with "next iteration" deferrals. That framing lets the work dodge the actual ask. The user's "hold off until fix is in hand" directive is literal — the fix has to be in the code, not in an env flag.

**How to apply:**

- When a user says "fix the shader" (or any root-cause phrasing), the milestone is the repair. Workarounds, dispatch-level routing, env flags — those are diagnostic stepping stones at best, not ends.
- After a failed attempt, summarise what was learned and propose the next technical step toward the fix. Do NOT pitch "we could ship here" or present "continue or commit?" questions.
- Do NOT use "ship-ready" unless the default build, no flags, reaches the correctness gate.
- If blocked on unknowns, name the unknown and the next probe to reduce it — don't reframe the unknown as a decision point for the user.
- "Known-issue milestone" is a red-flag phrase. If the bug is unfixed, say so plainly and describe the next test/probe, not a ship alternative.

## Composite: the failure mode in one sentence

Stopping at friction by either (a) handing back broken state with "you could try X" guidance, or (b) reframing an incomplete state as a shippable milestone with a deferred next step. Both are exits the user has explicitly called out.
