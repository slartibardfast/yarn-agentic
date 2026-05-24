---
name: Test-first discipline — three facets
description: Tests are first-class; write them upfront, execute every planned one, never collapse untested candidates into negative claims
type: feedback
originSessionId: 9b52ee2b-d969-442c-83ba-98396525f6fa
---
Three reinforcing rules that govern how tests relate to the work, consolidated from earlier separate entries. They are facets of the same principle: tests are first-class, not optional, not skippable, not collapsed into conclusions.

## 1. No deferrals on feature ports — port the full feature, test-first

When porting a feature (e.g., MTP from one fork to another), do not defer advanced capabilities like chained rollout, speculative decode loops, or "nice to haves" as future work. Port the full feature, all in.

**Why:** Partial ports are wasted work; deferred items tend never to get done; the partial implementation isn't useful without the full pipeline.

**How to apply:** Write failing tests for every capability (including advanced ones) before implementing. This ensures nothing is forgotten and progress is measurable. Expand test coverage from lessons learned in prior debugging.

## 2. Don't skip planned tests even when diagnosis is clear

Execute every planned test step even when an earlier diagnostic already points at the root cause. Example surfaced during MTP-IR Vulkan precision debugging: the per-layer diagnostic clearly identified `mul_mat_vec` batch-shape dependency. I proposed skipping the op-isolation test and going directly to the shader fix. User pushed back: "let's write tests for the future. skip nothing please."

**Why:** Reusable test infrastructure compounds. An op-level batch-invariance test is valuable beyond the specific bug — it catches regressions, helps future shader work (Vulkan/Metal), and documents the precision expectation formally. Skipping to the fix trades long-term testability for short-term speed.

**How to apply:** Execute each planned test step even when confident about the root cause. Plan steps exist because they build infrastructure, not just to satisfy the immediate bug. If tempted to shortcut, ask the user first; default is to write the test.

## 3. Never claim "no fix exists" without testing each candidate

When summarising research results, never assert "no fix exists" (or "capped", "impossible", "fundamentally limited") without having enumerated and actually tested the candidate fixes. If candidates remain untested, the honest claim is "no fix in the tested set — these remain untried."

**Why:** Research can produce strong-sounding negative claims from a narrow test surface. The user pushed back on my "no V=1 fix exists" conclusion (Path C's writeup literally listed LDL preconditioning + per-tile scale search + tile-to-tile state carry as untested) — I had rolled Path C's shallow-knob findings into a broader claim than the evidence supported. That framing hides the biggest untried lever and biases the next decision.

**How to apply:**

- When writing up a negative result, include an explicit list of candidate fixes that were not tested and why. No test → no verdict on it.
- For each untested candidate, propose a test-first experiment design: hypothesis, pass/fail threshold set before running, failure action, compute budget. (Path E in the HARP work was the exemplar.)
- When pivoting ("we should abandon X"), the bar is higher than "the N things we tried didn't work" — the pivot is only justified after the untested candidates have been tested or explicitly scoped out with a reason.
- If the compute budget doesn't allow testing everything, say that plainly ("X is the most promising untested candidate; deferring because…") rather than collapsing "deferred" into "doesn't work".
- Re-read your own writeups for this trap. If a prior writeup listed untested pieces and you're now summarising it as "no fix exists", the summary is wrong.

**Red-flag phrases** to avoid in summaries: "no fix exists", "fundamentally capped", "impossible without X" (where X itself wasn't tested), "exhausted", "we've tried everything".
