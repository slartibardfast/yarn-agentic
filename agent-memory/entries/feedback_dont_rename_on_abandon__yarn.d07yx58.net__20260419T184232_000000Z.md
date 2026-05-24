---
name: Don't rename a feature on the way out — abandoning is a status change, not a rename
description: When a line of work is being abandoned, don't rename it to "fix" misleading naming — that implies ongoing maintenance under a new name. Abandoning = one-line status change in the retrospective.
type: feedback
originSessionId: c1440a07-71f9-4a7a-a10b-10db7f6c5c11
---
When abandoning a ftype / feature / module, do not rename it on the
way out — not even to "fix" misleading naming. The rename implies the
thing is still being maintained under a different name; the correct
signal is "dead."

**Why:** 2026-04-19, I renamed `HARP_2B_S` → `UD_IQ2_S_QWEN35` after
we realized the recipe was just IQ2_S + Unsloth Dynamic 2.0 + Qwen3.5
SSM carveouts. The rationale was "honest naming." Within an hour we
ran the Unsloth comparison, saw strict dominance, and abandoned the
whole 2-bit lineage. The rename added commit noise, implied ongoing
work, and had to be reverted.

User feedback: "renaming it was a mistake." Correct.

**How to apply:**
- When a feature is being abandoned, update its status in the
  retrospective / MEMORY entry to "Abandoned" with a one-line
  explanation. Do not touch code to rename the enum / function /
  file.
- If the old name is misleading, say so in the retrospective. A
  future reader who looks up the code by the old name will find it
  and the retrospective together; that's the correct coupling.
- Especially: do not propose "let's rename it to X, then decide
  whether to abandon." That's two decisions bundled; the rename
  locks in a position (still maintaining) that we haven't made.
  Decide to abandon OR keep first, rename only if keeping.
- If the rename already happened and abandonment is the right call,
  revert the rename in the same branch. Net effect: zero code change,
  cleaner history.
