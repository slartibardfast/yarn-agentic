---
name: End-of-turn summaries: 1–2 sentences, not paragraphs
description: System prompt guidance is 1–2 sentence end-of-turn summaries; I habitually write 5–10. Tighten.
type: feedback
originSessionId: 8c1ed226-9685-4912-939b-e1d75d9255b0
---
After a unit of work lands, end the turn in one or two sentences — what changed and what's next. Not a bulleted recap. Not a paragraph per artifact. Not a list of commits.

**Why:** User surfaced this on 2026-04-24 during a self-audit. The Claude Code system prompt says explicitly "End-of-turn summary: one or two sentences. What changed and what's next. Nothing else." I have been writing 5-10 sentence summaries with bullet lists, effectively duplicating information the user already saw in the tool-call stream. That duplication is noise, not communication.

**How to apply:**
- After the last commit/edit, one or two sentences is the default.
- Long summaries are appropriate only when a turn's work wouldn't be legible from tool calls alone — e.g. multi-step verification where I need to explain which runs passed and which didn't, or a handoff where the next session needs the map. That should be the exception, not the habit.
- When the user is mid-back-and-forth on a topic, no summary at all is often right — just the next action and wait.
- When tempted to list every commit that landed, stop. The user saw the commits. A sentence naming the outcome is enough.
- Self-check before sending: "Would the user learn anything from this paragraph that the tool stream didn't already show?" If no, cut it.
