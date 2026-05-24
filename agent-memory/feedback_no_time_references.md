---
name: No time references in plans
description: Strip week/day/month/date references from plans, PHASEx.md, and work breakdowns — they demotivate both human and AI
type: feedback
originSessionId: c1440a07-71f9-4a7a-a10b-10db7f6c5c11
---
Never use "Week 1", "Day 5", "2-3 week effort", dated progress headers, or any other time-calibrated language in plan files, PHASEx.md documents, or work breakdowns.

**Why:** User explicitly called them "demotivators for man or machine" — time estimates age badly and create pressure that distorts decision-making. What matters is milestone gates (correctness / quality / speed), decision criteria, and dependency ordering — not calendar time.

**How to apply:**
- Use milestone names: "Correctness milestone", "Quality milestone", etc.
- Order scope items numerically within a milestone; do not label them by day.
- Strip any "as of YYYY-MM-DD" progress headers; track progress by what's landed vs what remains, not dates.
- If the user asks "how long will this take", answer in terms of decision gates and remaining work, not weeks.
- Exception: dates in git commit messages, changelogs, or external references (arXiv IDs, paper years) stay as-is — those are time-stamped facts, not estimates.
