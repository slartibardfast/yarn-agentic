---
name: Anchor projections to measured baselines, not estimates
description: Throughput projections drift sharply if anchored to estimated baselines instead of measured ones; always compute uplift % against the measured number
type: feedback
originSessionId: 0890bce8-8661-4b49-9766-2fd975b4920c
---
When projecting throughput uplift for a proposed optimization, always anchor the comparison to a MEASURED baseline number from the same hardware/model/context configuration — not to an estimated number from cycle-time arithmetic.

**Why:** During Phase 41+42 plan-mode design, I introduced a senior-CS-prof critical review pass that adjusted projections downward to match realistic component costs. The result was a +1.6% uplift estimate that "didn't bind" the +10% gate. But the baseline I used (37.2 t/s) was an estimate from cycle-time arithmetic, not the actual measured Phase 39 K=1 d=1 number (32.58 t/s). Recomputing against the measured baseline gave +16% — comfortably above gate. The pessimism was an artifact of comparing-apples-to-oranges (estimated vs measured), not a real concern.

**How to apply:**
- For every throughput claim, identify the closest already-measured number from runlog data and use it as the denominator.
- If only estimated baselines exist, run a measurement first before projecting.
- When senior-review-style pessimism deflates a projection, double-check whether the deflation is real (component cost changes) or apparent (baseline shift). Apparent deflation is fixable; real deflation needs a redesign.
- Document the anchor clearly: "vs Phase X measured K=Y baseline of Z t/s" not "vs hypothetical baseline of Z t/s".
