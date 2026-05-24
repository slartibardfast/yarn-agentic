---
name: Composition analysis before scope on port / replace work
description: When proposing to port external work or replace architecture, map composition (transfers / reshapes / drops) against existing stack BEFORE proposing scope and schedule
type: feedback
originSessionId: e17bdbd7-e0a3-4ec7-9818-e8f46eed5283
---
When proposing to port external work (upstream fork, paper algorithm) or replace existing architecture, map composition first: which existing components transfer directly, which reshape, which drop. Surface this analysis BEFORE proposing scope or schedule.

**Why:** During Phase 39 framing I proposed "port upstream tree drafting" with a token-budget schedule. The user pushed back: "what's the synergy with existing work?" That question forced explicit mapping — Phase 36 inline-KV-hook composes, Phase 38 B persist buffer reshapes, Phase 38 C extended-chain drops, Phase 38 E async APIs become moot. The analysis CHANGED the framing entirely: from "port wholesale" (overscoped, ignores reuse) to "collapse + replace + obsolete most of 36-38" (smaller scope, larger architectural simplification). Without the composition step, the schedule would have been wrong in both directions — too small (missed cleanup of obsolete pieces) and too large (proposed reimplementing what already transfers).

**How to apply:**
- For any port / replace proposal, produce a 3-column table BEFORE schedule: "Existing component | Disposition (transfers / reshapes / drops) | Why".
- Surface this table in the user-facing response when proposing the work, not buried in a doc.
- Drops are as important as transfers — they tell you what cleanup is in scope.
- Reshapes carry the most risk — flag specific shape/semantic mismatches in the table.
- The composition table feeds the schedule's "obsoletes" or "cleanup" phase. If your schedule has no cleanup phase but the composition has many drops, the schedule is wrong.
- Anti-pattern: "port upstream's X" without naming what it replaces locally. Name it.
