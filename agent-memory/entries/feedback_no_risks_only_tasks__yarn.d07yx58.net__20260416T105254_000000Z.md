---
name: No risks section — every gap is a task
description: When planning, don't list gaps as "risks with mitigations/fallbacks" — each is a required task to implement
type: feedback
originSessionId: c1440a07-71f9-4a7a-a10b-10db7f6c5c11
---
Never put implementation gaps in a "Risks" section with fallback approaches. Each gap is a task that must be done. "Wave32 support", "fused kernel", "multi-blocksize" are not risks to mitigate — they are work items to implement.

**Why:** The user views these as required functionality, not optional optimizations. Labeling them as "risks" implies they might not get done, which is the wrong framing.

**How to apply:** When planning, if something is needed for completeness, make it a numbered implementation step. Reserve risk/mitigation language for genuinely uncertain outcomes (e.g., "PPL might not improve" — that's a measurement, not a task).
