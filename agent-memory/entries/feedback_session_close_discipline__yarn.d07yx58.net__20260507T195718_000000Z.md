---
name: Close cleanly at ~700k+ rather than push through compaction
description: When context budget hits ~700k of 1M and the next chunk is a major pivot, close the session and write a complete pickup brief instead of starting the pivot
type: feedback
originSessionId: e17bdbd7-e0a3-4ec7-9818-e8f46eed5283
---
When the context window is ~700k+ of 1M and the next planned work is a substantial architectural pivot (multi-step refactor, port, reshape), STOP and execute a clean close: complete PHASE doc with primary-source refs, memory entry, cleanup commit, smoke test at parity. Don't start the pivot in the same session.

**Why:** Compaction at ~80%+ context degrades fidelity — exactly when implementing complex architectural mappings (file/line references, tensor shapes, attention masks) that need precision. Starting a 270–400k pivot at 720k means hitting compaction mid-implementation, where the cost of misremembering shapes or paths is high. The 30–50k tokens spent on a thorough close are recovered many-fold by the next session starting cold-cache with a complete brief vs. starting at 700k carrying half-state. CLAUDE.md §8 says this generally; this entry names the specific decision point.

**How to apply:**
- At ~700k of 1M, if the next chunk is non-trivial (>100k estimated), pivot to close mode.
- Close-mode deliverables: PHASEx.md with primary-source file/line refs, architectural comparison table, schedule with token budgets, binding closure criteria, anti-goals, risks, pickup brief; MEMORY entry; cleanup commit (revert in-flight half-state, gate diag knobs OFF); smoke test at parity GREEN; push.
- The pickup brief must be self-contained: a fresh session reading PHASEx.md alone should know what to do.
- "1M context bet" is the framing — commit the next session to a single bounded scope rather than spilling it across two windows. Cleaner narrative, less re-derivation.
- Anti-pattern: "let me just start the pivot" at 720k. Don't. The close is the work that makes the next session efficient.
