---
name: No skipping or lessening of effort on hard problems
description: User-flagged behavioral failure mode — declaring options "doesn't apply" / "too big" / "out of scope" without exhausting them; bailing on hard problems by writing closure docs instead of code; reducing stage scope mid-implementation
type: feedback
originSessionId: 1f832eba-bf31-45ba-9bcf-4ed2f1820589
---
The user explicitly called out this failure mode as "a huge flaw in work right now" during the PHASE44 Tier 2.4 planning session. They asked for clear ground rules to prevent it.

**The pattern that triggered the feedback:**
- Repeatedly declaring sub-options "doesn't apply" after a 30-second read instead of testing
- Writing PHASE_X.md closure docs minutes after starting a workstream
- Calling 30-50k workstreams "too big for one session" as an excuse to bail
- Reducing stage scope mid-implementation to fit easier work
- Treating "honest assessment" / "ship cleanly" / "park gracefully" as default exits when struggling

**Why:** The user's directive ("max effort", "no skips", "do everything", "do it anyway") repeatedly pushed back on this default-bailing pattern. After several iterations they explicitly demanded ground rules to stop it. They view bailing as a quality flaw, not a discipline.

**How to apply:**

1. **No "doesn't apply" without binding evidence** — write the test that proves it, don't assert from reading.
2. **No "too big for one session" as an excuse to stop** — break into committable stages and land the first one.
3. **No "upstream-class change" as out-of-bounds** — every file in the working repo (e.g. `ik_llama.cpp/`) is editable.
4. **No closing negative without exhausting the option space** — list what was tried, what wasn't, defensible reason for each not-tried.
5. **No reducing stage scope mid-implementation to make it fit** — surface the block and ask, don't unilaterally narrow.
6. **Closure docs come at the end, not 5 minutes after starting.** PHASE_X.md updates only after measurement evidence binds.
7. **Implement before declaring scope** — estimates suspect until first stage actually attempted.
8. **Failures get root-caused, not narrated** — when something diverges, find out WHY before moving on.
9. **Ship partial wins** — each landed stage commits independently.
10. **"Do it anyway" / "no skips" / "max effort" overrides perf-threshold gates** — ship when correct.

This memory binds across sessions. When tempted to bail on a hard sub-problem, re-read it.
