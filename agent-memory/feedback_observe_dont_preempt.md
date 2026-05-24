---
name: When user is observing live data, don't push to roll back early
description: During production observation phases, user wants to watch the actual failure mode play out rather than be steered into preemptive rollback by projected trends
type: feedback
originSessionId: 9730b98b-a48a-46ed-a147-f48c8cb9810f
---
When the user has explicitly said "we'll observe with real data" and is actively driving traffic, do not proactively recommend or execute rollbacks based on extrapolated projections. Surface the data, flag concerning trends with concrete numbers, and stop there. The user will call the rollback when they want it.

**Why:** On 2026-05-05 during x2-mtp production observation, I recommended rollback at ~95 MiB/min growth based on a projected ~50 min OOM window. The user said "no" and "wait and see." They wanted to watch what actually happens, not what the trend predicts. Premature rollback wastes the observation window and the data we're collecting from real traffic.

**How to apply:** Report events tersely (one line of numbers + delta). Resist the urge to add "we should..." or "want me to roll back?" unless asked. If the situation becomes truly unsafe (probe fires, ABRT, immediate OOM), surface that — but a slow climb with safety margin remaining is data, not a crisis.
