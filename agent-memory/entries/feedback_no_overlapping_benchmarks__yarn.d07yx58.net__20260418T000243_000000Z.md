---
name: Never overlap benchmarks
description: Never run concurrent inference or benchmark processes on shared GPUs — results are unreliable and waste time
type: feedback
originSessionId: e06c78f3-43de-46c4-b7cf-81ea2dbd7d8f
---
Never run overlapping benchmarks or inference processes on the same GPU(s). Run them sequentially, one at a time.

**Why:** Concurrent GPU processes contend for VRAM bandwidth, compute units, and dispatch queues. Timing results are meaningless. The user flagged this as a mistake.

**How to apply:** Before launching any llama-cli, llama-bench, or test binary, confirm no other inference process is running (`ps aux | grep llama`). Never use `run_in_background` for a second benchmark while the first is still running.

**Stronger rule — ALWAYS pause before benchmarking:** This host regularly runs multiple streams of work that aren't visible from within our shell session (other users, scheduled jobs, other agentic sessions). Before executing ANY benchmark (not just overlapping ones), explicitly pause and ask the user for confirmation that the host is idle. A `ps` check isn't sufficient — workloads may be idle-at-inspection-moment but about to resume. Write benchmark scripts freely; gate their EXECUTION behind user go-ahead.
