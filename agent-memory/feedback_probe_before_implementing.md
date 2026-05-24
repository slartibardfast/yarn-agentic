---
name: Probe before implementing — gate optimization work on cheap measurement
description: When an optimization's value depends on an empirical question (here α(top-K) ceiling), measure first via existing probe infrastructure; close on data alone if marginal
type: feedback
originSessionId: 0890bce8-8661-4b49-9766-2fd975b4920c
---
When an optimization plan asserts a throughput projection that depends on an empirical statistic (acceptance rate, hit rate, branch prediction quality), STOP and measure the statistic via cheap probe infrastructure BEFORE implementing the optimization. Close the work on probe data alone if the measured statistic falls below the build threshold.

**Concrete case (Phase 40, 2026-05-07):** Top-K=2 tree drafting projected +14-19% based on assumed α(top-2) - α(top-1) ≈ 0.15. The codebase already had `LLAMA_PROBE_TOP2` infrastructure (commit edc1f6a3) that measures this empirically without writing any tree-drafting code. Probe ran ~10 minutes, returned Δ=0.06 at production-relevant long context. +3.2% projected lift was below the +5% build threshold. Phase 40 closed negative on probe data alone — saved ~50-70k tokens of tree-drafting implementation that would have ended at the same negative conclusion.

**Why:** "Negative results land cheap when honest, expensive when rationalized" (CLAUDE.md §8). Implementing first then measuring multiplies the cost of negative results by the implementation effort, AND adds the cost of the abandon-the-work decision. Probe-first means the close is on hard empirical data, not on "we built it, it didn't help."

**How to apply:**
- When designing optimization phases, explicitly identify the empirical statistic that determines whether the optimization wins on this hardware/model.
- Look for existing probe infrastructure that measures the statistic without committing to the implementation. Search the codebase for env-gated probes (`LLAMA_PROBE_*`, `LLAMA_PROFILE_*`, etc.) before designing new ones.
- Set a quantitative gate (e.g., "build if Δ ≥ X, close if < Y") in the design doc BEFORE running the probe — prevents post-hoc rationalization either way.
- If no probe exists, write a minimal one as the first phase step, not embedded in the implementation.
- Run the probe at the production-relevant configuration (long context, real prompt distribution) — short-context numbers can mislead.
