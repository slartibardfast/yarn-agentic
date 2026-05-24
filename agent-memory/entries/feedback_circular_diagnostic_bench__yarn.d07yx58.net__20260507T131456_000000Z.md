---
name: Don't bench a partial implementation to decide whether to build a better one
description: Diagnostic benchmarks of an incomplete impl, used to gate building a fused/improved version, are circular and produce noise. Reach for primary-source evidence instead.
type: feedback
originSessionId: 9730b98b-a48a-46ed-a147-f48c8cb9810f
---
A diagnostic benchmark whose purpose is to classify the *current* (partial, in-progress) implementation in order to decide whether to build a *new* (better) implementation is circular. The current impl's quirks — flag mismatches, partial wiring, untuned defaults — confound the measurement, and the measurement is then used to authorise (or veto) the next build, when the next build is precisely what would fix those quirks.

Concrete failure: PHASE 36 Phase A' was scoped to bench our existing 27B MTP across `--draft N` configs and classify us as ceiling-bound / undertuned / mistuned to gate the fused-graph implementation slice (B.0). The harness was misflagged for our production GGUF; first run produced 0% accept and 11 t/s. I reported it as a "critical finding — MTP-head mistuned" instead of "the harness is broken."

**Why:** the actual evidence for whether per-step kernel-launch + D2H-sync overhead is the bottleneck (the thing the fused graph removes) was already in hand from `project_mtp_iter7_post_mortem.md` — 1.282× ceiling on 0.8B due to per-verify D2H sync. That diagnosis carries forward; the partial-impl bench was never going to add to it.

**How to apply:** when a planned diagnostic benchmark feeds a "should we build X" decision, ask first: does this measurement depend on the implementation we're about to replace? If yes, the bench is circular — replace it with a primary-source diagnosis (existing nsys traces, kernel-launch attribution, queueing-theory bound) or skip the diagnostic outright. Wall-clock sweeps are appropriate for *tuning* a working implementation, not for *deciding whether to build* one.

Tell on the harness too: 0%-accept or other "this can't be right" results are signals to verify the harness, not to propose plan-revising findings.

**Not the same as: real measurements against a fixed binding gate.** A test-first harness with a documented threshold (e.g. `tests/mtp-fused/gate.yaml`'s `accept_d3_ratio ≥ 0.97`) run against successive real fixes is *not* circular. The harness IS the binding test; each measurement is data; the next change is decided by the schedule, not by the measurement. The anti-pattern is using a *partial implementation's* metrics to decide whether to *replace* that implementation. Measuring real fixes against a fixed gate sits at the test-first / no-follow-up-cover / checkbox-semantics intersection, not this rule. PHASE37's schedule (each item: change → harness `--fast` → record → next) is the legitimate case; it does not violate this rule.
