---
name: override-locked-policy-pragmatically
description: "When a locked gate threshold blocks merge but the trade-off is well-understood and the structural correctness goal is delivered, the user may choose to override the locked policy. Don't ask \"should we hold?\" first — present all options including the override."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

When a locked perf-fail policy ("stop-and-surface, never silently
merge") is hit, present the override as a first-class option
alongside the hold-and-fix options. Don't reflexively assume the
user wants to hold — when correctness goals are met and the perf
trade-off is well-characterised, the user may pragmatically choose
to ship.

**Why:** 2026-05-20, PHASE_NSTREAM_KV_4D G3.h gate. TG NP=8 missed
±1% bound by -6.2%. Root cause: graph reuse disabled at n_stream>1
forces rebuild per single-token sub-batch. All 6 correctness gates
(G3.a–G3.f) green; Bug C structurally closed; decode-side prefill
gate removed. User was offered four options and selected
"Override locked policy — merge with -6.2% regression". The
locked policy was the right *default* but not the right *outcome*
once the trade-off was understood — and the user is the one who
gets to decide which way to resolve that.

**How to apply:** When a locked-policy gate fails:

1. Diagnose the root cause to the point you can characterise the
   trade-off precisely (the 5% delta has a 5-line explanation, not
   a hand-wave).
2. Distinguish "current-step gap" from "legitimate future work" per
   CLAUDE.md §4 "No follow-up cover". A regression with a known
   contained cause + the structural goal of the step being
   delivered = future-optimisation candidate, not a current-step
   gap.
3. Present the override as an explicit option with cost-explicit
   wording ("merge with -X% regression"). Pair it with the
   hold-and-fix paths so the user can pick.
4. If the user picks override, document the reasoning verbatim
   from their selected option in the phase doc + commit message,
   so the trade-off is recorded honestly rather than glossed.

This is NOT a licence to silently degrade gate rigour. The override
path requires (a) all correctness gates green, (b) regression cause
diagnosed and bounded, (c) explicit user selection — not implicit
"let it ride".

See [[feedback_no_followup_cover]] (the no-cover-language rule) and
[[feedback_surface_tradeoff_decisions]] (the surface-and-let-user-
decide rule). This memory composes them: surface the trade-off,
make override one of the options, let the user decide; if they
override, the gap is real future work, not cover language.
