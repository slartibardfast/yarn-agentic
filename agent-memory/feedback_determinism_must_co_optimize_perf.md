---
name: Determinism work must co-optimize perf, not trade against it
description: When planning correctness/determinism workstreams, perf is a co-equal binding gate from the start — not a regression check at the end. "Deterministic but slower" is not an acceptable ship state.
type: feedback
originSessionId: 9b52ee2b-d969-442c-83ba-98396525f6fa
---
Rule: when planning a determinism, batch-invariance, or numerical-correctness workstream, treat perf as a co-equal binding objective with the correctness fix. Design and verification both must include explicit perf targets cited against the relevant hardware ceiling (HBM bandwidth, tensor-core peak, NVLINK aggregate). Plans must NOT structure perf as an end-of-phase regression check that surfaces a tradeoff — that framing concedes the perf loss as plausible. Reject "deterministic but slower" outcomes.

**Why:** The user has named this multiple times across PHASE45 D10.e, DFlash, and the new DeltaNet np>1 work. The reasoning, in their own words: "noteably we must work to improve performance while we do this work. it is not good enough to be purely determisitic at any cost". The technical premise the user accepts is that heuristic dispatch (cuBLAS GEMM algo picker, FA split-size picker) is BOTH the source of batch-shape sensitivity AND the source of perf loss at decode shapes — replacing it with fixed-tile bespoke kernels that hit tensor cores deterministically tends to BEAT the heuristic on perf, not lose to it. The DFlash T8 finding that two named kernels run at ~1% of TU102 peak made this concrete: removing heuristic dispatch opens compile-time tile geometry, full nvcc unroll, and explicit tensor-core scheduling — all perf wins that come bundled with determinism, not in tension with it.

**How to apply:**

- Plan design (the spec / kernel-design step): every replacement kernel gets a perf contract — % of memory-bandwidth or tensor-core peak it targets, work amount cited, with the ceiling derived from the relevant hardware envelope (TU102 single, dual+NVLINK, etc.).
- Verification: a perf binding step that is GREEN/RED, not "regression check with thresholds". GREEN = baseline-or-better AND % of peak hit. RED = return to design with nsys evidence, not "acceptable tradeoff".
- Composite gate: list determinism AND perf as two separate-but-required gates that BOTH must close. Stating "if determinism passes but perf fails, surface tradeoff to user" is the wrong framing — it concedes the loss as plausible.
- Risk surface: do NOT include "if there's a perf regression, surface to user". That's cover for shipping the regression. Instead: "if D8 shows ANY regression, return to D5 design with nsys evidence".
- Reject the structural-cost frame. Per `feedback_no_skipping_lessening`: "this is the cost of correctness" is the same shape as "this is hardware-limited" — both are wrong-diagnosis cover. The cost is real ONLY after exhausting tensor-core enforcement, fixed-tile geometry, and nsys-driven kernel tuning.

Originated 2026-05-14 mid-DeltaNet-PLAN-write. The PLAN.md was initially structured with D8 as "perf regression check, surface tradeoff at >15%"; user immediately corrected. The corrected D8 binds positive perf outcomes (np=1 ≥ baseline, per-kernel envelope ≥ 30% of peak) and rejects "deterministic but slower" as ship state explicitly.

Sister entries: `feedback_zero_waste_mantra` (general perf framing, the floor this builds on); `feedback_no_skipping_lessening` (don't declare structural without instrumentation); `feedback_surface_tradeoff_decisions` (surface decisions that materially affect deliverable — but NOT as cover for shipping regressions).
