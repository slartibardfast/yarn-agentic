---
name: project-t5-probe-falsified-path-c-override
description: "T5.0-probe falsified perf-uplift mechanism on current workload; user selected Path C override with reframe to high-ctx feasibility; Tier 5 proceeds as forward-looking infra, not current-workload throughput"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

production/2026-q2-next 2026-05-22.

**Fact**: T5.0-probe analytical decomposition (`data/t5-probe-findings.md`)
falsified the implicit perf-uplift mechanism encoded in the decision rule's
`GP5.b = 26.49 × (1 + waste_pct/100)` formula.

- Structural waste % trips the rule trivially (~99% at production NP=2
  ctx=524288).
- BUT composition with T4 finding ([[project_t4_bundle_a_landed]]) — kernel
  saturation at NP=8 is binding — shows paging doesn't address the actual
  bottleneck on current workload.
- VRAM not capacity-bound (48 GiB aggregate, KV ~15 GiB).
- 5.84× gap to vLLM (154.77 vs 26.49 t/s) plausibly kernel-level (FA2 vs
  PSKV singlewarp, token-major dispatch, fusions), not K/V layout.

**User decision: Path C override (2026-05-22).** Verbatim: "larger ctx sizes
will come into play later so we must do this with complete sincerity".

**Why**: reframe shifts Tier 5's motivation from throughput-uplift on
production-current to **structural feasibility at upcoming high-ctx
workloads**. At ctx 1M NP=8 Q4_0, contiguous layout needs ~1.2 TB allocated
(we have 48 GiB) — contiguous cannot allocate; paged fits in ~10 GiB by
allocating only actually-written blocks. The lever flips from throughput
to feasibility.

**How to apply**:

- Tier 5 work proceeds. T5.A audit and T5.0-probe artifacts stay in tree as
  audit record.
- GP5.b is reframed from numeric uplift gate to feasibility gate (contiguous
  fails to allocate at ctx ≥ 1M NP=8; paged succeeds with finite TG).
  Current-workload uplift number (53 t/s) stays in ledger as measurement,
  not hard gate — acknowledged at-risk of structural FAIL by kernel
  bottleneck.
- "Complete sincerity" discipline: no shortcut specs, all 8 property tests
  RED-bound, allocator OOM at high ctx is first-class test, DFlash + paging
  composition tested at high-ctx workload.
- The 5.84× kernel-level gap to vLLM is a SEPARATE workstream (nsys profile
  vLLM vs ik_llama, OpenQ-E lever #3); Tier 5 does NOT claim to address it.
  If kernel-level work also lands later, Tier 5's paging layout will compose
  with whatever kernel ships.

**Related**: [[project-t4-bundle-a-landed]], [[project-t3-8-perf-gate-failed-tier4-justified]],
[[feedback-anchor-to-measured-baselines]], [[feedback-no-workarounds]],
[[feedback-override-locked-policy-pragmatically]].

The probe doing its job (catching the premise mismatch BEFORE 150-230k spend)
+ the user's reframe (forward-looking infra justification) together preserve
both audit discipline and forward velocity. The probe is not a failure mode;
it's the cheap insurance that surfaced a wrong-framing assumption in the rule,
which the user then corrected.
