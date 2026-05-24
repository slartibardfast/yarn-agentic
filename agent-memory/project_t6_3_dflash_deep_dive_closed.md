---
name: project-t6-3-dflash-deep-dive-closed
description: "production/2026-q2-next 2026-05-23: T6.3 DFlash characterisation closed across four axes. Axis 1 (per-prompt acceptance) — content-dominated, range 0.392 (King Lear prose) → 0.808 (haiku), mean 0.529. Axis 2 (draft_max sweep at NP=8) — dm=2 is sweet spot at 11.58 t/s (+5% over dm=4 default, acceptance 0.732) but still -43% vs no-DFlash 20.45. Axis 3 (NP ∈ {1,2,4,8}) — DFlash net-negative at EVERY NP (-37% at NP=1, -46% at NP=8); no flip. Axis 4 (nsys) — drafter forward mul_mat_f16_pinned_kernel_wmma at 17.8% of GPU time; target Q4_0 matmul dropped 31% → 13.3% but drafter adds more than savings recover. The 'DFlash wins' narrative was workload-locked to bench-t3.8-m3 (identical short prompts NP=2); does not generalise to mixed-prompt server workloads. Production keeping DFlash on is now informational; T6.3 records cost surface honestly per T6 discipline."
metadata:
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

Follows `[[project-t6-1-matrix-closed-with-segv]]` and `[[project-t6-6-segv-root-caused-and-fixed]]`. T6.1 surfaced DFlash -46.1% at gate0 NP=8 as the only material finding; T6.3 is the unconditional follow-on that characterises **why** with full axis-coverage.

## Four axes executed

| Axis | Approach | Tokens |
|---|---|---|
| 1 | Offline parse of matrix prod-baseline server.log via `scripts/analyze-dflash-accept-per-prompt.py` | ~10k |
| 2 | `draft_max ∈ {2,3,5,6}` at NP=8 via sibling profiles (sed-substituted from production profile) | ~30k |
| 3 | NP ∈ {1,2,4} × DFlash {on,off} via existing dflash/nodflash production profiles | ~25k |
| 4 | nsys decode trace at NP=8 dm=4 with DFlash on; bash `$PROFILE` under nsys `--trace-fork-before-exec=true` | ~15k |

Total session ~90k tokens. Bench wall-time ~35 min on 2× Quadro RTX 6000 locked 1455 MHz.

## Findings as record

**Axis 1 — per-prompt acceptance, NP=8 prod-baseline:**

| prompt | rate |
|---|---:|
| Write a haiku about a printing press | 0.808 |
| Write Python code that fits a 2nd-degree polynomial | 0.650 |
| Explain latent diffusion vs pixel-space diffusion | 0.608 |
| Translate to French (one sentence) | 0.488 |
| Describe role of telomeres in cellular aging | 0.454 |
| List five practical Rust memory tips | 0.426 |
| What are the main causes of the Peloponnesian War? | 0.408 |
| Summarize the plot of King Lear | 0.392 |

Range 0.416, mean 0.529. The matrix-cited 0.42 was the cum-dflash-stat (incl. warmup + intermediate decode states); per-task mean is higher. Content type is dominant — structured/code/haiku top of range, open prose bottom.

**Axis 2 — draft_max sweep at NP=8 dflash:**

| dm | t/s | Δ vs dm=4 | accept mean |
|---:|---:|---:|---:|
| 2 | 11.58 | +5.0% | 0.732 |
| 3 | 11.21 | +1.6% | 0.610 |
| 4 (default) | 11.03 | — | 0.529 |
| 5 | 11.27 | +2.2% | 0.533 |
| 6 | 10.59 | -4.0% | 0.502 |

Acceptance falls monotonically with dm (longer drafts = more terminal tokens with low target probability). dm=2 is the sweet spot but still -43% vs no-DFlash 20.45 t/s. **Tuning draft_max does NOT recover DFlash to net-positive on this workload.**

**Axis 3 — NP sensitivity:**

| NP | DFlash on | off | Δ |
|---:|---:|---:|---:|
| 1 | 11.46 | 18.32 | -37.4% |
| 2 | 12.99 | 20.69 | -37.2% |
| 4 | 11.14 | 20.58 | -45.9% |
| 8 | 11.03 | 20.45 | -46.1% |

DFlash is net-negative at EVERY NP. No-DFlash is near-flat (~18–21) because 2 production slots are kernel-saturated regardless of how many client prompts queue. DFlash hovers 11–13 because drafter+verify is the per-token bottleneck. Lower concurrency narrows the penalty slightly but never flips.

**Axis 4 — nsys decode trace at NP=8 dm=4 DFlash on:**

| kernel | T6.2 (no DFlash) | T6.3 (DFlash on) |
|---|---:|---:|
| NCCL AllReduce | 25.6% | 26.5% |
| `mul_mat_q_split_k<Q4_0>` (target) | 31.0% | 13.3% |
| `mul_mat_f16_pinned_kernel_wmma` (drafter) | — | **17.8%** |
| `cutlass_75_wmma_tensorop_h161616gemm` (f16) | 8.3% | 12.2% |
| PSKV singlewarp | 3.2% | 2.1% |

Drafter forward (17.8%) is the new dominant cost not present in T6.2. Target matmul drops 31.0 → 13.3 but drafter's addition exceeds the savings at 0.42-0.53 acceptance.

## Combined verdict

**DFlash is net-negative across all measured axes at gate0-shape workload.** The "DFlash wins" narrative from T3/T5 closures was workload-locked to bench-t3.8-m3 (identical short prompts, NP=2, no concurrency contention). It does not generalise to mixed-prompt server workloads. Drafter forward cost exceeds verify-savings at the content-dominated acceptance distribution gate0 exercises.

Production profile keeping DFlash on is now an informational data point; T6.3 records the cost surface honestly per T6 discipline ("T6 is not an optimization tier"). The keep/drop decision is downstream.

## T6.3 follow-ons opened (named, not deferred-as-cover)

- **T6.3.b** — re-measure axes 2+3 after NVLink install (2026-05-24). AllReduce 26.5% currently dominates; NVLink reduces small-message latency dramatically. Drafter F16 traffic may shift the dm=2 sweet spot.
- **T6.3.c** — characterise DFlash on bench-t3.8-m3 shape (identical short prompts) to quantify the upper-bound acceptance ceiling on this drafter/target pair. Bounds the workload range where DFlash net-helps.

## Discipline locks applied

- CLAUDE.md §1 (Think Before Coding) — methodology lock pre-execution: NP = client-side concurrent prompts, server stays `--parallel 2`. Avoided conflating multi-slot dispatch overhead with KV-cache-per-slot resource pressure.
- CLAUDE.md §4 (No follow-up cover) — net-negative finding is the result, not a deferred-to-future. Subtasks T6.3.b/c are named axes the deep-dive didn't cover (NVLink + identical-prompt regime), not gaps masquerading as deferrals.
- CLAUDE.md §5 (Checkbox semantics) — T6.3 closes `[x]` because evidence binds on the step's stated claim (per-prompt-shape acceptance distribution × draft_max sweep × NP sensitivity × kernel attribution). Step doesn't claim "DFlash is good"; it claims "characterise DFlash". Done.
- CLAUDE.md §8 (Negative results land cheap when honest) — applied. The net-negative landed with the numbers + the why, not rationalised into a positive narrative.

Related: `[[project-t6-1-matrix-closed-with-segv]]` (the closure that opened T6.3), `[[project-t6-6-segv-root-caused-and-fixed]]` (the fix making the matrix re-measurable), `[[feedback-no-followup]]` (no-follow-up-cover applied throughout), `[[feedback-oneshot-then-evaluate]]` (axes 1-4 oneshot, evaluated coherently). PHASE_T6_CHARACTERISATION.md T6.3 section is the in-tree record.
