---
name: Coherent oneshot + measured evaluation is the preferred work pattern for performance plans
description: When the user asks to implement a multi-step performance plan, write the bundle coherently without intermediate build/validate cycles, then evaluate measured results against the calculated ceiling — including failures.
type: feedback
originSessionId: e17bdbd7-e0a3-4ec7-9818-e8f46eed5283
---
For Phase-style multi-step performance plans, the user wants:
1. **Implement coherently across all steps** — don't pause to build, test, or commit per step. Treat the whole thing as one design pass.
2. **Then build once + evaluate against the calculated ceiling**. Measured numbers in tables, not prose.
3. **Honest about what broke**. "Step 1 fused crashed in build_std_attention because helper assumes lctx.n_tokens, fix is refactor or inline" is better than hiding the failure or rewriting the plan to pretend it didn't happen.
4. **Don't speculatively demote plan steps based on workload-specific data**. r7=0% in one X02 256K profile doesn't mean Step 5 is invalid — different workloads (longer prompts, p_min=0) may hit r7 hard. Leave the step in the plan; note the workload-specific finding.

**Why:** The user validated this pattern explicitly ("excellent. that's how we make progress.") after I:
- Reset speculative "Step 3 first" reordering once the back-of-envelope was proved wrong by tagged measurement (5× off).
- Wrote Steps 1.1-1.5, 2.1, 2.3, 3.2, 3.3, 4, 5 across 9 files in one pass, no intermediate commits.
- Reported "Step 3 hook +9% at d=1 = AT CEILING; Step 1 fused crashes; Step 2 deferred" plainly.
- Compared measured to ceiling table for both d=1 (hit) and d=5 (missed).

**Carve-out: per-step measurement when verification is cheap and sanctioned.** If verification per change is fast (<5 min — e.g. the Phase 37 harness `--fast` tier at ~3 min) AND the user has explicitly sanctioned per-step measurement (e.g. "we'll add each element after just a --fast run"), run the harness per change. The bundle pattern guards against *speculative reordering* on imagined gains; it does not forbid *measured progress* against a fixed gate when measurement is cheap. Concretely: PHASE37's eight-item schedule against `tests/mtp-fused/gate.yaml` is per-step legitimate; a hypothetical "implement #2/#5/#3 in parallel and decide which to keep based on intermediate ratios" would still violate the rule.

**How to apply:**
- When the user says "carefully oneshot" — *don't pause* to build between steps. Skip per-step validation. Land the bundle, then run profile/test once.
- When picking implementation order, follow the plan's stated dependencies (Step 2 depends on Step 1) rather than reordering on speculative gain estimates.
- Surface the actual win vs the projected ceiling in a side-by-side table. If a step broke, say "blocks Step 2's +X% lever", don't hide it.
- When the user says "be brief, expert in a hurry" — minimal commentary in code (one-line WHY at most), aggressive editing, no exploratory analysis beyond what the plan already specifies.
- When the user pushes back on a speculative demotion ("nothing is demoted, that was speculative"), restore the full plan immediately and don't re-argue.
