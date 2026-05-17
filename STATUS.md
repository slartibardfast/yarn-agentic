# Project status — Qwen 3.6 27B determinism on dual sm_75

Last updated: 2026-05-17

## The goal

Get a production multi-slot inference server that returns **byte-identical output at NP={1,2,4,8}** across the prompts a real user would type. The hardware is two Quadro RTX 6000s (TU102, sm_75); the software is a fork of llama.cpp called `ik_llama.cpp`.

Byte-identity matters because the alternative is "deterministic up to fp32-ULP noise" — which means argmax can flip, tokens can diverge, and an experiment that returns one answer at NP=1 returns a different answer at NP=2. For research reproducibility, regression debugging, and any system that stores or compares model responses, that's unacceptable.

We started this work thinking it was a kernel-level math problem.

## What we've built (and what works)

Four named fixes have shipped on `ik_llama.cpp`:

- **Phase A + B** — custom Q4_0_AR16 MMQ and MMVQ kernels. INT4 matmul paths that don't suffer from cuBLAS's per-shape algo selection.
- **Phase C** — cuBLAS algorithm pinning for F16/BF16/F32 weights. Algo is locked regardless of M; same shape always uses the same code path.
- **CY.F.17** — disabled MMQ's stream_K dispatch, which had a shape-dependent reduction order at prefill M ≥ 215.
- **CY.F.18** — fixed a scheduler `needs_sync` lifecycle race for multi-GPU peer writes. The bug let the scheduler clear its sync flag while peer-stream writes were still pending.

At the C API level, the unit test `test-cy-np2-multi-step-decode` runs 10/10 byte-identical comparing NP=2 against NP=1. At the server level, the production harness `scripts/test-production-np-determinism.sh` passes 5/5 multi-run with the default short prompt at NP={1,2,4,8}.

These are real, valid, shipped fixes.

## The catch

The "default short prompt" is fifteen tokens.

The production harness's hardcoded default prompt was *"The history of artificial intelligence began in earnest with the work of"* — short enough that prefill barely exercises the model's batched code paths. When we re-tested today with a realistic ~200-token prompt, the same harness fails 0/5 multi-run at NP>1. Specific prompts at specific sizes deterministically produce different output across slots.

We hadn't been catching this because the test fixture was overfit to a toy case.

## Today's bisection, and what it told us

The natural reflex was to assume the failure was somewhere in the kernel work we'd been doing — maybe CY.F.17 didn't extend to all M values, or CY.F.18 had a corner case at long context. So we built a bisection: start from the unit-test config (which passes 10/10) and step it toward the server config, one variable at a time. Five configs × six prompt sizes = thirty cells. We expected to find a single variable whose change flips PASS→FAIL.

The result was not what we expected.

**Several variables turned out to have no effect.** The unit test sets `mla_attn=3`, which we'd treated as a meaningful Qwen-specific knob. But `llama.cpp:7156` force-resets `mla_attn=0` for any model not in {DeepSeek2, GLM_DSA, Mistral4}. Qwen 3.6 never runs mla_attn=3 regardless of what the unit test asks for. The bisection cell was testing the same thing on both sides.

**The CY.A audit's named suspect was dead code.** The `cur->ne[1] > 32` conditional that casts the residual stream to fp16 for prefill — long flagged as the prime determinism risk — doesn't fire in our config. `reduce_type` is forced to F32 for Qwen 3.5/3.6 (shipped under CY.F.16). The cast condition `ne[1] > 32 && reduce_type != GGML_TYPE_F32` evaluates to false on the right side, always.

**The matrix is non-monotonic.** We see PASS-FAIL-PASS-FAIL across prompt sizes. ~200 tokens fails; ~350 tokens passes; ~1100 tokens fails. Same model, same config, same harness. Whatever's wrong isn't "prompts above some size threshold."

**And then the punchline: Hadamard isn't masking the race.** Turning Hadamard rotation off doesn't relax the determinism constraint — it dramatically tightens it. Without Hadamard, even the three-token "tiny" prompt fails. With Hadamard, only specific prompts fail. The reframe: Hadamard's stated purpose is precision optimization for Q4_0 KV cache quantization, but it has a second effect we hadn't recognized — it keeps cache values uniform enough that the upstream drift stays below the argmax-flip margin. The drift exists everywhere; Hadamard happens to keep most prompts safely above the line.

The race is at the **kernel level**, not the build-graph level. It's **content-dependent**: specific token sequences trigger it, others don't. It fires **identically on single-GPU and multi-GPU**. None of the fixes named in earlier phases address it directly — the kernels' shape-dependent paths run below the conditionals the audit was looking at.

## What this changes

Phase D was framed as multi-GPU peer-access determinism. That framing is dead — the race fires single-GPU. Phase CX was the non-MMQ shape-dependent op audit, but its bindings tested random tensors at unit-test shapes; production-state bindings at the same ops would not pass. Phase CY had been "closed" at narrow scope; the real binding doesn't hold.

We considered three paths:

- **A. Strict determinism-first**: block all forward work until the audit closes.
- **B. Ship NP=1 deterministic, document the NP>1 gap**: move on to perf tuning and other workstreams while the determinism work continues in the background.
- **C. Time-box the audit (~150k tokens of foundation + first audit cycle), then re-evaluate**: if a meaningful fix lands, continue; if not, fall back to B.

We chose C. The plan is `PLAN_DETERMINISM_AUDIT.md`: build a state-capture harness, build a per-kernel byte-identity binder, gather 100 diverse prompts, then audit the most likely culprits in priority order — singlewarp FA at production K/V cache first, then cache write/read paths, then MMQ at production weight distributions. The audit closure binding is 100 diverse prompts × 4 NP values × 3 sweeps, zero divergences. That's deliberately strict; today's bisection convinced us that anything narrower is overfit-able.

## Where to start the next session

1. Read `STATUS.md` (this file) and `PLAN_DETERMINISM_AUDIT.md`.
2. Read MEMORY.md entries from 2026-05-17 onward — they record the falsified facts (mla_attn force-reset, ne[1]>32 cast dead, harness bash bug that masked all probes for half a session) and the corrected understanding.
3. Begin task #211 — Audit F.1, the production-state capture harness.

## Other live workstreams

These are independent of the determinism audit and not blocked by it:

- **DFlash speculative decoding** — Qwen3.6-27B with bespoke sm_75 kernels. Kernel-pipeline closed at T1-T9.
- **MTP production** — Multi-token prediction for Qwen 3.6 27B, shipped Q2 2026.
- **Vulkan multi-GPU split-mode graph** — RDNA2 + Vega multi-GPU production stack, 22 phases of work landed.

See the workstream-specific docs in the mdBook nav.
