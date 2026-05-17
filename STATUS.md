# Project status — Qwen 3.6 27B determinism + perf on dual sm_75

Last updated: 2026-05-17

## What this project is

Produce a fully byte-deterministic multi-slot inference server for Qwen 3.6 27B on dual Quadro RTX 6000 (TU102 / sm_75), via a fork of llama.cpp (`ik_llama.cpp`). The deliverable: production llama-server returns byte-identical output at NP={1,2,4,8} across diverse realistic-size prompts.

The host runs the same engine for several downstream workstreams (DFlash speculative decoding, MTP, Qwen 3.5 tool-calling). Those have their own track records and are not described here — see workstream-specific docs.

## Where we are right now

**Active workstream**: determinism audit. `PLAN_DETERMINISM_AUDIT.md` is the live working plan.

**Branch state**: `main` and `production/2026-q2-next` are aligned (main is fast-forward equivalent). All today's work pushed to `production/2026-q2-next`; main has the same commits awaiting a push of `origin/main`.

**What's shipped and bound**:
- Phase A, B, C — Q4_0_AR16 MMQ kernels, MMVQ kernels, cuBLAS algo pinning.
- CY.F.17 — MMQ stream_K shape-dependence fix (env-gated, applied in production env).
- CY.F.18 — scheduler `needs_sync` lifecycle race fix (in-source, gated on `sched->has_reduce`).
- Production harness 5/5 multi-run at NP={1,2,4,8} for **short** prompts (~15 tokens).
- Unit test `test-cy-np2-multi-step-decode` 10/10 NP=2 byte-identical at the C API direct-decode level.

**What's not bound** (the actual ship gap):
- Multi-slot byte-identity at realistic prompt sizes (>~50 tokens). The bisection in task #210 showed the failure is content-dependent at the kernel level, not at the build-graph conditional. Same race fires on single-GPU and multi-GPU. Hadamard compensates Q4_0 quantization noise so it's sub-threshold for most prompts, but specific prompts cross the argmax-flip line.

## Why several phase docs are stale

Today's bisection invalidated several "closures" and framings:
- **Phase D** (multi-GPU peer-access determinism) — was based on a misdiagnosis. Race exists identically on single-GPU. Closed by supersession.
- **Phase CX** (non-MMQ shape-dependent op audit) — folded into the broader determinism audit at production state. CX.B/C bindings at random tensors are insufficient; need re-bind at production K/V cache distributions.
- **Phase CY** — narrow binding (unit test 10/10, short-prompt harness 5/5) is real. Realistic-prompt binding moved to the audit.
- **`PHASE_D_PLAN.md`** — superseded by `PLAN_DETERMINISM_AUDIT.md`.
- **`PHASE_CY_F18_PROPER_FIX.md`** — closure section is accurate for CY.F.18 the bug, but framed as Phase D pickup; that framing is dead.

These docs are kept for traceability (PR review, post-mortem). The active plan is `PLAN_DETERMINISM_AUDIT.md`. New work should not extend the older phase docs.

## What "real determinism" means here, and why it's hard

The bisection result reframed the problem:
- The race is **kernel-level**, not build-graph-level. The `ne[1]>32` cast conditional that the CY.A audit named is dead code in our production config (`reduce_type=F32` is forced). `mla_attn=3` from the unit test is overridden to 0 for Qwen 3.6.
- The race is **content-dependent**. A prompt's tokens influence intermediate activations, which influence the magnitude of accumulated drift. Some prompts have argmax margins large enough to absorb sub-ULP drift; others don't.
- **Hadamard isn't masking** the race — it's keeping Q4_0 cache quantization error small enough that downstream computation tolerates the drift.

So "fix" = find every kernel/op whose output differs across slot positions at production state, and pin each to be byte-identical. Tests at random tensors (CX.B for RMSNorm, CX.C for RoPE, CY.F.1 for MMQ) gave PASS but were not at the real distribution. The audit replaces those random-tensor bindings with production-state bindings.

## The active plan in one paragraph

`PLAN_DETERMINISM_AUDIT.md` lays out: build foundation (F.1 state-capture harness, F.2 per-kernel binder, F.3 diverse prompts, F.4 orchestration), then audit per-kernel byte-identity at production state for ~16 kernels in priority order (singlewarp FA first, then cache write/read paths, MMQ at production weights, DeltaNet, residual, fused MLP, cuBLAS quantized routes, RoPE/RMSNorm re-bind, Q/K/V/output projections, softmax, LM head, etc.). Closure: 100 prompts × 4 NP × 3 sweeps, zero divergences.

**Time-box**: Option 3 — ~150k tokens of investment on foundation + first audit cycle (A.0 + A.1 + A.7). At that point, fork in the road:
- If a meaningful fix lands → continue audit.
- If no clean culprit identified → pivot to ship-NP=1-deterministic, document the NP>1 gap, move on to Phase E (perf).

## Where to start the next session

1. Read `STATUS.md` (this file).
2. Read `PLAN_DETERMINISM_AUDIT.md`.
3. Read `MEMORY.md` recent entries (2026-05-17).
4. Task #211: start F.1 (production-state capture harness).

## Tracked tasks

Active (audit foundation + first cycle):
- #211 — Audit F.1 production-state capture harness
- #212 — Audit F.2 per-kernel byte-identity binder
- #213 — Audit F.3 diverse-prompt corpus
- #214 — Audit A.1 singlewarp FA at production K/V distribution

Pending (gated on Option 3 fork verdict):
- #155 — Phase E perf tuning + nsys/ncu data
- #156 — Phase F closure binding
- #178 — RESEARCH-5 reference oracle gathering

## Other live workstreams (independent of determinism audit)

- **DFlash speculative decoding** — kernel-pipeline closure on T1-T9 (see PHASE_DFLASH.md). Independent of determinism audit; tracks its own bindings.
- **MTP production** — shipped Q2 2026; see project_production_2026q2_landing memory.
- **Vulkan multi-GPU** — production Vulkan stack work, see phases 0-22 in PHASE_VULKAN_MULTIGPU_PLAN.md.

These are not blocked by the determinism audit; the audit only blocks Phase E and F on the CUDA production engine.
