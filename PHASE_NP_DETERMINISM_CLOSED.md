# NP-determinism CLOSED — canonical writeup

**Branch**: `production/2026-q2-next`
**Closed**: 2026-05-17
**Predecessors**: `PLAN_NP_CLOSURE.md`, `PHASE_NPC4_FIX_AUDIT.md`,
`PHASE_NPC_HANDOVER.md`. This file is the single doc to read for
the final state; the predecessors are kept for historical depth.

## Result

Production binary at `production/2026-q2-next` HEAD produces
byte-identical greedy-decode output across NP={1,2,4,8} on
`DEVICE=CUDA0,CUDA1` (multi-GPU, `--split-mode graph
--tensor-split 1,1`) for the Qwen 3.5/3.6 27B production GGUF, at
default `--ctx-checkpoints 3`. All slots match NP=1; all cross-NP
slot-0 pairs match each other. No env stack required — all six
fixes are baked default-on.

## How to verify

```bash
cd /home/llm/yarn-agentic
bash scripts/verify-production-determinism.sh
# Wraps test-production-np-determinism.sh with the deterministic
# profile's settings; exit 0 = PASS, exit 1 = FAIL.
```

Verified PASS at HEAD, multi-GPU, 2026-05-17.

## What ships

| Artifact | Path | Role |
|---|---|---|
| Six baked fixes | `ik_llama.cpp` submodule HEAD | Always-on in the binary |
| Multi-slot profile | `/home/llm/profiles/qwen36-27b-x8-deterministic.sh` | Opt-in; flip `active.sh` when ready |
| Acceptance wrapper | `scripts/verify-production-determinism.sh` | Pre-deploy gate |
| Cross-NP harness | `scripts/test-production-np-determinism.sh` | Underlying NP={1,2,4,8} byte-identity test |

Current `profiles/active.sh -> qwen36-27b-x1-mtp.sh` is intentionally
not flipped — single-slot MTP is the live serving profile. The
deterministic multi-slot profile is one symlink change away.

## The six fixes (all default-on, no env knobs)

All on submodule `production/2026-q2-next`.

| # | File:line | What |
|---|---|---|
| 1 | `ggml/src/ggml-cuda.cu:3724` (`ggml_cuda_up_gate_unary`) | Fused single-token kernel for n_tokens≤8 via `ne2`-packed launch (Ny slots in `blockIdx.y`, `nb02=0` shared weights). Replaces the prior 2026-05-17-morning per-slot loop. |
| 2 | `src/llama-build-context.cpp:1593` (`llm_build_kqv`) | Single-device FA routes through PSKV under `q->ne[0]==256 ∧ v->ne[0]==256 ∧ gqa≤16 ∧ no sinks`. |
| 3 | `ggml/src/ggml-cuda.cu:2756` (`ggml_cuda_mul_mat`) | Force MMQ for all quantized weights regardless of M. `LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH` env knob removed. |
| 4 | `ggml/src/ggml-cuda.cu:2867` (cuBLAS fallback) | Per-slot loop for non-quantized M>1. |
| 5 | `ggml/src/ggml-backend.cpp:1109` | `GGML_SCHED_MAX_SPLIT_INPUTS` 10 → 32. |
| 6 | `examples/server/server-context.cpp:3923` (`batch_pending_prompt`) | Remove mid-prefill "tolerance" break — prefill runs as one continuous ubatch-chunked pass. |

Supporting kernel fix that made #1's collapsed launch NP-invariant:
`ggml/src/ggml-cuda/mmvq-templates.cuh:445` keeps `nwarps=4`
whenever `ids_data==nullptr` regardless of `ne2`, instead of dropping
to `nwarps=1` for the MoE expert case. Different `nwarps` changes the
cross-warp reduction order and breaks bit-identity.

## Perf trade

`llama-batched-bench` multi-GPU, npp=200 / ntg=64, q4_0 KV +
Hadamard, env stack matching the deterministic profile:

| NP | PP HEAD | PP pre-NPC | TG HEAD | TG pre-NPC |
|----|---------|------------|---------|------------|
| 1  | 95.74   | 176.98     | 17.95   | 18.00      |
| 2  | 17.97   | 17.97      | 17.11   | 19.80      |
| 4  | 17.33   | 17.32      | 18.21   | 23.49      |
| 8  | 16.15   | 16.10      | 18.70   | 25.26      |

- NP=1 prefill: **-45%** (the single-shot prompt-processing throughput
  most user-visible at small concurrencies).
- NP={2,4,8} prefill: within noise.
- NP=1 decode: within noise.
- NP={2,4,8} decode aggregate: **-13% / -23% / -26%**.

The `≤3%` decode-regression budget from
`feedback_determinism_must_co_optimize_perf.md` is overrun.
Cost accepted by user 2026-05-17 given the volume of work
required to recover it (F.4.1', see below). Documented in
the deterministic profile's preamble as the known trade.

## What's still open (non-blocking)

- ~~**F.4.1'** — write a new MMVQ kernel template instance with
  `ncols_y>=2` AND `rows_per_cuda_block=1`.~~ **CLOSED 2026-05-17**
  via template-param lift + `force_rpcb1` flag on `mmvq_args`. NP
  byte-identical across {1,2,4,8} multi-GPU preserved; TG +5–8% over
  HEAD at NP≥2. See `PHASE_PERF_F4_1.md`. The remaining ~10–20% TG
  gap vs pre-NPC at NP≥2 is not owed by fix #1; the bisection of
  fixes #2 (PSKV) / #4 (cuBLAS per-slot loop) is the next subtask
  in `PHASE_PERF_F4_1.md`.
- **Evidence-dir prune** — `/opt/models/yarn-audit-data/npc4-*`
  (~50 GB) and `/tmp/npc4-f41-*` (~130 MB). Salient signatures
  captured in MEMORY; raw bytes reproducible from harness.
- **Clangd cleanup** — unused `#include <mtmd-helper.h>` in
  `server-context.cpp`; unused `llama-delta-net.h` + `unordered_set`
  in `llama-build-context.cpp`.

## Diagnostic methodology recorded

When the production harness shows `NP=4≡NP=8` mutually byte-identical
but both differ from NP=1, suspect a dispatch decision that depends
on `ne2` or another batch-shape parameter, not random ULP drift.
Single-GPU all-tensors-in-layer capture (`llama-state-capture
--all-in-layer --decode-only --layers <N1>,<N2>`) localizes the
first divergent tensor cheaply. Worked twice in this iteration:
first to find `ffn_up_gate-2` (NPC.4 original), second to find that
the `mmvq` `nwarps` dispatcher branches on `args.ne2` (F.4.1).

Companion `feedback_*` memory entries that landed this iteration:
- `feedback_bake_measurement_env_gates.md` — don't leave verified
  env knobs around; bake and delete.
- New methodology note in MEMORY (2026-05-17 NPC.6 entry) on the
  `NP=K≡NP=K'` partition pattern as a localizer cue.
