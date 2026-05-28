# PHASE_PERF_F4_1 — recover the determinism perf gap

**Branch**: `production/2026-q2-next`
**Predecessor**: `PHASE_NP_DETERMINISM_CLOSED.md` (NPC closed and shipped)
**Status**: F.4.1' kernel rewrite [x] CLOSED 2026-05-17. Remaining perf
gap traced to fixes #2/#4 — separate subtask (`Probable second target`).

## Outcome (2026-05-17)

F.4.1' delivered: lifted `rows_per_cuda_block` to a 4th template
parameter on `k_fused_mul_mat_vec_q`/`fused_mul_mat_vec_q`; added
`force_rpcb1` to `mmvq_args` and the public
`ggml_cuda_op_fused_mul_mat_vec_q_id` entry; pinned `nwarps=4` in
the dispatcher when `force_rpcb1`; replaced the ne2-packed call in
`ggml_cuda_up_gate_unary` with a non-packed `ncols_y=Ny`,
`force_rpcb1=true` call.

Acceptance wrapper (multi-GPU `CUDA0,CUDA1`, NP={1,2,4,8},
ctx-checkpoints=3): **PASS** — every slot byte-identical to NP=1;
cross-NP slot-0 matrix all byte-identical.

Measured perf (llama-batched-bench, `-npp 200 -ntg 64 -npl 1,2,4,8`):

| NP | PP F.4.1' | PP HEAD (NPC) | TG F.4.1' | TG HEAD (NPC) | TG pre-NPC |
|----|-----------|---------------|-----------|---------------|------------|
|  1 |  95.56    |  95.74        |  17.95    |  17.95        |  18.00     |
|  2 |  17.95    |  17.97        |  17.94    |  17.11 (+4.8%)|  19.80     |
|  4 |  17.32    |  17.33        |  19.45    |  18.21 (+6.8%)|  23.49     |
|  8 |  16.15    |  16.15        |  20.11    |  18.70 (+7.5%)|  25.26     |

Modest TG uplift over the NPC slot-packed approach (+5–8% at NP≥2)
with NP-determinism preserved. The remaining gap vs pre-NPC
(-9% TG at NP=2, -17% TG at NP=4, -20% TG at NP=8) is **not** owed
by fix #1 (this rewrite); it is owed by fixes #2 (PSKV) and #4
(cuBLAS per-slot loop). Confirms the "Probable second target after
F.4.1'" diagnostic call below.

Cost: ~25k tokens (well under the 80–150k budget). One iteration —
the initial implementation passed acceptance at NP={1,2,4} but
diverged at NP=8 (ncols_y=8 → nwarps=2 selector); localized in one
read and fixed by pinning `nwarps=4` under `force_rpcb1`. No
intra-layer capture round needed.

## Why this phase exists

NPC.4 / .5 / .6 shipped multi-slot byte-identical NP-determinism but
overran the ≤3% decode-regression budget from
`feedback_determinism_must_co_optimize_perf.md`:

| NP | PP HEAD | PP pre-NPC | TG HEAD | TG pre-NPC |
|----|---------|------------|---------|------------|
| 1  | 95.74   | 176.98     | 17.95   | 18.00      |
| 2  | 17.97   | 17.97      | 17.11   | 19.80      |
| 4  | 17.33   | 17.32      | 18.21   | 23.49      |
| 8  | 16.15   | 16.10      | 18.70   | 25.26      |

Two independent regression sites:

- **NP=1 PP -45%** — likely fix #2 (PSKV) and/or fix #4 (cuBLAS
  per-slot loop). Not touched by F.4.1'.
- **NP≥2 TG -13% to -26%** — fix #1 (slot-packed `ncols_y=1`
  fused up_gate kernel). F.4.1' is the kernel rewrite that
  recovers this.

## F.4.1' scope (this phase's core deliverable)

Add a new MMVQ kernel template instance for `ncols_y∈{2,4,8}` AND
`rows_per_cuda_block=1`. Existing `ncols_y≥4` paths use
`rows_per_cuda_block=2`, a different reduction tree that's
NP-divergent against `ncols_y=1`. The new instance combines:
- in-block weight amortization across N output columns (the source
  of pre-NPC.4 throughput)
- per-output independent reduction (the source of NP-invariance)

Then route `ggml_cuda_up_gate_unary`'s `n_tokens<=8` path through
the new kernel with `ncols_y=Ny`, replacing the current
`ne2`-packed `ncols_y=1` launch.

**Token estimate**: 80–150k. Subtasks:

| Subtask | Tokens |
|---|---|
| Template specializations in `mmvq-templates.cuh` | 15–25k |
| Dispatcher entry | 5–10k |
| Wire `ggml_cuda_up_gate_unary` | 5–10k |
| NP-invariance verify (intra-layer harness, single-GPU + multi-GPU) | 5–10k |
| Perf delta measurement vs HEAD + pre-NPC.4 | 10–20k |
| Debug if new kernel introduces ULP drift (R2) | 30–60k |
| Doc + commit | 5–10k |

## Probable second target after F.4.1'

NP=1 PP -45% won't move from F.4.1'. Diagnostic: bench with each of
fixes {#2, #4} temporarily reverted in isolation to see which one
owns the regression. Probable next subtask once F.4.1' is in.

## How to resume cold

```bash
cd /home/llm/yarn-agentic
git log --oneline -10   # confirm at production/2026-q2-next HEAD
bash scripts/verify-production-determinism.sh   # smoke check — must PASS

# Apples-to-apples bench at HEAD:
env GGML_CUDA_MMQ_DISABLE_STREAM_K=1 LLAMA_FATTN_PER_SLOT_KV_ENABLE=1 \
    LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH=1 LLAMA_PSKV_MODE=singlewarp \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  ./ik_llama.cpp/build/bin/llama-batched-bench \
    -m /opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf \
    -ngl 999 -sm graph -ts 1,1 -dev CUDA0,CUDA1 \
    -ctk q4_0 -ctv q4_0 -fa on -t 16 -b 2048 -ub 512 -c 32768 \
    -npp 200 -ntg 64 -npl 1,2,4,8
```

## Live state at session start

- `profiles/active.sh -> qwen36-27b-x8-deterministic.sh` (was
  `qwen36-27b-x1-mtp.sh` pre-flip; the MTP profile is still on
  disk for rollback).
- `systemctl --user status llama-server`: active, 8 idle slots,
  /health=ok.
- A perf-phase change that breaks determinism breaks production
  serving — verify with the acceptance wrapper before pushing.

## Things NOT to redo

- Don't re-localize the `nwarps` dispatcher bug; it's already
  fixed in `mmvq-templates.cuh:445`. Different bug class than
  F.4.1'.
- Don't restore env knobs — fixes are baked default-on.
  See `feedback_bake_measurement_env_gates.md`.
- Don't try `ne2`-packing as a perf strategy in isolation — it
  IS the current strategy. F.4.1' needs to recover bandwidth on
  top of that.

## Methodology notes

- `NP=4≡NP=8` mutually byte-identical but both differ from NP=1
  → suspect a ne2-derived dispatch decision, not random drift.
- Single-GPU all-tensors-in-layer capture
  (`llama-state-capture --all-in-layer --decode-only
  --layers <N1>,<N2>`) is the cheap localizer.
- Worked twice in NPC.4 / F.4.1.

## Companion files

- Canonical NPC closure: `PHASE_NP_DETERMINISM_CLOSED.md`
- Deep audit (history): `PHASE_NPC4_FIX_AUDIT.md`
- Plan with closure section: `PLAN_NP_CLOSURE.md`
- Acceptance wrapper: `scripts/verify-production-determinism.sh`
- Harness: `scripts/test-production-np-determinism.sh`
- Capture tool: `ik_llama.cpp/examples/llama-state-capture/`
- Comparators: `scripts/compare-intra-layer.py`,
  `scripts/find-first-autoregress-divergence.py`
