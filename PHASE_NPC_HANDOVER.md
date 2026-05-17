# PHASE_NPC_HANDOVER — NPC.4 CLOSED (single-GPU), NPC.5 next

**Branch**: `production/2026-q2-next`
**Plan**: `PLAN_NP_CLOSURE.md`, `PHASE_NPC4_FIX_AUDIT.md`
**Status**: NPC.4 fully closed on single-GPU. NPC.5 (multi-GPU) is the
next binding gate; NPC.6 is ship. Latency bench unmeasured.

## TL;DR

Production harness `scripts/test-production-np-determinism.sh` PASSes at
default `CTX_CHECKPOINTS=3` on `DEVICE=CUDA0` for the Qwen 3.5/3.6 27B
production GGUF — NP={1,2,4,8} slot-0 byte-identical at the server
level. Reproduced empirically multiple times.

What's NOT yet done:
1. **Latency bench** — F.4 in `PHASE_NPC4_FIX_AUDIT.md`. The six fixes
   add launches; budget says ≤3% decode regression.
2. **NPC.5 multi-GPU** — production runs `DEVICE=CUDA0,CUDA1`; we
   verified only `CUDA0`.
3. **NPC.6 ship** — wire the harness into `profiles/active.sh`, write
   the closure MEMORY entry.

## The six baked fixes (all default-on, no env knobs)

All on `production/2026-q2-next` submodule.

| # | File:line | What |
|---|---|---|
| 1 | `ggml/src/ggml-cuda.cu:3715` (`ggml_cuda_up_gate_unary`) | Per-slot loop over fused single-token kernel for n_tokens≤8. Dense FFN. |
| 2 | `src/llama-build-context.cpp:1593` (`llm_build_kqv`, FA branch) | Route single-device FA to PSKV op under predicate `q->ne[0]==256 ∧ v->ne[0]==256 ∧ gqa≤16 ∧ no sinks`. |
| 3 | `ggml/src/ggml-cuda.cu:2756` (`ggml_cuda_mul_mat`) | Force MMQ for all quantized weights regardless of M. `LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH` env knob removed. |
| 4 | `ggml/src/ggml-cuda.cu:2856` (cuBLAS fallback) | Per-slot loop for non-quantized M>1 (LM head shape-invariance). |
| 5 | `ggml/src/ggml-backend.cpp:1109` | `GGML_SCHED_MAX_SPLIT_INPUTS` 10 → 32 (accommodates PSKV's `inp_per_row_k_bound` input). |
| 6 | `examples/server/server-context.cpp:3923` (`batch_pending_prompt`) | Remove mid-prefill `tolerance` break. Prefill always runs as one continuous (ubatch-chunked) pass. |

## Verification (single-GPU, all PASS)

```bash
# Production harness, default CTX_CHECKPOINTS=3:
DEVICE=CUDA0 bash scripts/test-production-np-determinism.sh
→ RESULT: PASS — all slots at NP in {1 2 4 8} byte-identical to NP=1

# Kernel layer (all 64 layers, decode-step 0 + 1):
/tmp/run-npc4-lout.sh 1 /opt/models/yarn-audit-data/npc4-fixD-lout-np1
/tmp/run-npc4-lout.sh 8 /opt/models/yarn-audit-data/npc4-fixD-lout-np8
python3 scripts/compare-intra-layer.py \
  /opt/models/yarn-audit-data/npc4-fixD-lout-np1 \
  /opt/models/yarn-audit-data/npc4-fixD-lout-np8 --phase decode-0
→ All shared tensors are slot-0 byte-identical.

# Real autoregressive feedback loop (64 steps):
/tmp/run-npc4-auto.sh 1 /opt/models/yarn-audit-data/npc4-auto-np1 64
/tmp/run-npc4-auto.sh 8 /opt/models/yarn-audit-data/npc4-auto-np8 64
python3 scripts/find-first-autoregress-divergence.py \
  /opt/models/yarn-audit-data/npc4-auto-np1 \
  /opt/models/yarn-audit-data/npc4-auto-np8 --np-k 8
→ No divergence across 64 autoregressive steps.
```

## Tooling landed this iteration (all on submodule HEAD)

- **`examples/llama-state-capture/llama-state-capture.cpp`** —
  - `--all-in-layer` flag (capture every named tensor at listed layers)
  - `--decode-only` flag (skip prefill ubatches)
  - `--autoregress N` flag (real greedy generation, not synthetic decode)
  - Phase-tagged outputs: `{OUT_DIR}/{phase}/layer{LL}/{name}.ub{N}.bin`
  - Manifest carries `phase` + `order` fields
  - Per-slot text dump to `gen-slot{N}.txt`

- **`scripts/compare-intra-layer.py`** (parent repo) — walks the NP=1
  manifest in fire order, joins on (phase, name) with NPK, compares
  slot-0, prints first divergence.

- **`scripts/find-first-autoregress-divergence.py`** (parent repo) —
  walks auto-0..auto-N phases, reports first (step, layer) divergence.

- `/tmp/run-npc4-{capture,lout,auto,layer}.sh` are session-local repro
  scripts; rebuild from the handover commands above on resume.

## What to do on resume

### Step 1 — NPC.5 multi-GPU (the actual production binding)

```bash
cd /home/llm/yarn-agentic
DEVICE=CUDA0,CUDA1 bash scripts/test-production-np-determinism.sh
```

Expected outcome: PASS. If it fails, the gap is in the multi-device
split path. Fix #2 (PSKV) went into `llm_build_kqv` (single-device
FA). The multi-device branch at `llama-build-context.cpp:2697`
already used PSKV — it's where the predicate originally lived, copied
into `llm_build_kqv` per fix #2. So multi-device should be covered.
But this is empirical — verify before claiming closure.

If multi-device fails, the new capture tool's `--autoregress` mode
won't reproduce it (capture tool is single-context). Will need to
either (a) extend the harness to dump per-step per-slot KV state for
multi-device, or (b) run captures with `--device CUDA0,CUDA1 -ts 1,1`
and rebuild the comparator to handle split-buffer layouts.

### Step 2 — Latency bench (F.4)

```bash
# baseline (need a way to disable all 6 fixes — they're baked, so
# baseline means a commit-revert build. Cheapest path: tag the parent
# of the latest cluster of fix commits and bench HEAD vs that tag.)
cd ik_llama.cpp/build
./bin/llama-bench -m /opt/models/recast-out/qwen3.6-27b-...gguf \
    -ngl 999 --device CUDA0 -fa 1 -p 0 -n 64 \
    -ctk q4_0 -ctv q4_0 -t 16 -b 2048 -ub 512 \
    --parallel 1 --parallel 2 --parallel 4 --parallel 8
```

Acceptance: ≤3% decode-throughput regression at each NP per
`feedback_determinism_must_co_optimize_perf.md`. Bold guess: NP=1
takes the largest hit (lost MMVQ fast path → MMQ + extra single-token
launches). NP=8 might actually improve (the per-slot loop replaces
some redundant work).

### Step 3 — NPC.6 ship

- Add the harness call to `profiles/active.sh`'s acceptance step OR
  create `profiles/active-deterministic.sh`.
- Decide policy on `ctx_checkpoints` in production: keep at 3 (now
  safe), reduce to 0 (no rollback overhead), or `0` for deterministic-
  serving profile and `3` otherwise.
- Write the MEMORY closure entry per
  `feedback_claudemd_no_followup_and_checkbox_semantics_provisional.md`
  — what was delivered and what's binding it.

### Cleanup (low priority)

- clangd flagged unused `#include <mtmd-helper.h>` in
  `server-context.cpp` and a few unused includes in
  `llama-build-context.cpp` from this iteration. Tidy commit when
  convenient.
- `/opt/models/yarn-audit-data/npc4-*` evidence dirs (~50 GB) can be
  pruned once the MEMORY entry preserves the salient signatures.

## What NOT to redo

- All six fixes are baked, verified, no env knobs. Don't add a toggle
  back. (see `feedback_bake_measurement_env_gates.md`)
- Don't reopen the kernel localization — the captures prove every
  layer × every autoregressive step is byte-identical. The remaining
  work is multi-GPU plumbing + ship, not more kernel hunting.
- Don't try to re-introduce the mid-prefill `tolerance` checkpoint
  break (fix #6) without a kernel-level fix for incremental-prefill
  FA shape-invariance first. The break was correctly identified as
  the source of the production-harness gap; restoring it without
  that kernel fix will re-open NP=1 vs NP>=2.
- Don't trust `--ctx-checkpoints 0` as the "fix" — it's a workaround.
  Our fix #6 makes `CTX_CHECKPOINTS=3` (the default) safe.

## Evidence preserved

```
/opt/models/yarn-audit-data/
  npc4-fixD-lout-np{1,2,4,8}/         # all-layer l_out kernel verify (PASS)
  npc4-auto-np{1,8}/                  # 64-step autoregressive (PASS, no div)
  npc4-textauto-np{1,8}/              # 64-step + gen text dump
  npc4-out2-np{1,8}/                  # post LM-head-fix result_output verify
  npc4-fixD-harness/run-*/            # production harness PASS (CTX=0)
  npc4-fixF-harness/run-*/            # production harness PASS (CTX=3)
```

## Commits this iteration (most recent first)

- `NPC.4 FULL CLOSURE — production harness PASSes at default CTX_CHECKPOINTS=3`
- `NPC.4: stop splitting prefill mid-prompt for "tolerance" checkpoint`
- `NPC.4: cuBLAS per-slot loop + capture gen-text dump`
- `NPC.4 production harness CLOSED at CTX_CHECKPOINTS=0; ctx-checkpoint side-effect is the residual`
- `NPC.4 kernel-level CLOSED, prod harness narrowed to NP=1 vs NP>=2 tail`
- `NPC.4 closure (kernel-level): bake shape-invariant FA + mul_mat + FUSED_UP_GATE`
- `scripts: comparator for autoregressive cross-NP captures + submodule bump`
- `llama-state-capture: add --autoregress N for real greedy generation`
- `PHASE_NPC4_FIX_AUDIT: deep audit of ggml_cuda_up_gate_unary + fix plan`
- `NPC.4 LOCALIZED — ffn_up_gate MoE GEMM is the layer-2 divergence`
- `llama-state-capture: add --all-in-layer + --decode-only, phase-tagged outputs`
