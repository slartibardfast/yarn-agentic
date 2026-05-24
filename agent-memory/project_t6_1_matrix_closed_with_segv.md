---
name: project-t6-1-matrix-closed-with-segv
description: "production/2026-q2-next 2026-05-23: T6.1 binary ablation matrix CLOSED with 6 cells at gate0 NP=8 workload. 4 clean + 2 SEGV. Headline findings: (1) DFlash NET-NEGATIVE -50.3% at varied prompts (0.42 acceptance, drafter cost dominates) — workload-shape sensitive vs the bench-t3.8-m3 identical-prompt win story. (2) Hadamard NET-NEGATIVE -17.9% throughput-only (accuracy gain uncharacterised). (3) **defrag-on (0.1) + DFlash multi-slot CRASHES** under varied-prompt NP=8 — production default config is unsafe at this workload. T5.9.E closure measurement at NP=2 + bench-t3.8-m3 kept fragmentation ~1.0 so threshold never fired actual defrag passes. Real production bug exposed by T6.1. Three highest-priority follow-ons: T6.6 defrag root-cause, T6.3 DFlash acceptance-by-prompt-shape, T6.8 Hadamard accuracy delta."
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

Production state at landing: parent `8cca4ed` (PHASE_T6_CHARACTERISATION T6.1 closure), data `1d9efbb` (matrix data + scripts), submodule unchanged at `4f4da34f`. Branch: production/2026-q2-next, pushed.

## What this session produced

T6.1 binary ablation matrix at the gate0 reference workload (8 prompts × 256 max_tokens × NP=8 fired into --parallel 2 server, queue depth 6). Six cells under `data/t6.1-matrix-20260523T194240/`.

| cell_id | dflash | hadamard | defrag | t/s_agg | clean? |
|---|---|---|---|---|---|
| prod-baseline | on | on | 0.1 | 3.44 (2/8) | **SEGV** |
| no-defrag | on | on | -1 | 10.42 | ✓ |
| no-dflash | off | on | 0.1 | 20.47 | ✓ |
| no-dflash-nodefrag | off | on | -1 | 20.96 | ✓ |
| no-hadamard | on | off | 0.1 | 4.40 (2/8) | **SEGV** |
| no-hadamard-nodefrag | on | off | -1 | 12.69 | ✓ |

Baseline for delta = `no-defrag` (10.42 t/s) — the production-default-config-but-with-defrag-off cell, since the actual production default crashed.

## Three findings

### 1. DFlash is net-negative at this workload (-50.3%)

`no-defrag` (DFlash on) 10.42 t/s vs `no-dflash-nodefrag` (DFlash off) 20.96 t/s. Drafter forward cost dominates because gate0 prompts vary (some code, some translation, some haiku) and acceptance rate falls to 0.42 on harder prompts (measured in `cell-prod-baseline/server.log`: `draft acceptance rate = 0.42175 (159 accepted / 377 generated)`).

The "DFlash is a win" story from T3-T5 closure docs was workload-locked to bench-t3.8-m3 (identical short prompts, NP=2). It does not generalise. T6.3 owes the acceptance-rate distribution by prompt shape; that's the explanation for when DFlash is net-positive vs net-negative.

### 2. Hadamard is net-negative on throughput (-17.9%)

`no-defrag` (Had on) 10.42 t/s vs `no-hadamard-nodefrag` (Had off) 12.69 t/s. Hadamard exists to recover Q4_0 quantisation accuracy; the t/s cost is the price of that recovery, NOT a regression. The T6.1 matrix only measures throughput — the accuracy delta (Q4_0 with vs without Hadamard) is uncharacterised. T6.8 owes accuracy measurement; the trade-off decision lives there, not here.

### 3. defrag-on (0.1) + DFlash multi-slot CRASHES under varied-prompt NP=8

Both cells that combined `--defrag-thold 0.1` (T5.9.E default) with DFlash + concurrent firing of 8 varied prompts SEGV'd silently after 2/8 prompts completed. The crash signature in both `cell-prod-baseline/server.log` and `cell-no-hadamard/server.log`: fragmentation steady around 0.1-0.3 (just above threshold; defrag firing repeatedly), then on a slot-release-and-reuse boundary fragmentation jumps to 0.6-0.8 and the process disappears mid-decode. No GGML_ASSERT, no CUDA error message — clean SEGV in CUDA-land.

This is a REAL production bug, not a measurement artifact. T5.9.E closure validation at production NP=2 + bench-t3.8-m3 (identical short prompts, no slot churn) kept measured fragmentation ~1.0 throughout — threshold 0.1 never triggered actual defrag moves. The bug only fires when defrag passes actually run mid-flight under DFlash multi-slot with slot reuse.

**The production default config is broken at realistic workloads.** T6.6 must root-cause and either fix it or re-flip the default back to -1 with the fix tracked.

## What didn't get characterised in T6.1 (named, not cover)

Four features have no runtime knob to disable, so they couldn't be binary-cell tested:
- T4 chunked-prefill admission (`--prefill-chunk-budget > 0` overrides budget but doesn't disable)
- T5.9 paged BACKING (auto = byte-identical default; no off knob)
- per-slot-kv FA dispatch (needs build flag to revert to legacy `ggml_flash_attn_ext`)
- T3 unified-stream dispatch (needs build-time gate to revert to legacy per-stream)

The PHASE doc names them as scoped INTO the unconditional T6.4/T6.5/T6.7/T6.9 deep-dives — where the deep-dive owns the "what does this contribute" question via sweep, not on/off. The decision is honest: don't build build-flag gates just for binary cells when the deep-dive infrastructure can produce the same data better.

NP-sub-axis cells (NP ∈ {1, 2, 8}) are also not in T6.1. The matrix at NP=8 produced the unflinching signals it needed (especially the crash + workload-shape effects). NP sensitivity per feature lives in T6.3-T6.9.

## Artifacts

- `data/t6.1-matrix-20260523T194240/SUMMARY.md` + `.json` — aggregator output (per-cell table, per-feature delta, per-feature verdict).
- `data/t6.1-matrix-20260523T194240/cell-*/cell.json` — six schema-conformant cells (validated against `specs/t6-characterisation-cell.allium` via `scripts/validate-t6-cell.py`).
- `scripts/cross-engine-bench.sh` — harness extended to read 9 new env vars for the config block (`LLAMA_BENCH_{DFLASH,K_HADAMARD,V_HADAMARD,FLASH_ATTN,DRAFT_MAX,KV_POOL_BLOCKS,DEFRAG_THOLD,CTX_CHECKPOINTS,CACHE_RAM}`).
- `scripts/validate-t6-cell.py` — standalone shape+enum checker mirroring the 5 contracts in `specs/t6-characterisation-cell.allium` (Python-native so cells can be validated without booting allium).
- `scripts/run-t6.1-matrix.sh` + `scripts/run-t6.1-matrix-extension.sh` — sequential drivers.
- `scripts/aggregate-t6-matrix.py` — reads `cell-*/cell.json`, writes SUMMARY.md + SUMMARY.json with verdict logic.
- Sibling profiles (host config, not in repo): `/home/llm/profiles/qwen36-27b-x2-{nodflash,nohadamard,nodefrag,nodflash-nodefrag,nohadamard-nodefrag}.sh`.

## Discipline locks (per CLAUDE.md §4 + §8)

The matrix produced unflinching results because the discipline held:
- Two SEGVs were treated as findings, not "follow-up work to make the matrix complete." They became the headline. The matrix is CLOSED with the broken cells documented.
- "Net-negative" verdicts on DFlash and Hadamard were not softened. The fact that DFlash was a win at OTHER workloads does not entitle it to a different verdict at THIS workload — the verdict is workload-scoped.
- The "we re-ran with defrag-off because defrag-on crashed" pivot was an isolated extension (two cells), not a re-redesign of the matrix. Original cells stayed in the data with their broken status.

## Next session pickup

- T6.6 (defrag deep-dive) is now critical. Investigate the defrag-on + DFlash + slot-reuse crash. Likely candidates: the CUDA Q→Q same-type kernel landed at T3.6.I.c2 + T5.7c may not be DFlash-aware (DFlash holds extra per-slot KV that defrag moves invalidate?). Check with `defrag-thold 0.1` + DFlash + 4 varied prompts via cross-engine-bench.sh — should reproduce in minutes.
- T6.3 (DFlash deep-dive) — acceptance rate by prompt shape. Already have one cell of evidence: gate0 prompts yield 0.42 acceptance on the second prompt of `cell-prod-baseline/server.log`; that's BELOW the workload's break-even threshold.
- T6.8 (Hadamard deep-dive) — accuracy delta under Q4_0 KV with vs without Hadamard. This is the "is the -18% throughput cost worth it" question, owed.
- Production users: the default flip of `defrag_thold` to 0.1 is now known unsafe with DFlash multi-slot under varied prompts. Until T6.6 lands, the bigctx profile already sets `--defrag-thold -1`; the main production profile would benefit from the same. Decision for user: revert default to -1 in production profile pending T6.6 fix, or leave at 0.1 and rely on the bench-t3.8-m3 workload's safety?

Related: [[project-t5-9-closure-audit-and-t6-opened]] (the T5.9.E default flip that this matrix tested + broke), `PHASE_T6_CHARACTERISATION.md` §T6.1 (in-tree closure record), `[[feedback-claudemd-no-followup-and-checkbox-semantics-provisional]]` (the discipline that kept the SEGV finding as a finding instead of a "we should re-measure" cover), `[[feedback-no-skipping-lessening]]` (the discipline that locked the net-negative verdicts).
