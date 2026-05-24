---
name: dflash-workpath-2026-05-18
description: DFlash workpath as of 2026-05-18 — NP=4/8 vanilla drift empirically resolved (T9 blocker cleared); next active workitem is the libllama multi-slot API extension; kernel optimization comes after
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

DFlash workstream restart as of 2026-05-18.

**Status corrections vs PHASE_DFLASH (snapshot was 2026-05-14):**

1. **NP=4/8 vanilla drift IS RESOLVED.** PHASE_DFLASH lists "Resolve-drift
   workstream — named, not started" as item 1 of future work. The
   `scripts/verify-production-determinism.sh` harness on 2026-05-18
   PASSES at NP={1,2,4,8}: all slots byte-identical to NP=1, all 6
   cross-NP slot-0 comparisons byte-identical. The
   cuBLAS-GEMM-algo + FA-split-size drift T9 documented
   ([[dflash-t9-np-validity-drift-signature]]) has been fixed by NPC
   work that landed in the interval. The phase doc has not been updated.

2. **DFlash multi-slot libllama API IS the active blocker.** This is
   the named future-work item 2: ~115–185k tokens of work. With
   determinism cleared, this is what gates np>1 DFlash bench.

3. **Kernel optimization remains future work (item 3).** The two
   bottlenecks per T8 measurement of record
   ([[dflash-t8-closed]]):
   - `dflash_drafter_lm_head_kernel`: 41.3% GPU time, 1228 ms/call,
     **600× off TU102 memory-bandwidth ceiling**. One CTA per row
     (n_rows=1 at np=1 → 1 CTA on 72 SMs = 1.4% SM util), scalar fp32
     dot loop over V=248320 × D_emb=5120 against BF16 weights, no
     tensor cores, no SMEM weight tile.
     File: `ggml/src/ggml-cuda/dflash/dflash-drafter-lm-head.cu`.
   - `gemm_row_x_col_kernel`: 34.0% GPU time, 29 ms/call. 5 CTAs on
     72 SMs = 7% SM util, scalar fp32, no tensor cores.
     File: `ggml/src/ggml-cuda/dflash/dflash-drafter-forward.cu:121-141`.

**Decided ordering 2026-05-18:** Option B — multi-slot libllama API
first, then kernel optimization. Rationale: until DFlash can run
multi-slot the production server can't ship it; building the API
surface first means each subsequent kernel improvement is immediately
validatable at np=8 against the production NPC contract.

**How to apply:** Future session picking up DFlash should:
1. Start with `data/dflash-multi-slot-api-brief-2026-05-18.md`
   (handover brief in yarn-agentic data/).
2. Re-verify NPC PASSES at NP={1,2,4,8} before scoping (run
   `scripts/verify-production-determinism.sh`).
3. Confirm DFlash builds with `-DGGML_CUDA_DFLASH=ON`.
4. Read PHASE_DFLASH.md task T9 + the future-work pointers section to
   confirm scope.
