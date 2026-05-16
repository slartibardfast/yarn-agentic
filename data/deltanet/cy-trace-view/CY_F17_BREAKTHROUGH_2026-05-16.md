# CY.F.17 Breakthrough — MMQ stream_K root cause (2026-05-16)

## TL;DR

The "NP=2 multi-step decode race" symptom is, at its actual root, a
**shape-dependence in MMQ's stream_K reduction** at prefill M (>96). Stream_K
distributes work across nsm CTAs and fixes up partial results with an
accumulation order that depends on `ntiles_x = ceil(M / mmq_x)`. Different
M values (215, 430, 860, 1720) thus produce different floating-point
outputs.

Fixing this (`GGML_CUDA_MMQ_DISABLE_STREAM_K=1`) makes slot 0 of NP=2
byte-identical to NP=1 across 10/10 runs. Slot 1 still has a residual
intermittent race (CY.F.18) that surfaces because stream_K's noise was
previously masking it.

## What we did this session

1. Built a multi-step decode test (`test-cy-np2-multi-step-decode`) that
   reproduces the cross-NP non-determinism without an HTTP server. Long
   prompt (215 tokens), 20 decode steps, NP=2 batched.
2. With singlewarp + LLAMA_TEST_SERIAL_PREFILL=1 (each slot prefills in
   its own llama_decode call), NP=2 matched NP=1 byte-identically.
3. With batched prefill (default), NP=2 != NP=1 starting at step 2.
4. Therefore the bug is in BATCHED PREFILL — NP=2 hits a code path that
   serial prefill avoids.
5. The difference: serial prefill runs MMQ at M=n_prompt=215 (per slot).
   Batched prefill runs MMQ at M=2*n_prompt=430 (one call).
6. Extended `test-mmq-q4-0-ar16-shape-invariance-prod-dim` to include
   M ∈ {2, 215, 430, 860, 1720}. CY.F.1 only tested up to M=96.
7. Result: M ≤ 96 byte-identical (CY.F.1 was correct in its range);
   M ≥ 215 differs from M=1 AND from each other.

   | M    | bytes differ from M=1 |
   |------|-----------------------|
   | 1    | 0/5120 (ref)          |
   | 2    | 0/5120 ✓              |
   | 96   | 0/5120 ✓              |
   | 215  | 3774/5120             |
   | 430  | 4348/5120             |
   | 860  | 4624/5120             |
   | 1720 | 4753/5120             |

   Pairwise within prefill regime: M=430 vs M=215 = 1930/5120 differ.

8. Source-read MMQ dispatcher (`ggml/src/ggml-cuda/mmq.cuh`). Both
   `mul_mat_q_case` (line 4439) and `launch_mul_mat_q` (line 4389) check
   `use_stream_k = cc >= CC_VOLTA && cc < CC_OFFSET_AMD`. When true:
   - Launches `mul_mat_q<<<block_nums_mmq, ...>>>` with `nsm` CTAs
     (constant), each doing a slice of all tiles
   - Calls `mul_mat_q_stream_k_fixup` to combine partial results
   - The fixup accumulates partial sums; combine order depends on
     `ntiles_x = ceil(M / mmq_x)`, which is M-dependent

9. Added env-gate `GGML_CUDA_MMQ_DISABLE_STREAM_K=1`. When set, falls
   through to vanilla blocked GEMM: `block_nums_xy_tiling = (nty, ntx, 1)`,
   one CTA per output tile, no cross-CTA accumulation.

10. Verification:

    | Test                                     | Before | After |
    |------------------------------------------|--------|-------|
    | MMQ shape-invariance prefill M (test)    | FAIL   | PASS  |
    | NP=2 slot 0 == NP=1 (test-cy-np2)        | 0/10   | 10/10 |
    | NP=2 slot 1 == NP=1                      | 0/10   | 2/10  |

11. Slot-1 race characterization (10 runs):

    ```
    slot 1 divergence patterns:
      step  2 — 1 occurrence
      step  4 — 1 occurrence
      step 12 — 5 occurrences (2 with identical tokens, 3 separate)
      step 19 — 1 occurrence
      pass    — 2 occurrences
    ```

    Discrete race outcomes, not random ULP noise. The runs with identical
    "step 12 tokens" suggest a deterministic-per-race-outcome trajectory.

12. Without Hadamard (`k/v_cache_hadamard = false`), NP=2 deterministically
    diverges from NP=1 at step 2 (both slots, 5/5 runs, slot 0 == slot 1).
    Hadamard mitigates the residual bug for slot 0; what remains is a
    slot-position-asymmetric race for slot 1.

## What's open (CY.F.18)

Slot 1 intermittent race after CY.F.17 stream_K fix. Suspects (plausibility-ranked):

1. Hadamard K/V cache write — per-slot rotation kernel may race on slot-1
   region. (Hadamard is the binding observation: disable → no race.)
2. DeltaNet per-slot recurrent state buffer — slot-1's buffer may have
   uninitialized memory read or unsynchronized cross-stream write.
3. K-cpy at batched n_tokens=2 — order of slot-0 vs slot-1 cell writes
   may race with downstream FA read (less likely; same ggml_cpy op).
4. Multi-GPU reduce stream synchronization — slot-1's per-device output
   may not be flushed before reduce. CY.F.16 Option A forced F32 reduce
   but didn't add stream sync barriers.

Next probes:
- Disable peer access (LLAMA_CUDA_DISABLE_PEER_ACCESS or similar) to test
  cross-GPU race hypothesis.
- Capture slot 0 vs slot 1 K cache content at end of batched prefill via
  cb_eval. Compare position-by-position; if K[c=0..214] != K[c=215..429]
  byte-identically when prompts are identical, K-write/Hadamard is the source.
- Run with stream synchronization debug (CUDA_LAUNCH_BLOCKING=1) to test
  if forcing serial execution stabilizes slot 1.

## Production impact

- Production single-slot profile (np=1) unaffected — no batched prefill.
- Production np=2 server config: slot 0 reliable, slot 1 racy. Net pass
  rate is min(slot0, slot1) so production NP=2 multi-slot serving is
  still not byte-deterministic.
- Recommendation: ship np=1 as production (current state).
  Track CY.F.18 separately; gate multi-slot enable behind its closure.

Files at commits:
  `ik_llama.cpp@aa7b1eb7` (production/2026-q2-next)
  `yarn-agentic@be71cd2` (production/2026-q2-next)
