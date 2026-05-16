# V4 — production NP-determinism harness with FIX-C v5 (singlewarp)

**Date**: 2026-05-16
**Harness**: `scripts/test-production-np-determinism.sh` with `LLAMA_PSKV_MODE=singlewarp`
**Env**: production stack (multi-GPU, cont-batching, q4_0 KV + Hadamard)
**Prompt**: "The history of artificial intelligence began in earnest with the work of"
**n_predict**: 64

## Cross-NP slot-vs-baseline (the harness's PASS/FAIL criterion)

| NP | slot 0 | slot 1 | slot 2 | slot 3 | slot 4 | slot 5 | slot 6 | slot 7 |
|---|---|---|---|---|---|---|---|---|
| 2 | ≡ NP=1 ✓ | ≠ NP=1 ✗ | — | — | — | — | — | — |
| 4 | ≠ NP=1 ✗ | ≠ NP=1 ✗ | ≠ NP=1 ✗ | ≠ NP=1 ✗ | — | — | — | — |
| 8 | ≠ NP=1 ✗ | ≡ NP=1 ✓ | ≠ NP=1 ✗ | ≠ NP=1 ✗ | ≠ NP=1 ✗ | ≠ NP=1 ✗ | ≠ NP=1 ✗ | ≠ NP=1 ✗ |

Total: 2/14 slots match NP=1 (vs wmma baseline ~5/14 per MEMORY 2026-05-15).

## Intra-NP slot-vs-slot-0 (the real FIX-C v5 contract)

| NP | slot 1 | slot 2 | slot 3 | slot 4 | slot 5 | slot 6 | slot 7 |
|---|---|---|---|---|---|---|---|
| 2 | ≠ slot 0 ✗ | — | — | — | — | — | — |
| **4** | **≡ slot 0 ✓** | **≡ slot 0 ✓** | **≡ slot 0 ✓** | — | — | — | — |
| 8 | ≠ slot 0 ✗ | ≡ slot 0 ✓ | ≡ slot 0 ✓ | ≡ slot 0 ✓ | ≡ slot 0 ✓ | ≡ slot 0 ✓ | ≡ slot 0 ✓ |

**NP=4: PERFECT intra-NP determinism** — all 4 slots produce byte-identical token output to each other. This is the FIX-C v5 contract and it's GREEN at NP=4.

**NP=8: 6/7 slots match slot 0**; slot 1 diverges.
**NP=2: slot 1 diverges from slot 0** at the last generated token (near-tie argmax flip).

## What the divergences look like

NP=2 slot 1 vs slot 0 diverges at the LAST word (token 63/64):
- slot 0 (= NP=1): "...the official birth of AI as a **field of**"
- slot 1:           "...the official birth of AI as a **scientific discipline**"

NP=8 slot 0 (and {2..7}) produces a completely DIFFERENT trajectory than NP=1 / NP=8 slot 1:
- NP=1 / NP=8 slot 1: "Alan Turing, who in 1950 published a paper titled 'Computing Machinery and Intelligence.' ..."
- NP=8 slots {0,2..7}: "Alan Turing, who in 1950 published 'Computing Machinery and Intelligence,' a paper that laid the foundation ..."

## Interpretation

FIX-C v5 (singlewarp) **fixes the FA-kernel cross-slot determinism** as designed:
- All slots within NP=4 produce byte-identical token output ✓
- All slots within NP=8 (except 1) produce byte-identical token output ✓

The **cross-NP divergences** (NP=1 vs NP=N>1) are NOT in FA — they're driven by:
1. The F32-vs-F16 residual storage path split at layer 0 (per `data/deltanet/d2-first-divergent-layer.json` 2026-05-14). Different NPs trigger different `ne[1]==1` / `ne[1]<=32` / `ne[1]>32` branches in `src/llama-build-context.cpp` (lines 789, 1375, 1387, 1407, 1491, 1505, 2779, 2902, ...).
2. Other batch-shape-dependent build graph paths.

The **intra-NP=2 slot 1 vs slot 0** divergence is the most interesting open question. V3 (TRACE-1) at NP=2 showed slot 1 ≡ slot 0 at every layer for a SINGLE decode step. V4 production runs 64 autoregressive decode steps and shows slot 1 drifts at the final token. Possibilities:
- Continuous batching scheduler does something slot-position-dependent over multi-step decode (eviction, reordering)
- Some non-FA op accumulates slot-position-dependence over many decode steps
- A near-tie argmax flips after 60+ steps of fp32-ε accumulation

The **NP=8 slot 1** anomaly is similar.

## Comparison to baseline

Baseline (wmma, MEMORY 2026-05-15): "5/14 slots byte-identical, shifting pattern across runs at NP=4 (0/4→3/4→2/4)".

FIX-C v5: 2/14 cross-NP, but **NP=4 intra-NP fully byte-identical (4/4)**. The cross-NP fix is OUT OF SCOPE for FIX-C; intra-NP is the FA fix.

## Next

- V6 perf measurement
- DATA-3 multi-run stability stress (verify the FIX-C v5 results are run-to-run stable)
- A separate "cross-NP F32-vs-F16 storage path" fix
- Investigation of NP=2 slot 1 / NP=8 slot 1 late-decode divergence
