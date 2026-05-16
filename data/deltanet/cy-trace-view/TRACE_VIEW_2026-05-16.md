# Up-to-date trace view — post Phase A/B/C/CX/CY (2026-05-16)

## Environment

```
LLAMA_FATTN_PER_SLOT_KV_ENABLE=1
LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH=1
LLAMA_PSKV_MODE=singlewarp
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Model: Qwen 3.6 27B IQ4_KS / Q4_0_AR16 lossless target.
Hardware: 2× Quadro RTX 6000 (TU102, sm_75, 24 GiB each), CUDA 13.2.
Decode setup: 13-token prompt, single decode step.

## Cross-NP byte-identity at layer-0 residual (slot 0)

| Pair | First divergent layer | max\|Δ\| at first |
|---|---|---|
| **NP=1 vs NP=2** | layer 6 (DeltaNet) | 3.815e-06 |
| **NP=1 vs NP=4** | layer 6 (DeltaNet) | 3.815e-06 |
| **NP=1 vs NP=8** | layer 6 (DeltaNet) | 3.815e-06 |
| **NP=2 vs NP=4** | NONE (byte-identical through 64 layers) ✓ | 0 |
| **NP=2 vs NP=8** | layer 20 | 1.335e-04 |
| **NP=4 vs NP=8** | layer 20 | 1.335e-04 |

## Three isoclusters identified

```
ISOCLUSTER A: NP=1                                  (fast path)
ISOCLUSTER B: NP=2, NP=4    (byte-identical pair)   (slow path, narrow)
ISOCLUSTER C: NP=8                                  (slow path, wide)
```

### Cluster A vs B (layer 6 boundary)

Source: `all_same_seq` bifurcation in `delta_net::build_layer_attn_linear()` (`src/llama-delta-net.cpp:679`).

- NP=1: `all_same_seq=true` → single call to `build_layer_attn_linear_core` with the whole batch (ne[1]=1).
- NP≥2: `all_same_seq=false` → per-block split, one call per block (each ne[1]=1), `ggml_concat` outputs to [5120, N].

Confirmed via CY.B.1 intra-layer-0 capture: at layer 0, ALL DeltaNet internal tags (`q_in`, `k_in`, `v_in`, `q_fused`, `delta_net_fused_raw`, `new_state`) are byte-identical NP=1 inst0 vs NP=4 inst0 for slot 0. The divergence first manifests at layer 6 (3 DeltaNet layers and 1 FA layer downstream of layer 0).

Working hypothesis: the FFN that consumes the concatenated `[5120, N]` tensor produces a different slot-0 output column than the FFN that consumes a `[5120, 1]` tensor. MMQ at ne11=1 (NP=1) vs ne11∈{2,4} produces different slot-0 column even though MMQ mmq_x_best=8 for both. Investigated below.

### Cluster B vs C (layer 20 boundary)

Source: ne[1]=N MMQ template/kernel boundary. At NP=8 the FFN sees `[5120, 8]` (slow path with 8 blocks concatenated). MMQ at ne11=8 produces different slot-0 column than MMQ at ne11∈{2,4}.

Same mmq_x_best=8 by code analysis. But empirically diverges.

### Within-cluster B (NP=2 vs NP=4): byte-identical ✓

Both NP=2 (FFN sees `[5120, 2]`) and NP=4 (FFN sees `[5120, 4]`) produce byte-identical slot-0 output through all 64 layers. This is the most important green: the MMQ kernel IS byte-identical between ne11=2 and ne11=4 for the slot-0 column.

## Layer-by-layer drift magnitude (cumulative through 64 layers)

NP=1 vs NP=4:
```
layer  6:  3.8e-06 (DeltaNet first drift)
layer  7:  8.7e-04 (FA layer amplifies 200×)
layer  8:  4.3e-03
layer 19:  2.6e-01 (FA layer)
layer 31:  1.1e-01
layer 63:  3.6e+00 (final residual; argmax flip possible)
```

NP=2 vs NP=8:
```
layer 19:  OK (byte-identical 0..19)
layer 20:  1.3e-04 (first drift)
layer 22:  3.0e-02 (FA layer amplifies)
layer 31:  1.8e-01
layer 63:  7.8e+00
```

## What's closed by Phase A/B/C/CX/CY work

- **Intra-NP byte-identity** at FA (singlewarp) — V3 / V4 / V5 results stand.
- **NP=2 vs NP=4 cross-NP** — fully byte-identical at slot 0. The pair (slow path, ne[1]=2 vs 4) is the most-likely production target.
- **Layer-0 DeltaNet kernel itself** — byte-identical NP=1↔NP=4 slot 0 at all internal tags (`q_in`..`new_state`).
- **MMQ at ne11∈{2,4}** — byte-identical slot-0 output (empirical, from layers 0-63 NP=2 vs NP=4 all OK).

## What's open

1. **MMQ at ne11=1 vs ne11∈{2,4}**: produces different slot-0 output despite force_shape_invariant. The dispatch picks `is_gemv=false` for both with shape_invariant on, so both go through `quantize_mmq_q8_1_cuda` + `mul_mat_q`. Yet slot-0 column differs by 3.8e-06 at the first downstream FFN.

   **Likely cause**: at ne11=1 the slow path's `ggml_concat` is a no-op (only one block). But at NP=1 the build graph never enters the slow path — `cur` is just the layer-0 residual at ne[1]=1, and the FFN at layer 6 sees ne[1]=1 too. At NP≥2 the FFN sees ne[1]≥2 via concat. So the MMQ kernel is called with different ne[1] values: 1 vs ≥2.

   This is the **MMQ at ne11=1 vs ne11=N invariance gap**, NOT the MMVQ vs MMQ dispatch gap (which is already closed by SHAPE_INVARIANT_DISPATCH).

2. **MMQ at ne11=8 vs ne11∈{2,4}**: layer 20 boundary suggests another internal threshold. Same mmq_x_best=8 by code, but empirically diverges.

## Next subprobe (CY.B.2)

Unit test that drives MMQ at ne11∈{1,2,4,8} with the same first column and compares slot-0 (column 0) output. ~50 LOC.

If FAIL (output differs): MMQ at ne11=1 and ne11=8 are NOT byte-identical to ne11∈{2,4} for the slot-0 column. The fix is either:
- Force MMQ to always use ne11=4 (round up; waste 1-3 columns of compute) — ~25% decode-rate hit at NP=1.
- Find the per-column-independent reduction order inside MMQ and ensure it's truly column-independent.

If PASS: source is elsewhere. Look at elementwise SiLU/mul, ffn_up_gate fusion, or RMSNorm at ne[1]=1 vs ne[1]≥2.

## Production rollout impact

The NP=2/NP=4 byte-identical pair is the strongest empirical anchor. If we can fix the MMQ ne11=1 vs ne11≥2 boundary, **NP=1 joins cluster B**: NP={1,2,4} byte-identical. That's 7/8 production slot-count combinations covered. NP=8 closure requires the layer-20 boundary fix in addition.
