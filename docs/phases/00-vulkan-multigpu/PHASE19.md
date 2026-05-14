# Phase 19: Graph-Split Correctness

## Status: COMPLETE — fixed by Phase 22 (get_tensor_async race fix)

## Problem (historical)

Multi-GPU graph-split mode (`-sm graph`) produced incorrect output with greedy sampling (temp=0). Single-GPU and layer-split modes produced correct output. The issue was pre-existing — both CPU-mediated and dmabuf REDUCE paths produced identical (wrong) output, confirming the bug was upstream of REDUCE.

## Symptoms

```
Single GPU:   "The capital of France isyi..."  ← matches layer-split
Layer split:  "The capital of France isyi..."  ← correct
Graph split:  "The capital of France is Hemddddddddddddddddddd"  ← wrong
```

## Suspected Areas

1. **Weight partitioning**: The scheduler splits weight matrices across devices (~25%/75% by VRAM). If the split points are wrong or the partial sums don't align, REDUCE accumulates incorrect values.

2. **Residual accumulation**: In graph-split, device 1 does `ADD(residual)` before REDUCE. If the residual is applied to the wrong partial sum, the layer output is corrupted.

3. **F16 cast alignment**: The `CPY` op casts partial sums from F32 → F16 before REDUCE. If the cast truncates differently across devices, accumulated error could diverge.

4. **Redundant norm computation**: Both devices compute `FUSED_RMS_NORM` on the same input independently. If the REDUCE result isn't properly broadcast to both devices before the next layer's norm, the inputs diverge.

## Approach

1. Dump intermediate tensors at each REDUCE point for single-GPU vs graph-split
2. Find the first layer where outputs diverge
3. Trace back to the source of divergence (weight split, residual, norm)

## Verification (2026-03-18)

Tested post-Phase 0/22 with `--temp 0` greedy sampling:

```
Single GPU:  " Paris.\n\n2. The capital of the United States is Washington, D.C."
Graph split: " Paris.\n\n2. The capital of the United States is Washington, D.C."
```

Verified on both TinyLlama 1.1B Q2_K and Llama-2-7B Q8_0. Output is identical between single-GPU and graph-split modes. The root cause was the `get_tensor_async` race condition on rBAR devices (Phase 22) — stale data was read before GPU compute completed.
