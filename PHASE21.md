# Phase 21: Dispatch Reduction for Hybrid Mamba Token Generation

## Status: COMPLETE

## Problem

After Phase 20 eliminated all CPU-fallback ops for Qwen3.5-35B-A3B, profiling (`GGML_VK_PERF_LOGGER=1`) revealed the tg bottleneck was dispatch overhead, not compute:

- **RDNA2**: 41 ms CPU dispatch overhead across 810 dispatches (~50 us each)
- **Vega**: GPU pipeline stalls of 100-250x between dependent dispatches (CONCAT 2540 us vs 17 us on RDNA2 for the same 2 MB copy)

## Tier 1: Inplace State Writeback

DELTA_NET previously wrote SSM state to a temporary buffer, then a CONT+CONCAT+CPY chain copied it into the KV cache. Added `STATE_INPLACE` shader variants that write directly to the KV cache buffer (`src[5]`), eliminating the three-op chain per SSM layer.

Result: Vega GPU stall total 198.8 → 69.5 ms (-65%).

## Tier 2: GGML_OP_FUSED Framework

Ported the op-fusion framework from the `phase25-decode-perf` branch. Single `GGML_OP_FUSED` enum with `fusion_id` dispatch. First fusion:

- **GATE_PREP** (add + softplus + mul → 1 dispatch): saves 48 dispatches per token
- **SILU_MUL** stub: CPU kernel + Vulkan pipeline reusing existing `fused_mul_silu`

## Results

Qwen3.5-35B-A3B on 6800 XT:

| Metric | Phase 20o | Phase 21 | Change |
|---|---:|---:|---|
| pp256 t/s | 146.41 | ~157 | +7% |
| tg64 t/s | 18.18 | ~18.5 | +1.8% |

The 2.5x tg gap to dense models of similar active-parameter count is architectural (2x more ops per token for Mamba layers), not an optimization bug.

## Dead End: Megakernel/JIT

The `polaris-jit` branch's megakernel approach (fusing entire layer subgraphs into single dispatches via runtime JIT) was tested and confirmed 12% slower than standard dispatch. Op-level fusions are the proven pattern.

## Verify by

- RDNA2 pp +7%, tg +1.8% confirmed
- All backend-ops tests pass on both GPUs
- No regressions on dense models
