# DATA-3 / V5 — multi-run stability stress (singlewarp at NP=8 × 5 runs)

**Date**: 2026-05-16
**Setup**: `test-deltanet-d1-capture` at NP=8 same-prompt with `LLAMA_PSKV_MODE=singlewarp`
**Runs**: 5 back-to-back invocations (separate processes, fresh model load each)
**Total captured**: 8 slots × 64 layers × 5 runs = 2560 files

## Result

**0 / 2560 files diverge across runs**. Every captured residual is sha256-identical across all 5 runs.

## Significance

This closes the open concern from `specs/deltanet/fattn-per-slot-kv-sm75.md §15.17`:

> "At NP>=4: residual divergence is non-deterministic across runs (different
>  slots diverge in different runs with same code). This rules out batching
>  as the cause (which would be deterministic). The remaining source is
>  CUDA-side timing / kernel-internal non-determinism (atomic order,
>  scheduler interleaving, possibly cuda graph cache eviction)."

The single-warp per-row CTA architecture eliminates this:
- Each (token, head, seq) has its own CTA → no inter-CTA dependencies.
- Single warp inside the CTA → no cross-warp reduction order variance.
- Canonical k-loop in fp32 → no scheduler interleaving effects.
- No atomicAdd anywhere in the kernel.

The earlier wmma_f16_case_pb1 had `parallel_blocks=1` already, but its
batched WMMA mma_sync across 8 cols introduced the slot-position-dependent
chunk decomposition (per TRACE-1..6). With singlewarp, each row's k-loop
is independent of other rows' presence in the batch.

§15.17's "run-to-run residual" is RESOLVED at the FA kernel level.
