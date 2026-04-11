# Benchmark Expectations (sanity check reference)

## Build Status

| Binary | Target | Path | JIT Status |
|--------|--------|------|------------|
| Fork Vulkan | RADV (both GPUs) | `ik_llama.cpp/build/bin/llama-cli` | Ready (no JIT needed) |
| ROCm 6800 XT | gfx1030 only | `llama.cpp/build-hip-navi/bin/llama-cli` | Ready (JIT cached) |
| ROCm Vega | gfx900 only | `llama.cpp/build-hip-vega/bin/llama-cli` | Ready (JIT cached after ~9h first run) |
| ROCm dual | gfx900+gfx1030 | `llama.cpp/build-hip/bin/llama-cli` | 6800 XT cached, Vega JIT in progress |

### ROCm JIT compilation notes

HIP/ROCm compiles GPU kernels on first invocation. This is a one-time cost per binary+target combination:
- gfx1030 (RDNA 2): ~20-30 minutes first run
- gfx900 (GCN 5): ~9 hours first run (pathologically slow)
- Kernel cache persists between runs of the same binary
- Cache does NOT transfer between different binaries (single-target vs dual-target)
- Dual-target builds (`gfx900;gfx1030`) JIT both targets even when only one GPU is visible

## Hardware bandwidth limits

| GPU | Bandwidth | Compute | VRAM |
|-----|-----------|---------|------|
| RX 6800 XT (RDNA 2) | ~512 GB/s GDDR6 | ~20 TFLOPS | 16 GB |
| Vega 56/64 (GCN 5) | ~484 GB/s HBM2 | ~12 TFLOPS | 8 GB |

Token generation is memory-bandwidth bound: tok/s ≈ bandwidth / model_bytes.
Prompt eval is compute-bound for large batches.

## Theoretical token gen ceilings

| Model | Size | 6800 XT ceiling | Vega ceiling |
|-------|------|-----------------|--------------|
| TinyLlama Q2_K | 461 MiB | ~1060 tok/s | ~1000 tok/s |
| Llama-2-7B Q8_0 | 6.7 GiB | ~73 tok/s | ~69 tok/s |
| Llama-2-13B Q8_0 | 13 GiB | ~38 tok/s | WON'T FIT |

Real numbers are 50-80% of theoretical due to kernel launch overhead,
KV cache reads, and non-matmul ops.

## Expected single-GPU Vulkan (fork)

Based on old benchmarks — Phase 0 fixes (push constants, IQK scalar
fallbacks) shouldn't affect Q2_K/Q8_0 performance. Numbers should be
similar to pre-Phase 0 values:

| Config | Prompt tok/s | Gen tok/s | Old value |
|--------|-------------|-----------|-----------|
| TinyLlama / 6800 XT | ~2000-2500 | ~80-95 | 2372 / 91 |
| TinyLlama / Vega | ~1200-1500 | ~70-80 | 1354 / 76 |
| Llama-2-7B / 6800 XT | ~700-800 | ~35-42 | 767 / 38.5 |
| Llama-2-7B / Vega | ~400-450 | ~28-32 | 416 / 30.2 |
| Llama-2-13B / 6800 XT | ~400-500 | ~24-28 | 462 / 26.6 |
| Llama-2-13B / Vega | N/A (OOM) | N/A | N/A |

## Expected single-GPU ROCm (upstream)

ROCm typically matches or slightly beats Vulkan on AMD due to more
mature rocBLAS kernels and native ISA compilation. Expect:

| Config | Prompt tok/s | Gen tok/s | Notes |
|--------|-------------|-----------|-------|
| TinyLlama / 6800 XT | ~2500-3500 | ~80-100 | ROCm well-optimized for RDNA 2 |
| TinyLlama / Vega | ~1000-2000 | ~60-80 | gfx900 less optimized in ROCm |
| Llama-2-7B / 6800 XT | ~800-1200 | ~35-45 | |
| Llama-2-7B / Vega | ~400-600 | ~25-35 | |
| Llama-2-13B / 6800 XT | ~500-700 | ~25-30 | |
| Llama-2-13B / Vega | N/A (OOM) | N/A | |

## Expected multi-GPU Vulkan (fork, dmabuf)

From old Phase 12 benchmarks — multi-GPU with dmabuf eliminates most
transfer overhead. Expect similar or slightly better than old numbers:

| Config | Prompt tok/s | Gen tok/s | Old value |
|--------|-------------|-----------|-----------|
| TinyLlama / both 1:1 | ~1000-1100 | ~90-100 | 1043 / 97 |
| Llama-2-7B / both 1:1 | ~350-400 | ~38-42 | 356 / 39 |
| Llama-2-13B / both 1:1 | ~200-250 | ~20-22 | 214 / 21 |
| Llama-2-13B / both 2:1 | ~220-260 | ~19-21 | 234 / 20 |

## Red flags (investigate if seen)

- Any single-GPU number >2x off from expected → measurement error or regression
- ROCm slower than Vulkan by >30% → possible gfx target or driver issue
- Multi-GPU faster than single-GPU for 7B → suspicious (model fits on one GPU)
- Token gen >theoretical ceiling → measurement error (too few tokens, timing noise)
- Prompt eval <100 tok/s on 6800 XT for any model → something very wrong
- Vega numbers >6800 XT → device assignment reversed
