# Phase 20: Vulkan Token Generation Performance

## Status: COMPLETE (hybrid Mamba/MoE ops) — dense-model MMVQ deprioritized

## Problem

Vulkan token generation was 2-6x slower than ROCm on identical hardware for dense models, and 150-720x slower for hybrid Mamba/MoE models due to missing ops forcing CPU fallback.

### Starting baselines (6800 XT)

| Model | ROCm tok/s | Vulkan tok/s | Gap |
|-------|-----------|-------------|-----|
| TinyLlama 1.1B Q2_K | 380 | 62 | 6.1x |
| Llama-2-7B Q8_0 | 69 | 31 | 2.2x |
| Qwen3.5-35B-A3B IQ3_XXS | — | 0.31 | 322 graph splits |

## Sub-phases

### 20a: Subgroup Reduction (done, marginal)

Added `subgroupAdd` reduction variants to mul_mat_vec shaders, replacing shared memory tree reduction. Performance impact negligible — the reduction step is not the bottleneck.

### 20c: f16acc mul_mat_vec for Vega RPM

Exploited Vega's Rapid Packed Math (f16vec2 `v_pk_fma_f16` instructions) for the dequant mul_mat_vec path. Inner FMA loop uses f16vec2 packed arithmetic; accumulation and reduction in f32. ISA confirmed: 1175 `v_pk_fma_f16` instructions emitted on Vega.

### 20g: MoE Root Cause Analysis

Profiled Qwen3.5-35B-A3B-UD-IQ3_XXS. Found **322 graph splits per token** — caused by 6 missing/broken Vulkan ops in the hybrid Mamba/MoE layers. GPU compute was healthy (~15 ms/token); the bottleneck was CPU fallback sync overhead (~3.2 sec/token). Initial hypothesis blamed `GROUPED_TOPK` — later proved wrong (Qwen3.5 uses `ARGSORT` for routing, not `GROUPED_TOPK`).

Real breakdown of CPU ops per inference graph:

| Count | Op | Status |
|---|---|---|
| 120 | `L2_NORM` | Missing |
| 80 | `MUL_MULTI_ADD` | Missing |
| 60 | `UNARY` (softplus) | Missing |
| 60 | `SSM_CONV` | Missing |
| 60 | `DELTA_NET` | Missing |
| 40 | `FUSED_MUL_UNARY` | Shape mismatch in supports_op |

### 20h: MOE_FUSED_UP_GATE Vulkan Shader

Extended the existing dense `mul_mm_fused_up_gate.comp` with a `MUL_MAT_ID` build flag for the MoE path. Backported full IQ family (IQ1_S..IQ4_XS) into the dense fused_up_gate gen. Fixed a buffer subbuffer scoping bug where multi-token MoE dispatches read zeros for tokens > 0 because the B binding only covered one token's worth of data.

Result: pp256 0.71 → 3.47 t/s (fused MoE now at parity with unfused).

### 20i: GROUPED_TOPK Shader

Single fused compute shader, one workgroup per token row, all stages in shared memory. Correct and needed for DeepSeek-V3/BailingMoE-style models, but did NOT move the Qwen3.5 needle (wrong root cause in 20g).

### 20j–20m: Qwen3.5 Op Ports

| Sub-phase | Op | Source | CPU instances cleared | Splits |
|---|---|---|---|---|
| 20j | `L2_NORM` (non-contig) | Upstream PR #19604 — uncommented local shader + replaced contiguous-only variant | 120 | 322 → 262 |
| 20k | `UNARY` SOFTPLUS | Upstream PR #17319 — new pipeline arrays + CREATE_UNARY wiring | 60 | 262 → 202 |
| 20l | `MUL_MULTI_ADD` | Greenfield (ik-only op) — one workgroup per (k_block, token) | 80 | 202 → 122 |
| 20m | `FUSED_MUL_UNARY` broadcast | Local fix — new `fused_mul_sigmoid.comp`, BCAST define for scalar-broadcast, supports_op extended for `nelements(src0) == 1` | 40 | 122 (compute moved to GPU, no split reduction — ops were adjacent to SSM CPU stretches) |

### 20n: SSM_CONV Full Coverage

Ported all 5 SSM_CONV CUDA kernels from ik PR #1251 to Vulkan. Three dispatch tiers:

- **Single-sequence fast path**: 3 SPVs, parallel over (row, token)
- **Multi-sequence slow path**: 6 SPVs, serial per-row with recurrence/fanout handling
- **Multi-sequence unique-fast path**: 7 SPVs, GPU-side `fast_path_ok` atomic flag

Total: 6 shader files, 16 SPV variants, ~700 LOC.

Result: splits 122 → 62, pp256 3.47 → 12.69 t/s (+266%).

### 20o: DELTA_NET

Custom Vulkan shader for ik fork's 6-arg `ggml_delta_net` (transposed state layout, different from upstream `ggml_gated_delta_net`). One workgroup per (head, seq); each thread holds one row of state in registers. HEAD_DIM baked at SPV-gen time (64, 128). Two reduction strategies: shmem (universal) and subgroup-add (Vega wave64 + h64).

Result: splits 62 → **2**, pp256 12.69 → **146.41 t/s**, tg64 0.46 → **18.18 t/s**.

### 20p: f16acc Overflow Fix

Phase 20c's f16acc set `FLOAT_TYPE=float16_t`, causing the entire accumulation chain to be f16. Q4_K/Q5_K/Q6_K overflowed (6-bit or 8-bit scale × dot exceeds f16 range). Fix: promoted 3 scale-multiply sites to `float`, changed `FLOAT_TYPE=float` for all f16acc variants. Inner f16vec2 FMAs unchanged (1175 `v_pk_fma_f16` confirmed). 32 new stress tests (B in [-50,50]) for 8 quant types.

## Results

### Qwen3.5-35B-A3B-UD-IQ3_XXS on 6800 XT

| Metric | Before Phase 20 | After Phase 20 | Improvement |
|---|---:|---:|---|
| pp256 t/s | 0.71 | 146.41 | 206x |
| tg64 t/s | 0.31 | 18.18 | 58x |
| Graph splits/tok | 322 | 2 | -99.4% |

### Deployment (Q4_K_M, dual-GPU layer-split)

| Metric | Value |
|---|---|
| pp256 | 117 t/s |
| tg64 | 11.3 t/s |
| Graph splits | 3 |

### Remaining bottleneck

82% of wall time is CPU dispatch overhead (1482 dispatches x ~51 us each). GPU compute is only 16 ms. The Vulkan ops themselves (DELTA_NET 19 us, SSM_CONV 8 us, fusions 4 us) are negligible.

## MMVQ Investigation (deprioritized)

The original MMVQ port for dense-model DP4A acceleration hit NaN on every dispatch despite byte-identical shader SPIR-V to upstream. Extensive debugging ruled out shader code, Q8_1 format, NUM_ROWS, reduction variant, descriptor range, and binding count. Remaining suspects were fork buffer management patterns and push constant struct layout. Deprioritized when focus shifted to hybrid Mamba/MoE op coverage, which delivered far larger gains.

## Test Coverage

- 1309+ backend-ops tests pass on both Vega 64 and 6800 XT
- 19 DELTA_NET, 13 SSM_CONV, 7 GROUPED_TOPK, 40 MOE_FUSED_UP_GATE cases
- 32 f16acc stress tests (Q4_K/Q5_K/Q6_K overflow regression coverage)
