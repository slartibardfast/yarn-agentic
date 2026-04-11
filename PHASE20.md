# Phase 20: Vulkan Token Generation Performance

## Status: IN PROGRESS — subgroup reduction complete, MMVQ port next

## Problem

Vulkan token generation is 2-6x slower than ROCm on identical hardware.

### Measured performance (6800 XT, 512 GB/s peak)

| Model | ROCm tok/s | Vulkan tok/s | BW utilization | Gap |
|-------|-----------|-------------|----------------|-----|
| TinyLlama 1.1B Q2_K | 380 | 62 | 4.8% | 6.1x |
| Llama-2-7B Q8_0 | 69 | 31 | 42% | 2.2x |

### Measured performance (Vega, 484 GB/s HBM2 peak)

| Model | ROCm tok/s | Vulkan tok/s | Gap |
|-------|-----------|-------------|-----|
| TinyLlama 1.1B Q2_K | 244 | 31 | 7.9x |

## Root Cause: Missing MMVQ (Integer Dot Product) Path

Upstream llama.cpp Vulkan has two mul_mat_vec strategies:

1. **Dequant path** (what the fork uses): dequantize weights → F32, load F32 activations, FMA accumulate
2. **MMVQ path** (missing from fork): load quantized weights directly, quantize activations to Q8_1, use `dotPacked4x8EXT` (DP4A) for 4×int8 per instruction

The MMVQ path reads ~8× less data per element and uses ~4× fewer instructions. On AMD with k≥2048, upstream selects MMVQ for all K-quant types. The 6800 XT supports `VK_KHR_shader_integer_dot_product` (`int dot: 1`).

### Why the dequant path is slow

For Llama-2-7B Q8_0 token generation:
- Total weight reads per token: 7.02 GB
- At 512 GB/s: 13.7 ms theoretical minimum
- ROCm achieves 14.5 ms (95% bandwidth utilization)
- Vulkan dequant path achieves 32.3 ms (42% utilization)

The dequant path wastes bandwidth by expanding quantized weights to F32 in shader registers before multiply-accumulate. MMVQ keeps data in int8 throughout, reading less from memory and using the dedicated DP4A hardware.

### Why small models are even worse

TinyLlama (459 MiB Q2_K) has small weight matrices (~1.4 MB per 2048×2048). Each dispatch can't amortize Vulkan's ~3-5 µs per-dispatch overhead. With 511 graph nodes per token, dispatch overhead consumes ~2 ms of the 16 ms total. ROCm's HIP has ~10× lower dispatch overhead.

## Implementation Plan

### Step 1: MMVQ Shaders (for RDNA2 and other DP4A hardware)

Port upstream's MMVQ shader infrastructure:

| File | Lines | Purpose |
|------|-------|---------|
| `mul_mat_vecq.comp` | ~143 | Main MMVQ shader — loads Q8_1 B data, calls `mmvq_dot_product` |
| `mul_mat_vecq_funcs.glsl` → `.comp` | ~494 | Per-type `repack()` and `mmvq_dot_product()` using `dotPacked4x8EXT` |

Key functions per quant type:
- `repack(ib, iqs)` — load quantized A weights, rearrange for DP4A alignment
- `mmvq_dot_product(ib, iqs)` — `dotPacked4x8EXT(a_packed, b_packed)` + scale correction
- `get_dm(ib)` / `get_d(ib)` — load per-block scale factors

Types to implement (matching upstream AMD heuristic):
- Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 (legacy quants)
- Q2_K, Q3_K, Q4_K, Q5_K (K-quants, k≥2048)
- IQ types where applicable

### Step 2: Q8_1 Quantization Pipeline

Add GPU-side F32 → Q8_1 quantization:
- `quantize_q8_1.comp` shader (upstream already has this)
- Pipeline creation and pre-allocated Q8_1 buffer
- Dispatch Q8_1 quantization before MMVQ mul_mat_vec

### Step 3: C++ Dispatch Logic

Port upstream's `quantize_y` decision:
```
quantize_y = device->integer_dot_product
          && src1->type == GGML_TYPE_F32
          && ggml_is_contiguous(src1)
          && (ne11 * ne10) % 4 == 0
          && ggml_vk_should_use_mmvq(device, ne01, ne11, ne10, src0->type)
```

AMD heuristic from upstream:
- Use MMVQ when k ≥ 2048 for most types
- Q8_0: use MMVQ only on GCN (Vega), not on RDNA2
- Q3_K, Q6_K: skip MMVQ (2-byte alignment issues)

### Step 4: Pipeline Array Extension

Add `pipeline_dequant_mul_mat_vec_q8_1_f32[type][cols]` array alongside existing `_f32_f32` and `_f16_f32` arrays.

### Step 5: Vega (GCN5) — Packed FP16 Path

Vega has NO DP4A hardware (`int dot: 0`). The MMVQ path won't help Vega. Instead, exploit Vega-specific features:

**Rapid Packed Math (RPM):**
- Vega processes two FP16 values per instruction via `f16vec2` (`v_pk_*` ISA)
- Current dequant path works in F32 — switching to FP16 packed math doubles throughput
- Requires `VK_KHR_shader_float16_int8` (supported on RADV)

**Implementation:**
- New `mul_mat_vec_f16pack.comp` variant that:
  - Dequantizes to `float16_t` instead of `float`
  - Accumulates using `f16vec2` packed operations
  - Uses `float` only for the final reduction (avoid precision loss)
- Selected when `!integer_dot_product && fp16` (Vega, Polaris)

**VGPR Budgeting:**
- GCN5 has 256 VGPRs per SIMD unit
- Target ≤32 VGPRs per thread for maximum occupancy (8 wavefronts/SIMD)
- f16vec2 halves register pressure vs F32 — more accumulators fit in 32 regs

**Wave64 Exploitation:**
- Vega's native wavefront is 64 lanes
- Each subgroupAdd covers 64 elements — 2× the work per reduction vs RDNA2's wave32
- Shader should be tuned for 64-wide access patterns (coalesced 64×4B = 256B cache lines)

### Step 6: Benchmark and Tune

For each GPU × quant type combination:
1. Measure tok/s and bandwidth utilization
2. Compare against ROCm baseline
3. Tune NUM_ROWS, BLOCK_SIZE, and K_PER_ITER for optimal occupancy

Target: exceed ROCm on 7B Q8_0 token gen (>69 tok/s on 6800 XT) by combining DP4A efficiency with Vulkan's lower memory overhead.

## Completed Work

### Subgroup Reduction (done, marginal impact)

Added `subgroupAdd` reduction variants to mul_mat_vec shaders, replacing shared memory tree reduction. Requires `require_full_subgroups=true` + `required_subgroup_size`. Correctness verified (926/926 tests pass). Performance impact negligible — the reduction step is not the bottleneck.

### Bandwidth Analysis (done)

Profiled mul_mat_vec with test-backend-ops perf:
- Large matrices (m=128K, k=3K): kernel saturates bandwidth (~2100 GB/s effective)
- Realistic matrices (m=4K, k=4K): ~42% bandwidth on 6800 XT
- Small matrices (m=16, k=256): dispatch-overhead dominated (~9 GB/s)

## Architecture-Specific Strategy Summary

| GPU | Architecture | DP4A | Strategy | Expected Improvement |
|-----|-------------|------|----------|---------------------|
| RX 6800 XT | RDNA2 | Yes | MMVQ (dotPacked4x8EXT) | 2-3× (close to ROCm) |
| RX Vega | GCN5 | No | Packed FP16 (f16vec2 RPM) | 1.5-2× |
| Polaris | GCN4 | No | Packed FP16 (limited RPM) | 1.2-1.5× |
| NVIDIA | Turing+ | Yes | MMVQ (existing upstream) | Already optimized |

## Verify by

- 7B Q8_0 token gen on 6800 XT: >60 tok/s (from 31, target exceeding ROCm's 69)
- TinyLlama Q2_K on 6800 XT: >150 tok/s (from 62)
- TinyLlama Q2_K on Vega: >60 tok/s (from 31, via FP16 path)
- 926/926 backend-ops tests pass on both GPUs
- Perplexity unchanged (no precision regression)

## References

- Upstream MMVQ: `llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vecq.comp`
- Upstream MMVQ funcs: `llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vecq_funcs.glsl`
- Upstream quantize_y: `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` line ~7647
- Upstream should_use_mmvq: same file, search `ggml_vk_should_use_mmvq`
- AMD GCN5 ISA: Rapid Packed Math `v_pk_*` instructions
- Vulkan DP4A: `VK_KHR_shader_integer_dot_product` / `dotPacked4x8EXT`
- Benchmark data: [BENCHMARKS.md](BENCHMARKS.md)

## Debugging Log: MMVQ NaN (2026-03-21)

### Confirmed facts
- Upstream Vulkan MMVQ passes ALL tests on same 6800 XT hardware
- Our fork's MMVQ shader SPIR-V is functionally identical to upstream's
- Every MMVQ dispatch produces NaN; every non-MMVQ dispatch passes
- NaN persists after: x4 quantize format, NUM_ROWS=1, shmem-only reduction, VK_WHOLE_SIZE for Y buffer, 3 bindings instead of 5
- Disabling MMVQ (should_use_mmvq returns false) → 0 failures

### Ruled out
- Shader code difference (byte-for-byte identical)
- Q8_1 buffer format (x4 vs plain — both produce NaN)
- NUM_ROWS spec constant (changed to 1, still NaN)
- Subgroup reduction variant (tried shmem-only, still NaN)
- Y descriptor range (VK_WHOLE_SIZE, still NaN)
- Binding count mismatch (3 vs 5 — dequant also has this, works fine)

### Remaining suspects
1. **Fork's buffer management pattern**: Fork uses `vk_buffer` + explicit offset/size at dispatch. Upstream uses `vk_subbuffer` with `ggml_vk_tensor_subbuffer()`. The implicit conversion to `vk::DescriptorBufferInfo` may differ.
2. **Push constant field mismatch**: Our fork's `vk_mat_vec_push_constants` struct may have different field layout than upstream's. Fields like `fusion_flags` and `base_work_group_y` might be at different offsets.
3. **Pipeline descriptor set layout**: The way the fork creates descriptor set layouts from pipeline bindings may differ from upstream, causing descriptors to be bound to wrong slots.
4. **The quantize_q8_1 dispatch itself**: The Q8_1 data written by the quantize shader may be wrong — need to read back and verify.

### Next step
Do a byte-level comparison of the push constant struct layout between fork and upstream, and verify the Q8_1 quantize output by reading back the buffer.
