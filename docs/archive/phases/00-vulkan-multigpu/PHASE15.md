# Phase 15: Vulkan Support for GLM-4.7-Flash (deepseek2 Architecture)

## Problem

GLM-4.7-Flash uses the `deepseek2` architecture (47 layers, SIGMOID expert gating) with Mixture of Experts (MoE). It produces 186 graph splits on Vulkan, making inference catastrophically slow.

The attention path (MLA) and dense FFN layers use only supported Vulkan ops. All 186 splits come from the **MoE FFN layers** due to two fused ops that lack Vulkan support.

## Architecture Overview

- **47 transformer layers** (48 if MTP layer included)
- First `n_layer_dense_lead` layers: dense FFN with SILU (fully Vulkan-supported)
- Remaining ~44 layers: MoE FFN (6 experts selected from 64) + shared expert FFN
- MLA attention with KV compression (all ops supported)

## Unsupported Operations

### 1. `MOE_FUSED_UP_GATE` — 1 per MoE layer

**Graph location**: `llama-build-context.cpp:1149-1157` (via `llm_build_moe_ffn`), triggered by default (`fused_moe_up_gate=true`, `up_exps->type == gate_exps->type`).

**Semantics** (graph construction: `ggml.c:7763-7806`, CPU impl: `ggml.c:17326-17476`):

```
src[0] = as_up    (quantized weights, [K, N, n_expert])
src[1] = as_gate  (quantized weights, [K, N, n_expert], same type; or NULL if interleaved in as_up)
src[2] = b        (input, F32, [K, n_expert_used, n_tokens])
src[3] = ids      (expert indices, I32, [n_expert_used, n_tokens])
src[4] = up_b     (optional bias)
src[5] = gate_b   (optional bias)
op_params[0] = unary_op (SILU/GELU)
op_params[1] = limit (float, activation clamp; 0.0 = disabled)

For each token, for each selected expert:
  output = mul_mat(up_slice, b) * activation(clamp(mul_mat(gate_slice, b), limit))
```

This is the MoE version of Phase 13's `FUSED_UP_GATE`, combining `MUL_MAT_ID` expert routing with fused dual-matmul gated activation.

### 2. `MUL_MULTI_ADD` — 1 per MoE layer

**Graph location**: `llama-build-context.cpp:1213`, triggered by default (`fused_mmad=true`).

**Semantics** (graph construction: `ggml.c:6184-6207`, CPU impl: `iqk_cpu_ops.cpp:422-457`):

```
src[0] = a  (F32, [n_embd, n_expert_used, n_tokens])
src[1] = b  (F32, [1, n_expert_used, n_tokens])
dst         (F32, [n_embd, n_tokens])

dst[token] = sum_over_experts( a[expert, token] * b[expert, token] )
```

Fused multiply-accumulate: weight each expert's output by its routing weight, then sum across the expert dimension.

## Implementation Plan

### Step 1: `MUL_MULTI_ADD` Vulkan shader

**File**: `ggml/src/vulkan-shaders/mul_multi_add.comp`

Each workgroup processes one output row (one token). Inner loop over `n_expert_used` (typically 6) is trivially short.

```glsl
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint token = gl_WorkGroupID.y;
    uint idx = gl_WorkGroupID.x * 256 + gl_LocalInvocationID.x;
    if (idx >= p.ne00) return;

    float sum = 0.0;
    for (uint expert = 0; expert < p.ne01; ++expert) {
        float a_val = data_a[token * p.stride_d + expert * p.stride_a + idx];
        float w_val = data_b[token * p.stride_bz + expert * p.stride_b];
        sum += a_val * w_val;
    }
    data_d[token * p.ne00 + idx] = sum;
}
```

**Performance**: Each thread does `n_expert_used` multiply-adds. Memory access is coalesced across threads within a token row. `n_expert_used` is small enough (6-8) that the loop fully unrolls. No shared memory needed.

**Dispatch**: `ceil(n_embd / 256)` × `n_tokens` × 1 workgroups.

**Compatibility**: No shared memory, no subgroup ops, no extensions. Works on all Vulkan 1.2 hardware including Polaris (GCN 7790+).

### Step 2: Register `MUL_MULTI_ADD` pipeline

**File**: `ggml/src/ggml-vulkan.cpp`

1. Add `vk_pipeline pipeline_mul_multi_add_f32` to device struct
2. Create pipeline in `ggml_vk_load_shaders`: 3 bindings (a, b, dst), push constants for dimensions/strides
3. Add `case GGML_OP_MUL_MULTI_ADD` to `supports_op` (src[0] F32, src[1] F32, dst F32)
4. Wire into `ggml_vk_get_pipeline`, `ggml_vk_build_graph`, dispatch switch
5. Implement `ggml_vk_mul_multi_add` dispatch function

**Verify by**: Run with `fused_moe_up_gate=false` (decomposed matmuls) and confirm MoE layers complete on GPU with this op resolved.

### Step 3: `MOE_FUSED_UP_GATE` Vulkan shader

**File**: `ggml/src/vulkan-shaders/mul_mm_moe_fused_up_gate.comp`

This combines three existing patterns into a single optimal shader:

1. **`MUL_MAT_ID` expert routing** (from `mul_mm.comp:179-235`): Each z-slice (`gl_GlobalInvocationID.z`) processes one expert. The shader scans the `ids` buffer to build `row_ids[]` in shared memory — listing which tokens are routed to this expert. Workgroups with no routed tokens exit early.

2. **Dual A-tile matmul** (from `mul_mm_fused_up_gate.comp`): Two shared memory A tiles (`buf_a_up`, `buf_a_gate`) share one B tile (`buf_b`). The `LOAD_A_TILE` macro loads dequantized weight tiles for both up and gate matrices. Both accumulator arrays (`sums_up`, `sums_gate`) are updated per K-block.

3. **Gated activation** (from `mul_mm_fused_up_gate.comp` write-out): `output = sums_up * silu(clamp(sums_gate, limit))`.

**Bindings**: 5 total

| Binding | Buffer | Description |
|---------|--------|-------------|
| 0 | `as_up` | Quantized up weights [K, N, n_expert] |
| 1 | `b` | F32 input [K, n_expert_used, n_tokens] |
| 2 | `dst` | F32 output |
| 3 | `as_gate` | Quantized gate weights (or alias of binding 0 for interleaved) |
| 4 | `ids` | I32 expert indices [n_expert_used, n_tokens] |

Plus packed buffer aliases for bindings 0 and 3 (same pattern as Phase 13):
```glsl
layout (binding = 0) readonly buffer A_UP {A_TYPE data_a[];};
layout (binding = 0) readonly buffer A_UP_P16 {A_TYPE_PACKED16 data_a_packed16[];};
layout (binding = 0) readonly buffer A_UP_P32 {A_TYPE_PACKED32 data_a_packed32[];};
layout (binding = 3) readonly buffer A_GATE {A_TYPE data_a_gate[];};
layout (binding = 3) readonly buffer A_GATE_P16 {A_TYPE_PACKED16 data_a_gate_packed16[];};
layout (binding = 3) readonly buffer A_GATE_P32 {A_TYPE_PACKED32 data_a_gate_packed32[];};
layout (binding = 4) readonly buffer IDS {int data_ids[];};
```

**Push constants** (fits in 128-byte Vulkan minimum):

```glsl
layout (push_constant) uniform parameter {
    uint M;              // output rows per expert (N)
    uint N;              // number of routed tokens (dynamic via row_ids)
    uint K;              // input dimension
    uint stride_a;       // row stride for weight matrices
    uint stride_b;       // row stride for input B
    uint stride_d;       // row stride for output
    uint batch_stride_a; // expert stride in weight buffer
    uint batch_stride_b; // not used for MoE
    uint batch_stride_d; // not used for MoE
    uint nei0;           // n_expert_used
    uint nei1;           // n_tokens
    uint nbi1;           // ids stride
    uint ne11;           // input B ne[1]
    uint unary_op;       // SILU=0, GELU=1
    float limit;         // activation clamp (0.0 = disabled)
} p;
```

Total: 15 × 4 = 60 bytes. Within the 128-byte minimum.

**Dispatch**: Single dispatch with `z = n_expert`. Each z-slice is one expert. The shader-side routing (inherited from `MUL_MAT_ID`) scans `data_ids` to find tokens routed to this expert and builds `row_ids[]` in shared memory. Workgroups for experts with zero routed tokens exit immediately.

This is the optimal approach — single dispatch, no host-side IDs readback, no per-expert dispatch overhead. Identical to how `MUL_MAT_ID` works in `mul_mm.comp`.

**K-split**: Forced to 1 (same as Phase 13 — nonlinear activation prevents partial-sum reduction).

**Shared memory budget**: `(2*BM + BN) * SHMEM_STRIDE * sizeof(float)` + `row_ids[4096] * 4 bytes`.

| Warptile | buf_a_up + buf_a_gate + buf_b | row_ids | Total | Polaris (32 KB) |
|----------|------------------------------|---------|-------|-----------------|
| Small (32×32) | ~9.5 KB | 16 KB | ~25.5 KB | Fits |
| Medium (64×64) | ~19 KB | 16 KB | ~35 KB | Does NOT fit |
| Large (128×128) | ~19.7 KB | 16 KB | ~35.7 KB | Does NOT fit |

**Polaris (32 KB) compatibility**: Only the small warptile fits. Medium and large warptiles exceed 32 KB due to the doubled A-tiles plus `row_ids[4096]`. The shader selection logic must check shared memory availability and fall back to the small warptile on Polaris. This matches the existing `ggml_vk_matmul_shmem_support` pattern which already gates warptile selection by shared memory size.

Alternatively, `row_ids` can be reduced to `u16vec2[2048]` (8 KB) if `nei0 * nei1 <= 2048`, cutting shared memory enough for medium warptiles on Polaris. The existing `MUL_MAT_ID` path already asserts `nei0 * nei1 <= 4096` (`ggml-vulkan.cpp:5893`).

For Vega (64 KB) and RDNA2 (64 KB), all warptile sizes fit with room to spare.

**Quant types**: Same as Phase 13 — Q4_0 through Q6_K, Q8_0, IQ types. One pipeline per quant type × size tier (s/m/l).

**Interleaved weights** (`as_gate = NULL`): When `as_gate` is NULL, `as_up` contains both up and gate weights with `N/2` rows each. Binding 3 aliases binding 0. The shader uses `pos_a_gate = expert_idx * batch_stride_a + (N/2) * stride_a + ...` to offset into the gate half. A boolean push constant or specialization constant distinguishes interleaved from separate mode.

### Step 4: Register `MOE_FUSED_UP_GATE` pipelines

**File**: `ggml/src/ggml-vulkan.cpp`

1. Add `vk_matmul_pipeline` arrays for `mul_mm_moe_fused_up_gate` (one per quant type × size tier)
2. Create pipelines in `ggml_vk_load_shaders` with 5 bindings, same specialization constants as Phase 13
3. Update `ggml_vk_matmul_shmem_support` to check `(2*BM + BN) * SHMEM_STRIDE + row_ids_size` for the fused MoE pipeline
4. Add `case GGML_OP_MOE_FUSED_UP_GATE` to `supports_op`:
   - `src[0]` (up) quantized, supported type with registered pipeline
   - `src[1]` (gate) NULL or same type/shape as up
   - `src[2]` (b) is F32
   - `src[3]` (ids) is I32
   - `src[4]`, `src[5]` (biases) must be NULL (biases not supported in fused path — falls back to decomposed)
   - Unary op is SILU or GELU
5. Implement `ggml_vk_moe_fused_up_gate` dispatch function:
   - Mirror `ggml_vk_mul_mat_id_q_f16` structure
   - Set up buffers for 5 bindings (up, b, dst, gate, ids)
   - Handle optional dequant for non-contiguous inputs
   - Dispatch with z-dimension = `n_expert` (= `ne02`)
   - Push constants include `unary_op` and `limit`

### Step 5: Handle `get_op_mem`

Neither op needs scratch buffers — both write directly to `dst`. Return 0 for both.

### Step 6: Benchmark

Compare before (186 splits, ~0.01 tok/s) vs after (2-5 splits) on:
- GLM-4.7-Flash Q4_K, single GPU (RX 6800 XT)
- GLM-4.7-Flash, multi-GPU (RX 6800 XT + Vega) if model fits

## Register Pressure

The fused MoE shader doubles the accumulator arrays (same as Phase 13):

| Warptile | Accumulators (fused) | VGPRs (est.) | Vega (256 max) | Polaris (256 max) |
|----------|---------------------|-------------|----------------|-------------------|
| Small (32×32) | 32 floats | ~52 | 4 waves/SIMD | 4 waves/SIMD |
| Medium (64×64) | 32 floats | ~52 | 4 waves/SIMD | N/A (shmem) |
| Large (128×128) | 128 floats | ~148 | 1 wave/SIMD | N/A (shmem) |

Polaris is limited to small warptile by shared memory, which has comfortable register occupancy.

## Performance Analysis

### `MUL_MULTI_ADD`

Arithmetic intensity is very low — one multiply-add per element per expert. This is fully memory-bandwidth bound. The shader does `n_expert_used` (6) reads of `n_embd` floats plus 1 write. For n_embd=5120 (GLM-4.7-Flash hidden dim), that's ~7 × 20 KB = ~140 KB per token. Trivial compared to the MoE matmuls.

### `MOE_FUSED_UP_GATE`

Same bandwidth analysis as Phase 13: the fused shader reads the B tile once (shared between up and gate matmuls), halving B bandwidth vs decomposed 3-dispatch approach. Weight bandwidth is identical (both matrices must be read regardless). The main win is:
- **1 dispatch vs 3** (eliminating 2 pipeline barriers per MoE layer)
- **No intermediate buffer** for the unfused gate output
- **Shared B tile loads** (one load vs two)

For token generation (M=1, small batches), dispatch overhead dominates. 44 MoE layers × 3 dispatches = 132 dispatches eliminated, saving ~132 pipeline barriers.

## Files Modified

- `ggml/src/vulkan-shaders/mul_multi_add.comp` — new shader (~30 lines)
- `ggml/src/vulkan-shaders/mul_mm_moe_fused_up_gate.comp` — new shader (fork of `mul_mm_fused_up_gate.comp` + `MUL_MAT_ID` routing from `mul_mm.comp`)
- `ggml/src/ggml-vulkan.cpp` — pipeline registration, `supports_op`, dispatch, build_graph wiring
- `ggml/src/vulkan-shaders/vulkan-shaders-gen.cpp` — register new shaders for compilation

## Risks

1. **Polaris shared memory**: Only small warptile fits (25.5 KB < 32 KB). Performance on Polaris will be lower than Vega/RDNA2 which can use medium/large warptiles. This is acceptable — Polaris GPUs have limited compute anyway.

2. **Interleaved weight correctness**: When `as_gate=NULL`, the up/gate split point in the weight buffer must align with quant block boundaries. All standard quant types (Q4_0: 32 elements/block, Q8_0: 32, K-quants: 256) have block sizes that divide typical `N` values (e.g., N=10944 for GLM-4.7-Flash). Verify with assertion.

3. **Bias support**: GLM-4.7-Flash does not use expert biases (`src[4]`, `src[5]` are NULL). If a future model has biases, `supports_op` returns false and falls back to the decomposed path. Full bias support can be added later.

4. **Testing model**: Need GLM-4.7-Flash GGUF (~30B total, ~3B active). A Q4_K quantization ~17 GB fits on 6800 XT (16 GB) with partial offload or with multi-GPU.

## Verification

1. Build on remote host: `cmake -S ... -B ... -DGGML_VULKAN=ON && cmake --build build -j16`
2. Download GLM-4.7-Flash GGUF
3. Run with `--verbose` to check graph split count — target: 2-5 splits
4. Compare output tokens with CPU-only run for correctness
5. Benchmark tok/s: expect 10-100x improvement from eliminating 186 graph splits
6. Test on Polaris (small warptile path) to verify GCN 7790+ compatibility
