# Phase 13: Vulkan FUSED_UP_GATE Support

## Problem

ik_llama.cpp's `ggml_fused_up_gate()` creates an `GGML_OP_FUSED_UP_GATE` graph node that fuses two matrix multiplications (up-projection and gate-projection) with an activation function (SiLU) and element-wise gating. The CUDA backend implements this as a fused kernel. The Vulkan backend does not support this op at all.

**Impact**: Every transformer layer in a llama-architecture model hits this unsupported op, causing a GPU→CPU→GPU graph split per layer. This produces 67 splits for 7B (32 layers) and 83 splits for 13B (40 layers), making Vulkan inference catastrophically slow (~1 tok/s for 7B, ~0.5 tok/s for 13B).

## Semantics of FUSED_UP_GATE

```
src[0] = up    (quantized weight matrix, [K, N])
src[1] = gate  (quantized weight matrix, [K, N], same type as up)
src[2] = b     (input, F32, [K, M])
op_params[0] = unary_op (SILU, GELU, or RELU)
op_params[1] = limit    (float, activation clamp; 0.0 = no clamp)

output = mul_mat(up, b) * unary(clamp(mul_mat(gate, b), limit))
```

The fallback path in `ggml.c` (line 7873) decomposes this to `mul_mat + mul_mat + fused_mul_unary` when inputs aren't quantized. For quantized inputs (all real inference), the fused node is created.

## Approach: Fused Dual-Matmul Shader

A new compute shader `mul_mm_fused_up_gate.comp` based on `mul_mm.comp` that processes **both** weight matrices in a single dispatch, sharing the input tile `b` and applying the activation+gating inline.

### How it works

The existing `mul_mm.comp` inner loop:
```
for each K-block:
    load A tile → buf_a (shared memory)
    load B tile → buf_b (shared memory)
    barrier
    accumulate sums += buf_a × buf_b
    barrier
```

The fused version uses doubled `buf_a` for parallel load + accumulation:
```
shared buf_a_up[BM * SHMEM_STRIDE];     // ~8 KB
shared buf_a_gate[BM * SHMEM_STRIDE];   // ~8 KB
shared buf_b[BN * SHMEM_STRIDE];        // ~8 KB

for each K-block:
    load B    → buf_b        // loaded ONCE
    load UP   → buf_a_up     // all three loads overlap
    load GATE → buf_a_gate
    barrier                  // 1 — all tiles ready
    accumulate sums_up   += buf_a_up   × buf_b
    accumulate sums_gate += buf_a_gate × buf_b
    barrier                  // 2 — reads complete before next overwrite
```

At write-out:
```
float gate_val = sums_gate[i];
if (limit > 0.0) gate_val = clamp(gate_val, -limit, limit);
output[i] = sums_up[i] * silu(gate_val)    // or gelu/relu
```

### Resource analysis

- **Shared memory**: `buf_a_up` + `buf_a_gate` + `buf_b` ≈ 25 KB. Well within the 64 KB limit on all target GPUs.
- **Barriers**: 2 per K-block — identical to `mul_mm.comp`. Zero additional synchronization overhead.
- **Registers**: Two accumulator arrays instead of one. Non-coopmat: `sums_up[16]` + `sums_gate[16]` = 32 floats (128 bytes). Well within GPU register limits (256 VGPRs on GCN5, 1024 on RDNA2).
- **Bandwidth**: Input `b` loaded once (halved vs decomposed 3-dispatch approach). Two weight matrices loaded in parallel — unavoidable, same total bytes as two separate matmuls. Net saving: ~33% bandwidth vs decomposed. No intermediate buffer writes.
- **Subgroup size**: Handled via `WARP` specialization constant (constant_id=10), same as `mul_mm.comp`. Set to `device->subgroup_size` at pipeline creation. No `gl_SubgroupSize` needed, zero runtime overhead.
- **Dispatch overhead**: Single dispatch vs 3 in decomposed approach. Eliminates two pipeline barriers and two command buffer records.

### What quant types to support

Start with the types `ggml_fused_up_gate` actually encounters in practice:
- Q8_0 (our benchmark model)
- Q4_0, Q4_1, Q5_0, Q5_1 (common quantizations)
- Q2_K through Q6_K (k-quants)

The shader includes the dequantization code via `#ifdef DATA_A_*` — same as `mul_mm.comp`. The shader generator creates one pipeline per quant type. No new dequant code needed.

### K-split: forced to 1

`mul_mm.comp` supports K-dimension splitting where partial sums are written to a temp buffer and reduced by `split_k_reduce`. The fused shader **cannot** use K-split because `silu(sum_a + sum_b) != silu(sum_a) + silu(sum_b)` — the nonlinear activation must be applied after the full K reduction.

This is not a performance concern. K-split only triggers when M×N tiles underutilize the GPU (`k >= 2048 && m_tiles * n_tiles < shader_core_count / 2`). For FUSED_UP_GATE, N is the FFN hidden dim (11008 for 7B, 13824 for 13B), producing plenty of tiles. The dispatch function forces `split_k = 1`.

### Dequant code: macro-based deduplication

The A-tile load in `mul_mm.comp` is ~470 lines of per-quant dequantization code. Rather than duplicating this verbatim for `buf_a_gate`, a macro parameterizes the target buffer, position variable, and data source:

```glsl
#define LOAD_A_TILE(buf_target, pos_src, data_src, data_src_p16, data_src_p32) \
    // ... all dequant blocks, writing to buf_target ...
```

Called as:
```glsl
LOAD_A_TILE(buf_a_up, pos_a_up, data_a, data_a_packed16, data_a_packed32)
LOAD_A_TILE(buf_a_gate, pos_a_gate, data_a_gate, data_a_gate_packed16, data_a_gate_packed32)
```

This keeps the shader maintainable and ensures both tiles use identical dequant logic.

### Buffer aliases for gate binding

Quant types Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, and IQ4_NL use `data_a_packed16` / `data_a_packed32` aliases on binding 0 for packed reads. The gate binding (3) needs equivalent aliases:

```glsl
layout (binding = 0) readonly buffer A_UP {A_TYPE data_a[];};
#if defined(A_TYPE_PACKED16)
layout (binding = 0) readonly buffer A_UP_PACKED16 {A_TYPE_PACKED16 data_a_packed16[];};
#endif
#if defined(A_TYPE_PACKED32)
layout (binding = 0) readonly buffer A_UP_PACKED32 {A_TYPE_PACKED32 data_a_packed32[];};
#endif

layout (binding = 3) readonly buffer A_GATE {A_TYPE data_a_gate[];};
#if defined(A_TYPE_PACKED16)
layout (binding = 3) readonly buffer A_GATE_PACKED16 {A_TYPE_PACKED16 data_a_gate_packed16[];};
#endif
#if defined(A_TYPE_PACKED32)
layout (binding = 3) readonly buffer A_GATE_PACKED32 {A_TYPE_PACKED32 data_a_gate_packed32[];};
#endif
```

Without these aliases, all packed-read quant types would fail to compile or produce wrong results.

## Implementation Plan

### Step 1: Create `mul_mm_fused_up_gate.comp`

**File**: `ggml/src/vulkan-shaders/mul_mm_fused_up_gate.comp`

Fork from `mul_mm.comp` with these modifications:
1. Add binding 3 for gate weight matrix, with packed buffer aliases (see above).
2. Double the shared memory A tile: `buf_a_up[BM * SHMEM_STRIDE]` + `buf_a_gate[BM * SHMEM_STRIDE]`.
3. Add push constants for `unary_op` (uint32) and `limit` (float). No gate stride needed — `up` and `gate` are guaranteed same shape/type by `ggml_fused_up_gate`, so `stride_a` applies to both.
4. Extract the A-tile dequant code into a `LOAD_A_TILE` macro parameterized by target buffer, position, and data source.
5. Duplicate the accumulator: `sums_up[]` and `sums_gate[]`, plus `cache_a_up[]` and `cache_a_gate[]`.
6. In the K-block loop: load `buf_b`, `buf_a_up`, and `buf_a_gate` (via macro), then accumulate both after a single barrier.
7. At write-out: apply activation clamping (if `limit > 0`) and `output = sums_up * activation(sums_gate)`.
8. Strip all `MUL_MAT_ID`, `COOPMAT`, and K-split codepaths — not needed for the fused shader.

**Verify by**: Shader compiles for all target quant types via `vulkan-shaders-gen`.

### Step 2: Register pipelines in `ggml_vk_load_shaders`

**File**: `ggml/src/ggml-vulkan.cpp`

Add pipeline creation for `mul_mm_fused_up_gate` with same warptile configurations as `mul_mat_*`. One pipeline per quant type × size tier (s/m/l). Use the same specialization constants (BM, BN, BK, WM, WN, WARP, etc.). Pipeline uses 4 bindings instead of 3.

Update `ggml_vk_matmul_shmem_support` to calculate `(2*BM + BN)` instead of `(BM + BN)` when checking the fused pipeline (add a `bool fused` parameter or separate function).

**Verify by**: Pipelines created without errors at startup.

### Step 3: Add `FUSED_UP_GATE` to `supports_op`

**File**: `ggml/src/ggml-vulkan.cpp`, function `ggml_backend_vk_supports_op`

Return true when:
- `src[0]` (up) and `src[1]` (gate) are quantized, same type, same shape
- `src[2]` (b) is F32
- The unary op is SILU, GELU, or RELU
- The quant type has a registered `mul_mm_fused_up_gate` pipeline

**Verify by**: Model loads with 2-3 graph splits instead of 67+.

### Step 4: Implement dispatch in `ggml_vk_op_fused_up_gate`

**File**: `ggml/src/ggml-vulkan.cpp`

New dispatch function that:
1. Selects the right pipeline based on quant type and matrix dimensions (s/m/l)
2. Binds all 4 buffers (up, b, dst, gate) — binding order: 0=up, 1=b, 2=dst, 3=gate
3. Sets push constants (M, N, K, strides, batch params, unary_op, limit)
4. Forces `split_k = 1` — no K-split for fused activation
5. Dispatches workgroups same as `mul_mat`

Wire into `ggml_vk_build_graph` and the op dispatch switch.

**Verify by**: Llama-2-7B Q8_0 runs on Vulkan with 2-3 graph splits, produces correct output.

### Step 5: Handle `get_op_mem`

No scratch buffers needed — the fused shader writes directly to `dst`. Return 0.

**Verify by**: No allocation failures.

### Step 6: Benchmark

Compare before (67 splits, ~1 tok/s) vs after (2-3 splits) on:
- Llama-2-7B Q8_0, single GPU (6800 XT)
- Llama-2-13B Q8_0, multi-GPU (6800 XT + Vega)

**Expected**: 10-100x improvement in token generation speed from eliminating CPU bouncing. Fused shader should also be faster than 3-dispatch decomposed approach due to shared `b` tile loads and no intermediate buffers.

## Wave32 vs Wave64

Handled identically to `mul_mm.comp`:
- `WARP` is specialization constant (constant_id=10), set to `device->subgroup_size` at pipeline creation
- `NUM_WARPS = BLOCK_SIZE / WARP` adapts workgroup partitioning automatically
- No `gl_SubgroupSize` used, zero dynamic overhead
- Works on RX 6800 XT (wave32), Vega (wave64), and NVIDIA (wave32) without changes

## Files Modified

- `ggml/src/vulkan-shaders/mul_mm_fused_up_gate.comp` — new shader (forked from `mul_mm.comp`)
- `ggml/src/ggml-vulkan.cpp` — pipeline registration, `supports_op`, dispatch logic, shmem check
- `ggml/src/vulkan-shaders/vulkan-shaders-gen.cpp` — register new shader for compilation

## Shared Memory Budget

The fused shader uses `(2*BM + BN) * SHMEM_STRIDE * type_size` for shared memory (doubled `buf_a` for up and gate). Verified against all target GPUs:

| GPU | Shared Mem | Small (32×32) | Medium (64×64) | Large (128×128) |
|-----|-----------|--------------|----------------|-----------------|
| Polaris 12 (WX 2100) | 32 KB | 6.3 KB | 12.7 KB | 13.1 KB |
| Vega 10 (RX Vega) | 64 KB | 6.3 KB | 12.7 KB | 13.1 KB |
| Navi 21 (6800 XT) | 64 KB | 6.3 KB | 12.7 KB | 13.1 KB |

All configurations fit with room to spare, even on Polaris's 32 KB limit.

## Register Pressure

Non-coopmat accumulator size: `WMITER * TM * WNITER * TN` floats. Doubled for fused shader.

| Warptile | Accumulators (single) | Accumulators (fused) | Total VGPRs (est.) | Vega (256 max) |
|----------|----------------------|---------------------|--------------------|----------------|
| GCN medium (256,64,64) | 2×2×2×2 = 16 | 32 | ~52 | 4 waves/SIMD |
| Small (32×32) | 2×2×2×2 = 16 | 32 | ~52 | 4 waves/SIMD |
| Large (128×128) | 2×4×2×4 = 64 | 128 | ~148 | 1 wave/SIMD |

GCN medium and small warptiles have comfortable occupancy. The large warptile drops to 1 wave/SIMD but is only selected for very large matrices where compute dominates over latency hiding.

## Push Constants

Current `vk_mat_mat_push_constants` = 14 × uint32 = 56 bytes. The fused shader adds:
- `unary_op` (uint32) — SILU=0, GELU=1, RELU=2
- `limit` (float) — activation clamp threshold, 0.0 = disabled

Total: 64 bytes. Well within Vulkan's guaranteed 128-byte minimum.

No gate stride push constant needed — `up` and `gate` are guaranteed same shape/type by `ggml_fused_up_gate` (enforced at graph construction time), so `stride_a` applies to both. Each binding has its own buffer base, and a separate `pos_a_gate` variable tracks the gate tile position.

## Bandwidth Analysis

For Llama-2-7B Q8_0, per FFN layer (K=4096, N=11008):

| Approach | Weight reads | B reads | Intermediate | Dispatches | Barriers |
|----------|-------------|---------|-------------|------------|----------|
| Decomposed (3-dispatch) | 90 MB | 2× | ~44 MB write+read | 3 | 4 |
| Fused (1-dispatch) | 90 MB | 1× | 0 | 1 | 0 |

For token generation (M=1), the B tile is negligible (16 KB). The real win is eliminating dispatch barriers and intermediate buffer I/O.

For prompt eval (M=200), B savings = 200 × 4096 × 4 bytes = 3.2 MB per layer × 32 layers = 102 MB total.

## Risks

- **Shader compilation time**: One new pipeline per quant type × size tier. Adds ~15 pipeline variants. First-run compilation may take a few extra seconds. Cached after that.
- **Register pressure on Vega**: GCN5 has 256 VGPRs per SIMD. 32 accumulator floats + working registers should fit (see table above). If occupancy drops too low, the device will fall back to the "small" warptile (fewer accumulators).
- **Coopmat2 path**: The fused shader supports only the non-coopmat path (scalar accumulation). Coopmat2 fusion is a future optimization — the non-coopmat path already handles all our target hardware (RDNA 2 + GCN 5).
- **Macro complexity**: The `LOAD_A_TILE` macro wrapping ~470 lines of dequant code is large but straightforward — each call site differs only in buffer name, position variable, and data source. No conditional logic in the macro itself.
