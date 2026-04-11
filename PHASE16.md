# Phase 16: Backend-Ops Testing and Bug Fixes

## Problem

Phases 13 and 15 added three custom Vulkan ops — `FUSED_UP_GATE`, `MULTI_ADD`, and `MUL_MULTI_ADD` — but test coverage was thin (4 quant types for FUSED_UP_GATE, 4 basic MULTI_ADD cases) and exercised only the happy path. Running the test framework against real hardware exposed two latent bugs: an infinite hang on empty graphs and silent data corruption in MULTI_ADD.

## Bugs Found

### 1. Empty-Graph Fence Hang

**Symptoms**: `test-backend-ops` hung forever after the first computed op. `strace` showed `DRM_IOCTL_SYNCOBJ_WAIT` spinning at 100% CPU. GPU reported 0% utilization.

**Root cause**: `ggml_backend_vk_graph_compute` (`ggml-vulkan.cpp:11042`) set `compute_pending = true` unconditionally in the deferred-fence path, even when no GPU work was submitted:

```cpp
// BEFORE (bug):
} else {
    ctx->compute_pending = true;
}
```

The test framework's `ggml_backend_compare_graph_backend` iterates over every node in the graph, including sentinel nodes (`GGML_OP_NONE`). These produce single-node graphs with zero dispatches. When `compute_pending` is set without any submitted work, the next `synchronize()` call enters `ggml_vk_wait_for_fence` and spins forever on an unsignaled fence.

**Fix**: Guard with `submit_count > 0`:

```cpp
} else if (submit_count > 0) {
    ctx->compute_pending = true;
} else {
    ggml_vk_graph_cleanup(ctx);
}
```

**Impact**: All standard tests and custom tests can now run. Without this fix, the test framework is unusable on Vulkan.

### 2. MULTI_ADD Descriptor Range Too Small

**Symptoms**: MULTI_ADD returned only expert 0's values for all multi-expert cases. Single-expert (`n_experts=1`) passed. No crashes — silently wrong results.

**Root cause**: `ggml_vk_op_f32` has an incontiguous-ops code path (`ggml-vulkan.cpp:7605`) that recalculates descriptor buffer sizes:

```cpp
if (op_supports_incontiguous) {
    x_sz = ggml_nbytes(src0);   // overwrites any earlier x_sz
    ...
}
```

`MULTI_ADD` was registered as incontiguous (`ggml_vk_op_supports_incontiguous`), so this code overwrote `x_sz` with `ggml_nbytes(src0)`. For a strided `view_2d` (which is how `MULTI_ADD` receives its input from the model), `ggml_nbytes` returns the view's logical byte span — it does not know about the `nadd` expert blocks hidden in the stride.

For example, with `ne0=128, ne1=1, n_experts=6`:
- `ggml_nbytes(src0)` = `128 * sizeof(float)` = 512 bytes (one row of the view)
- Shader reads `6 * 128` floats = 3072 bytes (all expert blocks)
- Descriptor covers only 512 bytes; Vulkan's `robustBufferAccess` returns 0 for out-of-range reads
- Result: `expert_0 + 0 + 0 + 0 + 0 + 0 = expert_0` — only first expert contributes

**Fix**: Override `x_sz` *after* the incontiguous block:

```cpp
if (op == GGML_OP_MULTI_ADD) {
    uint32_t nadd = (uint32_t)dst->op_params[0];
    x_sz = (ne01 > 1 ? src0->nb[1] * (ne01 - 1) : 0)
         + ggml_type_size(src0->type) * ne00 * nadd;
    if (x_buf_offset + x_sz >= d_X->size) {
        x_sz = VK_WHOLE_SIZE;
    }
}
```

This computes the actual byte range the shader accesses: the last row's token offset plus one full expert span (`ne00 * nadd` elements).

**Impact**: MULTI_ADD now produces correct results for all expert counts and token counts.

## Test Expansion

### FUSED_UP_GATE: 50 → 143 tests

Previously tested 4 quant types (Q8_0, Q4_0, Q4_K, Q6_K) × 4 dimension combos × 3 activations = ~50 tests.

Expanded to all 11 supported quant types:
- Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 (block size 32)
- Q2_K, Q3_K, Q4_K, Q5_K, Q6_K (block size 256)
- IQ4_NL (block size 32)

Each type tested with 2 dimension pairs × 2 batch sizes × 3 activations (SILU, GELU, RELU) = 12 tests per type.

Additional edge cases:
- **M=1**: Single output row (K=64 to ensure numerically stable NMSE)
- **M=17, M=33, M=127**: Non-tile-aligned output dimensions
- **N=16, N=32**: Large batch sizes
- **M=512, M=1024**: Medium realistic dimensions
- **M=9216, K=3072**: Nemotron-3-Nano production dimensions

All 143 tests pass with NMSE < 5e-4.

### MULTI_ADD: 0 → 12 tests (custom eval)

The standard `test_case` framework cannot correctly test MULTI_ADD because:
1. The op expects a strided `view_2d` (created from a 3D tensor), not a contiguous 2D tensor
2. `init_tensor_uniform` writes contiguous bytes, corrupting stride gaps

Converted to a custom eval function (same pattern as `test_fused_up_gate`):
- Creates a 3D `experts` tensor `[ne0, n_experts, ne1]` on both backends
- Creates `view_2d` with stride `experts->nb[2]`
- Initializes identical random data on both backends via `ggml_backend_tensor_set`
- Computes on both, compares via NMSE

12 test cases:

| ne0 | ne1 | n_experts | Description |
|-----|-----|-----------|-------------|
| 128 | 4 | 6 | Standard (6 experts, 4 tokens) |
| 128 | 1 | 6 | Single token |
| 256 | 4 | 8 | 8 experts |
| 5120 | 1 | 6 | GLM hidden dim |
| 128 | 1 | 1 | Degenerate (identity) |
| 128 | 1 | 2 | Minimal multi-expert |
| 128 | 32 | 6 | Large batch |
| 128 | 4 | 16 | Many experts |
| 1 | 4 | 6 | Minimal ne0 |
| 3072 | 1 | 6 | Nemotron hidden dim |
| 3072 | 8 | 6 | Nemotron batch |
| 128 | 4 | 32 | Many experts, batch |

All 12 tests pass with NMSE = 0.00e+00 (f32 → f32, exact match).

### MUL_MULTI_ADD: 4 → 12 tests

Expanded the standard framework tests to cover:
- Single expert, single token (degenerate case)
- 2 experts (minimal multi-expert)
- 16 experts with batch
- Nemotron hidden dims (3072)
- Nemotron batch (3072 × 8)
- Minimal ne0 = 1
- Large batch (64 tokens)

## Final Test Results

All tests pass on RADV VEGA10 (AMD Radeon RX Vega, wave64):

| Suite | Count | Status |
|-------|-------|--------|
| Standard backend-ops | 1190/1190 | PASS |
| FUSED_UP_GATE | 143/143 | PASS |
| MULTI_ADD | 12/12 | PASS |
| **Total** | **1345** | **PASS** |

## Debugging Notes

### The `robustBufferAccess` trap

The MULTI_ADD bug was hard to diagnose because Vulkan's `robustBufferAccess` feature (enabled by default on RADV) silently returns zero for out-of-descriptor-range reads instead of crashing. This means:
- The shader runs to completion
- The output looks "reasonable" (correct structure, valid floats)
- But values are wrong because some reads returned 0 instead of actual data

For future reference: if a Vulkan shader produces values that look like only part of the input contributes, suspect the descriptor buffer range (`x_sz`) first.

### View tensor byte size vs data span

`ggml_nbytes()` computes the byte range covered by a tensor's logical dimensions and strides. For a view, this is the *view's* span, not the underlying buffer's span. When a shader reads beyond the view's logical extent (as MULTI_ADD does — it reads `nadd` expert blocks through the stride), `ggml_nbytes` underestimates the required descriptor range.

Any op that reads beyond its src tensor's logical extent through stride manipulation needs a post-hoc `x_sz` override after the incontiguous block in `ggml_vk_op_f32`.

## Files Modified

- `ggml/src/ggml-vulkan.cpp` — fence guard (+4 lines), MULTI_ADD descriptor fix (+8 lines)
- `tests/test-backend-ops.cpp` — MULTI_ADD custom eval, expanded test cases (+182/-29 lines)
