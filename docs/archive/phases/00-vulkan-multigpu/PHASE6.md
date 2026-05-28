# PHASE 6: Runtime Testing, IQK Fallbacks, and lavapipe Fixes

**Goal**: Make the multi-GPU implementation actually run on available hardware — RADV (Polaris 12) + lavapipe on a Xeon X5650 (no AVX2).

## Status: COMPLETE

## Context

Phases 1-5 built clean but had never been tested at runtime. First test attempts revealed three categories of issues: IQK crashes on non-AVX2 CPUs, missing `extern "C"` linkage on IQK stubs, and Vulkan backend assertions blocking lavapipe (CPU-type device).

## Changes Made

### IQK Scalar Fallbacks (`ggml/src/iqk/iqk_quantize.cpp`)

Two functions used AVX2 intrinsics without `#ifdef __AVX2__` guards, unlike all other AVX2 code in the file which was properly guarded with scalar `#else` paths:

- `quantize_row_q8_0_x4`: replaced empty `#else` block with scalar quantization loop
- `quantize_row_q8_1_x4_T`: replaced empty `#else` block with scalar quantization loop (handles both `block_q8_1` and BF16 variants via `if constexpr`)

The upstream `-mavx2 -mfma -mf16c` compile flag was restored in `CMakeLists.txt`. On AVX2 machines, the upstream fast path is used. The scalar code only activates when building without AVX2.

### IQK Graceful Degradation (`ggml/src/iqk/iqk_mul_mat.cpp`)

The `#else` (non-`IQK_IMPLEMENT`) stubs for 4 functions called `GGML_ABORT("Unsupported CPU")`, which crashes the process instead of allowing the scheduler to fall back to non-IQK implementations:

- `iqk_mul_mat` — changed to `return false`
- `iqk_mul_mat_4d` — changed to `return false`
- `iqk_mul_mat_moe` — changed to `return false`
- `iqk_moe_fused_up_gate` — changed to `return false`

Note: `iqk_moe_fused_up_gate` is called from `ggml_compute_forward_mul_mat_up_gate` in `ggml.c` which has no non-IQK fallback — it still hits `GGML_ABORT("fatal error")` when the function returns false. The workaround is `-no-fug` (disable fused up-gate), which reduces graph splits from 46 to 2.

### IQK Linker Fixes (`ggml/src/iqk/iqk_flash_attn.cpp`, `iqk_mul_mat.cpp`)

Without `IQK_IMPLEMENT` defined (no AVX2), the `#else` stubs had missing or incorrect linkage:

- `iqk_flash_attn.cpp`: added missing `iqk_fa_work_buffer_size` stub, fixed `iqk_flash_attn_noalibi` parameter count (35 → 36, missing `sinks` parameter)
- `iqk_mul_mat.cpp`: added missing `iqk_topk_moe` and `iqk_fused_delta_net` stubs with `extern "C"` linkage

### lavapipe Host-Visible Buffer Fixes (`ggml/src/ggml-vulkan.cpp`)

lavapipe's buffers are all host-visible (it's a CPU-based Vulkan driver). Two async buffer transfer functions asserted/aborted on host-visible buffers:

- `ggml_vk_buffer_write_2d_async`: replaced `GGML_ABORT("fatal error")` with direct `memcpy` for host-visible+coherent buffers
- `ggml_vk_buffer_read_2d_async`: added early-out `memcpy` path for host-visible+coherent source buffers (before the pinned memory check)

These are effectively zero-copy for lavapipe since "device memory" is already host RAM.

## Test Results

### Single-GPU (RADV only)
```
ggml_vulkan: 0 = AMD Radeon Pro WX 2100 (RADV POLARIS12)
graph splits = 2
prompt eval: 61.76 tokens/sec
eval: 45.66 tokens/sec
Output: ", World!\n [end of text]"
```

### Multi-GPU (RADV + lavapipe)
```
ggml_vulkan: 0 = AMD Radeon Pro WX 2100 (RADV POLARIS12)
ggml_vulkan: 1 = llvmpipe (LLVM 21.1.8, 128 bits)
Vulkan0 buffer: 211.27 MiB, Vulkan1 buffer: 227.33 MiB
KV cache: Vulkan0 = 24 MiB, Vulkan1 = 20 MiB
graph splits = 3
Output: ", World!\nI"
```

### Required CLI flags for non-AVX2 host
- `-no-fug` (disable fused up-gate — no non-IQK fallback exists)
- `-no-fa` (disable flash attention — IQK FA requires AVX2)
- `-ngl 99` (offload everything to GPU to minimize CPU compute)
- `GGML_VK_VISIBLE_DEVICES=0,1` (lavapipe is CPU-type, not auto-included)

## Verification
- Single-GPU Vulkan inference produces correct output
- Multi-GPU split mode graph produces correct output across 2 devices
- Weights distributed across both devices proportional to memory
- KV cache split across devices
- Topology logging shows both devices with type identification
