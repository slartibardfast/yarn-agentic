---
name: turbo_kv_4b wrapper pattern — per-kernel dispatch with single_block_dot
description: When adding SIMD kernels, use per-kernel single_block_dot functions with compile-time dispatch in ggml_vec_dot_*_cpu, not delegation to batched attention_multi
type: feedback
originSessionId: b5d1669a-845c-4be3-9664-d6bfaddbdcd2
---
**Rule:** When adding a new SIMD kernel to ggml, avoid delegating `ggml_vec_dot_*_cpu` to `turbo_kv_4b_attention_multi(..., valid_count=1)`. Instead:

1. Each kernel header defines its own `*_single_block_dot(block, q_rot, dim)` that handles one block's dot product (extracts norm/inv_std, computes per_block_scale, calls inner loop)
2. The AVX2 wrapper `ggml_vec_dot_*_avx2` iterates the batch and calls the AVX2 single_block_dot
3. The CPU dispatch `ggml_vec_dot_*_cpu` uses `#if defined(GGML_TURBO_KV_4B_HAVE_AVX2)` / `#elif defined(GGML_TURBO_KV_4B_HAVE_SSE)` / `#else` to select the right kernel
4. SSSE3 and scalar fallback paths each have their own single_block_dot and loop

**Why:** Delegation to `turbo_kv_4b_attention_multi` with `valid_count=1` is correct but adds unnecessary overhead (per-call query rotation, extra indirection). The per-kernel single_block_dot pattern gives direct control and keeps each path independent.

**How to apply:** When adding any new SIMD kernel (AVX2, AVX-512, NEON, etc.), follow this pattern. Each kernel gets its own `*_single_block_dot` in its header file, and the CPU dispatch selects via compile-time guards.
