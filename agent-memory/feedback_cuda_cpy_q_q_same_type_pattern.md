---
name: feedback-cuda-cpy-q-q-same-type-pattern
description: "When adding multi-stream graph builders that emit Q→Q same-type cpy ops with non-contiguous strides (e.g. defrag, future hole-fill compactors), the CUDA cpy dispatcher must explicitly support that combo. The pre-existing Qn cpy entries only covered F32→Qn and Qn→F32/F16. Without the entry, the scheduler falls back to CPU, which segfaults reading CUDA-resident data."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

# CUDA cpy Q→Q same-type non-contiguous pattern

**Rule:** When emitting a `ggml_cpy(src_view, dst_view)` where both src and dst are views of a quantized cache tensor (e.g. Q4_0 K cache, Q4_0_AR16 K cache with Hadamard) and the views have non-contiguous strides (e.g. 3D views into a 4D layout), the CUDA backend must explicitly support `Qn → Qn` with the operand's quant type. Without an entry in `ggml_cuda_cpy` / `ggml_cuda_cpy_fn` / `ggml_backend_cuda_supports_op`, the scheduler falls back to CPU, which can't read CUDA-resident data and segfaults in `ggml_compute_forward_dup_bytes`.

**Why:** Pre-T3.6.I.c2 the dispatcher only had F32→Qn and Qn→F32/F16 entries plus a special Q8_0 transpose path (for MLA mla=2). For same-shape contiguous src + same type, line 552 of cpy.cu enters the `src->type == src1->type && contiguous` branch which falls through to `cudaMemcpyAsync` — but only for contiguous src AND dst. For non-contiguous strided views (4D KV with per-stream sub-slicing), this branch is bypassed and the dispatcher hits `GGML_ABORT`. The scheduler's `supports_op` returns false for the unsupported combo, sending it to CPU.

**How to apply:** Add a generic kernel parameterized by `qk` + `block_bytes` at runtime so one kernel covers every quant type:

```cpp
// In ggml-cuda/cpy.cu, before ggml_cuda_cpy_dest_ptrs_copy:
static __global__ void cpy_q_q_same_type(
        const char * cx, char * cdst_direct, const int ne,
        const int ne00, ..., const int nb13,
        const int qk, const int block_bytes,
        char ** cdst_indirect, int graph_cpynode_index) {
    const int i = (blockDim.x*blockIdx.x + threadIdx.x) * qk;
    if (i >= ne) return;
    // ... compute strided offsets via (i00/qk)*nb00 + i01*nb01 + ...
    for (int b = 0; b < block_bytes; b++) {
        dst[b] = src[b];
    }
}
```

Wire into `ggml_cuda_cpy`:
```cpp
} else if (ggml_are_same_shape(src0, src1) && src0->type == src1->type
           && ggml_is_quantized(src0->type)
           && !ggml_is_transposed(src0) && !ggml_is_transposed(src1)) {
    const int qk = ggml_blck_size(src0->type);
    const int block_bytes = (int)ggml_type_size(src0->type);
    ggml_cpy_q_q_same_type_cuda(...);
}
```

Same in `ggml_cuda_cpy_fn` (for graph-reuse). Add a matching `supports_op` entry in ggml-cuda.cu so the scheduler keeps the op on CUDA.

**Discriminate from transpose_q8_0:** the pre-existing `ggml_are_same_shape && Q8_0→Q8_0` path was for MLA mla=2 transposed Q8_0 cache and does requantization. Add `(ggml_is_transposed(src0) || ggml_is_transposed(src1))` to its guard so it only matches the transpose case; the new non-transposed same-type Qn cpy takes the plain block-copy path.

**Quant types covered by the generic kernel** (production-relevant):
- Q4_0: QK4_0=32, block_bytes=18 (test KV cache)
- Q4_0_AR16: QK_AR16=16, block_bytes=10 (production with --k/v-cache-hadamard)
- Q8_0: QK8_0=32, block_bytes=34 (alternative KV cache)
- Plus Q4_1, Q5_0, Q5_1, Q6_0, IQ4_NL — all standard.

**Established:** PHASE_NSTREAM_KV_PERF T3.6.I.c2 (2026-05-22). Defrag of strided 4D KV views was previously gated by `n_stream == 1` assert in `build_defrag`; lifting the assert + adding the per-(device, stream) 3D-per-stream view exposed this CUDA cpy gap. Test passes on Qwen 3.6 27B production target under both LAYER and GRAPH split modes; `verify-production-determinism.sh` ACCEPTANCE PASS; DFlash composition GREEN.

**Apply to:** any future multi-stream graph builder that creates strided Q→Q same-type cpy ops (graph compactors, in-place quantized reshapes, …).
