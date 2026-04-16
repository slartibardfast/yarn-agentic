# Phase 23: TURBO_4B Weight Quantization (RHT + Lloyd-Max Codebook)

## Status: IN PROGRESS (CPU quantize/dequant/vec_dot complete, tests passing)

## Problem

Q4_K_M (the standard 4-bit k-quant) uses per-block min/max scaling. Outlier weights distort the scale for the entire block (32 elements), causing disproportionate quantization error. This matters especially for:

- **MTP head weights**: acceptance rate drops from 82% (Q8_0) to 22% (Q4_K_M) — the draft token quality is highly sensitive to weight precision
- **MoE router gates**: routing accuracy depends on small weight differences that outlier distortion corrupts
- **out_proj tensors**: KLD ~6.0 (worst tensor per unsloth dynamic 2.0 analysis), no preceding norm to absorb error

## Approach: Extend TURBO_KV_4B to weight storage

TURBO_KV_4B already implements a complete RHT + codebook quantization pipeline for KV cache tensors:

1. **Random Hadamard Transform** (RHT) rotates each block, spreading outlier energy uniformly across all elements (incoherence processing)
2. **Lloyd-Max Gaussian codebook** (16 centroids, 4-bit indices) quantizes the rotated values — optimal for the approximately Gaussian distribution produced by RHT
3. **Per-block scale + inverse-std** stored as FP16 metadata

This is the same principle as QuIP# (ICML 2024, arXiv 2402.04396), which uses RHT + E8-lattice codebook and achieves near-lossless quality at 4 bpw — roughly Q5_K quality at Q4_K bitrate. No GGUF implementation of RHT-based weight quantization exists anywhere.

## Expected Quality vs Q4_K_M (same ~4.5 bpw)

Based on QuIP# published results (Llama-2-7B, WikiText-2):

| Method | bpw | PPL | vs Q4_K_M |
|---|---|---|---|
| Q4_K_M | 4.58 | 6.74 | baseline |
| QuIP# 4-bit (E8 lattice) | 4.0 | ~5.9 | -0.8 PPL at lower bpw |
| TURBO_4B (Lloyd-Max, projected) | 4.5 | ~6.1-6.3 | -0.4 to -0.6 PPL |

Lloyd-Max Gaussian codebook is simpler than E8 lattice (table lookup vs lattice decode) but slightly less optimal. Still substantially better than k-quants because the RHT eliminates the outlier problem.

MTP-specific impact: smoother quantization error from RHT should preserve draft token quality at lower bitrates, potentially allowing Q3-class bitrates (~3.5 bpw) without the MTP acceptance collapse seen with k-quants.

## Implementation Plan

### New ggml type: GGML_TYPE_TURBO_4B

**File**: `ggml/include/ggml.h`

Add `GGML_TYPE_TURBO_4B` to the type enum. Block structure is identical to `block_turbo_kv_4b`:

```c
// Block size = 128 elements (matches head_dim, 7-stage FWHT)
// Storage: 2 bytes (fp16 norm) + 2 bytes (fp16 inv_std) + 64 bytes (128 × 4-bit indices)
// = 68 bytes per 128 elements = 4.25 bpw
typedef struct {
    uint16_t norm;        // fp16 L2 norm of the original block
    uint16_t inv_std;     // fp16 inverse standard deviation after RHT
    uint8_t  qs[64];      // packed 4-bit codebook indices (2 per byte)
} block_turbo_4b;
```

### Weight quantization (offline, CPU)

**File**: `ggml/src/ggml-common.h` (type traits), `tools/quantize/quantize.cpp`

Per-block quantization:
1. Load 128 floats from the weight tensor
2. Apply forward RHT (sign flip → Walsh-Hadamard butterfly → normalize)
3. Compute block norm and inv_std
4. Nearest-neighbor codebook lookup (16 Lloyd-Max Gaussian centroids)
5. Pack 4-bit indices into qs[64]

The forward RHT and codebook lookup are already implemented in `turbo_kv_4b_rht.glsl` (GPU) and `test-turbo-kv-vulkan.cpp` (CPU reference). Port the CPU reference into the quantizer.

### Dequantization (runtime, GPU + CPU)

**Vulkan**: The existing `dequant_turbo_kv_4b.comp` shader works almost unchanged — it reads block metadata, unpacks 4-bit indices, looks up codebook values, applies inverse RHT. Register as a new pipeline for `GGML_TYPE_TURBO_4B`.

**CPU**: Scalar dequant in `ggml-cpu/ops.cpp`. The FWHT is 7 stages of butterfly operations — fully unrollable, no branches.

### mul_mat kernel (the hot path)

**File**: `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_turbo_4b.comp` (new)

The dequant-then-FMA approach works (dequant block → F16/F32 → standard dot product). But a fused kernel that reads codebook indices and accumulates directly would be faster — each codebook lookup is a table read (16 entries), and the FWHT can be done on the accumulated partial sums instead of per-element.

Fused approach (advanced, optional):
- Accumulate `codebook[idx] * activation` in RHT-rotated space
- Apply inverse RHT to the accumulated result (one FWHT per output element instead of per block)
- This is mathematically equivalent because RHT is orthogonal: `dot(RHT(w), a) = dot(w, RHT^-1(a))`

Start with dequant-then-FMA. Optimize to fused kernel if dequant overhead is measurable.

### Pipeline registration

**File**: `ggml/src/ggml-vulkan/ggml-vulkan.cpp`

Add `pipeline_dequant_mul_mat_vec_turbo_4b_f32` array alongside existing `_f32_f32` and `_f16_f32` arrays. Register in `supports_op` for MUL_MAT when src0 type is TURBO_4B.

### Block size considerations

TURBO_KV_4B uses block_size=128 (matching head_dim=128, 7-stage FWHT). For weight tensors:

- **128** (current): 7-stage FWHT, proven correct, matches KV cache infrastructure
- **64**: 6-stage FWHT, smaller blocks = better per-block adaptation, but higher metadata overhead (4 bytes per 64 elements = 4.5 bpw vs 4.25 bpw at 128)
- **256**: 8-stage FWHT, lower overhead (4.125 bpw), but coarser blocks

Start with 128. The FWHT shader is parameterized by block size — changing it later is straightforward.

### llama-quantize integration

**File**: `tools/quantize/quantize.cpp`

Add `TURBO_4B` as a quantization target. Support both:
- Global: `llama-quantize model.gguf output.gguf TURBO_4B`
- Per-tensor: `--tensor-type blk.*.ffn_gate.weight=turbo_4b`

### imatrix support (optional, follow-up)

The Lloyd-Max codebook assumes Gaussian distribution after RHT. An importance matrix could weight the codebook centroids toward the distribution's actual shape (which may be sub-Gaussian for some tensors). This is analogous to unsloth's AWQ-style pre-scaling.

## Verify by

1. **Correctness**: `test-backend-ops` MUL_MAT with TURBO_4B × F32 on both Vega and 6800 XT
2. **Round-trip quality**: quantize → dequant → measure RMSE vs F16 on Qwen3.5-0.8B weight tensors. Must beat Q4_K_M RMSE at same bpw.
3. **PPL**: `llama-perplexity` on WikiText-2 with Qwen3.5-0.8B TURBO_4B vs Q4_K_M
4. **MTP acceptance**: Qwen3.5-0.8B with TURBO_4B weights + F16 MTP head. Acceptance rate must exceed Q4_K_M's 22%.
5. **Performance**: mul_mat_vec throughput within 20% of Q4_K_M (dequant overhead from FWHT)

## References

- QuIP#: arXiv 2402.04396, https://github.com/Cornell-RelaxML/quip-sharp
- QTIP: arXiv 2406.11235 (RHT + trellis-coded VQ, NeurIPS 2024 Spotlight)
- Existing TURBO_KV_4B: `ggml/src/ggml-vulkan/vulkan-shaders/turbo_kv_4b_rht.glsl`
- Lloyd-Max codebook: `TURBO_KV_CODEBOOK[16]` in the same file
- Unsloth dynamic 2.0 KLD sensitivity analysis: https://unsloth.ai/blog/dynamic-v2
- Phase 0 FWHT stride-32 fix (this session): LDS swap for `subgroupShuffleXor(val, 32)` on wave64
