# Phase 8: TURBO_KV_4B Flash Attention — Multi-Head Stride Bug

## Context

Phase 6 shipped TURBO_KV_4B Vulkan shaders (CPY quantize, get_rows dequant, standalone FWHT) and Phase 7 delivered SET_ROWS correctness. The remaining piece was **flash attention** — computing Q·K scores and P·V output directly from quantized TURBO_KV_4B blocks without dequanting to F32 first.

The TurboQuant paper (ICLR 2026, arXiv 2504.19874) describes a Randomized Hadamard Transform (RHT) approach: quantize K/V via forward RHT → Lloyd-Max codebook → 4-bit pack. For attention, either rotate Q (dot-product preservation) or inverse-RHT the stored K/V values.

## Results

| Prompt | F16 | Q4_0 | TURBO_KV_4B (before) | TURBO_KV_4B (after) |
|--------|-----|------|----------------------|---------------------|
| Capital of France | Paris | Paris | London | **Paris** |
| Largest city in Japan | Tokyo | Tokyo | **Tok | **Tokyo** |
| Water freezes at | 0°C | 0°C | $- 18 | **$0°C$** |
| 2+2= | 42+2 | 44+2 | 4.002 | **44+2** |

All prompts now match F16 output. 9B model also produces "Paris".

## The Bug

### Symptom

0.8B Qwen3.5 with `--ctk turbo_kv_4b --ctv turbo_kv_4b` produced "London" instead of "Paris". The 9B model produced "Paris" (larger models tolerate quantization noise). CPU FA fallback also produced "Paris".

### Root Cause

**Multi-head KV cache stride mismatch.** The shader indexed K/V blocks using `blocks_per_row = HSK / 128 = 2` as the inter-token stride:

```glsl
block_idx = offset + kv_token * blocks_per_row + blk;  // WRONG
```

With `n_head_kv=2`, the K/V cache interleaves heads:

```
Token 0: [head0_blk0, head0_blk1, head1_blk0, head1_blk1]  (4 blocks)
Token 1: [head0_blk0, head0_blk1, head1_blk0, head1_blk1]  (4 blocks)
```

The actual stride between same-head tokens is `k_stride = nb1/type_size = 4` blocks (from the FA push constants), not 2. Using stride 2, token 1's head 0 read blocks 2-3 — which is **head 1 of token 0**.

### Fix

```glsl
block_idx = offset + kv_token * k_blk_stride + blk;  // CORRECT
```

Where `k_blk_stride = k_stride` from the push constants.

## Why This Was Hard to Find

### 1. Coherent wrong output

Reading wrong-head data produces plausible attention scores. "London" is a capital city — the model generates fluent English from corrupted attention, not garbled output. This looks like quantization noise, not a data addressing bug.

### 2. Synthetic tests used single-head configuration

All standalone FA tests used `nh=1` or `n_head_kv = nh`, so `blocks_per_row == k_stride`. The bug only triggers when `n_head_kv > 1` with GQA. Tests showed:
- KV=32, nh=1: **ratio 1.0000** (exact match)
- KV=5 + proper mask, nh=1: **NMSE 0.000000** (exact match)

Both passed perfectly while the model (nh=8, n_head_kv=2) was broken.

### 3. The 9B model tolerated the bug

With `n_head_kv=4`, the 9B model also read wrong-head data. But its larger parameter count made attention robust enough to still produce "Paris". This reinforced the "codebook quality" explanation.

### 4. Extensive red herrings

We investigated and ruled out:

| Investigation | Time | Conclusion |
|---------------|------|------------|
| Mesa NIR miscompilation (72K ISA dump) | Hours | Wrong line range — ISA was correct |
| ACO optimizer (ACO_DEBUG=noopt) | Hours | No effect — not ACO |
| nir_opt_algebraic rule 2291 (fsub→fadd+fneg) | Hours | Disabled it, rebuilt Mesa — no effect |
| Custom Mesa build + fsub tracing | Hours | Found fsub count drops but irrelevant |
| Shared memory FWHT precision | Hours | Bit-identical to CPU (max_diff=0) |
| FP accumulation order (pre/post scale) | Hours | Mathematically equivalent (FWHT linearity) |
| subgroupShuffleXor in multi-subgroup WG | Real bug | But unrelated to "London" |
| Denormal flushing (FTZ mode) | Quick | No denormals in FWHT values |
| WorkGroupSize=64 (single subgroup) | Quick | FA structure needs WG≥128 |
| volatile shared memory | Quick | NIR sees through it |
| Integer sign-bit XOR workaround | Quick | NIR recognizes fneg pattern |
| memoryBarrierShared() | Quick | Changed output slightly but didn't fix |
| Out-of-place butterfly | Quick | Same result |

### 5. V dump proved V values were correct

Byte-by-byte dump of V kv_sh values from inside the FA shader showed they matched CPU dequant to 1.19e-07. Q·K scores matched to 2.24e-08. This "proved" the shader was correct — but the dumps used single-head test data.

### 6. The actual clue

The `k_stride=4` value was visible in FA parameter dumps early on, but we didn't connect it to the block indexing. The standard F16 FA uses `k_stride` directly for K buffer access. The TURBO_KV_4B FA used a separate `blocks_per_row` computed from head dimensions, which doesn't account for multi-head interleaving.

## Debugging Lessons

### For multi-head stride bugs

1. **Always test with the model's actual GQA configuration.** If `n_head_kv > 1`, single-head tests miss stride bugs entirely.

2. **Print tensor strides and compare with shader indexing.** The key values:
   ```
   k_stride = nbk1 / ggml_type_size(k->type)  // actual inter-token stride
   blocks_per_row = HSK / block_size            // blocks within one head
   ```
   If these differ, multi-head interleaving exists.

3. **The first generated token diagnoses the bug.** If token 1 is already wrong (before autoregressive accumulation), the bug is in FA, not accumulated precision error.

4. **Compare with a working quant type on the same model.** Q4_0 FA uses the same FA structure with `k_stride`. If Q4_0 works and TURBO_KV_4B doesn't, look for indexing differences.

### For Vulkan compute shader debugging

1. **Dump intermediate values to the output buffer.** Write kv_sh values, scores, or V dequant results to `data_o[]` and read them back. Simplest and most reliable technique.

2. **RADV_DEBUG=shaders produces massive dumps.** Use `grep -n "ACO shader stage"` to find section boundaries. The FA shader can be 40K+ instructions.

3. **Test infrastructure bugs hide real bugs.** We had:
   - Missing `ggml_init()` → fp16 lookup table uninitialized → CPU returned zeros
   - KV < Bc without mask padding → softmax diluted by `exp(0)` contributions
   - Single-head tests → stride bugs invisible

## Other Fixes in This Phase

- **CPU FA fallback**: `iqk_flash_attn` returns false (not abort) for unsupported types; added `from_float` for F32 type traits
- **Score clamping**: Positions ≥ KV get score=-inf (prevents softmax dilution when KV < Bc)
- **V buffer binding**: Fixed hardcoded `kv_np[0]` to parameterized `bind_idx`
- **Large-rows-only FA pipelines**: `CREATE_FA2_LARGE` avoids Bc=64 shared memory overflow
- **Codebook + sign function**: Corrected to match CPU reference (symmetric Lloyd-Max, Knuth hash)
- **GL_EXT_shader_8bit_storage**: Added to FA shader for uint8_t mse_indices access

## Architecture

The final TURBO_KV_4B FA shader computes:

1. **Q rotation** (once, before j-loop): `Qf = (1/√128) · FWHT(sign · Q · scale)` via shared-memory butterfly
2. **K scoring** (per j-tile): `kv_sh = codebook[idx] × norm/inv_std`, then `score = Qf · kv_sh`
3. **Score clamping**: positions ≥ KV → -inf
4. **Softmax**: standard online softmax with incremental max/sum
5. **V inverse RHT** (per j-tile): `kv_sh = codebook/inv_std → FWHT → sign × norm/√128`
6. **P·V accumulation**: `O += softmax_weight × kv_sh`

The Q rotation uses dot-product preservation (`Q_rot · K_stored ≈ Q · K`). V requires full inverse RHT because P·V output must be in the original domain for downstream layers.

## Files Changed

- `ggml/src/vulkan-shaders/flash_attn_turbo_kv.comp` — complete FA shader
- `ggml/src/vulkan-shaders/turbo_kv_4b_rht.comp` — FWHT + codebook (corrected)
- `ggml/src/vulkan-shaders/cpy_f32_turbo_kv_4b.comp` — quantize shader (fp16-consistent)
- `ggml/src/vulkan-shaders/copy_from_turbo_kv_4b.comp` — dequant CPY shader
- `ggml/src/vulkan-shaders/vulkan-shaders-gen.cpp` — shader compilation
- `ggml/src/ggml-vulkan.cpp` — pipeline creation, FA dispatch, supports_op
- `ggml/src/ggml.c` — F32 from_float type trait, fp32_to_fp32_row helper
- `ggml/src/iqk/iqk_flash_attn.cpp` — graceful fallback for unsupported types
- `tests/test-minimal-ops.cpp` — end-to-end FA tests with scheduler
