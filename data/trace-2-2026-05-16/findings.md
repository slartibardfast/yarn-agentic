# TRACE-2 — intra-layer-3 capture, slot 0 vs slot 1 at NP=2 same prompt

**Date**: 2026-05-16
**Run**: `LLAMA_FATTN_PER_SLOT_KV_ENABLE=1 ./build/bin/test-trace-2-intra-layer-capture` with `LLAMA_TEST_NP=2`, both slots prefilled same prompt (n_prompt=12), one decode step.

## Captures (slot 0 vs slot 1 per tensor)

| Tensor | ne | per-slot floats | n_diff | max \|Δ\| |
|---|---|---|---|---|
| `l_out-2` (input to layer 3) | [5120,2,1,1] | 5120 | 0 | 0.0 |
| `Qcur-1003` (post-RoPE Q, dev 0) | [256,12,2,1] | 3072 | 0 | 0.0 |
| `Qcur-2003` (post-RoPE Q, dev 1) | [256,12,2,1] | 3072 | 0 | 0.0 |
| `Kcur-1003` (post-RoPE K, dev 0) | [256,2,2,1] | 512 | 0 | 0.0 |
| `Kcur-2003` (post-RoPE K, dev 1) | [256,2,2,1] | 512 | 0 | 0.0 |
| `Qcur_hadamard-1003` (post-Hadamard Q) | [256,12,2,1] | 3072 | 0 | 0.0 |
| `Qcur_hadamard-2003` | [256,12,2,1] | 3072 | 0 | 0.0 |
| `Kcur_hadamard-1003` | [256,2,2,1] | 512 | 0 | 0.0 |
| `Kcur_hadamard-2003` | [256,2,2,1] | 512 | 0 | 0.0 |
| `Vcur_hadamard-1003` | [512,2,1,1] | 512 | 0 | 0.0 |
| `Vcur_hadamard-2003` | [512,2,1,1] | 512 | 0 | 0.0 |
| **`l_out-3` (layer 3 output)** | [5120,2,1,1] | 5120 | **5120** | **5.860e-03** |

## Conclusion

The divergence enters **between the post-Hadamard Q/K/V tagging and `l_out-3`**. That window covers:

1. K cache write (F32 Kcur_hadamard → Q4_0 K cache CPY+quantize at slot-region offset).
2. V cache write (F32 Vcur_hadamard → Q4_0 V cache CPY+quantize at slot-region offset).
3. FA op (`ggml_flash_attn_ext_per_slot_kv` → `wmma_f16_case_pb1<256,256,8,float>` per CX.A retraction).
4. Output projection (mul_mat wo, F16-pinned per Phase C).
5. Residual add (element-wise).

Items 4 and 5 are deterministic. Phase A/B/C bound the matmul deterministic regardless of batch row. Element-wise add is per-element.

Items 1-3 are the credible sources:

- The KV cache write is a F32→Q4_0 quantize+CPY at a slot-position-dependent target offset. If the quantizer is deterministic per-input (it should be), slot 0's region and slot 1's region end up with byte-identical Q4_0 contents (since the F32 inputs are byte-identical per this trace).
- The FA op reads Q4_0 K + V from these slot regions and dequantizes on-the-fly to F16, then runs wmma_f16. If the dequant + FA mathematically depend only on the read K/V values, slot 0 and slot 1 should produce identical FA output.

So either:
- **A. The Q4_0 quantize-on-CPY is slot-offset-dependent** — quantizing the same F32 to different memory addresses (slot 0 region vs slot 1 region) produces different Q4_0 bits. Possible if the quantizer uses block-position-mod something.
- **B. The FA per-slot-kv kernel's per-row dispatch treats slot 0 and slot 1 differently** — even though `per_row_k_bound[0] == per_row_k_bound[1] == n_prompt` (both prefilled identically), the kernel might index slot 1's K/V via a different memory access pattern.
- **C. The Q4_0 dequantize-during-FA has slot-offset-dependent behavior** — pointer alignment, cache-line crossing, etc.

## Next: TRACE-3

Add ONE MORE capture to disambiguate A vs B/C. Capture the K cache **after** the slot 0 and slot 1 writes, READ the per-slot bytes back, and bit-compare.

If slot 0's Q4_0 K region == slot 1's Q4_0 K region → Q4_0 cache content is identical → the bug is in FA per-slot-kv read/dequant logic (B or C).

If slot 0's Q4_0 K region ≠ slot 1's Q4_0 K region → the bug is in the quantize-on-CPY (A).

Both are testable without breaking production state — just read the Q4_0 cache via `llama_kv_cache_view` or via raw `ggml_backend_tensor_get` on `kv_self.k_l[3]`.

Alternative — TRACE-3': just confirm the FA kernel is the culprit by replacing it with the F16 fallback for layer 3 and re-running. If l_out-3 then becomes slot-0 = slot-1, FA per-slot-kv with Q4_0 KV cache is confirmed as the bug.

Magnitude observation: `max |Δ| = 5.860e-03` here vs `9.525e+00` in TRACE-1's layer 63. The drift starts small at layer 3 and amplifies through the residual stream over 60+ layers — fully consistent with a single-layer numerical drift propagating through all downstream FA + GEMM ops.
