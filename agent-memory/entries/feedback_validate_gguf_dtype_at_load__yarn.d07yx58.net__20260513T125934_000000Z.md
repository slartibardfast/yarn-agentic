---
name: Validate GGUF tensor dtype at load — type confusion silently produces garbage
description: GGUF stores tensor dtypes explicitly (F32, F16, BF16, Q4_0, Q4_0_AR16, etc.). Uploading raw bytes from a F32 tensor and addressing it as `__half *` interprets each pair of fp32 bytes as ONE fp16 value — garbage, NaN cascade. Always cast at load time, OR check the tensor type matches the pointer type.
type: feedback
originSessionId: 9b52ee2b-d969-442c-83ba-98396525f6fa
---
GGUF tensors carry their `ggml_type` explicitly. A loader that uploads `tn->data` raw and stores the device pointer as `__half *` (because that's what the kernel expects) silently produces garbage when the GGUF type is anything other than F16. fp32 → `__half *` is the most common trap: kernels read `__half2float(ptr[i])`, interpreting each 2-byte slice as fp16 → garbled values → NaN cascade.

**Why this is invisible at test time:**
- Unit tests that generate random fp16 weights in-test (not loaded from GGUF) never hit this. T3 tests passed byte-identity at random fp16 weights → bug stayed dormant.
- The bug only surfaces when REAL production GGUF weights are loaded. The T4 closure test was the first end-to-end run with real weights.

**How to apply:**
- Loader functions must check `ggml_type` against the pointer type they're returning. Either:
  - (a) Cast at load time — read F32 source data, allocate `__half` host buffer, fill via `__float2half`, upload as `__half`.
  - (b) Type-tag the pointer (e.g. `const float * output_norm` vs `const __half * attn_norm`) and let the kernel pick the right read.
  - (c) Assert at load time that the GGUF type matches the expected destination type, abort with clear error otherwise.
- For DFlash drafter / Qwen 3.5 / 3.6 norm tensors specifically: all RMSNorm weights (`attn_norm`, `attn_q_norm`, `attn_k_norm`, `ffn_norm`, `hidden_norm`, `output_norm`) are F32 in the GGUF. Production loaders MUST cast → F16 (or read as fp32 and accept the kernel-side type change).
- For Q4_0, Q4_0_AR16, IQ-quants: these are not fp32/fp16; they have their own block formats. The loader must know which kernel consumes each tensor and route appropriately.

**Concrete example (T4 DFlash closure):** Drafter loader uploaded F32 norm weights raw and stored as `const __half *`. Kernel reads `__half2float(weight[i])` → garbage normalization weight values → sum_sq computed against garbage → rsqrt produces ∞ or 0 → output NaN. T3 missed it because T3 unit tests generate random fp16 weights in-test rather than loading from GGUF. Fix: `upload_f32_as_f16` helper that casts at upload time. Applied to all 6 norm tensors in the drafter loader.

**Generalisable:** Any loader that promotes test-side stub code to production load path needs a dtype-validation pass. If the test-side code generated weights synthetically, the dtype-from-GGUF code path was never exercised; assume it's wrong until you've verified.
