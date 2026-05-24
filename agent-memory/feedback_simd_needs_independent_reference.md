---
name: SIMD vec_dot correctness needs an independent scalar reference, not test-backend-ops alone
description: test-backend-ops `-b CPU` compares the CPU backend against itself with `use_ref` toggled, but `use_ref` only gates tiling/flash-attn path — NOT SIMD dispatch. SIMD kernel bugs silently agree with themselves across both sides and pass the whole test battery. To catch them, drive the production dispatch via the CPU backend's mul_mat and compare against an independent scalar reference (e.g. the ggml-base scalar vec_dot symbol).
type: feedback
originSessionId: 60b8f2a3-4018-43ac-ae61-4b83f88e6a1b
---
**Rule.** For any per-type SIMD vec_dot kernel in llama.cpp, don't rely on `test-backend-ops -b CPU` for correctness. Add a PBT (or equivalent targeted test) that drives the production dispatch via `ggml_backend_cpu_init` + a minimal `ggml_mul_mat` graph, and compares the backend's output against the ggml-base pure-scalar vec_dot symbol for that type.

**Why:**

On 2026-04-24 I propagated a `SIMDEquivalence` property from `mul_mat_cpu.allium` into `llama.cpp/tests/test-turbo-kv-pbt.cpp`. The first run surfaced a ~1900× divergence. Root-cause found two distinct AVX2 kernel bugs in `ggml-cpu/arch/x86/turbo_kv_4b_avx2.h`:

1. `per_block_scale` inverted in the SSE AND AVX2 per-block wrappers (`127/CENT_MAX` vs the correct `CENT_MAX/127`). Inflated scores by ~2160×.
2. AVX2 interleave pulled elements 16..31 from upper 128-bit lanes that held garbage/duplicated VPSHUFB results, because the index vector was zero-padded in the upper lane.

Both passed `test-backend-ops -b CPU -o MUL_MAT` — 1113/1113 runs, 10 of which were `turbo_kv_4b` cases. Both passed `test-turbo-kv-4b-attn` (which compares two paths that both use the scalar per-block dot). Neither was caught until a test compared the dispatch against an independent scalar reference.

The `use_ref` flag on the CPU backend (toggled by `ggml_backend_cpu_set_use_ref`) gates only tiling/FA/KV-chunking paths — see `ggml/src/ggml-cpu/ops.cpp` lines 8992-9087. It does NOT route vec_dot through a different code path. So when test-backend-ops compares backend-with-use_ref vs backend-without-use_ref, both paths dispatch to the same type-trait `.vec_dot`, hit the same SIMD kernel, and produce the same (possibly wrong) score. The "comparison" is a tautology for SIMD correctness.

**How to apply:**

When adding or modifying a per-type SIMD vec_dot kernel in llama.cpp's CPU backend:

1. Ensure the type has a scalar reference vec_dot exposed from ggml-base (e.g. `ggml_vec_dot_turbo_kv_4b_f32` in `ggml/src/ggml-turbo-kv.c`). This is the independent oracle.
2. Add a test that:
   - Quantizes input via the type's `quantize_row_*_ref`.
   - Computes the reference score by calling the ggml-base scalar symbol directly.
   - Builds a minimal `ggml_mul_mat` graph with the quantized K and an f32 Q, runs it through `ggml_backend_cpu_init()` + `ggml_backend_graph_compute`.
   - Compares the two scores with a mixed tolerance (`abs_err <= abs_tol OR rel_err <= rel_tol`) — rel_err is unstable near zero dot products.
3. The int8 codebook quantization used by some SIMD kernels (TURBO_KV_4B, Q4_0-family) introduces ~1% per-element reconstruction error vs the fp32 scalar codebook. Budget `abs_tol ~ 0.1` and `rel_tol ~ 0.05` to cover it; tighter values won't pass unless the SIMD kernel uses the fp32 codebook.

The Allium spec `mul_mat_cpu.allium` (committed 2026-04-24) captures the SIMDEquivalence invariant formally; the corresponding PBT lives at `llama.cpp/tests/test-turbo-kv-pbt.cpp::property_SIMDEquivalence_turbo_kv_4b`. Use those as templates for other SIMD-backed types.
