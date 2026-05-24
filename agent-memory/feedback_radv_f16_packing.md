---
name: RADV ACO f16vec2 packing requirements
description: How to write Vulkan GLSL shaders that actually get v_pk_fma_f16 (Rapid Packed Math) on AMD GCN/RDNA via RADV ACO compiler
type: feedback
originSessionId: a5a02c13-3d1d-4998-badf-36c74b87c680
---
RADV ACO can pack two float16_t values into one 32-bit VGPR, but ONLY when the NIR
re-vectorization pass (nir_lower_alu_width with vectorize_vec2_16bit) can pair them.

**Why:** ACO's pipeline scalarizes ALL vector ALU ops first, then tries to re-pair
16-bit scalars back into f16vec2 for VOP3P packed instructions. Scalar fma chains
`fma(a, b, fma(c, d, e))` have sequential dependencies that prevent pairing.

**How to apply:** Structure FP16 dot products as explicit f16vec2 operations:

BAD (generates scalar v_fma_f16, zero packing):
```glsl
float16_t sx = fma(b.x, q.x, fma(b.y, q.y, fma(b.z, q.z, b.w * q.w)));
```

GOOD (generates v_pk_mul_f16 — two muls per cycle, packed registers):
```glsl
float16_t sx = dot(b.xy, q.xy) + dot(b.zw, q.zw);
// or equivalently:
f16vec2 p0 = b.xy * q.xy;  // independent component-wise mul → packable
f16vec2 p1 = b.zw * q.zw;
float16_t sx = (p0.x + p0.y) + (p1.x + p1.y);
```

The key: the two multiplications in each f16vec2 `*` are INDEPENDENT, so the
re-vectorization pass can pair them. In the fma chain, each fma depends on the
previous one.

**Diagnostics:** `RADV_DEBUG=shaders` dumps the ISA. Grep for `v_pk_` to see
packed instructions. Zero `v_pk_*` means the compiler failed to pack.

**Also important:**
- Scalar float16_t occupies a FULL 32-bit VGPR (low 16 bits used, high wasted)
- f16vec3 causes severe register pressure — always use f16vec2 or f16vec4
- ACC_TYPE pattern: use F32 accumulator with F16 intermediates to avoid precision
  loss across blocks (inner f16 dot, outer f32 += ACC_TYPE(result))
- `v_cvt_f16_f32` costs 1 cycle per conversion — minimize F32↔F16 conversions
