---
name: Declare vec_dot_type=Q8_K for any new low-bit ggml quant type
description: Low-bit (< 4 bpw) quant types must emit 8-bit integer output to ride vpmaddubsw; vec_dot_type=F32 forgoes that path and concedes 4× throughput before a single decoder instruction runs
type: feedback
originSessionId: c1440a07-71f9-4a7a-a10b-10db7f6c5c11
---
When designing a new ggml quant type at < 4 bpw, declare
`vec_dot_type = Q8_K` (or Q8_0) and have the decoder emit 8-bit
integer output. Do **not** declare `vec_dot_type = F32` as a convenience
default.

**Why:** Q4_K_M and other k-quants dispatch to `vpmaddubsw` on AVX2 —
32 MAC/cycle on signed 8-bit integers, saturating into 32-bit
accumulators. `vec_dot_type = F32` uses fp32 FMA — 8 MAC/cycle. That's
a 4× datatype-level throughput penalty before any decoder work. At low
bpw where decoder cost is already high, you compound the datatype
penalty with the decoder penalty and land ~100× behind k-quants.

Measured on Ryzen 9 3950X (2026-04-19, Track B of HARP_2B work):
HARP_2B AVX2-1MAD hits pp128=13 on qwen35-0.8b vs IQ2_S (also 2 bpw)
at 879. Profile shows both the trellis decoder's serial state chain
*and* the fp32 FMA throughput as contributors. Fixing the chain alone
would leave the datatype penalty intact.

**How to apply:**
- Any new HARP/TURBO/lattice/trellis quant type at 2-3 bpw: decoder
  emit path targets Q8_K scratch buffer + int8 dotprod, not F32.
- If the quality path genuinely needs fp32 intermediate values (e.g.
  per-block scale multiply before reconstruction), the scale multiply
  can stay fp but the final emit to vec_dot must round to int8.
- When evaluating a new type's AVX2 ceiling, compare against IQ2_S /
  IQ2_XS at the same bit budget, not against Q4_K_M at 4.5 bpw. The
  latter has both more bits and a faster arithmetic path.
- This applies on AVX-512 VNNI too (compounds favourably there).

**Related structural rule:** trellis decoders with carried state
(`state[i] = f(state[i-1])`) have a hard throughput ceiling
independent of SIMD width, because OoO can't reorder past the
dependency. N serial steps per block × 128 elements = 128 cycles of
critical path per block minimum. Pair with the Q8_K lesson: even a
perfect decoder contract on a serial trellis will plateau. Both must
be addressed together.

**Follow-up if this comes up:** the Track B prerequisite to any fused
decode-GEMM route is a one-file change: dispatch HARP_2B's AVX2 kernel
to emit Q8_K, not F32. Expected 2-3× lift on pp128 as a measurement
baseline, even with the trellis chain intact.
