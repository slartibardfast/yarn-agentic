# DATA-5 — ptxas resource profile of singlewarp kernel

**Date**: 2026-05-16
**Compiler**: nvcc 13.x, --generate-code=arch=compute_75,code=sm_75, -O3, -use_fast_math
**Kernel**: `flash_attn_per_slot_kv_singlewarp_kernel<256, 256, Q4_0, Q4_0>`

## ptxas output

```
ptxas info: 27104 bytes gmem, 136 bytes cmem[4]
ptxas info: Compiling entry function flash_attn_per_slot_kv_singlewarp_kernel<256,256,Q4_0,Q4_0> for 'sm_75'
ptxas info: Function properties: 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info: Used 95 registers, used 0 barriers, 524 bytes cmem[0]
ptxas info: Compile time = 19.9 ms
```

## Interpretation

- **95 registers per thread**. Above the "target ≤ 64" rule of thumb for 2 blocks/SM,
  but our kernel uses `__launch_bounds__(WARP_SIZE, 2)` = 32 threads × 2 blocks/SM
  minimum. At 95 regs × 32 threads = 3040 regs/CTA × 2 CTAs/SM = 6080 regs/SM out of
  TU102's 65536 regs/SM = 9% register pressure. Massive headroom.

- **0 register spills, 0 stack frame**. The whole kernel fits in registers.

- **0 bytes SMEM**. We use only registers; no shared memory allocated.

- **0 barriers used**. Single-warp kernel has no warp-sync requirements (warp-shuffles
  are sync'd by hardware).

## Occupancy analysis

Constraints on CTAs/SM for TU102:
- Max threads/SM: 1024
- Max warps/SM: 32
- Max CTAs/SM: 16

Per-CTA cost: 32 threads × 95 regs = 3040 regs, 0 SMEM, 1 warp.

Effective max CTAs/SM: min(16 CTAs/SM, 32 warps/SM ÷ 1 warp/CTA, 65536 regs/SM ÷ 3040 regs/CTA) = min(16, 32, 21) = 16 CTAs/SM.

So singlewarp CTAs can pack at 16/SM. For NP=4 (96 CTAs total): 96/72 SMs = 1.3 CTAs/SM avg — far below capacity. For NP=8 (192 CTAs): 192/72 = 2.7 CTAs/SM — still way below 16. No occupancy bottleneck.

## Why __launch_bounds__(WARP_SIZE, 2)?

The "2" tells nvcc to optimize register allocation for at least 2 CTAs/SM. nvcc will
spill to local memory if necessary to fit 2 CTAs/SM. We measured 0 spills with 95
regs, so the compiler made it without spilling. At higher CTA counts (e.g. 8/SM)
it might spill — but we don't need higher CTA density for our workloads.

## Comparison to wmma_f16_case_pb1

The wmma kernel uses cpb=8 (128 threads/CTA, 4 warps), and processes 8 cols of work
per CTA. Different occupancy and register pattern. Direct comparison not meaningful
without ptxas profile of wmma at same shape.

## Conclusion

Singlewarp's resource profile is excellent: no spills, low register pressure,
plenty of occupancy headroom. The 1-4% slowdown vs wmma (per V6) is not from
resource constraints; it's from the per-row CTA architecture's lower per-CTA
parallelism for the SAME total work.
