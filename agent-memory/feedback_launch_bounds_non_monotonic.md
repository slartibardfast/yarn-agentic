---
name: launch-bounds-non-monotonic
description: "nvcc's __launch_bounds__ second arg (minBlocksPerSM) does NOT monotonically reduce regs; tighter values may increase regs as compiler does more aggressive ILP/unrolling within the new cap"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

`__launch_bounds__(threads, minBlocksPerSM)` is a register-budget HINT,
not a hard cap. The compiler chooses how many regs to actually use
within the implied max budget (`floor(reg_per_SM / (threads × minBlocksPerSM))`).
The relationship between minBlocksPerSM and chosen reg count is
**not monotonic**.

**Why:** When given more reg headroom (smaller minBlocksPerSM, larger
implied max), the compiler may choose to use MORE regs to enable more
aggressive ILP/unrolling that produces faster code. When given less
headroom (larger minBlocksPerSM, smaller implied max), it may
compress live ranges OR spill — and spills are expensive (~200
cycles per local memory access on hot paths).

**Empirical sequence from PSKV 4-way ILP work (2026-05-18,
[[pskv-ilp-recovery-landed]]):**

| `min_blocks_per_SM` | Implied max regs | Actual regs | Spills | Notes |
|---|---|---|---|---|
| 2 (baseline) | 1024 | 95 | 0 | Compiler default cautious |
| 16 | 128 | 114 | 0 | Compiler did MORE aggressive ILP under tighter cap |
| 12 (after 3-way ILP) | 170 | 159 | 0 | Sweet spot for 3-way |
| 10 (after 4-way ILP) | 204 | 168 | **1,920** | Compiler chose to spill rather than expand to 204 |
| 8 (after 4-way ILP) | 256 | **254** | 0 | Full reg-budget recovery, spills eliminated |

**How to apply:**

1. Don't expect "tighter launch_bounds → fewer regs." Both directions
   can happen depending on what the compiler decides is faster.
2. The right `minBlocksPerSM` is the one where ncu shows 0 spills
   AND theoretical occupancy fits the achievable grid waves — not
   the one that maximizes theoretical occupancy on paper.
3. Probe by toggling `minBlocksPerSM` across a small range
   (e.g. 8, 10, 12, 16) and ncu after each. The "best" point shows
   itself by the per-CTA duration and 0 spills.
4. At grid-undersaturated decode (<1 wave/SM), achieved occupancy
   is grid-limited regardless of theoretical, so dropping
   theoretical from 50% to 25% doesn't hurt achieved if the grid
   stays the same. Per-CTA latency reduction dominates.
