---
name: latency-bound-vs-bandwidth-bound
description: "Low DRAM throughput at high cache hit rate + low IPC is the signature of latency-bound code; the fix is more in-flight requests (ILP, occupancy), NOT reducing memory traffic"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

When a kernel reports low DRAM throughput (<1%) + high cache hit rate
(>80%) + low IPC (~1.0 out of 4.0) + low warp issue rate (<30% of
cycles have an eligible warp), it is **latency-bound on cache hits**,
not bandwidth-bound on DRAM.

**Why:** The kernel is keeping data hot in L1/L2 (high hit rate) so
DRAM bandwidth isn't the binding constraint. But each L1 hit still
has a real latency (~30 cycles on Turing), and if the kernel's
dependency chain forces the scheduler to wait on that latency without
another warp to issue from, IPC drops.

**Empirical signature from PSKV singlewarp (2026-05-18,
[[pskv-ilp-recovery-landed]]):**
- DRAM throughput: 0.62% (basically nothing)
- L1/TEX hit rate: 86%, L2 hit rate: 97%
- IPC: 1.03 of 4.0 (25%)
- Issued Warp Per Scheduler: 0.26 of 1.0
- "No Eligible Warps": **73.55% of cycles**
- Top stall reason: **L1TEX scoreboard, 35% of cycles**

**The fix is the opposite of bandwidth optimization:**

- ✗ Don't reduce memory traffic — you're not bandwidth-bound.
- ✗ Don't pack values smaller — it doesn't help latency.
- ✓ DO increase concurrent requests per warp via instruction-level
  parallelism (multiple independent accumulators driving multiple
  in-flight loads).
- ✓ DO consider raising occupancy IF the grid can support it (if
  grid is undersaturated, occupancy lever is limited).

**How to apply:** Before declaring a kernel "memory-bound" and
reaching for SMEM staging or weight tiling, check DRAM throughput.
If it's <5%, the constraint is latency hiding, not bandwidth.
The intervention shape is per-warp ILP, not data movement.
