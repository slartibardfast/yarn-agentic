---
name: pskv-ilp-recovery-landed
description: "PSKV singlewarp FA kernel 4-way K-loop ILP + lb=8 landed on production/2026-q2-next 2026-05-18; recovered TG +2.95% / PP +9.17% over HEAD baseline, NPC PASS at NP={1,2,4,8} multi-GPU verified"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

PSKV singlewarp FA kernel optimization landed on `production/2026-q2-next`
on 2026-05-18.

**Commits:**
- Submodule `ik_llama.cpp@c37c161d`: `__launch_bounds__(WARP_SIZE, 8)` + 4-way K-loop ILP in `ggml/src/ggml-cuda/fattn-per-slot-kv-singlewarp-sm75.cu`
- Parent `yarn-agentic@a5c58aa`: submodule bump + ledger append
- Parent `yarn-agentic@8e03f56`: writeup `data/pskv-ilp-recovery-2026-05-18.md`

**Verified perf (dual RTX 6000, npp=200 ntg=64 npl=8):**
- ncu per-CTA: 188.86 µs → **127.26 µs** (−32.7%)
- TG @ NP=8: 27.10 → **27.90 t/s** (+2.95%)
- PP @ NP=8: 21.04 → **22.97 t/s** (+9.17%)
- 254 regs/thread, 0 local spills, 25% theoretical occupancy

**Verified determinism (`scripts/verify-production-determinism.sh`):**
NP={1,2,4,8} all slots byte-identical to NP=1; all 6 cross-NP slot-0
comparisons byte-identical. The NP=4/8 vanilla drift that T9 named
(`project_dflash_t9_np_validity_drift_signature`) is empirically
resolved by whatever NPC work landed between 2026-05-14 and 2026-05-18 —
the PHASE_DFLASH future-work doc says "named, not started" but the
harness now PASSES at the full NP={1,2,4,8} cross-product.

**Why:** The earlier ralph loop (`data/perf-ralph-pskv-ledger.md` iters
0–13) concluded singlewarp was at +2% per-CTA ceiling. That was wrong.
ncu re-profile revealed the kernel was latency-bound on L1TEX scoreboard
stalls (35% of cycles) with 0.6% DRAM throughput — a latency-hiding
gap, not a work-amortization plateau. 4-way K-loop ILP breaks the
scalar dot-product dependency chain by running four independent partial
accumulators in the same inner i-loop; compiler interleaves their
dequant+FMA streams. NPC preserved by construction: softmax+V still
run sequentially in canonical k-order, only the dot product is
parallelized via independent accumulators.

**How to apply:** When a per-CTA optimization sweep plateaus at small
percentages, profile with `--section WarpStateStats --section SchedulerStats`
to get the stall-reason breakdown before declaring infeasibility.
