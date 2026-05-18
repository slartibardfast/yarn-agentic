# PSKV singlewarp K-loop ILP recovery — 2026-05-18

**Result:** TG @ NP=8 went from 27.10 → **27.90 t/s** (+2.95%), PP from 21.04 →
**22.97 t/s** (+9.17%) on dual RTX 6000, Qwen 3.6 27B Q4_0 KV cache, npp=200
ntg=64 npl=8. NPC byte-identity preserved. Singlewarp kernel per-CTA latency
dropped from 188.86 µs to 127.26 µs (−32.7%).

**Commit:** [`c37c161d`](../../ik_llama.cpp/commit/c37c161d) in the submodule;
[`a5c58aa`](../../yarn-agentic/commit/a5c58aa) submodule bump in the parent.

## Where we started

The earlier ralph loop (iter 0–13, see `perf-ralph-pskv-ledger.md` and
`pskv-ralph-loop-final-summary.md`) concluded:

> **Singlewarp at 27.10 t/s TG / 21.04 t/s PP @ NP=8 is the ceiling for
> this kernel under the byte-identity-across-NP determinism contract.**
> The +21.8% TG target (33.0 t/s) is unreachable — per-CTA optimization
> plateaus at +1–2% (grid undersaturation is the bottleneck), and grid
> expansion via split-K breaks NPC at multi-GPU with stochastic 1/8 races.

That conclusion was wrong about the per-CTA family. The diagnosis missed
the dominant stall reason because the prior ncu sessions didn't pull the
warp-state breakdown.

## ncu re-profile

Pulling `WarpStateStats` + `SchedulerStats` at production decode shape
revealed:

| Metric | Value | Reading |
|---|---|---|
| **Issue Slots Busy** | **25.5%** | 75% of cycles idle |
| Warp Cycles Per Issued Instruction | 5.03 | 4 cycles of stall per issue |
| **L1TEX scoreboard stall** | **35% of cycles** | Top stall reason |
| L1/TEX Hit Rate | 86% | Cache is hot |
| L2 Hit Rate | 97% | L2 is hot |
| DRAM Throughput | 0.62% | **Not bandwidth-bound** |
| Eligible warps / scheduler | 0.29 | <1 ready warp per cycle |
| Local memory spills | 0 | Register-resident already |

**Latency-bound, not bandwidth-bound.** The kernel waits 35% of cycles on
L1 cache hits — not because L1 is slow, but because the scalar dot-product
chain has no other warp to issue from when one stalls. Achieved occupancy
was 16.25% (5 active warps/SM) of 50% theoretical; the gap is grid
undersaturation (0.33 waves), not register pressure.

This is the classic latency-hiding gap. With more in-flight requests per
warp, L1 latency can be hidden behind compute. **Per-warp ILP is the
right intervention.**

## The intervention chain

Each step was tested independently: build → NPC smoke (NP={1,8}
byte-identity) → ncu reverify (regs, spills, theoretical occupancy) →
bench (TG/PP at NP=8 over two runs).

| Step | Change | Regs | Spills | per-CTA | TG | PP | Verdict |
|---|---|---|---|---|---|---|---|
| HEAD | baseline | 95 | 0 | 188.86 µs | 27.10 | 21.04 | — |
| 1 | `__launch_bounds__(WARP_SIZE, 16)` | 114 | 0 | 173.02 µs | 26.97 | 21.00 | wash; kept as foundation |
| 2 | + 2-way K-loop ILP | 126 | 0 | 158.98 µs | 27.33 | 21.97 | PP +4.4% above gate |
| 3 | + 3-way (lb=12) | 159 | 0 | 131.30 µs | 27.83 | 22.84 | **first clear ratchet** |
| 4a | + 4-way (lb=10) | 168 | 1,920 | 132.80 µs | 27.77 | 22.89 | tied with 3-way |
| 4b | + 4-way + Q SMEM (lb=10) | 168 | 1,536 | 139.46 µs | 27.73 | 22.28 | regression |
| 4c | + 4-way (lb=8) | **254** | **0** | **127.26 µs** | **27.90** | **22.97** | **LANDED** |

The `__launch_bounds__` second argument is a hint to the compiler about
minimum blocks-per-SM. It bounds register allocation:
`max_regs = floor(65536 / (32 × min_blocks_per_SM))`. The compiler chooses
how many regs to actually use within that cap.

- At lb=10 with 4-way, compiler chose 168 regs and **spilled 1,920 times**
  rather than expand to the full 204-reg budget. The spills hurt per-CTA
  duration enough to wash out the extra ILP gain.
- At lb=8 (256-reg cap), compiler chose 254 regs and **eliminated spills
  entirely**. Per-CTA dropped to its global minimum of the session.

The Q SMEM probe (4b) was the textbook suggestion to free reg headroom
for ILP. In practice it didn't help: the compiler had already absorbed Q
into live ranges internally, and the per-iter SMEM-read latency exceeded
the modest spill reduction (1,920 → 1,536). Net regression.

## Determinism preservation

NPC contract: kernel must produce byte-identical output for the same
(K, V, Q, mask) input regardless of slot, NP value, or GPU layout.

The 4-way ILP preserves NPC by construction because:

1. **Each `partial_a/b/c/d` accumulates over the same per-thread element
   set in the same order** as the scalar kernel. The four streams are
   independent fp32 accumulators, each running the identical `partial +=
   k_val * Q_reg[i]` chain that singlewarp ran for a single k.
2. **Softmax and V·phi still run sequentially in canonical k-order**
   (k → k+1 → k+2 → k+3) after the K·Q dot products complete. The
   Welford state (`kqmax`, `kqsum`, `VKQ[]`) evolves through exactly the
   same fp32 chain as the scalar version.
3. **`warp_reduce_sum` is identical** for any per-lane fp32 set; same
   inputs → same butterfly-tree output.
4. **No cross-CTA reductions, no atomics, no SMEM**, no graph-capture
   lifetime races. Single-CTA, single-warp, register-resident.

Verified via `scripts/quick-pskv-npc-check.sh` on every step: all 8 slots
at NP=8 byte-identical to NP=1, cross-NP slot-0 byte-identical.

## Lessons

1. **Profile before declaring infeasibility.** The "+2% plateau"
   conclusion came from variant-sweep without source-level stall
   breakdown. One ncu session with `WarpStateStats` would have flagged
   the latency-hiding gap immediately.

2. **Latency-bound ≠ bandwidth-bound, and the fix differs.** Low DRAM
   throughput at high cache hit rate is the signature of latency-bound
   code. The lever isn't reducing memory traffic — it's increasing
   concurrent requests so the latency hides.

3. **`__launch_bounds__` is a *register-budget hint*, not a hard cap.**
   The "right" `min_blocks_per_SM` is the one where compiler's chosen reg
   count avoids spills, not the one that maximizes theoretical occupancy.
   Theoretical occupancy bounds achieved only when the grid is large
   enough to fill the SM — at 0.33 waves, achieved is grid-limited
   regardless of theoretical.

4. **Compiler register allocator is non-monotonic in `min_blocks_per_SM`.**
   Tightening lb=2 → lb=16 actually *increased* regs (95 → 114) because
   the compiler used the new tighter target to be more aggressive about
   unroll/ILP. Loosening lb=10 → lb=8 *increased* regs (168 → 254)
   because the compiler now had headroom to avoid spills. The relationship
   is "compiler chooses what produces the best code subject to your
   constraint," not "tighter constraint → fewer regs."

5. **Speculative reg-reduction interventions can backfire.** Q SMEM was
   proposed to free headroom for ILP. In practice the compiler had
   already done equivalent live-range compression, and the SMEM read
   latency was net-negative when applied. Always measure before
   committing to setup work.

6. **Multi-step bench/ncu verification per change matters.** Each step
   in the chain had its own NPC smoke + ncu pull + bench. Step 4a's
   "spills appeared at 4-way" was the cue to try lb=8 rather than
   declaring 3-way the winner. Without the explicit ncu check, the
   ledger would have closed at 3-way and missed the better config.

## What's next

The remaining gap to pre-NPC ceiling (36.68 t/s TG) is 8.78 t/s = 31% of
HEAD. Most of that gap is the determinism contract itself (the pre-NPC
config used the `flash_attn_ext_f16` path with cross-warp WMMA, which
breaks NPC). Recovering more requires either:

- **Determinism-safe grid expansion.** The 1/8 stochastic race in
  graph-captured multi-kernel pipelines (split-K E and WMMA Phase 2) is
  the open structural blocker. Diagnosing the race precisely (not just
  observing it) is a multi-iter investigation.
- **Reducing PSKV's share of decode time** via parent-kernel optimizations
  (NCCL reduce path, cuBLAS gemvx). Per the 2026-05-17 nsys diff, PSKV
  is ~30% of NP=8 decode wall time; the other 70% has its own recovery
  surface area.

For now, 27.90 t/s TG / 22.97 t/s PP @ NP=8 is the new baseline on
`production/2026-q2-next`.
