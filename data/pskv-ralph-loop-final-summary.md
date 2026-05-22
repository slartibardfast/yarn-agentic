# PSKV Ralph loop — final summary (2026-05-18)

## Outcome

After 11 iterations, the per-CTA optimization search has reached its
ceiling at the singlewarp+scalar architecture. The +21.8% TG target
(33.0 t/s) is NOT reachable through cheap single-kernel single-warp-
family edits. Singlewarp baseline restored on `production/2026-q2-next`.

## Per-CTA optimization plateau (Variants A, A', B, C, F)

All variants that preserve single-kernel + canonical K-loop + scalar
fp32 Welford softmax plateau at **+1-2% TG @ NP=8**:

- A (4-warp Dv-split): +1.96%
- A' (4-warp redundant-dot): +1.59%
- B (4-warp + SMEM K/V tiles): +1.14%
- C (8-warp + SMEM K/V tiles): +1.33%
- F (K-pair ILP unroll): +1.33%

Diagnosis: the kernel is **balanced** at 31% compute / 31% memory /
0.6% DRAM throughput per ncu. Per-CTA work amortization (multi-warp,
SMEM K-tile, ILP) gives 4-8× theoretical occupancy improvement but
the SM is **grid-undersaturated** at decode (320 CTAs / 72 SMs = 28%).
No per-CTA edit can increase the GRID size.

## Grid expansion via split-K — perf works, determinism broken (Variants E, E1)

Variant E (split-K PB=4) achieves the target perf signal:
- **+5.7% TG @ NP=8 (27.10 → 28.65)**
- **+17.2% PP @ NP=8 (21.04 → 24.65)**

Recovers 30-40% of the HEAD-to-pre-NPC gap. NPC smoke passes (single-
GPU). But full multi-GPU NPC FAILS with stochastic cluster patterns:

- Iter 2 (PB=4): cluster `{1,2,4}≡ ≢ {8}`
- Iter 4 (PB=1): cluster `{1,4,8}≡ ≢ {2}`
- Iter 9 (PB=4 + memset): cluster `{1,2}≡, {4,8}≡, ≢`

Three different cluster partitions in three runs at varying configs
proves the non-determinism is **stochastic ULP-level**, not deterministic
shape-dependent dispatch.

## Root cause identified (Iter 8)

`cudaStreamSynchronize(ctx.stream())` at dispatcher scope-exit errored
with: **"operation not permitted when stream is capturing"**.

The ggml-cuda backend uses **CUDA graph capture** for multi-GPU
execution. This is the architectural root cause:

1. The PSKV dispatcher runs ONCE during graph recording.
2. `ggml_cuda_pool_alloc<T>` RAII allocates scratch on construction
   and releases on dispatcher scope-exit — during recording.
3. The graph then EXECUTES later. Pool memory may have been
   reassigned in the interval.
4. Recorded kernels read/write addresses that no longer hold the
   intended data → stochastic corruption.
5. Different multi-slot NP counts → different concurrent ops in
   the graph → different memory layouts → different cluster patterns.

`cudaMemsetAsync` (iter 9) does NOT fix it — that just zeros memory
during recording, doesn't solve the lifetime issue.

**This affects ANY 2-kernel-per-PSKV dispatcher in this codebase.**
The standard `launch_fattn` template in `fattn-vec-common.cuh` uses
the same RAII pool pattern but works in production because its
specific access patterns may not hit the race, OR because that
template uses graph-capture-aware allocation internally that
`ggml_cuda_pool_alloc` doesn't expose to client code.

## Path forward — three options

### Option 1: Fix scratch lifetime for Variant E (mechanically tractable)

Restructure scratch allocation to survive graph capture. Approaches:
- **Persistent per-device buffer**: allocate once at backend init,
  size for max parallel_blocks × max tensor dims, reuse across calls.
- **Stream-ordered allocation that ggml-graph-aware**: investigate
  ggml's pool internals (`ggml_cuda_pool_leg_alloc` vs newer impls)
  to find a graph-safe allocation path.
- **Embed partial scratch in a tensor**: allocate a graph-resident
  intermediate ggml_tensor for VKQ_parts/VKQ_meta. Won't fragment
  across calls because tensors have graph-scoped lifetime.

If fixable, Variant E ships with +5.7% TG / +17.2% PP. ~2-3 iters
of careful work.

### Option 2: Implement Variant D (WMMA tensor cores)

Single-kernel architecture sidesteps scratch lifetime entirely.
Comprehensive design doc at:
`/home/llm/yarn-agentic/data/pskv-variant-d-wmma-design.md`

Phases:
1. K·Q WMMA scaffold (~150 lines)
2. V_tile + softmax integration (~250 lines total)
3. NPC + bench gates
4. ncu confirms TC utilization

Expected: 50-150 µs/call (4-12× speedup), TG ~32-35 t/s, target hit.

Risk: WMMA on sm_75 uses m16n16k16, NOT the sm_80+ m16n8k8 mentioned
in the original prompt. Existing `fattn-wmma-f16.cuh` provides
reference but is shaped for batched ncols (not our ncols=1 per-slot
use case). ~3-4 iters of careful work + debug.

### Option 3: Accept current state

Singlewarp baseline ships at 27.10 t/s TG / 21.04 t/s PP @ NP=8.
Pre-NPC ceiling (36.68 / 31.91) is unreachable while preserving
multi-GPU NPC. Document the perf cost as the price of byte-identity
serving across NP.

## What this loop produced

Beyond the immediate per-CTA variant exploration:

1. **Root cause for the scratch+combine pipeline failure** — names
   the architectural blocker explicitly. Future kernel work in this
   codebase that uses scratch buffers must address graph-capture safety.
2. **Comprehensive WMMA Variant D design doc** — unblocks a careful
   future implementation against a fixed blueprint.
3. **Diagnostic playbook for cluster-pattern non-determinism** — when
   NP cluster partitions vary stochastically across runs, the bug is
   in the multi-kernel pipeline (not shape-dispatch). Stable cluster
   = deterministic dispatch boundary. (Updates
   `feedback_np_cluster_partition_signature.md`.)
4. **Per-CTA optimization ceiling characterized** — +1-2% TG is the
   max achievable from multi-warp/SMEM/ILP variants without addressing
   the grid undersaturation.

## Files modified vs reverted

All kernel changes reverted to baseline:
- `ggml/src/ggml-cuda/fattn-per-slot-kv-singlewarp-sm75.cu` ← clean

New artifacts (kept):
- `data/perf-ralph-pskv-ledger.md` — 11-iter ledger
- `data/pskv-variant-d-wmma-design.md` — WMMA design doc
- `scripts/quick-pskv-npc-check.sh` — single-GPU NPC smoke helper
- `data/nsys-perf-2026-05-17/` — original kernel-time diff (HEAD vs pre-NPC)

## Memory entries to capture

Worth adding to auto-memory after this session ends:
- **graph-capture-vs-pool-RAII**: `ggml_cuda_pool_alloc` RAII at dispatcher
  scope-exit is unsafe under CUDA graph capture. Any 2-kernel-per-op
  dispatcher pattern needs persistent scratch or graph-safe allocation.
- **Stochastic cluster signature distinction**: stable cluster partitions
  across runs → deterministic shape-dependent dispatch; varying clusters
  → stochastic timing/race-dependent bug.
- **PSKV per-CTA optimization ceiling**: +1-2% TG @ NP=8 plateaus the
  multi-warp/SMEM/ILP family. Real perf needs grid expansion (split-K
  with proper lifetime) or tensor cores.
