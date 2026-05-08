# Synthesis: TU102 sm_75 + Qwen 3.6 27B + ggml-cuda — chain of wins

## What we tried (4 phases, exhaustively)

| Phase | Hypothesis | Outcome | Hard data |
|---|---|---|---|
| 41 | Tree-K MTP (top-K fan-out) → +X% | -29% K=2 vs K=1 baseline | verify cycle scales super-linearly with K (1→2: 1.53×; 2→3: 2.47×) |
| 42 | Identify cost driver via microbench + nsys | Launch-overhead diagnosis (correct) but critical-path attribution (wrong) | per-token kernels scale 1.5×, not bandwidth-bound |
| 43 | NCCL all-reduce + capture unlock | NCCL alone -22% on PCIe x8 (per Agent 1's pre-NVLink warning); capture has cuBLAS-class incompat | nccl symbol linked, runtime confirmed |
| 44 | Capture stability via cache-key fix | Cache-key fix doesn't engage capture; cudaLaunchKernel count essentially unchanged (340k → 327k all-node disc.) | ggml allocator gives different addresses each cycle → cache is graveyard of one-shot entries |

## The actual cost decomposition (hard nsys data)

K=1 cycle ≈ 82ms at 256K X02. By API-time share:

| Component | % wall | Approx ms/cycle | Capturable? |
|---|---|---|---|
| `cudaMemcpyAsync` traffic (KV stream + cross-device reduce data) | 72% (overlapped) | ~10-15ms net | partially |
| `cudaStreamSynchronize` | 11% | ~9ms | no (sync points are critical-path) |
| `cudaLaunchKernel` (host launch overhead) | 11% | ~9ms | YES if capture engages |
| GPU compute kernels (model forward) | residual | ~50-55ms | no (the actual work) |

**Real launch overhead = ~9ms/cycle = 11% of wall.** If capture eliminated all of it: K=1 22.09 → 24.5 t/s = **+11% max**. Not the +50% I'd projected at PHASE43 prep.

(The +50% projection misread the nsys decomposition by treating all "API time" as on-critical-path. Most cudaMemcpyAsync overlaps with kernels and isn't the binding constraint.)

## Why cache-key approach didn't engage capture

Empirically: every cycle's compute splits hit NEW cache entries because ggml's per-cycle scratch allocator gives different node addresses each cycle. Specifically, FUSED_RMS_NORM output and many other ops have data pointers that vary each cycle.

The cache slots become a graveyard of one-shot captures. cudaLaunchKernel count stays at 95-100% of baseline → no replay → no savings.

## Chain of wins (real, ranked by leverage × cost)

### Tier 1 — verified-feasible, modest single-digit wins

1. **Switch `ncclAllReduce` → `ncclReduce` for reduce-to-one cases** (Agent 1's recommendation). Currently both devices receive the broadcast result even when only the master device consumes it. Halves PCIe traffic. Estimated **+1-3%** when NCCL is enabled. Trivial code change. ~3k tokens.
   - Stacks with NVLink upgrade (becomes more impactful when cross-device BW is faster but also less needed).

2. **Pipelining: CPU bookkeeping during GPU verify** (SGLang Zero-Overhead Overlap pattern). After verify GPU kernels submit, run sample/accept logic on CPU while GPU completes. Saves ~5ms of cudaStreamSynchronize wait per cycle. Estimated **+5-7%**. ~15-25k tokens (modify decode loop in `examples/server/server-context.cpp` + speculative.cpp).

3. **Production no-MTP swap.** Current K=1 baseline 22.09 t/s. We measured no-MTP at 22.90 t/s (3.7% better). Production already runs no-MTP per the profile. **+0% over current production** but documents that MTP is throughput-negative. Already in place. 0k.

### Tier 2 — capture-related, requires non-trivial infra work

4. **Indirection extension** (Phase 44 Stage 1, Option 4). `cudaGraphExecKernelNodeSetParams` patches per-cycle pointers without rebuild → cache hits → replay → eliminates cudaLaunchKernel time. Estimated **+5-10%** if it lands cleanly. ~30-50k tokens. Real risk of negative result (cuBLAS incompatibility from PHASE43 ThreadLocal mode might still bite).

5. **Allocator stability** (Phase 44 Stage 1, Option 2). Make ggml's per-cycle scratch deterministic. Properties match → cache hits → replay. ~10-20k tokens but upstream-class change.

### Tier 3 — hardware-gated

6. **NVLink upgrade** (when bridge arrives). Cross-device traffic goes from PCIe 3.0 x8 (~7 GB/s) to NVLink (~50-100 GB/s). Eliminates the cross-device reduce as a major cost. Estimated **+15-25%** + makes Tier 2 wins more achievable. Phase 43's NCCL infrastructure already in place behind cmake flag.

### Tier 4 — model-level (different workstream class)

7. **Kernel fusion at ggml level**: combine multiple ops into single fused kernels (rms_norm + matmul + add, etc.). Reduces both kernel COUNT (less launch overhead absolute) and per-kernel overhead. Significant upstream-class work. Estimated **+10-20%**. ~80-150k tokens.

8. **KV cache compression to IQ4** (currently Q4_0 + Hadamard). Smaller per-token bandwidth. Estimated **+5-10%** at long context. Engineering well-understood. ~20-40k tokens.

9. **Single-GPU draft model spec decode** (separate small draft, verify on TP-split target). Avoids TP overhead entirely on the speculation path. Estimated **+10-15%** at short-medium context where draft cost is significant. ~30-60k tokens. Requires choosing/training a draft model.

## Recommended chain (production-aligned, lowest risk first)

```
Tier 1.1 ncclReduce switch (~3k)             +1-3%       LANDED
   ↓
Tier 1.2 SGLang-style overlap (~20k)          +5-7%       AWAITING TIER 1.1 DATA
   ↓
[Tier 2 only if Tier 1 binds; gates on Stage 1 measurement]
Tier 2.4 Indirection extension (~40k)         +5-10%      RISK: cuBLAS-class incompat
   ↓
[Tier 3 hardware-gated]
Tier 3.6 NVLink + retest                      +15-25%
   ↓
[Tier 4 only if Tier 1-3 insufficient; bigger workstream]
Tier 4.7-9 kernel fusion / KV compression / draft model   +10-20% each
```

**Stacked best case (Tier 1+2+3): ~+25-40%** over current. Achievable but multi-phase.

## What this synthesis lets us do

1. **Stop spinning on capture as the lever.** It's not. Cache-key isn't the fix. Indirection might be but is gated on Tier 2.4 pre-investigation.
2. **Pursue Tier 1 first** — both items are small, well-understood, and stackable. ~25k total for ~+6-10% combined.
3. **Wait for NVLink before re-attempting NCCL/capture work.** Hardware gates Tier 2-3 quality.
4. **Tier 4 is its own multi-month effort** — only if Tier 1-3 leaves us short of production goals.

## What I got wrong in earlier phases

- **PHASE43 +50% projection**: misread nsys API-share as critical-path share. Actual launch overhead is 11% of wall, not 50%. Should have framed PHASE43 as "+5-15% best case" workstream.
- **PHASE44 multi-slot patch claimed correctness fix**: TRUE, but I conflated correctness restoration with capture engagement. Capture wasn't engaging in either state — accept rate was preserved by NOT capturing (the cache fragmentation prevented bad captures from being cached). cudaLaunchKernel count is the binding evidence I should have looked at first.
- **"Capture provides ~0 throughput benefit"**: technically correct conclusion but premature attribution to "hardware doesn't benefit" instead of "capture isn't engaging". The latter is fixable; the former is not.

## Trust-update threshold experiment (post-synthesis correction)

Raised `consecutive_updates >= 4` to `>= 100000` to let capture stabilize. Result:

- nsys K=1: cudaLaunchKernel time **1.37s → 0.46s (-66%)**. cudaGraphLaunch fires 21,145 times. cudaGraphExecUpdate fires 20,999 times. **Capture IS engaging.**
- nsys K=2: cudaLaunchKernel **2.16s → 0.64s (-70%)**. cudaGraphLaunch 33,225, Update 33,019.
- Throughput on noMTP: 22.78 t/s (vs baseline 22.90 — within noise, no benefit despite launch reduction)
- Throughput on K=1 / K=2: **server hangs / empty output**. Capture replays produce wrong/invalid output without proper pointer indirection.

**This is the unambiguous proof:**
- Capture CAN engage on this hardware (refutes "hardware doesn't benefit")
- Engagement saves 66-70% of cudaLaunchKernel time (real)
- But replays are wrong without indirection (cudaGraphExecUpdate alone doesn't catch all moving pointers)
- noMTP throughput unchanged because noMTP has minimal capture targets relative to wall

## Tier 2.4 proper scope (the actual work needed)

Implementation that would work:

1. Modify `evaluate_and_capture_cuda_graph` (`ggml-cuda.cu:4720+`) to call `cudaGraphGetNodes` after `cudaStreamEndCapture` and populate the existing `graph->nodes` (already in graph.cuh:26 but unused).
2. For each captured node, capture its kernel params (`cudaGraphKernelNodeGetParams`) into `graph->params` (graph.cuh:27, also unused).
3. Map each captured node back to its originating ggml_tensor via order-preserving traversal (capture preserves kernel-launch order per CUDA docs).
4. At replay time, BEFORE `cudaGraphLaunch`: iterate `graph->nodes`, get each tensor's CURRENT data pointer from the live cgraph, call `cudaGraphExecKernelNodeSetParams` to update.
5. Skip the live re-execute path when `graph->instance != nullptr`.

The data structures exist (someone planned this). The logic is missing. Estimated 30-50k tokens of careful work + extensive testing because cudaGraphExecKernelNodeSetParams has strict requirements (param sizes must match exactly).

## Final recommendation (revised)

The "hardware doesn't benefit from capture" framing was wrong. **Capture provides real benefit (~+11% launch overhead recovery + replay savings) IF we land the indirection infrastructure.** The path is:

- Tier 2.4 proper indirection: 30-50k tokens, real engineering, real win
- Stack with NVLink (Tier 3) when bridge arrives
- Then re-evaluate Tier 1.2 overlap on the captured baseline

The next session should start with Tier 2.4 implementation. The data structures in graph.cuh suggest this was upstream work that got stalled — picking it up would be productive.

## Recommendation

Implement Tier 1.1 (ncclReduce switch) now — small, low risk, real win, and unlocks the assumption that NCCL infrastructure can be tuned. Then Tier 1.2 (overlap scheduler).

Tier 2 on capture indirection only if user wants to invest 30-50k tokens with non-trivial risk of negative outcome.

Production stays on `phase41-tree-foundation` baseline until Tier 1 lands and binds.
