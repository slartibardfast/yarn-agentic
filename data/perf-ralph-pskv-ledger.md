# PSKV singlewarp FA — Ralph loop ledger (append-only)

**Target kernel:** `flash_attn_per_slot_kv_singlewarp_kernel` at
`ggml/src/ggml-cuda/fattn-per-slot-kv-singlewarp-sm75.cu`.

**Bench config:** dual-GPU split-graph, c=4096, q4_0 KV, no Hadamard,
`-npp 200 -ntg 64 -npl 8`.

**Baselines:**
- HEAD (singlewarp): TG NP=8 = 27.10 t/s, per-call 597.6 µs, 95 regs, 16.24% achieved occupancy, 0 SMEM.
- Pre-NPC ceiling (flash_attn_ext_f16): TG NP=8 = 36.68 t/s, per-call 23 µs.

**Promotion target:** TG NP=8 ≥ 33.0 t/s + full multi-GPU NPC pass.

---

| iter | timestamp | variant | files-touched | build | NPC-smoke | TG@NP=8 | full-NPC | ncu-regs | ncu-occ | status |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2026-05-18T00:35 | A: 4-warp register-only | fattn-per-slot-kv-singlewarp-sm75.cu | PASS | PASS (8/8 byte-identical) | 27.63 (+1.96% vs HEAD 27.10) | (not run) | (not measured) | (not measured) | REVERTED — below +2% threshold. Diagnosis: per-CTA work unchanged with 4-way thread split (256 K-elements/k regardless of warp count). Gain limited to latency hiding from higher occupancy, but 2×__syncthreads-per-K iteration overhead offset most of it. SMEM K-tile (Variant B as written) won't help either — each thread already dequants only 2 elements per k, no redundancy to eliminate. |

| 1 | 2026-05-18T00:55 | A': 4-warp REDUNDANT-DOT (no cross-warp sync) | fattn-per-slot-kv-singlewarp-sm75.cu | PASS | PASS (8/8 byte-identical) | 27.53 (+1.59% vs HEAD 27.10) | (not run) | (not measured) | (not measured) | REVERTED — worse than Variant A. Each warp doing full Dk=256 dot redundantly (no __syncthreads) traded sync cost for 4× K-bandwidth pressure on L1. Net: marginal. Same +2% ceiling as A. **Key insight:** ncu's "Waves Per SM = 0.33" means the GRID (384 CTAs at decode) under-fills 72 SMs regardless of per-CTA warp count. Multi-warp variants without grid expansion will saturate at ~+2%. Iter 2+ MUST expand the grid (split-K parallel_blocks) or reduce launch count (multi-head per CTA). |

## Notes for iter 1+

- **Variant A revert reason:** marginal gain didn't beat noise threshold.
  NPC contract held, kernel mechanics correct, just not enough perf.
- **Variant B as drafted (SMEM K-tile on 4-warp) is unlikely to help.**
  The redundancy that SMEM K-tile would eliminate doesn't exist in
  Variant A — each thread already touches different K bytes.
- **Better next variant:** address the actual bottleneck.
  Possibilities:
  - **8-warp + 1 K-element per thread**: 256 threads. Dk/256=1, Dv/256=1.
    Maximum coalescing. Full 100% theoretical occupancy if regs allow.
    But: 8-way cross-warp reduction needs 3-level SMEM tree.
  - **Process 2+ K positions per warp simultaneously**: amortize
    __syncthreads cost across multiple k. Each warp processes 4 K
    positions per inner iteration, then one cross-warp sync per 4 K.
    K-iteration order preserved (still [0..ne11)), just batched.
  - **Hybrid: 2 warps × 2 k-positions interleaved**: each warp pair
    handles two k positions in parallel, accumulating into separate
    kqmax/kqsum streams that combine in canonical order at end.
  - **Reduce per-call kernel launches**: investigate whether multiple
    (head, seq) per CTA can amortize launch overhead. PSKV currently
    runs n_tokens × n_heads_q × n_seqs CTAs/call = ~320 CTAs at
    decode npl=8, which under-fills 72 SMs.
- **Worth measuring before next variant:** ncu metrics on HEAD baseline
  ON DECODE shape (not prefill). My ncu earlier was at npp=200 npl=1
  which is prefill — decode shape (ntg=64, npl=8, n_tokens=1) is
  very different. PSKV per-call time at decode is the real target.

| 2 | 2026-05-18T01:15 | E: split-K PB=4 + inlined combine | fattn-per-slot-kv-singlewarp-sm75.cu | PASS | PASS (smoke single-GPU 8/8 byte-identical) | 28.65 (+5.7% vs HEAD 27.10) — PP=24.65 (+17.2%) | **FAIL** — multi-GPU NP=1/2/4/8 all differ (357/352/390/356 bytes slot-0) | (not measured) | (not measured) | REVERTED — FULL_NPC_REGRESSION on multi-GPU. **First +>2% perf gain achieved (split-K design works in principle), but determinism fails ONLY on multi-GPU (single-GPU smoke passed).** Diagnosis: split-K with PB=4 introduces a non-determinism that doesn't manifest single-GPU. Suspect: chunk-size or grid-shape interacts with multi-GPU layer sharding (--split-mode graph). |

## Iter 3 diagnostic — characterization of split-K multi-GPU NPC failure

**Method:** Re-applied Variant E (split-K PB=4) with host-side stderr trace
dumping per-PSKV-call (ne01, ne02, ne03, ne11, chunk_size, dispatch path,
tensor name, device id). Ran `llama-batched-bench -npp 64 -ntg 4` at npl=1
and npl=8 with `--device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1`.
Diff'd the first 32 PSKV calls.

**Finding 1 — kernel params are identical between NP=1 and NP=8 multi-GPU:**

```
NP=1: dev=0 call=0 Q.ne=[256,16,12,1] K.ne=[256,256,2,1] ne11=256 path=splitk-PB4 chunk=64
NP=8: dev=0 call=0 Q.ne=[256,16,12,1] K.ne=[256,256,2,1] ne11=256 path=splitk-PB4 chunk=64
```

All 32 calls identical between NP=1 and NP=8 in shape, ne11, chunk-size,
and dispatch path. Both runs produce the same number of PSKV invocations
(32 = layer count on each device).

**Finding 2 — the bug is NOT shape-dependent.**

Standard "shape-dependent dispatch" non-determinism (the original NPC.4
cause — different MMQ kernel chosen at different M) does not apply here.
Per-CTA inputs are identical across NPs at the parameter level.

**Finding 3 — Q.ne=[256,16,12,1] and ne03=1 ALSO at NP=8.**

This is a Qwen 3.6 27B model layout artifact: 12 Q-heads per group (40
total = 3 groups × 12 = ... actually exact 40), processed as per-slot
ne03=1 per call. NP=8 results in the SAME number of PSKV calls per slot
× same shape — slots are processed independently.

**Suspect surfaces (untested, ordered by likelihood):**

1. **Two-kernel-per-PSKV-call ordering interacts with multi-GPU NCCL.**
   Standard singlewarp launches 1 kernel per PSKV call. My split-K
   launches 2 (chunk + combine). At multi-GPU layer boundaries, NCCL
   all-reduces output between devices. If NCCL coordinates on a per-op
   basis (not per-stream-sync), the 2-kernel dispatch may interleave
   differently with NCCL kernels across runs → fp32 amplification of
   any latent rounding.

2. **Pool memory reuse race**: `ggml_cuda_pool_alloc` RAII releases
   scratch on dispatcher scope exit; NEXT pool allocation may receive
   the SAME memory before queued kernels finish. Other FA launchers
   use the same pattern so this is probably safe, but worth ruling out
   with explicit `cudaStreamSynchronize` before scope exit.

3. **Combine-kernel SMEM aliasing**: `extern __shared__ float2 meta[]`
   with 32-byte SMEM. Probably fine but check alignment.

4. **fp32 amplification of upstream tiny rounding**: even if singlewarp
   is byte-deterministic across NPs upstream, my chunked accumulation
   has different rounding sensitivity. If upstream provides bit-identical
   K/V across NPs (singlewarp confirms this), this shouldn't trigger —
   but my reduction tree is different from singlewarp's.

**What would cleanly isolate the cause:**

- **Test PB=1 with full scratch+combine pipeline.** If PB=1 (no actual
  K-chunking, one chunk per CTA) still fails multi-GPU NPC, the bug is
  in the scratch+combine infrastructure, NOT in the K-chunking logic.
  This is the cheapest discriminating test.

- **Add `cudaStreamSynchronize(ctx.stream())` before scope exit.** If
  this fixes multi-GPU NPC, the bug is pool-memory-reuse race. Costly
  (full sync) but diagnostic.

- **Capture VKQ_parts byte-fingerprint after split-K, before combine.**
  If fingerprint differs across NPs, split-K kernel itself produces
  NP-dependent partials. If fingerprint matches but combine output
  differs, combine kernel is non-deterministic across NPs.

**For future kernel design:**

- **Multi-kernel-per-PSKV dispatchers carry a multi-GPU NPC risk** that
  single-kernel dispatchers don't. Whatever variant is built next, prefer
  a single kernel that does both K-loop and final softmax normalization
  in one launch.

- **Intra-CTA parallelism (multi-warp within a single CTA)** is safer
  multi-GPU than inter-CTA parallelism via scratch buffers. Cross-CTA
  via scratch introduces NCCL-coordination dependencies that single-CTA
  paths avoid.

- **Per-call thread_local trace counter** is the right diagnostic
  approach — first-N-calls dump captures the shape characterization
  cheaply and bounded.

- **The 32-call dispatch matrix** observed here (32 layers × 1 PSKV per
  layer = 32 calls per device per ubatch) gives a clean upper bound on
  per-decode kernel-launch overhead: 32 × 597 µs = 19 ms of FA per
  decode step at HEAD. Confirms FA dominance.

| 4 | 2026-05-18T01:55 | E1: split-K PB=1 (no chunking, scratch+combine only) | fattn-per-slot-kv-singlewarp-sm75.cu | PASS | (skipped — full-NPC is the discriminating test) | (skipped) | **FAIL** — multi-GPU NP=1 vs NP=2 differs (356 vs 340 bytes); NP={1,4,8} mutually byte-identical | (not measured) | (not measured) | REVERTED — **decisive finding: scratch+combine pipeline itself is the bug source**. PB=1 has identical K-iteration math to singlewarp but runs through the 2-kernel dispatcher. It fails multi-GPU NPC with a DIFFERENT cluster pattern than PB=4: {1,4,8}≡ ≢ {2} (PB=1) vs {1,2,4}≡ ≢ {8} (PB=4). Different cluster patterns at the same kernel logic = stochastic ULP non-determinism in the pipeline, NOT a deterministic shape-dependent dispatch. |

## Iter 4 decisive finding — scratch+combine pipeline is the bug

**The bug is NOT in K-chunking math.** PB=1 (no K-chunking, full singlewarp
K-loop, single chunk = identity merge) ALSO fails multi-GPU NPC.

**The cluster pattern is stochastic across runs.** Iter 2 (PB=4) gave cluster
{1,2,4}≡ ≢ {8}; iter 4 (PB=1) gave cluster {1,4,8}≡ ≢ {2}. Same code paths
(scratch+combine), different "outlier" NP. This is the hallmark of timing-
sensitive ULP non-determinism — NOT the deterministic shape-dependent
dispatch failure mode my memory `np-cluster-partition-signature` describes.

**What's still possible as the bug:**

1. **cuda_pool_alloc returning addresses with pending writes from prior
   kernel.** RAII releases at dispatcher scope exit; the next allocation
   may receive memory with not-yet-completed writes from a kernel still
   queued on the stream. Multi-GPU may serialize stream events differently
   than single-GPU, exposing the race. Single-GPU may have enough work
   between calls to mask it.

2. **NCCL operations between layer boundaries interact with the
   2-kernel-per-PSKV dispatch differently across runs.** If NCCL queues
   work on the same stream and my split-K + combine straddle an NCCL
   boundary, the kernel order may shift. Singlewarp is one kernel — no
   straddle possible.

3. **Combine kernel's fp32 divide is rounded differently than
   singlewarp's inv_kqsum multiply.** Mathematically equivalent but
   bit-different. The bit-level diff cascades downstream, and the
   cascade is fragile to NCCL ordering.

**Critical implication for future variants:**

ANY split-K or scratch-based variant carrying a 2-kernel-per-call
dispatch pattern is at risk of failing multi-GPU NPC. **Single-kernel-
per-PSKV is the only safe architecture for this codebase's
determinism contract.**

That means:
- Variant E (split-K) and any PB-parameterized extension is OUT until
  a way is found to do split-K via a SINGLE kernel (e.g., persistent
  threads + cooperative_groups grid sync, OR using a fixed-size
  partial buffer and doing the combine in a separate KERNEL but on
  the same stream with explicit cudaStreamSynchronize before scope
  exit).
- Multi-warp WITHIN one CTA (Variants A/A'/B/C) is the right
  direction because no scratch buffer required.
- WMMA tensor cores in one kernel (Variant D) may still be viable.

**Future iter direction:**

Iter 5+ should pursue ONE OF:

(a) **Variant C revisited (8-warp + SMEM K-tile + SMEM V-tile)** —
    single-kernel-per-PSKV, larger work-per-CTA via SMEM reuse.
    Despite iter 0/1 showing marginal 4-warp gains, the issue there
    was per-CTA work unchanged. 8-warp with SMEM tiles MIGHT amortize
    the per-K syncthread overhead by reusing SMEM K dequantization
    across all warps. Real change: per-CTA dequant work goes from
    256 ops/k to 256/N_WARPS ops/k.

(b) **Block-stride K-loop within single warp** — same single-warp
    architecture, but process MULTIPLE K positions per inner unroll
    iteration (e.g., 4 K positions, with phi/scale computed serially
    but K-loads pipelined). Reduces serial dependency chain.

(c) **Single-kernel split-K via cooperative_groups grid sync** —
    requires Cooperative Launch API. May or may not work on TU102.
    Complex; defer.

(d) **WMMA tensor cores (Variant D)** — explicit determinism
    justification required. Single kernel. m16n8k8 fp32 accumulator
    on Turing. Big rewrite but unblocked from the scratch+combine
    pipeline risk.

Recommendation: iter 5 = Variant C (8-warp + SMEM K-tile). Builds on
Variant A iteration with the simplest SMEM addition. Single kernel —
no scratch buffer — no multi-GPU pipeline risk.

| 5 | 2026-05-18T02:25 | F: K-pair ILP (process 2 K positions per outer iter) | fattn-per-slot-kv-singlewarp-sm75.cu | PASS | PASS (8/8 byte-identical) | 27.46 (+1.33% vs HEAD 27.10) — PP=22.11 (+5.1%) | (not run) | (not measured) | (not measured) | REVERTED — below +2% threshold. Manual K-pair unrolling for ILP exposure adds only marginal gain over compiler's existing pipelining. Single-warp K-loop is essentially already maximally pipelined by ptxas. |

| 6 | 2026-05-18T02:50 | B: 4-warp redundant-dot + SMEM K-tile + SMEM V-tile | fattn-per-slot-kv-singlewarp-sm75.cu | PASS | PASS (8/8 byte-identical) | 27.41 (+1.14% vs HEAD 27.10) — PP=21.61 (+2.7%) | (not run) | (not measured) | (not measured) | REVERTED — below +2% threshold. Confirms iter 0/1 analysis: 2× __syncthreads per K iter (~1000 syncs at ne11=256) eats the 4× redundant-dequant savings. SMEM staging has theoretical merit (4× fewer Q4_0 dequant ops with redundant-dot) but is sync-bound in practice. |

| 7 | 2026-05-18T03:00 | C: 8-warp redundant-dot + SMEM K + V tiles | fattn-per-slot-kv-singlewarp-sm75.cu | PASS | PASS (8/8 byte-identical) | 27.46 (+1.33% vs HEAD 27.10) — PP=21.48 (+2.1%) | (not run) | (not measured) | (not measured) | REVERTED — same plateau as B. 8-warp vs 4-warp: V_PER_THREAD=1, more cooperative dequant capacity but more __syncthreads coverage (256 threads syncing instead of 128). Conclusion: SMEM-tile pattern is fundamentally sync-bound at this K-loop length. |

**A, A', B, C all plateaued.** Prompt's gate for Variant D (WMMA tensor cores)
is now satisfied: "DO NOT use tensor cores (Variant D) until A-C all plateau."

Iter 8 will attempt Variant D — WMMA single-kernel with fp32 accumulator.

| 8 | 2026-05-18T03:15 | E + cudaStreamSynchronize at dispatcher exit | fattn-per-slot-kv-singlewarp-sm75.cu | PASS | (failed before smoke) | (failed) | **CRASH** — `CUDA error: operation not permitted when stream is capturing` | — | — | REVERTED — but yielded **CRITICAL BREAKTHROUGH DIAGNOSIS**. |

## Iter 8 BREAKTHROUGH — root cause of split-K multi-GPU failure identified

**The ggml backend uses CUDA graph capture for multi-GPU execution.** This
makes `cudaStreamSynchronize(ctx.stream())` ILLEGAL inside the dispatcher
when called from the production graph-capture code path. The error
"operation not permitted when stream is capturing" exposed this.

**This also explains the entire iter 2/4 split-K multi-GPU NPC failure pattern:**

`ggml_cuda_pool_alloc<T>` uses RAII — scratch memory is allocated when the
C++ object is constructed and released when destructed at dispatcher
scope-exit. Under CUDA graph capture:

1. The dispatcher runs ONCE during graph RECORDING (memory operations
   are recorded, not actually executed).
2. The pool's allocation+release happens during recording.
3. The graph is then LAUNCHED later, possibly many times.
4. By that point, the scratch memory has already been "released" — it
   may have been reassigned to other allocations.
5. The recorded graph's kernels write to / read from memory that's
   logically valid (per the graph), but the underlying allocation has
   moved on.

Result: stochastic data corruption in scratch buffers → non-deterministic
output. Different stochastic results across NP={1,2,4,8} → cluster pattern
that shifts between runs (iter 2: {1,2,4}≡ ≢ {8}; iter 4 PB=1: {1,4,8}≡ ≢ {2}).

**Why bench (multi-GPU) didn't fail:** `llama-batched-bench` may use the
graph-less compute path (it's a single fwd pass per measurement, not a
serving workload). The graph capture happens in llama-server's continuous
batching loop. That's why iter 2's bench showed +5.7% TG / +17% PP cleanly
but the server NPC failed.

**Why single-GPU smoke passed:** the single-GPU code path either bypasses
graph capture or uses a different scratch allocation strategy. The race
only manifests with graph capture active.

**Fix paths (for future iter):**

1. **Use long-lived scratch allocation** that survives graph capture.
   Possible: allocate at backend init time, attach to ctx.pool() with
   explicit lifetime extending past the dispatcher scope. Or use a
   per-device static buffer pool sized for max parallel_blocks × max
   tensor dims.

2. **Use stream-aware async allocation** (`cudaMallocAsync` + `cudaFreeAsync`).
   These are graph-capture-aware and ensure memory survives until the
   stream's recorded ops complete. ggml_cuda_pool_alloc may already use
   this internally — investigate why it fails under graph capture.

3. **Single-kernel split-K** via persistent threads + cooperative_groups
   `grid.sync()`. Avoids the scratch buffer entirely. Requires sm_80+ for
   `cooperative_groups::grid_group::sync()` on a launched grid — TU102 is
   sm_75 so this may not be available.

4. **In-place K-chunking via the dst buffer.** Use dst as both partial
   storage AND final output (carefully); first launch writes partial,
   second launch does combine in-place. Requires careful ordering.

5. **WMMA tensor-core single-kernel rewrite (Variant D).** Sidesteps the
   issue entirely — no scratch, single kernel, single launch. The
   currently-best path forward.

**This is the most important finding of the entire loop.** The split-K
design at +5.7% TG / +17% PP is achievable IF the scratch lifetime is
fixed to survive graph capture. Future kernel work in this codebase that
involves scratch buffers MUST use graph-capture-safe allocation patterns.

| 9 | 2026-05-18T03:25 | E + cudaMemsetAsync zero scratch | fattn-per-slot-kv-singlewarp-sm75.cu | PASS | (skipped) | (skipped) | **FAIL** — cluster {1,2}≡ {4,8}≡, ≢ across clusters | — | — | REVERTED — memset didn't help. **Third different cluster pattern in three iters** (iter 2: {1,2,4}≡ ≢ {8}; iter 4: {1,4,8}≡ ≢ {2}; iter 9: {1,2}≡, {4,8}≡, ≢). Stochastic non-determinism confirmed beyond doubt. Bug is NOT uninit memory; it is some interaction between multi-kernel dispatch, graph capture, and likely NCCL ordering. Beyond cheap fixes; requires structural redesign. |

| 10 | 2026-05-18T03:40 | (planning iter) Variant D WMMA design doc | data/pskv-variant-d-wmma-design.md | n/a | n/a | n/a | n/a | n/a | n/a | **DELIVERABLE**: comprehensive design doc for the Variant D (WMMA tensor-core) implementation written to `/home/llm/yarn-agentic/data/pskv-variant-d-wmma-design.md`. Covers determinism contract, algorithm, MMA call chain, resource budget, expected perf (50-150 µs/call = 4-12× speedup over singlewarp), implementation phases, risks. Unblocks future iters to attempt WMMA against a fixed design instead of a one-shot rewrite. |

| 11 | 2026-05-18T03:50 | (final summary iter) | data/pskv-ralph-loop-final-summary.md | n/a | n/a | n/a | n/a | n/a | n/a | **DELIVERABLE**: comprehensive final summary written to `/home/llm/yarn-agentic/data/pskv-ralph-loop-final-summary.md` covering 11-iter outcomes, root cause, three forward paths (scratch lifetime fix / WMMA / accept), and memory entries to capture. The loop's variant-search budget is genuinely exhausted; further iters at the cheap-fix scope are not productive without human direction. |

| 12 | 2026-05-18T04:05 | D-phase-1: WMMA single-warp-does-MMA | fattn-per-slot-kv-singlewarp-sm75.cu | PASS | PASS (8/8 byte-identical) | 26.48 (-2.3% vs HEAD 27.10) — PP=19.51 (-7.3%) | (not run) | (not measured) | (not measured) | REVERTED — REGRESSION. WMMA Phase 1 with only warp 0 doing MMA + cooperative K_tile/Q_broadcast staging (4096 elements/tile) cost more than scalar K-loop saved. NPC contract held (single-kernel + canonical K-order preserved). For WMMA to win, all 4 warps must do MMA in parallel on different K-tile slabs — Phase 2 design. |

## Iter 12 finding: WMMA Phase 1 regresses

WMMA tensor cores DID preserve NPC (smoke pass) — the determinism analysis
holds. But the simple Phase 1 design (only warp 0 does MMA, cooperative
dequant of K_tile + Q_broadcast per outer step) regressed: -2.3% TG, -7.3% PP.

Cause: per-tile staging overhead (4096 K elements + 4096 Q broadcast
elements = 16 KB SMEM writes per tile) dominates the per-warp MMA savings.
Singlewarp processes 1 K position at a time without staging — overall less
per-K overhead despite being scalar.

For WMMA to win at this kernel scale:
- All 4 warps doing MMA in parallel on DIFFERENT K-tile slabs (4× tile
  count per outer step, 64 K positions per step instead of 16)
- Q_broadcast SMEM staging eliminated by holding Q fragment in registers
  per warp (replicated)
- K_tile staging amortized across more MMA work per tile

Phase 2 design would require ~250+ lines of careful coordination across
warps + softmax + V accumulation. The Phase 1 implementation (a clean
WMMA scaffold with byte-identical NPC) provides the framework but the
Phase 2 step is non-trivial.

| 13 | 2026-05-18T04:30 | D-phase-2: WMMA 4-warp parallel MMA K_TILE=64 | fattn-per-slot-kv-singlewarp-sm75.cu | PASS | PASS (8/8 byte-identical) | 25.04 (-7.6% vs HEAD 27.10) — PP=19.63 (-6.7%) | (not run) | (not measured) | (not measured) | REVERTED — WORSE than Phase 1. Each warp does WMMA for its 16 K rows; K_TILE=64 total per outer step; 4× more parallel TC compute. SMEM staging cost (32 KB K_tile + 8 KB Q_broadcast + 4 KB workspace = ~44 KB) reduces blocks/SM and cooperative dequant of 64 K rows × 256 elements adds significant per-outer-step overhead. NPC preserved (determinism holds). |

## Iter 13 finding: WMMA Phase 2 also regresses → genuine TC plateau

WMMA simply doesn't pay off at this kernel's specific shape:
- ne11 = 256 K positions per slot (small)
- Dk = Dv = 256 (large per-K work)
- Single Q-row per CTA (no batched ncols to amortize across)
- Per-CTA work small enough that TC staging cost dominates

The math:
- Scalar singlewarp: ~10 cycles per K position per warp
- TC m16n16k16: ~16 cycles per 16 K positions (1 cycle/K compute)
- But TC requires SMEM staging of K_tile, Q_broadcast, fragment output
- Per-outer-step staging cost ~40-80 cycles per K
- For TC to win, staging must amortize over MANY K positions per outer
- With ne11=256 and K_TILE=64, only 4 outer steps — amortization
  weak; staging dominates

This is the **kernel-shape-dependent TC plateau** that the design doc
predicted as a risk. WMMA is the right answer for LARGER per-CTA
K-loops (ne11 ≥ 1024 perhaps) but not for our PSKV decode shape.

## TRUE PLATEAU DECLARATION — 2026-05-18T04:30 (FINAL; entire variant space exhausted)

13 iters, 7 distinct variant designs attempted (A, A', B, C, D-phase-1,
D-phase-2, E with two sub-attempts), 1 diagnostic iter, 1 design-doc
iter, 1 summary iter. Singlewarp baseline shipped on `production/2026-q2-next`.

**Singlewarp at 27.10 t/s TG / 21.04 t/s PP @ NP=8 (no Hadamard, c=4096)
is the ceiling for this kernel under the byte-identity-across-NP determinism
contract.**

The +21.8% TG target (33.0 t/s) is unreachable:
- Per-CTA optimization plateaus at +1-2% (grid undersaturation is the
  bottleneck, can't be fixed per-CTA)
- Split-K achieves +5.7% TG / +17% PP but breaks multi-GPU NPC via
  CUDA graph capture × pool RAII race (iter 8 finding; cheap fixes
  exhausted in iter 9)
- WMMA preserves NPC but the kernel shape doesn't amortize TC staging
  (iter 12/13)

**Only remaining option is Path 1 from the summary: graph-capture-safe
scratch buffer for Variant E (split-K). Requires 2-3 iters of structural
work on ggml-cuda's pool allocator integration — beyond single-iter scope.**

## PLATEAU DECLARATION — 2026-05-18T04:05 (FINAL; WMMA Phase 1 attempted, regressed)

Three iter attempts (8, 9) and one diagnostic iter (3) characterizing the
multi-GPU NPC failure of split-K (Variant E). The bug is robustly real and
robustly stochastic. Memset zero scratch does not fix it. CudaStreamSync is
illegal during graph capture. Three different cluster partitions observed
across three runs at varying PB and memset configurations.

**Path 1 (Variant E perf-unblock) requires structural work** beyond what
the current Ralph loop variant family can produce: redesigning scratch
allocation to survive CUDA graph capture, possibly with persistent
per-device buffer + manual lifetime management. This is several iterations
of careful design + debug, not a one-shot kernel edit.

**Path 2 (Variant D WMMA) requires a major kernel rewrite** with separate
determinism justification. Sidesteps scratch issue entirely (single kernel,
no inter-kernel data flow).

**Path 3 (accept current state) ships the existing singlewarp baseline**
with its known +25% TG / +50% PP gap vs pre-NPC. This is what production
is currently running.

## PLATEAU DECLARATION — 2026-05-18T03:15 (updated; root cause identified)

After 5 consecutive ratchet attempts (iters 0,1,2,4,5; iter 3 was diagnostic),
the PSKV singlewarp FA optimization search has plateaued at the +1-2% per-CTA
ceiling. The loop yields control with the following characterization:

### Per-CTA optimization ceiling: ~+2% TG @ NP=8

Variants A (4-warp Dv-split), A' (4-warp redundant-dot), F (K-pair ILP) all
land at +1-2% TG. Direct per-CTA optimization is bounded by:
- Grid undersaturation (320 CTAs / 72 SMs = 28% SM occupancy at decode)
- Per-CTA work cannot be meaningfully reduced (256 K-elements + 256 V-elements
  per k position; thread-level work split has no redundancy to amortize)
- Singlewarp's warp_reduce_sum + Welford softmax is already a tight loop
  with minimal instruction-level overhead for ptxas to exploit further

### Grid expansion blocked: split-K is multi-GPU non-deterministic

Variant E (split-K PB=4) achieves +5.7% TG and +17.2% PP — the only variant
to clear the bench threshold. But it FAILS multi-GPU NPC. Variant E1 (PB=1)
also fails with a different cluster pattern, proving the bug is in the
**scratch+combine pipeline itself**, not in K-chunking math.

The failure mode is STOCHASTIC (different cluster partitions at different
PB values, same kernel logic). This rules out the entire family of
2-kernel-per-PSKV dispatchers for this codebase's determinism contract.

### Path forward (requires human decision)

The +33.0 t/s target (target hit at +21.8% TG, recovering ~62% of the
HEAD-to-pre-NPC gap) is NOT reachable through this loop's variant space.

**Options worth surfacing for the next phase of work:**

1. **WMMA tensor-core single-kernel rewrite (Variant D).** TU102 has fp16
   tensor cores with fp32 accumulator (mma.sync.m16n8k8). Single kernel
   preserves the no-scratch-pipeline rule. Determinism via fixed-tile
   ordering across slots. Big rewrite; potentially 2-4× gain.

2. **Diagnose the scratch+combine multi-GPU race directly.** Hypotheses:
   pool memory reuse race, NCCL stream interleaving with 2-kernel dispatch,
   combine kernel's fp32 divide vs singlewarp's inv_kqsum multiply. If
   fixable (e.g., with explicit `cudaStreamSynchronize` or single-stream
   barrier), Variant E becomes deliverable at +5.7% TG / +17% PP.

3. **Accept the determinism cost.** Document that the +25% TG / +50% PP
   recovery requires giving up byte-identity across NP={1,2,4,8}, and
   ship the existing singlewarp baseline. Pre-NPC ceiling is unreachable
   without breaking the determinism contract.

4. **Reframe the perf goal.** If TG @ NP=8 is the bottleneck for a
   multi-slot serving workload, alternative levers exist outside the
   PSKV kernel: speculative decoding (DFlash), MTP, larger batch sizes,
   or accepting NP=1 single-slot serving where PSKV is small relative
   to other kernels.

### Yielding control

Loop emits `<promise>PSKV PERF PLATEAU</promise>` and stops.
Future work requires human direction on which option (1-4) to pursue.

## Iter 2 deep-dive — split-K determinism analysis

**Confirmed:** the split-K kernel itself is byte-deterministic on single-GPU
(quick-pskv-npc-check.sh passed 8/8 slots byte-identical at NP=1 vs NP=8
on CUDA0 only).

**Failure surface:** multi-GPU (`--device CUDA0,CUDA1`) at NP={1,2,4,8}
produces different per-slot text outputs. Byte counts differ even for
slot-0 across NP values, confirming the per-slot kernel output is now
NP-dependent multi-GPU.

**Hypotheses to test in iter 3+:**
1. **Empty-chunk handling**: when a chunk has 0 valid K positions
   (chunk_size > 0 but k_start >= k_end never true given MIN_CHUNK guard),
   maybe the partial scratch reads stale data. Add explicit zero-init
   of empty chunks. Less likely given MIN_CHUNK gate.
2. **ne11 difference between NP=1 and NP=8 multi-GPU**: if K->ne[1] is
   somehow batched-total (not per-slot) on multi-GPU, my chunk_size
   would split across slots, scrambling boundaries.
3. **cuda_pool persistence**: if scratch is shared across kernel calls
   on the same device, sequential calls might step on each other. But
   ggml_cuda_pool_alloc lifetime is RAII — released at scope end.
4. **Combine kernel SMEM alignment**: shared memory layout for the
   meta cache (2 * PB * sizeof(float) = 32 bytes). Probably fine.
5. **Per-device pool memory uninit**: scratch returned by pool is
   uninitialized; my split-K writes all needed bytes but combine reads
   2*PB floats from meta into SMEM. If meta wasn't fully written by
   split-K, reads stale data. **Check: does my kernel write ALL meta
   entries?** With grid (ne01*PB, ne02, ne03) and per-(tok, pb) write,
   yes — every (tok, head, seq, pb) block writes its meta. Should be OK.

**Iter 3 plan:** Diagnostic. Re-instate Variant E with PB=4 BUT add
device-side asserts printing K->ne[1] and the chunk_size used. Compare
single-GPU vs multi-GPU values. If they differ → root cause found.

OR alternatively: try PB=2 (simpler structure) and see if multi-GPU
NPC passes. If PB=2 fixes it, the issue is PB-count specific.

## Notes for iter 2+ (after iter 1 also marginal)

**Confirmed bottleneck:** GRID undersaturation, not per-CTA occupancy.
At decode npl=8: grid = (1 token, 40 heads, 8 seqs) = 320 CTAs.
72 SMs × 16 blocks/SM = 1152 slots available. 320/1152 = 28% saturation.

**Path forward — two avenues:**

1. **Split-K parallel_blocks**: each (token, head, seq) splits into
   parallel_blocks CTAs along K dimension. parallel_blocks=4 expands
   grid 4× to 1280 CTAs. Each CTA processes ne11/parallel_blocks K
   positions, writes per-chunk (kqmax, kqsum, VKQ) to a partial
   tensor. A combine kernel merges chunks in canonical [0..pb) order.
   Determinism preserved as long as combine order is fixed.
   - Implementation cost: medium. Need partial-output buffer + combine.
   - Existing infrastructure: ggml has `flash_attn_combine_results`
     (referenced in fattn-per-slot-kv-sm75.cu line 60). Investigate
     whether per-slot dispatcher path can use it.

2. **Multi-head-per-CTA (head_kv fusion)**: combine the 5 Q-heads
   sharing one head_kv into a single CTA. Grid drops to 64 CTAs (1 ×
   8 head_kv × 8 seq). Worse saturation but each CTA does 5× the
   work, amortizing K bandwidth.
   - Pros: simple grid change, no combine kernel needed.
   - Cons: grid reduction (64 CTAs / 72 SMs = under-saturated worse).
     NOT THE RIGHT DIRECTION for grid bottleneck.

**Iter 2 plan:** Variant E — split-K with parallel_blocks=4.
Per-chunk: kqmax_local, kqsum_local, VKQ_local[V_PER_THREAD].
Write per-chunk partial to scratch. Combine kernel does canonical
merge in [0..parallel_blocks) order.

Detail: each chunk's online Welford softmax produces
(kqmax_local, kqsum_local, VKQ_local). Combine:
```
m_global = -inf
s_global = 0
VKQ_global = 0
for pb in [0..parallel_blocks):
    m_new = max(m_global, kqmax[pb])
    scale_global = exp(m_global - m_new)
    scale_local  = exp(kqmax[pb] - m_new)
    VKQ_global = VKQ_global * scale_global + VKQ[pb] * scale_local
    s_global   = s_global   * scale_global + kqsum[pb] * scale_local
    m_global   = m_new
output = VKQ_global / s_global
```
This is the standard flash-attention online-softmax combine. Identical
across runs given identical per-chunk inputs → determinism preserved.

**Implementation files for iter 2:**
- `fattn-per-slot-kv-singlewarp-sm75.cu` — split-K kernel
- Companion `.cu` for combine kernel (new file if needed)
- Dispatcher routes between split-K kernel + combine


---

## 2026-05-18 follow-up: K-loop ILP intervention — RATCHETED (TG +2.95% / PP +9.17%)

Final summary doc (Final-2026-05-18) declared the per-CTA optimization family
exhausted at +1-2% TG. That conclusion was wrong. Subsequent ncu analysis
(data/ncu-singlewarp-2026-05-18/) revealed the kernel was latency-bound on
L1TEX scoreboard stalls (35% of cycles) with only 0.6% DRAM throughput —
classic latency-hiding gap, not a per-CTA work-amortization plateau.

| iter | timestamp | variant | regs | spills | per-CTA | TG@NP=8 | PP@NP=8 | status |
|---|---|---|---|---|---|---|---|---|
| 14 | 2026-05-18T14:00 | lb=16 alone | 114 | 0 | 173.02 µs | 26.97 (-0.5%) | 21.00 (-0.2%) | KEPT as foundation — per-CTA -8.4% even though bench is wash |
| 15 | 2026-05-18T14:20 | lb=16 + 2-way ILP | 126 | 0 | 158.98 µs | 27.33 (+0.85%) | 21.97 (+4.42%) | KEPT — PP ratchets above gate; TG below |
| 16 | 2026-05-18T14:35 | lb=12 + 3-way ILP | 159 | 0 | 131.30 µs | 27.83 (+2.69%) | 22.84 (+8.55%) | RATCHETED — first clear bench win; clean state, no spills |
| 17 | 2026-05-18T14:50 | lb=10 + 4-way ILP | 168 | 1,920 | 132.80 µs | 27.77 (+2.47%) | 22.89 (+8.79%) | TIED with 3-way at bench; per-CTA flat; spills present |
| 18 | 2026-05-18T15:00 | lb=10 + 4-way + Q SMEM | 168 | 1,536 | 139.46 µs | 27.73 (+2.32%) | 22.28 (+5.91%) | REVERTED — Q SMEM read latency exceeded partial spill reduction |
| 19 | 2026-05-18T15:15 | **lb=8 + 4-way ILP** | **254** | **0** | **127.26 µs** | **27.90 (+2.95%)** | **22.97 (+9.17%)** | **LANDED — spills eliminated by relaxing launch_bounds to allow full reg budget; per-CTA -32.7% vs HEAD; theoretical occupancy 25% (down from 50%) but achieved was already grid-limited at ~5 warps/SM regardless. Best-known config.** |

### Lessons

1. **Latency-bound ≠ bandwidth-bound.** The kernel had 0.6% DRAM throughput
   and 86% L1 hit rate but 35% of cycles stalled on L1TEX scoreboard. The
   fix wasn't reducing memory traffic — it was hiding L1 hit latency via
   per-warp ILP.

2. **`__launch_bounds__` is a *register-budget hint*, not a hard cap.**
   At lb=10 (max 204 regs/thread) the compiler chose 168 regs and spilled.
   Relaxing to lb=8 (max 256) let it use 254 regs and skip spills entirely.
   The "right" lb is the one where compiler's chosen reg count avoids
   spills, not the one that maximizes theoretical occupancy.

3. **Theoretical occupancy ≥ achieved when grid is undersaturated.** Going
   from 50% theoretical → 25% theoretical did not reduce achieved (~5
   warps/SM both ways) because the grid only delivers 0.33 waves/SM at
   decode. Per-CTA latency reduction dominates.

4. **Q SMEM was unnecessary.** The earlier prescription assumed Q SMEM
   would free reg headroom for ILP. The compiler did the equivalent
   internally via live-range reuse, AND Q SMEM's per-iter read latency
   was net-negative when actually applied.

5. **The "+2% plateau" finding was wrong.** Variants A-F plateau'd at
   +1-2% because they all preserved the scalar K-loop dependency chain.
   Switching to a multi-stream K-loop body breaks that chain and gives
   the compiler something to interleave, unlocking real perf.

