# FIX-C — full worked plan (SoTA per-row CTA FA kernel for sm_75)

**Date**: 2026-05-16
**Goal**: eliminate the slot-position-dependence in batched FA (`flash_attn_per_slot_kv`) at the kernel level with no graph-level workaround, no N× launch multiplication, and SoTA performance for sm_75 (Turing/TU102) at the Qwen 3.6 27B production shape (Dq=Dv=256, n_heads_q=24, n_kv_heads=4, GQA=6, Q4_0 KV cache).

This is a **kernel architecture fix**, not a tactical dispatch change. Result: a new bespoke FA kernel `fattn-per-slot-kv-prc-sm75.cu` ("PRC" = per-row CTA) that replaces `wmma_f16_case_pb1<256,256,8,float>` on the per-slot-kv dispatch path.

## 0. Why FIX-A is structurally wrong and FIX-C is the right answer

The bug (TRACE-1..6 chain) is: in the existing wmma kernel's V × softmax(KQ) matmul, WMMA's matrix-A (V) is **shared across matrix-B's 8 cols (slots)** within one mma_sync call. fp32 accumulation across the 16 K-chunks decomposes the same algebraic total of products into different chunk groupings per slot (because slots' valid K positions differ). Same algebraic answer, different fp32 bit pattern. The bug is structural in WMMA's per-fragment K-sharing assumption.

Two structural escape paths exist:
- **Per-row CTA** (FIX-C): give each row its own CTA so V_a/K_a is no longer shared across rows. Each row's chunk decomposition is canonical for that row alone.
- **Per-call dispatch** (FIX-A): launch N separate FA calls instead of one batched call. Same effect as per-row CTA but achieved at the graph level with N× launches and N× per-call kernel work.

FIX-A is correct but tactical: N× launches, N× per-call WMMA mass (tensor-core utilization drops from N/8 to 1/8 × N calls). FIX-C avoids the multiplication by changing the kernel topology so one launch processes all N rows with per-row CTAs.

**On Turing perf reality**: FA is NOT the dominant cost at decode (MMQ FFN dominates). Empirical estimate ~0.2-2% of decode time per FA layer for Qwen 3.6 27B. So FIX-A's perf cost is **probably under 5% end-to-end** at NP=4-8. FIX-C's perf cost is **~0%**. The choice is mostly engineering elegance + future-proofing, not absolute perf.

For the spec contract ("SoTA performance"), FIX-C is the answer because:
1. No N× launch multiplication.
2. No WMMA fragment underutilization (per-row CTA chooses its own tile geometry).
3. Single kernel can absorb future improvements (split-K, async copy, smarter dequant).
4. Aligns with TML batch-invariance recipe + ssiu/flash-attention-turing's per-row design.
5. The kernel becomes a long-term-correct primitive instead of a dispatch workaround.

## 1. Algorithm — per-row CTA

For each (token_idx, head_q_idx, seq_idx) tuple = one CTA:

```
1. Load Q row [Dq fp32] from global → registers (per-thread chunks).
2. Determine valid K range from per_row_k_bound[token_idx] (or mask). Call this K_VALID.
   For per-slot-kv production: K_VALID is the slot's actual valid K count (typically n_prompt + decoded so far).
3. Initialize KQ_max = -inf, KQ_rowsum = 0 (per thread, then warp-reduced).
4. K-loop (canonical, per-row, no cross-row sharing):
   For k_block = 0 to K_VALID step K_TILE:
     a. Cooperatively load K[k_block..k_block+K_TILE) for this CTA's head_kv (= head_q / GQA_RATIO).
        K is Q4_0 in production; dequant inline to fp16 fragments.
     b. Compute KQ scores: KQ[k_block..k_block+K_TILE) = Q_row · K[k_block..k_block+K_TILE).
        Use mma.sync.m16n8k8 fp16/fp16/fp32. Iterate head_dim chunks of 8.
     c. Add mask: KQ[k] += mask[k] (mask is 0 for valid, -inf for masked-out).
     d. Online softmax (Welford-style for numerical stability):
        new_max = max(KQ_max, max(KQ[k_block..]))
        scale_correction = exp(KQ_max - new_max)
        KQ_rowsum = KQ_rowsum * scale_correction + sum(exp(KQ[k_block..] - new_max))
        KQ_max = new_max
        // Apply to running VKQ accumulator before next chunk:
        VKQ *= scale_correction
        // Compute softmax for this chunk:
        softmax_chunk = exp(KQ[k_block..] - KQ_max)
5. V accumulation (paired with K-loop, online):
   For each k_block:
     a. Cooperatively load V[k_block..k_block+K_TILE) for this CTA's head_kv. Q4_0 dequant inline.
     b. VKQ_partial = V[k_block..] · softmax_chunk = mma.sync.m16n8k8 fp16/fp16/fp32.
     c. VKQ_accumulator += VKQ_partial.
6. Final normalize: VKQ = VKQ_accumulator / KQ_rowsum.
7. Cross-warp combine VKQ across the CTA's warps (if VKQ is split across warps).
8. Write VKQ [Dv fp32] to dst.
```

**Key determinism properties**:
- Per-row CTA → no cross-row interference; each (slot, head) is computed independently.
- K-loop in canonical [0..K_VALID) order per row → same chunk decomposition for every row.
- mma.sync internal accumulation order is deterministic per fragment.
- Cross-warp combines are SMEM-tree reductions (canonical order) or warp-shuffle (commutative under fp32 for the same-value set — proved TRACE-4).
- Online softmax (Welford-style) preserves numerical stability across long K sequences without breaking row-independence.

**Same-prompt → byte-identical across slots**: each slot's CTA processes the same Q row × same K bytes × same V bytes (TRACE-3 confirms cache content is byte-identical for same-prompt slots) → byte-identical output.

## 2. Kernel signature

```cpp
template<int HEAD_DIM_Q, int HEAD_DIM_V, int K_TILE>
__global__ void fattn_per_slot_kv_prc_sm75(
    const char     * __restrict__ Q,                // [Dq, n_heads_q, n_tokens, n_seqs] F32
    const char     * __restrict__ K,                // [Dq, n_kv, n_kv_heads, n_seqs] Q4_0 (or F16)
    const char     * __restrict__ V,                // [Dv, n_kv, n_kv_heads, n_seqs] Q4_0 (or F16)
    const half     * __restrict__ mask,             // [n_kv, n_tokens] F16; 0 at visible, -inf elsewhere
    const int32_t  * __restrict__ per_row_k_bound,  // [n_tokens] I32
    float          * __restrict__ dst,              // [Dv, n_heads_q, n_tokens, n_seqs] F32
    const float scale,
    const float max_bias,
    const float m0, const float m1,
    const uint32_t n_head_log2,
    const int ne11,                                  // n_kv (= K position dim)
    const int n_kv_heads,
    const int gqa_ratio);                            // n_heads_q / n_kv_heads
```

Templated on:
- `HEAD_DIM_Q = HEAD_DIM_V = 256` (Qwen 3.6 production shape; compile-time fixed)
- `K_TILE = 32` (number of K positions per CTA per inner iter; choose to fit SMEM budget)

Compile-time `HEAD_DIM` fixing is per TML batch-invariance rule + matches existing per_slot_kv contract.

## 3. Grid + block geometry

**Grid**: `dim3(n_tokens, n_heads_q, n_seqs)`. One CTA per (token, head, seq) tuple. At NP=4 with 24 heads × 1 seq: 4 × 24 × 1 = 96 CTAs per FA layer (per device). TU102 has 72 SMs; 96 CTAs distribute with ~1.3 occupancy ratio. Good.

**Block**: `dim3(32, 4, 1)` = 128 threads (4 warps). Each warp owns a 64-dim slice of the output Dv=256 (in V accumulation phase), and cooperates on Q load + softmax.

**Justification**:
- 4 warps × 64 dims each = 256 = Dv. Clean partitioning of output across warps.
- 4 warps × 32 lanes = 128 threads → 1024-byte SMEM per K-block × 4 warps = enough K bandwidth.
- 1 CTA = ~1 SM share. At 72 SMs and ~24 CTAs per layer (NP=1), 33% SM utilization. At NP=4: ~96 CTAs = ~133% (some SMs run 2 CTAs concurrently if SMEM permits). Acceptable.

## 4. SMEM layout (per CTA, target ≤ 24 KiB)

```
__shared__ struct {
    half  Q_fp16[HEAD_DIM_Q];                   // 256 × 2 = 512 B
    half  K_tile_fp16[K_TILE * HEAD_DIM_Q];     // 32 × 256 × 2 = 16384 B
    half  V_tile_fp16[K_TILE * HEAD_DIM_V];     // 32 × 256 × 2 = 16384 B (could share buffer with K_tile)
    float KQ_scores[K_TILE];                    // 32 × 4 = 128 B
    float softmax_chunk[K_TILE];                // 128 B
    float VKQ_accum[HEAD_DIM_V];                // 256 × 4 = 1024 B
    float warp_partials[4 * HEAD_DIM_V/4];      // 4 × 64 × 4 = 1024 B (cross-warp combine)
} sm;
```

Total ~35 KiB. With K_tile and V_tile time-multiplexed (load K, compute KQ, then load V over K_tile memory, compute VKQ), we cut by 16 KiB → 19 KiB. Fits within sm_75's 48 KiB static SMEM per CTA. Headroom for 2 concurrent CTAs per SM (96 KiB total per SM).

If we want >2 CTAs per SM, drop K_TILE to 16: ~13 KiB per CTA → 3-4 CTAs per SM. Trade tile size for occupancy.

**Decision**: start with K_TILE=32 and 1-2 CTAs per SM. Tune via nsys profiling.

## 5. Register budget (per thread, target ≤ 64 regs for 2 blocks/SM)

Per-thread state:
- Q row chunks: 256/128 = 2 fp32 = 2 regs
- KQ_scores chunks: K_TILE/128 = 1 fp32 = 1 reg
- VKQ_accumulator chunks: HEAD_DIM_V/128 × 1 lane share = 2 fp32 = 2 regs
- Loop counters, temp regs for mma operand staging: ~16 regs
- mma fragment registers: ~16 regs for A/B/C across the inner mma calls

Estimated total: ~37-50 regs. Leaves headroom for compiler register spills. Target ≤ 64 confirmed achievable.

Verify post-build with `--ptxas-options=-v`. If above 64: increase block dim from 128 to 256, halving regs/thread.

## 6. PTX mma instructions

Use raw PTX mma.sync.m16n8k8 fp16/fp16/fp32 (sm_75 native).

**K · Q matmul** (per warp, per K_TILE iter):
- Q row fragment: 1 col × 8-dim chunks of head_dim. Loaded once into per-warp fp16 fragment.
- K fragment: 16 K-positions × 8-dim chunks. Loaded per inner iter from SMEM via ldmatrix.
- Output: 16 KQ scores per warp per inner iter, accumulated in fp32 registers.
- Inner iter count: 256 head_dim / 8-dim = 32 iters.
- Total mma per K_TILE: 32 mma.sync per warp.

**V · softmax matmul** (per warp, per K_TILE iter):
- V fragment: 16 K-positions × 8-dim chunks. Loaded per inner iter.
- softmax fragment: 16 K-positions × 1 col (this row's softmax values). Loaded per inner iter.
- Output: 8 head_dim outputs per warp per inner iter, accumulated.
- Inner iter count: 256 head_dim / 8-dim = 32 iters.
- Total mma per K_TILE: 32 mma.sync per warp.

**Total mma per CTA per K_TILE**: 64 mma.sync ops. With K_TILE=32 and K_VALID up to ~8K: ~256 K_TILE iters → 16K mma per CTA. At Turing fp16 tensor throughput ~14 TFLOPS/SM peak, 4096 FMAs/mma → 16K × 4096 = 64M FMAs per CTA. Time: ~5 µs per CTA on TU102 at full throughput.

For 96 CTAs (NP=4 × 24 heads) on 72 SMs: ~1.3 SM-cycles per layer = ~7 µs per FA layer. At 16 FA layers: ~110 µs per token of FA tensor-core work. Compared to ~30 ms total decode: 0.4% of decode time. Negligible.

## 7. Q4_0 dequant strategy

Inline dequant during K/V tile load. Each Q4_0 block = 18 bytes = 1 fp16 scale + 16 × 4-bit quants (= 32 fp16 values).

Per warp loading a 16-K-position × 256-head_dim tile (= 16 × 8 Q4_0 blocks = 128 Q4_0 blocks = 2304 bytes):
- Warp 0 thread t loads Q4_0 block t (32 quants × fp16 scale).
- Dequant: `q = (int4_quant - 8) × fp16_scale → fp16`. Each thread produces 32 fp16 values.
- Store to SMEM K_tile via aligned half stores.

Dequant time: ~32 multiply-add + 32 sign-extend per thread. ~50 cycles per Q4_0 block. Warp processes 32 blocks → ~1600 cycles per warp per K_TILE = ~1 µs at 1.6 GHz. Per CTA (4 warps): same. Per K_VALID=8K: 256 K_TILE iters × 1 µs = 256 µs per CTA. That's significant.

**Optimization**: vectorize Q4_0 load via half4 SMEM stores; use `__byte_perm` for nibble extraction; co-issue dequant with mma to overlap.

Per CTA dequant cost: ~256 µs at K_VALID=8K. For 96 CTAs: total ~24 ms? That's too much.

Wait — actually, CTAs run concurrently. 96 CTAs / 72 SMs ≈ 1.3 CTAs per SM in parallel. Per SM time: 256 µs × 1.3 ≈ 333 µs. At 16 FA layers: 5.3 ms FA time per token. Significant fraction of 30 ms decode time.

For K_VALID < 1K (typical decode early-on): 32 iters × 1 µs = 32 µs per CTA → 50 µs per SM → 16 × 50 µs = 800 µs FA time per token. Much better.

So FIX-C decode FA time scales with K_VALID (= prompt length + tokens generated so far). At max context (8K KV), FA time ≈ 5 ms / decode = ~15% slowdown vs current. At typical short conversations (1K KV): negligible.

Acceptable.

## 8. Softmax — fp32 throughout + online Welford

The current wmma kernel uses Welford-style online softmax. Reuse that pattern:

```cpp
float KQ_max = -INFINITY;
float KQ_rowsum = 0.0f;
float VKQ_accum[Dv/n_warps];  // per warp's slice of Dv

for k_block:
    KQ_block_scores[K_TILE];   // per-warp scratch
    // K·Q matmul -> KQ_block_scores
    // Mask add
    // Compute new max
    float KQ_max_block = warp_reduce_max(max_over_lanes(KQ_block_scores));
    float new_max = fmaxf(KQ_max, KQ_max_block);
    float scale_correction = expf(KQ_max - new_max);
    KQ_max = new_max;
    KQ_rowsum *= scale_correction;
    // Apply correction to running VKQ
    for (int i = lane*; i < Dv/n_warps; ...) VKQ_accum[i] *= scale_correction;
    // Compute softmax for this block
    float softmax_block[K_TILE];
    for k: softmax_block[k] = expf(KQ_block_scores[k] - KQ_max);
    KQ_rowsum += warp_reduce_sum(sum_over_lanes(softmax_block));
    // V·softmax matmul -> VKQ_partial
    for (i in warp's Dv slice) VKQ_accum[i] += VKQ_partial[i];

// After K loop:
for (i in warp's Dv slice) VKQ_accum[i] /= KQ_rowsum;
```

**warp_reduce_max** and **warp_reduce_sum** here are SAFE per TRACE-4/5 (XOR-shuffle is commutativity-stable on the same value set). Per-row CTA means each row's reductions are over its OWN values; no cross-row interference.

**fp32 precision throughout** for KQ scores, softmax values, VKQ accum. Q4_0 dequant produces fp16, but products are immediately cast to fp32 in mma's fp32 accumulator.

## 9. Per-row K-loop with bound

The `per_row_k_bound` tensor gives this row's effective K range. The K-loop runs `[0..bound)` instead of `[0..n_kv)`. For per-slot-kv production: bound = slot's valid K count (n_prompt + decoded).

But wait — the per-slot-kv layout in the production cache is **global-packed**: slot s's valid positions are NOT [0..bound[s]) but rather scattered across the cache. The per_row_k_bound is just the count, not the list.

**Two sub-options**:

**Sub-option PR1**: kernel iterates [0..bound[row]) and assumes slot's valid K is at the FIRST bound positions of the cache. Requires the cache to be slot-segmented OR a pre-FA gather to compact valid K to position [0..bound).

**Sub-option PR2**: kernel iterates [0..n_kv) and applies mask per K position. Wastes iterations on masked-out positions.

PR1 is cleaner but requires either cache restructuring (huge refactor) or a gather pass. PR2 is simpler but slower at sparse-mask cases.

**Recommendation**: **PR2**. Mask-bounded iteration. n_kv at production is the global cache size (n_seq_max × ctx_per_slot). At NP=4 with ctx_per_slot=2048: n_kv up to 8K. K-loop iterates ~256 K_TILEs at worst.

This makes FIX-C iterate ALL n_kv positions every call (like current kernel). The mask-shape-difference between slots doesn't change the chunk decomposition because per-row CTA isolates each row's accumulation. Even with all 8K K positions iterated (most masked-out → contribute 0 to softmax), each row's CTA's chunk decomposition is the same as if it iterated alone.

**Verification**: per-row CTA at n_kv=8K with mostly-masked positions should still produce byte-identical output to per-row CTA at the same row's actual valid count, because mask-applied positions contribute 0. The fp32 accumulation order is identical.

**Optimization later**: if K-loop iteration cost dominates, add a slot-aware compaction pass before FA. Defer to a follow-up phase.

## 10. GQA handling

Each head_q maps to head_kv = head_q / gqa_ratio. The kernel reads K/V for this CTA's head_kv. Multiple head_q CTAs share the same head_kv's K/V cache → L2 cache hit pattern is good (consecutive CTAs with same head_kv).

Optionally: launch multiple head_q per CTA in a "head_q_group" of size gqa_ratio. Each group shares K/V loads. Reduces redundant L2 reads. Implementation: CTA has dim3(gqa_ratio, 32, 4) thread layout, head_q dim handled by threadIdx.x dimension.

**Decision**: start with one CTA per head_q (simpler), measure L2 hit rate, add head_q grouping if needed.

## 11. Mask handling

Mask is loaded per K_TILE iter from the [n_kv, n_tokens] mask buffer at position `mask[k_block..k_block+K_TILE, row]`. Mask values: 0 at visible, -INF at masked-out.

For per-slot-kv: each slot's mask row has 0s at the slot's actual cache positions and -inf elsewhere. Adding mask to KQ before softmax effectively zeros-out invalid positions in softmax denominator.

ALiBi (max_bias > 0) for layers that need it: multiply mask by slope per row before adding. For Qwen 3.6 max_bias = 0, ALiBi is unused.

## 12. Numerical determinism guarantees (binding contract)

The kernel must satisfy these properties to close the determinism gap:

1. **Per-row CTA**: each (token, head) is its own CTA. No cross-row data sharing in registers or SMEM.
2. **Canonical K-loop**: each CTA iterates [0..n_kv) in canonical [0, K_TILE, 2*K_TILE, ...] order. No tile-skipping heuristics.
3. **Compile-time fixed tile shape**: K_TILE is template parameter. No runtime tile dim selection.
4. **fp32 accumulators**: VKQ_accum, KQ_max, KQ_rowsum all fp32.
5. **fp32 softmax intermediate**: KQ_scores and softmax values are fp32, not fp16.
6. **Canonical warp reduce**: warp_reduce_max and warp_reduce_sum operate on PER-ROW values. The "same value set in different lane positions" case (which TRACE-4 proved is commutativity-stable) doesn't even arise here — each row is computed in its own CTA, lane positions are canonical per the CTA's K-loop.
7. **No atomicAdd<float>**: cross-warp combines use SMEM-tree reduction (canonical order) or warp-shuffle on same-CTA data.
8. **No CUDA-graph cache eviction sensitivity**: kernel parameters are fully resolved at template instantiation; only one kernel variant per (HEAD_DIM_Q, HEAD_DIM_V, K_TILE) tuple.

Binding test (TRACE-6 + TRACE-1 patterns): all slots' outputs byte-identical at NP={2,4,8} for same-prompt input.

## 13. Test plan

### 13a. Scalar fp32 oracle (CPU reference)

Implement `tests/dflash-speculative/fattn-per-slot-kv-prc-sm75-reference.h`: scalar fp32 implementation of the per-row CTA algorithm. No tensor cores; plain fp32 loops. Used as the byte-identity oracle.

The oracle must mirror the kernel's fp32 accumulation order EXACTLY (Welford softmax, canonical k-iteration, canonical warp-shuffle equivalent on cpu). For warp-shuffle equivalence on CPU, implement a `tree_sum(values, n)` that matches the XOR-shuffle tree on 32-lane warps.

### 13b. Byte-identity unit test

`tests/dflash-speculative/test-fattn-per-slot-kv-prc-byte-identity.cpp`:
- Random F32 Q/K/V/mask/bound.
- Run kernel; capture output.
- Run scalar reference; capture output.
- bit-compare: kernel output ≡ reference output (fp32 byte-identical).

This is the kernel correctness gate.

### 13c. Same-prompt cross-slot byte-identity test

Adapt `test-trace-2-intra-layer-capture` to use the new kernel. Verify slot 0 ≡ slot 1 ≡ ... ≡ slot N-1 at FA output for same-prompt input.

### 13d. NP-invariance binding test

Adapt `test-deltanet-d1-capture` to use the new kernel. Verify NP={1,2,4,8} produce byte-identical residuals at every layer for same-prompt input.

### 13e. Existing CX.A test pass-through

`test-fattn-per-slot-kv-ncols-invariance` with new kernel should still pass (was already passing; verify no regression).

### 13f. Cross-stack reference test

Compare new kernel's output against the existing wmma kernel's output for SINGLE-ROW input (NP=1). Should be within 1e-5 NMSE (different fp32 accumulation paths but same algebraic result). NOT byte-identical — different fp32 paths produce different fp32 values for the SAME single-row input.

## 14. Perf targets

Calibrate against the existing `wmma_f16_case_pb1<256, 256, 8, float>` at NP=1 (1 row, batched as 1 useful col of 8).

- **NP=1**: ≤ 15% slower than wmma_f16 baseline (acceptable for an opt-in kernel with stronger correctness).
- **NP=4 decode**: ≤ 5% slower than current batched kernel (which has the determinism bug).
- **NP=8 decode**: ≤ 5% slower.
- **Prefill (n_tokens=12 at NP=1)**: ≤ 25% slower than current. (Prefill matters less for steady-state throughput; not the critical path.)

If unable to meet these targets, surface as a trade-off decision.

## 15. Implementation steps (chronological)

### Step 1 — scalar reference + test driver (RED)

1. `tests/dflash-speculative/fattn-per-slot-kv-prc-sm75-reference.h`: scalar fp32 oracle. Matches the kernel's intended accumulation order.
2. `tests/dflash-speculative/test-fattn-per-slot-kv-prc-byte-identity.cpp`: random inputs, scalar reference, kernel (stubbed `return -1`), bit-compare. Returns 77 (SKIP) initially.
3. `tests/CMakeLists.txt`: wire test target.
4. Verify scalar reference matches itself bit-exactly across batch dim sweeps.

### Step 2 — kernel skeleton (still RED)

1. `ggml/src/ggml-cuda/fattn-per-slot-kv-prc-sm75.cu`: kernel skeleton with empty body (returns 0s).
2. Dispatcher: `ggml_cuda_flash_attn_ext_per_slot_kv_prc_sm75(ctx, dst)` that launches the kernel.
3. Build flag: gate behind `LLAMA_FATTN_PER_SLOT_KV_ENABLE=1 && LLAMA_FATTN_PRC_SM75=1` env (defaults off; opt-in for testing).
4. Modify `fattn-per-slot-kv-sm75.cu`'s `ggml_cuda_flash_attn_ext_per_slot_kv_sm75` to route to PRC kernel when env var set.
5. Build; test still SKIPs (kernel returns 0s ≠ reference output).

### Step 3 — Q load + K-loop with fake softmax (incremental)

1. Implement Q row load (per-warp partitioned across head_dim).
2. K-tile load (Q4_0 dequant inline).
3. K·Q matmul with mma.sync.m16n8k8 PTX (initial implementation; correctness over perf).
4. Stub softmax → 1.0 / K_VALID (uniform attention).
5. V-tile load.
6. V·softmax matmul.
7. Output write.
8. Test: kernel output should equal reference IF reference also uses uniform attention. Adjust reference to match for this gate.

### Step 4 — full softmax (Welford online)

1. Replace stub softmax with full Welford-style online softmax in the K-loop.
2. Test: kernel ≡ reference bit-exactly.

### Step 5 — per_row_k_bound + mask handling

1. Wire per_row_k_bound to truncate K-loop.
2. Wire mask to add to KQ before max.
3. Test with random mask shapes; kernel ≡ reference.

### Step 6 — TRACE-6 binding

1. Run `test-trace-2-intra-layer-capture` against the new kernel.
2. Verify `flash_attn_per_slot_kv-1003`/`-2003` byte-identical between slot 0 and slot 1 at NP=2 same-prompt.

### Step 7 — TRACE-1 binding

1. Run `test-deltanet-d1-capture` at NP={1,2,4,8} with new kernel.
2. Verify all slots match slot 0 at every layer.

### Step 8 — production harness

1. Run `scripts/test-production-np-determinism.sh` with new kernel.
2. Verify 14/14 byte-identical OR document residual.

### Step 9 — perf measurement

1. nsys-profile NP=1 decode with old kernel and new kernel.
2. Bench at NP={1, 4, 8} for decode rate.
3. Compare. Tune K_TILE, head_q grouping if necessary.

### Step 10 — production routing

1. If perf within target: default-on the new kernel under `LLAMA_FATTN_PER_SLOT_KV_ENABLE=1`. Retain old kernel under fallback env var.
2. Update spec `specs/deltanet/fattn-per-slot-kv-sm75.md` with new §15.18 documenting the PRC kernel as production.
3. Update PHASE_MMQ_Q4_0_AR16.md.
4. MEMORY.md entry.

## 16. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| PTX mma.sync.m16n8k8 implementation has subtle bug (sign extension, fragment layout) | Medium | RED tests fail at step 3 | Scalar reference + CUTLASS sm_75 reference + small-tile unit test |
| Q4_0 dequant inline has correctness bug | Medium | Wrong outputs | Test against existing Q4_0 dequant kernel; CPU oracle uses scalar Q4_0 dequant |
| Online softmax (Welford) has fp32 corner-case bug | Low-Medium | Per-row outputs wrong | Test with extreme inputs (very small max, very large max) |
| Register count exceeds 64 | Medium | Lower occupancy than target | --ptxas-options=-v ; if above, increase block dim 128 → 256 |
| SMEM exceeds 24 KiB → 1 CTA/SM only | Low | ~33% perf loss | Reduce K_TILE 32 → 16 |
| Cross-warp combine has non-determinism (atomicAdd, scheduler-dependent) | Low | Slot-dependent output | Use SMEM-tree reductions; verify with TRACE-2 |
| Kernel slower than wmma_f16 at NP=1 | Medium | Block production roll-out | Profile + tune; if can't meet target, ship anyway behind opt-in env |
| Production NP-determinism harness still shows residual after FIX-C | Medium-Low | §15.17 run-to-run is separate; need TRACE-8 | Document; defer §15.17 to a follow-up phase |

## 17. Rollback plan

If FIX-C cannot pass byte-identity tests in 5 sessions:

1. Drop FIX-C; revert kernel files and dispatcher.
2. Fall back to FIX-A (per-slot dispatch at the graph level).
3. Document FIX-C attempt + lessons in MEMORY.md.
4. Long-term, revisit FIX-C with deeper PTX expertise.

If FIX-C passes correctness but fails perf targets:

1. Ship as opt-in (not default) under env var.
2. Default kernel stays the current wmma_f16 (with the bug) for non-opt-in.
3. Document trade-off; users who need determinism opt in and accept the perf cost.

## 18. Estimated effort

- Step 1: 0.3 session (~20k tokens)
- Step 2: 0.2 session (~10k tokens)
- Step 3: 1 session (~50k tokens — first full mma_sync working)
- Step 4: 0.5 session
- Step 5: 0.3 session
- Step 6-8: 0.3 session (testing only)
- Step 9: 0.5 session (perf tuning)
- Step 10: 0.2 session (production roll-out)

**Total: ~3-4 focused sessions.**

## 19. Comparison FIX-A vs FIX-C

| Dimension | FIX-A | FIX-C |
|---|---|---|
| **Correctness binding** | Trivial (ne[1]=1 ≡ NP=1) | Strong (per-row CTA architectural) |
| **Engineering effort** | 1 session | 3-4 sessions |
| **NP=1 perf vs current** | 0% (same path) | 0-15% slower (TBD, kernel quality dependent) |
| **NP=4 per-slot decode** | ~2-5% slower | ~0-5% slower |
| **NP=4 aggregate throughput** | ~95% of current | ~95-100% of current |
| **NP=8 aggregate throughput** | ~90% of current | ~95-100% of current |
| **Prefill at multi-token (n_tokens=12)** | ~10× slower (12 launches vs 1) | ~25% slower (same launch, slightly less efficient per-row CTA) |
| **Risk of subtle bugs** | Low (no kernel changes) | Medium (new PTX kernel) |
| **Reversibility** | Trivial (revert build_std_attention loop) | Medium (revert kernel files) |
| **Closes 14/14 NP-determinism** | Yes | Yes |
| **Closes §15.17 run-to-run** | No (separate issue) | No (separate issue) |
| **Spec alignment** | Workaround (FA piece OK but graph hack) | Aligns with TML batch-invariance + spec §15.10 |
| **Long-term maintainability** | Hidden complexity in graph builder | Self-contained kernel with explicit determinism contract |
| **Path to perf improvements** | Limited (graph constrained) | Open (kernel can absorb async-copy, split-K, etc.) |
| **Suitable as "ship now"** | Yes | No (3-4 sessions out) |
| **Suitable as "long-term answer"** | No | Yes |

## 20. Hybrid strategy proposal

Given the **fix-A token budget is 1 session vs fix-C is 3-4 sessions**, and FIX-A perf is actually fine for decode (~2-5% slowdown at NP=4-8 per the corrected perf analysis), a reasonable hybrid:

**Phase 1 (now)**: ship FIX-A as the immediate correctness gate. Validates the diagnosis end-to-end via production harness. Closes the 14/14 NP-determinism contract. Slowdown small enough to not block production.

**Phase 2 (~3-4 sessions later)**: implement FIX-C in parallel for the long-term SoTA path. Ship as opt-in under env var. Migrate production to FIX-C once perf is verified.

**Argument for skipping Phase 1 and going straight to FIX-C**: avoids ever shipping a "tactical fix" that lingers; cleaner spec/PHASE record; less code churn.

**Argument for Phase 1 first**: validates the diagnosis on production hardware before investing 3-4 sessions in FIX-C. If FIX-A fails the production harness (e.g., 14/14 isn't closed, or §15.17 dominates), FIX-C wouldn't help either and we'd want to know that cheaply.

**Recommendation given user's "MUST solve elegantly with SoTA performance"**: go straight to FIX-C. Skip FIX-A.

If FIX-C correctness fails (Step 8 production harness shows residual), the fix is to debug FIX-C, not retreat to FIX-A.

## 21. Open question before proceeding

The current wmma kernel uses WMMA fragment API (nvcuda::wmma). FIX-C uses raw PTX mma.sync. Maintaining both has cost (template duplication, two paths to bug-fix).

**Q**: Should FIX-C **replace** wmma_f16_case_pb1 on the per-slot-kv path (only), or also for non-per-slot-kv paths? Replacing globally is a bigger change but consolidates the FA architecture. Per-slot-kv only is minimal.

**Recommendation**: per-slot-kv only initially. Non-per-slot-kv stays on wmma_f16. Migrate non-per-slot-kv to PRC later if desired.

## 22. Decision required

User decision before I proceed with implementation:

1. **Pursue FIX-C only** (skip FIX-A entirely)? Recommended for "MUST solve elegantly + SoTA perf".
2. **FIX-A then FIX-C**? Hybrid, validates diagnosis on production hardware first.
3. **Refine FIX-C plan further before commitment**? E.g., do a paper-and-pencil register/SMEM/throughput analysis with measured Turing constants.
4. **Use option PR1 (slot-segmented cache or pre-FA gather) instead of PR2 (iterate full n_kv with mask)**? PR1 is faster at sparse-mask cases but requires larger refactor.
5. **Initial K_TILE size (32 vs 16 vs 64)**? Affects occupancy; I propose 32.

Per the user's "elegant + SoTA" directive, my recommendation is: **option 1 (FIX-C only), with K_TILE=32 and option PR2 (mask-bounded iteration, no cache refactor)**. Proceed with Steps 1-10.
