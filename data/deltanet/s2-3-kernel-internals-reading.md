# DeltaNet S2.3 — FA + MMQ kernel internals reading

Date: 2026-05-14
Branch: production/2026-q2-next
Companion to `data/deltanet/s2-3-corrected-unified-mechanism.json`.

## FA kernel — `wmma_f16_case<128, 128, 8, half>` structure

Reading `ik_llama.cpp/ggml/src/ggml-cuda/fattn-wmma-f16.cuh` + `fattn-common.cuh`:

### Dispatch (case template)

At lines 463-497 of `fattn-wmma-f16.cuh`:

```cpp
constexpr int nwarps = 4;
constexpr int frag_m = cols_per_block == 8 && Dk % 32 == 0 ? 32 : 16;
const int blocks_num_pb1 = ((Q->ne[1] + cols_per_block - 1) / cols_per_block)*Q->ne[2]*Q->ne[3];
const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
// Pick parallel_blocks based on whether the SMs have enough work:
if (4*blocks_num_pb1 < 2*nsm) parallel_blocks = 4;
else if (2*blocks_num_pb1 < 2*nsm) parallel_blocks = 2;
else parallel_blocks = 1;
```

For our case on TU102 (nsm=72): `blocks_num_pb1 = ⌈ne1/8⌉ * n_heads * batch`. With ne1 ∈ {2,4,8} (NP=2/4/8 decode) all give `⌈n/8⌉=1`. `blocks_num_pb1 = 1 * 12 * 1 = 12` for non-split or `1 * 6 * 1 = 6` per device with 2-GPU split. Either way, `4*12 = 48 < 144` → **`parallel_blocks = 4` selected at all NP∈{2,4,8}**.

### Parallel-block cross-block partial-sum combine

With `parallel_blocks = 4`, FOUR blocks per Q column compute partial KQ sums on different K chunks. Each block iterates K/V in stride `parallel_blocks * FATTN_KQ_STRIDE = 4 * 256 = 1024` (line 180):

```cpp
for (int k_VKQ_0 = ip*FATTN_KQ_STRIDE; k_VKQ_0 < ne11; k_VKQ_0 += parallel_blocks*FATTN_KQ_STRIDE)
```

Block ip=0 handles K tokens [0..255], [1024..1279], …; ip=1 handles [256..511], [1280..1535], … etc.

Partial outputs land in `dst[j_dst*gridDim.y*Dv + ...]` and `dst_meta[j_dst*gridDim.y*parallel_blocks + ...]`. The separate `flash_attn_combine_results` kernel (`fattn-common.cuh:670-708`) combines them.

### Combine-results kernel deterministic per-config

```cpp
for (int l = 0; l < parallel_blocks; ++l) {
    const float diff = meta[l].x - kqmax;
    const float KQ_max_scale = expf(diff);
    VKQ_numerator   += KQ_max_scale * VKQ_parts[l*gridDim.y*Dv + blockIdx.y*Dv + tid];
    VKQ_denominator += KQ_max_scale * meta[l].y;
}
dst[blockIdx.y*Dv + tid] = VKQ_numerator / VKQ_denominator;
```

Combine order is `l = 0, 1, 2, 3` — fixed. Same `parallel_blocks` across NP → same combine pattern → combine kernel is batch-shape-INVARIANT given identical partial sums.

### Where the batch-shape sensitivity actually enters

**Most-likely mechanism: `n_kv` (= `K->ne[1]`) depends on max KV-cache occupancy ACROSS slots in the batch.**

`ggml_view_3d(split_kl, n_embd_head_k, n_kv, n_head_kv, …)` (build path in `llama-build-context.cpp:2686`) — `n_kv` is taken at the WHOLE-BATCH level, not per-slot. At NP=2 with prompts {p3=33, p4=24}, `n_kv = max(33, 24) = 33` (padded to FATTN_KQ_STRIDE=256 multiple). At NP=4 with {p3, p4, p5, p6}, `n_kv = max(33, 24, len(p5), len(p6))` — may differ.

The K-loop in FA (line 180 of fattn-wmma-f16.cuh) iterates `for k_VKQ_0 = ip*256; k_VKQ_0 < ne11; k_VKQ_0 += 4*256`. If `ne11` differs between NP=2 and NP=4, the per-block K-loop count differs. The "extra" K positions are mask-zero for slot 0 (its valid range is 0..32), so their attention contribution is mathematically 0 — but the floating-point computation of `attn_weight * V_token` for masked positions produces fp32-epsilon-level roundoff that accumulates into the online softmax denominator and VKQ numerator.

This is a **`ne11`-dependent (not `cols_per_block`-dependent)** source of fp32-eps variation in slot-0 output.

### Secondary candidate: tile-load OOB reads

Q is padded to `ncols=8` columns in shared memory (line 162: `KQ[j*Dk_padded + i] = ic0+j < ne01 ? Q_f[...] : 0.0f`). For NP=2 (real cols 0,1; padding 2-7) vs NP=4 (real cols 0-3; padding 4-7), the padding-source data is explicit 0.0f. WMMA fragment-load from shared memory should produce the same per-column outputs regardless of padding configuration (per-column reduction is fragment-internal and doesn't cross-pollinate columns at the math level).

But shared-memory bank conflict patterns differ between the two padding configurations, which could affect scheduling of the warp-shuffle-based softmax steps. Effect is fp32-eps level.

## MMQ kernel — `mul_mat_q<TYPE, mmq_x=8, MMQ_NWARPS=8, need_check>` structure

Reading `ik_llama.cpp/ggml/src/ggml-cuda/mmq.cuh`:

### Stream-K work partitioning

The kernel uses the stream-K parallel matmul scheme (Osama et al. 2023, arXiv:2301.03598). Lines 3962-4005:

```cpp
const int ntx = (ne11 + mmq_x - 1) / mmq_x;   // output tiles in M direction
const int nty = (ne01 + mmq_y - 1) / mmq_y;   // output tiles in N direction
int64_t kbc      = (int64_t) blockIdx.x     *blocks_per_ne00*ntx*nty / gridDim.x;
int64_t kbc_stop = (int64_t)(blockIdx.x + 1)*blocks_per_ne00*ntx*nty / gridDim.x;
```

For NP ∈ {2,4,8} with mmq_x=8: `ntx = ⌈n/8⌉ = 1` in all cases. `nty` depends on `ne01` (output dim — same projection target across NP). `gridDim.x = nsm = 72`. So work per block = `blocks_per_ne00 * 1 * nty / 72`. Same across NP at decode shapes.

Per-block does:

```cpp
for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
    load_tiles(x + stride01*it*mmq_y, tile_x, kb0, ...);
    load_tile_y(...);  // tile_y[l] = by0[l] unconditional
    __syncthreads();
    vec_dot(tile_x, tile_y, sum, 0);
    __syncthreads();
    load_tile_y(...);  // second half
    __syncthreads();
    vec_dot(tile_x, tile_y, sum, WARP_SIZE);
    __syncthreads();
}
```

The fixup kernel `mul_mat_q_stream_k_fixup` (line 4009) combines partial sums from blocks that crossed tile boundaries. Iteration order is fixed:

```cpp
for (int bidx = bidx_start; bidx < bidx_stop; ++bidx) {
    ...
    sum[...] += tmp_last_tile[bidx*(mmq_x*mmq_y) + j*mmq_y + i];
}
```

### Where the batch-shape sensitivity enters

**Most-likely mechanism (mirroring FA): `ne11` differs across NP → `tile_y_max_j` differs → OOB tile_y loads have different garbage content but contribute to fragment-internal computation that DOES NOT cross-pollinate valid columns at the math level but DOES affect bank-conflict patterns and accumulator scheduling.**

For valid output column j=0 with same per-row inputs, the formal sum is identical. The empirical fp32-eps variation in MMQ outputs (seen as the 0.00077 differences at p4 layer 3 kqv_wo) is most likely **propagation of the upstream FA's batch-shape sensitivity through MMQ's faithful matmul**, rather than fresh MMQ-introduced noise.

### Amplification factor in MMQ

A matrix multiply amplifies input variation by the operator-norm of the matrix. For a 5120×5120 projection GEMM, this factor is typically 10²-10³ at small operator norms (weight-spectrum-dependent). The chain Q-proj → FA → O-proj → FFN-gate → FFN-up → FFN-down can amplify a 1e-7 FA-internal fp-eps noise to 1e-3 by the end of the layer — matching the empirical p0 layer 19 amplification chain (4.8e-7 at flash_attn → 0.006 at l_out).

## Summary — Open_FA_InternalMechanism + Open_MMQ_InternalMechanism updated framing

The CURRENT-STATE invariants the spec captured (KernelFPArithmeticBatchShapeSensitivity, etc.) hold. The kernel-internal mechanism is now identified:

1. **FA kernel batch-shape entry point: `n_kv = K->ne[1]` (max KV across slots) varies with NP.** The K-loop iteration count varies; masked extra K positions produce fp32-eps roundoff in online-softmax accumulators that propagates to slot-0 output.
2. **MMQ batch-shape entry point: `ne11 = src1->ne[1]` (M dimension) varies with NP.** OOB tile_y reads have different garbage content. Per-block compute for valid output positions is FORMALLY batch-invariant; observed variation in valid-position outputs is propagation of FA-introduced fp-eps noise through subsequent GEMMs.
3. **Combine + fixup kernels are batch-shape-invariant given identical partial sums** — iteration order is fixed across NP.
4. **Amplification through chained GEMMs is the cascade visibility regime** (regime b in the unified mechanism).

Both kernels have the same root structure: their formal output for slot-K with identical per-slot inputs SHOULD be batch-invariant, but practical fp32 arithmetic at extra-K-position boundaries produces sub-eps noise that the unified mechanism then surfaces through one of three visibility regimes (a / b / c per `s2-3-corrected-unified-mechanism.json`).

### What would empirically confirm the n_kv hypothesis

To CLOSE `Open_FA_InternalMechanism` to a binding empirical result:

(a) **Hold `n_kv` constant across NP**: pad all slots' KV occupancy to the same maximum value at prefill (e.g., pad p4 from 24 to 33 tokens with mask). At NP=2 vs NP=4 with all slots having `n_kv=33`, verify slot-0 FA output becomes byte-identical (modulo tile-padding-induced effects).

(b) **Vary `n_kv` independent of np**: add deliberate padding to slot 0 at the SAME np value (e.g., NP=2 with slot 0 padded to 33 vs 128 vs 256 KV tokens), verify slot-0 output drifts in the same fp-eps pattern as observed across np changes.

If (a) gives byte-identity and (b) shows the same drift signature without changing np: `n_kv` is the dominant source.

If (a) does NOT give byte-identity: there's an additional batch-shape-sensitive code path in the FA kernel internals (likely the warp-shuffle accumulators or the shared-memory bank-conflict scheduler interaction).

Static reading alone gives high-probability identification but not a definitive bind. The empirical (a)+(b) test would lock the mechanism.
