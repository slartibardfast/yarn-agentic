# `fattn_per_slot_kv_sm75` — SoTA sm_75 batch-invariant flash-attention

Date: 2026-05-14
Branch: production/2026-q2-next
Status: DRAFT — Stage 2 design (S2.4)
Companion to `specs/deltanet/batch-invariance.allium` (locks the invariants this kernel must restore)
Template follows `specs/dflash/kernel-design.md §6.3`

---

## 1. Scope

This spec defines the SoTA replacement for `ggml_cuda_flash_attn_ext_wmma_f16` on sm_75 / TU102. The replacement:

- **Closes** the invariants `FA_NkvIsDominantBatchShapeEntryPoint` and `MMQ_FaithfulPropagation` (currently violated by `wmma_f16` at all NP > 1 decode shapes).
- **Targets all shape regimes** (decode 1-8, batched prefill, full prefill) — must not regress wmma_f16 at any measured shape per the SoTA-sm_75 mandate.
- **Preserves tensor-core throughput** at all regimes via multi-head-per-CTA tile packing (Approach C — locked).
- **Replaces `wmma_f16` at sm_75 in production** after env-gated A/B profiling closes D8 perf binding GREEN at all measured shapes.

Production target: Qwen 3.6 27B (general.architecture=qwen35, hybrid linear_attn + full_attention). Hardware: dual Quadro RTX 6000 (sm_75 / TU102, 24 GiB each, NVLINK). The kernel runs on every `full_attention` layer at decode and prefill.

**Out of scope:** Vulkan backend, sm_80+ (other ggml-cuda kernels handle those). The DeltaNet recurrence kernel (`delta-net.cu`) is not affected — it was already proven byte-identical across NP.

---

## 2. Architecture features cited (sm_75 / TU102)

| Resource | Per-GPU | Aggregate |
|---|---:|---:|
| HBM bandwidth | 624 GB/s | 1.25 TB/s w/ NVLINK |
| FP16 tensor-core peak | 130.5 TFLOPs | 261 TFLOPs |
| NVLINK | — | ~100 GB/s |
| SMs | 72 | 144 |
| Register file | 65536 × 32-bit per SM | — |
| SMEM | 64 KiB per SM (effective) | — |
| L2 cache | 6 MiB per GPU | — |

Tensor-core instruction: **`mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16`** (and the `.f32.f16.f16.f32` variant for fp32 accumulators). This is Turing's primary fp16 MMA. Output is 16×8 (= 2 warps × 8 outputs/lane). Q tile must be 16 rows; K tile is 8 cols × 8 K-dim per instruction.

Smem-load helper: **`ldmatrix.sync.aligned.x4.m8n8.shared.b16`** with swizzled SMEM layout (8×16 transposed loads per instruction) for fragment-aligned reads.

---

## 3. Kernel signature

**File**: `ggml/src/ggml-cuda/fattn-per-slot-kv-sm75.cu` + `fattn-per-slot-kv-sm75.cuh`

```cpp
template<
    int HEAD_DIM_Q,         // 256 for Qwen 3.6 27B (and 64/96/128 for compat)
    int HEAD_DIM_V,         // 128 for Qwen 3.6 27B (V dim differs from Q dim)
    int KV_BLOCK_SIZE,      // K-loop iteration tile; {32, 64} (see §8)
    bool USE_SOFTCAP        // whether attn_soft_cap is non-zero
>
__launch_bounds__(128, 2)  // ≤ 64 regs/thread, 2 blocks/SM, target 50% occupancy
__global__ void fattn_per_slot_kv_sm75(
    // Inputs
    const half *  __restrict__ Q,                 // [HEAD_DIM_Q, n_tokens, n_heads_q, n_seqs]
    const half *  __restrict__ K_cache,           // [HEAD_DIM_Q, n_kv_max, n_kv_heads, n_seqs] per-slot
    const half *  __restrict__ V_cache,           // [HEAD_DIM_V, n_kv_max, n_kv_heads, n_seqs] per-slot
    const half *  __restrict__ mask,              // [n_kv_max, n_tokens] (per-Q-row attention mask)
    const int  *  __restrict__ slot_seq_lens,     // [n_seqs] per-slot valid KV-cache occupancy
    // Outputs (split-K partial sums; combine kernel finishes)
    float *       __restrict__ dst_partial,       // [HEAD_DIM_V, parallel_blocks, n_tokens, n_heads_q, n_seqs]
    float2 *      __restrict__ dst_meta,          // [parallel_blocks, n_tokens, n_heads_q, n_seqs]  (max, rowsum)
    // Constants
    const int n_tokens,
    const int n_heads_q,
    const int n_kv_heads,
    const int n_seqs,
    const int n_kv_max,                           // K_cache stride dim (≥ max(slot_seq_lens))
    const int parallel_blocks,                    // K-range splits per (slot, query-tile)
    const float scale,
    const float softcap,                          // 0.0f if !USE_SOFTCAP
    const float max_alibi_bias,
    const uint32_t n_head_log2,
    const float m0,
    const float m1
);
```

**Companion combine kernel** (re-uses ggml's existing `flash_attn_combine_results` from `fattn-common.cuh` — already batch-invariant per the audit in `s2-3-kernel-internals-reading.md`):

```cpp
template<int HEAD_DIM_V, int PARALLEL_BLOCKS>
__global__ void flash_attn_combine_results(
    const float * VKQ_parts, const float2 * VKQ_meta, float * dst
);
```

---

## 4. Grid + block geometry

### Grid

```
dim3 grid(
    n_seqs * n_query_tiles_per_seq,   // x: (slot, query-tile-within-slot) flattened
    n_heads_q,                         // y: per-head
    parallel_blocks                    // z: K-range split for this (slot, tile, head)
);
```

where `n_query_tiles_per_seq = ⌈(n_tokens / 16)⌉`. At decode (n_tokens=1 per slot): 1 query tile per slot. At prefill (n_tokens=1024): 64 query tiles per slot.

Each CTA processes ONE 16-row Q tile × ONE head × ONE K-range chunk. The 16-row Q tile contains 16 consecutive query positions from the SAME slot (one slot only — never crosses slot boundary), enabling per-slot `n_kv` bound enforcement.

### Block

`blockDim.x = 32` (warp size), `blockDim.y = 4` (warps per CTA). `nwarps = 4`. Threads per CTA = 128.

### `parallel_blocks` decision (per-slot)

```cpp
parallel_blocks(slot_seq_lens[s]) =
    let n_kv_blocks = ⌈slot_seq_lens[s] / KV_BLOCK_SIZE⌉
    in min(n_kv_blocks, max_pb_for_perf_target(s))
```

`max_pb_for_perf_target(s)` chooses the smallest `parallel_blocks` such that:
- Total CTAs across grid fills ≥ 70% of SMs (`8 * n_seqs * n_query_tiles_per_seq * n_heads_q * parallel_blocks >= 0.7 * nsm * 2` — counting target 2 blocks/SM occupancy)
- `parallel_blocks ≤ KV_BLOCK_SIZE` (don't fragment past a single K-block of work per CTA)

For decode at NP=8, n_heads_q=12, n_query_tiles_per_seq=1: `8 * 1 * 12 * pb >= 100.8` → `pb ≥ 2`. Pick `pb = max(2, n_kv_blocks)`. For long-context decode (n_kv=4096, KV_BLOCK_SIZE=64): n_kv_blocks=64; pb = 4 to 8 typical.

For prefill at NP=1, n_heads_q=12, n_query_tiles_per_seq=64: `1 * 64 * 12 * pb >= 100.8` → `pb ≥ 1`. No split-K needed; `pb = 1`.

**`parallel_blocks` is a function of `slot_seq_lens[s]` only, NOT of NP.** This is the key batch-invariance property: at the SAME slot K with the SAME n_kv, the SAME grid geometry is selected regardless of how many other slots are in the batch.

---

## 5. Algorithm (per-CTA work)

Each CTA computes attention for a 16-row Q tile against a `KV_BLOCK_SIZE`-bounded K range.

```
slot       = blockIdx.x / n_query_tiles_per_seq
qtile_idx  = blockIdx.x % n_query_tiles_per_seq
head       = blockIdx.y
ip         = blockIdx.z                 # K-range chunk index ∈ [0, parallel_blocks)
n_kv_slot  = slot_seq_lens[slot]

# Per-slot K range for this CTA:
k_chunk_size_for_slot = ⌈n_kv_slot / parallel_blocks_for_slot(slot)⌉
k_start = ip * k_chunk_size_for_slot
k_end   = min((ip + 1) * k_chunk_size_for_slot, n_kv_slot)    # bound by n_kv_slot

# If this CTA's K range is empty (slot has < ip*k_chunk_size valid KV):
if k_start >= n_kv_slot:
    write (max = -inf, rowsum = 0) to dst_meta; return
```

### Phase 1: Load 16 Q rows into registers

Each CTA needs 16 Q rows. Q row r maps to:
- `q_pos_in_slot = (qtile_idx * 16 + r) % n_tokens`  (clamp to n_tokens-1 → mask in softmax)
- `slot, head` fixed by blockIdx

`Q[head_dim_q, q_pos_in_slot, head, slot]` is `HEAD_DIM_Q` halfs = 512 B per row × 16 rows = 8 KiB per CTA.

**Strategy**: each warp loads 4 of the 16 rows (32 threads × 16 halfs = 1 row per pass × 4 passes per warp). `ldmatrix.sync.x4` could be used; in fp16 the simpler vectorized `half4` loads via `__ldg` are sufficient bandwidth-wise.

After Phase 1: Q registers = 16 × HEAD_DIM_Q halfs distributed across the 128 threads.

### Phase 2: K-loop over `KV_BLOCK_SIZE` chunks

```
for kb in range(k_start, k_end, KV_BLOCK_SIZE):
    # 2a. Load K block (16 K positions × HEAD_DIM_Q) into SMEM, swizzled for ldmatrix
    ...

    # 2b. Compute QK^T using mma.sync.m16n8k8
    #     m=16 (Q rows), n=8 (K cols), k=8 fragment k-dim
    #     For HEAD_DIM_Q=256, k-loop unroll 256/8 = 32 mma.sync calls per K-block
    #     For KV_BLOCK_SIZE=64, processes 64/8 = 8 (n_warps=4 → 2 col blocks per warp)
    ...

    # 2c. Apply softcap (if USE_SOFTCAP) and mask
    #     Softcap: KQ = softcap * tanh(KQ / softcap)
    #     Mask: KQ += mask[kb..kb+block_size, q_row]
    ...

    # 2d. Online softmax update (per-Q-row)
    new_max = max(KQ_max_running, max(KQ_block))
    diff = KQ_max_running - new_max
    KQ_rowsum_running *= exp(diff)
    VKQ_acc *= exp(diff)
    KQ_max_running = new_max

    KQ_softmaxed = exp(KQ_block - new_max)
    KQ_rowsum_running += sum(KQ_softmaxed)

    # 2e. Load V block (KV_BLOCK_SIZE × HEAD_DIM_V) into SMEM, swizzled
    ...

    # 2f. VKQ_acc += KQ_softmaxed @ V_block via mma.sync.m16n8k8
    #     m=16 (Q rows), n=8 (V dim cols), k=8 fragment k-dim
    ...
```

### Phase 3: Normalize + write partial sums

```
# Final VKQ_acc (un-normalized) and (KQ_max_running, KQ_rowsum_running) are this
# CTA's contribution. Write to dst_partial and dst_meta.

dst_partial[head_dim_v_index, ip, q_row, head, slot] = VKQ_acc / KQ_rowsum_running  # OR un-normalized; combine handles
dst_meta[ip, q_row, head, slot] = (KQ_max_running, KQ_rowsum_running)
```

Combine kernel `flash_attn_combine_results` (ggml's existing) handles the across-parallel_blocks reduction in fixed iteration order — already batch-invariant per the analysis in `s2-3-kernel-internals-reading.md`.

---

## 6. Determinism contract (binding to closed Allium invariants)

Binds:
- `FA_NkvIsDominantBatchShapeEntryPoint` — by per-slot `k_end` bound.
- `RepDeterminismAtSameNP` — by no atomics, no cross-block reductions inside the kernel.
- `DispatcherTemplateUniformity_FA` — template selection depends on (HEAD_DIM_Q, HEAD_DIM_V, KV_BLOCK_SIZE, USE_SOFTCAP), all derived from model hyperparameters and op_params, not from runtime batch shape.

Explicit non-determinism sources eliminated:
- No `atomicAdd<float>` anywhere.
- No cross-block reduction (combine kernel iterates in fixed `ip = 0..parallel_blocks-1` order).
- No warp-shuffle that depends on a runtime-shape-dependent thread mapping.
- mma.sync.m16n8k8 is deterministic per PTX spec.
- K-loop iteration count = `(k_end - k_start) / KV_BLOCK_SIZE`. `k_end` is `min(per-slot-bound, ip-block-bound)` — depends only on `(slot_seq_lens[slot], ip, parallel_blocks_for_slot)`, not on NP. So K-loop count is invariant across NP for the same slot.
- Same `parallel_blocks_for_slot(slot)` decision across NP — per-slot only.

---

## 7. Numerical contract (byte-identity to scalar fp32 oracle)

Per `feedback_test_first_discipline`: scalar fp32 oracle and RED unit tests FIRST.

The kernel output must be byte-identical (or ≤ 1 fp16 ULP) to a scalar fp32 reference implementation that:
- Iterates K positions in the SAME order as the kernel (per `ip`, then per `KV_BLOCK_SIZE` chunk)
- Uses fp32 accumulators throughout (matching the kernel's fp32 KQ_acc and online-softmax state)
- Casts intermediates to half at the SAME points the kernel does (Q/K/V to half pre-mma; final VKQ to half post-normalize)

The oracle is hand-written CPU code, ≤ 300 LOC, in `tests/dflash-speculative/fattn-per-slot-kv-sm75-reference.h`. Mirror DFlash T3's `wmma-mimicking-oracle.h` pattern.

**Closure binding**: byte-identity unit test PASS at `np ∈ {1, 2, 4, 8}` × `(slot_seq_lens) ∈ {short, medium, long}` × `n_heads ∈ {1, 4, 12}` × random seeds, all configs.

---

## 8. Performance contract (% of TU102 peaks)

### Decode regime (n_tokens=1 per slot, NP ∈ {1..8})

| Shape | n_kv | Work | Memory traffic | HBM-bound target | Wall-clock target |
|---|---:|---:|---:|---:|---:|
| NP=1 | 4096 | ~12 MFLOPS | 4 MiB K + 2 MiB V = 6 MiB | 6 MiB / 624 GB/s = 9.6 µs | < 15 µs (62% of HBM) |
| NP=4 | 4096 | ~48 MFLOPS | 6 MiB K+V (shared via L2) | 9.6 µs | < 15 µs |
| NP=8 | 4096 | ~96 MFLOPS | 6 MiB | 9.6 µs | < 20 µs |

Decode is HBM-bandwidth-bound (compute << bandwidth). Target: ≥ 60% of per-GPU HBM bandwidth = 374 GB/s effective. Tensor-core utilization at decode is incidental (small compute load); target ≥ 5% of 130.5 TFLOPs = 6.5 TFLOPs, which is trivial.

### Prefill regime (n_tokens=1024, NP=1)

| Shape | n_kv | Work | Tensor-core target | Wall-clock target |
|---|---:|---:|---:|---:|
| 1024 × 4096 | 4096 | ~50 GFLOPS | ≥ 50% of 130.5 TFLOPs = 65.25 TFLOPs | 50/65.25 ms = 0.77 ms |

Prefill is tensor-core-bound. Target: **≥ 50% of fp16 tensor-core peak per-GPU** at the prefill shape. Reference: ssiu/flash-attention-turing achieves ~63% peak on T4 at similar shapes; we target 50% as the floor, 60% as the goal.

### Cross-shape baseline (must not regress)

Per the SoTA mandate, the new kernel must MATCH OR BEAT wmma_f16 at every measured shape. Measurement strategy in §10 (Integration).

---

## 9. Register / SMEM budget

### Per-thread registers (target ≤ 64 for 2 blocks/SM occupancy)

| Category | Count | Notes |
|---|---:|---|
| Q row registers | 16 (8 halfs × 2 = 4 fp32 equiv per thread, 16 rows / 4 = 4 rows per thread group) | Q tile 16 × HEAD_DIM_Q halfs distributed across 128 threads. At HEAD_DIM_Q=256: 16×256/128 = 32 halfs/thread = 16 fp32-equiv. **OVER BUDGET** — need to load Q from SMEM lazily per mma block. Spec to verify. |
| KQ accumulator | 16 × 8 / 4 warps = 32 fp32 per warp = 1 per thread per fragment, × n_fragments | TBD with --ptxas-options=-v |
| VKQ accumulator | HEAD_DIM_V × 16 / 128 = 16 fp32 per thread (HEAD_DIM_V=128) | Fits |
| Online softmax state | 16 fp32 (rowsum) + 16 fp32 (max) = 32 fp32 per CTA = 1 fp32 per 4 threads | Fits |
| Loop counters, indices, etc. | ~10 | — |

**OPEN — Q tile budget**: at HEAD_DIM_Q=256 with 16 Q rows, total Q tile is 16×256×2 = 8 KiB. If kept entirely in registers (4096 halfs / 128 threads = 32 halfs/thread = 16 fp32 regs/thread), Q alone is 16 regs/thread. Add VKQ + KQ + online softmax + control → likely 50-70 regs/thread total. Spec assumption: Q is partially in SMEM, fetched per mma block; ~16-24 regs for Q at any time. **Will be validated with --ptxas-options=-v during S2.5 implementation.**

### Per-CTA SMEM (target ≤ 32 KiB for 2 blocks/SM occupancy with 64 KiB cap)

| Buffer | Size | Notes |
|---|---:|---|
| Q tile (alternative to all-registers) | 8 KiB | 16 rows × HEAD_DIM_Q=256 × 2 bytes |
| K block (KV_BLOCK_SIZE × HEAD_DIM_Q halfs, swizzled) | 32 KiB at KV_BLOCK_SIZE=64; 16 KiB at KV_BLOCK_SIZE=32 |
| V block (KV_BLOCK_SIZE × HEAD_DIM_V halfs, swizzled) | 16 KiB at KV_BLOCK_SIZE=64; 8 KiB at KV_BLOCK_SIZE=32 |
| KQ scores (16 × KV_BLOCK_SIZE fp32) | 4 KiB at KV_BLOCK_SIZE=64 |

**Total at `KV_BLOCK_SIZE=32`, single-buffer K+V**: 8 + 16 + 8 + 2 = 34 KiB. Exceeds 32 KiB. Reduce Q SMEM (use registers more aggressively) → target 24 KiB. **Or use `KV_BLOCK_SIZE=64` single-buffer**: 8 + 32 + 16 + 4 = 60 KiB — fits in 64 KiB cap but `blocks/SM = 1` only. Half the occupancy.

**Decision point (proposed): `KV_BLOCK_SIZE=32` with double-buffered K+V** (=32 KiB K + 16 KiB V + Q-register-only = 48 KiB). Verifies with --ptxas-options=-v. At fallback: `KV_BLOCK_SIZE=32` single-buffer + Q in registers (24 KiB total), giving 2 blocks/SM but no K/V load/compute overlap.

---

## 10. Test plan

Per `feedback_test_first_discipline`: scalar fp32 oracle → RED unit test → kernel implementation → byte-identity bind.

### S2.5.a — Scalar fp32 oracle (`tests/dflash-speculative/fattn-per-slot-kv-sm75-reference.h`)

~300 LOC of header-only CPU code. Mirrors the kernel's compute structure exactly:
- Per-slot K-loop bound (uses slot_seq_lens)
- KV_BLOCK_SIZE-chunked iteration
- Per-CTA online softmax order
- Final normalize + cast to half

Self-test: hand-verified on (zero Q, zero K, zero V) → output zero; (unit Q, unit K, zero V) → output zero (V dominates); etc.

### S2.5.b — RED unit test (`tests/dflash-speculative/test-fattn-per-slot-kv-sm75.cpp`)

Random fp16 inputs × NP ∈ {1, 2, 4, 8} × slot_seq_lens ∈ {32, 128, 1024, 4096} × n_heads ∈ {1, 6, 12} × 4 seeds = 256 configs.

Bind: byte-identity (or ≤ 1 fp16 ULP) of kernel output vs scalar oracle output at all configs.

### S2.5.c — Production-shape integration test

Re-use `test-deltanet-d1-capture.cpp` and `test-np-validity-vanilla.cpp` infrastructure (now exercising the new kernel as the default sm_75 path — no env to flip):
- All 8 prompts × 64 generated tokens × NP ∈ {1, 2, 4, 8} × 3 reps each
- Slot-K residual byte-identical across NP for every (prompt, layer, slot)
- T9.1 5 validity asserts GREEN
- `healthcheck.sh` GREEN
- Production server boots and serves a smoke prompt successfully under the new kernel

Closure binding: all 96 configurations produce byte-identical token sequences vs NP=2 reference.

### S2.5.d — Performance binding (straight replacement; baseline pre-merge only)

Per §11 — there is no env gate, so the baseline must be captured BEFORE the merge:

1. **Pre-merge baseline (captured before any code lands)**: build current HEAD, measure wmma_f16 at every production shape regime. Commit nsys + ncu profiles to `data/deltanet/perf/baseline/`.

2. **Replacement measurement (after kernel commits)**: same shape regimes, new kernel as default. Commit nsys + ncu profiles to `data/deltanet/perf/replacement/`.

3. **Per-shape comparison and bind**: for every shape regime, replacement must:
   - Meet or beat baseline wall-clock (no regression at ANY measured shape — recall there is no fallback)
   - Hit its §8 perf contract (positive bind, % of TU102 peak)
   - Match byte-identity vs scalar oracle (closure of §7)

4. **Shape regimes covered (mandatory; no shape can be skipped)**:
   - Decode: NP ∈ {1, 2, 4, 8}, n_kv ∈ {short_prompt_32, medium_512, long_4096, max_16384}
   - Small-batch prefill: NP=1, n_tokens ∈ {16, 32, 64, 128}
   - Mid prefill: NP=1, n_tokens ∈ {256, 512}
   - Full prefill: NP=1, n_tokens ∈ {1024, 2048, 4096}
   - HEAD_DIM_Q ∈ {128, 256} (Qwen variants on production)
   - With and without softcap (Qwen 3.6 uses softcap for some layers)

   Total: ~80 (shape, baseline-vs-replacement) measurement pairs. nsys + ncu data for all of them committed.

5. **Surface failures, do not fall back**: if any shape pair shows regression or fails byte-identity, the spec is updated with the failure mode named, the implementation iterated, and the measurement re-run. Per §11 thoroughness bar 4: failure surfaces, not silent fallback.

This is the ship gate. No env-flip; the replacement kernel is the default path the moment the merge lands.

---

## 11. Integration — straight replacement at sm_75 (no env gate)

Per user direction (2026-05-14): this is a REPLACEMENT, not an A/B rollout. There is NO env arg for sm_75; the new kernel takes the wmma_f16 path directly. **Thoroughness bar:** every shape regime production uses must be covered by the spec, validated by §10 tests, and bound by §8 perf contracts BEFORE the kernel is committed. The wmma_f16 path is removed for sm_75 in the same commit that adds the replacement (or in a follow-up only for non-Qwen models).

### Dispatcher change (production)

`fattn.cu` lines 60-148 — current sm_75 path that ends at `ggml_cuda_flash_attn_ext_wmma_f16(ctx, dst)` becomes:

```cpp
// sm_75 fast path: batch-invariant per-slot-KV-bounded FA replacement.
if (!new_mma_available(cc)) {
    fattn_per_slot_kv_sm75_dispatch(ctx, dst);  // routes on (HEAD_DIM_Q, HEAD_DIM_V, USE_SOFTCAP)
    return;
}
```

The `_dispatch` function picks template parameters based on the tensor's `Q->ne[0]`, `V->ne[0]`, and `op_params[2]` (softcap). All template specializations are compiled in for every supported HEAD_DIM combination (see OQ-4 resolution below).

### What changes in `fattn.cu`

The lines that previously routed to `wmma_f16` (60-148 inclusive) are replaced with the single `fattn_per_slot_kv_sm75_dispatch` call. The `fattn-wmma-f16.cu` + `.cuh` files remain in the tree but are no longer reachable on sm_75 — they stay for non-Turing fp16-mma paths and for non-Qwen models that may still need the wmma_f16 layout. Removal of wmma_f16 entirely is a follow-up that requires auditing all callers across the supported model set.

### Compatibility surface

- Drop-in `dst` tensor shape — same as wmma_f16 produces.
- Combine kernel reuse — `flash_attn_combine_results` from `fattn-common.cuh` (already batch-invariant per the audit; verified compatible per OQ-3 closure).
- Mask layout — same `[n_kv, n_tokens]` mask read; per-slot bound via `slot_seq_lens` is applied IN ADDITION TO mask, not in place of it.

### What "thorough" means here (added per user direction)

Spec is incomplete and cannot ship until:

1. **All HEAD_DIM_Q values that production uses are template-instantiated and validated.** Qwen 3.5/3.6 use HEAD_DIM_Q=256. Other production-relevant arches: Qwen 3 (128), Qwen 3 small (256?), GLM-4.7-Flash (128?), Nemotron (?). Audit before commit; instantiate every shape production touches OR explicitly route non-supported shapes to a documented fallback (NOT wmma_f16 on sm_75 since we're removing it from that path).
2. **KV_BLOCK_SIZE picked from measured profile, not provisional.** §8 budget commits to KV_BLOCK_SIZE=32 with double-buffer; §10 test S2.5.d must include a sweep at {16, 32, 64} with nsys + ncu data showing the chosen value is the best at all production shapes. If 32 is best at decode but 64 is best at prefill, template instantiate both with shape-dispatched selection.
3. **All n_tokens regimes pass byte-identity + perf bind.** §10 unit test extended to cover n_tokens ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024} × n_kv ∈ {32, 256, 1024, 4096, 16384} × HEAD_DIM_Q × seeds. ≥ 50 byte-identity configs and ≥ 30 perf configs.
4. **Failure modes are explicit and surfaced — not silently fallen-back-on.** If any shape combination cannot meet either the byte-identity or perf bind, the spec is updated (with rationale + user sign-off) BEFORE shipping. Per `feedback_surface_tradeoff_decisions.md`: surface ALL of them.
5. **Production health-check + T9.1 validity binding GREEN with the new kernel default-on.** Includes the existing 8-prompt production capture set across NP ∈ {1, 2, 4, 8}.

Per `feedback_no_workarounds`: a straight replacement means no escape hatch. If something is broken at a shape we ship to, production breaks. The thoroughness bar above is the only way to land this safely.

---

## 12. Open questions for user input

Per §11 straight-replacement mandate: every OQ below must be RESOLVED (not deferred to implementation) before the spec is locked. Each carries a proposal; user sign-off on each closes the spec.

**OQ-1 — KV_BLOCK_SIZE: single value or shape-dispatched.** Proposal: instantiate BOTH `KV_BLOCK_SIZE=32` (double-buffered K+V, 2 blocks/SM) and `KV_BLOCK_SIZE=64` (single-buffer, 1 block/SM) as templates. Pre-merge profile sweep at §10 S2.5.d locks per-shape dispatch table (e.g., `KV_BLOCK_SIZE=32 if n_kv ≤ 1024 else 64`). Single value risks under-tuning at one regime.

**OQ-2 — slot_seq_lens build-graph path.** Proposal: (a) new ggml input tensor populated from `llama_context`'s per-slot KV-cache state. Plumbed through `build_qwen35.cpp` (and any other build_* that uses full_attention) and stashed on `default_decoder`. Required for production flexibility. (b) op_params-encoded is a non-starter since max_n_seqs varies across deployments and the op_params slot is small.

**OQ-3 — Combine kernel.** Proposal: REUSE `flash_attn_combine_results` from `fattn-common.cuh`. The output layout from `fattn_per_slot_kv_sm75` must match wmma_f16's `dst_partial` + `dst_meta` layout exactly. Implementation must verify with a unit test asserting layout compatibility before any production use. If a future shape needs a different layout, write `fattn_combine_per_slot_kv_sm75` as a sibling rather than altering the existing one.

**OQ-4 — HEAD_DIM_Q coverage.** Per §11 thoroughness mandate, every production HEAD_DIM_Q must be template-instantiated. Production models on this host:
- Qwen 3.5 / 3.6 (production target): HEAD_DIM_Q=256, HEAD_DIM_V=128
- Older Qwen / other models that may share this path: needs explicit audit

Proposal: instantiate `<256, 128>` for the production target, plus `<128, 128>` for likely-needed compatibility. Audit at S2.5 start; add more instantiations if any production model uses different values. The dispatcher routes (HEAD_DIM_Q, HEAD_DIM_V) tuples to the matching template; unsupported tuples → GGML_ABORT (NOT a silent fallback to wmma_f16, since wmma_f16 is gone on sm_75 after this lands).

**OQ-5 — Mask shape compatibility.** Proposal: kernel reads mask `[n_kv_max, n_tokens]` AND uses `slot_seq_lens[slot]` for K-loop bound. Mask is evaluated for in-range K positions (alibi, special-position masking). Mask read for K positions ≥ slot_seq_lens[slot] is SKIPPED (the K-loop terminates first). Equivalence with wmma_f16: at slot K's valid K range, the mask is applied identically; beyond it, the new kernel doesn't iterate (wmma_f16 iterates with mask-zero contribution; the new kernel doesn't iterate at all — the empirical n_kv-pad evidence shows this is the source of the difference we want to eliminate).

**OQ-6 — Soft-capping (`USE_SOFTCAP` template).** Proposal: instantiate both `USE_SOFTCAP=false` and `USE_SOFTCAP=true` templates. Dispatcher routes based on `op_params[2] != 0`. Qwen 3.6 27B uses softcap on some layers; both template specializations required.

**OQ-7 — ALiBi / position-bias handling.** wmma_f16 supports `get_alibi_slope` for ALiBi position bias. Qwen 3.6 27B does not use ALiBi (uses RoPE on Q/K instead). Proposal: kernel template-instantiates `USE_ALIBI=false` for the production-Qwen path. If other models need ALiBi on sm_75 after wmma_f16 removal, add `USE_ALIBI=true` instantiation. Audit during S2.5 start.

**OQ-8 — Attention sinks.** wmma_f16 supports attention sinks (`flash_attn_ext_add_sinks`). Production Qwen 3.6 27B does not currently use this. Proposal: skip in initial template; add `USE_SINKS=true` instantiation if/when needed. Audit before commit.

---

## 13. References

- `specs/dflash/kernel-design.md §6.3` — `dflash_verify_attn` (direct ancestor template; same fixed-tile + per-output-row CTA + mma.sync.m16n8k8 pattern)
- [ssiu/flash-attention-turing](https://github.com/ssiu/flash-attention-turing) — only public FA implementation tuned for sm_75 head_dim=128, ~63% peak on T4
- [TML batch_invariant_ops Triton FA](https://github.com/thinking-machines-lab/batch_invariant_ops) — fixed-split-size + per-row CTA reference
- [llama.cpp PR #16016 deterministic FA](https://github.com/ggml-org/llama.cpp/pull/16016) — CUDA implementation of fixed-split-size
- `feedback_kernel_replacements_must_be_sota_sm75.md` — the SoTA mandate driving §2, §8, §9
- `feedback_determinism_must_co_optimize_perf.md` — co-equal det + perf binding
- `data/deltanet/s2-3-nkvpad-confirmation.json` — empirical evidence locking the n_kv mechanism

---

## 14. Allium binding

This spec is the design realization of:
- `FA_NkvIsDominantBatchShapeEntryPoint` (the kernel's per-slot K-loop bound = this invariant's bind point)
- `MMQ_FaithfulPropagation` (MMQ is NOT replaced; FA fix suffices)
- `RepDeterminismAtSameNP` (preserved by no-atomics, no-cross-block-reduction within kernel)
- `LayerForwardIsPerSlotDeterministic` (the contract in batch-invariance.allium)
