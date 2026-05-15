# PHASE_MMQ_Q4_0_AR16 — write MMQ kernels for Q4_0_AR16

Date: 2026-05-15
Branch: production/2026-q2-next
Status: PLAN — design only, no code yet

Source artifacts:
- `specs/deltanet/fattn-per-slot-kv-sm75.md §15.22` (locked target).
- `MEMORY.md` entry `[NP-determinism complete closure (single-GPU)]` (current closure state).
- `specs/deltanet/batch-invariance.allium` (the invariants this phase restores at concurrent batched-decode).

## 1. Context

The current NP-determinism closure (committed 2026-05-15) requires `LLAMA_FATTN_STRICT_SEQUENTIAL_DECODE=1` — serializes all slot processing at the server level. Throughput drops to NP=1-effective. The user has named the next constraint to remove: enable concurrent batched-decode without losing byte-identity.

The structural blocker is **production weights are Q4_0_AR16** (AutoRound 16-step variant of Q4_0). At batched-decode, the matmul kernel dispatch in ik_llama.cpp routes Q4_0_AR16 through:
- ne[1] = 1: dequant → cuBLAS GEMV (algo picked per shape).
- ne[1] > 1: dequant → cuBLAS GEMM (algo picked per shape).

cuBLAS auto-algorithm selection means row-0 output differs across ne[1] values. This breaks NP-cross byte-identity at concurrent batched-decode.

The fix at the cost-effective seam: **add Q4_0_AR16 to the MMQ kernel family**. MMQ is per-tile, compile-time-fixed, has provable per-row independence at fixed `mmq_x`. Sister types Q4_0, Q5_0, Q6_0, Q8_0 all live in MMQ with the same structure; adding Q4_0_AR16 follows the established pattern.

### 1.1 Q4_0_AR16 vs Q4_0

| | Q4_0 | Q4_0_AR16 |
|---|---|---|
| Block size | 32 elements | 16 elements |
| Per-block storage | 1 fp16 scale + 16 B nibbles | 1 fp16 scale + 8 B nibbles |
| Total bytes/block | 18 | 10 |
| Nibble packing | k even = low nibble, k odd = high nibble | identical |
| Sign offset | -8 (recentered from unsigned 0..15) | identical |

The block layout is identical except for block-size. The asymmetry: Q8_1 (activation quantization) has 32-element blocks (`QK8_1 = 32`). For Q4_0 there's a 1:1 block alignment; for Q4_0_AR16 there's a 2:1 alignment (two AR16 weight blocks dot against one Q8_1 activation block).

### 1.2 Verification anchors (already exist, used as oracles)

- `ggml_vec_dot_q4_0_ar16_q8_0` in `ggml/src/ggml.c:1634` — scalar CPU vec-dot. Reference oracle for the per-block math.
- `dequantize_block_q4_0_ar16` in `ggml/src/ggml-cuda/convert.cu:115` — CUDA dequant kernel. Reference for the per-nibble unpacking.

## 2. Goal binding (single closure criterion)

Concurrent batched-decode at NP ∈ {2, 4, 8} produces byte-identical slot-0 token sequences to NP=1 baseline, under this stack:

```bash
LLAMA_FATTN_PER_SLOT_KV_ENABLE=1 \
LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
llama-server --device CUDA0 --no-cont-batching ...
```

Note: **`LLAMA_FATTN_STRICT_SEQUENTIAL_DECODE` is NOT set**. That's the whole point — we drop the serialization sledgehammer.

Bound by: `scripts/test-fattn-per-slot-kv-np-determinism.sh` PASS, all NP values, all slot outputs byte-identical.

## 3. Design

Pattern: clone Q4_0's MMQ touchpoints, adjust for 16-element blocks.

### 3.1 Block packing layout — kept identical to Q4_0

Each AR16 block is 10 bytes: 2 B scale + 8 B nibbles. The 8 B = 16 nibbles = 16 quants. Even k = low nibble, odd k = high nibble of byte k/2. After load + sign-subtract, each AR16 block gives 16 int8s in [-8, 7].

For MMQ this means **a 32-element K-chunk = 2 AR16 blocks back-to-back in memory**. The first AR16 block provides codes for k=0..15; the second for k=16..31. Both blocks have their own fp16 scale. The Q8_1 activation has one fp16 scale + 1 fp16 sum for the same 32-element span.

Per-32-element dot-product: `result = ar16_block[0].d * dot(int8 codes[0..15], q8_1.qs[0..15]) + ar16_block[1].d * dot(int8 codes[16..31], q8_1.qs[16..31])`. Then sum (or sum-correction) via the Q8_1 layout.

### 3.2 Q8_1 DS layout choice

Q4_0 uses `MMQ_Q8_1_DS_LAYOUT_DS4` (d + sum). The sum is the row-sum of the Q8_1 activation block. It's used for asymmetric quant types (Q4_1 etc) but is included for Q4_0 too because the recentered code (qs - 8) introduces a constant 8-bias that must be subtracted.

Q4_0_AR16 has the same recentering (-8 offset). It needs the sum too. **Use `MMQ_Q8_1_DS_LAYOUT_DS4` for AR16.** Same as Q4_0.

### 3.3 Tile-size and traits

Approach: try to share Q4_0's `MMQ_DP4A_TXS_Q4_0` and `MMQ_MMA_TILE_X_K_Q8_0` directly. The tile structure works in terms of `mmq_y` (output-tile-rows) and `mmq_x` (output-tile-cols) which are unrelated to AR16's smaller block; the K-direction inside a tile sums over WARP_SIZE elements regardless of source block size. The block-size only affects HOW the per-K-position quant is loaded.

### 3.4 Existing AR16 kernels to inherit

`dequantize_block_q4_0_ar16` (convert.cu:115) — gives us the validated per-nibble unpack formula. The MMQ load_tiles is essentially "do this dequant + write to SMEM in MMQ's expected layout". Reuse the unpacking math.

## 4. Step-by-step plan with verification

Each step is binding; closure requires its verification to PASS on the named oracle.

### S1 — Survey existing Q4_0 MMQ touchpoints; produce a delta sheet

Files to touch (audit by reading + grep):
- `ggml/src/ggml-cuda/mmq.cuh` — `mmq_get_q8_1_ds_layout`, `mmq_get_dp4a_tile_x_sizes`, `mmq_get_mma_tile_x_k`, `load_tiles_q4_0`, `vec_dot_q4_0_q8_1_dp4a`, `mmq_type_traits<...,GGML_TYPE_Q4_0>`, `DECL_MMQ_CASE`.
- `ggml/src/ggml-cuda/mmq.cu` — `ggml_cuda_should_use_mmq` supported-type list, `mul_mat_q_case<GGML_TYPE_Q4_0>` instantiation.
- `ggml/src/ggml-cuda/quantize.cu` — `quantize_mmq_q8_1_cuda` (the Q8_1 quantization of activation rows). Currently aborts on unsupported types.
- `ggml/src/ggml-cuda.cu` — `ggml_cuda_mul_mat` dispatch and the `LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH` env path.

Verification: PHASE_MMQ_Q4_0_AR16.md gains a §5 "delta sheet" listing exactly which lines in each file to mirror.

### S2 — Add Q4_0_AR16 to `ggml_cuda_should_use_mmq` supported-type list

Single-line edit in `mmq.cu` around line 195 area: add `case GGML_TYPE_Q4_0_AR16:` to the `switch (type)` that sets `mmq_supported = true`.

Verification: build succeeds; an empty test call `ggml_cuda_should_use_mmq(GGML_TYPE_Q4_0_AR16, /*cc=*/750, /*ne11=*/4)` returns true. No actual kernel invocation yet — just the type-support gate.

### S3 — Add `mmq_get_q8_1_ds_layout` case for Q4_0_AR16

Mirror Q4_0's `MMQ_Q8_1_DS_LAYOUT_DS4`. Single-line edit in `mmq.cuh:50` area.

Verification: build succeeds.

### S4 — Add `mmq_get_dp4a_tile_x_sizes` and `mmq_get_mma_tile_x_k` cases for Q4_0_AR16

Mirror Q4_0's `MMQ_DP4A_TXS_Q4_0` and `MMQ_MMA_TILE_X_K_Q8_0`. Two single-line edits.

Verification: build succeeds.

### S5 — Implement `load_tiles_q4_0_ar16<mmq_y, nwarps, need_check>`

Pattern: clone `load_tiles_q4_0` (mmq.cuh:316). Adjust for 16-element blocks:
- `QI_AR16 = QK_AR16 / 8 = 2` (32-bit ints per block).
- Where Q4_0 has `kbx = threadIdx.x / QI4_0` and `kqsx = threadIdx.x % QI4_0`, AR16 has `kbx = threadIdx.x / QI_AR16` and `kqsx = threadIdx.x % QI_AR16`.
- Per (i, kbx, kqsx), read one int from `block_q4_0_ar16[i*stride + kbx].qs[kqsx]` and one fp16 scale.
- Sign-subtract 8, same as Q4_0.
- SMEM writes: int8s into `x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + ...]`; scale into `x_df[...]`. Use the same SMEM layout as Q4_0 — the consumer kernels (vec_dot) read from the same offsets.

Test oracle: `dequantize_block_q4_0_ar16` in convert.cu:115 produces the same per-nibble values. Cross-check on random inputs.

Verification: a new `tests/dflash-speculative/test-mmq-q4-0-ar16-load-tiles.cpp` that loads tiles, reads back via SMEM-dump kernel, compares to dequant oracle. PASS at random fp16 scales + nibble patterns over a sweep of (mmq_y, nwarps).

### S6 — Implement `vec_dot_q4_0_ar16_q8_1_dp4a` (DP4A path)

Per-tile dot-product using DP4A instruction (sm_75 supports). Two AR16 blocks dot against one Q8_1 block per 32-element K span.

Reference: `vec_dot_q4_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>`. Adapt the inner k01 loop to handle AR16's 2-block stride.

Test oracle: scalar CPU `ggml_vec_dot_q4_0_ar16_q8_0` (in ggml.c:1634) — bit-equivalent at exact arithmetic; on GPU we accept fp32-rounded equality.

Verification: a new `tests/dflash-speculative/test-mmq-q4-0-ar16-dp4a.cpp` drives the dp4a path over random inputs, compares to scalar reference. Cosine ≥ 0.9999 at all tested shapes.

### S7 — Implement `vec_dot_q4_0_ar16_q8_1_mma` (Tensor-core MMA path)

For Turing sm_75, MMQ MMA path uses `mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32` (int8 mma). Adapt `vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_DS4>` for AR16's 2-block-per-K span.

Test oracle: matches the DP4A output bit-for-bit (the underlying math is the same; only the SIMD width differs).

Verification: `tests/dflash-speculative/test-mmq-q4-0-ar16-mma.cpp` drives the MMA path over random inputs, compares to DP4A path output at random (mmq_x, mmq_y) tile sizes. Byte-identical.

### S8 — Add `mmq_type_traits<...,GGML_TYPE_Q4_0_AR16>` specialization

Wires `load_tiles_q4_0_ar16` (S5), `vec_dot_q4_0_ar16_q8_1_mma` (S7), `vec_dot_q4_0_ar16_q8_1_dp4a` (S6). Mirror line 3605 (Q4_0's specialization). ~4 LOC edit.

Verification: build succeeds.

### S9 — Add Q4_0_AR16 case to `quantize_mmq_q8_1_cuda`

The activation row is fp32; quantize_mmq_q8_1_cuda converts to Q8_1. Currently aborts for unhandled types at mmq.cuh:111. Need to handle the case where the WEIGHT (src0) is Q4_0_AR16; the ACTIVATION (src1) Q8_1 is unaffected by AR16's block size (Q8_1 has its own 32-element block).

Likely the existing Q4_0 Q8_1-quantization path works as-is. Verify by tracing the abort path; if AR16 just maps to the same Q8_1 quantization (because Q8_1's block size is the activation's, not the weight's), then a single case-label addition `case GGML_TYPE_Q4_0_AR16:` to the same branch suffices.

Verification: round-trip test — F32 activation → Q8_1 → MMQ AR16 weights → output. Compare to scalar reference. PASS.

### S10 — Add Q4_0_AR16 case to `mul_mat_q_case` dispatch in mmq.cu

Mirror line 111 area: add Q4_0_AR16 case calling `mul_mat_q_case<GGML_TYPE_Q4_0_AR16>(ctx, args, stream)`.

Verification: end-to-end test calling `ggml_mul_mat` with Q4_0_AR16 weights, verify GGML_OP_MUL_MAT dispatcher routes to MMQ (via NVTX range "mmq" appearing in nsys trace, NOT "op_mul_mat_cublas").

### S11 — Add MMQ instance file for Q4_0_AR16

Mirror `template-instances/mmq-instance-q4_0.cu`: a 1-line `DECL_MMQ_CASE(GGML_TYPE_Q4_0_AR16);` in a new file `mmq-instance-q4_0_ar16.cu`. Add to CMakeLists if needed.

Verification: build succeeds; binary contains the symbol (nm | grep Q4_0_AR16).

### S12 — Unit test: shape-invariance binding

New test `tests/dflash-speculative/test-mmq-q4-0-ar16-shape-invariance.cpp`.

Random Q4_0_AR16 weight matrix [K, N]; random fp32 activation [K, M] for M ∈ {1, 4, 8, 16}. Slice row 0 = same across all M. Run `ggml_mul_mat` with `LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH=1`. Compare row 0 of output across M. **Byte-identical** = PASS.

This is the kernel-level binding for the concurrent-batched-decode goal.

### S13 — Production NP-cross harness (closure binding)

Run `scripts/test-fattn-per-slot-kv-np-determinism.sh` with:
- `LLAMA_FATTN_PER_SLOT_KV_ENABLE=1`
- `LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH=1`
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- `--no-cont-batching` only (NOT `LLAMA_FATTN_STRICT_SEQUENTIAL_DECODE=1`).
- Single GPU.

**Closure**: all slot outputs at NP ∈ {2, 4, 8} byte-identical to NP=1 baseline.

### S14 — Spec + MEMORY update + plan close

- `specs/deltanet/fattn-per-slot-kv-sm75.md` §15.23: closure delivered, production stack updated.
- `MEMORY.md` (auto) entry: shape-invariant dispatch closure on Q4_0_AR16 path.
- This file marked `[x]` overall and the steps marked `[x]` per their verification.

## 5. Cross-cutting discipline (per CLAUDE.md)

- §1 Think Before Coding — every step has a verification check. If S5's load_tiles fails its byte-equivalence with the dequant oracle, STOP. Do not proceed to S6 with a known-wrong load.
- §2 Simplicity First — don't add SoTA tile tuning; clone Q4_0's tile sizes exactly. Performance is OUT OF SCOPE for this phase. Correctness + byte-identity ONLY.
- §3 Surgical Changes — only AR16-named symbols and the AR16 case-label additions. No drive-by cleanup of Q4_0 code.
- §4 Goal-Driven — closure = S13 binding. Until S13 PASS, the phase stays `[ ]`.
- §5 Plan auditing — every step that adds verification evidence commits + pushes immediately. Code commits go separately from this plan file.
- §6 MEMORY.md — append a closure entry when S13 PASS. Append a "structural finding" entry if any step reveals something unexpected about the AR16 packing format.

## 6. Out of scope

- **Performance**: clone Q4_0 tile sizes exactly. AR16's smaller block means more blocks per K span (2x); we accept the SMEM/register cost increase as the price of correctness. Tuning is post-closure.
- **Multi-GPU PCIe peer-access timing**: separate workstream, named in §15.21 of the FA spec.
- **MMVQ for Q4_0_AR16**: this phase adds MMQ only. MMVQ remains absent for AR16; the dispatch falls through to MMQ at all batch sizes when `LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH=1`.
- **Other quant types**: this phase is AR16-only. Q4_1, Q5_K, etc. remain on their existing paths.

## 7. Open questions to resolve in S1

- Is `MMQ_DP4A_TXS_Q4_0` the right tile sizing for AR16? AR16 has half the block size; the K-direction tile width (`WARP_SIZE` elements) covers 2 AR16 blocks vs 1 Q4_0 block. Does that change `txs.qs` (the int-count per tile row) or `txs.dm` (scale-count)?
- Does `quantize_mmq_q8_1_cuda` have any per-weight-type branching, or is the activation quantization purely a function of the activation tensor? If the latter, S9 is a single case-label add.
- Does the MMA path need a different `MMQ_Q8_1_DS_LAYOUT` for AR16, given the 2-blocks-per-Q8_1-span structure? Or does the DS4 layout naturally accommodate (since the sum is over the activation's 32 elements which span 2 AR16 blocks)?

These need to be answered concretely in S1 before any code lands.

## 8. Snapshot — current state

`production/2026-q2-next` head:
- Q4_0_AR16 in `ggml_cuda_should_use_mmq` supported list: **NO**.
- `load_tiles_q4_0_ar16`: **does not exist**.
- `vec_dot_q4_0_ar16_q8_1_dp4a`: **does not exist**.
- `vec_dot_q4_0_ar16_q8_1_mma`: **does not exist**.
- `mmq_type_traits<...,GGML_TYPE_Q4_0_AR16>`: **does not exist**.
- `mul_mat_q_case<GGML_TYPE_Q4_0_AR16>`: **does not exist**.
- MMQ instance file for AR16: **does not exist**.

This phase changes all 7 to YES.
