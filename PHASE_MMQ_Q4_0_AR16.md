# PHASE_MMQ_Q4_0_AR16 — full shape-invariant dispatch on production stack

Date: 2026-05-15
Branch: production/2026-q2-next
Status: PLAN — design only, no code yet

Source artifacts:
- `specs/deltanet/fattn-per-slot-kv-sm75.md §15.22` (locked target).
- `specs/deltanet/batch-invariance.allium` (invariants restored at concurrent batched-decode).
- `MEMORY.md` entry `[NP-determinism complete closure (single-GPU)]` (single-GPU + strict-sequential baseline).

## 1. Scope

User direction (2026-05-15): everything previously listed as out-of-scope is now in-scope. Full SoTA delivery.

Closure binding: concurrent batched-decode at NP ∈ {2, 4, 8} produces byte-identical slot-0 token sequences to NP=1 baseline, **on multi-GPU (--tensor-split 1,1)**, **without strict-sequential decode**, **without --no-cont-batching**.

Production env stack (target):
```bash
LLAMA_FATTN_PER_SLOT_KV_ENABLE=1 \
LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
llama-server -m <gguf> \
    --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
    --parallel <N> --ctx-size <N*8192> \
    -ngl 999 -fa on \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --k-cache-hadamard --v-cache-hadamard
```

No `LLAMA_FATTN_STRICT_SEQUENTIAL_DECODE`, no `--no-cont-batching`, no `--device CUDA0` single-GPU restriction.

Bound by: `scripts/test-fattn-per-slot-kv-np-determinism.sh` PASS, all slot outputs byte-identical at all tested NP values, ALL ITERATIONS over 5 runs (catch non-deterministic-source residues).

## 2. Answers to open questions (resolved from source 2026-05-15)

### 2.1 Tile sizing — needs new macro for AR16

`MMQ_DP4A_TXS_Q4_0 = tile_x_sizes{mmq_y*WARP_SIZE + mmq_y, mmq_y*WARP_SIZE/QI4_0 + mmq_y/QI4_0, 0}` where `QI4_0 = 4` (= QK4_0/8).

For AR16: `QI_AR16 = QK_AR16/8 = 16/8 = 2`. The `qs` count per tile is identical to Q4_0 (`mmq_y*WARP_SIZE + mmq_y` — int-count doesn't care about block size). The `dm` (scale-count) is **2x larger** for AR16 because there are 2x more blocks per WARP_SIZE-row of K.

New macro required:
```cpp
#define MMQ_DP4A_TXS_Q4_0_AR16 tile_x_sizes{mmq_y*WARP_SIZE + mmq_y, mmq_y*WARP_SIZE/QI_AR16 + mmq_y/QI_AR16, 0}
// expands to: tile_x_sizes{mmq_y*32 + mmq_y, mmq_y*16 + mmq_y/2, 0}
```

For the MMA tile: `MMQ_MMA_TILE_X_K_Q8_0 = 2*WARP_SIZE + 2*WARP_SIZE/QI8_0 + 4 = 76`. For AR16's MMA tile:
```cpp
#define MMQ_MMA_TILE_X_K_Q4_0_AR16 (2*WARP_SIZE + 2*WARP_SIZE/QI_AR16 + 4)
// = 64 + 32 + 4 = 100. % 8 == 4: passes the existing static_assert pattern.
```

### 2.2 Q8_1 quantization is activation-only (weight-type-independent)

`quantize_mmq_q8_1` (quantize.cu:86) takes a Q8_1 DS layout template parameter, but the layout depends on whether the WEIGHT type needs sum correction. For Q4_0 and AR16, the weight has -8 recentering, so DS layout = `MMQ_Q8_1_DS_LAYOUT_DS4` (d + sum).

The actual Q8_1 BLOCK SIZE (`QK8_1 = 32`) doesn't change. The activation is always quantized in 32-element blocks. For AR16, two AR16 weight blocks line up with one Q8_1 activation block.

So `quantize_mmq_q8_1_cuda` requires only a single new case-label `case GGML_TYPE_Q4_0_AR16:` that re-uses the existing `MMQ_Q8_1_DS_LAYOUT_DS4` path. No new quantization kernel needed.

### 2.3 DS4 layout for AR16 — sum-correction computed per-half inline

The DS4 layout stores one `(d, s)` pair per 32-element Q8_1 block. For Q4_0 the sum-correction is:
```
-8 * d_w * d_a * s_a
```
where `d_w` is the single weight scale, `s_a` is the activation sum over 32 elements.

For AR16, two weight scales (`d_w0`, `d_w1`) cover the 32-element span. The sum-correction must split:
```
-8 * d_w0 * d_a * s_a_half0 + -8 * d_w1 * d_a * s_a_half1
```
where `s_a_half0 = sum(qs[0..15])` and `s_a_half1 = sum(qs[16..31])`.

`block_q8_1_mmq` only stores `s_a` (the full 32-element sum). Computing half-sums requires summing the int8 qs values inline in the AR16 MMQ kernel. Cost: ~16 int8 additions per 32-element K-chunk per per-warp output tile = negligible vs the dot-product itself.

So the existing DS4 layout works without storage change. The kernel does the extra half-sum reduction inline.

### 2.4 Multi-GPU peer-access already has events but non-deterministic in practice

`cudaStreamWaitEvent` is already used at ggml-cuda.cu:4383 and 5460. But empirically multi-GPU `--tensor-split 1,1` produced non-deterministic outputs in earlier tests (see spec §15.21).

The non-determinism source must be either:
(a) Event-based sync misses some path. Need to audit ALL peer-access call sites and confirm event sync.
(b) The order of cross-device memcpy is timing-dependent even with events. Solution: serialize device-by-device processing (force one device to finish before the other starts).

This phase (Phase D) audits and fixes.

## 3. Architecture overview

Six phases, each with binding closure. Sequential dependency: A → B → C → D → E → F. Each phase commits + pushes immediately upon its closure binding.

```
                            +---> Phase F (closure binding)
                            |          ↑
Phase A ─→ Phase B ─→ Phase C ─→ Phase D ─→ Phase E
(Q4_0_AR16  (Q4_0_AR16   (cuBLAS    (multi-GPU   (SoTA
 MMQ)        MMVQ)        pinning)   peer-access  perf
                                     determinism) tuning)
```

## 4. Phase A — Q4_0_AR16 MMQ kernels

Goal: Q4_0_AR16 weight + F32 activation → MMQ kernel path; byte-identical row-0 output across `mmq_x` ∈ {compile-time-supported set}.

### A.1 — Add Q4_0_AR16 to MMQ supported-types

`ggml/src/ggml-cuda/mmq.cu:180-228`: add `case GGML_TYPE_Q4_0_AR16:` to the `switch (type)` setting `mmq_supported = true`.

Verification: `ggml_cuda_should_use_mmq(GGML_TYPE_Q4_0_AR16, 750, 16)` returns true. Build passes.

### A.2 — Add Q4_0_AR16 cases to layout tables

`mmq.cuh:50-area` `mmq_get_q8_1_ds_layout`: case Q4_0_AR16 → `MMQ_Q8_1_DS_LAYOUT_DS4`.

`mmq.cuh:179-area`: add `MMQ_DP4A_TXS_Q4_0_AR16` macro (per §2.1).

`mmq.cuh:190-236` `mmq_get_dp4a_tile_x_sizes`: case Q4_0_AR16 → `MMQ_DP4A_TXS_Q4_0_AR16`.

`mmq.cuh:238-242`: add `MMQ_MMA_TILE_X_K_Q4_0_AR16` macro (per §2.1). Static_assert `% 8 == 4`.

`mmq.cuh:270-area` `mmq_get_mma_tile_x_k`: case Q4_0_AR16 → `MMQ_MMA_TILE_X_K_Q4_0_AR16`.

Verification: build passes.

### A.3 — Implement `load_tiles_q4_0_ar16<mmq_y, nwarps, need_check>`

Clone `load_tiles_q4_0` (mmq.cuh:316). Adjust:
- `QI_AR16 = 2` replaces `QI4_0 = 4` in `kbx`/`kqsx` decomposition.
- Source struct is `block_q4_0_ar16` (10 bytes) not `block_q4_0` (18 bytes).
- Each thread reads 1 byte (= 2 nibbles = 2 quants) into one int32 by `get_int_b2(bxi->qs, kqsx)`.
- Sign-subtract is same (`__vsubss4(... & 0x0F0F0F0F, 0x08080808)`).
- SMEM layout: x_qs uses same `MMQ_MMA_TILE_X_K_Q4_0_AR16` (= 100) row stride for INT8_MMA; x_df uses 2x stride for AR16's 2x more scales.

Verification: new test `tests/dflash-speculative/test-mmq-q4-0-ar16-load-tiles.cpp`:
- Generate random `block_q4_0_ar16` rows.
- Dispatch a kernel that calls `load_tiles_q4_0_ar16`, then dumps `x_qs` and `x_df` to global memory.
- Compute the expected `x_qs` and `x_df` from `dequantize_block_q4_0_ar16` (convert.cu:115) which IS the validated unpacker.
- Byte-equivalent across (mmq_y, nwarps, need_check) sweep.

### A.4 — Implement `vec_dot_q4_0_ar16_q8_1_dp4a<mmq_x, mmq_y, nwarps>`

Clone `vec_dot_q4_0_q8_1_dp4a` (mmq.cuh:443). Adjust per §2.3:
- For each K-chunk of 32 vals, compute the dot product as TWO half-dot-products with separate weight scales.
- Inside the inner accumulation: `sum += d_w0 * d_a * sum(int8(qx_half0) * int8(qy_half0)) - 8 * d_w0 * d_a * sum(qy_half0) + d_w1 * d_a * sum(qy_half1) ... -8 * d_w1 * d_a * sum(qy_half1)`.
- The half-sums of `qy` are computed inline from `qy[0..15]` and `qy[16..31]` (16 int8 additions each per half-sum). One DP4A instruction takes 4 bytes; sum across 4 DP4As per half.

Verification: new test `tests/dflash-speculative/test-mmq-q4-0-ar16-dp4a.cpp`:
- Random Q4_0_AR16 weight matrix.
- Random Q8_1 activation (run quantize_mmq_q8_1 with DS4 layout).
- DP4A kernel produces output `dst`.
- Scalar CPU reference: dequantize Q4_0_AR16 to F32 + dequantize Q8_1 to F32 + fp32 dot-product.
- Cosine ≥ 0.9999, NMSE ≤ 1e-4. Per row sweep over (mmq_x, mmq_y, K).

### A.5 — Implement `vec_dot_q4_0_ar16_q8_1_mma<mmq_x, mmq_y, nwarps>`

Clone `vec_dot_q8_0_q8_1_mma<...,DS4>` (mmq.cuh:area). The MMA path uses `mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32` (int8 MMA on sm_75). Same -8 correction with per-half scaling.

Implementation strategy:
- MMA produces int32 accumulator from int8 × int8.
- After MMA, apply the per-K-chunk scale: `(d_w0, d_w1) * d_a * mma_int32_result`.
- Sum-correction: `-8 * (d_w0 * d_a * s_a_half0 + d_w1 * d_a * s_a_half1)`. The half-sums come from the MMA input fragment summed inline.

Verification: new test `tests/dflash-speculative/test-mmq-q4-0-ar16-mma.cpp`:
- Same setup as A.4.
- MMA kernel output bit-identical to DP4A kernel output (the underlying math is the same; this verifies the MMA path).

### A.6 — Add `mmq_type_traits<...,GGML_TYPE_Q4_0_AR16>`

Clone mmq.cuh:3605 specialization. Wire `load_tiles_q4_0_ar16` (A.3), `vec_dot_q4_0_ar16_q8_1_mma` (A.5), `vec_dot_q4_0_ar16_q8_1_dp4a` (A.4).

Verification: build passes.

### A.7 — Add Q4_0_AR16 to `quantize_mmq_q8_1_cuda` switch

Single case-label add at quantize.cu:area. Re-uses `MMQ_Q8_1_DS_LAYOUT_DS4` path. No new quantization kernel.

Verification: quantize round-trip test in `tests/dflash-speculative/test-mmq-q4-0-ar16-quantize.cpp`. F32 activation → Q8_1 → ds4 layout → dequant → cosine ≥ 0.9999.

### A.8 — Add Q4_0_AR16 to `mul_mat_q_case` dispatch

`mmq.cu:111-area`: add `case GGML_TYPE_Q4_0_AR16: mul_mat_q_case<GGML_TYPE_Q4_0_AR16>(ctx, args, stream); break;`.

### A.9 — Add MMQ instance file

New file `ggml/src/ggml-cuda/template-instances/mmq-instance-q4_0_ar16.cu`:
```cpp
#include "../mmq.cuh"
DECL_MMQ_CASE(GGML_TYPE_Q4_0_AR16);
```
Add to CMakeLists if instance files are listed explicitly.

Verification: `nm build/ggml/src/libggml.so | grep Q4_0_AR16` returns the symbol.

### A.10 — Shape-invariance unit test

`tests/dflash-speculative/test-mmq-q4-0-ar16-shape-invariance.cpp`:
- Random Q4_0_AR16 weight `[K, N]`. Random F32 activation row `a0[K]`.
- Build activation tensors at `M ∈ {1, 4, 8, 16, 32}` with `M[0] = a0`, `M[1..M-1] = random`.
- Run `ggml_mul_mat` with `LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH=1` for each `M`.
- Compare output row 0 across `M`. **Byte-identical** = PASS.

This is the Phase A closure binding. If it fails, do NOT proceed to Phase B.

## 5. Phase B — Q4_0_AR16 MMVQ kernels

Goal: Q4_0_AR16 also has MMVQ kernels so the default dispatch (without `LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH=1`) can route to MMVQ for small batch. This is for completeness: under the shape-invariant dispatch flag, MMQ is always chosen; under default dispatch, MMVQ allows perf at small batch.

### B.1 — Add Q4_0_AR16 to `ggml_cuda_mmvq_type_supported`

Single case-label add.

### B.2 — Implement `mul_mat_vec_q_q4_0_ar16_q8_1_cuda`

Clone `mul_mat_vec_q4_0_q8_1_cuda` (mmvq.cu:50). Per-row CTA structure (each output row gets its own CTA per the MMVQ pattern). Apply -8 correction with per-half-block weight scales.

Verification: `tests/dflash-speculative/test-mmvq-q4-0-ar16.cpp`. Cosine + NMSE against scalar reference.

### B.3 — Add Q4_0_AR16 case to MMVQ dispatch in `mmvq.cu`

Mirror the Q4_0 case at mmvq.cu:50.

### B.4 — Shape-invariance binding for MMVQ

Repeat the A.10 test but route through MMVQ (via `LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH=0`, batch ≤ MMVQ_MAX_BATCH_SIZE). Row 0 byte-identical across `M ∈ {1, 2, 4, 8}`.

## 6. Phase C — cuBLAS algorithm pinning for F16/BF16/F32 matmuls

Goal: ops with F16 / BF16 / F32 weights (`token_embd`, `lm_head`, RMSNorm output projection) currently route through `ggml_cuda_mul_mat_batched_cublas` or the cuBLAS path inside `ggml_cuda_op_mul_mat_cublas`. cuBLAS picks `cublasGemmAlgo_t` per (M, N, K, dtype) — different M (batch) → different algorithm → different fp accumulator order.

### C.1 — Survey all `cublasGemmEx` call sites in ggml-cuda

Grep for `cublasGemmEx`, `cublasSgemm`, `cublasGemmBatchedEx`, `cublasGemmStridedBatchedEx`. Document call site, current algo argument (typically `CUBLAS_GEMM_DEFAULT_TENSOR_OP` or similar), and the data types.

### C.2 — Force a fixed algorithm under env

Modify each call to (under env `LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH=1`) pass a fixed `cublasGemmAlgo_t` AND set `cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH)` once at handle creation.

Algorithm choice: `CUBLAS_GEMM_ALGO0_TENSOR_OP` is a deterministic choice on Turing. Verify by:
- Per call site, run with this algo at M ∈ {1, 4, 8, 16}. Row 0 must be byte-identical.

### C.3 — cuBLAS pedantic math handle setup

Add `cublasSetMathMode(ctx.cublas_handle(id), CUBLAS_PEDANTIC_MATH)` under env. Affects all subsequent calls on that handle. Set once on first use.

### C.4 — Closure binding for Phase C

New test `tests/dflash-speculative/test-cublas-pinned-shape-invariant.cpp`:
- F16 weight `[K, N]` (e.g., `K=5120, N=5120` matching token_embd-class shapes).
- F32 activation `[K, M]` for `M ∈ {1, 4, 8, 16, 32}`.
- Row 0 of activation identical across M.
- Run `ggml_cuda_mul_mat_batched_cublas` with env on.
- Row 0 of output byte-identical across M.

## 7. Phase D — Multi-GPU PCIe peer-access deterministic ordering

Goal: with `--device CUDA0,CUDA1 --tensor-split 1,1`, the output is byte-identical to single-GPU output. Currently non-deterministic due to peer-access timing.

### D.1 — Survey all `cudaMemcpyPeerAsync` call sites

From earlier scan: 5 sites in ggml-cuda.cu at lines 816, 2184, 2190, 4348, 4368.

For each, document:
- Source/dest devices.
- Stream.
- Event sync before/after.

### D.2 — Identify the non-deterministic site

Hypothesis (per §15.21): one or more peer-access copies have no preceding `cudaStreamWaitEvent`, so the dest stream may start consuming before the source stream finishes writing. Result: race condition.

Audit each site. Where missing, add explicit event creation + waitEvent pair.

### D.3 — Force serialized cross-device boundary

Even with events, the ORDER of inter-device transfers may matter. Add (under env) a deterministic ordering policy:
- All cross-device transfers complete before the next graph node begins.
- Use a single global event chain: each transfer signals an event; the next dependent kernel waits.

### D.4 — Closure binding for Phase D

New test `scripts/probe-multi-gpu-shape-invariant.sh`:
- Single-GPU NP=1 baseline.
- Multi-GPU NP=1 (--device CUDA0,CUDA1 --tensor-split 1,1) — should match single-GPU baseline.
- Multi-GPU NP={2,4,8} concurrent — all slots match the single-GPU baseline.

Bind: PASS on all NPs and devices.

## 8. Phase E — SoTA performance tuning

Required by `feedback_kernel_replacements_must_be_sota_sm75`: every kernel replacement must include register budget, occupancy, target % of TU102 peaks, committed nsys/ncu data.

For each new kernel in Phases A and B:

### E.1 — `load_tiles_q4_0_ar16` budget

- Register count target: ≤ 32 regs/thread (matches Q4_0's load tile).
- Occupancy: 2 CTAs/SM at WARP_SIZE × nwarps threads/CTA.
- Target peak: not perf-critical (load step), goal is correctness + footprint.
- Measure: `--ptxas-options=-v` registers, `cudaOccupancyMaxActiveBlocksPerMultiprocessor` blocks/SM.

### E.2 — `vec_dot_q4_0_ar16_q8_1_dp4a` budget

- Register count target: ≤ 48 regs/thread.
- Occupancy: 2 CTAs/SM target.
- Target peak: ≥ 60% of TU102 DP4A peak (Turing INT8 = 130 TFLOPS peak; DP4A subset achievable ~80 TFLOPS).
- Measure: nsys range capturing one matmul + ncu metric `sm__inst_executed_pipe_xu_dp4a.sum` divided by elapsed time → effective DP4A throughput.
- Commit: `data/mmq-q4-0-ar16-dp4a.{nsys-rep,sqlite,csv}` with the measurement results.

### E.3 — `vec_dot_q4_0_ar16_q8_1_mma` budget

- Register count target: ≤ 56 regs/thread (MMA register pressure).
- Occupancy: 2 CTAs/SM.
- Target peak: ≥ 70% of TU102 INT8 tensor-core peak.
- Measure + commit: as E.2.

### E.4 — `mul_mat_vec_q_q4_0_ar16_q8_1_cuda` (MMVQ) budget

- Register count target: ≤ 32 regs/thread.
- Per-row CTA structure → grid scales with output rows.
- Target peak: ≥ 70% of DP4A peak.
- Measure + commit.

### E.5 — Overall production benchmark

After Phases A–D close: run `llama-bench` with the full stack (env + multi-GPU + shape-invariant + no strict-seq), measure:
- Prefill throughput (PP @ 512 tokens).
- Decode throughput (TG @ 32 tokens).
- Compare against the single-GPU + strict-sequential baseline.

Closure: shape-invariant stack ≥ 0.95× the strict-sequential baseline at prefill, ≥ 4× at decode (concurrent batched-decode now active).

### E.6 — Profile-driven tuning if Phase E targets miss

If any of E.1–E.4 misses target: investigate via ncu. Common levers:
- mmq_x / mmq_y tile-size grid search via `cudaOccupancyMaxActiveBlocksPerMultiprocessor` sweep.
- Register pressure: `--maxrregcount`.
- SMEM layout: bank-conflict elimination, padding.

Document the tuning iterations in this PHASE file.

## 9. Phase F — Closure binding

### F.1 — Production NP-cross harness, multi-GPU, no strict-sequential

Run `scripts/test-fattn-per-slot-kv-np-determinism.sh` with the §1 production stack (multi-GPU, no strict-seq, no --no-cont-batching).

**Bind**: all slot outputs at NP ∈ {2, 4, 8} byte-identical to NP=1 baseline. Multi-run stability: run 5 times, all results byte-identical.

### F.2 — Server perf benchmark

Run `llama-bench` (or `llama-batched-bench`) at production shape with NP=8 concurrent. Throughput must be ≥ 4× the single-GPU + strict-sequential baseline (concurrent parallelism restored).

### F.3 — Spec + MEMORY close

Update `specs/deltanet/fattn-per-slot-kv-sm75.md §15.23`: full closure delivered.

`MEMORY.md` (auto): append entry on the closure.

## 10. Discipline (CLAUDE.md)

- §1 Think Before Coding — open questions resolved in §2 of this file before any code lands.
- §2 Simplicity First — within each kernel, clone the closest sibling kernel (Q4_0). No speculative abstractions.
- §3 Surgical Changes — only AR16-named symbols + case-label additions + env-gated cuBLAS pinning + env-gated peer-access serialization.
- §4 Goal-Driven — every step has a binding verification check. Stop on failure.
- §5 Plan auditing — each phase commits + pushes immediately upon closure. Code commits go separately from plan-file updates.
- §6 MEMORY.md — append entries on phase closures and on structural findings.
- `feedback_no_skipping_lessening` — no declaring "good enough at single-GPU". The full stack is the goal.
- `feedback_kernel_replacements_must_be_sota_sm75` — every new kernel has the Phase E budget + committed measurement data.
- `feedback_test_first_discipline` — every step that adds a kernel has its test driver written BEFORE the kernel, in RED-first style.
- `feedback_anchor_to_measured_baselines` — Phase F.2 perf comparison must use fresh-measured baseline at the same hardware + same prompt set.

## 11. Verification matrix (single source of truth for closure)

| Step | Test driver | Closure binding |
|---|---|---|
| A.3 | test-mmq-q4-0-ar16-load-tiles.cpp | byte-equivalent to dequant_block oracle, sweep (mmq_y, nwarps, need_check) |
| A.4 | test-mmq-q4-0-ar16-dp4a.cpp | cos ≥ 0.9999, NMSE ≤ 1e-4 vs scalar fp32 ref |
| A.5 | test-mmq-q4-0-ar16-mma.cpp | byte-identical to A.4 DP4A path output |
| A.7 | test-mmq-q4-0-ar16-quantize.cpp | round-trip cos ≥ 0.9999 |
| A.10 | test-mmq-q4-0-ar16-shape-invariance.cpp | row 0 byte-identical at M ∈ {1,4,8,16,32} |
| B.2 | test-mmvq-q4-0-ar16.cpp | cos + NMSE vs scalar ref |
| B.4 | test-mmvq-q4-0-ar16-shape-invariance.cpp | row 0 byte-identical at M ∈ {1,2,4,8} |
| C.4 | test-cublas-pinned-shape-invariant.cpp | row 0 byte-identical at M ∈ {1,4,8,16,32} for F16/BF16/F32 weights |
| D.4 | probe-multi-gpu-shape-invariant.sh | multi-GPU NP={1,2,4,8} all byte-identical to single-GPU NP=1 |
| E.5 | llama-bench benchmark | shape-invariant stack ≥ 0.95× prefill, ≥ 4× decode vs single-GPU+strict-seq baseline |
| F.1 | test-fattn-per-slot-kv-np-determinism.sh | all slots at NP={2,4,8} byte-identical to NP=1, 5 runs stable |
| F.2 | llama-bench (full prod stack) | concurrent throughput restored |

## 12. Snapshot — current state of touchpoints

`production/2026-q2-next` head:

| File | Symbol | Status |
|---|---|---|
| ggml-cuda/mmq.cu | `Q4_0_AR16` in `ggml_cuda_should_use_mmq` switch | NO |
| ggml-cuda/mmq.cuh | `MMQ_DP4A_TXS_Q4_0_AR16` macro | NO |
| ggml-cuda/mmq.cuh | `MMQ_MMA_TILE_X_K_Q4_0_AR16` macro | NO |
| ggml-cuda/mmq.cuh | `load_tiles_q4_0_ar16` | NO |
| ggml-cuda/mmq.cuh | `vec_dot_q4_0_ar16_q8_1_dp4a` | NO |
| ggml-cuda/mmq.cuh | `vec_dot_q4_0_ar16_q8_1_mma` | NO |
| ggml-cuda/mmq.cuh | `mmq_type_traits<...,GGML_TYPE_Q4_0_AR16>` | NO |
| ggml-cuda/mmq.cu | `Q4_0_AR16` in `mul_mat_q_case` | NO |
| ggml-cuda/template-instances/mmq-instance-q4_0_ar16.cu | file | NO |
| ggml-cuda/quantize.cu | `Q4_0_AR16` in `quantize_mmq_q8_1_cuda` switch | NO |
| ggml-cuda/mmvq.cu | `mul_mat_vec_q_q4_0_ar16_q8_1_cuda` + dispatch case | NO |
| ggml-cuda.cu | env-gated `cublasSetMathMode` + fixed algo | NO |
| ggml-cuda.cu | env-gated peer-access serialization | NO |
| data/mmq-q4-0-ar16-{dp4a,mma}.* | profile data | NO |

All 14 transitions must complete for Phase F closure.
