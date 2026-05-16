# PHASE_MMQ_Q4_0_AR16 — full shape-invariant dispatch on production stack

Date: 2026-05-15
Branch: production/2026-q2-next
Status: **Phase A + Phase B + Phase C CLOSED 2026-05-15** — all unit-test sub-steps GREEN. Production NP-token-determinism PARTIAL (Phase F open).

## Progress log

- **2026-05-15** §2.5 layout decision LOCKED (unified Q8_0-style linear-K x_qs). Plan updated, see commit `103b6de`.
- **2026-05-15** A.3 GREEN — `load_tiles_q4_0_ar16` sweep 8/8 PASS (mmq_y ∈ {16,32,64,128}, nwarps ∈ {4,8}, need_check ∈ {false,true}). Submodule commit `cf708fe5`.
- **2026-05-15** A.4 GREEN — DP4A vec_dot sweep 6/6 PASS, cos=1.0, NMSE~8e-14. Includes dispatcher y-ptr arithmetic re-anchor on MMQ_ITER_K (fixes qk<32 integer-truncation latent bug). Submodule commit `76d39f7e`.
- **2026-05-15** A.5 GREEN — MMA (INT8 tensor-core) vec_dot sweep 6/6 PASS, cos=1.0, NMSE~8e-14 at production shapes (mmq_x ∈ {8,16,32}, mmq_y=128, nwarps=8). Submodule commit `95f0b460`.
- **2026-05-15** A.1 + A.6 + A.8 + A.9 wired: mmq_type_traits specialization, mul_mat_q_case dispatch, instance file, supported-types gate. Build clean. Existing binding tests still GREEN. Submodule commit `530eeab5`.
- **2026-05-15** A.10 CLOSURE BINDING GREEN — ggml_mul_mat (Q4_0_AR16 × F32) at M ∈ {1,4,8,16,32} produces BYTE-IDENTICAL dst column 0 under `LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH=1`. Phase A CLOSED. Submodule commit `eb9ee4ab`.
- **2026-05-15** Phase B GREEN — `vec_dot_q4_0_ar16_q8_1` (mmvq) + `mul_mat_vec_q4_0_ar16_q8_1_cuda` instance + supported-types + dispatch. B.2 cosine=0.999988 NMSE=2.3e-05 vs CPU fp32 ref; B.4 row-0 byte-identical across M ∈ {1,2,4,8}. Plus: **MMQ row-0 ≡ MMVQ row-0 byte-for-byte** on same prompt. mmvq-templates.cuh `kby = (kbx * qk) / QK8_1` is the surgical fix for `qk < QK8_1` (AR16's 16 < 32). Phase B CLOSED. Submodule commit pending.
- **2026-05-15** Phase C CLOSED — `ggml_cuda_mul_mat_f16_pinned` (16x16x16 WMMA row-pinned) for F16/BF16 weights AND `ggml_cuda_mul_mat_f32_pinned` (warp-scalar row-pinned) for F32 weights. test-cublas-pinned-shape-invariant: F16/BF16/F32 all byte-identical across M ∈ {1, 4, 8, 16, 32}. Production NP-token-determinism harness improves from 1/14 → 4/14 byte-identical slots vs NP=1 baseline at first F16/BF16 landing; F32 path empirically not hot on Qwen 3.6 production decode (verified by diagnostic counter). Single-GPU also fails token-determinism → upstream non-MMQ-non-cuBLAS op is the residual contributor; Phase D + further probes follow.

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

### 2.1 Tile sizing — unified Q8_0-style layout per §2.5

AR16 uses the unified unpacked sign-recentered layout (linear K per int) for **both** DP4A and MMA paths. Per-row sizing:

- `x_qs`: 64 ints (= 2*WARP_SIZE) per row, 16 AR16 blocks × 4 ints/block. Each int = 4 sign-recentered int8 weights at 4 consecutive K positions.
- `x_df`: 16 floats per row (one per AR16 block).

`QI_AR16 = QK_AR16/(4*QR_AR16) = 16/8 = 2` remains the source-side packing constant (2 ints per AR16 block in raw byte storage). For x_qs layout we use `QI_AR16_LINEAR = 4` (4 ints per block after unpack).

```cpp
// DP4A path: 64 ints + 1 pad per row of qs. 16 scales + small pad per row of df.
#define MMQ_DP4A_TXS_Q4_0_AR16 tile_x_sizes{mmq_y*(2*WARP_SIZE) + mmq_y, mmq_y*16 + mmq_y/4, 0}
// expands to: tile_x_sizes{mmq_y*64 + mmq_y, mmq_y*16 + mmq_y/4, 0}

// MMA path: same layout. Per-row stride MUST match the Q8_0-style pattern Q8_0 uses
// (so vec_dot_q8_0_q8_1_mma can be reused if desired):
#define MMQ_MMA_TILE_X_K_Q4_0_AR16 (2*WARP_SIZE + 2*WARP_SIZE/QI_AR16_LINEAR + 4)
// = 64 + 16 + 4 = 84. % 8 == 4: passes the existing static_assert pattern.
```

Note: previous (pre-§2.5) macro `MMQ_MMA_TILE_X_K_Q4_0_AR16 = 100` (with `2*WARP_SIZE/QI_AR16=32`) is **superseded** — it allocated 32 scale slots/row but only 16 are needed under the unified layout. Reducing to 84 saves SMEM.

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

### 2.5 SMEM storage layout — LOCKED to Q8_0-style linear-K-per-int (2026-05-15 finding)

**Root issue discovered while writing A.4**: AR16's source-byte nibble convention is `qs[i].low = K=2i (even), qs[i].high = K=2i+1 (odd)` (per `dequantize_block_q4_0_ar16` at convert.cu:115). Q4_0's source convention is different: `qs[i].low = K=i (linear), qs[i].high = K=i+16 (block-half-stride)` (per `dequantize_block_q4_0` at convert.cu:78). Storing AR16's raw bytes into x_qs without re-arrangement produces ints whose nibbles map to non-contiguous K positions — making byte-wise `dp4a(v, u)` with linear `y_qs` invalid (lanes cross-multiply mismatched K's).

**Decision (LOCKED)**: AR16's MMQ x_qs storage uses the **Q8_0-style unpacked sign-recentered layout** for BOTH MMA and DP4A paths. After load:

- Per AR16 block (16 K positions): 4 ints in x_qs row.
- Slot `kbx*4 + s` (s ∈ 0..3): byte `b` holds the sign-recentered int8 for K position `4s+b` of the block. Linear K within each int.
- Per warp tile row: 16 blocks × 4 ints/block = **64 ints** = `2*WARP_SIZE` (same total as MMA path).
- x_df: 16 floats per row (one scale per AR16 block).

This **unifies** the DP4A and MMA storage layouts (one `load_tiles` produces correct data for both vec_dot paths). The DP4A vec_dot becomes a trivial `dp4a(v[i], u[i])` accumulator (no inline nibble extraction, no -8 correction — sign recentering is done at load time). The MMA path reuses the existing Q8_0-style fragment loader.

**Tradeoff**: AR16's DP4A x_qs occupies 64 ints/row instead of Q4_0's 32 ints/row (raw-nibble layout). For mmq_y=64, that's 16 KiB x_qs (vs Q4_0's 8 KiB). Within Turing 48 KiB SMEM budget. Q4_0's existing layouts are unchanged.

**Implication for §2.1 macros**:
- `MMQ_DP4A_TXS_Q4_0_AR16` updated to `tile_x_sizes{mmq_y*2*WARP_SIZE + mmq_y, mmq_y*16 + mmq_y/4, 0}` (64 ints + 1 pad per row of qs; 16 scales + small pad per row of df).
- `MMQ_MMA_TILE_X_K_Q4_0_AR16` stays at 100 (64 qs + 32 df + 4 pad — same as before).

**Implication for §2.3**: Half-sum correction is **NOT NEEDED** in the new design. Weights are sign-recentered at load time. Each AR16 block's contribution is `d_w * d_a * dp4a_sum_over_block` directly. The "split into half-sums" complication evaporates.

**Implication for A.3**: Existing committed `load_tiles_q4_0_ar16` (which stored AR16's raw even/odd K layout) MUST be rewritten to produce the unified layout. The existing A.3 test passed against the buggy layout; both the impl and the test must be redone.

**Migration plan** (mid-Phase A reset):
1. Update §A.3 spec to lock unified layout.
2. Revert in-flight A.4 impl (mmq.cuh `vec_dot_q4_0_ar16_q8_1_dp4a` + vecdotq.cuh `vec_dot_q4_0_ar16_q8_1_impl`) — uses wrong layout assumption.
3. Re-implement load_tiles_q4_0_ar16 per unified layout. Re-run A.3 test (new oracle for linear K).
4. Implement new A.4 DP4A vec_dot. Run A.4 test.
5. Implement new A.5 MMA vec_dot (likely reusable `vec_dot_q8_0_q8_1_mma` with scale-stride adapter).

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
Phase A ─→ Phase B ─→ Phase C ─→ Phase CX ─→ Phase D ─→ Phase E
(Q4_0_AR16  (Q4_0_AR16   (cuBLAS    (non-MMQ     (multi-GPU   (SoTA
 MMQ)        MMVQ)        pinning)   non-cuBLAS   peer-access  perf
                                     ops:         determinism) tuning)
                                     DeltaNet
                                     etc.)
```

## 4. Phase A — Q4_0_AR16 MMQ kernels

Goal: Q4_0_AR16 weight + F32 activation → MMQ kernel path; byte-identical row-0 output across `mmq_x` ∈ {compile-time-supported set}.

### A.1 — Add Q4_0_AR16 to MMQ supported-types (DEFERRED to last in phase)

`ggml/src/ggml-cuda/mmq.cu:180-228`: add `case GGML_TYPE_Q4_0_AR16:` to the `switch (type)` setting `mmq_supported = true`.

Verification: `ggml_cuda_should_use_mmq(GGML_TYPE_Q4_0_AR16, 750, 16)` returns true. Build passes.

**Ordering note (2026-05-15, in-flight finding)**: A.1 enables MMQ dispatch for AR16. The dispatcher at `mmq.cu:36` switches `src0->type` to call `mul_mat_q_case<TYPE>(...)`. If A.1 is committed before A.8 (the dispatch case) and a user runs with `LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH=1` and AR16 weights, the switch hits its default branch and aborts.

To avoid this intermediate-state regression, A.1 is deferred to be the LAST mutation within Phase A. Steps land in order A.2 → A.3 → A.4 → A.5 → A.6 → A.7 → A.8 → A.9, then A.1 flips the gate, then A.10 binds.

### A.2 — Add Q4_0_AR16 cases to layout tables

`mmq.cuh:50-area` `mmq_get_q8_1_ds_layout`: case Q4_0_AR16 → `MMQ_Q8_1_DS_LAYOUT_DS4`.

`mmq.cuh:179-area`: add `MMQ_DP4A_TXS_Q4_0_AR16` macro (per §2.1).

`mmq.cuh:190-236` `mmq_get_dp4a_tile_x_sizes`: case Q4_0_AR16 → `MMQ_DP4A_TXS_Q4_0_AR16`.

`mmq.cuh:238-242`: add `MMQ_MMA_TILE_X_K_Q4_0_AR16` macro (per §2.1). Static_assert `% 8 == 4`.

`mmq.cuh:270-area` `mmq_get_mma_tile_x_k`: case Q4_0_AR16 → `MMQ_MMA_TILE_X_K_Q4_0_AR16`.

Verification: build passes.

### A.3 — Implement `load_tiles_q4_0_ar16<mmq_y, nwarps, need_check>` (per §2.5 unified layout)

Output storage (BOTH DP4A and MMA paths):
- x_qs: 64 ints per row, slot `kbx*4 + s` (s=0..3) holds 4 sign-recentered int8 weights at K positions `[4s, 4s+1, 4s+2, 4s+3]` of block `kbx`. Linear K per int.
- x_df: 16 floats per row, one per AR16 block.

Source-byte unpacking math (per lane, with `kqsx ∈ {0, 1}`):
```cpp
// Each lane reads ONE int (4 bytes = 8 raw AR16 K's via even/odd nibbles).
const int qs = get_int_b2(bxi->qs, kqsx);  // qs[kqsx*4..kqsx*4+3]
// AR16 byte i has K=2i (low) and K=2i+1 (high).
const int qs_evens = qs & 0x0F0F0F0F;          // bytes: [K=2*0, K=2*1, K=2*2, K=2*3] = [K=0..6 even]
const int qs_odds  = (qs >> 4) & 0x0F0F0F0F;   // bytes: [K=2*0+1, ...] = [K=1..7 odd]
// Interleave even/odd → linear K=0..7 packed in 2 ints, 4 K's per int.
const int lin_low  = __byte_perm(qs_evens, qs_odds, 0x5140);  // [K=0, K=1, K=2, K=3]
const int lin_high = __byte_perm(qs_evens, qs_odds, 0x7362);  // [K=4, K=5, K=6, K=7]
// Sign-recenter (each byte: 0..15 → -8..7).
const int lin_low_s  = __vsubss4(lin_low,  0x08080808);
const int lin_high_s = __vsubss4(lin_high, 0x08080808);
// Store linearly: kqsx=0 → slots {0, 1}; kqsx=1 → slots {2, 3}.
x_qs[i*MMQ_MMA_TILE_X_K_Q4_0_AR16 + kbx*4 + kqsx*2 + 0] = lin_low_s;   // K = kqsx*8 + {0..3}
x_qs[i*MMQ_MMA_TILE_X_K_Q4_0_AR16 + kbx*4 + kqsx*2 + 1] = lin_high_s;  // K = kqsx*8 + {4..7}
```

For DP4A path (no INT8_MMA): same per-row layout but using `MMQ_DP4A_TXS_Q4_0_AR16.qs / mmq_y` stride (= `2*WARP_SIZE + 1`).

Verification: `tests/dflash-speculative/test-mmq-q4-0-ar16-load-tiles.cu`:
- Generate random `block_q4_0_ar16` rows.
- Dispatch kernel; dump x_qs and x_df to global memory.
- CPU oracle: dequantize via `dequantize_block_q4_0_ar16` math (low=K=2i, high=K=2i+1), regroup to linear K per int, sign-recenter. **Both kernel output and oracle MUST match byte-for-byte** across (mmq_y, nwarps, need_check) sweep.

### A.4 — Implement `vec_dot_q4_0_ar16_q8_1_dp4a<mmq_x, mmq_y, nwarps>` (per §2.5)

With unified linear-K storage, the impl is a trivial dp4a accumulator (no nibble extraction, no -8 correction):

```cpp
template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q4_0_ar16_q8_1_dp4a(
    const int * x, const int * y, float * sum, const int & k00)
{
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_0_AR16, mmq_y);
    constexpr int x_qs_stride = 2*WARP_SIZE + 1;     // ints/row for unified layout
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;
    // 16 iters cover one block_q8_1_mmq (128 K's). Step VDR=4 (one AR16 block per iter).
    for (int k01 = 0; k01 < 2*WARP_SIZE; k01 += 4) {
        const int k0 = k00*2 + k01;  // k0 indexes x_qs in linear K-int units (64 per row).
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;
                int sumi = 0;
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    sumi = ggml_cuda_dp4a(
                        x_qs[i*x_qs_stride + k0 + l],
                        y_qs[j*MMQ_TILE_Y_K + k01/2 + l],  // y_qs uses K-stride 4 per int
                        sumi);
                }
                const float d_w = x_df[i*16 + (i+15)/16 + (k0/4)];
                const float d_a = __low2float(y_ds[j*MMQ_TILE_Y_K + (k01/2)/QI8_1]);
                sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += d_w * d_a * (float)sumi;
            }
        }
    }
}
```

(Final indices subject to tuning; the structure is the SoTA: per-block scale, no per-call -8 correction.)

Verification: `tests/dflash-speculative/test-mmq-q4-0-ar16-dp4a.cu`:
- Random Q4_0_AR16 weight `[N_rows, K_cols]` and F32 activation `[K_cols, M_cols]`.
- Quantize activation via `quantize_mmq_q8_1_cuda` with DS4 layout (after A.7 lands, OR a manual Q8_1 packer in the test driver).
- Kernel produces output rows.
- CPU oracle: dequantize Q4_0_AR16 + dequantize Q8_1 + fp32 dot product.
- Cosine ≥ 0.9999, NMSE ≤ 1e-4 across sweep over (mmq_x, mmq_y, K_cols).

### A.5 — Implement `vec_dot_q4_0_ar16_q8_1_mma<mmq_x, mmq_y, nwarps>` (per §2.5)

With unified linear-K storage, the MMA path's structure mirrors `vec_dot_q8_0_q8_1_mma<...,DS4>` (mmq.cuh:987). The differences from Q8_0:
- Scale stride: 16 floats per row (vs Q8_0's 8), since AR16 has 16 blocks/warp_tile vs Q8_0's 8.
- Scale lookup: `x_df[i*MMQ_MMA_TILE_X_K_Q4_0_AR16 + k0/QI_AR16_LINEAR]` where `QI_AR16_LINEAR = 4` (ints per scale).
- Sign-recentering already applied at load → no per-call correction.

Verification: `tests/dflash-speculative/test-mmq-q4-0-ar16-mma.cu`:
- Same setup as A.4.
- MMA kernel output byte-identical to DP4A kernel output AND within cos ≥ 0.9999 / NMSE ≤ 1e-4 of scalar fp32 reference.

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

## 6b. Phase CX — Non-MMQ / non-cuBLAS shape-dependent op audit + FATTN deep audit

Inserted 2026-05-15 after the Phase C closure-binding test (`scripts/test-production-np-determinism.sh`) revealed that 10/14 slots still diverge at NP > 1 EVEN with strict-sequential off, cont-batching on, single OR multi GPU. Phase A/B/C made every MMQ + cuBLAS path byte-identical across batch size M, but the production token-level harness still fails — so the residual non-determinism lives in **the remaining (non-MMQ, non-cuBLAS) CUDA ops** AND/OR **scheduler-level state effects**. Phase CX chases all of these.

Single-GPU probe (`/tmp/single-gpu-np-probe`) confirms: even with `--device CUDA0` only, the divergence is identical in shape (per-slot agree/disagree pattern). Multi-GPU (Phase D) is NOT the primary cause — it can only AMPLIFY whatever's already broken. So Phase CX comes BEFORE Phase D.

**Slot-pattern shifting across runs** (NP=4 went 0/4→3/4→2/4 across iterations with the same code) strongly indicates RUN-TO-RUN non-determinism, not deterministic shape-dependence. The spec at `specs/deltanet/fattn-per-slot-kv-sm75.md §15.17` documents prior identification of this as "**CUDA-side timing / kernel-internal non-determinism (atomic order, scheduler interleaving, possibly cuda graph cache eviction)**". §15.16 cb_eval probe established that all 60+ model layers are byte-identical when processing is single-prompt — so the broken contract is purely at concurrent-batching boundaries.

### CX.A — FATTN per-slot-kv at ne[1] > 1 — CLOSED (test bug, kernel was already correct)

**RETRACTION 2026-05-16.** The CX.A diagnosis was based on a broken test, not a broken kernel.

Original signal (2026-05-15): `test-fattn-per-slot-kv-ncols-invariance.cpp` reported 5888/6144 row-0 floats differing across `n_tok ∈ {1, 2, 4, 8}` with max |Δ| ≈ 9.6e-2. We attributed this to fp16 `frag_c_VKQ` inter-row contamination per spec §15.13, designed and implemented a fp32-VKQ promotion in `fattn-wmma-f16.cuh` (gated `ncols ≤ 16` to fit 48 KiB static-SMEM on sm_75).

**Empirical falsification** (2026-05-16 — Opus 4.7 session): Sentinel-bias debugging confirmed the fp32-VKQ branch was executing as designed, yet diff count was unchanged (5888/6144) and the bit-pattern of which cells differed was identical pre/post-fix. Forcing Q tile rows 1..7 to zero inside the kernel (regardless of `ne01`) ALSO left the diff count unchanged — meaning Q rows >0 weren't even consumed by the result the test was reading.

**Root cause**: the test extracted row-0 with the wrong tensor stride. `ggml_flash_attn_ext_per_slot_kv` produces output shape `[Dv, N_HEADS_Q, n_tok, N_SEQS]` (per `ggml.c:10284`: `int64_t ne[4] = { v->ne[0], q->ne[2], q->ne[1], q->ne[3] }`). The test computed `full_idx = d + h*Dv*n_tok + s*Dv*n_tok*N_HEADS_Q` (head stride `Dv*n_tok`), which assumes `[Dv, n_tok, N_HEADS_Q, N_SEQS]`. For `n_tok=1` the wrong and correct strides coincide; for `n_tok>1` the test reads memory cells that the kernel populated with DIFFERENT head's data. That's why exactly 1 head's data (head 0, where `h*stride == 0` regardless of stride) always matched and 23 heads always "diverged" — pure layout misread, not kernel non-determinism.

**Verification 2026-05-16**: with the test corrected (`full_idx = d + h*Dv + s*Dv*N_HEADS_Q*n_tok`), the stock kernel (no fp32 VKQ promotion, exactly upstream code) is **byte-identical across `n_tok ∈ {1, 2, 4, 8}`**. The fp32 VKQ promotion was reverted (working tree clean against HEAD `f40e3ee2` for `fattn-wmma-f16.cuh`).

**Implications**:
- Spec `fattn-per-slot-kv-sm75.md §15.13` is FALSIFIED. The wmma_f16-pb1<256,256,8,float> route does NOT have inter-row fp16 rounding contamination. The fp32 `KQ_acc_t` softmax + fp16 `frag_c_VKQ` configuration is already NP-invariant for batched decode.
- The residual 10/14 production NP-determinism gap is NOT in fattn-per-slot-kv. Hunt continues elsewhere: cb_eval residual capture (l_out_F16 path), RMSNorm batch sweep (CX.B), RoPE batch sweep (CX.C), or scheduler/atomic-order effects.
- Lesson reinforced (per `feedback_verify_test_mechanism_before_trusting`): before designing a fix from a test signal, verify the test mechanism reports what it claims to.

**Test side-effect**: fixed the row-0 extraction stride in `tests/dflash-speculative/test-fattn-per-slot-kv-ncols-invariance.cpp` (commits separately). The test now binds correctly on per-slot-kv FA NP-invariance and is GREEN against stock upstream code.

### CX.D — FATTN per-slot-kv ROOT CAUSE found via TRACE-1..6; FIX-C v4 lands (IN PROGRESS)

**2026-05-16.** Even with CX.A retracted, production NP-determinism harness still showed slot-position-dependent output. A four-trace dive (`data/trace-{1,2,3,6}-2026-05-16/findings.md`) localized the real bug:

- **TRACE-1** captured per-slot per-layer residuals at NP ∈ {1, 2, 4, 8} with same-prompt across all slots. Found a SLOT-PARITY pattern: even-indexed slots (0, 2, 4, 6) byte-identical to slot 0; odd-indexed slots (1, 3, 5, 7) all diverge from slot 0 starting at **layer 3** (the first full-attention layer per the d2 finding's `fa_layer_indices = [3, 7, 11, ..., 63]`). Max |Δ| = 9.5 at NP={2,4}, 2.1 at NP=8. Full-magnitude drift, not fp32-ε noise.
- **TRACE-2** drilled into layer 3: all FA inputs (`Qcur`, `Kcur`, `Qcur_hadamard`, `Kcur_hadamard`, `Vcur_hadamard`) are byte-identical between slot 0 and slot 1; the divergence enters at `flash_attn_per_slot_kv-1003`/`-2003` (the FA op output itself). Max |Δ| at FA output = 8.4e-04 (per device).
- **TRACE-3** verified the Q4_0 K and V cache content is byte-identical at slot 0's vs slot 1's regions of the global-packed cache (same prompt → byte-identical K/V projections + Hadamard). So CPY+quantize is innocent.
- **TRACE-4/5/7** initially hypothesised `warp_reduce_sum` mask-shape non-associativity, but the focused CUDA unit test (`tests/dflash-speculative/test-trace-4-warp-reduce-mask-shape.cu`) DISPROVED this: warp XOR-shuffle is commutative-stable on the same nonzero-value set at different lane positions. fp32 commutativity holds for the shuffle pattern. Hypothesis falsified.
- **TRACE-6** added captures of every cb-tagged intermediate around the layer-3 FA block. Confirmed FA op output is the first divergent stage. The corrected diagnosis: the **WMMA k-loop's 16-K-chunk decomposition of the V × softmax(KQ) matmul** distributes each row's nonzero softmax × V products into different chunks based on the row's mask shape. WMMA matrix-A (V) is shared across matrix-B's 8 cols (slots); fp32 accumulation across chunks is non-associative. Same algebraic total, different fp32 partial-sum decomposition per slot. Structural to WMMA cross-row K-sharing.

**Web research dive** (`RESEARCH_2026-05-16.md`, sources include [TML 2025-09 "Defeating Nondeterminism"](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/), [SGLang deterministic inference](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/), [llama.cpp PR #16016](https://github.com/ggml-org/llama.cpp/pull/16016) draft, [ssiu/flash-attention-turing](https://github.com/ssiu/flash-attention-turing), [Turing Tuning Guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html)) confirms the field-standard recipe is **per-row CTA + canonical k-loop + online streaming softmax + fp32 accumulator + no Split-K**.

**LOCAL DISCOVERY**: `ggml/src/ggml-cuda/fattn-vec-f32.cuh:17` `flash_attn_vec_ext_f32<Dk=256, Dv=256, ncols=1, F16, F16>` is exactly that architecture and is already compiled for sm_75. Per-row CTA when ncols=1 (always on NVIDIA per the dispatcher at line 377). F32 throughout. Online Welford softmax (line 257). Mask-bounded K-loop. No Split-K when launch_fattn's `parallel_blocks=1`.

**FIX-C v4** (chosen): in `ggml_cuda_flash_attn_ext_per_slot_kv_sm75` (`fattn-per-slot-kv-sm75.cu`), change the dispatcher from
```cpp
ggml_cuda_flash_attn_ext_wmma_f16_case_pb1<256, 256, 8, float>(ctx, dst);
```
to
```cpp
ggml_cuda_flash_attn_ext_vec_f32_case<256, 256, GGML_TYPE_F16, GGML_TYPE_F16>(ctx, dst);
```
`launch_fattn` (with `need_f16_K=true, need_f16_V=true`) pre-dequants Q4_0 K/V to F16 before the kernel. The per-row CTA vec_f32 kernel then iterates each row's K in canonical order with online softmax in fp32. Same algebraic answer regardless of slot position. Byte-identical across slots for same-prompt input.

**Closure binding (V1-V6)** per `RESEARCH_2026-05-16.md` §9d:
- V1: scalar fp32 oracle (`fattn-vec-f32-reference.h`) + unit test → kernel byte-identical to oracle.
- V2: TRACE-6 re-run with new dispatcher → `flash_attn_per_slot_kv-*` slot 0 ≡ slot 1.
- V3: TRACE-1 re-run at NP={1,2,4,8} → all slots byte-identical at every layer.
- V4: production NP-determinism harness → 14/14 byte-identical.
- V5: NP=1 regression (token sequence preserved within tight tolerance).
- V6: perf measurement → quantify vec_f32 vs wmma_f16 end-to-end decode delta.

**Spec impact**: `specs/deltanet/fattn-per-slot-kv-sm75.md` §15.13 retracted in CX.A; §15.18 (to be written) documents FIX-C v4 as the production path. §15.7's KQ_acc_t=float fix is no longer applied because we're switching to vec_f32 entirely.

**Open empirical questions**:
1. Real perf delta of vec_f32 vs wmma_f16 at our shape. Spec §15.13 cited "~12× slower" but that measurement is unverified for our binary + production shape. V6 measures.
2. Does `launch_fattn` for vec_f32 default `parallel_blocks=1`? Must verify; the per-row CTA only delivers determinism with no Split-K.
3. The Q4_0 dequant kernel itself — is it batch-shape-invariant? Likely yes (per-cell transformation), but worth a probe.

### CX.1 — DeltaNet shape-conditional dispatch (DONE)

`ggml/src/ggml-cuda/delta-net.cu`, lines 219 and 228:

```cpp
if (n_tokens <= 8) {
    constexpr int threads_per_block = 256;   // 8 warps
} else {
    constexpr int threads_per_block = 128;   // 4 warps
}
```

The DeltaNet recurrent kernel's per-step `reduce_sum<block_size>(...)` produces a different cross-warp reduction order depending on the warp count (8 vs 4). The cross-warp SMEM tree at line 144-147 of delta-net.cu is `for (i = 0; i < block_size/WARP_SIZE; ++i) sum += all_sumK[i*WARP_SIZE_S + row]`, summing `block_size/WARP_SIZE` partial sums — 8 vs 4 → different fp32 rounding outcomes.

Under continuous batching, the same slot's DeltaNet state path can flip between the two dispatch arms across calls when prompt-eval (n_tokens=12 per sequence) and decode (n_tokens=1) interleave. The recurrent state baked into the per-CTA register accumulator `state_local[i]` then drifts.

**Fix**: pin a single `threads_per_block` under `LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH=1`, regardless of `n_tokens`. Recommendation: 256 (matches the n_tokens≤8 path which is the production decode shape). Validate that the n_tokens>8 prompt-eval shape still works with 256 — adjust SMEM budget if needed.

**Verification**:
- Unit test `tests/dflash-speculative/test-delta-net-shape-invariance.cu`: random inputs at (n_tokens ∈ {1, 4, 8, 12, 24, 48}, n_seqs ∈ {1, 2, 4, 8}), bit-compare per-(slot, head) output and final state across n_tokens values FOR FIXED slot inputs. PASS = byte-identical for the slot-0 path regardless of total `n_tokens`.
- Production NP harness slot-pass count strictly INCREASES vs the 4/14 baseline.

### CX.2 — FATTN-per-slot under cont-batching audit

The `fattn-per-slot-kv-sm75` kernel was designed for NP-invariance per-slot (per `project_fattn_per_slot_kv_p2_landed_kernel_only`). Verify that the production code path still hits the kernel with `parallel_blocks=1` and `cols_per_block` fixed regardless of ubatch composition. Look for any dispatcher branch that picks a different FATTN variant based on `ne[1]` (token count) of the input.

**Verification**: probe-only — run production NP harness with a `LLAMA_FATTN_PER_SLOT_KV_DEBUG=1` env that logs each FATTN dispatch's kernel selection + tile geometry. Confirm same kernel + same tile dims for all calls.

### CX.3 — `rms_norm_f32` block_size dispatch audit

`ggml/src/ggml-cuda/norm.cu:440-450`:

```cpp
static void rms_norm_f32_cuda(... int ncols ...) {
    if (ncols < 1024) {
        rms_norm_f32<256><<<...>>>(...);
    } else {
        rms_norm_f32<1024><<<...>>>(...);
    }
}
```

(Pseudocode — verify exact dispatch.) `block_size` switches based on `ncols` (hidden_dim), which is FIXED for a given model. So this isn't batch-shape-dependent for production. **No fix needed here** — but the audit confirms it.

If `fused_rms_norm_f32` has its own dispatch on `n_tokens` or batch, fix per CX.1 pattern.

### CX.4 — RoPE batch-invariance audit

`ggml/src/ggml-cuda/rope.cu`: verify per-token RoPE is independent of total batch. Audit kernel launch geometry.

### CX.5 — Element-wise op audit (GLU / SiLU / add / mul)

Element-wise ops have no cross-position interaction. Should be batch-trivial. Confirm by grep.

### CX.6 — `ggml_set_rows` / KV cache write per-slot audit

KV cache writes use per-slot pointers but the writing kernel may have batch-dependent geometry. Audit.

### CX.7 — Closure binding for Phase CX

`scripts/test-production-np-determinism.sh` PASS at NP ∈ {1, 2, 4, 8}, 5 runs stable, on **single-GPU** (Phase D unlocks multi-GPU).

If still FAIL after CX.1-CX.6: instrument per-op output capture (cb_eval callback at every named tensor) and bisect to find the first op whose output diverges between NP=1 and NP=2 slot 1.

---

## 6c. Phase CY — Cross-NP build-graph determinism (layer-0 fp32-ULP divergence)

**Inserted 2026-05-16 after FIX-C v5 (singlewarp) closed the intra-NP FA gap.** Per `FIX_C_V5_FINAL_REPORT.md` and `data/fixc-v5-prod-harness-2026-05-16/findings.md`, the cross-NP harness still shows 12/14 slots diverging from NP=1 baseline — not because of FA (FIX-C v5 makes FA byte-deterministic), but because the build graph has batch-shape-dependent conditionals that pick different ops for different `cur->ne[1]` values.

### CY.A — Audit batch-shape conditionals in src/llama-build-context.cpp

Active conditionals identified (2026-05-16 grep):

| Line | Conditional | Active? | What it gates |
|---|---|---|---|
| 789 | `cur->ne[1] > 32` | YES | cast residual to reduce_type (fp16) for prefill (>32 tokens) |
| 1375 | `gate->ne[1] == 1` | YES (only on MoE path) | MoE shexp gate: ne[1]==1 uses `ggml_fused_mul_unary`; else split sigmoid+mul |
| 1387 | `shared_out->ne[1] > 32` | YES (only on MoE path) | cast to reduce_type — same as 789 |
| 1407 | `shared_gate->ne[1] == 1` | YES (only on MoE path) | similar fused vs split — for shared-expert path |
| 1491 | `gate->ne[1] == 1` | YES (only on MoE path) | similar fused vs split |
| 1505 | `cur->ne[1] > 32` | YES | cast to reduce_type — same as 789 |
| 2779 | `if (false && cur->ne[1] == 1)` | NO (disabled) | placeholder for future fused op |
| 2805 | `cur->ne[1] > 32` | YES | cast to reduce_type — same as 789 |
| 2902 | `if (false && cur->ne[1] == 1)` | NO (disabled) | placeholder |

**RETRACTED MoE shared-expert hypothesis (2026-05-16)**: production GGUF inspection of `qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf` shows `general.architecture = qwen35` (not `qwen35moe`). No `*_exps`, `*_shexp`, or `gate_inp` tensors. The model uses `build_qwen35()` (`src/graphs/build_qwen35.cpp:160`), not `build_qwen35moe()`. The MoE shared-expert gate fusion at lines 1375/1407/1491 NEVER fires for Qwen 3.6 27B. The MoE hypothesis is dead.

**Two classes of batch-shape-dependent behaviour that DO apply to `build_qwen35()`**:

1. **fp16 cast cluster (lines 789, 1505, 2805)**: `if (ne[1] > 32 && reduce_type != F32) cast to reduce_type`. Active only for prefill (>32 tokens). For decode at NP={1,2,4,8} (all ≤8), this DOESN'T trigger. Excluded as cause.

2. **DeltaNet (recurrent) at layer 0** + **FFN (line 776-790 in `llm_build_ffn`)**: layer 0 of Qwen 3.6 is a DeltaNet/linear-attention layer (`hparams.is_recurrent(0)==true`). Inside `delta_net::build_layer_attn_linear` and `llm_build_ffn`, there are MMQ/cuBLAS dispatches where the algorithm picked depends on `ne[1]`. Even with Phase A MMQ lockdown and Phase C cuBLAS pinning, sub-paths inside DeltaNet may not be covered.

### CY.B — Layer-localized capture: where does the fp32-ULP gap first appear?

Empirical (`data/deltanet/d1-capture` from 2026-05-16): NP=1 vs NP=4 at slot 0 in `l_out-0` (layer 0 residual output) differ at max|Δ| = **1.118e-07** (single fp32 ULP at exponent 0), with 5119/5120 cells differing. Amplification through 64 layers: layer 1 → 1.047e-03, layer 2 → 6.194e-02, etc. By final layer the logit gap flips argmax → different token sequences.

**Subprobes to run (CY.B.1 .. CY.B.4)**:

- **CY.B.1**: capture `inp_embd` (post embedding lookup, pre layer 0) at NP=1 vs NP=4. Expectation: byte-identical (embedding is index-lookup, no math).
- **CY.B.2**: capture intra-layer-0 tags: `norm-0` (pre-DeltaNet RMSNorm output), `ssm_in-0`, `ssm_out-0` (DeltaNet kernel internals if tagged), `ffn_norm-0` output, `ffn_up-0`, `ffn_gate-0`, `ffn_silu-0`, `ffn_down-0`. First tag with non-zero diff identifies the source op.
- **CY.B.3**: if DeltaNet is the source: re-examine `ggml/src/ggml-cuda/delta-net.cu` for batch-shape-dependent reductions or kernel dispatch.
- **CY.B.4**: if FFN is the source: examine the mul_mat call for `ffn_up` / `ffn_gate` / `ffn_down` at ne[1]=1 vs ne[1]=4. MMQ vs MMV vs cuBLAS dispatch may differ.

### CY.C — Fix candidates grounded in performance impact AND data-provability

Each fix is evaluated for: (a) likelihood it's the actual source, (b) **decode-rate cost** at NP=1, (c) **provability with data** — what binding test demonstrates both source identification AND fix closure, and (d) does the test require the full 27B production model or runs in unit-test fixtures.

| # | Candidate source | Likelihood | Decode-rate cost @ NP=1 | Binding test for source ID | Binding test for fix closure | Fixture vs full-model |
|---|---|---|---|---|---|---|
| 1 | **DeltaNet kernel batch-shape dispatch** (layer 0 is recurrent) | HIGH (layer 0 is DeltaNet; CX.1 pinned threads_per_block but not all internal reductions) | 0-3% (DeltaNet single-token specialization is minor; bandwidth-bound) | Intra-layer-0 cb_eval capture: tag `delta_net_out-0` differs at NP=1 vs NP=4 with max\|Δ\|≈1e-7 → confirms source | New `tests/test-delta-net-np-invariance.cpp` driving `delta_net::build_layer_attn_linear` at n_tokens ∈ {1,2,4,8} with identical-content slots → sha256 match all-slots-all-shapes | Both: unit test for kernel; full-model only for end-to-end binding |
| 2a | **MMVQ vs MMQ dispatch** for ffn_up/ffn_gate/ffn_down (Option: force MMQ at ne[1]=1) | HIGH — TML/SGLang identify as canonical NP-invariance breaker | **30-50% UNACCEPTABLE** | `test-mul-mat-q-vs-vec-q-byte-identity.cpp`: random Q4_0/Q4_0_AR16 weights × random fp32 input row; drive both MMQ (ne[1]=2 with dummy) and MMVQ (ne[1]=1); diff outputs. Single 50-line test, no model needed. | Same test after fix → bytes match | Fixture only — fastest closure |
| 2b | **MMVQ rewrite to match MMQ reduction order** | HIGH | **0%** target | Same as 2a | Same as 2a after rewrite | Fixture |
| 2c | **MMQ shimmed for ne[1]=1 with dummy row** | HIGH | **10-30%** | Same as 2a | Same as 2a after shim | Fixture |
| 3 | RMSNorm batch-shape dispatch | LOW — CX.B already excluded via test-rmsnorm-batch-shape-invariance | n/a | n/a | n/a | n/a |
| 4 | Token embedding lookup (`inp_embd`) | VERY LOW | n/a | Single-line test: get `inp_embd` for same token at NP=1 vs NP=4, diff. Same input → same output trivially. | n/a | Fixture |
| 5 | Elementwise SiLU / mul at different ne[1] | LOW | 0% if needed | Unit-test elementwise op at varying ne[1] | Same test after pin | Fixture |

**Most likely candidates by ULP signature** (1.118e-07 = single fp32 ULP at exp 0, affecting 5119/5120 cells):

- **Candidate #2** (MMVQ vs MMQ) is BOTH most likely AND most provable. The unit-test binding is ~50 LOC, runs in <1s, and conclusively answers "do MMVQ and MMQ produce byte-identical output for the same row?" One test, no model load, no NP harness.
- **Candidate #1** (DeltaNet) is harder to bind because the kernel is custom and there's no scalar reference yet. But the d1-capture data already shows layer-0 divergence at the residual level — if we capture intra-layer-0 tags and DeltaNet output is byte-identical (meaning the divergence is in the FFN AFTER DeltaNet), it falsifies #1 and confirms #2.

**Provability-first resolution path** (next subprobe order):

1. **CY.B.0** — write `test-mul-mat-q-vs-vec-q-byte-identity.cpp` first. ~50 LOC. If FAIL (bytes differ): candidate #2 confirmed, no need for further model-level capture. Move to fix 2b.
2. **CY.B.1** — only if CY.B.0 PASSES (bytes match → MMVQ and MMQ ARE byte-identical, candidate #2 falsified): then capture intra-layer-0 tags `norm`, `delta_net_out`, `ffn_norm`, `ffn_up`, `ffn_up_gate`, `ffn_down`, `ffn_combined` at NP=1 vs NP=4 to find the first divergent intra-layer-0 op.

This ordering inverts the typical "trace first, fix second" approach by leveraging that the MMVQ-vs-MMQ question has a much cheaper binding test (unit test) than the DeltaNet question (intra-kernel capture).

### CY.D — Closure binding for Phase CY

- Production NP-determinism harness: 14/14 byte-identical at NP ∈ {1, 2, 4, 8}.
- `scripts/test-production-np-determinism.sh` PASS with `LLAMA_PSKV_MODE=singlewarp` (FIX-C v5) + CY fix applied.
- Multi-run stability: 5 runs of the harness, all PASS at 14/14.

### CY.E — Reference path comparison

Verify that the CY fix doesn't break existing tests (`test-fattn-*`, `test-rmsnorm-*`, `test-rope-*`). All should still pass.

---

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
| CX.1 | test-delta-net-shape-invariance.cu | per-(slot, head) output byte-identical for slot-0 inputs at n_tokens ∈ {1,4,8,12,24,48} |
| CX.7 | test-production-np-determinism.sh (single-GPU) | NP ∈ {1,2,4,8} byte-identical, 5 runs stable |
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
| ggml-cuda/mmvq.cu | `mul_mat_vec_q_q4_0_ar16_q8_1_cuda` + dispatch case | YES |
| ggml-cuda.cu | env-gated `cublasSetMathMode` + fixed algo | YES |
| ggml-cuda/mul-mat-f16-pinned.{cu,cuh} | F16 + F32 row-pinned GEMM | YES |
| ggml-cuda/delta-net.cu | env-gated single threads_per_block | NO |
| ggml-cuda.cu | env-gated peer-access serialization | NO |
| data/mmq-q4-0-ar16-{dp4a,mma}.* | profile data | NO |

All transitions must complete for Phase F closure.
