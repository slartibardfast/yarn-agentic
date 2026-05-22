# Variant D — WMMA tensor-core PSKV FA kernel design

**Target:** Replace `flash_attn_per_slot_kv_singlewarp_kernel` with a
WMMA-based single-kernel design preserving byte-identity across
NP={1,2,4,8} multi-GPU. Target per-call latency ≤ 150 µs at the
NP=8 decode shape (vs current 597 µs). Single kernel = no scratch =
sidesteps the graph-capture+pool RAII race characterized in iters 2-9.

## Hardware target

- TU102 (sm_75), 2× Quadro RTX 6000.
- Tensor cores: m16n16k16 fp16-input fp32-accumulator via nvcuda::wmma.
- Theoretical TC peak FP16 → FP32: ~32 TFLOPS per RTX 6000.
- Current singlewarp uses scalar FP32 — ~3 TFLOPS sustained.

## Determinism contract — required justification

WMMA mma.sync is **deterministic on fixed inputs**: same fragment in
→ same fragment out, bit-by-bit. This holds AS LONG AS:

1. **Input fragments are byte-identical across slots.** K/V/Q
   dequant from Q4_0 to fp16 must produce identical fp16 bytes for
   the same Q4_0 input bytes. Q4_0 dequant is deterministic
   (`d_scale * (q_nib - 8)` cast to fp16) so this holds.

2. **MMA call sequence is identical across slots.** Each slot's CTA
   processes the same (M_tile_count × K_tile_count) sequence of MMA
   calls in the same order. The per-slot CTA layout is identical
   (one CTA per (token, head, seq)) so this holds by construction.

3. **fp32 accumulator.** Avoids fp16 catastrophic cancellation in
   the inner K-dim reduction. `nvcuda::wmma::fragment<accumulator,
   16, 16, 16, float>` provides this.

4. **No cross-CTA reduction.** Each per-slot CTA computes its own
   complete softmax output → no scratch → no inter-CTA data flow.

**Determinism risk:** zero relative to the existing singlewarp kernel
PROVIDED the dequant + tile-sequence invariants hold. Validate with
the standard NPC smoke + full multi-GPU harness.

## Algorithm

For each per-slot CTA at (tok, head, seq):

### Step 1: Q dequant + scale (one-time per CTA)

Load Q[d] for d in [0, Dk=256), apply scale, hold in registers as fp16.
With 4 warps × 32 threads = 128 threads, each thread loads
Dk / 128 = 2 fp16 elements of Q.

### Step 2: K-row tile loop (outer, canonical order)

```
for k_base in [0, ne11) step 16:
    // ne11 is per-slot K cache row count
    // Process 16 K rows per tile
    
    Step 2a: K dequant Q4_0 → fp16 (cooperatively, into SMEM K_tile[16][256])
    Step 2b: WMMA chain to compute scores[16] = K_tile @ Q
    Step 2c: Read 16 scores from accumulator fragment
    Step 2d: Per-row softmax serially in canonical [k_base..k_base+16) order
    Step 2e: V dequant Q4_0 → fp16 (cooperatively, into SMEM V_tile[16][256])
    Step 2f: V accumulation into per-thread fp32 VKQ[Dv/128 = 2]
```

### Step 3: Output write

Normalize VKQ by kqsum and write to dst[seq, tok, head, :].

## WMMA-specific design

### Fragment layout

```cpp
using namespace nvcuda::wmma;

// A: K_tile (16 K rows × 16 head_dim columns)
fragment<matrix_a, 16, 16, 16, half, row_major>     a_frag;

// B: Q broadcast (16 head_dim rows × 16 cols, only col 0 has Q data)
fragment<matrix_b, 16, 16, 16, half, col_major>     b_frag;

// C: accumulator (16 K rows × 16 cols, only col 0 holds scores)
fragment<accumulator, 16, 16, 16, float>            c_frag;
```

### MMA call chain for one 16-K-row tile

```cpp
fill_fragment(c_frag, 0.0f);

for (int d_base = 0; d_base < 256; d_base += 16) {
    load_matrix_sync(a_frag, K_tile + d_base, /*ldm=*/256);  // 16x16 K slice
    load_matrix_sync(b_frag, Q_fp16 + d_base, /*ldm=*/16);   // Q[d_base..d_base+16]
    mma_sync(c_frag, a_frag, b_frag, c_frag);
}

// c_frag now holds K_tile @ Q.T as 16 scores in col 0
// (cols 1..15 hold padding/garbage, ignored)
```

### Extracting scores

WMMA fragment storage is opaque per-thread. To extract column 0 only,
store the full fragment to SMEM and read column 0:

```cpp
__shared__ float scores_smem[16];
__shared__ float c_full[16][16];

store_matrix_sync(&c_full[0][0], c_frag, /*ldm=*/16, mem_row_major);
__syncthreads();

if (warp_id == 0 && lane < 16) {
    scores_smem[lane] = c_full[lane][0];
}
__syncthreads();
```

### Per-row softmax (serial canonical order)

Same Welford logic as singlewarp, but applied to 16 scores at a time:

```cpp
for (int i = 0; i < 16; ++i) {
    const int k = k_base + i;
    if (k >= ne11) break;  // tail
    
    const float kq = scores_smem[i] + slope * __half2float(maskh[k]);
    const float new_max  = fmaxf(kqmax, kq);
    const float diff_old = kqmax - new_max;
    const float diff_cur = kq    - new_max;
    const float scale_corr = diff_old >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_old) : 0.0f;
    const float phi        = diff_cur >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_cur) : 0.0f;
    kqmax = new_max;
    kqsum = kqsum * scale_corr + phi;
    
    // Scale existing VKQ
    #pragma unroll
    for (int j = 0; j < V_PER_THREAD; ++j) VKQ[j] *= scale_corr;
    
    // Add phi * V[k] from V_tile
    #pragma unroll
    for (int j = 0; j < V_PER_THREAD; ++j) {
        const int d = tid + j*BLOCK;
        VKQ[j] += phi * V_tile_fp32[i][d];  // V_tile is per-row-tile
    }
}
```

### V_tile staging

V also needs to be staged. Either:
- Cooperative dequant V_tile[16][256] (16 K rows × 256 head_dim) per outer step. SMEM: 16KB. Within sm_75's 64KB SMEM budget.
- OR: dequant V row-by-row inside the inner softmax loop (no SMEM staging).

For simplicity start with row-by-row inside softmax loop. SMEM only
needs K_tile + scores extraction = ~9 KB.

## Resource budget per CTA

| Resource | Estimate |
|---|---|
| Block size | 128 threads (4 warps) |
| Registers/thread | ~64 (Q_reg + VKQ + softmax state + WMMA fragments) |
| SMEM | K_tile 8 KB + scores 64 B + temp 256 B ≈ 9 KB |
| Per-SM limits (TU102) | 64 KB SMEM, 65536 regs, 32 warps |
| Blocks/SM at 64 regs × 128 thr = 8 KB regs | 8 blocks (reg-limit) |
| Blocks/SM at 9 KB SMEM | 7 blocks (SMEM-limit) |
| Block_limit_warps = 32/4 | 8 blocks |
| **Theoretical occupancy** | 7 × 4 = 28 warps/SM = 87.5% |

## Expected performance

- Current singlewarp: 597 µs/call at NP=8 decode, ne11=256, 16% achieved occupancy.
- WMMA mma.sync m16n16k16 fp16→fp32 on TU102: 16 cycles per fragment-MMA.
- Compute: 16 (Dk tiles) × 16 (K-tiles per ne11=256) × 16 (cycles) = 4096 cycles per warp.
- 4 warps × 4096 = 16384 warp-cycles per CTA.
- At ~1.7 GHz: ~10 µs of pure compute per CTA.
- Memory: K reads 256 × 16 = 4 KB per CTA per outer tile. L2-cached.
- Net per-call estimate: 50-150 µs (4-12× speedup).

If achieved: TG @ NP=8 jumps from 27.10 to ~32-35 t/s, clearing the ≥33.0 target.

## Implementation plan

### Phase 1: minimal MMA scaffold (build + correctness)
- Just replace the K·Q dot computation. V scalar.
- Validate vs singlewarp output (single-GPU, single batch).
- ~150 lines new code.

### Phase 2: add V_tile + softmax integration
- Stage V cooperatively.
- Validate output matches Phase 1.
- ~250 lines total.

### Phase 3: gates
- Smoke NPC (single-GPU NP=1 vs NP=8).
- Bench at NP=8.
- Full multi-GPU NPC.

### Phase 4 (if positive): ncu profile to confirm TC utilization
- `ncu --section ComputeWorkloadAnalysis` to verify TC throughput.
- Target ≥30% TC utilization.

## Risks

1. **MMA on sm_75 has different fragment layout than sm_80+.** The PSKV
   spec template comment mentioned m16n8k8 (sm_80+ syntax). sm_75
   uses m16n16k16 only. Reference: NVIDIA PTX ISA docs.
2. **Q broadcast pattern requires careful B_frag layout.** Q is 1-row;
   B_frag wants 16-col matrix. Padding cols 1..15 with zero must not
   affect output bits — should be fine since fp16 0.0 × anything = 0.0.
3. **fp16 dequant rounding.** Q4_0 → fp16 via __float2half_rn. Same
   rounding mode across slots → deterministic.
4. **First-time WMMA in this kernel family.** No template to copy from
   — fattn-vec-common.cuh's other kernels are scalar/MMVQ, not WMMA.
   Look at fattn-new-mma.cu and fattn-wmma-f16.cuh for reference.

## Implementation prerequisites for next iter

Before writing code:
- Read `fattn-wmma-f16.cuh` for the existing WMMA pattern in this codebase
- Read `fattn-new-mma.cu` if present for any sm_75-specific WMMA tricks
- Confirm nvcuda::wmma headers are available and the build system links them
- Confirm sm_75 supports m16n16k16 with fp16 input + fp32 accumulator

## Done when

- Build PASS
- Single-GPU NPC smoke PASS (byte-identical slots at NP=8 vs NP=1)
- Bench TG @ NP=8 ≥ 33.0 t/s
- Full multi-GPU NPC PASS
- ncu confirms WMMA throughput

## Yield note

This document is the iter 10 deliverable. The kernel rewrite is iter 11+
work. Variant D requires ≥3-4 iters of careful implementation +
debug + measurement; not a one-shot kernel edit. The current ralph
loop's per-iter budget (~5-10 min build+test cycle) makes any one iter
high-risk for a WMMA attempt. The structured design above lets a future
session (or next iter) make incremental progress against a fixed plan.
