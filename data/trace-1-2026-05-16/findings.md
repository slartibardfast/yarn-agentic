# TRACE-1 — concurrent NP same-prompt per-slot per-layer residual capture

**Date**: 2026-05-16
**Hardware**: 2× Quadro RTX 6000 (sm_75), CUDA 13.2
**Binary**: `ik_llama.cpp/build/bin/test-deltanet-d1-capture` (built post-CX.A retraction)
**Model**: `/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf`
**Env**: `LLAMA_FATTN_PER_SLOT_KV_ENABLE=1`
**Cache**: q4_0 K, q4_0 V, Hadamard rotation on both
**Prompt**: same content across all slots (md5 `be7695391b1b61f26217a495e5a8ba64`)
**Setup**: prefill each slot sequentially with distinct seq_id; then ONE decode step at N tokens (one per slot, all SAME prompt)

## Method

For each NP ∈ {1, 2, 4, 8}: configure cb_eval for all 64 layers; capture `l_out-<il>` per slot per layer as raw float32 binaries (5120 floats each); diff:

- **Analysis A** — intra-NP slot uniformity: at fixed NP, diff slot s vs slot 0 at each layer.
- **Analysis B** — cross-NP slot-0 invariance: diff NP=1's slot-0 vs NP={2,4,8}'s slot-0.

## Result

### Analysis A — intra-NP slot uniformity (smoking gun)

Same prompt across all slots → if kernels are slot-position invariant, all slots should be byte-identical at every layer.

| NP | slot 1 vs slot 0 | slot 2 vs slot 0 | slot 3 vs slot 0 | slot 4–7 vs slot 0 |
|---|---|---|---|---|
| 2 | first div @ layer 3, max \|Δ\|=9.525 @ layer 63 | — | — | — |
| 4 | first div @ layer 3, max \|Δ\|=9.525 @ layer 63 | BYTE-IDENTICAL | first div @ layer 3, max \|Δ\|=9.525 @ layer 63 | — |
| 8 | first div @ layer 3, max \|Δ\|=2.090 @ layer 63 | BYTE-IDENTICAL | first div @ layer 3, max \|Δ\|=2.090 @ layer 63 | slot 4,6 BYTE-IDENTICAL; slot 5,7 first div @ layer 3, max \|Δ\|=2.107 @ layer 63 |

**Pattern**: even-indexed slots (0, 2, 4, 6) are byte-identical to slot 0; odd-indexed slots (1, 3, 5, 7) diverge from slot 0.
**First divergent layer**: 3 (the first full-attention layer per `data/deltanet/d2-first-divergent-layer.json` `fa_layer_indices = [3, 7, 11, ..., 63]`).
**Magnitude**: ≈ 9.5 at NP={2,4} vs ≈ 2.1 at NP=8 — divergence is real-magnitude, not fp32-epsilon.

### Analysis B — cross-NP slot-0 invariance

| Pair | First divergent layer | Max \|Δ\| | Layer of max |
|---|---|---|---|
| NP=1 slot 0 vs NP=2 slot 0 | 0 | 5.432 | 63 |
| NP=1 slot 0 vs NP=4 slot 0 | 0 | 5.432 | 63 |
| NP=1 slot 0 vs NP=8 slot 0 | 0 | 3.978 | 63 |

This is the **F32 single-token decode vs F16 multi-token storage** split that `data/deltanet/d2-first-divergent-layer.json` already identified. NP=1 batches one token and stores residuals in F32; NP>=2 batches multiple tokens and stores residuals in F16. The "divergence" at layer 0 is precision-truncation of the storage path, not a computational drift.

This is NOT the cross-slot bug. The bug is in Analysis A.

## Diagnosis

The slot-parity pattern at layer 3 says: **a kernel called at layer 3 produces output that depends on the slot index modulo 2, even when slot inputs are byte-identical**.

Layer 3 is the **first full-attention layer** in the qwen3.6 hybrid model (deltanet layers at indices 0,1,2,4,5,6,..., full-attn at 3,7,11,...,63). The kernel handling FA at this layer routes through `ggml_cuda_flash_attn_ext_per_slot_kv_sm75` → `wmma_f16_case_pb1<256, 256, 8, float>` (CX.A retraction confirmed this is the production path).

Per the CX.A retracted ncols-invariance test (fixed 2026-05-16), this kernel IS byte-identical across batch ne[1] when row 0's input is fixed. But the test uses **synthetic Q/K/V**, not real per-slot KV cache offsets.

**Working hypothesis (TRACE-2 needs to prove)**: the per-slot-kv mechanism that handles per-slot K/V offsets in the cache exposes slot-position-dependence. Either:
1. Q/K/V projection at layer 3 takes slot-position-dependent paths (unlikely — proj is shared)
2. RoPE at layer 3 uses different pos values per slot (yes — but algebraically each (token, head) pair is independent, and we test the same prompt at the same position; so equivalent inputs → equivalent outputs)
3. **The FA kernel reads per-slot KV cache with slot-offset-dependent geometry** — the `per_row_k_bound` (src[5]) plumbing or the slot index passed to FA changes the kernel's K/V memory access pattern in a way that breaks slot-position invariance.
4. **The KV cache write/dequant from Q4_0 + Hadamard rotation has slot-parity-dependent behavior** — perhaps Q4_0 block alignment changes between slots, or the Hadamard transform applies differently.

Option 3 or 4 is most likely. The "wmma_f16_case_pb1" kernel itself is row-CTA-independent per CX.A retraction, but the per-slot-kv DISPATCHER and the K/V cache READ are NOT the same code path as the FA test fixture. The FA test fixture uses synthetic fp16 K/V; production uses Q4_0 + Hadamard with per-slot offsets.

## Next: TRACE-2 plan

Localize WITHIN layer 3 by capturing intra-layer intermediates and comparing slot 0 vs slot 1 at NP=2:

1. **Q proj output** (`Qcur` post-rotate-or-not, name it `Qcur-<il>`) — if slot 0 and slot 1 differ here, the divergence is upstream of FA.
2. **K proj output** (`Kcur-<il>`) — similar.
3. **V proj output** (`Vcur-<il>`) — similar.
4. **K cache write target** (`k_cache_view-<il>`) — after Q4_0 quantization + Hadamard rotation.
5. **V cache write target** (`v_cache_view-<il>`) — same.
6. **FA output** (`kqv_out-<il>` or whatever the build graph names it).
7. **Output projection** result (already covered as `l_out-<il>`).

If 1-3 are byte-identical between slot 0 and slot 1 but 6 isn't, the FA per-slot-kv mechanism IS the problem.

If 4-5 differ, the cache write path (Q4_0/Hadamard) is the problem.

To add these intra-layer named captures: extend `src/graphs/build_qwen35.cpp` to tag the relevant intermediates with `cb(tensor, "name", il)` so cb_eval can match them. Then update test-deltanet-d1-capture to also extract these.

Alternative cheaper path: an instrumentation switch in the per-slot-kv kernel that prints first-cell-of-output for each slot at known points. Less informative than full residual capture but useful to confirm slot-parity entry point.

## Conclusion

TRACE-1 succeeded. Production NP-determinism residual is localized to:

- **Layer 3** (first full-attention layer) — and presumably propagates through layers 7, 11, 15, ..., 63 (every FA layer).
- **Slot-parity pattern** — odd-indexed slots diverge from even-indexed slots even with byte-identical inputs.
- **Magnitude** large enough (≈ 9.5 at NP={2,4}) to flip argmax tokens downstream.
- **NOT in the per-row FA kernel math** — CX.A retraction proved that; this is something the production FA call does WITH the per-slot KV cache state that the synthetic FA test fixture didn't exercise.

The "5/14 slots match np=1 in the production harness" observation (MEMORY 2026-05-15) is now coherent with the slot-parity finding: ≈ half the slots match because half of slots are even.
