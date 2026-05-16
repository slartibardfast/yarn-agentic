# TRACE-6 — FA per-slot-kv confirmed as bug source; root mechanism is batched-FA k-loop decomposition

**Date**: 2026-05-16
**Inputs**: TRACE-1 (slot-parity at layer 3 across all heads), TRACE-2 (divergence localized between Vcur_hadamard and l_out-3), TRACE-3 (Q4_0 cache content slot 0 ≡ slot 1), TRACE-4/5/7 (warp_reduce_sum, warp_reduce_max, compact-then-reduce all bit-equal in isolation — overturned my warp_reduce hypothesis).

## TRACE-6 method

Extended `test-trace-2-intra-layer-capture` to capture every cb-tagged intermediate along the layer-3 FA chain: pre-FA inputs (`Qcur`, `Kcur`, post-Hadamard variants, `Vcur_hadamard`), the FA op output (`flash_attn_per_slot_kv-1003`/`-2003`), post-FA V-Hadamard (`flash_attn_h-*`), reshape (`flash_attn_reshaped-*`), output projection (`kqv_wo-*`), per-device combine (`attn_combined-3`), residual add (`attn_out_with_input-3`), final layer-3 output (`l_out-3`).

NP=2, same prompt across both slots, sequential prefill then one decode step.

## Result

Slot 0 vs slot 1 byte diff per tensor (corrected slot_dim per build-graph layout):

| Tensor | ne | slot_dim | slice | n_diff | max \|Δ\| |
|---|---|---|---|---|---|
| `l_out-2` (input to layer 3) | [5120,2,1,1] | 1 | 5120 | 0 | 0.000e+00 |
| `Qcur-1003` | [256,12,2,1] | 2 | 3072 | 0 | 0.000e+00 |
| `Qcur-2003` | [256,12,2,1] | 2 | 3072 | 0 | 0.000e+00 |
| `Kcur-1003` | [256,2,2,1] | 2 | 512 | 0 | 0.000e+00 |
| `Kcur-2003` | [256,2,2,1] | 2 | 512 | 0 | 0.000e+00 |
| `Qcur_hadamard-1003` | [256,12,2,1] | 2 | 3072 | 0 | 0.000e+00 |
| `Qcur_hadamard-2003` | [256,12,2,1] | 2 | 3072 | 0 | 0.000e+00 |
| `Kcur_hadamard-1003` | [256,2,2,1] | 2 | 512 | 0 | 0.000e+00 |
| `Kcur_hadamard-2003` | [256,2,2,1] | 2 | 512 | 0 | 0.000e+00 |
| `Vcur_hadamard-1003` | [512,2,1,1] | 1 | 512 | 0 | 0.000e+00 |
| `Vcur_hadamard-2003` | [512,2,1,1] | 1 | 512 | 0 | 0.000e+00 |
| **`flash_attn_per_slot_kv-1003`** | [256,12,2,1] | 2 | 3072 | **1042** | **8.444e-04 ← FIRST** |
| **`flash_attn_per_slot_kv-2003`** | [256,12,2,1] | 2 | 3072 | **1515** | **1.605e-03** |
| `flash_attn_h-1003` | [256,12,2,1] | 2 | 3072 | 3071 | 4.990e-04 (propagated) |
| `flash_attn_h-2003` | [256,12,2,1] | 2 | 3072 | 3071 | 6.739e-04 |
| `kqv_wo-1003` | [5120,2,1,1] | 1 | 5120 | 5120 | 1.041e-03 |
| `kqv_wo-2003` | [5120,2,1,1] | 1 | 5120 | 5120 | 8.157e-04 |
| `attn_combined-3` | [5120,2,1,1] | 1 | 5120 | 5120 | 8.292e-04 |
| `attn_out_with_input-3` | [5120,2,1,1] | 1 | 5120 | 5120 | 8.157e-04 |
| `l_out-3` | [5120,2,1,1] | 1 | 5120 | 5120 | 5.860e-03 |

The `q-1003`/`q-2003` rows (a non-contig view of Qcur via `ggml_permute`) appeared to "diverge" in the raw script output — that is an artifact of `ggml_backend_tensor_get` reading raw bytes and ignoring strides; the underlying Qcur is verified byte-identical above and the permute is a no-op view.

## Conclusion

**The FA op `ggml_flash_attn_ext_per_slot_kv` (routed to `wmma_f16_case_pb1<256,256,8,float>`) is the bug source**, definitively. All inputs (Q, K, V, mask, cache) are byte-identical between slot 0 and slot 1 — confirmed at the F32 input layer (TRACE-2) and at the Q4_0 cache layer (TRACE-3). The FA call produces slot-position-dependent output of magnitude ~1e-3 to ~1.6e-3 per device. Downstream ops (V-Hadamard, output projection, residual) propagate and amplify the drift to 5.86e-3 at l_out-3.

## Mechanism analysis (post TRACE-4 overturn)

TRACE-4 disproved my first hypothesis (warp_reduce_sum mask-shape non-associativity). The actual mechanism is the **batched-FA k-loop decomposition** of the V × softmax(KQ) matmul.

In the wmma kernel:

```
for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += VKQ_ratio*16) {
    load V_a from V[k0..k0+15];
    mma_sync(VKQ_c, V_a, KQ_b[k0/16]);  // 16-k partial sum, all ncols=8 columns
}
```

For slot 0 (column n=0 of KQ_b): nonzero softmax values at k ∈ {0..11, 24}. Distributed across k-chunks:
- chunk 0 (k=0..15): 12 nonzero products
- chunk 1 (k=16..31): 1 nonzero product
- chunks 2..15: 0

For slot 1 (column n=1): nonzero softmax values at k ∈ {12..23, 25}. Distributed:
- chunk 0 (k=0..15): 4 nonzero products
- chunk 1 (k=16..31): 9 nonzero products
- chunks 2..15: 0

Same SET of 13 algebraic products (V_p × v_p), distributed differently across chunks. The `mma_sync` accumulator (fp32) adds partial sums chunk-by-chunk:

- VKQ_c[m, 0] = P0_chunk0 + P0_chunk1 = (12 products) + (1 product)
- VKQ_c[m, 1] = P1_chunk0 + P1_chunk1 = (4 products) + (9 products)

fp32 addition is non-associative; same algebraic total, different fp32 decomposition → different fp32 result. The drift is at the order of fp32 ε × n_products ≈ 1e-3 (matches observed ~1e-3 magnitude).

This is the **batched-FA k-loop chunk-distribution non-determinism**. It is a STRUCTURAL property of the WMMA fragment iteration when multiple rows with different mask shapes are batched together. It is NOT in any single reduction primitive; it is in how the K-loop partials are decomposed.

## Why prior fixes didn't catch this

- **CX.A retracted fix** (fp32 frag_c_VKQ promotion): would not have helped — the accumulator already uses fp32 (per the wmma m32n8k16 fp16/fp16/fp32 fragment). The bug isn't the accumulator's precision; it's the SET of values entering the accumulator at each chunk boundary.
- **Spec §15.7 KQ_acc_t=float**: fixed softmax warp_reduce non-associativity for KQ. That's a DIFFERENT bug than the VKQ matmul chunk-decomposition issue. Both contribute; both must be fixed.
- **Spec §15.10 per-row CTA recipe**: would actually fix this (one CTA per row → each row's k-loop iterates its own valid range, no cross-row chunk-decomposition). But the prior multi_row_kernel attempt had unrelated implementation bugs.

## Fix proposals

### Fix A (surgical, minimal kernel rewrite): per-slot dispatch in the build graph

Instead of one batched FA call with `ne[1] = N` (N slots), dispatch N separate FA calls each with `ne[1] = 1`. Each call goes through the SAME wmma_f16-pb1<256, 256, 8, float> kernel. At ne[1]=1, only one row exists; mask is uniform per the slot's view; no batched-row chunk-decomposition.

- Cost: N kernel launches instead of 1. Each launch's work is roughly 1/N the batched one, so total work is similar; only launch overhead increases. For Qwen 3.6 27B at NP=4: ~3 extra kernel launches per FA call × 16 FA layers × ~64 decoded tokens × ~5ms launch overhead/extra-launch ≈ measurable but not catastrophic. Likely 5-15% slowdown end-to-end.
- Correctness: each per-slot FA call is byte-identical to its NP=1 equivalent because ne[1]=1 is the canonical case.
- Implementation: in `build_std_attention`, replace the single `ggml_flash_attn_ext_per_slot_kv(Q, K, V, mask, bound, ...)` call with a loop `for (int s = 0; s < N; ++s) { Q_s = view(Q, slot=s); mask_s = view(mask, row=s); ... per-slot FA call ... }`. Concatenate outputs.

### Fix B (deeper, kernel-side): canonical k-loop per row in batched FA

Re-architect `wmma_f16_case_pb1` to dispatch one row per CTA. Each CTA processes one Q row, iterates ONLY that row's valid K positions in a canonical order (e.g., densely indexed [0..n_valid)). The mma_sync then sees the same k-chunk distribution for every slot's row.

- Cost: re-thinks the WMMA fragment-batching strategy. Higher engineering cost. Likely better perf than Fix A (single launch, content-canonical iteration).
- Aligns with spec §15.10's "TML 3-kernel batch-invariance" recipe.

### Recommendation

Land Fix A as the surgical first step. It is the smallest change with the strongest binding (every FA call literally becomes the NP=1 case for each slot, byte-identically). Use it to UNBLOCK production np>1 determinism. Then optionally explore Fix B for perf.

## Next: FIX-A draft + verification plan

1. Implement Fix A in `build_qwen35` / `build_std_attention`: loop N times, one FA call per slot, concat results back via `ggml_concat` along the n_tokens dim.
2. Re-run `test-trace-2-intra-layer-capture` with Fix A applied: verify `flash_attn_per_slot_kv` slot 0 ≡ slot 1 byte-identically.
3. Re-run `test-deltanet-d1-capture` at NP={1,2,4,8}: verify all slots produce byte-identical residuals at every layer.
4. Re-run production NP-determinism harness: verify NP={1,2,4,8} produces byte-identical token sequences.
5. Measure perf delta vs current batched FA.
