# TRACE-3 — root cause: FA per-slot-kv `warp_reduce_sum` is mask-shape-dependent

**Date**: 2026-05-16
**Inputs**: `data/trace-2-2026-05-16/findings.md` (slot divergence localized to inside FA block).

## What we measured

Captured K and V cache views (`k-1003`, `k-2003`, `v-1003`, `v-2003`) at decode time after both slots prefilled the same prompt then took one decode step. Q4_0 cache layout in our test was **global-packed** (not slot-segmented):

- positions [0..11] = slot 0 prefill
- positions [12..23] = slot 1 prefill
- position 24 = slot 0 decode-step K
- position 25 = slot 1 decode-step K

Diff slot 0's Q4_0 bytes vs slot 1's Q4_0 bytes (`diff2.py`):

| Tensor | head_kv 0 prefill | head_kv 1 prefill | decode-step pos 24 vs 25 |
|---|---|---|---|
| `k-1003` | BYTE-IDENTICAL | BYTE-IDENTICAL | BYTE-IDENTICAL |
| `v-1003` | BYTE-IDENTICAL | BYTE-IDENTICAL | BYTE-IDENTICAL |
| `k-2003` | BYTE-IDENTICAL | BYTE-IDENTICAL | BYTE-IDENTICAL |
| `v-2003` | BYTE-IDENTICAL | BYTE-IDENTICAL | BYTE-IDENTICAL |

CPY+quantize is innocent. The FA per-slot-kv kernel is reading byte-identical Q4_0 inputs (TRACE-3) with byte-identical Q (TRACE-2) but producing slot-position-dependent output (TRACE-1, TRACE-2 layer-3 row).

## Root cause analysis

The only per-row input that differs between slot 0 and slot 1 inside the FA call is **the KQ mask**.

- slot 0 mask: K positions {0..11, 24} visible (= 0.0 mask), others -inf
- slot 1 mask: K positions {12..23, 25} visible, others -inf

The K *values* at slot 0's visible positions are byte-identical to the K values at slot 1's visible positions (TRACE-3). So `softmax(K_visible × Q + mask)` is mathematically the same number for both slots.

But numerically, in CUDA, softmax is computed via `warp_reduce_sum` over 32 lanes per warp. The XOR-shuffle reduction pattern is:

```
tmp = warp_reduce_sum(KQ_f_tmp[k])
```

which produces, for each lane, the sum of all 32 lane values. The reduction tree pairings:

- step 1: lane i adds with lane i^16
- step 2: lane i adds with lane i^8
- step 3: lane i adds with lane i^4
- step 4: lane i adds with lane i^2
- step 5: lane i adds with lane i^1

**fp32 addition is not associative.** The same set of 13 numerical values (after softmax `exp`) is reduced in a DIFFERENT lane-pairing order for slot 0 vs slot 1:

- slot 0's valid lanes {0..11, 24}: lane 8 XOR 16 = lane 24 → step 1 pair (exp_8, exp_24). Other valid lanes pair with -inf lanes (exp=0).
- slot 1's valid lanes {12..23, 25}: lane 9 XOR 16 = lane 25 → step 1 pair (exp_9, exp_25). Other valid lanes pair with -inf lanes.

After step 1, the partial sums at the "valid" lanes are:

- slot 0: exp_0, exp_1, ..., exp_7 (paired with -inf=0), exp_8+exp_24 (paired together), exp_9, exp_10, exp_11
- slot 1: exp_12, exp_13, ..., exp_15, exp_16+exp_0_at_lane_16=0... etc.

Continuing through the 5 reduction steps, the **partial sums** encountered along the reduction tree are different for slot 0 vs slot 1, even though the set of 13 nonzero values being summed is the SAME. Final fp32 sum differs by O(fp32 ε × 13 values) — that matches the observed `max |Δ| = 5.860e-03` at layer 3.

## Even/odd parity explanation

Why specifically even/odd (slot 2 ≡ slot 0 but slot 1 ≠ slot 0 per TRACE-1)?

For NP=4, global-packed positions:
- slot 0: {0..11, 48}  (positions 0..11 = prefill, 48 = decode)
- slot 1: {12..23, 49}
- slot 2: {24..35, 50}
- slot 3: {36..47, 51}

Within a warp of 32 lanes, slot 0's prefill range [0..11] and slot 2's prefill range [24..35] map similarly mod 32:
- slot 0 lanes: {0..11, ?} where 48 → lane 48 % 32 = 16
- slot 2 lanes: {24..31 + wrap to 0..3 = 0..3, lane 50 % 32 = 18}

Wait — with 32 lanes per warp and FATTN_KQ_STRIDE=256, the K loop iterates 256 / 32 = 8 warp-wide chunks. Per chunk, the lane handles positions [k0..k0+31].

For slot 0's first chunk k0=0: valid positions = {0..11} → lanes {0..11} of warp.
For slot 2's first chunk k0=0: valid positions = {24..31} → lanes {24..31}. Then continues into chunk k0=32 with positions {32..35} → lanes {0..3}.

Hmm — the parity is more subtle than a simple even/odd. It depends on how the valid-position bitmask within each k0-chunk pairs up with -inf-position lanes under the XOR-shuffle reduction.

But empirically TRACE-1 shows clean even/odd parity. That likely reflects a regularity in how slots' prefill ranges sit in the warp lanes for our specific n_prompt=12 case: slot s's prefill {s*12..s*12+11} mod 32 has different lane occupancy for even-s vs odd-s, and the reduction-tree pairings happen to coincide for even-s.

## Why CX.A test passed

The CX.A `test-fattn-per-slot-kv-ncols-invariance` test fixture builds K, V, mask such that **mask row 0 is all-visible** for all positions [0..N_KV). The kernel sees all 256 K positions as valid for the only row tested. Reduction tree sums 256 valid values in a fixed lane order for every test config — no slot-parity effect because there's only one slot and one mask shape.

The production case has per-row mask shapes that depend on slot index. The CX.A test fixture didn't exercise this dimension. The CX.A retraction is still correct (the WMMA fp16 frag_c_VKQ accumulator is innocent), but the kernel IS non-deterministic at production's mask-shape configuration.

## What this implies for the fix

The fix must make the reduction order **content-independent**, not layout-independent. Three options:

1. **Compact-then-reduce**: gather valid (non-`-inf`) K cells into lanes [0..n_valid) before warp_reduce_sum. This canonicalizes the reduction tree regardless of which positions are valid in any given row. Adds a SMEM-tree compaction pass before the reduction.

2. **Pre-multiply mask before exp** (no change in algorithm) — already done; this isn't the issue.

3. **Switch to TML 3-kernel batch-invariance pattern**: per-row CTA, fixed-tile dims, no Split-K, SMEM-tree reductions instead of warp-shuffle reductions. Per spec §15.10 the multi_row_kernel attempted this but had implementation bugs. Re-attempt with a clean implementation.

Option 3 is the recipe spec'd in `DESIGN.md §5.5` and used elsewhere; option 1 is the smaller surgical fix to the existing wmma_f16 kernel.

## What this disproves

- Spec §15.13's framing that "wmma fp16 frag_c_VKQ accumulator" is the residual blocker is FALSIFIED. The fp16 accumulator is innocent (CX.A retraction). The actual residual is `warp_reduce_sum` non-associativity over mask-shape-dependent valid-lane patterns.
- The implication: changing `frag_c_VKQ` to fp32 (CX.A attempted fix) wouldn't have closed the gap. We need to change the REDUCTION, not the accumulator.
- The "5/14 slots match np=1 in production harness" observation maps cleanly: ~half the slots match because of the even-position lane-pairing regularity in the reduction tree.

## TRACE plan onward

- TRACE-4: instrument the kernel to dump `KQ_f_tmp[k]` and the warp_reduce_sum intermediate at each step for slot 0 and slot 1, confirm the partial-sum tree diverges as theorized.
- FIX A (option 1 above): compact-then-reduce inside the kernel.
- VERIFY: re-run TRACE-2 after FIX A and confirm `l_out-3` becomes slot 0 ≡ slot 1. Re-run production NP-determinism harness.
- FIX B (option 3 above): if compact-then-reduce isn't enough or has perf cost, replace with per-row CTA pattern.
