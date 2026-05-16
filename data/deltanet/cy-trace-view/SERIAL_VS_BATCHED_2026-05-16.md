# Phase CY.F.9 + CY.F.9b — slot-id hypothesis FALSIFIED, multi-seq batched is the actual source (2026-05-16)

## Test design

Two complementary fixtures with the same n_seq_max=8 context:
- **Serial** (CY.F.9): prefill each seq_id ∈ {0..7} in a SEPARATE `llama_decode` call, capture per-seq logits. One seq active per decode.
- **Batched** (CY.F.9b): prefill all 8 seqs (8 × 12 = 96 tokens) in ONE `llama_decode` call with interleaved seq_ids. Capture per-seq last-token logits.

Then compare batched[seq=K] vs serial[seq=K] bit-wise.

## Results

### CY.F.9 serial — PASS

```
[CY.F.9] seq=0..7  argmax=24480  logits[0..3]=4.246263 5.334656 0.502013 1.247608  (identical for all 8)
[CY.F.9] seq=1..7: BYTE-IDENTICAL to seq=0 (n_vocab=248320)
[CY.F.9] PASS — all seq_ids bit-identical in serial path
```

256K floats × 8 seqs all matching bit-for-bit. **NO slot-id > 0 addressing bug in the serial path.**

### CY.F.9b batched — FAIL

```
[CY.F.9b] seq=0..7: BATCHED != SERIAL — 248077/248320 logits differ, max|Δ|=2.521e-01 at idx 96675.
argmax: serial=24480  batched=24480  (matches, but barely)
```

ALL 8 seqs in batched produce identical-to-each-other logits but DIFFERENT from serial baseline. Magnitude 0.25 — large. argmax of the very next token happens to match but probability mass is materially redistributed (~99.9% of vocab differs).

### CY.F.9b batched + LLAMA_DELTA_FORCE_SLOW=1 — STILL FAIL

Same magnitude divergence (max|Δ|=0.252). FORCE_SLOW makes serial also take the slow path (per-block split) — but that doesn't close the batched-vs-serial gap. So the bifurcation is NOT just about `all_same_seq` fast/slow path.

## Hypothesis triangulation

**FALSIFIED**: per-slot addressing bug (KV stride, DeltaNet state offset, Hadamard rotation, ssm_conv t==0 special case) — CY.F.10..13 fixtures obsoleted.

**CONFIRMED**: batched multi-seq build differs from serial single-seq build, even when the seq_id-0 row of the batched build is being compared to a standalone seq_id-0 serial run.

The bug lives in **how the graph handles `n_seqs > 1` at the build/kernel level**, NOT in how it addresses any specific seq slot.

## Where to look (CY.F.14)

Multi-seq specific code paths that differ from single-seq:

1. **ssm_conv multi-seq fast path** (`ssm_conv_multi_seq_unique_f32_kernel`, ssm-conv.cu:361) — fires only when each token has a unique seq. Has internal logic that doesn't fire at single-seq.
2. **FA per-slot-kv at n_seqs > 1** — singlewarp kernel grid Z dim. CTAs per seq.
3. **Per-slot KV cache write/read interleaving** — at multi-seq batched, multiple slot regions are written in interleaved fashion.
4. **`build_layer_attn_linear` slow path with > 1 block** — even with FORCE_SLOW serial, serial = 1 block, batched = 8 blocks. Per-block kernel call is the same, but the surrounding graph (concat, reduce) differs.
5. **Cross-device reduce at n_seqs > 1** — `ggml_reduce(OP_ADD, n_device)` at multi-seq vs single-seq. The per-device tensors might have different shapes.
6. **inp_out_ids gather** at multi-seq — at serial, only the last token's logit is needed (n_tokens=12, gather idx=11). At batched, 8 last-token logits are gathered (one per seq).

## Next probe (CY.F.14)

Capture per-layer residual outputs (using the d1-capture extract-layers mechanism with dtype branching) at:
- Serial: seq=0 alone, 12 tokens
- Batched: 8 seqs × 12 tokens, extract seq=0's row of each layer's residual

Find first layer where serial[seq=0] vs batched[seq=0, row=0] differ. This isolates which sub-system introduces the multi-seq drift.

If layer 0 already differs — the qkv projection or DeltaNet kernel handles n_seqs > 1 incorrectly.
If layer N differs — narrow to that layer's specific op.

## What's confirmed for production

- **NP=1, 2, 3, 4 fully byte-deterministic** — production-ready at these slot counts.
- **NP≥5 has multi-seq batched processing bug** — the multi-seq batched build is computationally inequivalent to per-seq processing.
- **Per-slot addressing is correct** — no need to audit KV stride or DeltaNet state offsets.

## Production rollout consequence

Either:
1. **Ship at NP≤4** (current production guarantee).
2. **Fix the multi-seq batched build** to be bit-equivalent to per-seq processing. This is a much narrower fix than chasing per-slot addressing.

If at-NP=4 is acceptable for production (current canonical assumption), this is shippable as-is. The NP≥5 path remains a known limitation, scoped to multi-seq batched-graph determinism (a different problem from the original cross-NP determinism work).
