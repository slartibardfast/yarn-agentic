# Production-graph cross-NP measurement (2026-05-16)

## Setup

V4 production harness extended with cross-NP slot-0 comparison matrix. No
`cb_eval` instrumentation — uses raw llama-server completion output. This is the
production-faithful graph (full in-place optimization, graph_reuse=1).

Env stack:
```
LLAMA_FATTN_PER_SLOT_KV_ENABLE=1
LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH=1
LLAMA_PSKV_MODE=singlewarp
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

13-token prompt, 32 decode steps, NP_LIST=`{1, 2, 4, 8}`.

## Cross-NP slot-0 comparison matrix

```
np1 vs np2 slot 0: BYTE-IDENTICAL
np1 vs np4 slot 0: BYTE-IDENTICAL
np1 vs np8 slot 0: BYTE-IDENTICAL
np2 vs np4 slot 0: BYTE-IDENTICAL
np2 vs np8 slot 0: BYTE-IDENTICAL
np4 vs np8 slot 0: BYTE-IDENTICAL
```

**ALL slot-0 outputs at NP ∈ {1, 2, 4, 8} are byte-identical to each other.** Cross-NP byte-determinism for slot 0 is ACHIEVED in production with current env stack.

## The d1-capture trace was misleading

Earlier d1-capture (with `cb_eval` instrumentation marking 64 `l_out-<il>` tags as outputs) reported:
- NP=1 vs NP=4 first diverges at layer 6 (max|Δ|=3.815e-06)

But production says: **NP=1 slot 0 == NP=4 slot 0 byte-identical**. The "layer 6 cross-NP gap" was a **cb_eval instrumentation artifact** — the act of marking residuals as outputs disabled in-place optimizations and **changed the actual computation**.

Same caveat applies to the d1-capture "NP=2/4 vs NP=8 first diverges at layer 20" finding — production says NP=4 slot 0 == NP=8 slot 0 byte-identical.

**The entire cy-trace-view layer-by-layer table is suspect** as a guide to production behavior. The cb_eval-marked graph != production graph.

## The real open problem: intra-NP=8 slot-position

| Slot pair (NP=8) | Result |
|---|---|
| 0 vs 1 | DIFFERS |
| 0 vs N for N ∈ {2..7} | DIFFERS |
| Any pair in {1..7} × {1..7} | BYTE-IDENTICAL |

**Slot 0 is the odd one out.** Slots 1-7 form one cluster, byte-identical to each other AND to slot 0 of NP={1,2,4}. Slot 0 of NP=8 takes a different code path.

Output diffs:
- slot 0 (NP=8): "Alan Turing, who in 1950 published a paper titled..."  (canonical, matches all NP=1/2/4/8 slot 0)
- slot 1 (NP=8): "Alan Turing, who in 1950 published "Computing Machinery..."  (matches slots 2-7 at NP=8)

Note: at NP=4 and NP=2 ALL slots are byte-identical. The regression is **specifically at NP=8 for slot 0 vs slots 1-7**.

## Hypotheses for NP=8 slot-0-vs-slots-1+ effect

1. **First-block-in-batch special case**: the slow path's per-block dispatch may handle block 0 differently than blocks 1+ (e.g., a one-time initialization that doesn't fire for later blocks). At NP=4 this works correctly, but at NP=8 some divergence accumulates.

2. **FA grid scheduling at n_tokens=8**: singlewarp Grid (n_tokens, n_heads_q, n_seqs) has 8 X-blocks at NP=8. The first X-block (slot 0) may launch earlier or read freshly-initialized state, while later X-blocks read shared state from earlier blocks.

3. **KV cache contiguity at n_seq_max=8**: slot 0's cache region starts at offset 0; slots 1+ start at later offsets. If addressing has an off-by-one effect or aliasing for slot index > 0, slots 1+ would all share the bug (consistent with empirical pattern).

4. **Concurrent ubatch processing**: maybe at NP=8 a sub-batch boundary divides slot 0 from slots 1-7 (e.g., n_ubatch=512 splits 8 tokens into 1+7 for some reason).

## What this means for production

- **NP ∈ {1, 2, 4} is fully byte-deterministic in production** with singlewarp + shape_invariant + cublas_pin. Ship-ready for these slot counts.
- **NP=8 has an intra-NP slot-position issue affecting slots 1-7**. Slot 0 of NP=8 matches NP=1; slots 1-7 produce a consistent but different output. The cross-NP slot-0 picture is clean, just the intra-NP within NP=8.
- **The trace view at the layer level is unreliable** — cb_eval distorts the graph. Future probes should use uninstrumented measurement (token output, nsys timeline, ncu kernel-level profiling).

## Pivot for next probe

Drop layer-level cb_eval captures. Use token-output measurement (V4) + nsys per-kernel timing as the authoritative cross-NP probe stack. To find the NP=8 intra-NP source, the cleanest probe is:
- Add per-slot greedy-argmax instrumentation **inside the production server** (no cb_eval; emit token IDs directly).
- Bisect the slot-0-vs-slot-1 divergence by progressively replacing slot 1's computation with slot 0's at each layer.
- Or: use ncu to find a kernel that takes different code paths based on slot index.
