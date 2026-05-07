# Phase 39: Tree-Style MTP Decoding (in-flight)

## Status

| Area | State |
|---|---|
| Inline MTP head (linear chain) | ✅ working — coherent text, 18% accept (FastMTP on) / 33.78% (off) |
| Tree drafting on MTP head | ⏳ designing → implementing |
| Server tree-aware speculative | ⏳ pending |

## Why tree drafting

Linear chain MTP at depth=N picks 1 candidate per depth (argmax). Tree drafting at K-fan-out produces K^N candidate paths per cycle. Even if per-step accept stays at the current ~33% (post-FastMTP-off), tree drafting gives K parallel shots per depth — effective per-cycle accept lifts substantially.

Hardware estimate (2× RTX 6000, split-graph): linear chain at 33% accept, depth 1 → 1.33 useful tokens per ~37ms cycle = ~36 t/s. Tree at K=2 depth=2 → effective 1 + (1−(1−0.33)²)×2 = ~2.1 useful tokens per ~42ms cycle (slightly more compute) = ~50 t/s = +1.5× over baseline 33 t/s.

## Architecture

### Graph builder (`build_mtp_head_qwen35`)

Replace per-iter `argmax` with `top_k`:

```cpp
// At each rollout iter k:
const int K = LLAMA_MTP_TREE_K;          // env knob, default 2
const int N = LLAMA_MTP_ROLLOUT;          // chain depth, default 1
ggml_tensor * top_k_tokens = ggml_top_k(ctx0, mtp_logits, K);  // [K, n_tokens]
```

Each of K tokens spawns a branch. At depth k, total branches = K^(k+1) (capped at MAX_TREE_NODES = 16).

Per branch:
- Distinct K/V cell at MTP layer: cell index = `kv_head + tree_depth`. (Multiple branches at same depth share a cell? — no, each gets its own KV row.)
- Independent eh_proj → attention → FFN → lm_head per branch.
- Stack output logits with parent indices for tree reconstruction.

### Output tensor

`lctx.t_mtp_logits` shape: `[mtp_n_vocab, n_tree_nodes]`.
`lctx.mtp_tree_parent[]` (new field): `[n_tree_nodes]` parent index per node (-1 = root).
`lctx.mtp_tree_depth[]`: depth of each node.

Post-compute extraction: pull all tree nodes' logits to host.

### Speculation glue

`common_mtp_read_drafts_tree(ctx_tgt, k_max)` returns `struct mtp_tree { tokens[], parent[], depth[] }`.

### Server speculative engine

Three additions in `common/speculative.cpp` + `examples/server/server-context.cpp`:

1. **Tree batch construction**: each tree node at depth d gets position = base_pos + d. Custom KQ_mask: node i attends to its ancestor chain + base committed prefix only.
2. **Tree-aware verify**: single `llama_decode` runs the tree as a batch. Mask is per-node.
3. **Accept walk**: from root, take verify-argmax at each depth, descend to matching child if any. Longest matching path = accepted prefix.

### Implementation sequence

1. **Scaffold tree-aware ggml ops**: `ggml_top_k` exists. Need to confirm or add.
2. **Build tree in graph**: extend `build_mtp_head_qwen35` to top-K + branch.
3. **Extract tree**: extend post-compute extraction in `llama_decode_internal`.
4. **Read tree drafts**: extend `common_mtp_read_drafts` to return structured tree.
5. **Tree batch**: extend `mtp_speculative_gen_draft` → `common_speculative_state_mtp::draft` → server tree-batch construction.
6. **Tree-aware mask**: server hooks into KQ_mask building for the tree batch.
7. **Tree accept**: server walks tree to pick longest matching prefix.
8. **Harness measure**: re-run `--fast` and `--slow` with tree drafting on.

## Env knobs

- `LLAMA_MTP_TREE_K=N` — branching factor per depth (default 1 = linear).
- `LLAMA_MTP_ROLLOUT=N` — chain depth (already exists, default 1).
- `LLAMA_MTP_TREE_MAX_NODES=N` — total tree node budget (default 16).
- `LLAMA_MTP_VOCAB_TRIM=N` — FastMTP top-K vocab trim (already exists, default 32768; observed dropping ~14pp accept on Qwen 3.6 — recommend bumping to 65536 or disabling for production).

## Open question: linear-chain accept gap (33.78% vs 60% literature)

Linear-chain MTP currently accepts at 33.78% (FastMTP+RHT off) on this setup, vs literature's expected 60-70% for Qwen 3.6. Plausible causes:
- Numerical drift in chain rollout's intermediate state vs. upstream's reference compute (maybe FP32-vs-FP16 precision in RoPE / attention scales).
- KV cache state at MTP layer 64 differs from what the trained head expects (split-graph fragmentation across 2× RTX 6000s).
- Subtle norm-tensor mismatch (attn_q_norm / attn_k_norm not applied where expected).

These are *additive* with tree drafting: a 33% per-step linear with 2-fan-out tree gives effective ~55%; with 3-fan-out tree gives ~70%. So tree drafting is independent of solving the linear gap; both can land.

Tracked as a separate diagnostic: load main forward and MTP head into a unit test, compare token-by-token argmax against upstream's reference.
