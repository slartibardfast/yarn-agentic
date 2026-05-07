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

#### Path A: Slot-allocator extension (chain rollout > 1)

Smaller refactor; lifts effective accept by ~accept^N at modest extra overhead:

1. **Plumb chain depth into cparams**: add `cparams.mtp_chain_extra_cells = max(0, LLAMA_MTP_ROLLOUT - 1)` read at init. Only set when `cparams.mtp && hparams.nextn_predict_layers > 0`.
2. **Extend find_slot reservation**: `cells_to_reserve = n_tokens + cparams.mtp_chain_extra_cells`. Bump `cache.head` and set `cache.n` to cover all reserved cells.
3. **Per-iter kv_head_offset in build_mtp_head_qwen35**: pass `kv_head_offset = k` to `build_std_attention` so iter k writes K/V at cell `kv_head + k`.
4. **Per-iter KQ_mask view**: each chain iter's attention reads cells `[0..kv_head + k]`. The shared `KQ_mask` (built for verify's single Q position at `kv_head + 0`) needs an EXTENDED variant covering `kv_head + k` for k = 0..rollout-1. Build the extended mask in `build_inp_KQ_mask` when `cparams.mtp_chain_extra_cells > 0`, then index into it per-iter.
5. **Confirm n_kv covers reserved**: `kv_self.n` at verify decode time should be `kv_head + n_tokens + chain_extra`. Adjust the bookkeeping in find_slot.
6. **Test**: re-run harness with `LLAMA_MTP_ROLLOUT=3`. Expect accept × rollout effective (e.g., 39% × 3 → ~1.17 effective drafts/cycle).

#### Path B: Tree drafting (top-K branching)

1. **Scaffold tree-aware ggml ops**: `ggml_top_k` exists. Need to confirm or add.
2. **Build tree in graph**: extend `build_mtp_head_qwen35` to top-K + branch.
3. **Extract tree**: extend post-compute extraction in `llama_decode_internal`.
4. **Read tree drafts**: extend `common_mtp_read_drafts` to return structured tree.
5. **Tree batch**: extend `mtp_speculative_gen_draft` → `common_speculative_state_mtp::draft` → server tree-batch construction.
6. **Tree-aware mask**: server hooks into KQ_mask building for the tree batch.
7. **Tree accept**: server walks tree to pick longest matching prefix.
8. **Harness measure**: re-run `--fast` and `--slow` with tree drafting on.

### Next-iteration priority

Path A is the smaller-blast-radius win. With current measured 39.44% per-iter
accept, chain rollout=3 gives expected effective drafts/cycle:
1 - (1 - 0.394)^3 = 0.78 → average 1 + 0.394 + 0.394² + 0.394³ ≈ 1.55 drafts.
Compute overhead: chain adds ~3× the MTP head cost (per iter), but the
amortization across multiple drafts per cycle should net positive.

If Path A works empirically (positive uplift over no-MTP baseline of 33 t/s),
tree drafting becomes a stretch optimization on top, not a blocker.

### Path A implementation depth (revised after deep dive)

The chain rollout's cache semantics require care across TWO decodes:

**Decode 1 (chain-generating)**: positions [head+0..head+chain] reserved.
Layer 0..63 K/V at head+0 only (single verify). Layer 64 K/V at
head+0..head+chain (chain rollout writes per-iter). cells[].pos =
[P+1, P+2, ..., P+chain+1] all committed to the same seq_id.

**Decode 2 (verify)**: batch [draft_P+1, draft_P+2, ..., draft_P+chain].
positions match. Layer 0..63 process the batch normally, writing K/V
at cells head+0..head+chain. Layer 64's K/V at those cells was
already written by the previous chain — gets OVERWRITTEN by this
decode's chain. That overwrite is fine because we're verifying the
same positions.

**Reject handling**: walk drafts vs. argmax(verify). Find longest
matching prefix. seq_rm positions past the prefix.

Concretely, the implementation steps:

1. **Plumb cparams.mtp_chain_extra_cells** from `LLAMA_MTP_ROLLOUT - 1`
   when MTP + nextn weights are present.
2. **Extend find_slot**: reserve `n_tokens + chain_extra` cells. Assign
   pos = base_pos + i and seq_id to all reserved cells.
3. **Extend KQ_mask**: build_inp_KQ_mask creates an [n_kv, chain] sized
   mask (chain Q rows). Row k covers cells [0..head+k].
4. **Multi-Q attention in MTP head**: stack chain iter Q tensors into a
   [n_embd_head, n_head, chain] tensor. Single attention call processes
   all chain Q rows in parallel with the per-row mask. K/V written at
   cells head+0..head+chain.
5. **Per-iter K/V offsets in build_std_attention**: pass `kv_head_offset`
   so iter k writes at head+k. (Already supported via existing
   parameter.)
6. **MTP head's chain rollout**: produces N mtp_logits (one per iter)
   stacked. Output is [vocab, chain].
7. **Speculative engine**: chain → drafts list (length chain). Server
   submits batch [draft_1..draft_chain] for verify. Walk + accept like
   standard speculative decoding.

The crucial novel piece: step 4's multi-Q attention. ggml's
flash_attn_ext supports multi-Q (n_q > 1). The mask is [n_kv, n_q];
each Q row's column mask defines visible K cells. Building this mask
correctly is the central design point.

### What blocks the simpler approaches

- "Skip K/V write for iter > 0": ggml's attention mixes cache K/V with
  freshly-built K/V via the kv_store path. There's no "compute attn
  with provided K/V tensors only, no cache" variant in build_std_attention.
  Adding one is a few-day refactor of llm_build_kv.
- "Sequential chain via N graph builds": ggml's graph is built once
  per decode. Building N times per decode = N graph allocations.
  Slow + fragments scheduler state.
- "Accept top-K drafts in flat batch (no tree mask)": the existing
  speculative engine uses linear positions; can't represent K
  alternatives at the same logical position.

The multi-Q attention is the cleanest design even though it requires
tree-aware mask construction.

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
