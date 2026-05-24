---
name: project-t5-6-paged-write-read-end-to-end
description: production/2026-q2-next 2026-05-23 — T5.6 paged KV WRITE+READ end-to-end; identity block_table preserves attention math but NOT physical byte layout vs legacy; verify-prod-determinism still PASSes because outputs are layout-invariant
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

T5.6 landed Bundle B's WRITE+READ flip end-to-end on `production/2026-q2-next`. Submodule HEAD `0e3cc592`, parent HEAD `1ae12b9`. After T5.5 wired the PSKV singlewarp kernel's `block_table` indirection (legacy nullptr branch preserved), T5.6 activated the paged path at `kv.n_stream > 1`:

- SET_ROWS K/V view reshape: row count = `total_blocks * BLOCK_SIZE_TOKENS * n_head_kv` (4 sites: `llm_build_kv_store` × 2 + `llm_build_std_attention` × 2).
- `inp_kv_idxs` formula flipped to paged: `bid = btbl[p_t/blk_tokens]; row = bid*blk_tokens*n_head_kv + h*blk_tokens + p_in_block`.
- New `inp_block_table` input tensor I32 `[nbps, n_seqs]`, populated from `kv.paged.block_table(s)`, padded with `-1`.
- FA op gets `src[6] = block_table`; kernel derives `paged_nb12 = nb11 * BLOCK_SIZE_TOKENS`, `paged_nb13 = paged_nb12 * ne12` (and same for V), used in all three paged branches (main K-loop, V softmax, tail).
- NP=1 retains legacy contig branch (no block_table, kernel takes nullptr branch).

**Load-bearing non-obvious finding (paged ≠ legacy physical byte layout):**

The cross-NP byte-identity gate passes despite paged and legacy storing K/V cache in physically different in-memory layouts. Legacy:
`[head_dim, kvps_per_stream, n_head_kv, n_stream]`. Paged: `[head_dim, BLOCK_SIZE_TOKENS=64, n_head_kv, total_blocks]` with `total_blocks * BLOCK_SIZE_TOKENS == kvps * n_stream`. The byte sequences at corresponding seq-position addresses are NOT byte-equal across these layouts.

verify-production-determinism PASS because the gate compares **per-slot decode token outputs (logits → sampled tokens)**, which are invariant under the storage permutation when block_table is the identity (`btbl[s][i] = s*nbps + i`). The attention math reads K/V via the new row formula and gets the same VALUES it would have under the legacy formula; the bytes happen to be at different addresses. NP=1 (legacy branch) and NP>1 (paged branch) producing byte-identical slot-0 outputs is therefore a **mathematical invariance result, not a memory-layout equality result**.

**Why:** the SET_ROWS reshape + paged stride derivation are the load-bearing pieces that make this work — if the kernel had instead used `nb12 = head_dim * head_dim_bytes * n_head_kv * n_total_blocks` (i.e., read the reshaped tensor's actual strides) every paged access would walk into wrong cells. Kernel-internal derivation of `nb11 * BLOCK_SIZE_TOKENS` is what re-establishes intra-block contiguity over the flattened view.

**How to apply:** when extending paged work in T5.7 (K-shift) and T5.8 (gate sweep / closure), don't assume byte equality with the legacy layout is required or even meaningful at NP>1. The closure gate is value-equality (NPC slot-0 byte-identical to NP=1 reference), and that holds because the math is right, not because the bytes are placed identically. If a future change touches K-shift, the indirection must read+write through `block_table[s][p/64]*64 + (p%64) + h*64` — applying a pos-shift directly to `p` and addressing legacy-style would silently corrupt cells under the paged layout.

verify-prod-determinism evidence of record: `data/t5-perf-gate-ledger.md` T5.6 row, NP_LIST="1 2 4 8", CTX_CHECKPOINTS=3, 1455 MHz locked, cross-NP slot-0 matrix BYTE-IDENTICAL across all six pairs, batch-shape invariance 4/4 PASS.

Next: T5.7 — port K-shift to block_table indirection + remove CPY-fallback at single-stream legacy sites; drive `test-paged-kshift-byte-identity` GREEN. Then T5.8 — full gate sweep (M1, M3-steady, M4 high-ctx feasibility, ncu PSKV, trace validator 60s) + remove LLAMA_T5_TRACE env-gate + remove paged-nullptr fallback + PHASE doc Tier 5 closure + MEMORY entry + submodule bump.

Related: [[project-t5-bundle-a-closed]], [[project-t5-1-paged-allocator-landed-dormant]], [[project-t5-2-shadow-landed-t5-3-next]], [[project-t5-3-scope-corrected-bundle-a-close]], [[project-t5-probe-falsified-path-c-override]].
