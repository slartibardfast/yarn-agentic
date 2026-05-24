---
name: tier2-diagnostic-findings
description: "Tier 2 FA-probe diagnostic — mask/bound are correct under bailout-dropped, K/V cache contents diverge; per-slot output is uniquely garbled (interleaving-order-dependent corruption). Intel for Tier 3 design."
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

`production/2026-q2-next` 2026-05-21 (probe at submodule c2a142a4 + uncommitted scratch):

Ran kernel-launcher FA-probe under `GGML_CUDA_DISABLE_GRAPHS=1` against three pairs:
- **active NP=1** vs **active NP=8 concurrent slot-0** — output byte-identical (expected baseline).
- **dropped NP=1** unchanged (bailout doesn't gate when `n_stream=1`).
- **dropped NP=8 concurrent slot-0** vs **active NP=1** — output text DIVERGES.

Probe captured per-FA-call: K_hash, V_hash, mask_hash (64-bit FNV1a-style over the addressed view bytes), per_row_k_bound[0], K view offset, parent stride.

**Output under DROPPED NP=8 (per-slot, all 8 with identical prompt+seed):**
```
slot0: " the idea that the brain could be be"
slot1: " the mind could be be be be be"
slot2: " the human thought could be be bearded"
slot3: " the be simulated simulated by the the the"
slot4: " the could be simulated simulated by by the"
slot5: " the the by a a a a a"
slot6: " the the the the the the the the"
slot7: " the by machines.\n\n<think>\n\n<think>"
```
Each slot produces UNIQUE degraded output — interleaving-order-dependent corruption, not a uniform shared error.

**FA-probe hashes for stream-0, layer flash_attn_per_slot_kv-1003 (device-0 layer-3):**

At bound 1..9 — ACTIVE NP=1 and DROPPED NP=8 stream-0 K_hash, V_hash, mask_hash ALL MATCH.

At bound=10 onward — K_hash and V_hash DIVERGE; mask_hash IDENTICAL; bound value identical; K view offset identical.

Example bound=10:
```
ACTIVE NP=1   K=0x38470ddd8ccfa8f7 V=0xdca5fcd332a41ea9 mask=0x521fb34eba4c3580
DROPPED NP=8  K=0x9cfba89d7eec0990 V=0x066957185361cdf4 mask=0x521fb34eba4c3580
```

Cross-stream K_off (= view_offs) is correctly patched per-tick under dropped mode — verified via call 11 (stream 1) at K_off=0x120000 = 1·parent->nb[3], call 14 (stream 2) at K_off=0x240000, etc. So `update_cache_copies()`'s read-view loop IS patching the K view's data pointer correctly.

**The corrupted bytes live IN the K/V cache memory at the (correctly-patched) stream offset.** Some prior write put bad data into stream 0's slice between its bound=9 and bound=10 FA calls. Stream-0 didn't touch its own slice in that interval — streams 1..7 did their own decodes between.

What's known good under DROPPED mode:
- `find_slot` allocates cells in the correct stream's slice (mask construction agrees byte-for-byte with active mode → cells[].pos and cells[].seq_id are consistent).
- `per_row_k_bound[0]` is correct per-stream (read from `llama_kv_cache_seq_pos_max(seq_id)`).
- K read view's view_offs and data pointer are correctly patched per-tick (verified).
- CPY destination view's view_offs is correctly computed (`stream_id * parent->nb[3] + pos * step`).

What's broken:
- The K cache bytes AT THE CORRECTLY-ADDRESSED STREAM SLICE are wrong after stream interleave.

Hypothesised cause (NOT confirmed by this diagnostic — requires sched-internal instrumentation to confirm):
- ggml-backend-sched optimises intermediate buffer allocation. When a graph is reused across streams, the K projection scratch buffer (or some other intermediate) is reused; stream-N's K projection writes into a buffer that gets read by stream-0's CPY before stream-0's re-execution refreshes it. OR: sched aliases a buffer between the K projection output and the K cache READ view, so what FA reads is the LAST STREAM's K projection rather than the cache.
- Alternative: cuda-stream ordering. CPY writes happen on one CUDA stream, FA reads on another, without proper synchronisation across ggml-backend-sched's per-backend streams under reuse.

**Implication for Tier 3 (unified-stream dispatch via PSKV `ne[3] > 1` packing):**

The structural fix: ONE `llama_decode` per tick processes all active streams' tokens in a single batch with multi-seq ubatch (à la upstream `split_equal`). The graph builds K/V/mask/Q as 4D tensors with `ne[3]=n_stream`. The PSKV kernel iterates `ne[3]` internally — each block targets `(token, head, seq)`. No graph reuse across stream boundaries is needed because every tick already processes every stream.

This avoids:
- Cross-tick buffer aliasing in sched (each tick has its own intermediate-buffer lifecycle).
- Per-tick view_offs patching dependencies.
- The bailout / can_reuse_graph branching at `n_stream>1`.

T3 Step 1 (PSKV kernel `nb33` mask addressing for ne[3]>1) is already landed at submodule `c2a142a4`. Remaining T3 work (find_slot multi-seq, llama_decode multi-seq batch, 4D K/V/mask build, server dispatch, bailout removal) is substantial but well-scoped.

How to apply: Tier 3 implementation should NOT attempt to patch any intermediate state per-tick — instead make all data flow through the graph topology as 4D tensors. Treat the `update_cache_copies` per-tick patching as a legacy mechanism that Tier 3 removes (the cache_copies and cache_read_views registries become unused once dispatch is unified).
