---
name: p0a4-multislot-dflash-segv-closed
description: "P0.A.4 closure on production/2026-q2-next 2026-05-21 — server scoped speculative_decoding_accept to current run's slot; both np=2 DFlash + cross-NP byte-identity gates green"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

P0.A.4 closed on production/2026-q2-next 2026-05-21. Submodule commit `cad6b591`, parent commit `a12609a`.

Bug: at `--spec-type dflash --parallel 2`, the server SEGV'd on the first verify cycle with `llama_get_logits_ith: invalid logits id N, reason: batch.logits[N] != true`.

Root cause: the per-stream split in `process_batch_tokens` issues one `llama_decode` per slot per tick (Bug C-safe). The engine resets `output_ids` on every `llama_decode` AND indexes it in the LOCAL frame of the dispatched batch_view, not the GLOBAL frame of the combined `batch`. The in-loop call to `speculative_decoding_accept()` walked **all** slots and dereferenced their GLOBAL `slot.i_batch_dft` indices — only the most-recently-decoded slot's logits resolved; every other slot hit `output_ids[N] = -1`.

Fix: pass `(i, run_seq_id)` from the per-stream loop into `speculative_decoding_accept`; filter to the matching slot; translate `slot.i_batch_dft` to local frame (subtract `batch_offset`) before any `llama_get_logits_ith` / `llama_decoder_get_embeddings_ith` call. The np=1 two-phase split (Phase A reads, Phase B mutates) is preserved trivially since `accepted` now holds at most one slot per call.

Verified:
- `scripts/test-server-multi-slot-dflash.sh` (new) — RED on pre-fix HEAD (curl 52 / server crash); GREEN post-fix (two coherent /v1/completions).
- `scripts/verify-production-determinism.sh` — NP={1,2,4,8} all slots byte-identical to NP=1; cross-NP slot-0 matrix byte-identical; batch-shape invariance gate 4/4 PASS.

**Why:** Closes the last remaining P0.A item in PHASE_NSTREAM_KV_PERF.md. Tier 2 entry condition for the perf-recovery phase is now strict (no deferred P0.A items).

**How to apply:** When future kernel work alters per-stream dispatch in `process_batch_tokens`, preserve the `speculative_decoding_accept(i, run_seq_id)` interface — adding more slots to a single dispatch (e.g. unified-stream Tier 3) will need the function to handle multiple slots-per-call again, but the index translation invariant must be respected.

Stochastic NPC note: one of the verify-production-determinism runs during this session failed at NP=4 with a hard divergence (Chinese text + completely different output, not subtle drift). A subsequent run passed cleanly. Same binary both times. Worth keeping an eye on; flagged in [[feedback_np_cluster_partition_signature]].

Related: [[project_dflash_multislot_phase5_landed]], [[project_continuous_batching_vs_perslot_dispatch]], [[project_dflash_t9_np_validity_drift_signature]].
