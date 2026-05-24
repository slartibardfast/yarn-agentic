---
name: dflash-multislot-phase5-landed
description: DFlash multi-slot Phase 5 (common_speculative orchestrator fan-out) landed on production/2026-q2-next 2026-05-18; 5 gates green; server CLI wiring for --spec-type dflash + --model-draft <sidecar> still routes through standalone draft loader and fails on missing tokenizer.ggml.tokens — separate pre-existing wiring gap
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

DFlash multi-slot libllama API extension Phase 5 on `production/2026-q2-next` on 2026-05-18.

**Plan:** `data/dflash-multi-slot-impl-plan-2026-05-18.md` (6 phases).
**Prior phases:** [[dflash-multislot-phase4-landed]] (and earlier).

## Phase 5 landed (commits `92fbe36c` submodule, `8d83779` parent)

### Adapter glue in `common/speculative.cpp`
- `common_speculative_state_dflash` now carries `seq_id` and `owns_drafter`.
- First per-slot state on shared ctx_tgt loads + binds; later states detect the existing binding via new public getter `llama_get_dflash_drafter(ctx)` and share without rebinding. Destructor only frees the drafter when `owns_drafter` is set.
- `state_dflash::draft()` routes through `llama_dflash_draft_batch(n_slots=1, …)` — single- and multi-slot share one impl path; byte-identical to the legacy `llama_dflash_draft` per Phase 4's n_slots=1 trampoline.
- `common_speculative_draft_batched`: all-DFlash fan-out path mirroring the all-MTP gate. Packs per-slot anchor_token/anchor_pos/seq_id arrays and dispatches one `llama_dflash_draft_batch` per draft cycle. Per-slot unpack stride = `rc / n_slots` (the operating BS — can be smaller than drafter's advertised `block_size`). Falls through to per-slot serial fallback on mismatch.

### New binding test
`tests/dflash-speculative/test-dflash-spec-batched-fanout.cpp`:
- Symmetric: `draft_batched({spec0, spec1})` byte-identical to two serial `common_speculative_draft` calls.
- Asymmetric: per-seq prefills with different prompts yield per-slot outputs that differ — proves seq_id flows through the gate.

### Critical bug caught by the new binding test
First fanout impl used `BS = block_size = 16` (drafter's advertised max) for the slot-major output stride. The actual kernel writes BS=4 per slot. Slot 1's real data at flat_out[4..8) was read from flat_out[16..20) → zeros. Fix: derive `per_slot = rc / n_slots` from the kernel's return value. See `[[drafter-forward-n-slots-cap]]` for the related lesson on bind-time vs dispatch-time stride confusion in the same kernel.

## Verifications (locked clocks 1455 MHz, dual Quadro RTX 6000)
1. Build clean: llama, llama-server, all dflash tests.
2. `test-dflash-closure` 8/8 prompts argmax-equivalent.
3. `test-dflash-np-invariance` (T7) 4/4 seeds × N ∈ {1,2,4,8} byte-identical FNV-1a64.
4. `test-dflash-extract-multi-seq` (Phase 3) per-seq buffer counts unchanged.
5. `test-dflash-batch-vs-serial` (Phase 4) byte-identical.
6. **NEW** `test-dflash-spec-batched-fanout`: symmetric ≡ serial + asymmetric per-slot routing.
7. `scripts/verify-production-determinism.sh` NPC PASS NP ∈ {1,2,4,8} multi-GPU CUDA0,CUDA1.

## Out-of-scope: server CLI wiring gap (pre-existing)
`llama-server --spec-type dflash --model-draft <sidecar.gguf>` does NOT work: `--model-draft` sends the sidecar through the standalone draft-model loader which rejects it on missing `tokenizer.ggml.tokens` (the sidecar shares the target's tokenizer by design). The orchestrator glue is now complete and exercised by the binding test, but reaching it from llama-server CLI requires wiring `--spec-type dflash` to call `common_speculative_init` with `mparams_dft.path` set without going through `load_model_draft` first.

No DFlash profile in `profiles/` exists. End-to-end llama-server smoke at `--parallel 2 --spec dflash` is gated on this server-side wiring fix, which is a separate workitem.

## What Phase 5 actually enables (right now)
Any caller using `common_speculative_init` + `common_speculative_draft_batched` directly (i.e. via libcommon, not llama-server) gets multi-slot DFlash draft cycles bundled into one kernel pipeline.
