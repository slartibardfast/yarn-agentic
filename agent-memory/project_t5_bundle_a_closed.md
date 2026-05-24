---
name: project-t5-bundle-a-closed
description: Tier 5 Bundle A (T5.1-T5.4) closed — paged allocator landed dormant + shadow integration + partial seq_rm + LLAMA_T5_TRACE producer + NPC PASS + GP5.a perf gate PASS at +0.04% vs T4 baseline; Bundle B (T5.5+) is the byte flip + kernel block_table indirection
metadata:
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

production/2026-q2-next 2026-05-23.

**Tier 5 Bundle A CLOSED** across T5.1-T5.4 per `PHASE_NSTREAM_KV_PERF.md`
lines 2154-2160 (Bundle A = allocator + WRITE-side bookkeeping; Bundle B
= READ path + kernel flip).

## Bundle A delivered

- **T5.1** (sub `622558ba`): `llama_paged_kv_allocator` class, 5 spec
  invariants (BlockUniquelyOwned, FreeListDisjoint, AllocLazy,
  Deterministic, IdentityMappingAtSingleSeq), block_size=64,
  transactional `write_tokens`, LIFO `std::deque` free list. Initialised
  at `kv_cache_init`; not consumed by find_slot or K/V WRITE/READ paths.
- **T5.2** (sub `9cc32cd4`): shadow `paged.write_tokens` at find_slot
  multi-seq + single-seq commit phases; `paged.free_seq` per stream at
  `kv_cache_clear`.
- **T5.3** (sub `68686e17`): partial `seq_rm` shadow — after the
  existing cell-freeing loop, recount per-stream used cells and resync
  via `free_seq + write_tokens(count)`. Plus the `LLAMA_T5_TRACE`
  NDJSON producer in `src/llama-paged-kv-trace.{h,cpp}`, env-gated,
  emits BlockAllocEvent records hooked into the allocator's alloc/free.
- **T5.4** (parent `c0111e1` ledger row + `e93055e` perf gate close):
  GP5.a regression band PASS — 26.50 t/s mean (CV 0.14%) vs T4
  C1-steady baseline 26.49 t/s = **+0.04%** delta, comfortably inside
  ±2% band [25.96, 27.02].

## Verification gates (all GREEN at T5.3/T5.4)

- `verify-production-determinism.sh` @ 1455 MHz, NP={1,2,4,8},
  CTX_CHECKPOINTS=3 → ACCEPTANCE PASS (cross-NP byte-identity +
  batch-shape invariance both PASS).
- `test-kv-block-allocator` PASS (7 invariant checks).
- `test-paged-allocator-determinism` PASS (3×3 traces + OOM + LIFO).
- LLAMA_T5_TRACE producer smoke: 10 events validated by
  `validate-paged-allocator-trace.py`; OFF-path zero output.
- GP5.a regression band: PASS at +0.04%.

## Key architectural finding — why Bundle A vs B split is correct

The PSKV singlewarp kernel
(`ggml/src/ggml-cuda/fattn-per-slot-kv-singlewarp-sm75.cu:99-100,145-148`)
reads K via uniform stride: `K_base = K_direct + nb13*seq + nb12*head_kv`,
then `K_row = K_base + k*nb11`. There is no `block_table` indirection.
Any paged layout where seq's blocks scatter the pool (i.e., non-identity
mapping) breaks linear k-stepping at every block boundary.

⇒ No `nb[*]` strides setting can make the kernel paged-aware. Block_table
indirection must happen INSIDE the kernel. This is Bundle B (T5.5).

The prior compact-survival memory
`[[project-t5-2-shadow-landed-t5-3-next]]` mis-scoped T5.3 to bundle the
formula flip + view reshape + CPY-fallback removal into Bundle A.
Corrected in `[[project-t5-3-scope-corrected-bundle-a-close]]`.

## What Bundle B (T5.5-T5.8) must deliver

Per PHASE doc §"Bundle B":

- T5.5: PSKV singlewarp + sm_75 kernels take `block_table` tensor as
  `src[6]`, dereference per `(seq, k/block_size)` to compute byte
  address inside the K-loop. Kernel NPC at the new signature is the
  largest single risk (PHASE doc line 2384).
- T5.6: `inp_kv_idxs` formula replacement at `src/llama.cpp` (the
  WRITE-time mapping must match the kernel's READ-time interpretation).
- T5.7: SET_ROWS target view reshape (`src/llama-build-context.cpp`)
  + CPY-with-view-offset fallback removal (T3.6 same-type Q→Q kernel
  composes).
- T5.8: Bundle B perf gate + M4 high-ctx feasibility binding (Path C
  reframe) + LLAMA_T5_TRACE env-gate bake-out per
  `[[feedback-bake-measurement-env-gates]]` + Tier 5 closure.

Token estimate: 60-100k (PHASE doc line 2384). Must ship as one
coherent commit per `[[feedback-oneshot-then-evaluate]]`.

## Critical context

- Branch: `production/2026-q2-next`.
- Latest parent HEAD: `e93055e` (T5.4 close).
- Latest submodule HEAD: `68686e17` (T5.3).
- Locked clocks 1455 MHz required for verify-prod-determinism.
- Production profile: `profiles/qwen36-27b-x2-dflash.sh` — unchanged.
- GPUs IDLE in coord/.
- Path C reframe (`data/t5-probe-findings.md` §9): T5 is forward-
  looking infra for high-ctx feasibility (ctx ≥ 1M NP=8 unallocatable
  under contiguous, ~1.2 TB needed vs 48 GiB available). GP5.b
  hard-gate is M4 feasibility, not numeric current-workload throughput.

## Related

[[project-t5-3-scope-corrected-bundle-a-close]],
[[project-t5-2-shadow-landed-t5-3-next]] (superseded),
[[project-t5-1-paged-allocator-landed-dormant]],
[[project-t5-probe-falsified-path-c-override]],
[[feedback-oneshot-then-evaluate]],
[[feedback-bake-measurement-env-gates]],
[[feedback-no-workarounds]].
