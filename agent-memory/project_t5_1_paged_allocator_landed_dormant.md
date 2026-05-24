---
name: project-t5-1-paged-allocator-landed-dormant
description: "T5.1 paged KV block allocator landed as dormant class; data structures + init only, find_slot untouched; verify-production-determinism PASS at NP={1,2,4,8}; integration is T5.2-T5.4"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

production/2026-q2-next 2026-05-23.

**Fact**: T5.1 — block allocator + block table — landed as a standalone,
dormant class.

- New: src/llama-paged-kv-allocator.{h,cpp}. Class implements 5
  binding invariants from [[paged-block-allocator-allium-spec]]:
  BlockUniquelyOwned, FreeListDisjoint, AllocLazy,
  DeterministicAtFixedSequence, IdentityMappingAtSingleSeq.
- Block size locked at 64 tokens (BLOCK_SIZE_TOKENS constant per
  PHASE doc Mechanism section; not configurable per
  [[feedback-no-workarounds]]).
- LIFO std::deque free_list ensures determinism + L1-cache locality
  for recently-freed blocks.
- Transactional write_tokens(): partial-alloc OOM is rolled back so
  seq state is unchanged from before the call.

Wiring:
- llama_kv_cache (src/llama-context.h) gets a `paged` field.
- kv_cache_init initialises paged.init(n_blocks, n_seqs) alongside
  the contiguous layout. n_blocks = ceil(kv_size / 64).

**Design discipline**: DORMANT at T5.1.

- The allocator is initialised but NOT consumed by find_slot or the
  K/V WRITE/READ paths.
- Production behaviour at n_stream==1 single-seq workloads is
  byte-identical to pre-T5.1 because the contiguous v_heads / cells[]
  path remains the sole source of truth.
- T5.2 (WRITE-path inp_kv_idxs formula), T5.3 (CPY-fallback removal),
  T5.4 (find_slot integration) progressively migrate WRITE to consume
  paged.block_table(seq).

**Gates GREEN**:
- test-kv-block-allocator: 7 invariant checks + OOM signal + LIFO +
  determinism. Exercises the production class directly (transitioned
  from the T5.0 RefAllocator+RED-flag pattern).
- test-paged-allocator-determinism: 3 deterministic traces × 3 runs
  + pool-exhaustion stress + LIFO post-free re-alloc.
- verify-production-determinism.sh @ 1455 MHz locked, NP_LIST="1 2 4 8":
  ACCEPTANCE PASS — cross-NP determinism + cross-shape invariance
  unchanged. Confirms the dormant-allocator design choice (skipping
  find_slot integration at T5.1) preserves production.

**Why dormant**: per the user's "complete sincerity" directive +
[[feedback-oneshot-then-evaluate]], each T5.x card lands coherently.
T5.1's scope is data structures + algorithm; integration is its own
verification challenge belonging to T5.2-T5.4. Keeps the verify-
production-determinism gate trivially GREEN at T5.1 (additive change
only) and isolates the failure modes if T5.2-T5.4 introduce
regressions.

**How to apply**:
- Future T5.2-T5.4 work can rely on the allocator being initialised
  at kv_cache_init time. Just call `cache.paged.write_tokens(seq, n)`
  / `cache.paged.block_table(seq)` etc.
- The LANDED build flag is NOT used yet — the test compiles + drives
  the production class directly. The 6 remaining RED-bound tests
  (test-paged-*) still gate on LLAMA_PAGED_KV_LANDED for their T5.x
  cards.
- Block-size = 64 is hardcoded. OpenQ-T5-A (drop to 128) is a
  contingency if kernel indirection > 5%; not exercised at T5.1.

**Related**: [[project-t5-probe-falsified-path-c-override]],
[[feedback-oneshot-then-evaluate]], [[feedback-no-workarounds]],
[[paged-block-allocator-allium-spec]] (specs/kv-cache/paged_block_allocator.allium).
