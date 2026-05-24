---
name: project-t5-2-shadow-landed-t5-3-next
description: T5.2 shadow paged-allocator integration landed; production NPC PASS; T5.3 is the coherent Bundle A close (formula + K/V view + CPY-fallback removal) where production bytes first flow through paged
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

production/2026-q2-next 2026-05-23. **Compact-survival entry.**

## What's done — T5.0 through T5.2

- **T5.0** (`PHASE_NSTREAM_KV_PERF.md` Tier 5 scope-lock):
  - 4 Allium specs in `specs/kv-cache/`: `paged_block_allocator.allium`,
    `paged_write_path.allium`, `paged_read_path.allium`,
    `paged_kshift_defrag.allium`. `allium check` 0 errors.
  - 2 TLA+ triplets: `PagedKVAllocator{.tla,MC.tla,MC.cfg}` (6140 states),
    `PagedKVByteIdentity{.tla,MC.tla,MC.cfg}` (1282 states). TLC GREEN.
  - 8 RED-bound property test stubs in `ik_llama.cpp/tests/spec/test-*`.
    Build, run, FAIL with explicit T5.x-not-landed messages.
  - `scripts/validate-paged-allocator-trace.py` validator (NDJSON
    trace per BlockAllocEvent schema).
  - `ik_llama.cpp/src/llama-paged-kv-trace.h` stub header.
  - `data/t5-perf-gate-ledger.md` initialised.

- **T5.1** (parent commit `cacb5f2`; submodule `622558ba`):
  - `src/llama-paged-kv-allocator.{h,cpp}` standalone class.
  - `BLOCK_SIZE_TOKENS = 64` constant. LIFO `std::deque` free_list.
    Transactional `write_tokens()` rolls back partial allocs on OOM.
  - `llama_kv_cache.paged` field; initialised in `kv_cache_init` with
    `n_blocks = ceil(kv_size / 64)`.
  - **DORMANT** — not consumed by find_slot or K/V WRITE/READ paths.
  - `test-kv-block-allocator` + `test-paged-allocator-determinism`:
    PASS (transitioned from RefAllocator+RED-flag to production class).
  - `verify-production-determinism.sh @ 1455 MHz, NP={1,2,4,8}`:
    ACCEPTANCE PASS.

- **T5.2** (parent commit `00339f4`; submodule `9cc32cd4`):
  - Shadow integration: `paged.write_tokens(seq, n)` wired at BOTH
    find_slot commit phases (multi-seq line ~1525 + single-seq
    line ~1585), after `cells[]` write + `v_heads` advance.
  - `kv_cache_clear`: `paged.free_seq` for every seq.
  - Partial `seq_rm` NOT wired (deferred to T5.3 as prerequisite).
  - `(void)` cast suppresses `write_tokens` return — non-fatal at
    shadow stage; becomes hard error at T5.3 when paged becomes
    load-bearing.
  - `verify-production-determinism.sh`: ACCEPTANCE PASS again.

## What's next — T5.3 coherent Bundle A close

**Scope deviation from the plan's literal T5.2-T5.4 split** (called out
in T5.2's commit message):

The plan's literal T5.2 (formula + K/V view reshape) cannot ship in
isolation because legacy and paged formulae produce DIFFERENT indices
at `n_head_kv > 1`:

- Legacy (multi-seq path): `idx[t,h] = stream_base + h * kvps + p_t`
  (head-major within stream — heads SEPARATED by `kvps` positions)
- Paged: `idx[t,h] = block_table[seq][t/64] * 64 + (t%64) * n_head_kv + h`
  (block-major — heads PACKED within block, n_head_kv consecutive
  entries per position)

Setting equal at single-seq identity mapping: `(t%64)*(n_head_kv - 1)
= h*(kvps - 1)` — only holds at `n_head_kv = 1`. Production
(Qwen 3.6 27B GQA) has `n_head_kv = 8`. ⇒ formula change ALONE breaks
production byte-identity.

**Therefore T5.3 = coherent Bundle A close** comprising:
1. Wire partial `seq_rm` shadow updates (prerequisite — partial removal
   must keep `written_tokens` and `block_table` consistent with `cells[]`).
2. Replace `inp_kv_idxs` formula (`src/llama.cpp` ~5115-5168) with the
   paged formula.
3. Reshape SET_ROWS target view (`src/llama-build-context.cpp`) to
   `ggml_view_2d(k_cache, [head_dim_gqa, n_blocks * block_size], ...)`.
4. Reshape kernel K/V views to `[head_dim, block_size, n_head_kv,
   n_blocks]` — strides set so bytes-at-identity-mapping match legacy
   (the kernel reads via `nb[1]/nb[2]/nb[3]` so it follows automatically
   IF the strides are correct).
5. Remove the CPY-with-view-offset fallback (T5.3 binding contract:
   `PagedKVWriteEquivToLegacyAtIdentity` in `paged_write_path.allium`).
6. Binding gate: `verify-production-determinism.sh` PASS at NP={1,2,4,8}
   via paged path end-to-end. Hard.

**Estimate**: 40-60k tokens including 1-2 verify-prod-determinism
rounds for diagnosis if it fails.

**Highest risk**: getting the K/V kernel-view strides right at step 4.
The PSKV singlewarp kernel uses `nb[1]/nb[2]/nb[3]` (verify by
reading `fattn-per-slot-kv-singlewarp-sm75.cu`). If yes, the bytes
at identity mapping are byte-equivalent automatically. If the kernel
has any hardcoded arithmetic that bypasses `nb[*]`, we have to update
it (which crosses into T5.5 territory).

## Critical context

- Branch: `production/2026-q2-next`.
- Production profile: `profiles/qwen36-27b-x2-dflash.sh` — Qwen 3.6 27B
  Q4_0 + Hadamard, NP=2, DFlash, ctx=524288.
- Locked clocks: 1455 MHz both GPUs (verify-production-determinism
  enforces). GPUs IDLE in coord/.
- Per Path C reframe (`data/t5-probe-findings.md` §9): T5 is
  forward-looking infra for high-ctx feasibility (ctx ≥ 1M), NOT
  current-workload throughput uplift. GP5.b is a feasibility gate,
  not numeric throughput.
- "Complete sincerity" discipline: every commit must be tested + verify-
  prod-determinism GREEN. No shortcut specs, no deferred tests, no env
  knobs left around (per `[[feedback-bake-measurement-env-gates]]`).
- The Bundle A coherent commit pattern is per
  `[[feedback-oneshot-then-evaluate]]` — write the bundle coherently,
  evaluate at the end with measured results.

## Files touched at T5.1+T5.2

- `ik_llama.cpp/src/llama-paged-kv-allocator.{h,cpp}` (new)
- `ik_llama.cpp/src/llama-context.h` (paged field added)
- `ik_llama.cpp/src/llama.cpp` (kv_cache_init + find_slot shadow +
  kv_cache_clear shadow)
- `ik_llama.cpp/src/CMakeLists.txt` (allocator sources)
- `ik_llama.cpp/tests/spec/test-kv-block-allocator.cpp` (drives prod class)
- `ik_llama.cpp/tests/spec/test-paged-allocator-determinism.cpp` (same)
- `ik_llama.cpp/tests/CMakeLists.txt` (link allocator into tests)
- `data/t5-perf-gate-ledger.md` (T5.1 + T5.2 rows)

## Related memories

[[project-t5-1-paged-allocator-landed-dormant]],
[[project-t5-probe-falsified-path-c-override]],
[[project-t4-bundle-a-landed]],
[[feedback-oneshot-then-evaluate]],
[[feedback-bake-measurement-env-gates]],
[[feedback-no-workarounds]].
