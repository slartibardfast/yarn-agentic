---
name: project-t5-9-b-landed-npc-pass
description: "production/2026-q2-next 2026-05-23 — T5.9.A + T5.9.B landed & pushed. Paged BACKING layout flip complete: K/V tensor [head_dim, BLOCK, n_head_kv, total_pool_blocks] block-major; allocator seed_identity_per_stream at auto-size preserves byte semantics; --kv-pool-blocks N CLI override for under-allocation; find_slot pre-checks paged exhaustion and propagates HTTP 503; build_defrag reworked per-position via block_table. GP5.9.NPC ACCEPTANCE PASS + batch-shape invariance ALL PASS at NP={1,2,4,8}. T5.9.C/D/E pending."
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

T5.9.A and T5.9.B both landed and pushed on `production/2026-q2-next` 2026-05-23 same day as T5 reopen. Parent HEAD `f160361`; submodule HEAD `41b6f99d`. T5.1–T5.8 infrastructure preserved as-is; T5.9 builds on top.

**Status grid:**
- T5.9.A — spec (`paged_kv_pool_sizing.allium`, 3 contracts, allium GREEN) + TLA (PagedKVAllocator extension + sibling MC config that forces OOB) + trace validator (`PoolBoundsRespected` check + header parsing) + 2 RED-acting-as-GREEN tests (`test-pool-bounds-respected`, `test-paged-backing-feasibility`, both PASS on HEAD as defence-in-depth).
- T5.9.B — the layout flip. ✓ Submodule sub-commit `41b6f99d`; parent submodule bump `f160361`.
- T5.9.C — bench harness + sibling profile + ledger. PENDING.
- T5.9.D — gate sweep. GP5.9.NPC + batch-shape invariance ALREADY PASS. GP5.9.regression (bench-t3.8-m3 at locked clocks) + GP5.9.feasibility (ctx 1M NP=8 sibling profile with `--kv-pool-blocks` sized to fit VRAM) + GP5.9.exhaustion + GP5.9.spec PENDING.
- T5.9.E — closure docs (PHASE_NSTREAM_KV_PERF.md T5.9 A/S/M/E/C, MEMORY entry, auto-memory). PENDING.

**T5.9.B layout flip — what changed:**
1. K/V tensor shape `[head_dim, kv_size_per_stream, n_head_kv, n_stream]` → `[head_dim, BLOCK_SIZE_TOKENS, n_head_kv, total_pool_blocks]`. Block-major. Buffer total bytes unchanged at auto-size (`total_pool_blocks = ceil(kv_size/BLOCK_SIZE_TOKENS)` = `nbps × n_stream`); proportionally smaller at user override.
2. Allocator's `seed_identity_per_stream()` called at auto-size — pre-allocates each seq's nbps blocks at IDs `[s*nbps, s*nbps+nbps-1]`. This preserves byte semantics relative to T5.8 stream-major layout. Skipped at `--kv-pool-blocks N` override — admissions go through lazy alloc + admission gate.
3. Kernel arithmetic (`paged_nb13 = nb1 × BLOCK × ne12`) unchanged in formula; ne12 (= n_head_kv) and nb1 (= head_dim × type_bytes) both unchanged across shapes; same bytes addressed.
4. `build_k_shift` was already per-block-via-block_table at T5.7b. Only relaxed the `total_blocks() == nbps × n_stream` assertion (under user override this may not hold).
5. `build_defrag` REWORKED: was stream-major (`s * parent->nb[3] + p_local * nb[1]`); under T5.9 `nb[3]` is the block stride not stream stride, so legacy formula silently corrupted bytes. New formula: per-(seq, position) `bid * nb[3] + offset_in_block * nb[1]` where `bid = block_table[s][p_local/BLOCK]`. Run-batching dropped (positions not necessarily co-block on src and dst; defrag is rare under `defrag_thold=-1.0f` default).
6. `find_slot` multi-seq path: pre-check the paged allocator across ALL seq runs BEFORE committing cells writes. If any run would exceed pool capacity, return false → `llama_decode` returns ALLOC_FAILED → server's existing 503 + Retry-After path fires. Single-seq path: same pattern, with cells rollback on paged failure.
7. CLI `--kv-pool-blocks N` added to common args; default 0 = auto. Production profile (`qwen36-27b-x2-dflash.sh`) DOES NOT set this — production stays at auto-size, byte-identical to T5.8.

**Binding evidence at T5.9.B close:**
- GP5.9.NPC `verify-production-determinism.sh NP_LIST="1 2 4 8" CTX_CHECKPOINTS=3`: ACCEPTANCE PASS. All NP slots byte-identical to NP=1; all cross-NP slot-0 byte-identical.
- Batch-shape invariance gate: all 4 tests PASS (libllama-verify-batch-width-sweep, libllama-multi-cycle-restore-drift, kernel-mulmat-batch-shape, kernel-mulmat-mmq_x-dispatch).
- Production NP=2 ctx 524288 server smoke: coherent output (`"Hello"` → `", I am trying to create a"`).
- `--kv-pool-blocks 8192` (= nominal max at this ctx/np) smoke: identical output as auto.
- Spec tests: `test-pool-bounds-respected`, `test-paged-backing-feasibility`, `test-paged-allocator-determinism`, `test-paged-defrag-preserves-contents`, `test-kv-block-allocator`, `test-fattn-per-slot-kv-dispatch-np-invariance` — all PASS.

**Stale-build trap that bit me & flushed:**
Initial run of verify-production-determinism failed with `*** stack smashing detected ***` in test-dflash-{verify-batch-width-sweep,multi-cycle-restore-drift}. Cause: `cmake --build build --target X` only rebuilds X's direct dependencies, not transitive dependents of changed PUBLIC headers. The two dflash tests were compiled against an older `include/llama.h` (before I added `kv_pool_blocks` to `llama_context_params`), causing ABI mismatch when linked against the new libllama.so. **Future fix**: after any change to public headers (`include/llama.h`, `common/common.h`), always `cmake --build build -j 32` (full target, no `--target` flag) before running gates.

**What's pending — quick-start for next session:**

1. T5.9.C deliverables (~20-30k tokens):
   - `scripts/bench-ctx-feasibility.sh` (mimic `bench-t4-m3-staggered.sh`). Args: CTX_PER_SLOT (default 1048576), NP (8), POOL_BLOCKS (default 1/16 nominal), N_PROMPTS (16). Pass criteria: ≥1 status 200, ≥1 status 503 + Retry-After, `/health` 200 after.
   - `profiles/qwen36-27b-x2-dflash-bigctx.sh` (sibling profile; same engine + `--ctx-size 1048576 --parallel 8 --kv-pool-blocks N`). N picked to fit VRAM after measurement.
   - `data/t5.9-perf-gate-ledger.md`.

2. T5.9.D remaining gates:
   - **GP5.9.regression**: `bash scripts/bench-t3.8-m3.sh` 3 runs at 1455 MHz. Pass if aggregate t/s ≥ T5.8 baseline 26.45 × 0.98.
   - **GP5.9.feasibility** (HARD): run new bench-ctx-feasibility at sibling profile ctx 1M NP=8.
   - DFlash gates (test-dflash-{np-invariance, np-multislot, closure}) — likely GREEN given GP5.9.NPC pass but verify.
   - K-shift / defrag model-level tests (test-paged-kshift-byte-identity, test-kv-shift-per-stream, test-kv-defrag-per-stream) at production sizes.

3. T5.9.E closure: PHASE_NSTREAM_KV_PERF.md T5.9 A/S/M/E/C section appended to existing T5 closure; close the reopen note. MEMORY.md append-only entry. Auto-memory `project_t5_9_paged_backing_closed.md` (supersedes this entry once gates close).

**Operational constraints in force (carried verbatim):**
- No yarn-agentic nomenclature in code, scripts, tests, directories, branch names
- Production profile `qwen36-27b-x2-dflash.sh` UNCHANGED (auto-size, no `--kv-pool-blocks`)
- Locked clocks 1455 MHz required for verify-production-determinism
- coord/gpu BUSY/IDLE state machine before/after benchmarks
- Never run concurrent verify-production-determinism runs
- Never run concurrent inference benchmarks
- Don't write large artifacts to /tmp (tmpfs)
- All PLAN/PHASE/MEMORY edits commit+push immediately
- Never skip hooks, force-push to main/master, or amend commits
- Plan file `/home/llm/.claude/plans/cached-crunching-tiger.md` is the T5.9 plan.

Related: [[project-t5-9-paged-backing-reopen]] (reopen decision), [[project-t5-8-tier-5-closed]] (preserved audit record, closure-verb overridden by T5 reopen), [[project-t5-7-bundle-b-complete]] (T5.7b paged-K-shift this builds on), [[feedback-no-followup]], [[feedback-no-workarounds]], [[feedback-oneshot-then-evaluate]].
