---
name: project-t5-7-bundle-b-complete
description: "production/2026-q2-next 2026-05-23 — T5.7 Bundle B all three sub-steps landed (a paged K-shift, b n_stream==1 pretence drop, c allocator-level paged defrag); pretence drop was net -295/+130 LOC (dual-path code reduction); kernel nullptr branch + defrag graph-level integration deferred to T5.8"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

T5.7 Bundle B closed on `production/2026-q2-next` 2026-05-23. Three coherent sub-commits then T5.8 = perf-gate sweep + closure.

**Heads of record:**
- T5.7a (paged K-shift port + binding test): submodule `ee67864c`, parent `9557936`
- T5.7b (n_stream==1 contiguous-layout pretence drop): submodule `0544573b`, parent `5801fa0`
- T5.7c (allocator-level paged defrag + binding test): submodule `7f6fadf6`, parent `6fa59cf`

**T5.7b structural finding — pretence drop is NET CODE REDUCTION:**
Diff was -295 / +130 LOC across 2 files (`src/llama-build-context.cpp` + `src/llama.cpp`). The dual-path dispatch machinery (paged vs legacy view-3d) at every K/V touchpoint was the bulk of the conditional logic; removing it cut more code than the always-on paged path added. Sites: WRITE (kv_store + std_attention) drop CPY-fallback else; READ K/V views always 4D with ne[3]=n_seq_in_batch (single-seq → ne[3]=1); FA op always set_block_table; Q reshape always 4D; build_k_shift drops paged_mode local. At single-seq batches with active stream s > 0, the global `[nbps, n_stream]` block_table tensor is viewed as `[nbps, 1]` at byte offset `s * nb[1]` so kernel blockIdx.z=0 indexes the active stream's blocks.

**Load-bearing non-obvious finding — NP=1 byte layout shifts but output stays equal:**
At NP=1 paged, the K cache buffer's bytes for (stream s=0, head h, position p) move from legacy addresses `p*nb[1] + h*nb[2]` to paged addresses `((p/64)*64*n_head_kv + h*64 + p%64) * nb[1]`. **Attention math output is invariant under this layout permutation** (identity block_table at NP=1, no defrag); the cross-NP byte-identity gate (NP=1 vs NP>1) holds because all NPs now use paged consistently. Note: this means NP=1 output BYTES post-T5.7b may differ from pre-T5.7b NP=1 baseline (different code paths can produce different Q4_0 quantization rounding) but cross-NP byte-identity within the build still binds. The verify-production-determinism gate compares within-build NP=1 vs NP>1, not within-build vs prior baseline.

**T5.7c allocator-level paged defrag — caller applies byte moves:**
`llama_paged_kv_allocator::defrag()` returns `std::vector<defrag_move>{old_bid, new_bid}` via greedy pairing of highest-allocated with lowest-free. Method updates pool_owner_, tables_ (replaces old_bid with new_bid at its logical position), rebuilds free_list_ LIFO. Idempotent. Caller responsibility: physically copy each block's bytes from old_bid's region to new_bid's region via the T3.6 generic CUDA Q→Q same-type non-contig cpy kernel. Trace event LLAMA_PAGED_KV_TRACE_DEFRAG_MOVE fires per move when LLAMA_T5_TRACE=1.

**Graph-level defrag integration into `llama_kv_cache_defrag_internal` is deferred to T5.8** — `defrag_thold = -1.0f` default in production (defrag disabled), so the trigger path is not exercised. Forward-looking work, not closure-blocking.

**Kernel nullptr branch removal is T5.8** — the FA per-slot-kv kernel still has `if (block_table != nullptr)` paged-vs-legacy branching, but with T5.7b's always-on `set_block_table` in build-context, the nullptr branch is now unreachable in production. The kernel header at `fattn-per-slot-kv-singlewarp-sm75.cu` line 77 explicitly anticipates this removal at T5.8 closure.

**Why:** PHASE_NSTREAM_KV_PERF.md §"T5.7 — Drop the n_stream==1 contiguous-layout pretence" said "every path is paged" — T5.7b realised that aspiration at the build-context layer. The PHASE doc's find_slot/cell_max language is partially aspirational (cells[] metadata stays as the seq-position bookkeeping layer with the paged allocator running in parallel; full cells[] replacement would be a much larger refactor) — current implementation keeps both as parallel structures and that's the honest scope.

**How to apply:** when extending T5.x work, the n_stream==1 special case has been removed from build-context K/V WRITE/READ/K-shift/Q paths. Do NOT reintroduce it. The remaining n_stream==1 specialisation is in `find_slot` (single-seq scan at `cells[stream*kvps + ...]`) and `cell_max` (which scan cells[] metadata), and in the FA kernel's nullptr branch — all three are honest forward-looking deferrals.

**Gate evidence:**
- verify-production-determinism ACCEPTANCE PASS @ 1455 MHz NP={1,2,4,8} CTX_CHECKPOINTS=3 (after all three sub-steps)
- test-paged-kshift-byte-identity PASS (non-boundary + boundary-crossing, LLAMA_PAGED_KV_LANDED=1)
- test-paged-defrag-preserves-contents PASS (3 moves, byte-identical reads for seqs {0,2,3}, idempotent)
- test-kv-shift-per-stream PASS (LAYER + GRAPH split, NP=4)
- test-fattn-per-slot-kv-dispatch-np-invariance PASS (6144 floats byte-identical)
- test-dflash-np-invariance PASS (drafter_forward, N ∈ {1,2,4,8}, 4 seeds)
- test-kv-block-allocator + test-paged-allocator-determinism PASS (unchanged)

Related: [[project-t5-6-paged-write-read-end-to-end]], [[project-t5-bundle-a-closed]], [[project-t5-1-paged-allocator-landed-dormant]], [[project-t5-probe-falsified-path-c-override]], [[feedback-cuda-cpy-q-q-same-type-pattern]], [[feedback-bake-measurement-env-gates]].
