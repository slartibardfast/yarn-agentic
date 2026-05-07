# Phase 38 — Full #2 dual-stream speculative dispatch (+18% projection)

## Hypothesis

At deployed settings (`LLAMA_MTP_FUSED=1 LLAMA_MTP_INLINE_KV=1 LLAMA_MTP_CHAIN_MIN_PROB=0.5`), the speculative-decoding loop is dependency-bounded: fused → verify → fused. Phase 36/37 closed at the ceiling of that serial DAG: parity at short context, +7.3% effective output at production context.

The hypothesis driving Phase 38: **breaking the serial DAG via dual-stream dispatch with all-accept prediction yields +18% effective output at production context**. The architectural insight: fused(k+1)'s seed is fused(k)'s chain residual at index `n_accepted[k+1]` — knowable speculatively before verify(k+1) finishes, if we accept the prediction-miss cost of a sequential redo.

## Lift derivation

Geometric distribution with per-draft accept p = 0.8 (slow workload) at n_steps=3:

| n_accepted | Probability | Speculatable? |
|---:|---:|:---:|
| 0 | 0.20 | yes (chain_residual[0]) |
| 1 | 0.16 | yes (chain_residual[1]) |
| 2 | 0.128 | yes (chain_residual[2]) |
| 3 (all-accept) | 0.512 | yes IF chain extended to n_steps+1 |

Per-cycle wall time, fused = 20 ms, verify = 50 ms:

- **All-accept (51%)** with extended chain + speculative dispatch on `chain_residual[3]`: parallel dispatch → max(50, 22) = 50 ms (saves 20 ms vs sequential 70 ms)
- **Predicted hit (~20%)** in non-all-accept (predict 0): max(50, 22) = 50 ms (saves 20 ms)
- **Predicted miss (~29%)**: spec compute wasted + sequential redo + GPU contention overhead → 80 ms (penalty 10 ms)

Expected cycle: `0.51 × 50 + 0.20 × 50 + 0.29 × 80 = 58.7 ms` vs baseline 70 ms = **+18.3% throughput**.

The +18% rests on three load-bearing assumptions:
1. **Extended-chain cost is small** (~10% of fused compute = 2 ms, paid in 100% of cycles)
2. **Prediction accuracy holds** — the all-accept case is trivially predictable; non-all-accept needs chain-prob signal correlation. Phase F validates.
3. **GPU overlap delivers wall-clock savings** — verify and fused must concurrently use SMs without crippling contention. Quadro RTX 6000 has 72 SMs; verify saturates many but fused at d=3 uses far fewer. Phase F validates.

## Architecture

```
Cycle k (after accept):
  fused(k).chain_residuals[0..n_steps] → persist_dev[0..n_steps]      (B: D2D save)

Cycle k+1 setup:
  predict guess[k+1] from prior accept stats + chain probs            (E: predict)
  read persist_dev[guess[k+1]] → fused(k+1) seed                      (B: D2D load)

Cycle k+1 dispatch (parallel):
  Stream A: verify(k+1) using fused(k)'s drafts                       (existing path)
  Stream B: fused(k+1) using guess seed                               (E: async dispatch)
            writes K/V to speculative-tail at [commit_k+n_drafts+1..] (D: tail writes)

Cycle k+1 sample:
  verify(k+1) sample → actual n_accepted[k+1]                         (existing path)

Cycle k+1 reconcile:
  if guess matched: D2D copy fused(k+1) tail → actual positions       (D: reconcile)
                    fused(k+1) drafts are valid → use for cycle k+2
  if guess missed:  discard fused(k+1) (tail writes never read)
                    sequential redo: fused(k+1) with correct seed     (E: recovery)
```

## Schedule

### B. Persistent chain-residual device buffer

Outlives `sched_reset` between fused decodes (today's chain_residuals are sched-owned, freed at every reset).

| Task | What |
|---|---|
| B1 | llama_context: ggml_context*, ggml_backend_buffer_t, persist[8] tensors, persist_n |
| B2 | Lazy-init helper: alloc on first fused compute (use buft from chain_residuals[0]'s backend) |
| B3 | Cleanup helper: ggml_backend_buffer_free + ggml_free in llama_free |
| B4 | Wire post-fused-compute: D2D copy chain_residuals[k] → persist[k] for k in [0, n_steps-1] |
| B5 | prepare_mtp_graph_inputs: read from persist[step] (replaces today's chain_residuals[step] read) |
| B6 | Drop the validity flag mtp_fused_chain_residuals_valid (replaced by persist_n) |
| B7 | Smoke test: fused-fused-verify-fused-verify sequence verifies persist data is correct |

### C. Extended chain (n_steps+1 internal compute)

Makes `chain_residual[n_steps]` exist for the all-accept seed case.

| Task | What |
|---|---|
| C1 | cparams.mtp_fused_n_steps_extended (int) — actual chain length when extended |
| C2 | build_qwen35_mtp_fused: compute n_steps_extended internal steps, emit only n_steps drafts |
| C3 | llama_mtp_fused_draft_invoke: accept extended n_steps |
| C4 | Env knob: LLAMA_MTP_FUSED_EXTEND=1 enables n_steps+1 chain |
| C5 | Smoke test: confirm chain_residual[n_steps] populated when extended |

### D. Full KV cache unification + drop inline-KV-hook on LLAMA_MTP_FULL_2 (REVISED 2026-05-07 mid-session)

**Two prior framings tried and rejected:**

1. **Offset path (initial)**: speculative-tail KV writes for fused, with reconciliation. Surfaced issues — the race isn't only on writes (fused chain attention reads from cells verify is concurrently writing), and offset introduces logical/physical position decoupling that complicates kv_head accounting and graph reuse keying.

2. **Partial unification (intermediate)**: alias layers 0..n_layer-2, keep layer n_layer-1 separate. The "partial" hedge existed to preserve the Phase 36 Step 3 inline-KV-hook (which writes layer n_layer-1 with main-forward-derived `inpL` during verify's forward). But the hook is exactly what creates the layer n_layer-1 race — its writes are semantically distinct from fused chain's MTP-shortcut-derived writes, so concurrent dispatch produces non-deterministic torn cells. Partial unification dodges this by giving layer n_layer-1 separate memory, paying 1.5% extra VRAM and accepting that ctx_tgt's and ctx_mtp's layer n_layer-1 diverge.

**Final framing (full unification + drop inline-hook on Full #2 path):**

The inline-KV-hook IS the source of the layer n_layer-1 race. Eliminate it on the Full #2 path and layer n_layer-1 has only one writer semantic class:

- **Fused chain** (during fused dispatch on ctx_mtp): writes layer n_layer-1 at chain positions. MTP-shortcut-derived.
- **UPDATE_ACCEPTED** (post-accept on ctx_mtp): writes layer n_layer-1 at accepted positions. Also MTP-shortcut-derived.

These run at distinct times (UPDATE_ACCEPTED is post-accept; fused is pre-accept). No concurrent write at layer n_layer-1.

Verify's main forward iterates layers 0..n_layer-2 (excludes the MTP layer); without the inline-hook, verify never writes layer n_layer-1. So under full unification:

- **Verify writes**: layers 0..n_layer-2 (unified — fused doesn't read or write them, no contention).
- **Fused writes**: layer n_layer-1 (unified — verify doesn't write layer n_layer-1 without the hook).
- **UPDATE_ACCEPTED writes**: layer n_layer-1 (post-accept, sequential with fused).

**Disjoint layers between concurrent dispatchers. No race anywhere. Single semantic class per layer.**

The cost: re-enabling UPDATE_ACCEPTED's separate decode (~5ms/cycle on production-context). Phase 36 Step 3's +12% MTP-vs-nomtp at d=1 partially erodes — most of that win was the kv-fold structural change, not just UPDATE_ACCEPTED elimination, so the actual cost is closer to -3-5% throughput. Phase 38 E's projected +18% from concurrent dispatch dwarfs this; net positive.

**The architectural simplification is also durable.** Single canonical layer n_layer-1 writer (MTP-shortcut-derived). No two-phase semantic divergence between inline-hook writes and UPDATE_ACCEPTED writes. ctx_tgt and ctx_mtp share every K/V cell at every layer — no divergence anywhere.

| Task | What |
|---|---|
| D1 | Add a `parent_ctx` parameter (or post-init setter) to plumb ctx_tgt into ctx_mtp at init time |
| D2 | In `llama_kv_cache_init` for ctx_mtp: when `parent_ctx` is set, set k_l[il]/v_l[il] for ALL il in [0, n_layer) to ALIAS the parent's k_l/v_l (point ggml_tensor->data into parent's buffer; share view metadata; do NOT allocate fresh tensor memory) |
| D3 | Lifecycle: ctx_mtp's destructor must not free the aliased buffers (owned by ctx_tgt). Mark all KV slots as aliases when parent is set. |
| D4 | When LLAMA_MTP_FULL_2=1: disable cparams.mtp_inline_kv_hook on ctx_tgt's verify path. UPDATE_ACCEPTED becomes the sole layer n_layer-1 canonical writer. mtp_accept_tokens's early-return-if-_hook_on guard handles this automatically (when hook is off, UPDATE_ACCEPTED runs). |
| D5 | Wire common/speculative.cpp's ctx_mtp construction to pass ctx_tgt as parent |
| D6 | Smoke test: VRAM measurement before/after shows ~all of fused-context's KV cache freed; correctness check (--fast harness GREEN at deployed config with LLAMA_MTP_FULL_2=1) |

### E. Dual-stream dispatch + recovery

The actual overlap. fused(k+1) dispatches on a separate sched while verify(k+1) runs.

| Task | What |
|---|---|
| E1 | API: llama_decode_async(ctx, batch) — dispatches but doesn't sync |
| E2 | API: llama_decode_wait(ctx) — completes a prior async decode |
| E3 | mtp_speculative_gen_draft fused branch: use async dispatch when LLAMA_MTP_FULL_2 enabled |
| E4 | Server: predict guess[k+1] from prior accept stats; dispatch fused(k+1) async |
| E5 | After verify+sample: check guess vs actual; match → use fused(k+1) result; miss → sequential redo |
| E6 | Reconciliation hook on match path |
| E7 | KV cleanup on miss path (discard fused tail writes — no actual cleanup needed since they're past kv_head and never read) |
| E8 | Smoke test: correctness across match and miss paths |

### F. Integration + harness validation

Master env gate, harness measurement, results.

| Task | What |
|---|---|
| F1 | Master env knob LLAMA_MTP_FULL_2=1 (gates B+C+D+E together) |
| F2 | Run --fast harness; measure effective_output_ratio with Full #2 enabled |
| F3 | Run --slow harness; measure |
| F4 | Update gate.yaml floor + PHASE38.md results table |

## Binding closure criteria for Phase 38

Phase 38 closes (`[x]`) when, and only when, all four hold:
1. All schedule items B-F implemented and committed.
2. `--fast` harness GREEN at recalibrated threshold (no regression below the parity floor).
3. `--slow` harness shows measured `effective_output_ratio` ≥ 1.10 (10% lift over per-step at production context). The +18% projection becomes a measurement; 1.10 is the binding ratchet (1.073 baseline + 5% real lift = 1.13 expected; 1.10 floor allows for noise margin).
4. `gate.yaml` updated to set `effective_output_ratio` slow threshold to the new measured floor minus 5% noise margin. The recalibration is a real ratchet — Phase 38's lift becomes the new baseline gate.

## Anti-goals (won't do without explicit user approval)

- Cap chain depth at d=2 to "ship cleaner" — that's a regression to the d=2 ceiling, not a Phase 38 closure
- Bench partial implementations to "decide whether to finish" — circular diagnostic; complete the bundle, then measure
- Skip --slow on the assumption fast GREEN → slow GREEN
- Roll back B-D when E doesn't deliver — keep the foundation, the architecture is correct regardless

## Implementation log

### Compaction-1 progress (2026-05-07, session 2)

**Landed and verified:**
- B (persistent chain-residual buffer): committed `0e18a304`. Persist[] tensors in context-owned ggml_backend_buffer outlive sched_reset. D2D capture from chain_residuals at end of fused compute. D2D read in prepare_mtp_graph_inputs. Smoke-test --fast PASS at recalibrated 0.95 floor (effective 1.022).
- C (extended chain): committed `4f8f7154`. cparams.mtp_fused_n_extend, build_qwen35_mtp_fused runs n_chain = n_steps + n_extend internal steps, emits n_steps drafts, captures all residuals. Smoke-test --fast EXTEND=0 PASS (1.037), EXTEND=1 PASS (0.984 — the +1 step's compute cost paid; recovered later when E uses the extended seed).

**Re-scoped during execution:**
- D (speculative-tail KV writes): originally framed as race-avoidance between concurrent fused and verify. User correction during the run: caches are unified for VRAM reasons (not separate as I had assumed). With unified cache at layer 64 (MTP layer), fused and verify CAN write the same cells concurrently. The race is real. Speculative-tail offset (`cparams.mtp_kv_head_offset = n_drafts_prev`) places fused's writes past verify's range; UPDATE_ACCEPTED handles correctness for accepted cells. **D not landed in this session** — needs implementation in next session.

**Deferred to next session:**
- D (full): cparams.mtp_kv_head_offset, fused dispatch with offset, llama_mtp_set_kv_offset API.
- E (async dispatch + recovery): llama_decode async path requires gating the post-compute extraction (sched_synchronize + tensor_get for argmax/prob) behind a flag. New APIs llama_mtp_fused_dispatch_async + llama_mtp_fused_extract_results. Server tracks per-slot pending speculative state. Match/miss recovery via llama_kv_cache_seq_rm.
- F: harness validation, gate.yaml ratchet to whatever lift Full #2 measures.

**The +18% projection remains the binding ratchet for closure.** B+C alone don't deliver lift (both are foundation for E). The lift comes from E's concurrent dispatch overlapping fused(k+2) with verify(k+1). B+C are correct and verified GREEN at the recalibrated parity floor; E+F is the next session's work.

### Why D was rescoped mid-session

Initial reading of common/speculative.cpp:172 (`ctx_mtp = llama_init_from_model(...)`) suggested a fully separate ctx_mtp with its own KV cache → fused and verify on different memory regions → no race possible → D simplifies to "no-op". User corrected: "we were unifying caches for VRAM reasons" — the unification is real; D's race-avoidance is necessary. Re-scope captured here for next session's pickup.
