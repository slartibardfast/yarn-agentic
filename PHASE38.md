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

### D. Speculative-tail KV writes + reconciliation

Eliminates the verify/fused KV cell race for accepted positions by giving fused its own write region.

| Task | What |
|---|---|
| D1 | cparams.mtp_kv_head_offset (int) — base position for fused's KV writes |
| D2 | build_qwen35_mtp_fused: use kv_head_offset + k as KV write position (kv_head_offset replaces "+k" today) |
| D3 | API: llama_mtp_set_kv_offset(ctx, offset) |
| D4 | Reconciliation function: per-layer D2D copy from tail [commit+n_drafts+1..] to actual [commit+n_accepted+1..] |
| D5 | Cell allocation: ensure tail region [commit+n_drafts+1..commit+n_drafts+n_steps] is reserved before dispatch |
| D6 | Smoke test: reconciled K/V matches sequential-baseline K/V byte-identical for accepted positions |

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

(Appended as work lands. Per CLAUDE.md §5: every PHASE38 edit commits + pushes immediately.)
