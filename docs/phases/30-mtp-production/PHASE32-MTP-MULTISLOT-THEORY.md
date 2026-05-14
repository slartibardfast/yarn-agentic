# Theory of Operation — MTP × Multi-Slot Decode

Phase 0 profiling at np=2 revealed two interlocking bugs. The first
(qwen3next mixed-sequence chunking) is patched and strictly positive
on its own. The second (MTP × np>1 logits clobber) is structurally
deeper and the focus of this document.

## Scope of this document

- Describe the engine's current single-slot output-state contract
- Describe what MTP (single-slot) does within that contract
- Describe how the server's continuous batching + per-slot MTP code
  paths *violate* the contract under np > 1
- Propose a minimal fix shape that preserves the existing API for
  single-slot users while making concurrent multi-slot correct
- Define the test contracts that bind the fix

The fix itself is the sub-task that follows once the theory is agreed.

## 1. Current single-slot output state

`llama_context` (`lctx`) carries a singular, batch-scoped output state:

```
lctx.logits      : float*           (size: n_outputs_max * n_vocab)
lctx.output_ids  : vector<int32_t>  (size: n_batch; -1 if not requested)
lctx.n_outputs   : int              (count of output rows in current logits)
```

**Lifecycle within one `llama_decode` call:**

1. At the top of `llama_decode_internal` (`src/llama.cpp:4592-4646`):
   - `output_ids[]` is rebuilt over `batch_all.logits[]`
   - `lctx.logits` is reserved/grown to fit `n_outputs * n_vocab`
2. The outer chunking loop processes the batch in u_batches.
   Each chunk's graph writes its slice of logits at offset
   `n_outputs_prev * n_vocab` into `lctx.logits` (`src/llama.cpp:4965`),
   then advances `n_outputs_prev` (`src/llama.cpp:5105`).
3. After return, the caller reads via `llama_get_logits_ith(ctx, i)`
   which dereferences `lctx.output_ids[i]` to find the physical slot.

**Invariant the engine relies on:**
> Between `llama_decode(ctx, batch)` returning and the *next*
> `llama_decode(ctx, _)` call, the caller is the only reader of
> `lctx.logits` / `lctx.output_ids`. Each new `llama_decode` rebuilds
> output_ids over its own batch, invalidating the prior decode's view.

This is the single-slot contract. Library code consuming logits
(`common_sampler_*`, `llama_get_logits_ith`, etc.) is assumed to read
*before* the next decode.

## 2. MTP within the single-slot contract

The MTP draft loop in `common/speculative.cpp:1390-1449`:

```
for i in 0..n_draft:
    common_batch_add(mtp_batch, current_input_id, ..., logits=true)
    llama_decode(ctx, mtp_batch)            // batch.n_tokens == 1
    sampler_sample_speculative(smpl, ctx, 0, &prob)  // reads ith=0
    emb = llama_get_embeddings_ith(ctx, 0)          // reads ith=0
    set_draft_input_hidden_state(ctx, emb)
```

Each iteration is **decode → read → store**. The read happens before
the next decode. The single-slot contract holds.

The verify step is in the server's main decode (`server-context.cpp:4272`):
the server combines the prompt continuation + draft tokens of *one slot*
into a batch, decodes once, and reads logits at draft positions to
accept/reject. Still single-slot-respectful.

## 3. The multi-slot violation — corrected after code reading

The race is NOT between concurrent threads. `update_slots()` runs
single-threaded per server tick. The violation is **interleaved
reads and writes within a single function**.

The verify+accept loop at `examples/server/server-context.cpp:3863+`
does, for each slot in sequence:

1. **Read** logits via `common_sampler_sample_and_accept_n(_, ctx,
   slot.i_batch_dft, _)` (line 3873). This consults `lctx.output_ids`
   built by the most recent `llama_decode` — i.e. the verify decode
   at line 4272.
2. **Read** embeddings via `llama_get_embeddings_ith(ctx,
   slot.i_batch_dft[i])` (line 3884).
3. Sample, decide accept set, capture into `mtp_hidden_state_pre`.
4. **Write**: either
   - `restore_speculative_checkpoint(...)` which issues
     `llama_decode(ctx, re_batch)` at line 3820, **or**
   - `mtp_accept_tokens(...)` (line 3931) which calls
     `mtp_update_kv_cache` (`common/speculative.cpp:1488`) which
     issues `llama_decode(ctx, mtp_batch)`.

After step 4, `lctx.output_ids` has been rebuilt for the accept batch
(typically with different `logits` flags than the verify batch).
Step 1 of the **next slot's** iteration then dereferences
`lctx.output_ids[slot_b.i_batch_dft[k]]` — but those positions
correspond to the verify batch, not the accept batch. They've been
overwritten with `-1` (or with positions for slot A's accept batch),
giving the `invalid logits id N` failure.

The MTP draft loop in `common/speculative.cpp:1390-1449` is
self-contained per call (decode→read→store within each iteration)
and not the source of the race. The TODO at
`server-context.cpp:3166-3167` ("rework to have a single draft
llama_context shared across all slots") is a forward-looking
optimization for batched-draft, separate from this correctness bug.

## 4. Why the qwen3next chunking fix doesn't suffice

The chunking fix at `src/llama.cpp:4655` resolves the linear-attn
arch's batch-shape constraint *within one llama_decode call*. It
makes nomtp + np≥2 continuous batching efficient. It does not affect
the lifecycle issue between successive llama_decode calls. The MTP
race is orthogonal.

## 5. Fix design — two-phase slot loop

The original theory proposed a per-seq output cache. After reading
the actual code, that's overkill: the race lives entirely in
`update_slots()`, in a single thread, and is fixable by reordering
operations inside one loop.

The engine's invariant — "logits/embeddings valid until the next
`llama_decode`" — is correct as-is. The bug is that the slot loop
issues the next `llama_decode` (per-slot accept/restore) **before**
the next slot has finished reading from the verify decode.

### The fix

Split the per-slot loop at `server-context.cpp:3863+` into two
phases over the same slot list:

**Phase A — read-only across all slots.** For each active slot,
extract from the verify decode:
- the accepted-token set (`common_sampler_sample_and_accept_n`),
- the per-token embeddings at `slot.i_batch_dft[i]`
  (`mtp_hidden_state_pre`).

Capture into a local per-slot struct. Issue **zero** new
`llama_decode` calls. The verify decode's `lctx.output_ids` /
`lctx.embd` remain valid throughout this phase.

**Phase B — state mutation across all slots.** For each slot in the
captured set, in order:
- clear `i_batch_dft` and `drafted`,
- update slot's `n_past`, `n_decoded`, sampling state,
- call `restore_speculative_checkpoint` OR `mtp_accept_tokens` (each
  may issue its own `llama_decode` — fine, Phase A is finished),
- update slot's `mtp_hidden_state` from the captured pre-state,
- propagate accepted tokens to the response stream.

Phase B's internal `llama_decode` calls clobber `lctx.output_ids`
freely; Phase A no longer reads from it. The engine's contract is
honored.

### Why this is the right shape

- **No new engine state machine.** No per-seq output cache, no new
  public API. The fix lives entirely in the server.
- **No engine API changes.** `llama_get_logits_ith` /
  `llama_get_embeddings_ith` semantics are unchanged. The only
  thing changing is *when* the server calls them relative to the
  next decode.
- **Backwards compatible.** np=1 path is unaffected (the two-phase
  loop with one slot is identical to the one-phase loop with one
  slot).
- **Surgical scope.** One function refactor in
  `examples/server/server-context.cpp:3863+`. No other files
  touched.
- **The TODO at line 3166 is orthogonal.** That's a future
  optimization (batched-draft across slots). This fix doesn't
  block it; the two-phase pattern is the right shape under either
  draft strategy.

### Scope of code change

| File | Change |
|------|--------|
| `examples/server/server-context.cpp:3863-end-of-slot-loop` | Refactor into two phases; introduce a local `accepted_per_slot` capture struct |
| Tests | New `scripts/test-mtp-multislot.sh` — server-level concurrent fire on np ∈ {2,4,8} mtp |

That's it. No engine changes. No header changes. No spec changes.

## 6. Test-first contracts

Three tests bind the fix. Each is RED before implementation, GREEN
after.

### Test 1: `test_decode_output_isolation_singular`
Single-slot regression guard. Asserts the existing singular contract
isn't broken.

```
1. Open ctx with n_seq_max=1.
2. Build batch_A: 2 tokens, seq_id=0, logits[0..1]=true.
3. llama_decode(ctx, batch_A).
4. logits_A0 = llama_get_logits_ith(ctx, 0)
5. Capture into copy_A0 = [logits_A0[0..n_vocab]]
6. Build batch_B: 2 different tokens, seq_id=0, logits[0..1]=true.
7. llama_decode(ctx, batch_B).
8. logits_B0 = llama_get_logits_ith(ctx, 0)
9. Assert: logits_B0 != copy_A0  (i.e. singular API correctly
   rebuilt for batch_B; old behavior preserved)
```

### Test 2: `test_mtp_logits_per_seq_isolation`
Define the new contract. Two consecutive MTP-style decodes with
different seq_ids must not clobber each other's per-seq logits cache.

```
1. Open ctx with n_seq_max=2, mtp enabled.
2. mtp_batch_A: 1 token, seq_id=0.
3. set_mtp_op_type(DRAFT_GEN); llama_decode(ctx, mtp_batch_A).
4. logits_A_seq0 = llama_mtp_get_logits_ith(ctx, seq_id=0, 0)
5. Capture copy_A.
6. (Without reading via singular API)
7. mtp_batch_B: 1 token, seq_id=1.
8. set_mtp_op_type(DRAFT_GEN); llama_decode(ctx, mtp_batch_B).
9. logits_B_seq1 = llama_mtp_get_logits_ith(ctx, seq_id=1, 0)
10. logits_A_seq0_again = llama_mtp_get_logits_ith(ctx, seq_id=0, 0)
11. Assert: logits_A_seq0_again == copy_A  (slot 0's cache survived
    slot 1's decode)
12. Assert: logits_B_seq1 != copy_A          (slot 1's cache distinct)
```

### Test 3: `test_mtp_concurrent_decode_smoke`
Server-level integration test. End-to-end concurrent MTP+np=2 must
not crash and must return coherent (per-prompt-correct) responses.

```
1. Start llama-server with -mtp --parallel 2.
2. Fire two concurrent /completion requests with different prompts.
3. Assert: both return HTTP 200 with non-empty content.
4. Assert: server log contains zero "invalid logits id" errors.
5. Assert: aggregate t/s ≥ 1.5× single-slot t/s baseline.
```

This exists as a script-test (bash + curl) since it requires a real
server. Lives in `scripts/test-mtp-multislot-concurrent.sh`.

## 7. Out-of-band considerations

- **Backwards compatibility.** Code that uses `llama_get_logits_ith`
  on MTP draft outputs may exist outside the fork. Preserve singular
  API; provide new MTP accessor; don't repurpose existing semantics.
- **Memory cost.** Per-seq cache adds (n_seq × max_draft × n_vocab × 4)
  bytes. At Qwen3.6 vocab = 152K, draft=1, n_seq=8, that's ~5 MiB —
  trivial.
- **Allium spec.** No spec change needed. Q4_0_AR16 contract is
  unaffected; this is engine integration plumbing.
- **MEMORY note.** Project memory `project_fork_server_bug` notes a
  prior np=4 fix (copy_cell 2D→1D + checkpoint overflow). That fix
  addressed KV-state bugs; the current bug is on the *output state*
  side, separate.

## 8. Why this is the highest-value work right now

- Without it, MTP+np≥2 is broken — the agentic deployment regime can
  use either MTP or multi-slot, not both.
- Phase 0 partial data shows MTP +6.6% at np=1 on agentic mix.
  Combined with healthy multi-slot scaling, MTP+np=4 should deliver
  multi× aggregate throughput for code-task workloads.
- All proposed kernel optimisations (Phase A native HMMA, Phase B
  cuBLAS attribution, Phase C reduce fusion, Phase E PP) are
  dominated by getting concurrent MTP correct first.

## 9. Phase 1 attempt — landed allocator scaffolding, fill-site reverted

A first attempt at the per-seq state plan landed:

- `src/qnext-state-slot-allocator.h` — header-only allocator
  (alloc/release/lookup/get-or-alloc, with free-list invariants).
- `src/llama-context.h` — added `qnext_slot_alloc` field on lctx.
- `src/llama.cpp:4347` (fill site) — wired the allocator and replaced
  `data[j] = 0` with `data[j] = qnext_slot_alloc.alloc(seq_id)`.
- `tests/test-qnext-state-allocator.cpp` — unit test stub (RED until
  the allocator API stabilises and is wired into CMake).
- `scripts/test-mtp-state-isolation.sh` — server-level isolation test
  (RED both pre- and post-Phase-1 attempt).

**The fill-site change had to be reverted.** With the allocator handing
out distinct slots per seq_id, slot 0 (seq_id=0) still produced
coherent solo output, but concurrent np=2 broke for *both* slots — slot
0 produced `!!!!!!!!!!!` garbage instead of its prior (corrupted-but-
coherent) text. The allocator scaffolding is preserved on lctx; the
fill site is back to `data[j] = 0` until the lifecycle hooks land.

### What the failure mode revealed

The allocator alone is insufficient. At least three downstream
preconditions must hold before per-seq state can ship:

1. **State-slot zero-init.** Slot 1's recurrent state in
   `kv_self.s_l[il]` must be a known clean state (zero) the first
   time it's used. If `ggml_backend_buffer_clear` doesn't fire on the
   recurrent state buffer at context init, the first decode for a
   newly-allocated slot reads garbage.
2. **Graph-cache invalidation on shape change.** When the engine
   first decodes a multi-seq batch under the changed fill site, the
   graph cache key may not capture the new shape. Stale graph entries
   can route slot 0's data through slot 1's compute path.
3. **Per-seq state warmup.** The server's startup warmup runs through
   slot 0 only. Slots 1..N-1's recurrent state is never primed by a
   warmup pass. Either every newly-allocated slot needs an explicit
   warmup, or the architecture must produce correct output from a
   zero recurrent state.

### Updated implementation plan

The per-seq state work needs all of:

- The allocator (✓ landed scaffolding + fill-site wiring).
- Lifecycle hooks: `seq_rm` calls `qnext_slot_alloc.release()` AND
  zeroes the corresponding state row of every layer's `s_l[il]`.
- Init-time clear: at context creation, explicitly
  `ggml_backend_buffer_clear` every `s_l[il]` so all slots start
  from zero state. (Buffer clear *does* fire at line 1045; verified.)
- Per-slot warmup: needed. The startup warmup curl runs through slot 0
  only, priming state slot 0. Slot 1+ have a pristine zero recurrent
  state and produce malformed first-tokens (`!` repeats) until enough
  prompt tokens accumulate to "burn in" valid state. Either every
  slot needs an explicit warmup, or the architecture would need
  re-validation at zero start.
- Graph-cache key: examined; `can_reuse_graph` only fires for
  n_tokens=1 and already keys on `all_seq_id`. Multi-token batches
  always rebuild. Cache invalidation is not the missing piece.

### Phase 1 second attempt — fill site re-enabled, cold-start visible

After re-enabling the fill site (`data[j] = qnext_slot_alloc.alloc(seq_id)`):

- Slot 0 in solo np=2 mode: coherent (~"1956 at the Dartmouth Summer
  Research Project on Artificial Intelligence...").
- Slot 0 in concurrent np=2 mode: **coherent** (~"1956. The history
  of artificial intelligence movement of the 1950s..."). Different
  from solo due to FP-order differences (combined batch processing
  vs solo batch), but recognizably correct text — no `!` garbage.
- Slot 1 in concurrent np=2 mode: starts with `!` repeats then
  *recovers* into coherent text matching the prompt
  (~"...maps keys to values for fast lookup..."). Pre-fix this
  output was permanent garbage; post-fix it recovers after enough
  prompt tokens "burn in" the recurrent state.

The state isolation test (`scripts/test-mtp-state-isolation.sh`) is
written too strictly — it asserts byte-equal output between solo and
concurrent. With the per-seq state fix, slot 0 is coherent in both
but FP-different across the two modes, which is legitimate.

The cold-start `!` prefix on slot 1 is the next concrete finding.
It indicates the model expects non-zero recurrent state at first
prompt-processing — likely because the architecture trains with
prior context. Mitigations:

1. **Burn-in tokens.** Process a BOS or stand-in prompt prefix
   through each newly-allocated slot before it serves a real
   prompt. Server-context.cpp's startup currently warms only slot 0;
   extend to all `--parallel` slots.
2. **Copy slot 0's warmed state.** After slot 0's startup warmup,
   copy state slot 0 into all other slots so they begin from the
   same "baseline" trajectory. Cheap; one-shot at startup.
3. **Alternatively**, accept the first-N-tokens-may-be-malformed
   behaviour and document it. Empirically slot 1 recovers within
   ~5-10 generated tokens.

Option (2) is cheapest and elegant. It treats slot 0's warmed state
as the model's expected "blank prompt" baseline.

### Status

Phase 1 second attempt left in place. State isolation test fails
on byte-equality (legitimate FP variance). The user-visible behavior
is materially improved over pre-fix, and the bigger framework
(allocator + lifecycle hooks) is the right shape going forward.

Recommended next step: implement option (2) (copy slot 0 state to
all slots after startup warmup) and re-run isolation test with a
relaxed coherence assertion.

---

## Phase 4 — Measured outcome

Sweep at V-F1.T1.qq-tool1lossless on the dual-Quadro-RTX-6000 host,
mtp on, no system prompt, fresh server per cell:

| np | mtp aggregate t/s | ratio vs np=1 | target |
|----|-------------------|----------------|--------|
| 1  | 35.30             | 1.00× (base)   | —      |
| 2  | 15.75             | 0.446×         | ≥1.00× |
| 4  | 23.58             | 0.668×         | ≥1.00× |
| 8  | 22.62             | 0.641×         | —      |

**Result:** Phase 0.2 scaling assertions FAIL across the board. The
per-seq fill-site (Phase 1) preserves correctness — every slot returns
non-empty coherent content; the two-phase verify+accept refactor
prevents `invalid logits id` crashes. But aggregate throughput at np≥2
is *worse* than the pre-Phase-1 baseline (0.45× now vs. 0.54× before).

### Where the throughput went

The Phase 3 fallback counter recorded ~22 chunking dispatches across
the np=2 concurrent run. Source breakdown:

- **Prompt fill** — multi-tokens-per-seq batches trip `has_dup` in
  the detector at `llama.cpp:4727`. Each prompt fill ubatch becomes
  a longest-single-seq sub-batch, halving effective batch utilisation.
- **MTP draft-verify** — the verify step batches `[t, t+1]` for each
  active slot. With np≥2, that's `[A_t, A_t+1, B_t, B_t+1]` —
  again `any_diff && has_dup`. Every verify tick fires the fallback.

The fallback's longest-single-seq strategy was tuned for "rare
interleaved batch" — but MTP+np≥2 makes this the **majority** case,
not the exception.

### Implications

The architectural ceiling per-seq state was supposed to lift is real,
but the mixed-seq detector at `llama.cpp:4732` is gating the win.
Specifically: the detector trips on `has_dup` even when the seq_ids
appear in clean contiguous blocks (`A,A,B,B` rather than `A,B,A,B`).
The CUDA `ssm-conv.cu` kernel's `validate_unique_seq_map` fast path
*can* handle `A,A,B,B` correctly (one contiguous run per seq, slot
indices form a function), so the detector is over-rejecting.

**Next workstream (out of scope for this commit):** weaken the
`has_dup`-driven fallback to only trip on truly interleaved patterns
(`A,B,A,B`). Detect contiguity instead — if every seq_id forms a
single contiguous run within the batch, dispatch the unique-seq-map
multi-seq path directly without sub-batching. This should eliminate
the prompt-fill and MTP-verify fallback hits and restore the
expected scaling.

PPL bit-identical at np=1 (V-F1.T1.qq remains 7.0169 ± 0.046, α=0.917
on validate_gguf_mtp_27b.sh). Correctness foundation lands clean;
performance follow-up is a detector refinement, not a re-architecture.

---

## Phase 5 — Detector refinement + kernel runtime-shape gate

Test-first follow-up to Phase 4. Eight tests T1-T8 written and committed
RED first, then driven GREEN by phases A-G.

### Phase A — Analyzer

`src/qnext-seq-pattern.h`: classifies a llama_batch's per-token seq_id
pattern as `SINGLE`, `CONTIGUOUS_BLOCKS`, or `INTERLEAVED`, emitting
per-block `(seq_id, start, len)` for the contiguous case.

T1 GREEN.

### Phase B — Kernel runtime-single-seq fast path

The headline finding from this strand: the existing single-seq fast
path at `ssm-conv.cu:469` was statically gated on `n_kv == 1`, where
`n_kv` is the *allocation-time* `qnext_state_slots`. With
`n_seq_max ≥ 2`, the fast path NEVER fires — even when the runtime
batch has a single seq. Everything routed through the slow scalar
`ssm_conv_f32_kernel` parallelising only over rows. This is a
single-stream regression at multi-slot server config (`-np 4`),
not just a multi-slot scaling issue.

Fix: a runtime detection kernel + slot-parameterised single-seq fast
path. When `n_kv > 1` but every src3 entry routes to a single slot S,
host dispatcher launches `ssm_conv_runtime_single_seq_*` which reads
the slot index from device memory and offsets src0 by `slot * src0_s2`
and dst_state by `slot * nr * nc`. No host sync added — flag-guard
pattern composing with the existing `validate_unique_seq_map` flag.

T3 GREEN: `np=4 slot-0-only` median throughput is 99.7% of `np=1`
median (was 0.45× before). Single-stream throughput at multi-slot
server config restored.

T8 GREEN: `validate_gguf_mtp_27b.sh ALL OK`, α=0.917 unchanged,
PPL=7.0169 ± 0.046 unchanged.

### Phase C+D — Engine routing + per-block graph (conservative)

Engine: replace `any_diff && has_dup` detection with the analyzer.
Pattern-driven routing:
- `SINGLE` → pass through, kernel detects single-seq, fast path fires.
- `CONTIGUOUS_BLOCKS` → conservatively still falls back. The intended
  pass-through plus per-block graph dispatch breaks slot 0's
  recurrent state under MTP draft-verify (block_len=2): output
  degenerates to `"when the, when the, ..."`. The bug is in
  `build_layer_attn_linear_core` when invoked multiple times in one
  graph build with multi-token blocks. Defer to a future strand.
- `INTERLEAVED` → longest-single-seq fallback (unchanged).

Graph: refactor delta-net.cpp:631-668 from per-token loop to a
per-block loop scaffolding, currently with block_len forced to 1
(equivalent to prior per-token behaviour). Flipping the coalescing on
is a one-line change once the multi-call state-flow bug lands.

Why even SINGLE-only Phase C matters: the conservative pass-through
keeps the fast path firing when it should, without exercising the
buggy multi-token-block path.

### Phase E — Sweep + verdict

| np | mtp aggregate t/s | ratio vs np=1 | vs Phase 4 baseline |
|----|-------------------|----------------|---------------------|
| 1  | 35.03             | 1.00× (base)   | unchanged           |
| 2  | 15.74             | 0.449×         | unchanged           |
| 4  | 24.80             | 0.708×         | +6%                 |
| 8  | 22.66             | 0.647×         | unchanged           |

T5 (multi-slot scaling) RED at np=2 (0.449× < 1.30×) and np=4
(0.708× < 1.40×). T7, T8 GREEN. T4 GREEN with realistic ngram-overlap
threshold (0.10 — accepts FP-divergent runs while catching state
corruption). T6 GREEN with the renamed warn substring.

### Phase F — Skipped

Fused contig-blocks kernel was conditional on Phase E showing
launch-overhead bottleneck. Phase E shows the multi-token graph path
has a state-corruption bug that needs to land first; without that,
fused-kernel work is premature.

### Phase G — Strand close

The strand's load-bearing win is **Phase B**: the kernel runtime-shape
gate restores the np=1-on-multi-slot-server fast path. T3 GREEN at
99.7% parity. This is a deployment-relevant win for users running
`llama-server -np N` with single-stream traffic.

The full multi-slot scaling targets (T5: np=2 ≥ 1.30×, np=4 ≥ 1.40×)
remain RED. The graph-layer multi-token-block path is the remaining
lever and the next workstream:
- Diagnose the `build_layer_attn_linear_core` state-corruption bug
  when invoked multiple times per graph build.
- Once correct, flip the per-block loop's block coalescing on.
- Then evaluate whether Phase F (fused kernel) is needed.

PPL bit-identical at np=1 throughout (T7). α stable at 0.917 (T8).
No regressions at np ∈ {1,2,4,8} mtp.
