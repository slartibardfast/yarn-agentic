# Phase 36 Implementation Plan: Multi-GPU MTP Draft Throughput

## Context

MTP speculative decoding on Qwen3.6 27B with 2x RTX 6000 (`--split-mode graph --tensor-split 1,1`) shows d=1 as optimal (+5% over baseline) while d>=3 regresses even after per-step checkpoint fix. Root cause: draft is on the critical path -- each draft step adds ~11 ms (build + alloc + compute + host bounce). Five drafts add ~55 ms to an ~85 ms cycle.

**Design invariant:** Draft generation must be off the critical path. If draft cost is zero, any acceptance rate > 0% is profitable. No break-even threshold, no adaptive truncation.

**Goal:** d=5 throughput > d=1 throughput on 2-GPU graph split. Conservative target: 2x baseline (~67 t/s). Stretch: match upstream single-GPU 2.5x (~84 t/s).

Phase 36 design doc: `PHASE36-MULTI-GPU-MTP-DRAFT-THROUGHPUT.md`

## Ground state (verified 2026-05-06)

**ik_llama.cpp repo:**
- Branch: `phase36-mtp-throughput` at `5bd0d740` (forked from `slartibardfast/phase33-concat-probe`)
- Fork is 50 commits ahead of upstream `origin/main` -- includes Q4_0_AR16 (type 159), per-step checkpoint, DeltaNet state, cuda graph cache
- Two builds exist: `build/` (standard Release), `build-profile/` (IK_PRINT_TIMING=1). Both from `5bd0d740`, different MD5s (confirmed)
- `build-debug/` does not have a server binary

**Uncommitted changes (carry forward):**
- `common/speculative.cpp`: per-step component timing (`mtp_draft_step_decode[i]`, `mtp_draft_step_emb_d2h[i]`, `mtp_draft_step_hidden_h2d[i]`) behind existing `LLAMA_PROFILE_DECODE` env gate (lines 1419-1456)
- `src/llama.cpp`: `can_reuse_graph` HIT/MISS logging behind `#if IK_PRINT_TIMING` (lines 4856, 4890)
- `ggml/src/gguf.cpp`: untracked file -- needs cleanup (delete or .gitignore)

**Already committed RED test scaffolding:**
- `tests/mtp-fused/` (8 tests) -- commit `b34e661b`. API: `llama_mtp_fused_draft_invoke()`, `llama_mtp_fused_result`, `LLAMA_MTP_OP_DRAFT_GEN_FUSED`
- `tests/mtp-ubatch-hook/` (5 cpp + 3 shell) -- commit `b34e661b`. API: `llama_mtp_kv_pos_max()`, `llama_main_graph_h_pre_norm()`, `llama_mtp_hook_fire_count()`, `llama_mtp_decode_count()`
- `tests/mtp-verify-accept/` (9 cpp + 2 shell) -- commit `58009a77`. API: `LLAMA_ACCEPT_MODE_ARGMAX_MATCH`, `llama_accept_decision`, `llama_mtp_accept_verify()`

**yarn-agentic repo:**
- Branch: `phase32-q4_0_ar16-integration`
- Submodule pin shows diff from `5bd0d740` to `b86670a` (stale pointer from previous session -- need to reset to `5bd0d740`)
- Profile script exists: `scripts/profile-mtp-draft-cycle.sh` (NOT committed)
- Existing instrumentation: `#define IK_PRINT_TIMING 0` at `src/llama.cpp:32`, `LLAMA_PROFILE_DECODE` env gate at `common/speculative.cpp:1397`

**Key source locations (at 5bd0d740):**
- `src/graphs/build_qwen35.cpp:145-246` -- `build_qwen35_mtp()` single MTP step builder
- `common/speculative.cpp:1356-1465` -- `mtp_speculative_gen_draft()` per-step draft loop
- `src/llama.cpp:560-571` -- `can_reuse_graph()` 10 conditions (condition 7 kills reuse between drafts)
- `src/llama.cpp:4184-4195` -- `prepare_mtp_graph_inputs()` host bounce path
- `ggml/src/ggml-cuda/common.cuh:850` -- `cudaStream_t streams[MAX_DEVICES][8]`, only stream 0 used
- `include/llama.h:287-290` -- MTP_OP enum: NONE=0, WARMUP=1, UPDATE_ACCEPTED=2, DRAFT_GEN=3

## Dependency graph

```
Step 0 (profile) --> Step 1 (fused cgraph) --> Step 2 (async pipeline)
                                                     ^
                     Step 3 (kill UPDATE_ACCEPTED) ---+
                     Step 4 (device-resident relay) --+
                     Step 5 (KQ_mask bucketing) -- independent fallback path
                     Step 6 (MTP head precision) -- independent measurement
```

Steps 3, 4, 5, 6 are independent of each other but Step 2 benefits from Step 3 (smaller accept tail) and Step 4 (faster draft start).

---

## Step 0: Profile draft cycle [~] (instrumented + first profile landed; gate failed; ordering revised)

Instrumentation and measurement. Profile script and per-step timing landed at submodule `d15dd96` (phase36-mtp-throughput) on 2026-05-06.

### 0.1 Instrumentation [x] (commit `d15dd96`)

Single submodule commit on `phase36-mtp-throughput`:
- `common/speculative.cpp`: per-draft-step `decode/emb_d2h/hidden_h2d` timers, env-gated by `LLAMA_PROFILE_DECODE`. Output format: `<label>  <microseconds>  step=<i>` — matches the awk pattern in `scripts/profile-mtp-draft-cycle.sh`.
- `src/llama.cpp`: `can_reuse_graph` HIT/MISS counter at the call site, with per-condition reason histogram (1..10). Gated by `IK_PRINT_TIMING`.
- `src/llama.cpp`: `#ifndef IK_PRINT_TIMING` guard so the script's CLI `-DIK_PRINT_TIMING=1` actually takes effect (the unconditional `#define ... 0` was silently overriding it; existing instrumentation never fired before this fix).

### 0.2 First profile pass [x] (logs in `data/profile-step0/`)

Ran `scripts/profile-mtp-draft-cycle.sh` at d=0,1,3,5 (200 tokens, temperature=0) on the production AutoRound quant with the production server stopped. Full results table now lives in `PHASE36-MULTI-GPU-MTP-DRAFT-THROUGHPUT.md` under "Step 0 results."

### Performance gate — RESULT: FAILED, ordering revised

The gate ("≥ 40% of per-step cost is build+alloc") **failed** at the populated draft steps:

| step | n   | build+alloc / decode |
|-----:|----:|---------------------:|
| 0    | 116 | 16.0%                |
| 1    |  59 | 18.7%                |
| 2    |   2 | 22.1%                |
| 3    |   1 | 43.8% (single-call noise) |
| 4    |   1 | 43.7% (single-call noise) |

Compute dominates draft-step cost, not scheduling. The gate forced a re-evaluation, which surfaced three more findings that changed the plan:

1. **Effective draft depth at d=5 = 1.54** (179 step calls / 116 cycles). The early-exit on `prob < p_min` clamps draft depth aggressively — the d=5 mode is structurally close to "d=1 plus extra rejected drafts." The original "5 sequential drafts × 11 ms = 55 ms on critical path" framing was wrong on both factors.
2. **`can_reuse_graph` MISS reasons:** r1=59% (no_prev), r2=24% (multi_token), r9=17% (mtp_op_changed). **r7 (n_kv changed) = 0%.** Step 5 (KQ_mask bucketing) was scoped specifically to fix r7 — its premise is invalidated.
3. **Hidden-state host bounce:** measured `emb_d2h` = 2 µs, `hidden_h2d` = 0 µs per step. Step 4's premise of "1.5 ms × 5 = 7.5 ms saved per cycle" is invalidated by 3 orders of magnitude.

### Revised step ordering

Steps 4 and 5 are demoted (premise invalidated by data). Steps 1–3 remain but with smaller projected gains. See `PHASE36-MULTI-GPU-MTP-DRAFT-THROUGHPUT.md` "Step 0 results → What this changes" for the full table.

New priority for the work that follows:

1. **Step 3** (kill UPDATE_ACCEPTED) — fastest path to a measurable win; the actual update_accepted cost was not separated in the first profile pass and needs Step 3 telemetry to size precisely
2. **Step 1 + Step 2 together** — the d=5 path to 1.5–2× baseline (not 4×+)
3. **Step 6** (MTP head F16) — multiplicative with the above; needs a precision audit of the production AutoRound quant first
4. **Step 5** (bucketing): demoted — revisit only if a new measurement shows r7 firing in a different operating regime
5. **Step 4** (D2D relay): demoted — revisit only if profiling under a different workload shows the bounce regrowing

---

## Step 1: Fused multi-draft cgraph (6 commits)

Single `ggml_cgraph` chaining N MTP draft steps. One build + alloc + compute instead of N.

### 1.0 RED test stubs -- DONE (commit b34e661b)

Already committed at `b34e661b`. 8 RED test files in `tests/mtp-fused/`, wired into `tests/CMakeLists.txt` behind `if(GGML_CUDA)`.

Committed tests (from `tests/mtp-fused/README.md`):

| File | Spec invariant |
|------|----------------|
| `test-mtp-fused-symbols.cpp` | `llama_mtp_fused_draft_invoke()` and `LLAMA_MTP_OP_DRAFT_GEN_FUSED` compile |
| `test-mtp-fused-step-count-bound.cpp` | `StepCountConformsToBound`, `StepCountWithinBound` |
| `test-mtp-fused-single-compute.cpp` | `SingleGraphCompute`, `NoSyncBetweenSteps` |
| `test-mtp-fused-argmax-correctness.cpp` | `ArgmaxCorrectnessIntrinsic`, `NTokensProducedInOrder` |
| `test-mtp-fused-determinism.cpp` | `DeterminismUnderArgmax` |
| `test-mtp-fused-kv-coverage.cpp` | `KvWritesAreExactlyNSteps`, `KvWrittenAtPositionsPThroughPPlusN` |
| `test-mtp-fused-kv-chop-rewrite.cpp` | `KvChopRewriteIsIdempotent` |
| `test-mtp-fused-prob-populated.cpp` | `ProbabilityFieldPopulated` |

**API names from the committed tests** (these are authoritative -- implementation must match):
- `LLAMA_MTP_OP_DRAFT_GEN_FUSED = 4` (enum extension)
- `llama_mtp_fused_draft_invoke(ctx, seed_token, seed_hidden, n_steps, &out)` (not `llama_mtp_fused_draft()`)
- `struct llama_mtp_fused_result { n_steps, tokens[], probs[] }`
- `llama_mtp_fused_last_compute_count(ctx)` (observability)

### 1.1 API surface (commit 1)

Add to `include/llama.h`:
- `LLAMA_MTP_OP_DRAFT_GEN_FUSED = 4` in enum
- `struct llama_mtp_fused_result`
- `llama_mtp_fused_draft_invoke()` declaration
- `llama_mtp_fused_last_compute_count()` declaration

Add stub implementation in `src/llama.cpp` that returns -1.

**Files:** `include/llama.h`, `src/llama.cpp`
**RED->GREEN:** `test-mtp-fused-symbols.cpp` passes (compiles, links).

### 1.2 Fused graph builder (commit 2)

Implement `build_qwen35_mtp_fused(n_draft)` in `src/graphs/build_qwen35.cpp`. This is the core graph construction:

- Takes existing `build_qwen35_mtp()` (line 145) as the template for a single step
- Unrolls N steps in a single cgraph
- Step 0: uses `inp_mtp_states` (hidden state from verify)
- Step k>0: `ggml_argmax(step_{k-1}_logits)` -> `ggml_get_rows(tok_embd, argmax_{k-1})`
- Each step gets its own KQ_mask tensor (pre-sized for `n_kv_start + k`)
- DeltaNet state: sequential in-place update through steps
- Each step writes per-step checkpoint tensors
- Output nodes: N `argmax` results + N `softmax` probability scalars

Pre-allocate N KV positions via `llama_kv_cache_find_slot()` before graph build.

**Files:** `src/graphs/build_qwen35.cpp`, `src/llama-build-context.h` (declare builder)
**RED->GREEN:** `test-mtp-fused-kv-coverage.cpp`, `test-mtp-fused-argmax-correctness.cpp` pass.

### 1.3 Fused evaluation runtime (commit 3)

Wire `MTP_OP_DRAFT_GEN_FUSED` into `llama_decode_internal()` (llama.cpp:4197+). When `mtp_op_type == MTP_OP_DRAFT_GEN_FUSED`:

- Call fused graph builder instead of single-step builder
- Single `ggml_backend_sched_graph_compute()` call
- Extract N argmax results + N probabilities from output nodes
- Single D2H of ~40 bytes (N x {token_id, prob})

Implement `llama_mtp_fused_draft_invoke()` body: set op type, prepare inputs, decode, extract results into `llama_mtp_fused_result`, clean up KV on early exit.

**Files:** `src/llama.cpp`
**RED->GREEN:** `test-mtp-fused-single-compute.cpp`, `test-mtp-fused-kv-coverage.cpp`, `test-mtp-fused-prob-populated.cpp` pass.

### 1.4 Wire into speculative.cpp (commit 4)

In `mtp_speculative_gen_draft()` (speculative.cpp:1356): detect trivial sampler (greedy/argmax). If trivial and `n_draft > 1`, call `llama_mtp_fused_draft_invoke()` instead of the per-step loop.

Fall back to per-step loop for non-trivial samplers (temperature, top-p, grammar).

**Files:** `common/speculative.cpp`
**RED->GREEN:** `test-mtp-fused-determinism.cpp` passes -- fused path produces identical tokens to per-step path on same input.

### 1.5 Qwen3.5 MoE variant (commit 5)

Implement `build_qwen35moe_mtp_fused(n_draft)` for the 35B-A3B architecture. Same structure as 1.2 but routes through MoE FFN layers.

**Files:** `src/graphs/build_qwen35.cpp`

### 1.6 Integration test (commit 6)

Shell script `tests/mtp-fused/test-fused-e2e.sh`: run `llama-server` with `--mtp --draft 5`, send 3 diverse prompts via curl, verify:
- Coherent output (not garbled)
- Acceptance rate within 5% of per-step baseline
- Token generation logged as fused (grep for `MTP_OP_DRAFT_GEN_FUSED` in debug log)

**Commit:** Shell test.

### Performance gate -- Step 1

| Metric | Current (d=5) | Target | Fail |
|--------|--------------|--------|------|
| Per-draft scheduling cost | 5 x (build+alloc) | 1 x (build+alloc) | >= 2x calls |
| d=5 throughput | 32.4 t/s | >= 40 t/s | < 36 t/s |
| d=5 acceptance | 59% | 59% +/- 2% | > 5% regression |
| d=1 throughput | 35.3 t/s | 35.3 t/s +/- 1% | > 3% regression |
| Baseline (no MTP) | 33.5 t/s | 33.5 t/s +/- 1% | > 3% regression |

---

## Step 2: Async dual-stream pipeline (5 commits)

Draft runs on a low-priority CUDA stream, overlapped with accept tail. This is the core architectural change.

### 2.0 RED test stubs (commit 1)

Create `tests/mtp-pipeline/` with RED tests:

| File | Validates |
|------|-----------|
| `test-pipeline-stream-separation.cpp` | Draft and accept use different CUDA streams |
| `test-pipeline-overlap.cpp` | Draft kernels overlap temporally with accept tail (nsys-parseable output) |
| `test-pipeline-priority.cpp` | Draft stream is lower priority than accept stream |
| `test-pipeline-correctness.cpp` | Pipelined path produces identical output to sequential |
| `test-pipeline-sync.cpp` | Both streams sync before verify_{k+1} starts |

**Commit:** RED tests + CMakeLists wiring. All fail.

### 2.1 Draft CUDA stream creation (commit 2)

In `ggml-cuda.cu` backend context init, create a low-priority stream per device for draft work:

```cpp
cudaStreamCreateWithPriority(&draft_streams[device], cudaStreamNonBlocking, lowest_priority);
```

Expose via `ggml_backend_cuda_context::draft_stream(int device)`.

No functional change yet -- stream exists but is unused.

**Files:** `ggml/src/ggml-cuda/common.cuh`, `ggml/src/ggml-cuda.cu`
**RED->GREEN:** `test-pipeline-stream-separation.cpp` passes (draft stream exists and differs from main stream).

### 2.2 Pre-allocated fused graph scheduler (commit 3)

Create a second `ggml_backend_sched` instance dedicated to draft. At init, build the fused draft graph once, allocate its buffers once. On each cycle, only `set_inputs()` changes -- no rebuild, no realloc.

This avoids scheduler contention between accept tail (on main sched) and draft (on draft sched).

**Files:** `src/llama.cpp` (init path + decode path), `src/llama-context.h` (store `sched_draft`)
**RED->GREEN:** `test-pipeline-correctness.cpp` passes (draft uses separate scheduler, produces correct output).

### 2.3 mtp-argmax stream parameter (commit 4)

Modify `ggml_cuda_argmax()` (ggml-cuda/argmax.cu:82) to use the stream from its backend context rather than hardcoded `nullptr`. When called within the fused draft graph, this dispatches to the draft stream.

**Files:** `ggml/src/ggml-cuda/argmax.cu`

### 2.4 Pipeline cycle implementation (commit 5)

In the speculative decode loop (`common/speculative.cpp`), replace sequential:

```
verify -> accept -> UPDATE_ACCEPTED -> draft_0 -> draft_1 -> ... -> draft_N
```

with pipelined:

```
verify_k -> accept_decision ->
  stream_0: accept_tail (per_step_restore, state_advance)
  stream_1: fused_draft_{k+1}
  sync(stream_0, stream_1) -> verify_{k+1}
```

The fused draft graph is dispatched on the draft scheduler with draft streams. Accept tail runs on the main scheduler with main streams. `cudaStreamSynchronize` on both before the next verify.

**Files:** `common/speculative.cpp`, `src/llama.cpp`
**RED->GREEN:** `test-pipeline-overlap.cpp`, `test-pipeline-priority.cpp`, `test-pipeline-sync.cpp` pass.

### Performance gate -- Step 2

| Metric | After Step 1 | Target | Fail |
|--------|-------------|--------|------|
| d=5 cycle time | ~50 ms (est.) | <= 25 ms | > 35 ms |
| d=5 throughput | ~40 t/s | >= 60 t/s | < 50 t/s |
| Draft on critical path | 100% of draft time | < 30% of draft time | > 50% |
| d=1 throughput | 35.3 t/s | >= 35 t/s | > 3% regression |
| nsys overlap | 0% | >= 50% overlap | < 30% |

---

## Step 3: Eliminate UPDATE_ACCEPTED decode (4 commits)

Fold MTP KV update into verify forward via per-ubatch hook. Shrinks accept tail from ~8 ms to ~3 ms.

### 3.0 RED test stubs -- DONE (commit b34e661b)

Already committed at `b34e661b`. Tests in `tests/mtp-ubatch-hook/` (5 cpp + 3 shell):

| File | Spec invariant |
|------|----------------|
| `test-hook-tag-tensor.cpp` | `llama_main_graph_h_pre_norm()` tensor available |
| `test-hook-fires-once.cpp` | `llama_mtp_hook_fire_count()` == 1 per verify |
| `test-hook-no-secondary-decode.cpp` | `llama_mtp_decode_count()` == 0 after hook |
| `test-hook-cross-ubatch-pairing.cpp` | `llama_mtp_kv_pos_max()` covers all ubatch positions |
| `test-hook-idempotent-chop.cpp` | Repeated chop is idempotent |
| `test-hook-lockstep.sh` | Server-level lockstep coordination |
| `test-hook-reject-tail.sh` | Rejected KV entries are trimmed |
| `test-hook-sole-populator.sh` | Hook is sole populator of MTP KV |

Also committed: `tests/mtp-verify-accept/` (9 cpp + 2 shell) at `58009a77` for the verify-accept contract:
- `llama_accept_decision`, `llama_mtp_accept_verify()`, `LLAMA_ACCEPT_MODE_ARGMAX_MATCH`
- Tests for longest-prefix, bonus-becomes-seed, chop-coordination, probabilistic-rejected

### 3.1 Tag result_mtp_embd in verify graph (commit 1)

During verify forward (`MTP_OP_NONE` or `MTP_OP_WARMUP`), the hidden state tensor `result_mtp_embd` is already computed for all batch positions (it's the pre-lm_head activation). Tag it so the per-ubatch hook can find it.

Currently found by name search at `llama.cpp:4508`. Make the pointer available directly on the context for the hook.

**Files:** `src/llama.cpp`, `src/llama-context.h`

### 3.2 Per-ubatch MTP KV hook (commit 2)

After each ubatch's main forward during verify, run the MTP layer on all batch positions and write KV entries. This is the ik_llama adaptation of upstream PR #22673's pattern.

In `llama_decode_internal()` (llama.cpp:4197+), after `ggml_backend_sched_graph_compute`:
- If verify mode: extract `result_mtp_embd` for current ubatch positions
- Build single-step MTP graph for those positions
- Compute and write MTP KV entries

**Files:** `src/llama.cpp`
**RED->GREEN:** `test-hook-tag-tensor.cpp`, `test-hook-fires-once.cpp` pass.

### 3.3 Remove UPDATE_ACCEPTED decode (commit 3)

Remove `MTP_OP_UPDATE_ACCEPTED` forward pass from `mtp_accept_tokens()` (speculative.cpp:1463-1490). After accept/reject, call `llama_kv_cache_seq_rm()` to trim rejected MTP KV entries instead.

**Files:** `common/speculative.cpp`, `src/llama.cpp`
**RED->GREEN:** `test-hook-no-secondary-decode.cpp`, `test-hook-idempotent-chop.cpp`, `test-hook-cross-ubatch-pairing.cpp` pass.

### 3.4 Verify overhead measurement (commit 4)

Profile verify forward with and without hook. Expect <= 15% overhead (1 MTP layer on top of 64 main layers).

### Performance gate -- Step 3

| Metric | Before | Target | Fail |
|--------|--------|--------|------|
| Accept tail duration | ~8 ms | <= 4 ms | > 5 ms |
| UPDATE_ACCEPTED decodes | 1 per cycle | 0 | > 0 |
| Verify overhead | 13.5 ms | <= 15.5 ms | > 17 ms (>25% overhead) |
| Acceptance rate | baseline | +/- 2% | > 3% regression |

---

## Step 4: Device-resident hidden state relay (4 commits)

Kill the `inp_mtp_states` host bounce. Draft starts faster, improving overlap window.

### 4.0 RED test stub (commit 1)

Create `tests/mtp-relay/`:

| File | Validates |
|------|-----------|
| `test-relay-d2d.cpp` | Hidden state relay is D2D, no PCIe D2H+H2D |
| `test-relay-acceptance.cpp` | Acceptance rate matches host-bounce baseline (within 1%) |
| `test-relay-tensor-identity.cpp` | Relay copies correct tensor (`result_mtp_embd`, not draft output) |

**Commit:** RED tests. All fail.

### 4.1 Root-cause 70150c6d failure (commit 2)

Read commit 70150c6d. Reproduce the acceptance collapse (85% -> 3%). The `llama.cpp:4769` comment says "semantically wrong -- needs main-forward residual, not DRAFT_GEN forward's own output." Investigate:

- Is the relay grabbing the wrong tensor? (DRAFT_GEN embd vs verify embd)
- Is it a stale buffer? (graph reuse invalidated the pointer)
- Is it an offset error? (split tensor, wrong device sub-tensor)

Document the root cause.

**Commit:** Analysis doc + regression test that reproduces the failure.

### 4.2 D2D relay implementation (commit 3)

After verify compute, `result_mtp_embd` is on-device. Instead of:
```
llama_get_embeddings_ith() -> D2H -> draft_input_hidden_state -> prepare_mtp_graph_inputs() -> H2D
```

Do:
```
cudaMemcpyAsync(inp_mtp_states_device_ptr, result_mtp_embd_device_ptr, 14KB, D2D, stream)
```

For split tensors: D2D within each device's sub-tensor.

**Files:** `src/llama.cpp` (new `prepare_mtp_graph_inputs_device_resident()`)
**RED->GREEN:** `test-relay-d2d.cpp`, `test-relay-acceptance.cpp`, `test-relay-tensor-identity.cpp` pass.

### 4.3 Remove host bounce path (commit 4)

Remove `llama_get_embeddings_ith()` and `llama_set_draft_input_hidden_state()` calls from `mtp_speculative_gen_draft()` (speculative.cpp:1391-1393) for the fused path. Keep the host path as fallback for non-fused.

**Files:** `common/speculative.cpp`

### Performance gate -- Step 4

| Metric | Before | Target | Fail |
|--------|--------|--------|------|
| Verify->draft latency | ~1.5 ms (D2H+H2D) | <= 0.05 ms (D2D) | > 0.5 ms |
| PCIe transfers per cycle | 2 (D2H + H2D) | 0 | > 0 |
| Acceptance rate | baseline | +/- 1% | > 2% regression |

---

## Step 5: KQ_mask bucketing for graph reuse (3 commits)

Fallback path optimization for non-fused (non-trivial sampler) draft.

### 5.1 Bucket n_kv in build_inp_KQ_mask (commit 1)

In `src/llama-build-context.cpp:364` (`build_inp_KQ_mask`):
```cpp
int64_t n_kv_bucketed = GGML_PAD(n_kv, 64);
lctx.inp_KQ_mask = ggml_new_tensor_2d(ctx0, type, n_kv_bucketed, padded_n_tokens);
```

Update `can_reuse_graph()` condition 7 (llama.cpp:567):
```cpp
GGML_PAD(kv_self.n, 64) == GGML_PAD(prev->n_kv, 64)
```

Mask fill in `set_inputs()` already handles padding with `-inf`.

**Files:** `src/llama-build-context.cpp`, `src/llama.cpp`

### 5.2 Graph reuse logging -- PARTIALLY DONE (uncommitted)

`can_reuse_graph: HIT/MISS` logging already exists in uncommitted `src/llama.cpp` diff (lines 4856, 4890) behind `#if IK_PRINT_TIMING`. Will be committed in Step 0.1. May need hit/miss counter addition.

**Files:** `src/llama.cpp`

### 5.3 Validation (commit 3)

Shell test: run d=5 with bucketing, verify `can_reuse_graph` hit rate >= 80% during drafting. Verify acceptance unchanged.

**Files:** `tests/mtp-bucketing/test-bucketing-reuse-rate.sh`

### Performance gate -- Step 5

| Metric | Before | Target | Fail |
|--------|--------|--------|------|
| Graph reuse during drafting | 0% | >= 80% | < 60% |
| Non-fused d=5 throughput | 32.4 t/s | >= 38 t/s | < 35 t/s |
| Acceptance rate | baseline | +/- 2% | > 3% regression |

---

## Step 6: MTP head precision audit (3 commits)

Measurement only -- no code changes to the inference engine.

### 6.1 Check AutoRound GGUF MTP head precision (commit 1)

Use `llama-gguf-info` to inspect `mtp.fc.weight`, `mtp.eh_proj.weight`, `mtp.enorm.weight`, `mtp.hnorm.weight` tensor types in the production GGUF. Document current precision.

**Commit:** Findings in Phase 36 doc.

### 6.2 Rebuild with F16 MTP head (commit 2)

If MTP head is quantized: rebuild GGUF with MTP head tensors in F16. ~100 MB overhead.

**Commit:** Script or conversion command.

### 6.3 Acceptance rate comparison (commit 3)

Measure acceptance at d=1,3,5 with F16 head vs quantized head. Compare to community numbers (Q6_K: 92/81/67%).

**Commit:** Results in Phase 36 doc.

### Performance gate -- Step 6

| Metric | Current | Target | Note |
|--------|---------|--------|------|
| d=1 acceptance | 86% | >= 88% | Any improvement worth keeping |
| d=3 acceptance | ~55% | >= 65% | Match community Q6_K slope |
| Tokens per cycle (d=5) | ~4.0 | >= 4.5 | Multiplicative with cycle time |

---

## Cumulative performance gates (revised after Step 0 measurement)

Pre-measurement table assumed ~85 ms cycle and 4 tokens/cycle. Measurement says ~57 ms cycle and 1.72 tokens/cycle. Targets are scaled proportionally; the Step 4 row is dropped (premise invalidated; demoted to "revisit later").

| Milestone                            | d=5 t/s | vs baseline | Gate |
|--------------------------------------|--------:|------------:|-----:|
| **Current (measured 2026-05-06)**    |   30.00 |       0.90× | --   |
| After Step 1 (fused)                 |   ≥ 31  |       0.93× | ≥ 30 t/s |
| After Step 2 (pipeline)              |   ≥ 50  |       1.5×  | ≥ 40 t/s |
| After Step 3 (kill UPDATE_ACCEPTED)  |   ≥ 57  |       1.7×  | ≥ 50 t/s |
| Final (Steps 1–3 + Step 6 F16 head)  |   ≥ 85  |       2.5×  | ≥ 67 t/s (2×) |

Fail at any gate → stop, diagnose, re-profile before continuing.

The original ~80–96 t/s ceiling was based on multiplicative errors (5 drafts vs measured 1.5; 11 ms per step vs measured 5 ms). The revised ceiling matches the upstream single-GPU 2.5× benchmark — a defensible engineering target on this 2-GPU setup.

---

## Files summary

| File | Steps | Change |
|------|-------|--------|
| `include/llama.h` | 1 | `MTP_OP_DRAFT_GEN_FUSED`, `llama_mtp_fused_draft_invoke()`, `llama_mtp_fused_result` |
| `src/llama.cpp` | 0,1,2,3,4,5 | Instrumentation, fused eval, pipeline dispatch, ubatch hook, D2D relay, bucketing |
| `src/llama-context.h` | 2,3 | `sched_draft`, `result_mtp_embd` pointer |
| `src/graphs/build_qwen35.cpp` | 1 | `build_qwen35_mtp_fused()`, `build_qwen35moe_mtp_fused()` |
| `src/llama-build-context.h` | 1 | Declare fused builder |
| `src/llama-build-context.cpp` | 5 | Bucket `n_kv` in `build_inp_KQ_mask()` |
| `common/speculative.cpp` | 0,1,2,3,4 | Instrumentation, fused dispatch, pipeline cycle, remove UPDATE_ACCEPTED, D2D |
| `ggml/src/ggml-cuda/common.cuh` | 2 | Draft stream per device |
| `ggml/src/ggml-cuda.cu` | 2 | Draft stream creation, multi-stream graph compute |
| `ggml/src/ggml-cuda/argmax.cu` | 2 | Stream parameter from context |
| `tests/mtp-fused/` | 1 | 8 RED->GREEN tests (SCAFFOLDING DONE, b34e661b) + integration shell test |
| `tests/mtp-pipeline/` | 2 | 5 RED->GREEN tests (scaffolding TODO) |
| `tests/mtp-ubatch-hook/` | 3 | 5 cpp + 3 shell RED->GREEN (SCAFFOLDING DONE, b34e661b) |
| `tests/mtp-verify-accept/` | 3 | 9 cpp + 2 shell RED->GREEN (SCAFFOLDING DONE, 58009a77) |
| `tests/mtp-relay/` | 4 | 3 RED->GREEN tests (scaffolding TODO) |
| `tests/mtp-bucketing/` | 5 | 1 shell test (scaffolding TODO) |

## Execution order

Step 0 -> Step 1 -> Step 2 -> Step 3 (can start parallel with Step 2) -> Step 4 -> Step 5 (independent) -> Step 6 (independent)

Total: ~25 commits across 7 steps. Steps 5 and 6 can run in parallel with Steps 2-4.

## Immediate next actions (Step 0 closed; next is Step 3 telemetry)

Step 0 setup + first profile pass done at parent commits `59e926f`..`885e2a0` and submodule `d15dd96`. Next:

1. Decide whether to rerun the profile under different `--draft` / `p_min` settings to map effective draft depth (currently 1.54 at d=5; would be informative to see what depth actually fires at p_min=0)
2. Begin Step 3 (kill UPDATE_ACCEPTED) — it's now the highest-priority lever per the revised ordering. The RED tests already exist in `tests/mtp-ubatch-hook/` (b34e661b) and `tests/mtp-verify-accept/` (58009a77). Step 3.1 (tag `result_mtp_embd` in verify graph) is the first concrete commit.
3. After Step 3 lands and the `update_accepted` cost is precisely sized, decide whether Step 1 (fused) is worth the engineering vs going straight to Step 2 (pipeline) on the per-step path.

## Verification plan

After each step:
- Run production model at d=0,1,3,5 with `scripts/profile-mtp-draft-cycle.sh`
- Compare t/s and acceptance rates against performance gate table
- Verify no regression in d=1 throughput (the known-good operating point)
- For Steps 1-4: corresponding RED tests must go GREEN before proceeding
