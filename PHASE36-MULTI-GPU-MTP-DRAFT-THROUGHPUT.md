# Phase 36: Multi-GPU MTP Draft Throughput

Make draft depth > 1 faster than draft depth 1 on 2-GPU graph split.

## Design invariant

**Draft generation must be off the critical path.**

If draft cost is zero on the critical path, then any acceptance
rate > 0% is a net performance improvement. There is no break-even
acceptance threshold. There is no "diminishing returns at higher
draft depths." Every accepted token is free. This is the only
architecture worth building.

The current implementation puts draft squarely on the critical
path — each draft step adds ~11 ms to the cycle. Five drafts add
~55 ms. This means draft must "earn back" its time through
accepted tokens, and at 59% acceptance for d≥2, it barely breaks
even. That framing is wrong. Draft should cost nothing.

## Status

| Step | State | Summary |
|------|-------|---------|
| 0. Profile draft cycle | [ ] | Instrument every ms of the cycle — no estimates, only measurements |
| 1. Fused multi-draft cgraph | [ ] | Single ggml_cgraph chaining N draft steps into one compute call |
| 2. Async dual-stream pipeline | [ ] | Draft runs on a low-priority CUDA stream, overlapped with accept tail |
| 3. Eliminate UPDATE_ACCEPTED decode | [ ] | Fold MTP KV update into verify via per-ubatch hook (shrinks accept tail) |
| 4. Device-resident hidden state relay | [ ] | Kill the inp_mtp_states host bounce — draft starts faster |
| 5. KQ_mask bucketing for graph reuse | [ ] | Pad n_kv to bucket boundaries for non-fused fallback path |
| 6. MTP head precision audit | [ ] | Test F16 MTP head preservation on acceptance rate |

## Context

Per-step checkpoint for split DeltaNet state landed (see
[Multi-GPU Per-Step Checkpoint](MULTI-GPU-PER-STEP-CHECKPOINT.md)).
The ~36 ms re-decode penalty is eliminated. But d=1 still wins:

| Config | Throughput | Accept |
|--------|-----------|--------|
| No MTP | 33.5 t/s | — |
| d=1 | 35.3 t/s | 86% |
| d=3 | 32.5 t/s | 63% |
| d=5 | 32.4 t/s | 59% |

Root cause: draft is on the critical path. Every draft step adds
to cycle time. The scheduling overhead (graph build, alloc, launch)
per step makes it worse, but even with zero overhead, sequential
draft compute (~5 ms × 5 = 25 ms) dominates the cycle.

## Prior art

| System | Technique | Applicability |
|--------|-----------|---------------|
| [Upstream PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673) | Per-ubatch hook folds MTP KV update into verify forward | Directly portable — Step 3 |
| [SwiftSpec](https://doi.org/10.1145/3779212.3790246) | Disaggregated draft/target on separate GPU groups, async pipeline | Architecture principle applies; topology doesn't (we share GPUs) |
| [Mirror SD (Apple)](https://machinelearning.apple.com/research/mirror) | Dual pipeline: draft and target speculate simultaneously | Dual-stream concept applies to Steps 1–2 |
| [SGLang SpecV2](https://docs.sglang.io/advanced_features/speculative_decoding.html) | Overlap scheduler: CPU metadata prep concurrent with GPU compute | Subsumed by async pipeline |
| [NVIDIA CUDA graphs](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/) | Graph capture eliminates kernel launch overhead | KQ_mask bucketing is the ggml equivalent — Step 5 |
| [EVICT](https://arxiv.org/abs/2605.00342) | Adaptive draft tree truncation | **Contradicts the invariant.** If draft is free, never truncate |
| [Qwen3.6 27B quant studies](https://huggingface.co/froggeric/Qwen3.6-27B-MTP-GGUF) | Q6_K+BF16 head: 92/81/67% at d=1/2/3; NVFP4+BF16: 87/72/61% | Step 6 — improves yield, not critical path |
| [P-EAGLE](https://arxiv.org/abs/2602.01469) / [DFlash](https://arxiv.org/html/2602.06036v1) | Parallel draft heads (single forward for K tokens) | Requires different head architecture; inapplicable to Qwen3.6 MTP |
| [EasySpec](https://arxiv.org/abs/2502.02493) | Layer-parallel fuzzy speculation | MTP head is 1 layer — nothing to parallelize |

## Architecture: async draft pipeline

### Current (sequential)

```
                    CRITICAL PATH
                    ─────────────
VERIFY_k ──────────────────────────────────  ~13.5 ms   GPU×2
UPDATE_ACCEPTED ───────────────              ~7 ms      GPU×2
DRAFT step 1 ────────────────────            ~11 ms     GPU×2
DRAFT step 2 ────────────────────            ~11 ms     GPU×2
DRAFT step 3 ────────────────────            ~11 ms     GPU×2
DRAFT step 4 ────────────────────            ~11 ms     GPU×2
DRAFT step 5 ────────────────────            ~11 ms     GPU×2
ACCEPT TAIL ──────────                       ~8 ms      GPU×2
                                             ─────
                                             ~83 ms → ~4 tokens → 48 t/s theoretical
```

Every block is on the critical path. Draft adds 55 ms.

### Target (pipelined)

```
                    CRITICAL PATH
                    ─────────────
VERIFY_k ──────────────────────────────────  ~13.5 ms   GPU×2, stream 0
  (accept decision at end)                   ~0.1 ms    CPU
ACCEPT TAIL ──────────                       ~3 ms      GPU, stream 0
DRAFT_{k+1} ────────── (fused, async)        ~7 ms      GPU, stream 1
  ↑ overlapped ↑                             ─────
                                             ~16.6 ms → ~4 tokens → 241 t/s theoretical
```

Draft runs on a separate CUDA stream, overlapped with the accept
tail. The critical path is: VERIFY → max(ACCEPT_TAIL, DRAFT).
Draft adds **zero** time if DRAFT ≤ ACCEPT_TAIL. If DRAFT >
ACCEPT_TAIL, it adds only the difference.

With Step 3 (eliminate UPDATE_ACCEPTED), the accept tail shrinks
from ~8 ms to ~3 ms (just per_step_restore + state advance).
Fused draft at ~7 ms would then extend the critical path by ~4 ms.
But that's 4 ms for 4 tokens instead of 55 ms for 4 tokens.

### Why this works on shared GPUs

The MTP draft head is 1 transformer layer + 1 output matmul. The
verify forward is 64 layers. Both use the same 2 GPUs.

Three mechanisms enable overlap:

1. **CUDA multi-stream concurrency.** TU102 (sm_75) has 72 SMs per
   GPU. The MTP layer's attention + FFN at batch=1 uses a small
   fraction of available SMs. Accept tail kernels (per_step_restore,
   small memcpy) use even less. Two streams with non-conflicting
   resources execute concurrently.

2. **Bandwidth partitioning.** The output matmul (~3 ms, bandwidth-
   limited) will compete with accept tail for memory bandwidth if
   they overlap. But accept tail's bandwidth demand is small (~1 MB
   of checkpoint data) vs the output matmul's (~2.2 GB read). The
   interference is < 1%.

3. **Stream priority.** CUDA stream priorities ensure that if
   resources conflict, the accept tail (high priority, stream 0)
   finishes first. Draft (low priority, stream 1) fills in around
   it. This guarantees the critical path isn't extended by resource
   contention.

### Data dependencies

```
verify_k(draft_tokens_k) → hidden_state_k → draft_{k+1} → draft_tokens_{k+1} → verify_{k+1}
```

- Draft_{k+1} depends on verify_k's hidden state → draft can't
  start until verify_k's embedding output is available
- Verify_{k+1} depends on draft_{k+1}'s tokens → verify can't
  start until draft finishes

With device-resident hidden state (Step 4), the dependency is a
D2D copy (~14 KB, ~0.01 ms) instead of D2H+H2D (~1.5 ms). Draft
starts within microseconds of verify completing.

The remaining sequential dependency (verify must wait for draft)
is fundamental — you can't verify tokens you haven't drafted.
The pipeline hides draft behind accept tail, not behind verify.

---

## Step 0: Profile draft cycle

Instrument every component. No estimates — measure.

```bash
cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF \
  -DCMAKE_CXX_FLAGS="-DIK_PRINT_TIMING=1"
cmake --build build -j 32 --target llama-server

LLAMA_PROFILE_DECODE=1 build/bin/llama-server \
  -m /opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless.gguf \
  --mtp --draft 5 ...
```

**Per draft step:** build_graph, sched_alloc_graph, compute
(MTP layer vs output matmul), inp_mtp_states H2D, hidden_state
D2H, argmax kernel + D2H, `can_reuse_graph` hit/miss.

**Per cycle:** verify total, UPDATE_ACCEPTED total,
mtp_accept_tokens, per_step_restore, total cycle time, tokens
accepted per depth.

**Verify by:** measured timing table replaces every estimate in
the architecture section above.

---

## Step 1: Fused multi-draft cgraph

**Why first:** The async pipeline (Step 2) needs draft to be a
single compute call. You can't overlap 5 separate decode calls
with accept tail — each decode has its own build/alloc/sync cycle.
Fusion is the prerequisite for pipelining.

Build a single `ggml_cgraph` chaining N draft steps. One
`build_graph()` + `sched_alloc_graph()` + `compute()` for all N.

```
inp_mtp_states →
  [step_0: embed→RMSNorm→eh_proj→Attn(KV@pos₀)→FFN→lm_head→argmax₀]
  → [step_1: get_rows(tok_embd, argmax₀)→...→argmax₁]
  → ...
  → [step_{N-1}: ...→argmax_{N-1}]
output: N token IDs + N probabilities (on device)
```

**Graph structure:**
- Each step's `ggml_argmax` (I32 scalar) feeds the next step's
  `ggml_get_rows(tok_embd, argmax)` — standard ggml ops
- Each step gets its own attention KV slot and KQ_mask
- DeltaNet state flows sequentially through steps (in-place update)
- Per-step checkpoint tensors capture state after each step

**KV pre-allocation:** Reserve N positions starting at
`n_kv_start`. Step k reads `[0, n_kv_start + k)` and writes at
`n_kv_start + k`.

**Output:** N argmax results + N softmax probabilities on-device.
Single D2H copies all (~40 bytes) instead of N separate transfers.

**Constraint:** Requires trivial sampler (greedy/argmax). Non-trivial
samplers (temperature, top-p, grammar) fall back to per-step loop.
Acceptable: MTP speculative decoding already requires
`fast_argmax_for_verify` in the production path.

**Files:**
- `src/graphs/build_qwen35.cpp`: `build_qwen35_mtp_fused(n_draft)`
  + `build_qwen35moe` variant
- `src/llama.cpp`: `MTP_OP_DRAFT_GEN_FUSED` enum, fused eval path
- `common/speculative.cpp`: Use fused path when trivial sampler
- `include/llama.h`: `llama_mtp_fused_draft_invoke()` API
- RED test stubs in `tests/mtp-fused/` define acceptance criteria

**Expected cost:** 1 × build (~1 ms) + 1 × alloc (~3 ms) + 1 ×
compute (N × MTP_layer + N × output_matmul ≈ 5N ms) + 1 × D2H
(~0.3 ms). For N=5: ~29 ms → ~7 ms (eliminating 4 × 4 ms
scheduling overhead + 4 × 1.5 ms host bounce).

**Verify by:** `test-mtp-fused-single-compute.cpp` passes.
`test-mtp-fused-determinism.cpp` confirms token-for-token match
with per-step loop.

---

## Step 2: Async dual-stream pipeline

**This is the core step.** With fused draft (Step 1), draft is a
single `ggml_backend_sched_graph_compute_async()` call. Run it on
a separate CUDA stream, overlapped with the accept tail.

**Implementation:**

```cpp
// After verify_k completes and accept decision is made:

// Stream 0 (high priority): accept tail
async_launch(stream_0, [&] {
    per_step_restore(n_accepted);       // ~1 ms
    deltanet_state_advance(n_accepted); // ~2 ms
});

// Stream 1 (low priority): fused draft for k+1
async_launch(stream_1, [&] {
    prepare_mtp_graph_inputs(hidden_state);  // device-resident after Step 4
    ggml_backend_sched_graph_compute_async(sched_draft, fused_graph);
});

// Sync both streams before starting verify_{k+1}
cudaStreamSynchronize(stream_0);
cudaStreamSynchronize(stream_1);
```

**ggml scheduler integration:** The current `ggml_backend_sched`
assumes single-stream execution. Options:
1. Two separate `ggml_backend_sched` instances — one for
   verify/accept (stream 0), one for draft (stream 1)
2. Extend `ggml_backend_sched` with stream selection
3. Bypass the scheduler for draft — pre-allocate fused graph
   buffers once, reuse across cycles

Option 3 is simplest. The fused draft graph shape is constant
(same N, same tensor sizes every cycle). Allocate once at init,
then `ggml_backend_sched_graph_compute_async()` on the pre-
allocated graph with only `set_inputs()` changing per cycle.

**CUDA stream creation:**
```cpp
cudaStream_t draft_stream;
cudaStreamCreateWithPriority(&draft_stream, cudaStreamNonBlocking, low_priority);
```

**Multi-GPU:** Both GPUs need their own draft stream. The fused
graph's splits execute on each GPU's draft stream. Sync between
GPUs within the fused graph uses the same mechanism as the verify
graph (events between streams).

**Critical path analysis:**
- Accept tail: ~3 ms (after Step 3 eliminates UPDATE_ACCEPTED)
- Fused draft: ~7 ms (Step 1)
- Overlap: max(3, 7) = 7 ms on critical path
- vs sequential: 3 + 7 = 10 ms
- Savings: ~3 ms

But the real win is vs the current architecture where draft is
~55 ms: 55 ms → 7 ms on critical path = 48 ms saved.

**If draft > accept tail:** The pipeline still works. Draft
extends the critical path by (draft - accept_tail), not by the
full draft duration. With fused d=5 at ~7 ms and accept_tail at
~3 ms, the extension is ~4 ms. For ~4 tokens, that's 4 ms / 4
tokens = 1 ms per token of overhead. Baseline is ~30 ms per token.
Draft overhead is 3% of cycle time.

**Verify by:** `nsys` profile shows draft and accept tail kernels
overlapping on separate streams. Critical path measured at
max(accept_tail, draft), not sum.

---

## Step 3: Eliminate UPDATE_ACCEPTED decode

**Problem:** After verify+accept, a separate `MTP_OP_UPDATE_ACCEPTED`
forward pass syncs the MTP KV cache with accepted tokens. This is
~7 ms of accept tail that directly extends the critical path in the
pipelined architecture.

**Upstream solution (PR #22673):** Per-ubatch hook inside
`process_ubatch`. After each main forward during verify, the hook
syncs, D2Hs the hidden state, and decodes on the MTP context for
all batch positions. After accept/reject, `seq_rm` trims rejected
positions. Zero additional forward passes.

**ik_llama adaptation:** ik_llama uses inline MTP (not separate
context). The MTP layer runs as part of the main graph when
`mtp_op_type != MTP_OP_NONE`. Change: during verify, run the MTP
layer on all batch positions and write KV entries for each. After
accept/reject, trim.

**Impact on pipeline:** Accept tail shrinks from ~8 ms to ~3 ms
(per_step_restore + state advance only). If fused draft is ~7 ms,
this means draft is the longer operation and extends the critical
path by ~4 ms. Without Step 3, accept tail at ~8 ms would hide
draft entirely — but would add 8 ms to the critical path instead
of 3 ms. Net: Step 3 trades "draft perfectly hidden" for "5 ms
less on the critical path." Clear win.

**Files:**
- `src/llama.cpp`: Remove `MTP_OP_UPDATE_ACCEPTED` decode. Add MTP
  KV writes to verify graph for all batch positions.
- `common/speculative.cpp`: Remove `mtp_update_kv_cache` from
  `mtp_accept_tokens` (lines 1463–1490).

**Verify by:** No UPDATE_ACCEPTED in profile. Acceptance rates
unchanged.

---

## Step 4: Device-resident hidden state relay

**Problem:** Each draft step's input (`inp_mtp_states`) bounces
through host: D2H from verify/previous-draft embedding → H2D into
next draft's input tensor. ~1.5 ms per step, ~7.5 ms total at d=5.

With the fused cgraph (Step 1), steps 2–N chain internally (no
host bounce between them). But the first step still needs the
verify forward's hidden state. And even within the fused graph,
the step-to-step connection is via `ggml_get_rows(tok_embd, argmax)`
which stays on-device.

The remaining host bounce is: verify → D2H embedding → H2D
`inp_mtp_states` → fused draft step 0. This is ~1.5 ms.

**Fix:** After verify's compute, the `embd` tensor holds the
hidden state on-device. D2D copy into `inp_mtp_states` (~14 KB,
~0.01 ms). Skip `llama_get_embeddings_ith()` D2H and
`llama_set_draft_input_hidden_state()` H2D.

**70150c6d failure analysis:** A previous attempt (commit 70150c6d)
tried device-resident relay but acceptance collapsed 85% → 3%.
The comment at `llama.cpp:4769` says the captured residual was
"semantically wrong — the next DRAFT_GEN needs the main-forward
residual (post-24-layer pre-MTP), not the DRAFT_GEN forward's own
output."

However, examining the data flow: `llama_get_embeddings_ith()`
returns the pre-lm_head hidden state — the same tensor that
`embd` points to on-device. The DRAFT_GEN forward's embedding
output IS what the next step needs (it's H', the MTP layer's
output, which becomes the next step's input). The 70150c6d failure
was likely a bug (wrong tensor pointer, stale buffer, incorrect
offset), not a semantic error.

**Action:** Read commit 70150c6d, reproduce the failure, identify
the actual bug, fix it.

**Expected savings:** ~1.5 ms for the verify→draft transition.
Small individually, but it reduces the time before draft can start
on stream 1, allowing more overlap with accept tail.

**Verify by:** Acceptance rate matches host-bounce baseline (within
1%). No PCIe transfers between verify and draft except argmax D2H.

---

## Step 5: KQ_mask bucketing for graph reuse

**Problem:** `can_reuse_graph()` (llama.cpp:567) fails between
consecutive draft steps because `kv_self.n` increments by 1 per
step. This matters for the non-fused fallback path (when sampler
is non-trivial and can't use the fused cgraph).

**Fix:** Pad `n_kv` to bucket boundaries (e.g., 64).

```cpp
// In build_inp_KQ_mask:
int64_t n_kv_bucketed = GGML_PAD(n_kv, 64);
lctx.inp_KQ_mask = ggml_new_tensor_2d(ctx0, type, n_kv_bucketed, ...);

// In can_reuse_graph:
if (GGML_PAD(kv_self.n, 64) != GGML_PAD(prev->n_kv, 64)) return false;
```

All 10 `can_reuse_graph` conditions (llama.cpp:560–570):
1. `!prev || !prev->graph` — no prior graph
2. `u_batch.n_tokens > 1` — batch > 1 (draft is 1 ✓)
3. `u_batch.embd` — using embeddings (draft uses tokens ✓)
4. `!cparams.graph_reuse` — reuse disabled
5. `u_batch.all_seq_id != prev->all_seq_id` — seq changed
6. `kv_self.head == 0` — empty KV
7. **`kv_self.n != prev->n_kv`** — KV size changed ← fixed by bucketing
8. `n_outputs != prev->n_outputs` — output count changed
9. `cparams.mtp_op_type != prev->mtp_op_type` — op type changed
10. `!update_cache_copies()` — cache views invalid

**Impact:** ~80% graph reuse during drafting. At ~4 ms saved per
reused step: ~16 ms saved per cycle for the non-fused path.

**Not needed for the fused path** (Step 1) — the fused graph is
built once. This is a fallback optimization.

**Verify by:** `can_reuse_graph` hit rate > 80% during drafting.

---

## Step 6: MTP head precision audit

**Problem:** Our INT4 AutoRound quant (V-F1.T1.qq) shows acceptance
decay from 86% at d=1 to ~55% at d=5. Community data on the same
Qwen3.6 27B model:

| Quant | d=1 | d=2 | d=3 |
|-------|-----|-----|-----|
| Q6_K + BF16 MTP head | 92% | 81% | 67% |
| NVFP4 + BF16 MTP head | 87% | 72% | 61% |
| Our INT4 AutoRound | 86% | ~60% | ~55% |

Depth-decay is intrinsic to the Qwen3.6 MTP head. Higher precision
reduces the severity but doesn't eliminate it.

**Investigation:**
1. Check whether the AutoRound GGUF quantized the MTP head tensors
   (`mtp.fc.weight`, `mtp.eh_proj.weight`, etc.)
2. If quantized: rebuild with MTP head in F16 (~100 MB overhead)
3. Measure acceptance at d=1,3,5 with F16 head vs quantized head

**Relationship to the invariant:** With pipelining, acceptance rate
doesn't determine whether MTP is profitable — it always is. But
higher acceptance means more tokens per cycle. At d=5: 59% yields
~2.95 accepted + ~1 bonus = ~3.95 tokens. At 75%: ~3.75 + ~1 =
~4.75 tokens. That's 20% more throughput for the same cycle time.

**Verify by:** Measured acceptance at d=3 matches community numbers
for the corresponding precision level.

---

## Cumulative impact model

| State | Cycle time | Tokens/cycle | Throughput |
|-------|-----------|--------------|------------|
| Current (d=5, sequential) | ~85 ms | ~4.0 | 32.4 t/s |
| After Step 1 (fused draft) | ~50 ms | ~4.0 | ~80 t/s |
| After Step 2 (async pipeline) | ~20.5 ms | ~4.0 | ~195 t/s |
| After Step 3 (kill UPDATE_ACCEPTED) | ~16.6 ms | ~4.0 | ~241 t/s |
| After Step 6 (F16 MTP head, 75% accept) | ~16.6 ms | ~4.75 | ~286 t/s |
| Baseline (no MTP) | ~30 ms | 1.0 | 33.5 t/s |

These are theoretical upper bounds. Realistic throughput depends on:
- GPU resource contention between streams (expect 70–80% of
  theoretical overlap efficiency)
- Remaining sync points not yet identified
- Actual measured cycle times (Step 0)

**Conservative estimate:** 60% of theoretical after Steps 1–3 →
**~145 t/s (4.3× baseline)**. Even at 40% efficiency: **~96 t/s
(2.9× baseline)**.

The upstream single-GPU 2.5× benchmark (83 t/s) is achievable and
beatable with correct pipelining on 2-GPU.

## Execution order

1. **Step 0** — profile. Replaces every estimate with measurement.
2. **Step 1** — fused cgraph. Prerequisite for pipelining. The RED
   test stubs define acceptance criteria.
3. **Step 2** — async dual-stream. The core win. Draft moves off
   the critical path.
4. **Step 3** — kill UPDATE_ACCEPTED. Shrinks accept tail, reduces
   the critical-path extension from draft.
5. **Step 4** — device-resident hidden state. Reduces draft start
   latency, improves overlap window.
6. **Step 5** — KQ_mask bucketing. Fallback path optimization.
7. **Step 6** — MTP head precision. Multiplicative with cycle time
   improvements.

## Hardware

2× Quadro RTX 6000 (TU102, sm_75, 24 GiB each), CUDA 13.2,
`--split-mode graph --tensor-split 1,1`, 262K context.

Model: Qwen3.6 27B INT4 AutoRound (V-F1.T1.qq) with q4_0
Hadamard KV cache.
