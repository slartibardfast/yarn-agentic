# Phase 36: Multi-GPU MTP Draft Throughput

Make draft depth > 1 faster than draft depth 1 on 2-GPU graph split.

## Status

| Step | State | Summary |
|------|-------|---------|
| 0. Profile draft cycle | [ ] | Instrument every ms: build, alloc, compute, argmax D2H, hidden-state bounce, accept decode |
| 1. Eliminate MTP_OP_UPDATE_ACCEPTED decode | [ ] | Port upstream per-ubatch hook — fold MTP KV update into the verify forward |
| 2. Kill the inp_mtp_states host bounce | [ ] | Correct device-resident residual relay: main-forward post-backbone, not DRAFT_GEN output |
| 3. KQ_mask bucketing for graph reuse | [ ] | Pad n_kv to bucket boundaries so consecutive draft steps reuse the cached graph |
| 4. Overlap CPU graph prep with GPU compute | [ ] | Double-buffer: CPU builds graph k+1 while GPU computes step k |
| 5. Fused multi-draft cgraph | [ ] | Single ggml_cgraph chaining N draft steps — one build+alloc for all N |
| 6. Adaptive draft truncation | [ ] | Stop drafting when marginal cost exceeds expected acceptance benefit |
| 7. MTP head precision audit | [ ] | Test BF16/F16 MTP head preservation vs IQ4_XS quantized head on acceptance rate |
| 8. Pipeline overlap: accept tail + state advance | [ ] | Overlap mtp_accept_tokens with DeltaNet re-advancement on separate streams |

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

Three categories of waste remain: per-draft scheduling overhead,
unnecessary data movement, and acceptance rate decay.

## Prior art survey

| System | Technique | Relevance |
|--------|-----------|-----------|
| [Upstream PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673) | Per-ubatch hook folds MTP KV update into verify forward; eliminates separate UPDATE_ACCEPTED decode | Directly portable — Step 1 |
| [NVIDIA CUDA graphs for llama.cpp](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/) | CUDA graph capture eliminates kernel launch overhead; `cudaGraphExecUpdate` for reuse | ik_llama has ggml graph cache instead; KQ_mask bucketing is the equivalent — Step 3 |
| [SGLang SpecV2](https://docs.sglang.io/advanced_features/speculative_decoding.html) | Overlap scheduler: CPU prepares next batch metadata while current batch runs on GPU | Same principle applies to ggml graph build/alloc — Step 4 |
| [P-EAGLE](https://arxiv.org/abs/2602.01469) | Parallel multi-token prediction in single forward pass | Requires trained parallel head; not applicable to existing Qwen3.6 sequential MTP head |
| [DFlash](https://arxiv.org/html/2602.06036v1) | Block diffusion draft: K tokens same cost as 1 | Requires different head architecture; not applicable |
| [SwiftSpec](https://doi.org/10.1145/3779212.3790246) | Disaggregated draft/target on separate GPU groups; 3K lines fused CUDA | Different topology (dedicated draft GPUs); our MTP head is tiny (1 layer), not worth dedicating a GPU |
| [EasySpec](https://arxiv.org/abs/2502.02493) | Layer-parallel fuzzy speculation across GPUs during draft idle | Draft model is 1 layer — no inter-layer dependency to break; not applicable |
| [Mirror SD (Apple)](https://machinelearning.apple.com/research/mirror) | Dual pipeline: draft and target speculate simultaneously on separate accelerators | Interesting for Step 8 but our GPUs cooperate on every forward |
| [MoE-Spec](https://arxiv.org/abs/2602.16052) | Top 50% experts capture 93% routing probability; budget verification | Applicable if we target 35B-A3B MoE; not relevant for 27B dense |
| [EVICT](https://arxiv.org/abs/2605.00342) | Training-free adaptive draft tree truncation before verification | Principle applies to Step 6 — stop drafting when it stops paying |
| [Qwen3.6 quantization studies](https://huggingface.co/shieldstar/Qwen3.6-35B-A3B-int4-AutoRound-EC) | BF16 MTP head preservation gives 85–90% acceptance; quantized head degrades sharply | Directly relevant to Step 7 |

**Conclusion from survey:** P-EAGLE, DFlash, EasySpec, and SwiftSpec
are architecturally inapplicable — they require either a different
draft head or a different GPU topology. The applicable techniques are:
upstream's per-ubatch hook (Step 1), CUDA graph / graph-shape reuse
(Step 3), CPU-GPU overlap scheduling (Step 4), and adaptive truncation
(Step 6). The MTP head precision finding (Step 7) may be the single
highest-leverage item if our IQ4_XS quant has quantized the MTP head.

## Anatomy of one speculative cycle

To apply the zero-waste mantra — 100% CPU, 100% memory bandwidth,
not one wasted or repeated byte — we must account for every
millisecond in a speculative cycle. Estimated breakdown for d=5 on
2-GPU graph split:

```
VERIFY (1 forward pass, batch=N_accepted+1)
  ├─ build_graph .................. ~1 ms   CPU
  ├─ sched_alloc_graph ........... ~3 ms   CPU
  ├─ compute (64 layers) ......... ~9 ms   GPU×2
  ├─ mtp_argmax D2H .............. ~0.3 ms PCIe
  └─ accept/reject decision ...... ~0.1 ms CPU
                                   ─────
                                   ~13.5 ms

MTP_OP_UPDATE_ACCEPTED (fold accepted tokens into MTP KV)    ← WASTE: Step 1 eliminates
  ├─ build_graph .................. ~1 ms   CPU
  ├─ sched_alloc_graph ........... ~3 ms   CPU
  ├─ compute ..................... ~3 ms   GPU×2
                                   ─────
                                   ~7 ms

DRAFT (×5 sequential steps)
  Per step:
  ├─ inp_mtp_states H2D ......... ~1.5 ms PCIe  ← WASTE: Step 2 eliminates
  ├─ build_graph ................. ~1 ms   CPU   ← WASTE: Steps 3–5 eliminate
  ├─ sched_alloc_graph ........... ~3 ms   CPU   ← WASTE: Steps 3–5 eliminate
  ├─ compute (1 MTP layer) ....... ~2 ms   GPU×2
  ├─ output matmul ............... ~3 ms   GPU×2
  ├─ argmax D2H .................. ~0.3 ms PCIe
  └─ hidden_state D2H ........... ~0.5 ms PCIe  ← WASTE: Step 2 eliminates (stays on device)
                                   ─────
                                   ~11.3 ms × 5 = ~56.5 ms

ACCEPT TAIL
  ├─ mtp_accept_tokens ........... ~5 ms   GPU   ← Step 8: overlap with state advance
  ├─ per_step_restore ............ ~1 ms   GPU   (already on-device, shipped)
  ├─ DeltaNet state advance ...... ~2 ms   GPU   ← Step 8: overlap with accept
                                   ─────
                                   ~8 ms

TOTAL CYCLE ...................... ~85 ms
Tokens generated ................ ~4.0 (d=5, 59% accept → 2.95 accepted + ~1 bonus)
Effective throughput ............. ~47 t/s theoretical, 32.4 measured
```

The gap between theoretical 47 and measured 32.4 suggests additional
sync points, scheduler overhead, or measurement error in the
estimates above. Profiling (Step 0) will resolve this.

**Waste inventory:**

| Waste | Per cycle | Source | Fix |
|-------|-----------|--------|-----|
| UPDATE_ACCEPTED decode | ~7 ms | Separate forward pass to sync MTP KV | Step 1: per-ubatch hook |
| inp_mtp_states host bounce | ~7.5 ms (5 × 1.5) | D2H main-forward residual + H2D next step | Step 2: device-resident relay |
| hidden_state D2H between drafts | ~2.5 ms (5 × 0.5) | Embedding extraction for next MTP input | Step 2: stays on device |
| Graph rebuild per draft | ~20 ms (5 × 4) | n_kv changes by 1 → cache miss | Steps 3–5 |
| Drafting past break-even | variable | d=4,5 acceptance ~55% may cost more than they save | Step 6 |
| Acceptance decay from quant noise | ~27% rate loss | 86% → 59% across 5 depths | Step 7 |

**Target after all steps:** eliminate ~37 ms of waste per cycle.
With 85 ms → 48 ms and ~4 tokens/cycle: ~83 t/s theoretical.
At 70% of theoretical (sync, overhead): **~58 t/s, 1.7× baseline**.

---

## Step 0: Profile draft cycle

Instrument every component. No estimates — measure.

The codebase has `#if IK_PRINT_TIMING` instrumentation around
`build_graph`, `sched_alloc_graph`, and `compute` in `llama.cpp`
(lines 4851–4878). Also `LLAMA_PROFILE_DECODE` in
`speculative.cpp:1397`.

```bash
# Build with timing
cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF \
  -DCMAKE_CXX_FLAGS="-DIK_PRINT_TIMING=1"
cmake --build build -j 32 --target llama-server

# Run with profiling
LLAMA_PROFILE_DECODE=1 build/bin/llama-server \
  -m /opt/models/... --mtp --draft 5 ...
```

**Measure per draft step:**
1. `build_graph` wall time
2. `sched_alloc_graph` wall time
3. `compute` wall time (split: MTP layers vs output matmul)
4. `inp_mtp_states` tensor set (H2D) duration
5. `hidden_state` extraction (D2H) duration
6. `mtp_argmax` kernel + D2H duration
7. `can_reuse_graph` hit/miss rate

**Measure per cycle:**
8. `mtp_accept_tokens` duration
9. `per_step_restore` duration
10. Verify forward total
11. UPDATE_ACCEPTED forward total
12. Total cycle time and tokens accepted

**Verify by:** published timing table with measured values replacing
every estimate in the anatomy section above.

---

## Step 1: Eliminate MTP_OP_UPDATE_ACCEPTED decode

**Problem:** After verify+accept, a separate `MTP_OP_UPDATE_ACCEPTED`
forward pass syncs the MTP KV cache with accepted tokens. This is a
full `build_graph → sched_alloc → compute` cycle (~7 ms) that exists
only because ik_llama's MTP KV isn't updated during the verify forward.

**Upstream solution:** PR #22673 uses a per-ubatch hook inside
`process_ubatch`. After each main forward during verify, the hook:
1. Synchronizes
2. D2H of `t_h_pre_norm` (the pre-normalization hidden state)
3. Builds `[hidden_state, next_token]` pairs for all batch positions
4. Calls `llama_decode` on the MTP context with all positions
5. After accept/reject, `seq_rm` trims rejected positions from MTP KV

This folds MTP KV update into the verify forward — zero additional
forward passes.

**ik_llama adaptation:** ik_llama uses an inline MTP path (not a
separate model context), so the port is different. The MTP layer
runs as part of the main graph when `mtp_op_type != MTP_OP_NONE`.
The key change: during verify, run the MTP layer on all batch
positions (not just the output position), and write KV entries for
each. After accept/reject, trim rejected entries.

**Files:**
- `src/llama.cpp`: Remove `MTP_OP_UPDATE_ACCEPTED` decode call after
  accept. Add MTP KV writes to the verify graph for all batch positions.
- `common/speculative.cpp`: Remove `mtp_update_kv_cache` call from
  `mtp_accept_tokens` (lines 1463–1490).

**Expected savings:** ~7 ms/cycle → ~7% throughput improvement.

**Verify by:** Step 0 profiling shows no UPDATE_ACCEPTED forward pass
in the cycle. Acceptance rates unchanged.

---

## Step 2: Kill the inp_mtp_states host bounce

**Problem:** Each draft step requires the main-forward residual
(post-backbone, pre-MTP-layer hidden state) as input via
`inp_mtp_states`. Currently this is a D2H + H2D bounce:

1. After verify or previous draft: `llama_get_embeddings_ith()` does
   D2H of the embedding tensor (~14 KB for dim=3584)
2. `llama_set_draft_input_hidden_state()` stores pointer on host
3. Next draft's `prepare_mtp_graph_inputs()` does H2D into
   `inp_mtp_states` tensor

At 5 draft steps: ~1.5 ms × 5 = 7.5 ms of PCIe round-trips for
14 KB payloads. Plus ~0.5 ms × 5 = 2.5 ms for the hidden_state
extraction D2H between drafts.

**Why the previous attempt failed:** Commit 70150c6d tried a
device-resident fast path but captured the DRAFT_GEN forward's own
output (the MTP layer's post-FFN residual) instead of the
main-forward residual (the backbone's post-layer-64 output). The MTP
layer transforms the hidden state — feeding its own output back as
input creates a divergent autoregressive chain. Acceptance collapsed
from 85% → 3%.

**Correct approach:** The residual needed is the output of the last
backbone layer (layer 63), before the MTP layer (layer 64) processes
it. During DRAFT_GEN, the MTP layer's input IS this residual — it
arrives via `inp_mtp_states`. The MTP layer's attention + FFN produce
a transformed state, and the output matmul produces logits. The
*input* to the MTP layer is what the next step needs.

Implementation:
1. After the MTP layer's input is consumed but before it's
   overwritten, D2D copy `inp_mtp_states` → a device-resident buffer
   (`draft_residual_dev` fields already exist on `llama_context`)
2. For the next draft step, `prepare_mtp_graph_inputs()` reads from
   the device-resident buffer instead of host memory
3. The semantic issue from 70150c6d is avoided because we capture the
   *input* to the MTP layer, not its *output*

Wait — this is circular. `inp_mtp_states` IS the input we set from
host. The actual main-forward residual is the `embd` tensor at the
output position after the backbone forward. For DRAFT_GEN, the graph
doesn't run the backbone — it only runs the MTP layer. So the
residual must come from the *previous* step's MTP layer output
projected back to the embedding space... No. Let me re-examine.

**Actual data flow:**
1. Verify forward runs full backbone (64 layers) → produces hidden
   state H at output position
2. H is extracted via `llama_get_embeddings_ith()` (D2H)
3. H is set as `draft_input_hidden_state` (host pointer)
4. Draft step 0: `prepare_mtp_graph_inputs()` copies H into
   `inp_mtp_states` (H2D). MTP layer transforms H → H'. Output
   matmul on H' → logits → argmax → token_0.
5. H' is extracted via `llama_get_embeddings_ith()` (D2H)
6. H' is set as `draft_input_hidden_state`
7. Draft step 1: H' copied into `inp_mtp_states` (H2D). MTP layer
   transforms H' → H''. And so on.

So each step's *output* embedding IS the next step's input. The
70150c6d attempt was semantically correct in principle — the
DRAFT_GEN output embedding IS what the next step needs. The 85% → 3%
collapse must have been a different bug (wrong tensor, wrong offset,
stale buffer).

**Revised approach:** Investigate what 70150c6d actually captured.
If it grabbed the post-output-matmul logits instead of the
post-MTP-layer hidden state, that's the bug. The correct tensor is
the `embd` output at the embedding extraction point, before the
output projection. If that's what `llama_get_embeddings_ith`
returns (and it is — it returns the pre-lm_head hidden state), then
a D2D copy of that tensor to a device buffer eliminates the bounce.

**Implementation:**
1. After `compute()` in DRAFT_GEN, the `embd` tensor holds the
   hidden state on-device
2. D2D copy `embd` → `draft_residual_dev` (same device, ~14 KB,
   negligible)
3. `prepare_mtp_graph_inputs()` reads from `draft_residual_dev`
   instead of host `draft_input_hidden_state`
4. Skip `llama_get_embeddings_ith()` D2H and
   `llama_set_draft_input_hidden_state()` H2D entirely

**Risk:** The 70150c6d failure needs root-cause analysis before
re-attempting. Read the commit, understand exactly which tensor was
captured, reproduce the 85% → 3% collapse, then fix.

**Expected savings:** ~10 ms/cycle (7.5 ms bounce + 2.5 ms extraction).

**Verify by:** Acceptance rate matches host-bounce baseline (within
1%). Profiling shows zero PCIe transfers between draft steps except
argmax D2H (~0.3 ms × 5 = 1.5 ms, unavoidable for token IDs).

---

## Step 3: KQ_mask bucketing for graph reuse

**Problem:** `can_reuse_graph()` (llama.cpp:567) checks
`kv_self.n != prev->n_kv`. Every draft step increments `n_kv` by 1,
so graph reuse NEVER fires during drafting. Each step pays ~4 ms for
`build_graph` + `sched_alloc_graph`.

All 10 `can_reuse_graph` conditions (llama.cpp:560–570):
1. `!prev || !prev->graph` — no prior graph
2. `u_batch.n_tokens > 1` — batch > 1 (draft is always 1 ✓)
3. `u_batch.embd` — using embeddings not tokens (draft uses tokens ✓)
4. `!cparams.graph_reuse` — reuse disabled
5. `u_batch.all_seq_id != prev->all_seq_id` — seq changed
6. `kv_self.head == 0` — empty KV
7. **`kv_self.n != prev->n_kv`** — KV size changed ← THIS ONE
8. `n_outputs != prev->n_outputs` — output count changed
9. `cparams.mtp_op_type != prev->mtp_op_type` — op type changed
10. `!update_cache_copies()` — cache tensor views invalid

Only condition 7 triggers between consecutive draft steps.

**Fix:** Pad `n_kv` to bucket boundaries when constructing KQ_mask
and related tensors sized by `n_kv`.

**File:** `src/llama-build-context.cpp` (`build_inp_KQ_mask`)

```cpp
int64_t n_kv_bucketed = GGML_PAD(n_kv, 64);
lctx.inp_KQ_mask = ggml_new_tensor_2d(ctx0, type, n_kv_bucketed, ...);
```

Also need to pad in `can_reuse_graph`:
```cpp
// Compare bucketed n_kv, not raw
int64_t n_kv_buck = GGML_PAD(kv_self.n, 64);
int64_t prev_buck = GGML_PAD(prev->n_kv, 64);
if (n_kv_buck != prev_buck) return false;
```

And in `set_inputs()`: fill `[n_kv, n_kv_bucketed)` with `-inf`.

**Over-read cost:** Flash attention iterates up to `n_kv_bucketed`.
63 extra KV rows × 28 heads × 128 dim × 2 bytes = ~450 KB. At
~900 GB/s HBM bandwidth: ~0.5 µs. Negligible.

**Impact:** At d=5 near a bucket boundary, 4 of 5 drafts reuse the
graph. ~16 ms saved per cycle. Worst case (all 5 cross a boundary):
0 ms saved. Average: ~13 ms saved.

**Verify by:** Log `can_reuse_graph` hit/miss. 80%+ hit rate during
drafting. d=3 and d=5 throughput measurably improve.

---

## Step 4: Overlap CPU graph prep with GPU compute

**Problem:** Even with bucketing, bucket-crossing draft steps and
the first draft step still need `build_graph` + `sched_alloc_graph`
(~4 ms CPU). During this time both GPUs are idle.

**Approach:** While GPU executes draft step k's compute, CPU
prepares step k+1's graph in a separate thread. When GPU finishes
step k, step k+1's graph is already built and allocated — compute
starts immediately.

This is the same principle as SGLang's overlap scheduler: CPU
metadata preparation runs concurrent with GPU execution.

**Implementation:**
1. After launching `ggml_backend_sched_graph_compute_async()` for
   step k, immediately start building step k+1's graph on the CPU
2. Use a simple double-buffer: two `ggml_cgraph` slots, alternating
3. Need `n_kv` for step k+1 (= step k's `n_kv` + 1) to build the
   graph — this is known before step k's compute finishes
4. The only unknown is the argmax token ID for step k+1's embedding
   lookup — but with graph reuse (Step 3), the graph shape doesn't
   depend on the token ID, only `set_inputs()` does

**Interaction with Step 3:** When graph reuse fires (same bucket),
there's nothing to build — this step is a no-op. This step only
helps for bucket-crossing steps and the initial draft step.

**Expected savings:** ~4 ms for each non-reused step. With
bucketing, that's ~1 step per cycle → ~4 ms saved.

**Verify by:** Profiling shows GPU idle time between draft steps
drops below 1 ms.

---

## Step 5: Fused multi-draft cgraph

**Problem:** Even with bucketing and overlap, each draft step is a
separate `compute()` call with separate GPU kernel launches, sync
points, and scheduler dispatch. The GPU-side cost per step is ~5 ms
(MTP compute + output matmul), but launch overhead and sync add
~1-2 ms.

**Approach:** Build a single `ggml_cgraph` that chains N draft steps
end-to-end. One `build_graph` + `sched_alloc_graph` + `compute`
for all N drafts.

```
inp_mtp_states →
  [step_0: embed→RMSNorm→eh_proj→Attn(KV@pos0)→FFN→lm_head→argmax₀]
  → [step_1: get_rows(tok_embd, argmax₀)→RMSNorm→eh_proj→Attn(KV@pos1)→FFN→lm_head→argmax₁]
  → ...
  → [step_N: ...→argmax_N]
output: N token IDs + N probabilities (on device)
```

Each step's `ggml_argmax` output (I32 scalar) feeds the next step's
`ggml_get_rows(tok_embd, argmax)` — both are standard ggml ops.

**KV cache:** Pre-allocate N positions. Step k's attention reads
`[0, n_kv_start + k)` and writes at position `n_kv_start + k`.
Each step gets its own KQ_mask tensor, pre-sized to
`n_kv_start + k + 1`.

**DeltaNet state:** Each step updates the recurrent state
in-place. Per-step checkpoint tensors capture the state after
each step for later rollback.

**Argmax + probability:** The fused graph ends with N argmax results
and N softmax probabilities on-device. A single D2H copies all N
token IDs + N probs (~40 bytes total) instead of N separate D2H
transfers.

**Constraint:** Requires trivial sampler (greedy/argmax). If the
sampler is non-trivial (temperature, top-p, grammar), fall back to
the per-step loop. This is acceptable: MTP speculative decoding
already requires `fast_argmax_for_verify` in the production path.

**Files:**
- `src/graphs/build_qwen35.cpp`: New `build_qwen35_mtp_fused(n_draft)`
  (and `build_qwen35moe` variant)
- `src/llama.cpp`: `MTP_OP_DRAFT_GEN_FUSED` enum, fused evaluation path
- `common/speculative.cpp`: Use fused path when trivial sampler
- `include/llama.h`: `llama_mtp_fused_draft_invoke()` API
- RED test stubs already exist in `tests/mtp-fused/`

**Cost reduction:** 1 × (build + alloc + launch) instead of
N × (build + alloc + launch). For N=5: save ~25 ms scheduling +
~5 ms sync overhead = ~30 ms/cycle.

**Risk:** ggml's graph scheduler may not handle the N×KQ_mask
tensors efficiently across the 2-GPU split. May need to pin all
MTP-layer tensors to one device to avoid per-step cross-device
splits within the fused graph.

**Verify by:** `test-mtp-fused-single-compute.cpp` passes (single
`ggml_backend_sched_graph_compute_async` call). Token-for-token
determinism with the per-step loop
(`test-mtp-fused-determinism.cpp`). d=5 throughput exceeds d=1.

---

## Step 6: Adaptive draft truncation

**Problem:** At d=5, step 4 and 5 have ~55% acceptance. Each draft
step costs ~5 ms compute. A step with 55% acceptance yields 0.55
tokens but costs 5 ms + its share of the verify batch. If the
marginal cost exceeds the marginal benefit, that step is waste.

**Break-even analysis:** Each additional draft token adds ~1 ms to
the verify batch (one more position in the batch matmul). Draft
step cost: ~5 ms compute. Expected yield: acceptance_rate × 1 token.
Break-even acceptance: verify_marginal / (draft_cost + verify_marginal)
= 1 / (5 + 1) ≈ 17%. So even 55% acceptance is well above
break-even for compute cost alone.

But each draft step also delays the start of the next cycle. The
opportunity cost of a low-acceptance draft is the time that could
have been spent on the next verify+draft cycle with higher-acceptance
early tokens.

**Approach:** Track per-depth acceptance rates over a rolling window.
If depth k's rolling acceptance drops below a threshold (calibrated
by Step 0 profiling), truncate drafting at depth k-1.

The `p_min` threshold in `speculative.cpp:1446` already does a
per-token version of this using the draft probability. Enhance it
with historical acceptance data per depth.

**Implementation:**
- `speculative.cpp`: Rolling average acceptance per depth (window=100)
- Truncate when `accept[k] < threshold` (threshold from profiling)
- Log truncation events for tuning

**Expected savings:** Variable. If d=3 is the optimal depth
(cutting d=4,5), saves ~10 ms/cycle with minimal acceptance loss.

**Verify by:** Throughput at `--draft 5` with adaptive truncation
exceeds fixed `--draft 3`.

---

## Step 7: MTP head precision audit

**Problem:** Our Qwen3.6 27B model is quantized to IQ4_XS. Upstream
results with BF16 or Q6_K models show 75–85% acceptance at d=3.
Our d=1 acceptance is 86% (good), but d≥2 drops to 59–63%.

Community findings: preserving the MTP head weights in BF16/F16
while quantizing the backbone gives 85–90% acceptance across all
depths. Quantizing the MTP head amplifies rounding error through
autoregressive chaining — each draft step's prediction error
compounds into the next step's input.

**Investigation:**
1. Check whether our IQ4_XS GGUF has quantized the MTP head
   (`mtp.fc.weight`, `mtp.eh_proj.weight`, etc.) or preserved it
   in F16/BF16
2. If quantized: rebuild the GGUF with MTP head weights in F16
   (using `convert_hf_to_gguf.py` with per-tensor type overrides
   or binary patching)
3. Measure acceptance at d=1,3,5 with F16 MTP head vs IQ4_XS head
4. If F16 head recovers acceptance to 75%+, ship it as the
   production quant

**MTP head size for 27B:** 1 transformer layer ≈ 4 × 3584² × 2
bytes (F16) ≈ 100 MB. Negligible vs the ~15 GB model.

**Expected impact:** If acceptance recovers to 75% at d=3 (from
63%), each cycle produces ~3.25 tokens instead of ~2.89 → 12%
more tokens per cycle. Combined with scheduling improvements,
this is multiplicative.

**Verify by:** d=3 acceptance ≥ 75%. d=5 acceptance ≥ 65%.
Throughput at d=3 exceeds d=1 after other steps land.

---

## Step 8: Pipeline overlap — accept tail + state advance

**Problem:** After verify+accept, three operations run sequentially:
1. `mtp_accept_tokens` — update MTP KV with accepted tokens (~5 ms)
2. `per_step_restore` — restore DeltaNet state to accepted position (~1 ms)
3. DeltaNet state re-advancement — if needed (~2 ms)

These use separate GPU contexts and could overlap.

**Prerequisite:** Step 1 (per-ubatch hook) may eliminate or reshape
`mtp_accept_tokens`, changing what's available to overlap.

**Approach:** Launch `mtp_accept_tokens` on one CUDA stream while
`per_step_restore` + state advance runs on another. Synchronize
before the next verify forward.

**Expected savings:** ~3 ms/cycle (overlap the longer of accept
vs restore+advance).

**Verify by:** Profiling shows accept and restore overlap on
separate streams. Total accept-tail time drops from ~8 ms to ~5 ms.

---

## Cumulative impact model

| Step | Savings/cycle | Cumulative cycle | Tokens/cycle | t/s |
|------|---------------|-----------------|--------------|-----|
| Baseline (d=5, current) | — | ~85 ms | ~4.0 | 32.4 |
| Step 1: Kill UPDATE_ACCEPTED | −7 ms | ~78 ms | ~4.0 | ~51 |
| Step 2: Kill host bounce | −10 ms | ~68 ms | ~4.0 | ~59 |
| Step 3: KQ_mask bucketing | −13 ms | ~55 ms | ~4.0 | ~73 |
| Step 5: Fused cgraph | −10 ms | ~45 ms | ~4.0 | ~89 |
| Step 7: F16 MTP head | — | ~45 ms | ~4.5 | ~100 |

These are theoretical upper bounds assuming no new bottlenecks
emerge. A realistic 60–70% of theoretical gives **60–70 t/s**
(1.8–2.1× baseline). The upstream 2.5× target (83 t/s) would
require near-perfect execution of all steps plus acceptance
recovery.

Steps 4, 6, and 8 provide diminishing returns (~4+variable+3 ms)
and serve as polish after the high-leverage steps land.

## Execution order

1. **Step 0** — profile. No code changes, just measurement. Replaces
   every estimate with a measured value.
2. **Step 7** — MTP head precision audit. Cheapest to test (rebuild
   one GGUF), highest potential multiplier (acceptance recovery is
   multiplicative with everything else).
3. **Step 1** — kill UPDATE_ACCEPTED. Well-understood upstream
   pattern, ~7 ms, low risk.
4. **Step 2** — kill host bounce. Requires root-causing the 70150c6d
   failure, moderate risk.
5. **Step 3** — KQ_mask bucketing. Localized change, ~13 ms, medium
   complexity.
6. **Step 5** — fused cgraph. Largest code change, highest single-step
   savings, highest risk. The RED test stubs provide the acceptance
   criteria.
7. **Steps 4, 6, 8** — diminishing returns, schedule as polish.

## Hardware

2× Quadro RTX 6000 (TU102, sm_75, 24 GiB each), CUDA 13.2,
`--split-mode graph --tensor-split 1,1`, 262K context.

Model: Qwen3.6 27B IQ4_XS with q4_0 Hadamard KV cache.
