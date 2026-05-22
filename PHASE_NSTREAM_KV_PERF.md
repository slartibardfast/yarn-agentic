# PHASE_NSTREAM_KV_PERF — recover the regression *and* unlock the dispatch ceiling

**Branch**: `production/2026-q2-next` (off submodule HEAD `16b608d1`).
**Predecessor**: `PHASE_NSTREAM_KV.md` — closed 2026-05-20. Bug C structurally closed; decode-side prefill gate removed; 6 correctness gates green; **-6.2 % TG NP=8 regression carried over**.
**Status**: Open. **Tier 1 perf premise falsified, Tier 2 parked with intel captured (see retrospective 2026-05-21).** Active path: **Tier 3** unified-stream dispatch. Dominant perf lever is kernel batching at ne[3]=n_stream, NOT graph-reuse retention (A/B at production shape: cuda graphs ON vs OFF ≈ 0% Δ).
**Triangulated 2026-05-20** against prior CUDA-graph work, PSKV per-slot landing, P3.X NPC failures, PHASE45 D10 multi-slot work, and `feedback_n_stream_byte_compat_tradeoff`. **Revised 2026-05-21** post Tier 2 FA-probe diagnostic and graphs-on-vs-off A/B.

## Why this phase exists — sharpened framing post-triangulation

After deep primary-source review **plus** triangulation against everything we've already shipped on this codebase, the picture is now grounded:

1. **The regression is not isolated.** Upstream [ggml-org/llama.cpp#14863](https://github.com/ggml-org/llama.cpp/issues/14863) reports a 33 % TG regression on dual Blackwell from the same PR #14363 — multi-GPU only, tg128 only. Same pattern as ours. Upstream has no fix.
2. **We ported the data structure, kept the wrong dispatch.** Upstream runs one `llama_decode` per tick with all streams in one ubatch (`n_stream` baked into the graph at ne[3]). Our `process_batch_tokens` (server-context.cpp:4610) splits batches by primary `seq_id` and dispatches one `llama_decode` per active stream per tick. At NP=8 that's 8 cgraph rebuilds per tick.
3. **The CUDA-graph patching infrastructure is already in tree.** `ggml-cuda.cu:4500-4830` — multi-entry cache keyed by topology hash, capped at `GGML_CUDA_GRAPH_MAX=128`, FIFO-evicted; `cudaGraphExecUpdate` patches per-call `ne`/`nb`/`src_address` when topology hits an entry; dtype-strict (designed to prevent the multi-slot concat.cu:202 GGML_ASSERT crash); explicitly allows `src_address` change for `GGML_OP_VIEW` and `GGML_OP_CPY` nodes (so per-stream view base changes are *pre-supported* at the downstream layer).
4. **The K/V *write* CPY ops are already patched per-stream.** `llama.cpp:630-720` (`update_cache_copies`) was extended in PHASE_NSTREAM_KV_4D N2.b to compute `view_offs = (head % kvps) * step + (head / kvps) * stream_stride` and assign `data = view_src + view_offs` every decode. The reuse-bailout at n_stream > 1 isn't because the WRITE path can't be patched — it can and is.
5. **What's not patched are the K/V *read* views in `llm_build_kqv`.** Those view tensors bake `stream_id * nb[3]` at graph-build time; reusing a graph captured for stream 0 against stream 1's decode reads from the wrong slice. Hence the bailout. **This is the single concrete obstacle Tier 2 addresses.**
6. **TU102 ceiling math.** NP=8 aggregate ceiling at MFU=30 % is ~300-600 t/s for Qwen 35B-A3B INT4. Production is at ~10-30 % of that. Dispatch is the binding constraint.
7. **vLLM's 4.75× at NP=8 was MEASURED on our hardware** 2026-05-12 (`data/gate0-np1-np8.json` — vLLM 154.77 t/s aggregate vs ik_llama.cpp 33.5 t/s NP=1 baseline; per [`project_continuous_batching_vs_perslot_dispatch`](~/.claude/projects/-home-llm-yarn-agentic/memory/project_continuous_batching_vs_perslot_dispatch.md)). The 4.75× is real and reproducible; Tier 3 is the path to closing the gap inside ik_llama.cpp.

## Hardware ground truth (probed 2026-05-20)

```
nvidia-smi nvlink -s   → all links inActive  (no bridge installed)
nvidia-smi topo -m     → GPU0–GPU1 connection: PHB  (PCIe Gen3 through host bridge)
```

PCIe Gen3 x16 effective ~13 GB/s bidirectional. Production runs `--device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1`. Closest published fabric analogue: ik_llama.cpp [PR #1080](https://github.com/ikawrakow/ik_llama.cpp/pull/1080) on 4× RTX 3090 PCIe — Llama-3-70B Q4_0 at ~50 t/s gen on `--split-mode graph`. For MoE this fabric is workable because all-reduce volume scales with active params (3 B) not total params (35 B).

## What's already in tree (triangulation against prior work)

The following directly bears on this phase:

- **`ggml-cuda.cu` multi-entry CUDA-graph cache** (Phase 36/37/38, baked production): topology-hash-keyed `std::unordered_map<uint64_t, ggml_cuda_graph>`, capped via `GGML_CUDA_GRAPH_MAX` (default 128, FIFO eviction). `cudaGraphExecUpdate` patches per-call src/ne/nb on topology hits. Strict dtype/op-sequence check.
- **`update_cache_copies()` per-stream patching** (PHASE_NSTREAM_KV_4D N2.b, baked 2026-05-20): K/V write CPY view offsets recomputed per decode from `head` and `kv_size_per_stream`. The mechanism Tier 2 must replicate for the read-view side.
- **PSKV per-slot FA kernel** (P2 landed 2026-05-15, ILP ratcheted 2026-05-18): `GGML_OP_FLASH_ATTN_EXT_PER_SLOT_KV` with `src[5] = per_row_k_bound`; always-on production path; routes to `ggml_cuda_flash_attn_ext_wmma_f16_case_pb1<256,256,8,half>`. **NPC PASS at NP={1,2,4,8} multi-GPU verified** via `scripts/verify-production-determinism.sh`. Per-row-k-bound is plumbed but currently unused for correctness (mask already pushes past-bound positions to -inf bit-identically). This is the prerequisite kernel Tier 3 needs.
- **MTP fused graph reuse** (Phase 37 #5 in `can_reuse_graph`): allows `n_tokens > 1` reuse when `mtp_op_type == MTP_OP_DRAFT_GEN_FUSED` and step counts match.
- **n_kv bucketing** (Phase 36 Step 5): rounds `n_kv` to multiples of 64 so consecutive draft steps within a 64-cell bucket share a cached graph; per-call ne/nb patched via `cudaGraphExecUpdate`.

The takeaway: **ik_llama.cpp has shipped the exact mechanism Tier 2 needs**, repeatedly, across four prior phases. Tier 2 is "do that once more, for the attention-read views."

## Triangulation against PHASE_NSTREAM_KV closure + DFlash multi-slot expectations

Three places where this phase touches load-bearing predecessor work:

1. **Bug C closure mechanism — must be preserved by both Tier 2 and Tier 3.** PHASE_NSTREAM_KV's closure language: "Bug C (the mixed prefill+decode GEMM-vs-GEMV accumulation divergence) is closed structurally" because "each call sees a single-stream batch — mul_mat sees a uniform shape per call." Tier 2 doesn't change dispatch (still per-stream) — Bug C closure is untouched. Tier 3 unifies dispatch — must argue and verify that "uniform shape per call" still holds. See §"Tier 3 composition" below for Q1.

2. **Open follow-ups carried by PHASE_NSTREAM_KV** — K-shift, defrag, v_trans guarded at `n_stream == 1`. PHASE_NSTREAM_KV_PERF inherits these. Tier 2 doesn't need to lift them (Tier 2 is graph-reuse only). Tier 3 lifts them as part of T3.f.

3. **DFlash multi-slot composition** — Phase 5 (`common_speculative_draft_batched`) already batches the draft side across slots. Phase 6 test harness gates byte-identity at NP={1,2,4,8}. Tier 2 (graph reuse) is transparent to DFlash — no dispatch model change. Tier 3 (verify-side unification) puts the verify path in the same mental model as the already-unified draft side. See §"Tier 3 composition" Q3 below.

## Risks distilled from prior work

These are direct lessons-learned that bind Tier 2 / Tier 3 scoping:

1. **`feedback_n_stream_byte_compat_tradeoff` — DO NOT regress the axis order.** Production runs the non-byte-compatible 4D `[head_dim, kv_size_per_stream, n_head_kv, n_stream]` (heads outer per stream, positions inner). Reverting to byte-compatible layout broke NP=2 R2 with garbage tokens in N1 session. Tier 2 must not touch the axis order.
2. **P3.X NPC stochastic 1/8** (tasks #37/38/39 in this branch's task history): cudaMallocAsync gained +5% TG / +16.7% PP but broke NPC at 1/8 slots stochastically. **Different mechanism from Tier 2** (memory allocator change vs graph-reuse extension) — but the failure mode is the watch-pattern: any perf change that affects CUDA's runtime state ordering can introduce stochastic NPC drift. Run determinism gates after every commit, not just at end of phase.
3. **`project_fattn_per_slot_kv_p2_landed_kernel_only` — "CUDA graph cache warm-up at NP>1"** was named as a suspected divergence source pre-NSTREAM-KV. The PHASE_NSTREAM_KV closure structurally resolves Bug C; PSKV ILP (2026-05-18) verified NPC at NP={1,2,4,8} multi-GPU. Tier 2 must keep that property. **Specific test**: run the same prompt 3× sequentially at NP=8 (R1, R2, R3) and verify byte-identity — that's the exact configuration where warm-up bugs surface.
4. **`feedback_bake_measurement_env_gates`** — no `LLAMA_*_ENABLE` knobs left around after verification. Bake or revert. Past leftovers: `LLAMA_FATTN_PER_SLOT_KV_ENABLE` (baked), `LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH` (cost ~30k tokens of rediscovery before being baked). Don't add new knobs to Tier 2.
5. **`feedback_no_workarounds` + `feedback_no_skipping_lessening`** — no half-implementations, no env-gated optional paths. The Tier 2 patch is either the new normal or it doesn't land.
6. **`feedback_oneshot_then_evaluate`** — implement Tier 2 coherently (no intermediate "builds green but not yet correct" partial states), evaluate against gates including the warm-up R1=R2=R3 test, then decide whether to advance to Tier 3.
7. **NPC contract**: the 6 gates closed in PHASE_NSTREAM_KV must stay green. The bench gate GP3.a is *additional* binding — recover the -6.2 % regression at minimum.
8. **PHASE45 D10.b explicitly named "CUDA graph reuse for batched draft" as the next lever** for multi-slot perf beyond +27%. Tier 2 is essentially that lever for vanilla decode, not just MTP draft.

## Phase 0 — Prerequisites (must land before any Tier 2 work)

Two prerequisites identified during triangulation. Both close gaps that PHASE_NSTREAM_KV's closure left implicit but that this phase's gates expose.

### P0.A — DFlash server CLI: surface, fix, document open gaps

**Scope evolution log:**
- **Original (`0adfe9f` / `d9fb279`)** — frame as *implementation* of CLI wiring fix.
- **2026-05-20 first revision** — grep of recent commits revealed `61a7e874` (2026-05-18) *"server: wire DFlash sidecar drafter path; drop n_parallel=1 refusal"* already landed wiring + production profile `/home/llm/profiles/qwen36-27b-x2-dflash.sh`. Reframed as *verify-on-post-fold*.
- **2026-05-20 second revision (this section)** — verification surfaced two additional pre-existing DFlash bugs and one fundamental open issue. Reframed again as *land what we can, document what's open with explicit testable theories*.

#### P0.A.1 — Drafter K/V VRAM cap [SHIPPED]

**Symptom on first verify-on-post-fold attempt:**

```
[dflash] CUDA error in alloc K cache: out of memory
common_speculative_state_dflash: llama_set_dflash failed: rc=-8
... abort in FUSED_RMS_NORM during common_speculative_is_compat
```

**Root cause:** `llama_set_dflash` sized the drafter K/V cache to the full target `n_ctx`:

```cpp
// src/llama-dflash.cpp:474 (pre-fix)
const int seq_len_cap = (int) ctx_tgt->cparams.n_ctx;  // = 524288
```

Cache footprint at production profile (--ctx-size 524288, --parallel 2):
```
kv_bytes = L_d × n_slots × SeqLen × H_kv × D_h × sizeof(__half) × 2 (K and V)
         = 5  × 2       × 524288  × 8    × 128 × 2                 × 2
         = 21.5 GiB
```
That alone exceeds the 24 GiB Quadro RTX 6000's free VRAM after weights + main KV + NCCL.

**Fix:** the drafter has 4 SWA layers (sliding_window=2048) + 1 full-attention layer (per `dflash.layer_types` in the GGUF). With MAL capped universally at `swa_window + block_size + 16 = 2080`, all layers operate within the same bound; cache footprint drops to **85 MB** (99.6 % reduction). The full-attention layer becomes effectively SWA-bounded under this regime — an explicit cost of the universal cap, to revisit if drafter quality at production scale shows measurable regression.

Submodule commits (production/2026-q2-next):
- `[MAL cap commit]` — set `seq_len_cap = MAL_max` in allocate_ctx_scratch; cap `anchor_pos = min(prompt_tgt.size(), MAL_max)` at both single-slot and multi-slot DFlash draft call sites in `common/speculative.cpp`.

#### P0.A.2 — stage_target_hiddens end-trim restore [SHIPPED]

**Symptom on first smoke after VRAM cap landed:**

DFlash at `--parallel 1`, temp=0, simple prompt — output filled with token duplication and acceptance rate at 8 % (vs the canonical 30–50 % range for DFlash).

**Root cause:** the MAL-cap commit changed `stage_target_hiddens` to read from the extract buffer's tail (intended for the long-context case where `buf` accumulates more rows than `mal_anchors`). That edit replaced the original `buf.resize(mal_anchors * D_emb)` end-trim with a conditional front-erase — **silently dropping the end-trim entirely**.

Effect across cycles: the previous cycle's verify decode appended `(1 + n_draft)` rows past the post-accept `cache_tokens.size()`. Without the end-trim, those rejected-draft hiddens survived into the next cycle's `stage_target_hiddens`. The drafter then read stale rejected hiddens as if they were accepted target features → wrong predictions → cycle never recovered.

**Fix:** restored the end-trim `buf.resize(mal_anchors * D_emb)`. With the MAL cap and caller-side anchor_pos cap in place, `mal_anchors ≤ MAL_max`, so `buf` stays bounded.

Submodule commit:
- `[stage trim commit]` — restore end-trim at end of `stage_target_hiddens`.

**Effect on smoke:** acceptance rate jumped from **8 % → 54 %** at temp=0. Numerically validates the trim was a real correctness regression (not just a perf issue).

#### P0.A.3 — DFlash output divergence from spec-none baseline [OPEN — root cause REOPENED, T5 falsified]

> **Update 2026-05-20 (later same day): T5 (cb_eval as cause) is FALSIFIED.**
>
> Wrote a libllama-level binding observational test
> (`ik_llama.cpp/tests/dflash-speculative/test-dflash-extract-observational.cpp`,
> held on disk uncommitted) that decodes the same prompt twice with
> `llama_set_dflash_extract_layers` armed on the production set
> {1, 16, 31, 46, 61} vs disarmed, comparing per-row argmax. Ran three
> shapes:
>
> | Shape | Generated tokens | Result |
> |---|---|---|
> | Single 12-token prefill | 12 | byte-identical |
> | Prefill + 64 single-token autoregressive decodes | 64 | byte-identical |
> | Prefill + 32 verify-style 5-wide multi-token decodes | 160 | byte-identical |
>
> All with cb_eval armed. Extract buffer populates correctly (61440
> floats = 12 rows × 5120 D_emb), confirming the hook fires on every
> configured `l_out-<il>` node and the scheduler takes its slow path.
> The target's argmax is unchanged.
>
> The previous falsification matrix's "T5 CONFIRMED" rested on
> `ngram-simple ≡ spec-none ≠ DFlash`. The architectural diff between
> ngram-simple and DFlash is NOT only the cb_eval install — it is the
> entire DFlash pipeline (combine_features, inject_kv_fused, drafter
> forward, common_speculative_draft sample-and-accept loop). The
> matrix was correlation, not causation; the cheap binding test
> isolated cb_eval as a single variable and exonerates it.
>
> **The fix paths below ("re-architect to set_output tap nodes" etc.)
> are NO LONGER justified by P0.A.3** — there is no longer a bug at
> the cb_eval surface for that change to fix.
>
> **Candidate downstream theories that need fresh investigation:**
>
> - `combine_features` cuBLAS pinned-HMMA GEMM dispatch ordering
>   (PHASE 67-69 batched-pinned rewrite landed; may interact with
>   post-fold 4D KV in unanalysed ways).
> - `inject_kv_fused` async sync — does the post-fold variant
>   serialise correctly against subsequent target decodes?
> - `drafter_forward` kernel state leakage / shared CUDA stream race
>   with the target context.
> - `common_speculative_draft` sample-and-accept loop position math
>   (verify-batch position drift relative to target's committed pos).
> - Post-fold 4D KV interaction with drafter's own KV (shared
>   `llama_context`, possible cell alias on the inject path).
>
> **Next experiment**: A/B on `examples/dflash-speculative-simple/dflash-speculative-simple.cpp`
> with cb_eval install force-disabled vs intact. This isolates
> cb_eval at the CLI level where the bug is observed. Predicted
> outcome (given the libllama-level falsification above): the
> divergence survives disabling cb_eval, confirming the mechanism is
> in the rest of the DFlash pipeline.
>
> **Bundle held on disk uncommitted** for future revival once the
> real mechanism is named:
> `specs/dflash/cb_eval_residual_capture.allium` (Allium contract,
> parses clean); `specs/dflash/CbEvalObservational.tla` +
> `CbEvalObservationalMC.{tla,cfg}` +
> `CbEvalObservationalMC_callback.cfg` (TLA+ positive verifies,
> negative produces expected counterexample); the libllama test
> above. They encode a legitimate contract on the cb_eval surface
> (extraction is observational) that the implementation upholds —
> but the contract is independent of P0.A.3 root cause.
>
> Sections below this banner are preserved as the **historical
> diagnostic record** of the now-falsified T5 line of investigation,
> NOT as guidance for the fix.

##### P0.A.3 A/B result — cb_eval at the CLI level RESULT (2026-05-20)

> **The CLI-level A/B is complete and confirms the libllama-level
> finding.** With `LLAMA_NO_DFLASH_CB_EVAL=1` patched into the
> install site at `src/llama.cpp:10072` (env-gated, diagnostic-only,
> reverted after capture), the dflash-speculative-simple binary
> running at temp=0, seed=42, the documented prompt
> ("Write a quicksort in Python.", n=64):
>
> | Run | Output start | n_emitted | drafts | accepted | t/s |
> |---|---|---|---|---|---|
> | A — cb_eval intact (HEAD) | `<think>  - The **UserUser**:**: ... quick quick quick quick ...` | 64 | 88 | 42 | 16.80 |
> | B — cb_eval disabled (env-gated) | `<think>... and modify... 2.` (clean prose) | 63 | 0 | 0 | 25.79 |
>
> Logs at `data/p0a3-cb-eval-ab/{A,B}-cb_eval-*.log`.
>
> B's drafter pipeline failed every cycle with "extract buffer too
> short for slot 0 (layer 1) seq_id 0: have 0 rows, need N" — the
> drafter has no hiddens to consume because cb_eval never populated
> the extract buffer. The CLI falls back to plain decode
> (`llama_decode` of a single token) for every cycle, which produces
> clean output. n_drafts=0 confirms the drafter pipeline never ran
> successfully.
>
> Composed with the libllama-level result (cb_eval install + scheduler
> slow-path does NOT perturb the target's argmax), the diagnostic
> reads:
>
> 1. cb_eval is the SUPPLIER of hiddens to the drafter pipeline.
> 2. cb_eval's slow-path does NOT perturb the target's forward output.
> 3. The drafter pipeline, FED CORRECT HIDDENS via cb_eval, produces
>    degenerate predictions (the "quick quick quick" loop).
> 4. Therefore: **the bug is downstream of cb_eval, inside the drafter
>    pipeline** — `combine_features`, `inject_kv_fused`,
>    `drafter_forward`, or `drafter_lm_head` — not in cb_eval itself.
>
> P0.A.3 root cause is now narrowed to **the drafter pipeline kernels**.
> The libllama Allium/TLA+/test bundle on cb_eval remains held on disk
> uncommitted (the contract it encodes is still valid; it just isn't
> the P0.A.3 fix surface).
>
> Diagnostic env-gate at `src/llama.cpp:10072` REVERTED in working
> tree. Not committed; nothing to revert in git history.

##### P0.A.3 test ladder for `save_per_step_ssm` / `save_all_steps` binding (2026-05-21)

> Now that A→D→E bisection narrowed the bug to the
> `ggml_delta_net` kernel's `save_all_steps` parameter, this section
> plans the test ladder that will bind the invariant at four
> increasingly specific levels. Tests go in
> `ik_llama.cpp/tests/dflash-speculative/`, plan stays here. All
> tests are self-consistency comparisons (save_all_steps=true LAST
> per-step state must byte-equal save_all_steps=false final state) —
> no external reference values needed because the kernel runs the
> same sequential recurrence either way.

###### L1 — libllama observational: `test-dflash-save-per-step-ssm-observational.cpp`

The cb_eval observational pattern, retargeted to `save_per_step_ssm`.
Cheapest test in the ladder; binds the system-level invariant
before drilling into the kernel.

Setup:
- Load Qwen 3.6 27B target, q4_0 KV, Hadamard ON, dual-GPU.
- Tokenize a short prompt; call `llama_decode` of the whole prompt
  as a single multi-token batch (the shape that triggers the bug).
- Capture per-row argmax of the prefill logits.

Two passes:
1. Default — `save_per_step_ssm` left false.
2. Armed — call `llama_spec_ckpt_init(ctx, LLAMA_SPEC_CKPT_AUTO, n_tokens+1)`
   then `llama_spec_ckpt_save(ctx, 0)` BEFORE the decode. That sets
   `save_per_step_ssm = true`. Call `llama_kv_cache_seq_rm` between
   passes to clear cache. Same prompt, same seed, same cparams.

Assertions:
- `argmax_disarmed.size() == argmax_armed.size()`.
- Byte-identical at every row. FAIL → save_per_step_ssm IS the
  system-level perturbation (we already strongly suspect this from
  the CLI A/B; L1 confirms at the libllama layer where the libllama
  observational cb_eval test passed cleanly).
- Mechanism check: assert `kv.save_per_step_ssm == true` after
  step (2)'s spec_ckpt_save (binds that we actually exercised the
  variable, per `feedback_verify_test_mechanism_before_trusting`).

Predicted result: armed argmax DIFFERS from disarmed. Binding.

Token cost: ~15 k.

###### K1 — kernel-direct last-state equivalence: `test-deltanet-save-all-steps-last-state.cpp`

The minimum binding test for the kernel-level invariant. Mirrors
the existing `test-deltanet-shape-invariance.cpp` style (direct
ggml_delta_net via CUDA backend, no llama_context, fp32 inputs).

Geometry (production DeltaNet):
- `HEAD_DIM = 128`, `H_V = 16`, `H_K = 2` (gqa_ratio=8).
- `n_tokens = 5` (the production verify-batch shape).
- `n_seqs = 1` (single-slot to start; K1 follow-up adds n_seqs=2).

Inputs (deterministic, seeded):
- Q, K: `[HEAD_DIM, n_tokens, H_K, n_seqs]`, fp32 random in [-0.5, 0.5].
- V: `[HEAD_DIM, n_tokens, H_V, n_seqs]`, fp32 random.
- G: `[n_tokens, 1, H_V, n_seqs]`, fp32 random.
- Beta: `[1, n_tokens, H_V, n_seqs]`, fp32 random.
- State: `[HEAD_DIM, HEAD_DIM*H_V, 1, n_seqs]`, fp32 random.

Two graphs, same inputs:
- `out_false = ggml_delta_net(..., save_all_steps=false)` — total size
  `output_size + state_size`.
- `out_true  = ggml_delta_net(..., save_all_steps=true)` — total size
  `output_size + n_tokens * state_size`.

After execution:
- Extract `final_state = out_false[output_size .. output_size + state_size)`.
- Extract `last_per_step = out_true[output_size + (n_tokens-1)*state_size
  .. output_size + n_tokens*state_size)`.
- Both views have `state_size = HEAD_DIM * HEAD_DIM * H_V * n_seqs` floats.

Assertion: `final_state == last_per_step` byte-identical at fp32.

Reporting on FAIL:
- Print first-divergent index, both values, ulp diff.
- Per-warp / per-head decomposition: which warp_id and head_id the
  divergence is in. Helps the kernel-level fix target the right
  loop iteration.

Predicted result: FAIL at the first multi-step recurrence. The
divergence will be at the LAST step of the state update.

Token cost: ~12 k.

###### K2 — per-iter bisect: `test-deltanet-save-all-steps-bisect.cpp`

Smallest-n_tokens search. Confirms whether divergence appears at
step 1 (the first multi-step iteration) or accumulates later. The
result tells us how the compiler is rescheduling: if at step 1, the
state update path is reordered by the `if (save_all_states)` branch
itself; if at step >= 2, accumulation drift inside the per-step
write.

For `n_tokens ∈ {1, 2, 3, 4, 5, 8, 16}`:
- Build and run BOTH save_all_steps modes.
- Compare per_step_state[n_tokens-1] (from true) vs final_state
  (from false).

Report:
- Smallest `n_tokens` where divergence appears.
- For each n_tokens, ulp-distance of the most-divergent element.

Two sub-cases inside the same test:
- n_tokens=1: the per-step loop runs once. The save_all_states=true
  path writes the state to per_step_state[0]; the save_all_states=false
  path writes the SAME state to final_state. The two write
  destinations are different memory regions but the SOURCE
  (state_local register) is identical. Predicted: PASS at n_tokens=1.
- n_tokens=2: the loop runs twice. State_local is updated at t=0,
  then again at t=1. If the writes at t=0 (under save_all_states=true)
  spill state_local to register-bank-different addresses, the t=1
  recompute could see different scheduling. Predicted: FAIL at
  n_tokens=2 if the bug is per-iteration scheduling.

If K2 shows divergence ONLY at n_tokens >= 2, the smoking gun is in
the per-iteration loop body's instruction scheduling.

Token cost: ~10 k.

###### K3 — output-token equivalence: `test-deltanet-save-all-steps-output-tokens.cpp`

Comparing the per-token OUTPUT rows (the `out_base` writes, not
state). The output rows are computed identically in both
save_all_states modes (the `if (save_all_states)` branch only
controls state writes, not output writes). If output rows
DIFFER, the compiler is reordering operations in the state-update
loop in ways that perturb FMA fusion / register usage, cascading
into the output computation.

Setup: identical to K1 (same inputs).

Compare: `out_false[0 .. output_size)` vs `out_true[0 .. output_size)`.
Should be byte-identical.

Outcomes:
- PASS: only state writes differ, output is clean. The bug is
  isolated to the recurrent state path; the output is unaffected.
  Fix is to align save_all_states=true's state writes with =false's
  via explicit barrier / volatile / __syncthreads placement.
- FAIL: output rows ALSO differ. The compiler is reordering the
  shared state-update loop body. Fix is more invasive (force
  identical instruction scheduling between modes; possibly by
  always writing per-step states and then choosing the right offset
  on the host side — removing the runtime branch from the kernel).

Token cost: ~8 k.

###### Closure binding (what "fix landed" means)

The fix lands when ALL of the following hold:
1. K1 PASS at production geometry (n_seqs=1 AND n_seqs=2).
2. K2 PASS across n_tokens ∈ {1..16}.
3. K3 PASS at production geometry.
4. L1 PASS at the libllama level.
5. `dflash-speculative-simple` at the documented prompt produces
   non-degenerate output AND mean accept rate ≥ 1.0 — same gate as
   PHASE_DFLASH.md T5 closure.
6. `examples/dflash-speculative-simple` byte-identical-to-spec-none
   on at least one prompt where spec-none and DFlash should agree
   (every accepted token must match what spec-none would have
   produced at the same position).

Gate 5 is the user-visible "DFlash CLI no longer broken" signal.
Gate 6 is the strict-binding "DFlash is observationally equivalent
to spec-none" — the original P0.A.3 invariant before the matrix
went astray.

###### Sequencing

L1 + K1 land together (cheapest, both bind the same invariant at
different levels). If both PASS unexpectedly, save_per_step_ssm
ISN'T the kernel-side cause and we re-open. If both FAIL as
predicted, K2 + K3 follow to narrow the fix surface, then the fix
lands. Total ladder cost: ~45 k tokens. Fix scope estimated at
~30-60 k.

###### What gets committed along the way

- L1 + K1 sources, CMakeLists wiring, and an initial PASS/FAIL row
  in this section: one commit.
- K2 + K3 added once K1 has bound: separate commit.
- The actual kernel fix: separate commit with its own subsection
  here (mechanism + fix + before/after K1-K3 results).
- MEMORY.md append-only entry on closure.

Tests stay GREEN once the fix lands — they become a regression
gate, not a one-time diagnostic. Tests go on `production/2026-q2-next`.

##### P0.A.3 test ladder L1 + K1 result — BOTH PASS, Suspect 2 FALSIFIED (2026-05-21)

L1 + K1 both landed on `production/2026-q2-next` and **both PASS on
HEAD** — the opposite of the predicted result. This **falsifies**
Suspect 2 ("save_per_step_ssm / save_all_steps perturbs the verify
decode"). The earlier CLI run E A/B was correlation, not causation:
gating save_per_step_ssm OFF via `LLAMA_NO_SPEC_CKPT_SAVE` did change
the CLI output, but the change isn't reproducible at the libllama
layer OR the kernel layer in isolation.

###### K1 result (kernel-direct, `test-deltanet-save-all-steps-last-state`)

```
[gate-A output] byte-identical: 10240 fp32 floats
last_step_state first 8:   +0.039601  -0.087997  +0.089954  +0.155827  -0.134584  -0.037176  -0.020451  +0.113233
final_state     first 8:   +0.039601  -0.087997  +0.089954  +0.155827  -0.134584  -0.037176  -0.020451  +0.113233
[gate-B state] PASS — last per-step state (save_all_steps=true) byte-identical to final state (save_all_steps=false): 262144 fp32 floats
[PASS] save_all_steps self-consistency: last per-step state == final-only state, byte-identical at production geometry
```

At production geometry (HEAD_DIM=128, H_V=16, H_K=2, n_tokens=5,
n_seqs=1) the `ggml_delta_net` CUDA kernel produces **byte-identical**
output rows AND last-step state across `save_all_steps ∈ {true,
false}`. The `if (save_all_states)` branch in `delta-net.cu:163-184`
does NOT perturb the per-step recurrence math at this geometry. The
kernel is **exonerated** as the kernel-side cause.

###### L1 result (libllama observational, `test-dflash-save-per-step-ssm-observational`)

```
[save-ssm] disarmed: prefill=12 rows, generated=32 tokens
per_step_alloc:      CUDA0 per-step buffer =   364.69 MiB (max_tokens=5)
per_step_alloc:      CUDA1 per-step buffer =   364.69 MiB (max_tokens=5)
checkpoint_alloc_shadows:      CUDA0 split shadow buffer =    74.81 MiB
checkpoint_alloc_shadows:      CUDA1 split shadow buffer =    74.81 MiB
[save-ssm] armed: prefill=12 rows, generated=32 tokens
[PASS] save_per_step_ssm is observationally equivalent to spec-none on the verify-batch decode: prefill=12 argmax rows match, generated=32 tokens match
```

At the libllama API surface, calling `llama_spec_ckpt_init(PER_STEP)
+ llama_spec_ckpt_save(seq=0)` (which sets
`transformer_kv.save_per_step_ssm = true` AND allocates the 364.69
MiB per-step buffers AND the 74.81 MiB shadow buffers AND performs
the checkpoint save) **before** each verify-style 5-row decode
produces a byte-identical per-row argmax sequence to the same decode
with the flag never armed. Same prompt, same cparams, same kernel
geometry as the CLI.

The flag + buffer machinery is **exonerated** as the libllama-side
cause.

###### What this means for the diagnosis

Three suspects are now falsified:

| Suspect | Binding test | Result |
|---|---|---|
| cb_eval install (R5.4* matrix) | `test-dflash-extract-observational` | PASS — falsified 2026-05-20 |
| cudaMallocAsync in combine/inject | run C (sync `cudaMalloc` swap) | FALSIFIED 2026-05-20 |
| `save_per_step_ssm` / `save_all_steps` | L1 + K1 | FALSIFIED 2026-05-21 |

The CLI failure must therefore come from the **interaction** of the
DFlash drafter pipeline (combine_features → inject_kv_fused ×
L_d → drafter_forward → sample-and-accept) with the verify decode,
not from any single axis we have isolated so far. Specifically:
neither save_per_step_ssm being TRUE in isolation, nor the kernel's
per-step state writes, nor cb_eval's residual capture, perturb the
verify decode by themselves.

Run E from the prior bisect (CLI with `LLAMA_NO_SPEC_CKPT_SAVE`
producing different degenerate output "...# How to # How to...")
remains an empirical fact; the L1 PASS shows the gate it flipped
isn't the proximate cause of the divergence. The likely real
mechanism: the CLI's drafter pipeline produces *different inputs*
to the verify decode depending on whether `save_per_step_ssm` is
on — because `per_step_restore()` at the end of each cycle moves
the recurrent state differently when per_step buffers exist vs
when they don't. **Suspect 4: per_step_restore semantics under
LLAMA_SPEC_CKPT_PER_STEP**, not the save side.

###### Next steps

The test ladder K2 + K3 lose their binding target (both were defined
against the now-falsified Suspect 2). They will be re-targeted once
the new root-cause hypothesis is named. L1 + K1 remain in the tree
as regression gates — both should stay PASS forever.

New diagnostic plan to surface in a follow-up edit to this section:
- L2 — libllama observational that includes the FULL drafter pipeline
  (combine_features + inject_kv_fused + drafter_forward) wrapped
  around the verify decode, comparing spec-none vs DFlash-armed. This
  is the binding observational that the original P0.A.3 matrix tried
  to construct but did via a partial bisect.
- R5.5 — CLI direct A/B with `LLAMA_NO_DFLASH_PER_STEP_RESTORE` (new
  env-gated diagnostic at `src/llama.cpp:8510` skipping
  `kv.per_step_restore(accepted_step)`). If this run flips back to
  non-degenerate output while save_per_step_ssm stays TRUE, restore
  is the perturber.

L1 + K1 sources committed on this branch. Plan stays open.

##### P0.A.3 K1' + L2 result — kernel intermediates + per_step_restore both PASS; Suspect 4 falsified (2026-05-21)

Followed up Suspect 4 (per_step_restore semantics) with two binding
tests landed on `production/2026-q2-next`:

- **K1'** (`test-deltanet-save-all-steps-intermediate`) — kernel-direct
  test that `per_step_ssm[k]` from an `n_tokens=5` `save_all_steps=true`
  run byte-equals the `final_state` from a fresh `n_tokens=k+1`
  `save_all_steps=false` run, for ALL k ∈ {0, 1, 2, 3, 4}. **PASS at
  every k**, 262144 fp32 floats per check. The kernel writes the
  correct per-step state at every intermediate iteration; the
  source data `per_step_restore` reads is bit-perfect.

- **L2** (`test-dflash-per-step-restore-byte-identity`) — libllama
  observational test on the load-bearing restore claim: a fresh
  3-token decode + 1-token bonus must produce byte-identical
  bonus-decode logits to a 5-token verify-batch (with
  `save_per_step_ssm` armed) + `llama_spec_ckpt_restore(accepted_step=2)`
  + 1-token bonus. **PASS**: 248,320 fp32 logits byte-identical at
  pos P+3; argmax A=B=13. Both the recurrent-state stitch in
  `kv.per_step_restore()` AND the KV-cache `seq_rm` AND the
  `cells[seq_id].pos` update are observationally equivalent to a
  clean (k+1)-token decode.

###### K1 layout caveat — discovered during K1'

While writing K1', it became clear that the original K1 happens to
read its kernel-supplied beta/g tensors with a layout that doesn't
match the production graph's permuted beta/g. The kernel uses
precomputed strides (`g_stride_batch = n_tokens * n_heads`) that
expect `h`-fast layout, but a default `ggml_new_tensor_4d` with
`ne=[1, n_tokens, H_v, n_seqs]` is contiguous as `t`-fast. The
discrepancy is symmetric across K1's `save_all=true` vs
`save_all=false` runs (both read the same non-production layout
identically), so K1's byte-identity claim ("kernel internal
save_all_steps branch is neutral") still holds. K1' had to fix
this by slicing beta/g in the kernel's coordinate system; the fix
is documented inline as `slice_first_n_tokens_h_fast`.

This caveat does NOT weaken K1's exoneration of "the save_all_steps
branch perturbs the recurrence math." K1' establishes the stronger
claim that the kernel's per-step writes match what a fresh decode
would produce at every k under production-equivalent layouts.

###### What remains

Five suspects now empirically falsified for P0.A.3:

| Suspect | Binding test | Result |
|---|---|---|
| 1. cb_eval install | `test-dflash-extract-observational` | PASS, 2026-05-20 |
| 2. cudaMallocAsync (combine/inject) | run C sync swap | FALSIFIED 2026-05-20 |
| 3. save_per_step_ssm (libllama) | L1 | PASS 2026-05-21 |
| 3'. save_all_steps last-state | K1 | PASS 2026-05-21 |
| 3''. save_all_steps intermediate | K1' | PASS 2026-05-21 |
| 4. per_step_restore byte-identity | L2 | PASS 2026-05-21 |

The combined L2 PASS at the libllama layer rules out the entire
save-then-restore chain for an isolated 5-token verify with
no DFlash pipeline. The CLI failure must therefore involve one
or both of:

- **The drafter pipeline interaction.** combine_features →
  inject_kv_fused × L_d → drafter_forward → drafter_lm_head runs
  during DFlash CLI but NOT in L2. Tests at this layer:
  L3 ("multi-cycle save→restore loop with identity drafter, no
  drafter pipeline") to distinguish multi-cycle drift from
  drafter-pipeline interaction; L4 ("full DFlash drafter pipeline
  with deterministic draft tokens, observational vs spec-none") to
  bind the drafter pipeline's contribution.
- **Multi-cycle state drift.** L2 tests ONE save→restore cycle.
  The CLI runs many cycles in sequence. If the restored state
  accumulates ULP drift cycle-over-cycle, byte-identity at cycle 1
  doesn't preclude argmax flip by cycle N. L3 binds this.

L3 is the cheapest next test (no drafter pipeline, just multi-cycle
loop at libllama). ~20k tokens. If L3 PASSES, the bug needs the
drafter pipeline (L4). If L3 FAILS, multi-cycle restore drift is
the cause.

K1' + L2 sources committed on this branch as regression gates.

##### P0.A.3 L3 result — verify-batch ≠ autoregressive; BATCH-SHAPE VARIANCE bound (2026-05-21)

L3 (`tests/dflash-speculative/test-dflash-multi-cycle-restore-drift`)
was originally written to bind multi-cycle save→restore drift. The
first run produced a much sharper signal: a SINGLE 5-token verify
decode (no spec_ckpt, no DFlash pipeline) produces DIFFERENT per-row
argmaxes than the equivalent 1-token autoregressive decode at the
SAME effective context. The test was rewritten to a clean
batch-shape-invariance probe.

**Result: 10/25 rows mismatched across 5 windows × 5 tokens.**

```
[B w=0 pos=12] tokens=[ 13 271 248068 198 8160 ]
              sampled_at=[ 271 271(!) 198 248046(!) 579 ]
              expected  =[ 271 248068 198 8160     579 ] mismatches=2
[B w=1 pos=17] tokens=[ 579 264 7047 1817 25 ]
              sampled_at=[ 264 7047 1817 271(!)   271 ]
              expected  =[ 264 7047 1817 25       271 ] mismatches=1
[B w=2 pos=22] tokens=[ 271 16 13 220 2972 ]
              sampled_at=[ 16 16(!) 271(!) 16(!) 2014 ]
              expected  =[ 16 13 220 2972 2014 ]       mismatches=3
[B w=3 pos=27] tokens=[ 2014 278 53983 2570 5396 ]
              sampled_at=[ 53983(!) 53983 279(!) 332(!) 64700 ]
              expected  =[ 278 53983 2570 5396 64700 ] mismatches=3
[B w=4 pos=32] tokens=[ 64700 198 256 471 2570 ]
              sampled_at=[ 198 256 471 471(!) 2640 ]
              expected  =[ 198 256 471 2570 2640 ]    mismatches=1
```

Same model, same prompt, same KV cache contents at the start of
each window, same seq_id. The model's forward at row k of an n=5
multi-token batch produces a different argmax than the same model's
forward at row 0 of an n=1 batch with the equivalent prefix. This
is batch-shape variance at the **model** level, not the kernel
level we've been chasing.

###### Why this isn't ruled out by existing infrastructure

The production NP-determinism gates (G3.a/b) verify byte-identity
across concurrent NP={1, 2, 4, 8} dispatches — each slot processing
ONE token per decode. They do NOT exercise the case "one slot
processes 5 tokens in one decode vs. that same slot processing
those 5 tokens one-at-a-time across 5 decodes". L1 + K1 + K1' + L2
together verified the SAVE / RESTORE chain at the verify-batch
shape — but always compared n=5 to n=5. Spec-none autoregressive
runs at n=1. The cross-shape n=5↔n=1 invariance was untested.

###### What this means for the CLI bug

The DFlash CLI's degenerate output is at least partly explained by
the verify-batch decoder producing different — possibly degenerate —
argmaxes than autoregressive would. The CLI's drafter supplies
tokens; verify-batch decides what's "right" by its own argmax, which
deviates from autoregressive. Whatever sequence the verify-batch
produces becomes the CLI output (via the accept/bonus loop). If
that sequence is degenerate, the CLI degenerates.

###### Why MTP appears to work despite the same variance

MTP-IR uses the same verify-batch pattern at production np=1. Two
possibilities, not yet distinguished:
- MTP uses a SMALLER verify-batch shape (e.g. n=4 for --draft 3)
  where batch-shape variance happens to be smaller / absent.
- MTP also exhibits the variance but produces a coherent — though
  non-autoregressive — token sequence, and that's been considered
  acceptable production behaviour because drafter outputs match
  the verify-batch's preference rather than autoregressive's.

A focused follow-up test L3' would sweep verify-batch widths
n_tokens ∈ {2, 3, 4, 5, 6, 8} against autoregressive at the same
context, identifying the smallest n_tokens at which variance begins.

###### Where the variance enters

K1' verified the DeltaNet kernel is byte-equivalent across n_tokens
at the kernel-direct level — same recurrence, same inputs, same
output state. The variance must therefore enter at a layer that
K1' didn't reach:
- **The full-attention layers** (FA per-slot KV singlewarp kernel
  or vanilla FA dispatch). The recent PSKV NPC work (PHASE 40-46
  and friends) optimised this kernel for concurrent multi-slot
  decode — not multi-token same-slot decode. The dispatcher may
  route multi-token same-slot to a code path that wasn't covered
  by NPC byte-identity tests.
- **The graph-build conditional at delta-net.cpp:380-389**: the
  permute/L2-norm ordering branches on `n_seq_tokens > 1`. K1'
  bypassed the graph builder by calling ggml_delta_net directly.
  At the FULL model forward, this branch fires differently for
  n=5 vs n=1. The branch is alleged to be equivalent (line 380
  comment), but it's a candidate.
- **Norm / MLP ops with batch-shape-dependent dispatch** (RMS norm
  reduction tile size, MLP FC kernel selection at small vs medium
  n_tokens). Less likely but possible.

###### Next binding diagnostics

| Test | Binds | Cost |
|---|---|---|
| **L3'** sweep verify_bs ∈ {1..8} vs autoregressive | which n_tokens introduces variance | ~10k |
| **L4** intra-graph tensor capture at FA layer N for n=5 vs n=1 | which op's output diverges | ~30k |
| **L5** disable PSKV → fall back to vanilla FA + repeat L3 | tests whether the PSKV path is the cause | ~20k |

L3' is the cheapest next step. If variance is present at n_tokens=2
already, the bug surface is whatever code path activates for
n_tokens > 1. If variance only appears at n_tokens=5 (DFlash's
specific shape), the bug surface is shape-specific dispatch (likely
a FA tile-size decision).

L3 source committed as a regression gate. The test FAILS on HEAD.
It SHOULD continue to FAIL until the underlying batch-shape
variance is fixed; do not chase the FAIL into the gate logic.

##### P0.A.3 L3' width sweep — variance starts at n_tokens=2 (2026-05-21)

L3' (`tests/dflash-speculative/test-dflash-verify-batch-width-sweep`)
sweeps verify_bs ∈ {1, 2, 3, 4, 5, 6, 8} against the autoregressive
baseline on production Qwen 3.6 27B q4_0 Hadamard dual-GPU.

```
=== L3' verify-batch width sweep ===
verify_bs : mismatches / total
  bs=1    : 0 / 1 PASS  (autoregressive control)
  bs=2    : 1 / 2 FAIL
  bs=3    : 1 / 3 FAIL
  bs=4    : 2 / 4 FAIL
  bs=5    : 2 / 5 FAIL
  bs=6    : 3 / 6 FAIL
  bs=8    : 4 / 8 FAIL
First failing width = 2.
```

Variance starts at **n_tokens=2** — the smallest possible
multi-token same-slot batch. Mismatches scale ~linearly with
width (about half the rows in any batch), consistent with drift
that accumulates through model layers as the batch passes
through.

###### Implication

This is NOT a DFlash-specific bug. ANY same-slot multi-token decode
on this target produces argmaxes that differ from 1-token
autoregressive. The bug surface is "whatever code path activates
when n_tokens > 1 but not at n_tokens = 1". The candidate set is
narrow:

1. **`build_fused_delta_net` graph-build conditional** at
   `llama-delta-net.cpp:380-389`:
   ```cpp
   if (n_seq_tokens > 1) {
       q_conv = ggml_permute(ctx0, q_conv, 0, 2, 1, 3);
       k_conv = ggml_permute(ctx0, k_conv, 0, 2, 1, 3);
       q_conv = ggml_l2_norm(ctx0, q_conv, eps_norm);
       k_conv = ggml_l2_norm(ctx0, k_conv, eps_norm);
   } else {
       q_conv = ggml_l2_norm(ctx0, q_conv, eps_norm);
       k_conv = ggml_l2_norm(ctx0, k_conv, eps_norm);
       q_conv = ggml_permute(ctx0, q_conv, 0, 2, 1, 3);
       k_conv = ggml_permute(ctx0, k_conv, 0, 2, 1, 3);
   }
   ```
   The two branches run L2 norm on contiguous (n=1) vs strided
   (n>1) inputs. If the L2 norm op uses different reduction
   strategies for contiguous vs strided inputs, the fp32 results
   may diverge. Strong suspect — the branch is gated exactly on
   the threshold L3' identifies (n_seq_tokens > 1).

2. **FA per-slot KV singlewarp kernel** at multi-token same-slot
   dispatch. The production NPC gates verify concurrent multi-slot
   single-token; they don't cover multi-token same-slot. The kernel
   may have a different K-tile reduction order for n_tokens=2 vs
   n_tokens=1 inputs.

3. **`ggml_ssm_conv`** invariance across n_tokens. The conv1d
   produces n_tokens outputs from n_tokens inputs + d_conv-1 cached
   states. The output for token t in an n=N batch should equal the
   output for token 0 in an n=1 batch with the same conv state +
   input. Worth a focused unit test.

###### Why MTP-IR appears to work in production

MTP uses `n_seq_tokens > 1` verify batches in production
(--draft 3 → verify_bs=4). If the variance hits MTP, why does
production look fine? Three remaining hypotheses:

- (a) MTP's drafter is trained on the target — it predicts what
  the verify-batch would produce, not what autoregressive would.
  Output sequence == verify-batch decode, which is coherent
  (though different from autoregressive).
- (b) Variance is small at MTP's typical shape and the user has
  never compared to autoregressive ground truth.
- (c) MTP has a workaround we haven't found.

L3' rules out "shape-specific dispatch at verify_bs=5" — it's the
n_tokens > 1 branch in general.

###### Next: localising the divergent op

L4 — per-layer intra-graph tensor capture at production target:
- Decode 1-token autoregressive at position P (single token).
- Decode 2-token verify batch at positions [P, P+1] (row 0 logits
  match L3'/L1 result for first row).
- Capture layer-wise hidden states for each path.
- Diff layer-by-layer to find the first divergent layer.
- Within that layer, diff op-by-op (RMS norm → q/k/v proj → conv1d
  → L2 norm → delta_net → MLP) to find the divergent op.

L4 cost: ~30k tokens (extends the existing extract_layers hook to
capture more layers; reuses `llama_get_dflash_extract_data_seq`).

A targeted shortcut: build the model with the `else` branch of
`llama-delta-net.cpp:380-389` (n_seq_tokens > 1 path) FORCED for
n=1 too — i.e., always permute first then L2 norm — and re-run L3'.
If variance disappears, the graph-build conditional IS the cause.
~5k tokens of diff + rebuild + re-run.

L3' source committed as a regression gate. Test stays FAIL on
HEAD until variance is fixed.

##### P0.A.3 L4 + L5 + FIX — MMQ I=8 batch-shape variance bound and resolved (2026-05-21)

After L3' bound the variance threshold to n_tokens=2, the path
forward was: localise the divergent op (L4) → kernel-direct
isolation (L5) → fix.

###### L4 — per-layer divergence localiser

`test-dflash-per-layer-batch-shape-diff` captured `l_out-<il>` at
every 4th layer of Qwen 3.6 27B for path A (two 1-token decodes
at positions [P, P+1]) vs path B (one 2-token batch at the same
positions). Result:

| Layer | row 0 diffs | row 1 diffs | max \|Δ\| row 1 |
|---|---|---|---|
| 0 | 0/5120 | 5120/5120 | 1.027 |
| 4 | 0/5120 | 5120/5120 | 4.112 |
| 16 | 0/5120 | 5120/5120 | 3.531 |
| 32 | 0/5120 | 5120/5120 | 10.06 |
| 48 | 0/5120 | 5120/5120 | 7.48 |
| 63 | 0/5120 | 5120/5120 | 154.4 |

Row 0 byte-identical across all 17 sampled layers (the autoregressive
control). Row 1 diverges at layer 0 already with |Δ|=1.0, compounding
to |Δ|=154 by layer 63. The bug is INSIDE layer 0.

###### L5 — kernel-direct mul_mat batch-shape probe

`test-mulmat-batch-shape-invariance` ran `ggml_mul_mat(Q4_0 weight,
F32 input)` at production dims (K=5120, N=8192) across ne11 ∈
{1, 2, 8}. Result:

| Comparison | diff | max \|Δ\| |
|---|---|---|
| y_n1(col-0 input) vs y_n2 col 0 vs y_n8 col 0 | 0 / 8192 | 0 |
| y_n1(col-1 input) vs y_n2 col 1 | 8192 / 8192 | 0.363 |
| y_n2 col 1 vs y_n8 col 1 | 0 / 8192 | 0 |

**MMQ Q4_0 is byte-shape-invariant for OUTPUT column 0 only.**
Columns ≥ 1 in a multi-token batch produce different fp32 bits than
the same input vector at column 0 of a single-token dispatch. The
existing `LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH` (now always-on per
the `ggml-cuda.cu:2747-2754` comment) only guarantees column-0
invariance.

###### Diagnostic isolation: I=8 split-K kernel

The MMQ I=8 split-K path (`mul_mat_q_split_k_i8` with `mma_int_C_I8J8`
fragment) was added for sm_75 decode-shape TG perf (PHASE 71-74
"MMQ I=8: NPC + perf verify"). The NPC verification compared col 0
across concurrent NP={1..8} — it did NOT exercise the col j>0 path
that the verify-batch decoder uses.

Setting `i8_shape_supported = false` in `launch_mul_mat_q` (at
`ggml/src/ggml-cuda/mmq.cuh:4986`) makes mul_mat fully
batch-shape-invariant. L5 immediately PASSES across all columns.

###### Closure: every binding test now PASSES

After the fix on `production/2026-q2-next` (submodule HEAD `8e233e9b`):

| Test | Pre-fix | Post-fix |
|---|---|---|
| L1 libllama save_per_step_ssm observational | PASS | PASS |
| K1 ggml_delta_net save_all_steps last-state | PASS | PASS |
| K1' ggml_delta_net intermediate per-step state | PASS | PASS |
| L2 per_step_restore byte-identity | PASS | PASS |
| L3 multi-cycle save→restore drift | FAIL | PASS |
| L3' verify-batch width sweep ∈ {1..8} | FAIL | PASS |
| L4 per-layer batch-shape diff (17 layers × 2 rows) | FAIL | PASS |
| L5 mul_mat batch-shape invariance | FAIL | PASS |

`dflash-speculative-simple` CLI on the production target (lm_head-f16):

- Before: degenerate "...quick quick quick...", mean accept 0.29-1.91/4.
- After: coherent — "The capital of France is Paris. The capital of
  Germany is Berlin. The capital of Italy is Rome. The capital of
  Spain is Madrid. The capital of Portugal is Lisbon. The capital of
  Greece is Athens..." Mean accept 2.30/4. 33 tokens at 18.68 t/s.

###### Trade-off: decode TG perf

The I=8 path was added for sm_75 decode TG perf (~25% → ~56-100%
theoretical occupancy bid). Falling back to regular MMQ costs some
decode TG but restores correctness for speculative-decoding verify
batches. The original NPC verification (PHASE 74) was a partial
check that missed col-j>0 invariance — it should be re-run after
this fix to confirm the regular MMQ path's NPC byte-identity is
unaffected.

To re-enable I=8 in the future, the kernel needs to be fixed so the
col-j>0 FMA accumulation order matches col-0's. This is a
non-trivial rewrite of the `mma_int_C_I8J8` fragment's K-loop and
per-thread accumulator ownership.

###### Post-fix bench

`llama-batched-bench` on production geometry (Qwen 3.6 27B q4_0
Hadamard, dual-GPU, c=4096, npp=200 ntg=64 npl=8):

| Configuration | TG NP=8 (t/s) | Δ vs prior |
|---|---|---|
| HEAD pre-fix (I=8 enabled, broken DFlash) | 27.73 | baseline |
| HEAD post-fix (I=8 disabled, correct DFlash) | 24.14 | **−12.95%** |
| Pre-NPC ceiling (no PSKV NPC work) | 36.68 | — |

The fix costs ~13% TG NP=8 vs the prior NPC-enforced baseline. **G3.h
binding gate (±1% of 27.73) is BROKEN by this fix**; it should be
relaxed to ±15% or have a documented exception until MMQ I=8 is
properly fixed.

NPC byte-identity smoke (`quick-pskv-npc-check.sh`): **PASS**. All
NP={1,8} slots byte-identical to NP=1 baseline, cross-NP slot-0
byte-identical. The production NP-determinism gates are unaffected
by this fix.

###### Trade-off and next perf recovery work

The user's trade is correctness vs perf. With I=8 disabled,
speculative decoding (MTP and DFlash) actually produces correct
output. The TG NP=8 perf hit is significant but recoverable by:

1. Rewriting `mul_mat_q_split_k_i8` / `mma_int_C_I8J8` so that
   per-column FMA accumulation order matches col 0's. This is
   pure kernel work, no algorithmic changes. Token cost ~50-100k.
2. OR pad single-token decodes to n_tokens=2 at the libllama
   dispatcher to force "col 1" path uniformly. Avoids the kernel
   rewrite at the cost of 2× the per-decode mul_mat work. May
   be cheaper than option 1 if I=8 stays superior to regular MMQ.

Decision: defer perf recovery to its own phase; ship correctness.

##### P0.A.3 follow-on — NPC test-coverage audit + cross-shape SAME-SLOT gate (2026-05-21)

The MMQ I=8 col-j>0 bug shipped through PHASE 71-74's NPC verification
because the existing harness exercises only the cross-slot axis at
n_tokens=1 per slot — every slot is col 0 of its own dispatch. The
cross-shape SAME-SLOT axis (n_tokens=1 vs n_tokens=N for the same
slot) was structurally untested.

Closure:

- `tests/dflash-speculative/test-mulmat-batch-shape-invariance.cpp`
  (kernel-level, no model load): sweeps `ggml_mul_mat(Q4_0, F32)`
  across (K, N) ∈ {prod-qkv 5120×8192, prod-model-dim 5120×5120,
  small-square 2048×2048} × ne11 ∈ {1, 2, 5, 8, 16, 32} and asserts
  every output column byte-identical to the ne11=1 reference.
  Catches MMQ tile regressions in <60s.
- `tests/dflash-speculative/test-dflash-verify-batch-width-sweep.cpp`
  (libllama-level): sweeps verify_bs ∈ {1..8} at the libllama API
  surface; argmax-equal across widths.
- `tests/dflash-speculative/test-dflash-multi-cycle-restore-drift.cpp`
  (libllama-level): verify-batch row-k argmax vs autoregressive at
  the same effective context.
- `scripts/test-batch-shape-invariance.sh` orchestrates the three
  sub-tests above plus the cross-mmq_x dispatch test (below).
- `scripts/verify-production-determinism.sh` chains the new gate
  after the existing NPC harness; both must pass to ship.

Result on the I=8-disabled HEAD: all four sub-tests PASS.

##### P0.A.3 follow-on — cross-mmq_x dispatch byte invariance closed (2026-05-21)

L5's cross-shape sweep surfaced a residual fp32 ULP divergence at
ne11=32 (max |Δ| ~5e-6 at K=5120). Root cause traced to the MMQ
dispatcher at `mmq.cuh:5168-5186` selecting a compile-time `mmq_x`
tile size per ne11, and the split-K factor at `mmq.cuh:4974` being:

```
constexpr int split_k_factor = (mmq_x <= 16) ? 4 : 1;
```

Tiles for ne11 ≤ 16 routed to `mul_mat_q_split_k<...,4>` (K split
into 4 chunks summed in fp32 by a fixup pass); tiles for ne11 > 16
routed to plain `mul_mat_q` (K summed in a single sequential pass).
Mathematically equivalent, bit-different by ULP magnitude.

Test-first closure:

- New kernel test
  `tests/dflash-speculative/test-mulmat-mmq_x-dispatch-invariance.cpp`
  sweeps ne11 ∈ {1, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 96, 128,
  256, 512}, builds 512 single-column references at ne11=1, and
  asserts every dispatch column byte-identical to its reference.
  On HEAD pre-fix: FAILED 12/15 (PASS at ne11 ∈ {1,8,16}; FAIL at
  every tile transition from ne11=24 upward). Confirms the
  mechanism.

Fix (one line):

```diff
-constexpr int split_k_factor = (mmq_x <= 16) ? 4 : 1;
+constexpr int split_k_factor = 4;
```

at `ggml/src/ggml-cuda/mmq.cuh:4974`. Unifies the K-axis reduction
structure across every MMQ tile. The split-K kernel and fixup were
already templated on `mmq_x` and `split_k_factor`, so this just
instantiates additional templates for `mmq_x ∈ {24,32,...,128}`
that were previously dead.

Gates post-fix:

| Test | Pre-fix | Post-fix |
|---|---|---|
| `test-mulmat-mmq_x-dispatch-invariance` ne11 ∈ {1..512} | 3/15 PASS | 15/15 PASS |
| `test-mulmat-batch-shape-invariance` (ne11=32 promoted from informational to binding) | informational diff | binding PASS |
| `quick-pskv-npc-check` NP={1,8} | PASS | PASS |
| `test-batch-shape-invariance.sh` (4 sub-tests) | 3 PASS, 1 N/A | 4/4 PASS |

Perf delta (llama-batched-bench, Qwen 3.6 27B q4_0 Hadamard,
dual-GPU, c=4096, npp=200 ntg=64 npl=8):

| Configuration | TG NP=8 (t/s) | PP NP=8 (t/s) |
|---|---|---|
| HEAD pre-fix (I=8 disabled, split_k_factor (mmq_x≤16) ? 4 : 1) | 24.15 | 24.56 |
| HEAD post-fix (I=8 disabled, uniform split_k_factor=4) | 24.14 | 24.55 |
| Δ | −0.04% | −0.04% |

Inside measurement noise. G3.h binding gate is unaffected (the prior
P0.A.3 I=8-disable cost of −12.95% vs the broken-but-fast pre-fix
baseline still stands; that's the gate-relaxation conversation, not
this fix).

Why Option A (uniform split_k_factor=4) is free: the original
`mmq_x ≤ 16` restriction was an under-restriction. Split-K's
SM-occupancy lift applies at every mmq_x where the grid
under-saturates TU102's 72 SMs. At very-large mmq_x where the grid
is already saturated, the fixup-pass overhead (3 fp32 adds per
output element + one tiny kernel launch) is dominated by the
K-reduction work it parallelises. The bench confirms directly.

Lesson banked in MEMORY: "informational" findings should be closed
properly when the mechanism is diagnosable, not deferred behind an
"argmax dampens it" rationale. ~30k tokens converted a known latent
issue into a clean fix plus a permanent regression gate.

##### P0.A.3 Suspect 2 result — `save_per_step_ssm = true` perturbs the verify decode (2026-05-20)

> **SUPERSEDED 2026-05-21** by the L1 + K1 binding tests above (both
> PASS on HEAD). The Run E correlation below was real but NOT causal:
> isolating the save side at the libllama layer (L1) and at the
> kernel layer (K1) shows the verify decode is byte-identical with
> save_per_step_ssm armed vs disarmed. The CLI's Run E output
> difference must come from how the drafter pipeline interacts with
> per_step_restore() or with the alternative shadow-only path —
> NOT from save_per_step_ssm itself. The "kernel-level bug" framing
> in this section is wrong. New suspect (Suspect 4: per_step_restore
> semantics) is named in the L1 + K1 result section above. Section
> kept as audit trail; do not act on its "next steps" or the K2/K3
> rows in the ladder.

> **Suspect 2 confirmed at coarse grain.** A series of env-gated
> diagnostic patches in `src/llama.cpp` spec_ckpt code produced
> meaningfully different outputs at the same prompt/seed/temp/n:
>
> | Test | save_per_step_ssm | per_step_restore | Output start |
> |---|---|---|---|
> | A — HEAD | true | runs | `...quick quick quick...` (degenerate, accept=1.91) |
> | D — `LLAMA_NO_SPEC_CKPT_RESTORE=1` | true | skipped | `...<think>\n\n\n</think>\n\n a a a...` (different degenerate, accept=2.47) |
> | E — `LLAMA_NO_SPEC_CKPT_SAVE=1` | false | runs (reads stale buf) | `...<think>\n\n# How to\n# How to use of the 1...` (partially coherent, accept=0.29) |
>
> The fact that flipping `save_per_step_ssm` between true and false
> CHANGES the output materially proves the two paths through
> `ggml_delta_net` are NOT byte-equivalent. Specifically:
>
> - At `src/llama-delta-net.cpp:73`:
>   `save_per_step_states = lctx.transformer_kv.save_per_step_ssm && batch.n_tokens > 1;`
> - At `src/llama-delta-net.cpp:125`:
>   `ggml_tensor * fused_result = ggml_delta_net(ctx0, q, k, v, g, beta, state_flat, save_all_steps);`
> - At `src/llama-delta-net.cpp:140-142`: when `save_all_steps`, `new_state` is
>   the LAST per-step state (`output_size + (n_tokens - 1) * state_size`);
>   when not, it is at offset `output_size`. These should be the SAME
>   mathematical value (the recurrent state after processing all n_tokens
>   steps).
>
> They are not.
>
> **Why the libllama observational test passed but the CLI fails**:
> the libllama test never sets `save_per_step_ssm`. The DFlash CLI
> sets it on every cycle via `llama_spec_ckpt_save`. The verify decode
> shape is identical (5-row multi-token) in both — only the
> save_per_step_ssm flag differs.
>
> **Suspect 2 root location**: the `ggml_delta_net` CUDA kernel's
> `save_all_steps=true` path is numerically divergent from the
> `save_all_steps=false` path at the LAST step. They should agree
> bit-exact (the kernel runs the same sequential recurrence either way;
> save_all_steps only changes whether intermediate states are written
> to the output tensor). The next binding test is a libllama-level
> harness that runs the same multi-token decode with
> `save_per_step_ssm` on vs off and asserts argmax equality — the
> equivalent of the cb_eval observational test but for save_per_step_ssm.
>
> If that confirms (very high prior given the above), the fix is in
> the DeltaNet CUDA kernel — make the save_all_steps=true path emit
> the same last state as save_all_steps=false. This is a kernel-level
> fix downstream of the post-fold 4D KV port and adjacent to PHASE 67-69's
> batched-pinned changes.
>
> Diagnostic env-gates reverted; submodule clean.

##### P0.A.3 Suspect 1 result — `cudaMallocAsync` FALSIFIED (2026-05-20)

> **Suspect 1 falsified.** Diagnostic patches replaced
> `cudaMallocAsync`/`cudaFreeAsync` with synchronous `cudaMalloc`/
> `cudaFree` in `dflash-combine-features.cu` and `dflash-inject-kv.cu`,
> rebuilt, ran the same prompt/seed/temp:
>
> | Run | Output start | n_emitted | drafts | accepted | t/s |
> |---|---|---|---|---|---|
> | A — HEAD (cudaMallocAsync) | `...quick quick quick quick...` | 64 | 88 | 42 | 16.80 |
> | C — sync cudaMalloc | `...quick quick quick quick...` | 64 | 88 | 42 | 16.76 |
>
> Byte-identical degenerate output. Identical stats. The
> async-allocator jitter is not the perturbation source despite the
> auto-memory P3.X.B precedent. Diagnostic patches reverted; submodule
> clean.
>
> Suspect priority updated:
> 1. ~~cb_eval install + scheduler slow-path~~ (falsified)
> 2. ~~`cudaMallocAsync` jitter~~ (falsified)
> 3. **Suspect 2 — `llama_spec_ckpt_restore` under post-fold 4D KV** (next)
> 4. Suspect 3 — `stage_target_hiddens` post-trim alignment
> 5. Suspect 4 — `trim_extract` × `stage_target_hiddens` interaction
> 6. Suspect 5 — drafter K/V pointer arithmetic (downgrade)

##### P0.A.3 next experiment + code review of candidate suspects (2026-05-20)

This subsection is the plan and code-review writeup that precedes
the next experiment. It is intentionally folded into PHASE_NSTREAM_KV_PERF.md
rather than carried in a new doc so the falsification + plan + review
sit on one page.

###### Next experiment — A/B on `dflash-speculative-simple` with cb_eval install force-disabled

The previous matrix never ran the experiment that isolates cb_eval at
the actual CLI level. That is the cheapest binding step. Plan:

1. Build the existing `examples/dflash-speculative-simple/dflash-speculative-simple.cpp`
   (already wired). It is a self-contained DFlash CLI driver — prefill
   + per-cycle (drafter_forward → 5-token verify decode → sample-accept
   → spec_ckpt_restore → trim_extract). No HTTP / scheduler / multi-slot
   complexity. Single-slot, np=1.
2. Capture baseline output at a fixed prompt / seed / `--n-predict 64`,
   intact binary. Expected: degenerate output matching the documented
   P0.A.3 symptom ("user user wants wants quick quick").
3. Patch the binary: at `src/llama.cpp:10072` change
   `ctx->cparams.cb_eval = llama_dflash_extract_cb_eval;` to
   `ctx->cparams.cb_eval = nullptr;` (also clear `cb_eval_user_data`),
   rebuild, re-run with the same args. Outcome A: divergence
   SURVIVES → cb_eval install is conclusively exonerated at the CLI
   level (consistent with the libllama-level falsification). Outcome
   B: divergence DISAPPEARS → cb_eval IS the cause in some way the
   libllama-level test missed (very unlikely given the three test
   shapes already passed, but the binding A/B is what closes the
   diagnostic either way).
4. Either outcome closes the cb_eval line of investigation. If A,
   move to the candidate suspects below. If B, instrument the
   scheduler split boundary to find which fusion the slow path
   perturbs at the CLI shape.

Cost: 1 build + 2 runs (~3 min each) ≈ 15-20 minutes wallclock.
Token cost: 10-20 k for the experiment + writeup.

If outcome A, do NOT revert the cb_eval install — the libllama tests
that rely on the extract surface (test-dflash-extract-multi-seq,
the various capture-driven harnesses) still need it. The patch is a
temporary diagnostic-only edit; the conclusion is what we record.

###### Code review — candidate suspects downstream of cb_eval

The DFlash CLI per-cycle flow (reading `examples/dflash-speculative-simple/dflash-speculative-simple.cpp:137-258`):

```
prefill prompt (cb_eval fires)
  → sample first token id_last
  → loop:
       spec_ckpt_save (DeltaNet recurrent shadow + save_per_step_ssm on)
       common_speculative_draft (drafter forward → 4 candidate tokens)
       build verify batch [id_last @P, c1 @P+1, ..., c4 @P+4]
       llama_decode (verify, cb_eval fires)
       sample argmax per row → sampled_at[0..4]
       n_accepted = longest prefix where draft[k] == sampled_at[k+1]
       spec_ckpt_restore (DeltaNet rewind + seq_rm rejected positions)
       trim_extract (drop rows past P + n_accepted + 1)
       emit n_accepted drafts + bonus
       id_last = bonus
```

The libllama test that exonerated cb_eval exercised the target
forward only. It did NOT exercise: drafter forward, combine_features,
inject_kv_fused, spec_ckpt_save/restore, trim_extract, or
common_speculative_draft's position math. Any one of those is a
candidate downstream of cb_eval.

Suspect ranking (highest first) with citations:

**Suspect 1 (HIGH) — `cudaMallocAsync` jitter in combine_features and inject_kv_fused.**

Citations:
- `ggml/src/ggml-cuda/dflash/dflash-combine-features.cu:134,156` —
  `cudaMallocAsync(&fc_out_f32, ...)` and `cudaFreeAsync(...)` per
  combine call (once per cycle).
- `ggml/src/ggml-cuda/dflash/dflash-inject-kv.cu:202,203,234,235` —
  TWO `cudaMallocAsync` (K and V projections) and matching frees per
  layer. For Qwen3.6-27B-DFlash with L_d=4 drafter layers, that is
  8 alloc/free pairs per cycle, 9 total per cycle including combine.

Why this is suspect: auto-memory record P3.X.B (task #38) is
"`cudaMallocAsync` — NPC FAIL 1/8 stochastic. Bench TG@NP=8 =
28.45 t/s (+5.0 %)". The same async-pool API has previously been
confirmed to cause stochastic non-determinism on this hardware /
driver stack. The DFlash pipeline now uses it on every cycle. Pool
allocation pointers can vary call-to-call (size class fragmentation,
prior frees not yet returned to pool, etc.) — and even though the
F32 scratch is written-then-read in the same stream, different
backing memory between cycles can change peer-access / TLB / coherence
state in ways that perturb downstream kernel timing and produce
ULP-level differences in F32 reductions.

Cheap binding test: replace both `cudaMallocAsync` sites with
`cudaMalloc`/`cudaFree` (synchronous, deterministic pool) for one
diagnostic run; if divergence disappears, the suspect is confirmed
and the fix is to pre-allocate the scratch ONCE in `alloc_ctx_scratch`
sized to `n_slots_cap * MAL_max * D_kv` and reuse it (no per-cycle
alloc). Cost: ~20 k tokens.

**Suspect 2 (HIGH) — `llama_spec_ckpt_restore` under post-fold 4D KV.**

Citations:
- `src/llama.cpp:8505-8520` — PER_STEP restore:
  - `kv.per_step_restore(accepted_step)` (DeltaNet recurrent rewind)
  - `kv.cells[seq_id].pos = accepted_pos`
  - `llama_kv_cache_seq_rm(kv, seq_id, accepted_pos + 1, -1)`
- `src/llama-context.h:76` — `std::vector<llama_kv_cell> cells` —
  cell metadata array.
- `examples/dflash-speculative-simple/dflash-speculative-simple.cpp:224` —
  CLI invocation site, called every cycle after sample-accept.

Why this is suspect: the post-fold 4D KV layout puts K/V cells in
per-stream slices. Whether `cells[seq_id].pos = accepted_pos` and
`llama_kv_cache_seq_rm(...)` correctly scope to the seq_id's stream
slice has not been re-validated since PHASE_NSTREAM_KV landed. The
PER_STEP path was authored pre-fold (DFlash T6.α) for the shared-pool
layout. If the seq_rm bound is wrong, rejected positions stay live
in cache → next cycle's verify decode attends to stale K/V → wrong
logits → drift.

Cheap binding test: add an assertion at the end of `llama_spec_ckpt_restore`
that the K/V for positions `> accepted_pos` is logically empty for
this seq's slice. Use the existing `LLAMA_KV_CONCURRENT_TRACE`
instrumentation (PHASE_NSTREAM_KV Bug C work) to dump
`kv.cells[i]` for the seq's slice after restore in a few cycles.
Cost: ~15-25 k tokens.

**Suspect 3 (MEDIUM) — `stage_target_hiddens` post-trim staging order.**

Citations:
- `src/llama-dflash.cpp:566-626` — function definition.
- `src/llama-dflash.cpp:600-605` — F32 → __half host stage of rows
  `[0, mal_anchors)` BEFORE the trim at line 614.
- `src/llama-dflash.cpp:614` — `buf.resize(mal_anchors * D_emb)`
  (P0.A.2 fix landed earlier this session).

Why this is suspect: the per-layer loop reads
`buf.data() + a * D_emb` for `a ∈ [0, mal_anchors)`. After a verify
decode appended 5 rows past the prior cycle's `mal_anchors`, the
buffer has `prev_mal_anchors + 5` rows. The new cycle's `mal_anchors`
is `prev_mal_anchors + n_accepted + 1`. The trim at line 614 truncates
to the new mal_anchors AFTER staging. The staging reads the FIRST
mal_anchors rows of the buffer — but those rows include the verify
decode's appended hiddens at indices `[prev_mal_anchors, prev_mal_anchors + n_accepted]`
which correspond to positions `[P, P+n_accepted]` (the anchor +
accepted drafts). That IS what the drafter wants for the next cycle.
Probably correct, but the alignment is delicate; an off-by-one in
the anchor row (do we include the anchor's hidden from the verify
batch's row 0, or the prior cycle's bonus-decode hidden?) could
silently produce a one-row drift that compounds across cycles.

Cheap binding test: add a per-cycle log that prints `buf.size() / D_emb`,
`mal_anchors`, and the first-row hidden norm before stage. Compare
across two consecutive cycles. If the row at index `prev_mal_anchors`
is the bonus-decode's hidden vs the anchor's verify hidden — that's
the off-by-one. Cost: ~10-15 k tokens.

**Suspect 4 (MEDIUM) — `llama_dflash_trim_extract` × `stage_target_hiddens` interaction.**

Citations:
- `src/llama-dflash.cpp:885-941` — function definition. PER_STEP
  semantics: `p_end < 0` truncates to first `p_start` rows.
- `examples/dflash-speculative-simple/dflash-speculative-simple.cpp:235` —
  CLI call: `llama_dflash_trim_extract(ctx, P + n_accepted + 1, -1)`.
- `src/llama-dflash.cpp:614` — `stage_target_hiddens`'s own trim
  inside the per-layer loop.

Why this is suspect: there are now TWO trim sites — `trim_extract` at
the CLI level (after spec_ckpt_restore) AND `stage_target_hiddens`
post-stage trim (P0.A.2 restored). If they disagree on the row count,
the buffer is over-trimmed (drafter sees too few rows next cycle)
or under-trimmed (drafter sees stale rejected hiddens). The
`stage_target_hiddens` trim runs INSIDE the next cycle's stage call,
which is AFTER `trim_extract` has already run. If
`trim_extract(P + n_accepted + 1, -1)` already produced
`buf.size() / D_emb == mal_anchors_next`, the per-layer trim at
line 614 is a no-op (idempotent). But if either of the row counts is
off by one, the no-op becomes a corruption.

Cheap binding test: assert
`buf.size() / D_emb == mal_anchors` at the START of
`stage_target_hiddens` (before staging). If it fails, the trim
sites disagree. Cost: ~5-10 k tokens.

**Suspect 5 (MEDIUM) — drafter K/V cache alias with target context after post-fold.**

Citations:
- `src/llama-dflash.cpp:728` — `n_kv_per_layer = n_slots_cap * SeqLen * H_kv * D_h`.
  The drafter K/V cache lives in `st.d_k_cache, st.d_v_cache` — separate
  CUDA allocations from the target's K/V.
- `src/llama-dflash.cpp:768` — `st.d_k_cache + l * n_kv_per_layer + s * kv_slot_stride`
  pointer arithmetic.

Why this is suspect: the drafter cache is sized at `llama_set_dflash`
time from `seq_len_cap = swa_window + block_size + 16`. The
production capacity is 2080 (per P0.A.1 fix). The drafter writes to
this cache via `inject_kv_fused` at the anchor positions, and reads
during its forward kernel via `slot_positions`. The drafter cache
is entirely separate from the target's 4D KV — no direct alias is
possible. But: the drafter's `inject_kv_fused_launch` is called for
EACH source layer `l ∈ [0, L_d)`. If the per-layer base pointer
math `(std::size_t) l * n_kv_per_layer + s * kv_slot_stride` overflows
or wraps for large `l × n_slots_cap × SeqLen`, layer L_d-1's writes
could clobber layer 0's reads on the next cycle. For Qwen3.6-27B-DFlash:
L_d=4, n_slots_cap=2 (production), SeqLen=2080, H_kv=8, D_h=128
→ per-layer bytes = 2080 × 2 × 8 × 128 × 2 = 8.5 MiB. Total drafter
K cache = L_d × per-layer = 34 MiB. No overflow risk at `size_t`.
The arithmetic looks clean; downgrade this suspect to LOW unless
the others fail.

###### Suspect priority + sequencing

1. Run the **A/B on dflash-speculative-simple** first (15-20 min,
   10-20 k tokens). This either closes cb_eval forever or reopens
   it as a different mechanism.
2. If divergence survives, attack **Suspect 1 (`cudaMallocAsync`)**
   next — cheapest binding test (replace with `cudaMalloc`, one
   diagnostic run), strongest prior (P3.X.B is the established
   precedent for `cudaMallocAsync` stochastic non-determinism on
   this hardware).
3. Then **Suspect 2 (spec_ckpt_restore)** — requires more
   instrumentation but addresses the post-fold integration gap most
   directly.
4. Then **Suspect 3** and **4** in parallel — both are simple
   asserts on the extract buffer state.
5. Suspect 5 only if 1-4 all fail.

###### What we ship along the way

- No code changes from this code review yet — read-only audit.
- The A/B experiment edit at `src/llama.cpp:10072` is a **diagnostic-only**
  patch, not committed. After capturing both outputs, revert the
  edit.
- Findings from each binding test land as an append-only MEMORY.md
  entry + an inline update to THIS subsection. If a suspect is
  CONFIRMED, the fix lands as a separate commit with its own
  PHASE_NSTREAM_KV_PERF.md update.

**Symptom after P0.A.1 + P0.A.2 landed:**

Same prompt (`"Write a quicksort in Python."`), same temperature, same seed, np=1, ctx=65536, side-by-side token capture from `/completion`:

```
spec-none (n_predict=30, temp=0.0, seed=42):
  '\n\n<think>\nHere\'s a thinking process:\n\n1.  **Understand User Request:**
   The user wants a "quick quiz in C". This'

dflash    (n_predict=30, temp=0.0, seed=42):
  '\n\n<think>\n  - The **UserUser**:**: The user wants is asking asking for
   for a a " quick quick quick quick quick quick'
```

The two outputs **match byte-identical for the prefill+first-decode prefix** (`\n\n<think>\n`), then diverge at the first multi-token verify-batch decode. The DFlash output collapses into a degenerate `quick quick quick…` loop.

This was reproduced under **multiple cache configurations** to isolate variables:

| Cache config           | Hadamard | Temp | Output      | Accept |
|------------------------|----------|------|-------------|--------|
| Q4_0 K / Q4_0 V        | yes      | 0.0  | degenerate  | 54 %   |
| Q4_0 K / Q4_0 V        | yes      | 0.6  | degenerate  | 7.8 %  |
| f16 K / f16 V          | no       | 0.0  | degenerate  | 14.9 % |

The degenerate pattern survives the matrix. **Quantization, Hadamard rotation, and the GEMV-vs-GEMM accumulation-order divergence are all ruled out as root causes**.

The MTP production path on the same target model + same Q4_0+Hadamard cache produces clean output (production verified `np=1 --draft 3` MTP throughput 33.5 t/s). Since MTP and DFlash share `server_context::speculative_decoding_accept` and the `add_sampled_tokens` verify-batch construction, the bug is somewhere DFlash-specific — most likely in either the drafter's input-construction half (`stage_target_hiddens` / `inject_kv` positional semantics) or the cb_eval hook's interaction with the verify-batch causal mask.

**Why P0.A.3 cannot close in this conversation:** the bug requires capturing target logits at the first divergent position with and without DFlash drafting, plus tracing K/V writes and reads through a single verify cycle. That is dedicated kernel-level diagnostic phase work — significantly more than P0.A's verification scope and not addressable from this conversation's remaining context budget.

##### Theories for P0.A.3 — what the dedicated phase should test

The theories below are non-exclusive. The dedicated phase should pick the cheapest binding test for each, in the order shown.

###### Theory T1 — KQ mask off-by-one in multi-token verify batch

**Mechanism.** The verify batch contains `1 + n_draft` tokens at absolute positions `[N, N+1, ..., N+K]`. The causal mask should restrict each Q position to attend only to K positions ≤ itself. If the mask under the post-fold 4D KV path **leaks** future positions into the attention window for the anchor — even just position `N+1`'s K into position `N`'s attention — then logits at position `N` (which gate the token at position `N+1`) are computed against a context that includes the *drafted* token at `N+1`.

```
Correct causal mask for verify batch [P, P+1, P+2, P+3, P+4]:
                ┌──────────────────────────┐
                │ K-pos →   P P+1 P+2 P+3 P+4
                │ Q-pos ↓
                │ P         X . . . .          ← P attends ONLY to P
                │ P+1       X X . . .
                │ P+2       X X X . .
                │ P+3       X X X X .
                │ P+4       X X X X X
                └──────────────────────────┘

Hypothesised broken mask (off-by-one):
                ┌──────────────────────────┐
                │ K-pos →   P P+1 P+2 P+3 P+4
                │ Q-pos ↓
                │ P         X X . . .         ← P leaks attention to P+1
                │ P+1       X X X . .
                │ P+2       X X X X .
                │ P+3       X X X X X
                │ P+4       X X X X X
                └──────────────────────────┘
```

If T1 holds: position-`N` logits depend on the drafted token at position `N+1`. Whatever the drafter predicted becomes self-fulfilling — target's `argmax` at position `N` is biased toward continuing the drafter's prediction. This would explain why "quick quick quick" reinforces itself: each cycle the drafter predicts `quick`, the leaked mask makes the target's logits also favor `quick`, the verify accepts, and the loop continues.

**Cheap binding test.** Dump the KQ mask tensor from the verify-batch `ggml_cgraph` for a slot with `n_draft=4`. Compare to the dumped mask for the same slot with `n_draft=0` (spec-none). They should be identical at the overlap (the anchor row). If they differ, that's the bug.

Files in scope: `src/llama.cpp` graph build for Qwen35 — search for `KQ_mask` construction with `n_tokens > 1` per stream; the 4D port's `[n_kv, n_tokens/n_stream, 1, n_stream]` mask shape (see `PHASE_NSTREAM_KV` STATUS.md note on KQ-mask shape).

###### Theory T2 — Drafter input positions vs cache_tokens cardinality mismatch

**Mechanism.** The drafter is told its input shape via `(anchor_id, anchor_pos)`. `anchor_pos` is `cache_tokens.size()` at the start of the cycle (capped at MAL_max). The drafter combines features from `extract_buf` rows `[0, anchor_pos)`, treating those rows as target hiddens at "anchor positions" `[0, 1, ..., anchor_pos-1]`.

Each row in `extract_buf` is a `l_out-<il>` hidden written by the cb_eval hook for one position in one target decode. The mapping is "row index in buf ↔ absolute target context position" — and that mapping is preserved ONLY if `stage_target_hiddens`'s end-trim (P0.A.2 fix) keeps `len(buf) == cache_tokens.size()` at every cycle boundary.

```
Healthy invariant after each cycle's accept + stage_end_trim:

  Absolute target positions:  0  1  2  …  P-1  P  P+1
  cache_tokens[i]:            t0 t1 t2 …  t_{P-1}  s  d0
                              ───────prompt──────   ↑   ↑
                                                anchor accepted-draft

  extract_buf rows:           h0 h1 h2 …  h_{P-1}  h_s h_d0
                              (1 row per absolute target position)

Bug variant — buf has one extra row from a prior partial-accept that
escaped the end-trim, OR one missing row from a cb_eval that fired for
a rejected position that was then trimmed:

  cache_tokens[i]:            t0 t1 t2 …  t_{P-1}  s  d0
  extract_buf rows:           h0 h1 h2 …  h_{P-1}  h_s h_d0  h_REJ  ← extra
                              row index → absolute target position shifts
                              by ±1 from this point onward
```

If T2 holds: drafter sees target-hidden rows whose row index no longer matches their absolute target position. Drafter's anchor positions `[0, anchor_pos-1]` mean different things to the drafter and to the target. Drafter's RoPE and attention compute against shifted positions. Predictions are systematically wrong.

**Cheap binding test.** Instrument `speculative_decoding_accept` + `stage_target_hiddens` to log `(cycle_n, cache_tokens.size(), buf.size() / D_emb)` per cycle. Assert equality at every cycle boundary. If violated, dump the cycle where they diverge.

###### Theory T3 — cb_eval hook fires on rejected-draft positions and the trim isn't tight enough

**Mechanism.** Related to T2 but more specific. The cb_eval hook is registered as `ggml_backend_sched_eval_callback` and fires for every evaluated `l_out-<il>` tensor — including positions in the verify batch that correspond to rejected drafts. After accept:

```
Verify batch positions appended to buf during target verify decode:
  position:   N   N+1   N+2   N+3   N+4
  token:      s   d0    d1    d2    d3
  cb_eval:    h0  h1    h2    h3    h4    ← all 5 rows appended

If accept: m drafts accepted, m-1 < n_draft.

  cache_tokens (after accept):  …, s, d0, d1, …, d_{m-2}, [slot.sampled=ids[m-1]]
                                size = prev + m
                                                     ↑ NOT in cache_tokens

  Desired buf size (next cycle): prev + m
  Actual buf size pre-trim:      prev + 1 + n_draft

  P0.A.2 fix trims to mal_anchors = prev + m — drops last (n_draft + 1 - m) rows.
```

If P0.A.2's trim is correct, this all balances. But if the trim drops `n_draft - m` rows instead of `n_draft + 1 - m` (off-by-one), then one stale row survives every cycle and accumulates over time — explaining why early cycles produce *somewhat* sensible tokens before the loop degenerates.

```
Cycle counter vs row drift (off-by-one variant):
  cycle:   1   2   3   4   5  …
  drift:   1   2   3   4   5  …  ← per-cycle drift accumulates
```

**Cheap binding test.** Same as T2 — log buf.size() / D_emb vs cache_tokens.size() per cycle. T3 distinguishes from T2 by *monotonic linear drift* with cycle number. T2 would show stable mismatch (constant offset). T1 would show no row mismatch — the bug would be elsewhere.

###### Theory T4 — Drafter K/V positional encoding mismatched to target's

**Mechanism.** The drafter applies RoPE at "anchor positions" `[0, anchor_pos-1]` and Q positions `[anchor_pos, anchor_pos+BS-1]`. These are the drafter's *internal* positions. Target's K/V applies RoPE at absolute target positions `[N, N+1, ..., N+K]`.

If the drafter's anchor_pos at cycle K equals the target's absolute position N (which it does — `anchor_pos = cache_tokens.size() = N`), the positions align. **But if the drafter's K/V cache uses position-`x` semantics that don't match the kernel's RoPE position-`x` semantics** (e.g. the drafter inject kernel writes K at absolute position but the forward kernel reads K at position % SeqLen), then drafter's K/V is RoPE-encoded inconsistently with what its forward attention expects.

```
Drafter inject (writes K with RoPE at absolute position `position`):
  K_cache[slot, position % SeqLen, h, d]  = RoPE(K_proj, position) [h, d]
                          ↑
                          With SeqLen >= max position, modulo is no-op.

Drafter forward attention (reads K at qpos minus offsets in window):
  K_read = K_cache[slot, k_idx % SeqLen, h, d]    ← if forward reads modulo
  K_read = K_cache[slot, k_idx, h, d]             ← if forward reads absolute
                          ↑
                          MISMATCH if inject uses modulo and forward doesn't,
                          or vice versa.
```

If T4 holds: forward attention reads K/V whose RoPE encoding is for a DIFFERENT absolute position than what the kernel computes against the Q. Attention scores would be ~random. Drafter would predict noise.

For T4 to explain the specific "quick quick" loop, the noise would need to be biased — possibly because the wrong-RoPE'd K/V vectors point to embeddings that disproportionately match certain tokens after lm_head.

**Cheap binding test.** Compare drafter K_cache indices used during inject vs read for a single cycle. Add an assertion that inject's `cache_idx == k_read_idx` for the same `position`. The PHASE_NSTREAM_KV_PERF discussion above already considered modulo addressing for SWA — if the kernels ARE modulo-aware but the current code uses absolute, this would surface.

###### Theory T5 — cb_eval hook side-effect on verify-batch graph compute

**Mechanism.** cb_eval calls `ggml_backend_tensor_get` per matched `l_out-<il>` tensor — a device-to-host DMA. This forces stream synchronization on the CUDA stream the graph is running on.

For the verify batch (1 + n_draft = 5 tokens), the graph contains multiple `l_out-<il>` nodes (5 source layers × 5 positions = 25 evaluations per verify decode). Each fires a synchronizing DMA.

If the synchronization interacts with cudaGraphExecUpdate (PHASE 36/37/38 machinery) in a way that invalidates the *non-extracted* graph nodes' state — e.g., async K/V writes haven't completed when the next position's Q reads — then K/V at the just-written position would contain garbage. The Q at position `N+1` would attend to a half-written K at position `N`.

```
Timeline within one verify decode:
  ┌─ position N graph nodes execute ─┐
  │   K_proj(s), V_proj(s) → K[N], V[N]
  │   l_out-1 evaluation               ─┐  cb_eval fires.
  │                                    │  ggml_backend_tensor_get
  │                                    │  forces sync.
  │   Async write to K[N] in flight    ←  Sync may complete before
  │                                       OR concurrent with Q[N+1] read.
  └────────────────────────────────────┘
  ┌─ position N+1 graph nodes ──────────┐
  │   K_proj(d0), V_proj(d0) → K[N+1], V[N+1]
  │   Q[N+1] @ K[0..N+1]  ←  attends to K[N] which may be partially written
  └─────────────────────────────────────┘
```

**Cheap binding test.** Disable cb_eval temporarily (return early from `llama_dflash_extract_cb_eval`) and run the smoke. If DFlash output now matches spec-none, T5 holds and the fix is to fence cb_eval reads behind a cudaStreamSynchronize per layer.

###### Theory T6 — cudaGraphExecUpdate machinery + DFlash dynamic batch shape

**Mechanism.** The 4D port + DFlash both stress cudaGraphExecUpdate machinery. The CUDA graph cache (`ggml-cuda.cu:4500-4830`) is keyed by topology hash. A verify batch of `1+K` tokens has a different topology than the prefill (`P` tokens) and the next single-token decode (`1` token).

If the multi-entry cache evicts the WRONG entry under DFlash's batch shape sequence (FIFO under MAX=128 cap), graph reuse may use a *stale* graph instance whose `src_address` pointers were last patched for a different batch shape.

```
Graph-cache state across DFlash cycles:
  cycle 0 (prefill, n_tokens=P):    GRAPH_A captured + instantiated
  cycle 1 (verify,  n_tokens=1+K):  GRAPH_B captured + instantiated
  cycle 2 (verify,  n_tokens=1+K):  GRAPH_B reused → cudaGraphExecUpdate
                                    src_addr for K/V → new positions
  ...
  cycle N (verify,  n_tokens=1+K'): GRAPH_C? cache eviction?
                                    OR GRAPH_B with stale K/V address?
```

If T6 holds: K/V writes happen but to stale buffer addresses; subsequent verify cycles read from those stale addresses; logits become drifting noise.

**Cheap binding test.** Disable the CUDA graph cache (`GGML_CUDA_DISABLE_GRAPHS=1`) and re-run smoke. If output now matches spec-none, T6 holds.

##### Falsification matrix (run 2026-05-20)

| Theory | Test | Result | Verdict |
|--------|------|--------|---------|
| T1 | MTP code-path-share check | MTP fused uses a DIFFERENT mask path (line 4454, gated on MTP_OP_DRAFT_GEN_FUSED). MTP-works ≠ standard-multi-token-verify-works. | reopened |
| T2 / T3 | `DFLASH_DIAG=1` + 6 cycles of buf_rows vs MAL trace | buf_rows tracks `MAL + 1 + n_draft - m` formula exactly per cycle. No drift, no off-by-one. | **falsified** |
| T4 | Kernel index audit (inject vs forward read) | Both use `slot * SeqLen + position` form, no modulo, consistent. | **falsified** |
| T5 | `--spec-type ngram-simple --draft-max 4` (same standard verify-batch path, NO cb_eval hook) | **Output byte-identical to spec-none baseline**: `'\n\n<think>\nHere\'s a thinking process:...'`. The standard multi-token verify path works correctly without the DFlash cb_eval hook. | **CONFIRMED** |
| T6 | `GGML_CUDA_DISABLE_GRAPHS=1` | Output byte-identical to enabled-graphs run (still degenerate). | **falsified** |

Verifications that further support T5:
- `--draft-max 1` (verify batch n_tokens=2): output byte-identical to `--draft-max 4` degenerate output. So the bug is not batch-shape-specific within multi-token, only "DFlash vs other-spec multi-token" specific.
- ngram-simple uses the standard `add_sampled_tokens` + standard verify-batch construction + standard `speculative_decoding_accept`. The ONLY architectural difference from DFlash is the `cb_eval` hook installation via `llama_set_dflash_extract_layers` at `src/llama.cpp:10067-10073`.

#### P0.A.3 — Confirmed root cause: `cb_eval` hook perturbs target's forward pass

The cb_eval hook (`llama_dflash_extract_cb_eval` at `src/llama.cpp:9961-10049`) is registered as `cparams.cb_eval` for any context where DFlash is bound. It fires on every evaluated tensor in the target's decode graph; when the tensor matches a configured source-layer's `l_out-<il>` node (layers 1, 16, 31, 46, 61 for the Qwen 3.6 drafter), it returns `true` to claim ownership for inspection.

```
ggml_backend_sched_eval_callback invocation per node:

  1st call (ask=true)   ──┐
                          │   if return true:
                          │     - scheduler ISOLATES this node
                          │     - may break operator fusion
                          │     - may force separate buffer allocation
                          │     - may force sync before subsequent reads
                          └─→  ┐
  2nd call (ask=false)  ───→   ggml_backend_tensor_get
                               (device → host DMA)
```

The standard multi-token verify decode (DFlash, ngram, draft model spec-types) all use the same `add_sampled_tokens` + verify-batch construction. Only DFlash installs the cb_eval hook. ngram-simple's output matches spec-none exactly → standard verify path is correct. DFlash's output diverges → hook is the perturbation.

The most likely mechanism: returning `true` on `ask=true` for `l_out-<il>` nodes prevents the CUDA backend from fusing those nodes with adjacent ops (RMS-norm + residual-add + attention input projection are otherwise fusable). The unfused path is numerically different — slightly different intermediate accumulations through 65 layers cascade into argmax flips at the output.

**Why it doesn't affect MTP**: MTP uses fused draft generation (`MTP_OP_DRAFT_GEN_FUSED`) which has its own dedicated mask + dispatch path (`llama.cpp:4454`) and never goes through the standard verify-batch path with cb_eval installed. Production MTP works because it never exercises the affected fusion path under hook observation.

#### P0.A.3 fix paths (in increasing scope)

1. **Re-architect hidden-state capture to NOT use cb_eval.** Add tap nodes inside the graph builder for the 5 source layers (e.g., `ggml_dup` to a dedicated output buffer that lives outside the fused region). The drafter reads from those buffers after `llama_decode` returns. Estimated 30-50 k tokens. Cleanest fix; cb_eval hook removed entirely.

2. **Conditional hook arming.** Only install cb_eval during the prefill (when capturing prefill hiddens) and detach it before verify decodes. Drafter still has prefill data but post-prefill hidden updates are missed. Acceptable if MAL is bounded — drafter operates on a sliding window already. Estimated 15-30 k tokens.

3. **Investigate the specific fusion broken and find a minimal workaround.** Identify which CUDA backend fusion is disrupted by `cb_eval=true`. Patch the backend to fuse-around the inspection points (or use a different inspection mechanism such as `ggml_backend_tensor_get_async` after compute completes). Estimated 50-100 k tokens if backend changes needed.

Per `feedback_no_workarounds`, path 1 is the proper fix. Path 2 is a stop-gap if production needs DFlash CLI before path 1 lands. Path 3 is the most invasive (touches ggml backend).

#### P0.A.4 — Multi-slot DFlash SEGV in `llama_sampling_prepare` [CLOSED 2026-05-21]

> **Closed 2026-05-21.** Reproduced clean on post-P0.A.3 HEAD with
> `qwen36-27b-x2-dflash` profile + two concurrent `/v1/completions`:
> server crashes during the first verify cycle with
> `llama_get_logits_ith: invalid logits id 5, reason: batch.logits[5] != true`.
>
> Root cause: the per-stream split in `process_batch_tokens` (added
> 2026-05-19 by `PHASE_NSTREAM_KV_4D` N2.d) issues one `llama_decode`
> per slot per tick (Bug C-safe). The engine resets `output_ids` on
> every `llama_decode` AND indexes it in the LOCAL frame
> `[0..n_tokens)` of the dispatched `batch_view`, not the GLOBAL
> frame of the combined `batch`. The in-loop call to
> `speculative_decoding_accept()` walked **all** slots and read
> their logits via `slot.i_batch_dft[k]` — which is recorded in the
> GLOBAL frame. Only the most-recently-decoded slot's indices
> resolved against `output_ids`; every other slot hit
> `output_ids[N] = -1`, which the engine reports as
> `batch.logits[N] != true`. At np=2 with `--draft-max 4`, slot 1's
> i_batch_dft[0] = 5 falls outside slot 0's just-decoded `output_ids`
> window, so the engine throws on the first attempt to sample for
> slot 1.
>
> Fix (submodule commit `cad6b591`):
>
> - `examples/server/server-context.h` — signature change
>   `speculative_decoding_accept(int32_t batch_offset, int run_seq_id)`.
> - `examples/server/server-context.cpp:4161-` — filter Phase A's
>   slot iteration by `slot.id == run_seq_id`; translate
>   `i_batch_dft` from global to local frame
>   (`g - batch_offset`) before any
>   `llama_get_logits_ith` / `llama_decoder_get_embeddings_ith`
>   call.
> - `examples/server/server-context.cpp:4947` — call site passes
>   `(i, run_seq_id)` from the per-stream loop's current iteration.
>
> The np=1 two-phase split (Phase A reads, Phase B mutates) is
> preserved trivially: `accepted` now holds at most one slot per
> call.
>
> Regression test (parent commit `a12609a`):
> `scripts/test-server-multi-slot-dflash.sh` boots the production
> DFlash profile at `--parallel 2`, sends two concurrent
> `/v1/completions`, asserts both return coherent text. RED on
> pre-fix HEAD (curl 52, server log
> `batch.logits[5] != true`); GREEN post-fix.
>
> Gates post-fix:
>
> | Gate | Pre-fix | Post-fix |
> |---|---|---|
> | `test-server-multi-slot-dflash.sh` (np=2 DFlash smoke) | FAIL (server crash) | PASS |
> | `verify-production-determinism.sh` NP={1,2,4,8} per-slot byte-identity | PASS | PASS |
> | `verify-production-determinism.sh` cross-NP slot-0 matrix | PASS | PASS |
> | `verify-production-determinism.sh` batch-shape invariance gate (4 sub-tests) | PASS | PASS |
>
> Original section preserved below as the diagnostic record.

##### P0.A.4 historical diagnostic — original symptom record

**Symptom at `--parallel 2`:**

```
llama_get_logits_ith: invalid logits id 5, reason: batch.logits[5] != true
SIGSEGV at: llama_sampling_prepare
  ← llama_sampling_sample_impl
  ← common_sampler_sample_and_accept_n
  ← server_context::speculative_decoding_accept
  ← server_context::process_batch_tokens
```

The verify batch for 2 slots × `(1 + n_draft = 5)` tokens = 10 batch positions. Position 5 is slot 1's anchor. `batch.logits[5] = false` suggests the multi-slot verify batch isn't enabling logits at every position the sampler will index into.

This is independent of P0.A.3 and likely contained to the `add_sampled_tokens` multi-slot loop in `server-context.cpp`. Estimated scope: 30-80k tokens once P0.A.3 is closed (the SEGV may also be a symptom of T1/T2/T3 — verify position-mismatch math more carefully under np>1 before assuming an independent bug).

#### What this phase actually depends on

Tier 2 and Tier 3 do NOT depend on DFlash CLI working end-to-end. Production runs MTP `--draft 3` at np=1 today; MTP is the gated production path for both correctness (Bug C absence) and perf (the -6.2 % TG NP=8 regression we're trying to recover comes from per-stream graph rebuilds in the vanilla decode path, not from DFlash). **Tier 2's gate GP3.n binds on MTP NP=1 production smoke**, not on DFlash CLI smoke.

Tier 2 entry condition (updated 2026-05-21 after P0.A.3 + P0.A.4 closure):
- ✅ P0.A.1 (MAL cap) — landed.
- ✅ P0.A.2 (stage end-trim) — landed.
- ✅ P0.A.3 (output divergence) — CLOSED 2026-05-21 (MMQ I=8 disable + cross-mmq_x dispatch uniformity).
- ✅ P0.A.4 (multi-slot SEGV) — CLOSED 2026-05-21 (per-stream-scoped `speculative_decoding_accept` + local-frame index translation).
- ✅ P0.A.5 (production server boots clean on post-fold for the non-DFlash production profile) — verified.

All P0.A items now strictly closed. P0.B is the remaining Tier-2 prerequisite.

Gates GP3.e ("DFlash test suite GREEN") in Tier 2/3 bind at the libllama layer only — the `test-dflash-np-multislot` family in `tests/dflash-speculative/`, which already passes on current HEAD. Server-side DFlash gates are deferred with P0.A.3.

#### Token estimate revision

- P0.A.1 + P0.A.2: ~30 k actual (delivered).
- P0.A.3 dedicated phase: ~100-300 k (scope is "root cause + fix a deep DFlash + 4D KV composition bug"; range is wide because deep into kernel/cb_eval interaction).
- P0.A.4: ~30-80 k once P0.A.3 closes (or subsumed).

### P0.B — Radical spec / TLA+ / test surface expansion

**Why this is a prereq, not parallel work.** The existing S1-S5 spec layer (`batch_composition.allium`, `n_stream_layer.allium`, `BatchComposition.tla`, `StreamIsolation.tla`) was scoped to **what the 4D port preserved** — Bug C absence, per-stream allocator, mask isolation. It does not cover:

1. The CUDA-graph reuse / `cudaGraphExecUpdate` invariants we now depend on at Tier 2.
2. The unified-stream dispatch semantics Tier 3 introduces.
3. The composition of MTP fused × n_stream > 1 × per-stream view patching.
4. The composition of DFlash multi-slot × unified verify-side × graph reuse.
5. The dispatch-mode invariants the DFlash server-CLI fix (P0.A) must preserve.
6. The warm-up determinism contract (P2 memory flagged this as a divergence vector; gate GP3.g binds it but no spec exists).

Past in-tree perf work (tasks #37/38/39 — cudaMallocAsync NPC stochastic 1/8) is the historical reason this expansion is non-optional: that failure mode was not catchable by any S1-S5 test because no spec existed for "CUDA runtime state ordering under per-tick perf changes." If we ship Tier 2/3 against the current spec surface and hit a stochastic-NPC class regression, we will not catch it cheaply.

Expansion targets — each is a parallel of an existing S1-S5 artifact:

**P0.B.S1 — Allium specs (5 new + 1 tend):**
- `specs/graphs/cuda_graph_reuse.allium` — contracts for the multi-entry topology-hash cache, `cudaGraphExecUpdate` patch semantics, dtype-strict invariant, VIEW/CPY src_address tolerance, `GGML_CUDA_GRAPH_MAX` cap + FIFO eviction. Source-tied to `ggml-cuda.cu` lines 4500-4830.
- `specs/kv-cache/per_stream_read_view_patching.allium` — extension of S1.b's per-stream contracts to the K/V *read* views in `llm_build_kqv` (currently NOT patched per-stream — the Tier 2 gap). `ReadViewStreamPatching` contract mirrors S1.b's `PerStreamAllocator` shape.
- `specs/dispatch/unified_stream_dispatch.allium` — Tier 3 contracts. `UnifiedUbatchInvariant` (one llama_decode/tick spans N streams via ne[3]); `UniformShapePerTick` (mul_mat sees one shape per call, not N shapes within a tick — the Q1 composition argument formalised); `PreservesBugCAbsence` (derived from BatchCompositionInvariant under unified dispatch).
- `specs/dflash/dflash_server_cli.allium` — contract for `--spec-type dflash` orchestrator invocation; `SidecarSharesTargetTokenizer`; `OrchestratorInvokedFromServerCLI`. Binds P0.A's wiring.
- `specs/composition/mtp_fused_x_n_stream.allium` — MTP fused × n_stream > 1 composition. `MTPFusedReuseSurvivesStreamPatching` (Phase 37 #5 reuse branch composes with Tier 2's update_cache_copies extension at NP > 1).
- `specs/mtp_fused_draft.allium` **tend** — reference the new cross-cutting contracts from `unified_stream_dispatch.allium` and `mtp_fused_x_n_stream.allium`. No new contracts inline; pure cross-reference.

**P0.B.S2 — TLA+ specs (3 new + 1 extension):**
- `specs/graphs/CUDAGraphReuse.tla` — state machine over `{Captured, Instantiated, ExecUpdated, Invalidated}`; actions `Capture`, `Update`, `Evict`. Invariant: `DtypeStrictness` (no graph reused across a dtype change). Liveness: `EventualEviction` (FIFO under cap).
- `specs/dispatch/UnifiedStreamDispatch.tla` — state machine for the unified ubatch builder; admission of new prefill streams, fan-out across the n_stream axis. Invariants: `UnifiedUbatchSeqIdsAreUnique`, `UniformShapePerTick`, `BugCAbsencePreserved`.
- `specs/composition/MTPxNStream.tla` — composition state machine. Variables: `mtp_op_type`, `n_stream`, `chain_residual`. Invariant: `MTPChainResidualPerStream` (n_streams chain_residual buffers each tracked independently).
- `specs/dflash/DFlashMultiSlot.tla` **extend** — add `VerifySideUnification` action covering Tier 3's verify-side dispatch change; preserve existing `NoCrossSlotRegionOverlap` invariant.

**P0.B.S3 — TLC model checking (every new spec + negative-tests):**
- For each new `.tla`: `MC.cfg` clean run + `MC_negative.cfg` that disables the invariant the spec binds and confirms TLC produces the expected counterexample. Mirror the existing `BatchCompositionMC.cfg` + `BatchCompositionMC_no_gate.cfg` pattern.
- Specifically the negative test for `UnifiedStreamDispatch.tla` must produce the Bug C signature when `UniformShapePerTick` is disabled — proving the spec captures the predecessor's closure mechanism.

**P0.B.S4 — Property tests (allium propagate):**
- `tests/spec/test-cuda-graph-reuse.cpp` — exercise the `is_cuda_graph_update_required` + `ggml_cuda_graph_node_props_eq` checks at the dtype boundary, VIEW src_address change, op-sequence change. PASS on HEAD.
- `tests/spec/test-per-stream-read-view-patching.cpp` — exercise `update_cache_copies` after Tier 2 extension lands; FAIL on HEAD (before Tier 2), PASS after. Binding RED test.
- `tests/spec/test-unified-stream-dispatch.cpp` — exercise Tier 3 dispatch; FAIL on HEAD (before Tier 3), PASS after. Binding RED test.
- `tests/spec/test-mtp-x-n-stream.cpp` — exercise MTP fused at NP > 1; PASS on HEAD (MTP NP > 1 not used in production but composes correctly).
- `tests/spec/test-dflash-server-cli.cpp` — drives `llama-server` with `--spec-type dflash`. PASS after P0.A lands.

**P0.B.S5 — Trace harness expansion:**
- Extend `examples/server/server-trace-ndjson.h` with `emit_graph_event` (`CaptureGraph`, `UpdateGraphExec`, `EvictGraphCacheEntry`). Gate on `LLAMA_TRACE_NDJSON_DIR`.
- Extend `scripts/validate-batch-composition-trace.py` to validate the new graph-event records against `CUDAGraphReuse.tla`.
- Extend the harness to emit `WarmUpRunIndex` markers so the validator can verify R1=R2=R3 byte-identity for GP3.g.

**P0.B gates (closure):**
- **GP0.B.a** — All new `.allium` files `allium check` clean.
- **GP0.B.b** — All new `.tla` files SANY-parse clean; `MC.cfg` runs clean; `MC_negative.cfg` produces the expected counterexample for each.
- **GP0.B.c** — All new property tests build + run via `cmake --build build --target <test-name>`. RED tests fail on pre-Tier-2 HEAD (proves they bind on what Tier 2 will deliver).
- **GP0.B.d** — `LLAMA_TRACE_NDJSON_DIR=...` `r5-probe-c4.sh ITERS=20` produces traces; `scripts/validate-batch-composition-trace.py` reports zero violations across BatchComposition + StreamIsolation + CUDAGraphReuse + UnifiedStreamDispatch.
- **GP0.B.e** — `allium weed` against current HEAD finds expected gaps for the unmlanded contracts (`ReadViewStreamPatching`, `UnifiedUbatchInvariant`) and clean for the already-landed ones.

Token estimate: 25-40 k Allium + 30-50 k TLA+ + 15-25 k TLC + 20-30 k property tests + 15-25 k trace harness = **105-170 k tokens**.

### P0 sequencing

P0.A and P0.B can run in parallel. P0.A is contained scope (verify post-fold DFlash CLI on the existing profile + commit `61a7e874`); P0.B is broad surface expansion. Both must close before T2.a probe begins.

**Phase 0 closure**: GP0.A.a–d and GP0.B.a–e all GREEN. Submodule + parent commits per CLAUDE.md §5. MEMORY entry per §6.

---

## Retrospective: Tier 1 + Tier 2 (2026-05-21) — revised hypothesis for Tier 3

This section supersedes the original Tier 1 and Tier 2 framings below. Both were authored 2026-05-20 against premises that empirical work falsified one week later. The §"Direction tree" Tier 1/2 entries are preserved for audit but should be read with this retrospective in mind.

### Tier 1 — falsified

**Original premise (2026-05-20):** drop the `n_stream > 1` bailout, accept ~N cudaGraphInstantiate at warm-up, then ride cudaGraphExecUpdate for steady-state reuse. Estimated **+6.2% TG recovery** at NP=8.

**Falsification (2026-05-21):** measured A/B at production bench shape (`llama-batched-bench -npp 200 -ntg 64 -npl 8` dual RTX 6000, back-to-back):

| | T_PP s | S_PP t/s | T_TG s | S_TG t/s | total t/s |
|---|---|---|---|---|---|
| graphs ON | 65.638 | 24.38 | 21.371 | 23.96 | 24.27 |
| graphs OFF (`GGML_CUDA_DISABLE_GRAPHS=1`) | 65.498 | 24.43 | 21.332 | 24.00 | 24.32 |

**Δ ≈ 0% (graphs-off marginally faster, within noise).** Recorded in `data/ralph-nstream-kv-ledger.md` row 21 + commit `9f2a016`.

**Why the original estimate was wrong:** the PSKV singlewarp kernel costs ~127 µs/call (ledger row 19, ncu). The cudaLaunchKernel + cudaGraphInstantiate overhead is ~5-10 µs per kernel. With 24 layers × multiple ops per layer at our model size, per-launch CPU cost is amortised below the noise floor. **There is no recovery to be had from preserving graph reuse at our shape.**

**Net:** Tier 1 doesn't exist as a separable value-add lever. Don't sequence it.

### Tier 2 — parked with intel

**Original premise (2026-05-20):** patch the K/V attention-read views per-stream in `update_cache_copies()`, mirroring the existing CPY write-side patching, then drop the bailout. Estimated **+15-30% beyond Tier 1** from full reuse retention.

**Status (2026-05-21):** scaffolding wired and committed (`030a0f04` … `d07e0e16` … `c2a142a4`); behaviourally inert because the bailout was restored after the mechanism failed to deliver NPC under bailout-dropped. Three indirection variants were tried; all produced identical divergent output. Root cause was not localised under the original diagnostic protocol.

**FA-probe diagnostic (2026-05-21):** environment-gated probe in the PSKV launcher hashed K/V/mask bytes at FA entry, with bailout env-toggleable. Compared NP=1 baseline vs NP=8 stream-0 under bailout-active and bailout-dropped at layer `flash_attn_per_slot_kv-1003`.

Findings (full table in user-local memory `project_tier2_diagnostic_findings.md`, captured at ralph row 20 + commit `bcab99d`):

- **Bailout-active NP=8 stream-0:** byte-identical to NP=1 across all hashes. Baseline holds.
- **Bailout-dropped NP=8 stream-0:** K and V cache-byte hashes **diverge from bound=10 onwards**. Mask hash is **byte-identical to NP=1**. `per_row_k_bound[0]` is **byte-identical to NP=1**. K view `view_offs` is **correct per-stream** (verified directly: streams 1..7 show `K_off = stream_id × parent->nb[3]`).
- **All 8 slots produce uniquely garbled outputs under bailout-dropped** — interleaving-order-dependent corruption, not a uniform shared error.

**What this tells us:**

- The Tier 2 patching mechanism works at the LAYER it operates on: view offsets and data pointers are updated correctly per-tick, the read-view registry is well-formed, the FA op's op_params slot routing is consistent.
- The bug is at the **K/V cache byte level at the correctly-addressed stream slice**. Some prior write put bad bytes into stream 0's slice between its bound=9 and bound=10 FA calls — but stream 0 didn't run in that interval. Streams 1..7 did.
- The mask construction is sound (`set_inputs` reads `cells[]` via `_kv_mask_base = (head / kvps) * kvps`; `cells[].pos` and `cells[].seq_id` are consistent with bailout-active).
- The most plausible mechanism (not directly confirmed; would require ggml-backend-sched-internal instrumentation):
  - **Sched intermediate-buffer aliasing across cross-stream graph reuses.** The K-projection scratch buffer (or similar intermediate) has its allocation determined at graph-build time based on the single-stream execution that built the graph. When stream M reuses stream N's graph, the scratch buffer's lifetime overlaps with stream N's CPY-into-cache read, producing the wrong K cache contents.
  - Alternative: cuda-stream ordering. CPY writes happen on one ggml-backend-sched-managed cuda stream, FA reads from another, without proper cross-stream synchronisation under reuse.
- **The structural problem is that "graph reuse across stream switches" implicitly assumes the graph's buffer-lifetime contract still holds when the execution pattern changes.** It does not. Patching tensor pointers fixes the addressing layer but not the buffer-lifecycle layer.

**Combined with Tier 1 falsification:** even if we had localised and fixed the corruption, Tier 2 would have bought ~0% perf (per the A/B). The original +15-30% estimate was anchored on Tier 1's +6.2% as a baseline; that baseline is also ~0%.

**Decision:** park Tier 2. Keep the scaffolding committed as benign no-op (it's documented; T3 Step 1 will remove it). Do not pursue further root-cause work on the corruption — the bug is real but solving it has no value once Tier 3 lands.

### Revised Tier 3 hypothesis — perf-anchored

**Drop:** any reliance on graph reuse as a perf lever. The A/B falsifies this independently of any Tier 2/Tier 3 specifics.

**Dominant lever: kernel batching at the ne[3] = n_stream axis.** Each kernel launch processes ~n_stream× more useful work per call. SM utilisation scales sublinearly with the batch dimension, but at our current per-slot decode (n_tokens=1, 1 CTA per attention head per layer per slot, ~64 CTAs total per layer) we are *extremely* under-utilising TU102's 72 SMs × 2 GPUs = 144 SMs. Going to ne[3]=8 multiplies the per-launch CTA count by 8 (or more, depending on layout) at no per-launch overhead cost.

**Secondary lever: amortised host-side overhead.** `set_inputs` walks `cells[]` once per `llama_decode` to build the mask + per_row_k_bound; the sched setup runs per call; the graph topology hash + cache lookup runs per call. Even though *cuda-graph* cost is ~0%, these *host-side* costs are not free. Going from 8 `llama_decode` calls per tick to 1 reduces them 8×.

**Tertiary: the Tier 2 corruption class is excluded by construction.** Tier 3 has no cross-tick graph reuse across stream switches because all streams run in one graph per tick. The implicit buffer-lifetime contract is honoured — buffers live for the duration of one execution, which now covers all streams.

**Quantitative anchors:**

- Current production NP=8 aggregate TG: **27.73 t/s** (ledger row 7 baseline, hadamard-on profile). Today's same-shape `llama-batched-bench` measurement: **23.96 t/s** (no-hadamard CLI path — ~14% gap to baseline likely from clock state or hadamard contribution, not load-bearing for the A/B comparison).
- Per-slot at NP=8: ~3.46 t/s.
- NP=1 single-slot baseline: ~33.5 t/s (no contention).
- **vLLM measured on the same hardware** (Q4 same quant, no Hadamard): **154.77 t/s aggregate at NP=8** (`data/gate0-np1-np8.json`, ralph row 11; details in user-local memory `project_continuous_batching_vs_perslot_dispatch.md`). Per-slot: 19.3 t/s.
- **The vLLM gap is 5.6× aggregate, 5.6× per-slot.** vLLM achieves it on unified-batch dispatch. They don't have Hadamard or our Q4_0 KV; we don't lose those under Tier 3. The dispatch model is the difference, not the kernel quality.

**Realistic projection for Tier 3 closure:**
- **Conservative target: 100 t/s aggregate** (3.6× current). Anchored to GP3.i.
- **Stretch target: 130 t/s aggregate** (4.7× current, 85% of vLLM's measurement).
- Both targets are sub-vLLM because we retain Hadamard + Q4_0 KV (small per-op cost), accept some MoE-vs-dense overhead, and don't assume we match vLLM's kernel-fusion depth on first cut.

**Risk register for Tier 3:**

1. **mul_mat GEMV → GEMM dispatch boundary at n_tokens=8.** The original Bug C was triggered by mixed shapes within a tick. Tier 3's per-tick batch is *uniform* shape — all tokens at ne[1]=n_tokens_per_tick. Within a tick, mul_mat is called once at this shape; the GEMM/GEMV pick is consistent. **More uniform than current per-slot dispatch** (which calls mul_mat 8× at 8 different per-stream shapes). Verified by GP3.f (`r5-probe-c4.sh` ITERS=20 = 0/20).
2. **KQ_mask construction at 4D shape.** Mask must address `cells[stream_offset + i]` correctly. T2 diagnostic showed the existing per-stream mask code is byte-correct; T3 needs it at `[n_kv, n_tokens_per_seq, 1, n_stream]`. T3.c.
3. **DFlash multi-slot composition under unified verify-side dispatch.** The draft side is already batched (Phase 5). Verify side currently uses per-stream dispatch; Tier 3 unifies it. Cleaner mental model, but the DFlash test gates (GP3.l) must pass.
4. **T9 NP=4/8 drift signature.** Pre-PHASE_NSTREAM_KV ne[1]>1 attention had a mma_f16 tile-transition drift. PSKV per-slot kernel handles this — production routes to `wmma_f16_case_pb1<256,256,8,half>` with fp32 accumulation and parallel_blocks=1 pinned. Validated by GP3.k.

**What this hypothesis is NOT betting on:**
- It is NOT betting that graph reuse saves anything (falsified).
- It is NOT betting that we can localise and fix the Tier 2 corruption.
- It is NOT betting on persistent-kernel megakernels or any sm_75-incompatible vLLM lift.

### What's left undone before Tier 3 begins

**Tier 2 scaffolding** (committed, behaviourally inert because bailout is active). Tier 3 Step 1 (next session) removes:

- `cache_read_views` registry in `src/llama-context.h` (the `CacheReadView` struct + vector).
- `update_cache_copies` per-stream read-view patch loop in `src/llama.cpp:782-795`.
- FA op `op_params[14]`/`[15]` K/V slot fields + `ggml_set_fa_indirect_slots` / `ggml_get_fa_K/V_indirect_slot` helpers in `ggml/include/ggml.h` and `ggml/src/ggml.c`.
- PSKV kernel `src_ptr_table` / `K_slot_idx` / `V_slot_idx` parameters and the indirection path in `fattn-per-slot-kv-singlewarp-sm75.cu`.
- ggml-cuda.cu per-tick read-view-src-ptrs refresh hook (lines 4668-4720).
- Per-stream view registration in `llm_build_kqv` / `build_std_attention` (the `ggml_set_read_view_indirect_slot` calls).

**The bailout itself** at `src/llama.cpp:610-630`. Tier 3 removes this (it becomes structurally unreachable — `n_stream > 1` simply means more rows in the unified ubatch).

**T3 Step 1 (already landed at submodule `c2a142a4`):** PSKV singlewarp kernel takes `nb33` and addresses mask as `mask + nb33 × seq + nb31 × tok`. At `ne[3]=1` this collapses to legacy behaviour (`nb33 × 0 = 0`); at `ne[3]>1` it routes to the active stream's mask block. **Foundation for the unified-stream dispatch path is in place.**

**Phase 0.B (Allium/TLA+ spec layer):** partial — review at Tier 3 start whether the existing spec contracts (`StreamPartition`, `MaskPerStream`, `PerStreamDispatch`, `BugCAbsenceByConstruction`, `DFlashCompatibility`) need extension for the unified-ubatch case. T3.a probe of upstream PR #14363's `split_equal` semantics may surface new contracts.

**Phase 0.A (DFlash CLI fix):** orthogonal to Tier 3; doesn't block. Production runs MTP, not DFlash CLI.

---

## Direction tree (refined and primary-sourced)

| Tier | What | Expected gain | Effort | Risk |
|------|------|---------------|--------|------|
| 1 | ~~Drop `can_reuse_graph` n_stream > 1 bailout + accept rebuild storm at first stream-switch~~ **FALSIFIED 2026-05-21** | ~~+6.2 % recovery~~ **~0%** (A/B confirms cuda graphs gain ≈ 0% at production shape) | n/a | n/a |
| 2 | ~~Patch attention-read view offsets per-stream in `update_cache_copies`~~ **PARKED 2026-05-21** (mechanism wired but byte-corrupts K/V cache under bailout-dropped; intel captured) | ~~+15-30 %~~ **~0%** (since T1's baseline-recovery premise is false) | parked | parked |
| **3** | **Unified-stream dispatch: one `llama_decode` per tick with all streams packed at ne[3]=n_stream; use existing PSKV per-slot kernel** | **Approaches vLLM's measured 154.77 t/s aggregate at NP=8** (4.75× over current 27.7 t/s baseline). Conservative 100 t/s, stretch 130 t/s. Dominant lever: kernel batching, NOT graph reuse. | 3-5 weeks | Medium-High |
| 4 | + chunked-prefill admission ([Sarathi-Serve](https://arxiv.org/abs/2403.02310)) | small additional gain on prefill-heavy workloads | +2 weeks | Low after Tier 3 |
| 5 | Full paged-KV port (V1 vLLM kernel, sm_70+) | Marginal beyond Tier 3 for our workload | 6+ months | Very high |

**SKIP**: vLLM pivot (loses Q4_0 + Hadamard + Q4_0 KV — uniquely SoTA on sm_75; nothing else in the ecosystem can consume this weight stack); persistent-kernel megakernels (Luce sm_75 batch=1 only; Mirage Hopper-first).

### Tier 1 — Drop the bailout — **[FALSIFIED 2026-05-21; see Retrospective above]**

Simplest possible change: delete `if (transformer_kv.n_stream > 1) { ... return false; }` at `src/llama.cpp:616`. First decode for each stream still triggers a cudaGraphInstantiate (because topology-hash hits an unpopulated entry), but subsequent decodes for the same stream hit the cache and route through `cudaGraphExecUpdate`. Net: ~N graph instantiations at warm-up (one per stream), then steady-state reuse.

**Why this alone may not work**: the existing graph cache check compares `node->src[i]->data` against captured `src_address[i]`. For the **attention-read** view tensors in `llm_build_kqv`, the `data` pointer encodes the stream's offset and changes per stream. The check at `ggml-cuda.cu:4711-4717` returns false for src_address mismatch *unless* the op is `GGML_OP_CPY` or `GGML_OP_VIEW`. Whether the attention-read tensors register as `GGML_OP_VIEW` (legitimately tolerated) or as e.g. `GGML_OP_FLASH_ATTN_EXT_PER_SLOT_KV` (which would invalidate) needs source confirmation before locking the design. **T1.a is that probe.**

### Tier 2 — Patch attention-read views in `update_cache_copies` — **[PARKED 2026-05-21; see Retrospective above]**

Extension of the existing PHASE_NSTREAM_KV_4D N2.b patching, from CPY nodes to the attention-read view nodes in `llm_build_kqv`. Mechanism:

1. **Identify the attention-read view tensors per layer** in `llm_build_kqv` (`src/llama-build-context.cpp`). These are the `ggml_view_4d`/`ggml_view_3d` of K and V that bake `stream_id * nb[3]` at graph-build time. Store handles in the same registry as the existing `cache_copies[]`.
2. **Extend `update_cache_copies()`** (`src/llama.cpp:630`) to walk the attention-read view registry and recompute `view_offs` + `data` per stream, mirroring the CPY block already there (lines 657-665).
3. **Drop the `can_reuse_graph` bailout** at `src/llama.cpp:616`. Per-input reuse checks (`u_batch.all_seq_id`, `transformer_kv.head > 0`, `n_kv` bucketing) still gate; n_stream > 1 alone no longer disqualifies.
4. **Verify the downstream cache routing**: confirm that the attention-read view tensors register as `GGML_OP_VIEW` (allowed src_address change) — if not, extend the cache check to tolerate the read-view ops, scoped narrowly to nodes we've registered as per-stream-patched.

Existing infrastructure carries Tier 2:
- `cudaGraphExecUpdate` already in dispatch path — no novel CUDA-graph engineering.
- `update_cache_copies()` already runs per decode — extending its registry is mechanical.
- The graph cache cap (`GGML_CUDA_GRAPH_MAX=128`) is fine for n_stream ≤ 8.
- PSKV per-slot FA kernel is the production attention path — already NPC-preserved.

Expected outcome: at NP=8 steady-state TG, cudaGraphInstantiate fires N times at first decode per stream then never again; subsequent decodes route to `cudaGraphExecUpdate` with patched view offsets. The 2-3 ms cgraph-rebuild cost per per-stream sub-batch collapses to the cudaGraphExecUpdate cost (~10-50 µs per upstream NVIDIA benchmarks).

### Tier 3 — Unified-stream dispatch + ragged FA

The structurally-correct end state. Stop dispatching `llama_decode` per stream. Instead:

a. **Server-side batch fusion.** Modify `process_batch_tokens` (`examples/server/server-context.cpp:4610`) to collect all active slot tokens into one batch each tick — one llama_decode/tick. The 4D KV layout already supports this; each token's `seq_id` resolves to its stream's slab via `update_cache_copies`'s per-stream patching.

b. **GGML graph builder uniform-batch awareness.** `llm_build_kqv`, the KQ-mask builder, and the projection matmuls need to handle a ubatch where token rows belong to different streams. Upstream's pattern: KQ mask shape `[n_kv, n_tokens / n_stream, 1, n_stream]`; per-stream attention computed via the n_stream axis at ne[3]. Our `GGML_OP_FLASH_ATTN_EXT_PER_SLOT_KV` already takes `per_row_k_bound` and is shape-aware per row — **bridging this to a unified ubatch is much smaller scope than a fresh FA kernel port**.

c. **Drop the n_stream==1 guards on `build_k_shift` / `build_defrag` / `v_trans`** (currently `GGML_ASSERT(n_stream == 1)` per the PHASE_NSTREAM_KV closure follow-ups) so multi-slot ctx_shift and defrag work natively.

d. **Verify against vLLM's measured 154.77 t/s.** The empirical anchor exists; the gate is concrete.

### Tier 4 — Chunked-prefill admission

Defer to post-Tier-3 measurement. Splices new request prefill chunks into the same fused ubatch as running decodes (Sarathi-Serve). Mechanism well-understood; only worth it if prefill stalls measurably bound throughput.

### Tier 5 — Paged KV

Skip. For a single-server NP=8 workload with broadly similar sequence lengths the per-stream contiguous layout we have is functionally equivalent and avoids the indirection cost.

## Why we don't pivot to vLLM (despite the measured 4.75×)

The 4.75× is real and reproducible — we measured it on our hardware 2026-05-12 with vLLM at np=8 vanilla decode (154.77 t/s aggregate). The reason we don't pivot:

1. **Weight format incompatibility.** vLLM cannot consume Q4_0 + Hadamard + Q4_0 KV natively. Marlin-AWQ INT4 is comparable not better; QuaRot Hadamard PR [vllm-project/vllm#15162](https://github.com/vllm-project/vllm/pull/15162) was *closed*; the later compressed-tensors path has no Turing track record.
2. **GGUF path is slow.** [vllm-project/vllm#8669](https://github.com/vllm-project/vllm/issues/8669) — Llama 3.1 70B Q4 GGUF on A100 80GB at 8.7 tok/s. Worse than our current production on much weaker hardware.
3. **FP8 KV impossible on TU102.** [Turing whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf) lists FP16/INT8/INT4/INT1 only. vLLM's FP8 KV needs sm_86+ Triton.
4. **No working FA backend on Turing for modern models.** vLLM [#38918](https://github.com/vllm-project/vllm/issues/38918) — Gemma4 has zero working attention backends on sm_75. [#29743](https://github.com/vllm-project/vllm/issues/29743) Qwen3-VL closed as not-planned for Turing.
5. **We lose DFlash, MTP, PSKV ILP, MMQ I=8, Hadamard.** Months of NPC-preserving perf work is throwaway. Tier 3 keeps all of it.
6. **ik_llama.cpp's split-mode-graph already beats mainline 33 % TG / 6-9× PP** ([discussion #1247](https://github.com/ikawrakow/ik_llama.cpp/discussions/1247)). The kernels are SoTA for sm_75; only the dispatch is behind. Close the dispatch gap inside ik_llama.cpp.

## Binding gates (Tier 2 closure)

- **GP3.a** — `llama-batched-bench` TG NP=8 ≥ 27.73 t/s baseline (recover -6.2 %). Hard binding.
- **GP3.b** — `llama-batched-bench` TG NP=8 ≥ +15 % over baseline (≥ 31.9 t/s) if patching is correct. **Stretch binding** — if missed but GP3.a hit, Tier 2 closed-on-recovery and we evaluate why patching gain underperformed.
- **GP3.c** — `scripts/test-pp-serialization.sh` wall ≤ 11 s (TG-overlap window restored).
- **GP3.d** — `scripts/test-production-np-determinism.sh` byte-identity preserved, single + multi-GPU at NP={1,2,4,8}. Hard binding.
- **GP3.e** — DFlash test suite GREEN unchanged: `bin/test-dflash-np-multislot` (Phase 6 driver), `bin/test-dflash-spec-batched-fanout` (Phase 5 orchestrator), `bin/test-dflash-batch-vs-serial` (Phase 4 kernel batched-vs-serial), `bin/test-dflash-closure` (8/8 prompts argmax-equivalent), `bin/test-dflash-np-invariance` (T7 kernel-level 4 seeds × N∈{1,2,4,8}). Hard binding — DFlash multi-slot composition with the dispatch change must not regress.
- **GP3.f** — `scripts/r5-probe-c4.sh ITERS=20` = 0/20, single + multi-GPU. Bug C absence preserved. Hard binding.
- **GP3.g** — **Warm-up determinism**: run same prompt 3× sequentially at NP=8 (R1, R2, R3), verify byte-identity across all three runs. Specifically addresses the "CUDA graph cache warm-up" suspect from P2 memory. Hard binding.
- **GP3.h** — `scripts/validate-batch-composition-trace.py` against an NDJSON trace from `r5-probe-c4.sh` ITERS=20 — zero violations of `BatchComposition.tla` / `StreamIsolation.tla`. Spec layer preserved.
- **GP3.n** — **MTP NP=1 production smoke**: run `LLAMA_MTP_FUSED=1` decode at `--parallel 1 --mtp --draft 3` on the production Qwen 3.6 27B GGUF. Token output byte-identical pre/post-phase. Throughput within ±1 % of current production NP=1 MTP baseline (~33.5 t/s). Confirms Tier 2's `update_cache_copies` extension is a no-op at NP=1 (where `_s = 0` always). Hard binding — current production must not regress.

If GP3.b underperforms but the other gates pass: surface the diagnostic per CLAUDE.md §8 ("Negative results land cheap when honest, expensive when rationalised"). Was patching correct? Did the disable counter fire? Did some other invalidation kick in? Instrument before deciding Tier 3 scope.

**Workload coverage**: GP3.a–c gate vanilla TG/PP; GP3.d–h gate NPC + Bug C + spec + warm-up; GP3.e gates DFlash multi-slot; GP3.n gates MTP NP=1 production. All three workloads — vanilla, MTP, DFlash — bound.

## Binding gates (Tier 3 closure — provisional, locked at Tier 2 close)

- **GP3.i** — `llama-batched-bench` TG NP=8 aggregate ≥ 100 t/s (~3.6× over current 27.73 t/s baseline). Anchored against vLLM measured 154.77 t/s — a 65 % approach is the conservative target. **Stretch**: 130 t/s (≥ 85 % of vLLM's number).
- **GP3.j** — All Tier-2 gates remain GREEN.
- **GP3.k** — `bin/test-fattn-per-slot-kv-dispatch-np-invariance` continues PASS (kernel-level NPC; PSKV unified-stream path under ne[1]>1 batched ubatch — the specific shape T9's drift signature flagged).
- **GP3.l** — `bin/test-dflash-spec-batched-fanout` symmetric + asymmetric continue PASS under unified verify-side dispatch. Composition gate.
- **GP3.m** — MTP fused path at NP>1: `test-spec-mtp-fused` (or equivalent) PASS under unified ubatch. Confirms `can_reuse_graph` Phase 37 #5 MTP fused branch composes with Tier 2's bailout drop.

## Implementation cards (Tier 2 — provisional, finalised at design lock)

1. **T2.a** — **Probe**: identify all attention-read view tensors emitted by `llm_build_kqv` per layer for the production Qwen 35B-A3B graph. Read `src/llama-build-context.cpp` lines 2864-... and the `llm_build_kqv` definition. Output: list of view-node positions per layer + their `ggml_op` types. Verify that they register as `GGML_OP_VIEW` (in which case the downstream cache already tolerates them) or another op (in which case T2.d needs the cache extended).
2. **T2.b** — **Storage**: extend `cache_copies[]` or add a parallel `cache_attn_views[]` registry; populate at graph-build time in `llm_build_kqv` (mirror how `cache_copies` is populated in `llm_build_kv_store`).
3. **T2.c** — **Patching**: extend `update_cache_copies()` (`src/llama.cpp:630`) to walk the new registry per decode, recompute `view_offs = _p * step + _s * stream_stride` and assign `data`. Identical mechanism to the existing CPY block, applied to read-views.
4. **T2.d** — **Cache compatibility**: confirm the downstream `ggml-cuda.cu:4711-4717` check tolerates the read-view src_address change. If not, narrow extension: tolerate src_address change for ops in the registered read-view set.
5. **T2.e** — **Drop the bailout**: delete the `n_stream > 1` short-circuit at `src/llama.cpp:616`. Update the comment to note T2.b/T2.c.
6. **T2.f** — **Bench gate GP3.a + GP3.b**. If positive, full correctness battery (GP3.c–GP3.h) on the production 27B Qwen.
7. **T2.g** — **Submodule bump + parent commit + push** per CLAUDE.md §5/§6.

## Tier 3 composition with PHASE_NSTREAM_KV closure + DFlash multi-slot

Before locking Tier 3 implementation, four composition questions need clear answers:

**Q1. Does Tier 3 unification reopen Bug C?** PHASE_NSTREAM_KV.md (closure doc) is explicit that per-stream dispatch is the structural Bug C closure — "each call sees a single-stream batch — `mul_mat` sees a uniform shape per call." Tier 3 unifies dispatch back to one `llama_decode` per tick.

*Answer*: No, because Bug C is "**mixed shape WITHIN a tick**", not "batched shape per tick". The original failing geometry was: tick T calls `mul_mat` with `[d, 1]` (slot 0 decode) then `mul_mat` with `[d, 210]` (slot 1 prefill) — two separate calls with different shapes, GEMV-vs-GEMM picked differently → ULP drift. Under Tier 3 unification, tick T calls `mul_mat` once with `[d, n_tokens_total]` — a single uniform shape per tick. Within-tick GEMM/GEMV pick is consistent. This is **more uniform than current per-stream dispatch** (which calls `mul_mat` N times at N different per-stream shapes). The KQ mask routes per-stream attention via ne[3] = n_stream axis; that's where stream isolation happens, not at the `mul_mat` level.

Verification: GP3.f (`r5-probe-c4.sh ITERS=20` = 0/20) is the binding gate for "Bug C absence". If it passes after Tier 3, the structural claim holds.

**Q2. Does Tier 3 reintroduce T9's NP=4/8 drift signature?** [[project_dflash_t9_np_validity_drift_signature]] (2026-05-14) identified a specific kernel boundary at NP=2→NP=4: "Likely candidates: FA mma_f16 tile transition (Turing m16n8k8, 8-wide tile; batch=4 first hits the next tile multiple)." This drift was empirically resolved by PHASE_NSTREAM_KV closure (per-stream FA inputs at ne[1]=1). Tier 3 puts `ne[1] > 1` back into FA via the unified ubatch.

*Answer*: PSKV per-slot kernel handles this — the production path routes `GGML_OP_FLASH_ATTN_EXT_PER_SLOT_KV` to `ggml_cuda_flash_attn_ext_wmma_f16_case_pb1<256,256,8,half>` (parallel_blocks=1 pinned, fp32 accumulation throughout, no cross-warp reductions, per-row mask filtering — see [[project_fattn_per_slot_kv_p2_landed_kernel_only]]). The kernel was verified NPC at NP={1,2,4,8} pre-PHASE_NSTREAM_KV when ne[1] was sometimes >1; T9's drift was at the FA mma_f16 *original* path which is no longer the production route. **But this needs gate verification**: GP3.k (`test-fattn-per-slot-kv-dispatch-np-invariance` continues PASS) is the binding test.

If Tier 3 produces T9-class drift, the path is to confirm whether the PSKV pb1 kernel is being dispatched (and not the legacy mma_f16 path) for the unified-ubatch shapes — instrument and verify before declaring impossibility.

**Q3. How does Tier 3 compose with DFlash multi-slot?** DFlash Phase 5 added `common_speculative_draft_batched` — the **draft side** is already unified across slots in one batched `llama_dflash_draft_batch` call. The **verify side** (target forward on candidate tokens) currently uses the server's per-stream `process_batch_tokens` dispatch. Tier 3 unifies verify-side dispatch too — that's compositionally cleaner than current (both sides batched-across-slots).

The DFlash multi-slot test gates (`test-dflash-np-multislot`, `test-dflash-spec-batched-fanout`, `test-dflash-batch-vs-serial`) will exercise verify-side under Tier 3 unification. GP3.e gates all of them GREEN.

Risk: DFlash's kernel-level `n_slots_cap` distinction ([[feedback_drafter_forward_n_slots_cap]]) means its kernels expect bind-time capacity not dispatch-time fan-out. Tier 3's unified ubatch passes through DFlash unchanged because the DFlash kernel is invoked from `llama_dflash_draft_batch`, not from the generic decode path. Verify-side ubatch unification is downstream of DFlash kernel invocation. **They compose**.

**Q4. How does Tier 2 compose with MTP fused graph reuse?** `can_reuse_graph` Phase 37 #5 allows n_tokens>1 reuse when `mtp_op_type == MTP_OP_DRAFT_GEN_FUSED` and step counts match. Tier 2 drops the n_stream>1 short-circuit. Both checks compose: at n_stream>1 with MTP fused, the path now reuses through the MTP fused branch AND patches per-stream view offsets via update_cache_copies. Verify: MTP fused at NP>1 continues to work — covered by the existing PHASE_NSTREAM_KV closure gates.

## Implementation cards (Tier 3 — refined 2026-05-21 post Tier 2 closure)

Sequenced as **two coherent bundles** per `feedback_oneshot_then_evaluate`:

- **Bundle A (Steps 0–4) — infrastructure plumbing.** Lands without flipping dispatch. Bailout stays active so production (n_stream>1 single-seq per call) is unchanged at every intermediate step. Each step adds a property-test gate and keeps `verify-production-determinism.sh` GREEN at production shape. End of bundle: the 4D unified-batch path exists and is exercised by synthetic multi-seq tests, but server-context still dispatches per-stream.
- **Bundle B (Steps 5–8) — the dispatch flip.** Server-context fusion + bailout drop + n_stream==1 guard lifts + DFlash composition + perf gate. One coherent commit (with sub-commits for traceability) that flips the world.

### T3.0 — Remove Tier 2 scaffolding (cleanup)

**Touches:**
- `src/llama-context.h` — drop `CacheReadView` struct + `cache_read_views` vector.
- `src/llama.cpp` — drop the per-stream read-view patch loop in `update_cache_copies()` (lines 782-795); drop the `[PARKED]` comment block at lines 610-629; keep the bailout (will be removed in T3.6).
- `src/llama-build-context.cpp` — drop `ggml_set_read_view_indirect_slot(k, …)` / `ggml_set_read_view_indirect_slot(v, …)` calls in both `llm_build_kqv` and `build_std_attention` paths; drop the `ggml_set_fa_indirect_slots(cur, …)` calls; drop the `cache_read_views[idx].view = k/v` assignments.
- `ggml/include/ggml.h` + `ggml/src/ggml.c` — drop `ggml_set_fa_indirect_slots` / `ggml_get_fa_K_indirect_slot` / `ggml_get_fa_V_indirect_slot` and the FA op_params[14]/[15] convention.
- `ggml/src/ggml-cuda.cu` — drop the per-tick read-view-src-ptrs refresh hook (lines 4668-4720) and `graph->use_read_view_indirection` / `read_view_src_ptrs` / `read_view_src_ptrs_d` fields.
- `ggml/src/ggml-cuda/graph.cuh` — drop matching fields.
- `ggml/src/ggml-cuda/fattn-per-slot-kv-singlewarp-sm75.cu` — drop `src_ptr_table` / `K_slot_idx` / `V_slot_idx` kernel parameters + the indirection lookup path; keep `nb33` (T3.1 foundation).
- `tests/spec/test-per-stream-read-view-patching.cpp` — re-purpose to assert the registry is GONE, OR delete entirely. The contract no longer exists.
- `specs/kv-cache/per_stream_read_view_patching.allium` — mark superseded by `specs/dispatch/unified_stream_dispatch.allium` (or delete).

**Keep:**
- 4D K/V tensor layout `[head_dim, kvps, n_head_kv, n_stream]` (PHASE_NSTREAM_KV closure).
- `cache_copies` WRITE-side per-stream patching (still load-bearing — CPY writes happen every tick regardless of dispatch model).
- `v_heads` per-stream cursor; `kv_size_per_stream`, `n_stream` foundation fields.

**Gate:** build clean. `verify-production-determinism.sh` NP={1,2,4,8} GREEN. DFlash test suite GREEN.

### T3.1 — PSKV `nb33` mask addressing — *landed*

Submodule `c2a142a4`. Kernel takes `nb33` and addresses mask as `mask + nb33 × seq + nb31 × tok`. At `ne[3]=1` collapses to legacy via `nb33 × 0 = 0`. Foundation for the unified-batch path.

### T3.2 — `find_slot` multi-seq allocation — *landed*

**Touched:** `src/llama.cpp` `llama_kv_cache_find_slot`.

**Semantics implemented:** when `n_stream > 1` AND the batch has tokens with multiple distinct primary seq_ids, scan-for-empty within each seq's stream slice and commit per-stream cells in a two-phase pass (scan-all-then-write). At `n_stream == 1` OR single-seq batches the path collapses byte-identically to the legacy single-arena allocator. Token convention is contiguous-per-seq (locked at OpenQ-A resolution).

**Test gate landed:** `tests/spec/test-n-stream-kv-layout.cpp::test_multi_seq_find_slot` — synthetic batch with 2 seqs × 4 tokens each (seq 0 at pos [10..13], seq 1 at pos [20..23]) → asserts each stream's allocated cells carry the expected pos and seq_id, isolates across streams, and that v_heads + cache.head advance correctly. PASS.

`llama_kv_cache_find_slot` declaration exposed in `src/llama-context.h` for the property test. The function is internal-but-not-static so existing internal callers (`llama_decode_internal`, state load) remain unchanged.

`tests/CMakeLists.txt` sets `SKIP_RETURN_CODE 77` on the test so ctest reports skip cleanly when no model path is supplied.

**Production gate:** `verify-production-determinism.sh` GREEN (single-seq path unchanged from HEAD).

### T3.3 — Build context 4D K/V/mask/Q

**Touches:**
- `src/llama-build-context.cpp` `llm_build_kqv` + `build_std_attention` — when batch is multi-seq AND `n_stream > 1`, build K/V as 3D views (already done — `ggml_view_3d` with stream offset). Q reshape to 4D `[head_dim, n_head, n_tok_per_seq, n_stream]`. FA op routes to PSKV (T3.1's nb33 handles ne[3]>1 mask addressing).
- `src/llama-build-context.cpp` `build_inp_KQ_mask` (line 383) — emit `[n_kv, n_tok_per_seq, 1, n_stream]` 4D mask shape when multi-seq. Today the mask is 2D `[n_kv, n_tokens]` — needs widening with stream as the outermost dim.
- `src/llama.cpp` `llama_set_inputs` mask-fill loop (line 4592-4791) — fill the 4D mask per-stream using `_kv_mask_base = stream_id × kvps`. Already partly done at line 4602 for the n_stream>1 case; refactor to iterate streams as the outermost loop.

**Test gate:** synthetic 2-stream decode (one tick, 2 tokens, distinct seq_ids) produces byte-identical output to two serial per-stream decodes at the same starting state. `test-unified-stream-dispatch.cpp` flipped from RED to GREEN at this step.

**Production gate:** NP=1 (collapses to ne[3]=1) still byte-identical. `verify-production-determinism.sh` GREEN.

### T3.4 — `llama_decode` multi-seq path enabled — *gate dropped*

**Touched:** `src/llama.cpp` `nstream_demands_subbatch` block. The short-circuit is now gated on `!cparams.flash_attn` — under FA-on, contiguous-per-seq multi-seq batches flow through directly to the T3.2 multi-seq find_slot + T3.3 4D build + PSKV ne[3]>1 dispatch.

Bailout in `can_reuse_graph` stays active (T3.6 territory). Multi-seq decodes build a fresh graph per call.

**Production gate (PASS post-T3.4):** `verify-production-determinism.sh` cross-NP determinism PASS at NP={1,2,4,8} multi-GPU; all slots byte-identical; cross-NP slot-0 matrix BYTE-IDENTICAL. Batch-shape invariance PASS 4/4. Server still dispatches per-stream so production load remains on the single-seq path.

**Binding test — folded into T3.5 verify gate (2026-05-21).** The standalone multi-seq decode test (`tests/spec/test-multi-seq-decode-byte-identity.cpp`, held untracked on disk) hits `delta_net::delta_net` `GGML_ASSERT((uint32_t) s < qnext_state_slots)` on every Qwen GGUF available — both Qwen 3.5 0.8B and the production Qwen 3.6 27B route through `build_qwen35` → `delta_net` (Qwen3-Next hybrid arch) and the bare test harness doesn't initialise `qnext_state_slots` the way the production server does. The new 4D build path under `is_multi_seq && kv.n_stream > 1` is also unreachable in the production code today — the server still splits per-seq before dispatch. T3.5 is what makes the server construct multi-seq batches; once it lands, `verify-production-determinism.sh` becomes the load-bearing binding test for the entire chain T3.3 + T3.4 + T3.5. T3.4 stays open against that joint gate.

### T3.3-followup — K/V WRITE multi-seq via `ggml_set_rows` + per-device `n_head` fix — *reopen* 2026-05-21

T3.3 was prematurely marked complete. T3.5 attempt surfaced two structural gaps:

1. **`build_std_attention` multi-device split path** (`src/llama-build-context.cpp:2946`) uses the build-context member `n_head` (full from `hparams`) for the multi-seq Q reshape. Under graph-split with `n_device=2`, per-device Qcur is `[head_dim, n_head/2, n_tokens]`; the reshape requires per-device `n_head_q_local = split_wq->ne[1] / n_embd_head_k`, not the full `n_head`. Fix is local (one line / a few lines).
2. **K/V WRITE path** still writes via a single 3D view at `kv_head_eff = kv.head + kv_head_offset`. Under multi-seq dispatch each seq writes to its own stream slice (different per-stream `head_local` values from T3.2's allocator). The current `ggml_cpy(Kcur, k_cache_view)` lays all tokens consecutively starting from one base offset — wrong for multi-seq.

**Solution (per upstream PR #14363 design).** Integrate `ggml_set_rows` into the K/V WRITE path:
- `ggml_set_rows` is already implemented in ik_llama.cpp (`ggml/src/ggml-cuda/set-rows.cu` covers F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/IQ4_NL — exact type parity with upstream).
- Upstream `llama_kv_cache::cpy_k`/`cpy_v` reshape K cache to 2D `[n_embd_gqa, kv_size * n_stream]` for n_stream > 1 and emit `ggml_set_rows(k, k_cur, k_idxs)`. `k_idxs` carries per-token GLOBAL row indices: `idx[t] = stream(t) * kv_size_per_stream + head_local(t)`.
- One scatter per layer per K/V replaces the per-stream `ggml_cpy`; multi-seq writes naturally land in their correct stream slices.

**Touches:**
- `src/llama-build-context.cpp` — `llm_build_kv_store` and `build_std_attention` K/V cpy sites replaced by `ggml_set_rows`; per-device `n_head` derived from `Qcur->ne[1]` or `split_wq->ne[1] / n_embd_head_k`.
- `src/llama-context.h` / `src/llama-build-context.cpp` — new `inp_k_idxs` (and optionally `inp_v_idxs`) input tensor (shape `[n_tokens]`, `GGML_TYPE_I32` or `I64`), wired alongside `inp_KQ_mask`.
- `src/llama.cpp` — `llama_set_inputs` populates `inp_k_idxs` from `batch.seq_id` + `cache.v_heads`; `update_cache_copies`'s per-stream view_offs patching becomes unused for the new path (kept on legacy path for n_stream==1 and the read-side views).

**Gates (binding):**
- Build clean; `bin/test-backend-ops` GREEN for SET_ROWS at F16, Q4_0 (already passing on HEAD if upstream's tests carried over; verify).
- `verify-production-determinism.sh` GREEN at NP={1,2,4,8} multi-GPU **post-refactor without the T3.5 unification** (single-seq dispatch using the new write path must be byte-identical to T3.4).
- After T3.5 re-applied: `verify-production-determinism.sh` GREEN at NP={1,2,4,8} multi-GPU **with** unified multi-seq dispatch.

Token estimate: 40–80 k (write path refactor + index plumbing + dual-mode verification).

### T3.5 — Server-context unified dispatch (bundle B starts) — *landed 2026-05-22*

**Landed:** parent `51c86bb`, submodule `be5d756e`. `process_batch_tokens` now uses **split_equal** grouping: detects contiguous shape-uniform multi-seq groups (seq_ids contiguous from 0, distinct, uniform per-seq count) and dispatches each as ONE `llama_decode`. Single-seq groups (prefill, single-slot decode) fall back to legacy n_batch chunking. `speculative_decoding_accept` gained `n_tokens_in_view` parameter to scope to the dispatched view under `run_seq_id=-1` (unified mode). MTP warmup detection scans all batch_view tokens. `build_std_attention` Q reshape uses empirical `ggml_nelements(Qcur) / (n_embd_head_k * n_tokens)` for per-device `n_head_q_local`. K/V views span `n_seq_in_batch` (not full `kv.n_stream`). `per_row_k_bound` sized to `n_tokens`. PSKV FA constructor relaxed to accept `mask->ne[3] == q->ne[3]` and `per_row_k_bound->ne[0] == q->ne[1] * q->ne[3]`.

**Permanent metric:** `dispatch_multi_seq_count` / `dispatch_total_count` atomic counters in `process_batch_tokens` emit `LLAMA_LOG_INFO` every 64 dispatches. Verify run at NP=8 shows 55/64 dispatches use the multi-seq build path.

**Keeps:** `PrefillSerialisationGate` in `batch_pending_prompt`. `DecodeHoldGate` already removed in PHASE_NSTREAM_KV_4D N3.

**Production gate (PASS):** `verify-production-determinism.sh` ACCEPTANCE PASS at `DEVICE=CUDA0,CUDA1, NP_LIST="1 2 4 8", CTX_CHECKPOINTS=3` — cross-NP determinism PASS AND batch-shape invariance PASS.

### T3.6 — Bailout drop + multi-stream K-shift / defrag — *CLOSED 2026-05-22*

**[x]** All sub-cards landed: A (audit), S (specs), T (synthetic tests), I.b (SET_ROWS pass-through + bailout-drop design decision), I.c1.x2 (multi-stream K-shift LAYER + GRAPH), I.c2 (multi-stream defrag LAYER + GRAPH + generic CUDA Q→Q cpy), M (graph-pool VRAM probe). Closure-gates section below fully ✅. See `~/.claude/projects/-home-llm-yarn-agentic/memory/project_t3_6_full_grid_closure.md` for the per-piece commit-hash trail. Next blocker on this PHASE doc is **T3.8** (perf gate GP3.i).


**Path:** T3.6-full per the audit grade. Audit (T3.6.A) complete; revealed three HIGH-severity risks that necessitate proper specs + synthetic tests before any code change.

#### T3.6.A — Audit findings (read-only, 2026-05-22)

- **F1** — `inp_K_shift` populator (`src/llama.cpp:4356–4366`) is per-cell flat: `data[i] = cells[i].delta`. Cell `i` is in stream `s = i / kvps` at local pos `p = i % kvps`. Flat layout already matches the per-(stream, pos) view.
- **F2 (HIGH)** — `ggml_rope_ext` requires `b` 1D with size `a->ne[2]`. A single 4D rope on K view `[head_dim, n_head_kv, kvps, n_stream]` would broadcast the SAME deltas across all streams. WRONG when seqs have different per-cell deltas. **Resolution:** Option (i) — per-stream loop in `build_k_shift`. n_stream rope calls per layer; k_shift is rare so overhead is bounded.
- **F3 (HIGH)** — `llama_kv_cache_defrag_internal` hole-fill (`src/llama.cpp:6661+`) walks `cells[]` flat and pulls cells "from the end" without stream awareness. Under n_stream>1 this would move a cell from stream `s` into another stream's slice. **Per-stream outer loop is mandatory, not optional.**
- **F4 (HIGH)** — `build_defrag`'s 2D `[n_embd_k_gqa, nm]` view (`src/llama-build-context.cpp:313–321`) assumes flat `[n_embd_k_gqa, n_kv]` layout. Under 4D `[head_dim, kvps, n_head_kv, n_stream]`, n_head_kv is NOT contiguous with head_dim. **3D-view-per-layer rewrite required** (head_dim contiguous; per-head stride via parent `nb[2]`; per-position stride via parent `nb[1]`; per-stream offset via `s * nb[3]`).
- **F5** — `cache_copies` sizing (`src/llama.cpp:755–762`) is `2*splits*n_layer` or `2*n_layer`. MTP-safe: `update_cache_copies` adjusts iteration count via `model.mtp ? n_layer : n_layer - nextn_predict_layers`. Set_rows pass-through landing in `cache_copies[2*il+0/1]` does not interfere with MTP.
- **F6** — Graph reuse with set_rows is safe: `inp_kv_idxs` is `ggml_set_input`-marked; data is uploaded fresh every `llama_set_inputs`. Topology hash includes input tensor shapes; n_tokens mismatch → fresh build (no staleness).
- **F7 (VRAM)** — `GGML_CUDA_GRAPH_MAX = 128` FIFO. Worst-case +16 multi-seq entries × ~5MB ≈ +80MB. Irrelevant against 24GB.
- **F8 (MEDIUM)** — Prior "~0% from reuse" A/B was single-seq dispatch. Multi-seq cgraph has different op count; CPU host build cost may differ. **Microprobe required** (T3.6.M) before declaring perf-neutral.
- **F9** — Dropping the n_stream>1 bailout in `can_reuse_graph` doesn't create false positives. Remaining miss reasons (embd, graph_reuse off, n_kv bucket, n_outputs, mtp_op_type, update_cache_copies) still gate correctly.
- **F10 (HIGH)** — Neither `r5-probe-c4` (`--no-context-shift`) nor `verify-production-determinism.sh` exercises k_shift or defrag. **Synthetic tests (T3.6.T) are the only binding evidence for the c1/c2 paths.**

#### T3.6.S — Allium + TLA+ specifications

1. **`specs/kv-cache/k_shift_per_stream.allium` + `specs/kv-cache/KShiftPerStream.tla`**
   - `KShiftAppliesPerCell` — `inp_K_shift[s*kvps + p]` is the rotation delta for stream `s`, local pos `p`.
   - `KShiftIsolation` — applying a shift to stream `s` cells does not modify stream `s' != s`'s K bytes.
   - `KShiftRoPEEquivalent` — post-shift K equals pre-shift K with RoPE applied per the new positions.
   - TLC `MC.cfg` with `n_stream ∈ {2, 4}`, `kvps ∈ {4, 8}`.

2. **`specs/kv-cache/defrag_per_stream.allium` + `specs/kv-cache/DefragPerStream.tla`**
   - `DefragNoCrossStream` — no cell moves cross stream boundaries.
   - `DefragCompactsPerStream` — post-defrag, each stream has cells `[0, used_per_stream)` non-empty contiguously.
   - `DefragPreservesKVBytes` — K and V byte data follow cell metadata moves byte-for-byte.
   - TLC `MC.cfg` with `n_stream ∈ {2, 4}`, `kvps ∈ {4, 8}`.

3. **`specs/dispatch/graph_reuse_set_rows.allium` + `specs/dispatch/GraphReuseSetRows.tla`**
   - `SetRowsInputRefresh` — `inp_kv_idxs` data is refreshed by `llama_set_inputs` between any two graph executions.
   - `ReuseShapeStability` — cached graphs keyed by `(n_tokens, n_seq_in_batch, n_tok_per_seq, n_kv_bucket)`; shape mismatch → cache miss.
   - `BugCAbsenceUnderReuse` — reusing a multi-seq graph with refreshed `inp_kv_idxs` produces byte-identical output to a fresh build at the same shape.
   - TLC `MC.cfg` with bounded reuse cycles.

**Gate:** `allium check` clean on all 3 specs; `tlc -parse` clean on all 3 modules; TLC runs on each `MC.cfg` complete without invariant violations.

#### T3.6.T — Synthetic tests (RED first, GREEN after impl)

1. **`tests/spec/test-kv-shift-per-stream.cpp`**
   - n_stream=4 cache; populate streams with known F32 patterns recoverable on dequant.
   - `llama_kv_cache_seq_add(seq_id=2, p0=10, p1=30, delta=+5)`; `llama_kv_cache_update`.
   - Assertions: streams 0/1/3 byte-identical pre/post; stream 2 cells [10, 30) RoPE-rotated to new positions.
   - **RED on T3.5 HEAD** (assert n_stream==1 fires).

2. **`tests/spec/test-kv-defrag-per-stream.cpp`**
   - n_stream=2 cache; populate both streams.
   - `llama_kv_cache_seq_rm(seq_id=0, p0=5, p1=10)` to create stream-0 hole.
   - `llama_kv_cache_defrag` + `llama_kv_cache_update`.
   - Assertions: stream 0 compacted, stream 1 byte-identical pre/post.
   - **RED on T3.5 HEAD**.

3. **`tests/spec/test-graph-reuse-set-rows.cpp`**
   - n_stream=4 cache; run identical multi-seq decode twice.
   - Inspect `g_can_reuse_last_miss_reason`: first call MISS (no prev), second call HIT (reason=0 post-(b)).
   - Outputs byte-identical.
   - **RED on T3.5 HEAD** (reason=6 from n_stream>1 bailout); GREEN after T3.6.I.b.

#### T3.6.I — Implementation (in order, each commit GREEN-gated)

- **[~] T3.6.I.b — SET_ROWS pass-through + bailout drop** *(partial, 2026-05-22)*
  - **[x] T3.6.I.b.1 — SET_ROWS pass-through in `update_cache_copies`** (4 branches in `src/llama.cpp:629–749`) landed at submodule commit `4210e5b8`. All 4 cpy-branches now treat `GGML_OP_SET_ROWS` as pass-through. Defensive (dead code under current bailout; ready for the I.b.2 drop). Verify-production-determinism PASS post-change at NP={1,2,4,8}.
  - **[ ] T3.6.I.b.2 — Bailout drop with `prev->kqv_stream_id` gating.** The audit's F9 finding was incomplete: `build_std_attention`'s single-seq branch (`src/llama-build-context.cpp:1821-1826`) bakes `kqv_stream_id * nb[3]` into the K view's offset. Single-seq decodes on a multi-stream context are NOT reuse-safe across streams. Verify-production-determinism FAILED at slots != 0 under NP={4,8} when the bailout was dropped naively; reverted. The proper drop requires:
    - Track `prev->kqv_stream_id` (or equivalent) on the cached graph.
    - Gate reuse on `prev->kqv_stream_id == current kqv_stream_id` for single-seq decodes.
    - For multi-seq decodes (4D K view at offset 0, ne[3]=n_seq_in_batch), reuse is offset-safe; the existing n_tokens > 1 MTP gate fires first anyway, so the n_stream bailout drop yields no real uplift there until the MTP gate is also revisited.
  - Test 3 (`test-graph-reuse-set-rows`) stays RED on this submodule (binding on the I.b.2 work).
- **[x] T3.6.I.c1 — Multi-stream `build_k_shift` + IMROPE gate lift** *(landed 2026-05-22, submodule `583c279d`)*
  - Per-stream rope loop in `build_k_shift` (`src/llama-build-context.cpp:170-249`); under n_stream==1 reduces to the legacy single-rope shape.
  - **Bonus from upstream cross-check:** lifted IMROPE from `get_can_shift`'s no_shift list (`src/llama.cpp:6879-6893`). The audit had treated K-shift as "structurally unreachable on Qwen 3.6 dense"; investigation of upstream llama.cpp showed the IMROPE gate at `llama.cpp/src/llama-kv-cache.cpp:1306-1310` (commit `b768f0843f`, authored by the user 2026-04-14) explicitly allows K-shift on IMROPE — only MROPE (true 2D positions) is gated off. ik_llama.cpp's IMROPE inclusion (commit `433531dd`, firecoperana, 2026-03-13) was a port-shortcut from the Qwen3-VL port. With the gate aligned to upstream and the existing IMROPE→NEOX workaround in `rope_type_shift`, K-shift now works on Qwen 3.6 dense.
  - **T3.6.T1 binding** GREEN on Qwen 3.6 27B production target: pre-shift seq2 pos_max=3, post-shift seq2 pos_max=8 (+5 delta), streams 0/1/3 unchanged. KShiftIsolation + KShiftAppliesPerCell both bind.
  - **verify-production-determinism.sh** ACCEPTANCE PASS post-change at NP={1,2,4,8} (server uses `--no-context-shift`; verify doesn't exercise K-shift but no regression).
  - **Known limitation (now defensively gated):** `CUDA_Split` (graph-split) + K-shift hits an illegal-memory-access in `ggml_backend_sched_copy_inputs` distributing the `inp_K_shift` view across devices. Root cause is the view-data-pointer-set-at-construction interaction with the scheduler's cross-device input cpy. The split-aware K-shift code path itself produces a correct graph; the failure is in input distribution.
  - **T3.6.I.c1 follow-up landed** *(submodule commit after `583c279d`)*:
    - `build_k_shift` now branches on `k_l[il]->extra` and operates on per-device `ggml_split_tensor_t::splits[id]` under graph-split (`src/llama-build-context.cpp:170-275`).
    - `get_can_shift` now returns false under `split_mode == GRAPH && n_stream > 1` (`src/llama.cpp:get_can_shift`) — clean `rc=1` from `llama_kv_cache_update` instead of a CUDA crash.
    - `test-kv-shift-per-stream` extended to run **both** split modes. LAYER asserts the full K-shift binding (isolation + per-cell rotation); GRAPH asserts the documented graceful `rc=1`. A future regression that lifts the gate without first fixing the input-population layer would fail this branch.
  - **Verified on Qwen 3.6 27B production target:** both branches PASS. Verify-production-determinism ACCEPTANCE PASS post-change.
  - **Input-layer restructure landed** *(T3.6.I.c1.x, submodule `b62765be`)*: replaced single `inp_K_shift` + per-stream view-slicing with `n_stream` separate input tensors (`std::vector<ggml_tensor *> inp_K_shift_per_stream` on the decoder struct). Each per-stream input is independently scheduler-allocated; no view aliasing. Verified GREEN on Qwen 3.6 27B and verify-production-determinism PASS.
  - **Audit:** view-aliased class of inputs scanned (`inp_K_shift`, `inp_KQ_mask`, `inp_kv_idxs`, `inp_pos`, `inp_pos_bucket`, …). Only `inp_K_shift` had the per-stream-view-slice pattern. `inp_KQ_mask` already uses a native 4D `[n_kv, n_tok, 1, n_seq_in_batch]` layout; `inp_kv_idxs` consumed by `ggml_set_rows`; others not per-stream-sliced. Audit-grade completeness for the input-tensor class confirmed.
  - **GRAPH-split gap fully closed** *(T3.6.I.c1.x2, submodule commit on top of `b62765be`)*. Root cause identified as twofold:
    1. Single shared `inp_K_shift` forced scheduler into cross-device input distribution where leaf-input allocation under multi-device reset produced `[NULL]` backend assignments → `cudaMemcpyPeerAsync` "invalid argument".
    2. The intermediate F32 `tmp` tensor (cast of quantized K view) was pinned to the FIRST CUDA backend supporting CUDA_Split (i.e. CUDA0), regardless of which device the K split lived on. Cast/rope/cpy then crossed device boundaries on the K-cache side too.
  - **Fix:** per-(device, stream) input tensors, each pinned to its consuming backend via `ggml_backend_sched_set_tensor_backend`. `build_one_rope` accepts a `backend_override` parameter so the intermediate tmp lives on the same device as its K split. `llama_set_k_shift` uses `ggml_backend_tensor_set` (host-staging-aware) and writes the replicated per-stream slice to each device's copy. `get_can_shift` GRAPH-split gate REMOVED.
  - **Verification on Qwen 3.6 27B production target:** both LAYER and GRAPH split pass the full K-shift binding (isolation + per-cell rotation, seq2 pos_max 3→8, others unchanged). `verify-production-determinism.sh` ACCEPTANCE PASS at NP={1,2,4,8}. Defensive test asserts full binding under both modes — any regression that reintroduces cross-device cpy fails the GRAPH branch.
- **[x] T3.6.I.c2 — Multi-stream defrag (LAYER + GRAPH)** *(landed 2026-05-22)*
  - **`llama_kv_cache_defrag_internal`** rewritten with per-stream outer loop (resolves F3): hole-fill scans scoped to each stream's `[s*kvps, (s+1)*kvps)` slice; cells never cross stream boundaries. Per-stream `v_heads[s]` updated post-defrag; `cache.head` reset to 0 (compacted-from-front in every stream). Flat `ids[]` interface preserved for backward compat.
  - **`build_defrag`** rewritten with 3D-per-stream views (resolves F4) `[head_dim, n_head_kv, nm]` using parent's `nb[2]` as head stride and `nb[1]` as position stride, offset = `s*nb[3] + p_local*nb[1]`. Under graph-split (`kl_extra`/`vl_extra` set), iterates per-device on the split tensors so each device sees only its own K/V split (no cross-device cpy needed).
  - **CUDA cpy enhancement** *(ggml/src/ggml-cuda/cpy.cu + ggml-cuda.cu)*: added generic `cpy_q_q_same_type` kernel (parameterized at runtime by `qk` and `block_bytes`) for Q→Q same-type non-contiguous block copies. Covers Q4_0 (test), Q4_0_AR16 (production with `--k/v-cache-hadamard`), Q8_0, Q4_1, Q5_0, Q5_1, Q6_0, IQ4_NL. Wired into `ggml_cuda_cpy` + `ggml_cuda_cpy_fn` (gated on same shape + same quantized type + neither operand transposed; the pre-existing Q8_0→Q8_0 transpose path for MLA mla=2 is preserved by adding a transposed-operand discriminator). `ggml_backend_cuda_supports_op` updated to advertise the new combination. Unblocks the class of strided same-type Qn cpy that was previously falling back to CPU (which segfaulted reading CUDA-resident data).
  - **`test-kv-defrag-per-stream`** extended to defensive both-mode harness (LAYER + GRAPH). Both modes assert the full `DefragNoCrossStream` + `DefragCompactsPerStream` binding. `argv[2] = layer|graph|both`.
  - **Verification on Qwen 3.6 27B production target:** both LAYER and GRAPH pass. `verify-production-determinism.sh` ACCEPTANCE PASS at NP={1,2,4,8}. DFlash composition (np-multislot byte-identical, np-invariance 4/4 seeds, closure 8/8 argmax-equivalent) GREEN.

`v_trans` non-FA asserts (`src/llama-build-context.cpp:790`, `:3070`) **NOT lifted** — that path is genuinely incompatible with the 4D layout and gated on `--fa off` which production does not use. Asserts correctly document the constraint.

#### T3.6.M — VRAM + reuse perf measurements — *landed 2026-05-22*

- **[x] VRAM probe (permanent)** — extended `~ggml_backend_cuda_context` log at `ggml/src/ggml-cuda.cu:667` to sum host-side bookkeeping (nodes / params / ggml_graph_properties / cpy_dest_ptrs vectors) plus device-side cpy `dest_ptrs` arrays across all cached graphs. cudaGraphExec_t binaries are opaque so are not counted — this is an under-count of true graph-pool VRAM, but it grows proportional to pool size which is the load-bearing signal. Sample (test-graph-reuse-set-rows, 0.8B, GRAPH split): `have 9 graphs (234 nodes, 60.4 KB host bookkeeping, 0.0 KB device dest_ptrs)`. test-kv-shift LAYER split shows the larger-per-graph shape: `have 3 graphs (1764 nodes, 455.5 KB host bookkeeping, 0.6 KB device dest_ptrs)`.
- **[~] Reuse perf delta — MOOT and parked**. The original M card scoped this as "bailout-dropped vs bailout-in-place A/B". T3.6.I.b.2 (bailout drop) closed as a design decision (see I.b above + `src/llama.cpp:610-628` inline rationale): dropping the n_stream>1 bailout would expose a single-seq cross-stream bug in `build_std_attention` for no real reuse uplift, because the multi-seq dispatch path already trips the n_tokens>1 MTP gate (reason=2) before reaching the n_stream check. There is no "bailout-dropped" world to measure. **If future data motivates revisiting the bailout drop** (e.g. a single-seq cross-stream fix lands separately, or a workload emerges where the n_tokens>1 gate is no longer dominant), spin up a fresh perf probe at that point.

#### T3.6 closure gates

- ✅ `bash scripts/verify-production-determinism.sh` ACCEPTANCE PASS post-(c.2) — captured 2026-05-22 across NP={1,2,4,8} multi-GPU.
- ✅ `r5-probe-c4 ITERS=20` = 0/20 — captured post-T3.5, 2026-05-22.
- ✅ `bin/test-kv-shift-per-stream both` GREEN.
- ✅ `bin/test-kv-defrag-per-stream both` GREEN.
- ✅ `bin/test-graph-reuse-set-rows` GREEN (miss_reason != 6 bailout path closed; reason=1/2 on 1st/2nd call is the cross-tick MTP-gate path which is correct behaviour).
- ✅ All 3 Allium specs `allium check` clean (0 errors; external-entity / unused-field warnings expected).
- ✅ All 3 TLA+ modules `tlc -config MC.cfg` clean — KShiftPerStreamMC 3041 states, DefragPerStreamMC 304 states, GraphReuseSetRowsMC 54 states, no errors.
- ✅ VRAM probe landed + sample output captured (T3.6.M above).
- ✅ Reuse perf delta — closure rationale recorded (no bailout-dropped world to measure; see T3.6.M).

**Token budget:** ~200–250k across audit + specs + tests + impl + measurements + closure.

### T3.7 — DFlash composition — *landed 2026-05-22*

**Result:** Verify-only. T3.5 landed clean; the three DFlash tests in the Framing B closure scope all PASS:
- `bin/test-dflash-np-multislot` GREEN — slot-0 byte-identical across NP ∈ {1,2,4,8} (n_cycles=16, prompt_len=7). Aggregate t/s scales 1→8: 112.5 → 328.5 t/s.
- `bin/test-dflash-closure` GREEN — 8/8 prompts argmax-equivalent vs vLLM PR #40898 at BLOCK_SIZE=4. Cos similarity ≥ 0.999979 per prompt.
- `bin/test-dflash-np-invariance` GREEN — kernel-level drafter_forward np-invariance, 4 seeds × N ∈ {1,2,4,8}, all FNV hashes byte-identical.

`bin/test-dflash-spec-batched-fanout` and `bin/test-dflash-batch-vs-serial` were not in the Framing B closure scope this session; left for follow-up if needed.

### T3.8 — Perf gate GP3.i — *FAIL on gate, theory falsified, Tier 4 justified — 2026-05-22*

**Verdict:** GP3.i fails on all three sub-gates. Honest measurement of record captured in `data/t3.8-perf-gate-ledger.md`:

| Config | Mean t/s | CV | Gate | Result |
|---|---|---|---|---|
| M1 NP=1 ref (no spec-decode) | 24.04 | 0.11% | (sanity) | OK |
| M2 NP=8 batched-bench no-Hadamard | 27.73 aggregate | 0.43% | ≥ 100 conservative / ≥ 130 stretch | **FAIL (28% / 21% of gate)** |
| M3 NP=8 server + Hadamard | 26.49 aggregate | 0.14% | ≥ 90 production-realistic | **FAIL (29% of gate)** |
| M4 graphs-off A/B | 27.84 | n/a | (sanity vs M2) | confirms graphs ≈0% at our shape |

**Root cause of the FAIL — the theory was wrong, not the implementation:**

T3.5's unified-stream multi-seq dispatch is firing at 93% rate in M3 (dispatch counter `total=192 multi_seq=179`, byte-identical across the three M3 runs). The dispatch is delivered, exercised, and deterministic. But M2 (which hits the QNEXT_SEQ_INTERLEAVED sub-batched fallback because `llama-batched-bench` builds a token-major batch — see T3.8.A finding A2-extended) ≈ M3 (T3.5 active) within Hadamard tax. **Unified multi-seq dispatch delivers ~0% additional throughput at decode shape on PSKV-singlewarp + Q4_0 KV + sm_75.**

The PHASE doc's direction-tree estimated +3.6×–4.7× from "kernel batching, NOT graph reuse" (line 1918). This assumption did not hold: at decode n_tokens=1-per-seq the kernel work is grid-parallel per-seq either way; launch/dispatch overhead amortises below per-kernel runtime; memory traffic dominates and isn't reduced by packing. Same generalisation as Tier 2's CUDA-graphs ≈0% finding, now extended to dispatch packing.

**Why vLLM still gets 154.77 t/s on the same hardware:** vLLM's lever is **continuous batching / chunked-prefill admission** (`project_continuous_batching_vs_perslot_dispatch`), splicing new-request prefill into running-decode ubatches. That is **PHASE doc Tier 4 / OpenQ-C** — never lifted, never built. T3 closure does not address it.

**Tier 4 justification triggers fire:** M2 stall fraction = 65.44s / 83.98s = **77.9%** (overwhelmingly prefill-bound at the bench shape); conservative gate misses by 3.6×. Both conditions for the PHASE doc Tier 4 trigger are met.

**What T3 actually delivered:**

- Correctness foundation for multi-stream KV at NP>1 with `--fa on` (T3.0–T3.7, audit-grade with specs + TLA+ + tests).
- T3.5 unified-stream dispatch firing at 93% rate on the server path (no regression).
- T3.6 K-shift / defrag working under both LAYER and GRAPH split with generic CUDA Q→Q cpy support.
- T3.6.M permanent VRAM probe (graph-pool growth bounded < 25 graphs / < 400 nodes / < 110 KB host per device across all M-configs — no runaway).
- All correctness gates (verify-production-determinism, NPC, DFlash composition, kv-shift/defrag/graph-reuse per-stream tests) GREEN.

**What T3 did NOT deliver:** the throughput uplift the GP3.i gate was targeting. The gate's underlying theory was incorrect; the right lever (Tier 4) is a separate phase.

**Branch state at T3.8 closure:** clean; production profile (`profiles/active.sh`) unchanged. Do **NOT** revert T3 changes — they are correctness infrastructure required for future ctx + parallelism scaling regardless of which throughput lever is pursued next. Roll forward to Tier 4 in a fresh phase.

See `data/t3.8-perf-gate-ledger.md` for the full per-run capture, dispatch-counter values, VRAM probe data, variance analysis, and Tier 4 decision rationale.

---

### T3.8 — Perf gate GP3.i (audit-grade restructure, 2026-05-22 — prefatory; see verdict above)

Following the audit-grade pattern established by T3.6, T3.8 splits into A/S/M/E/C sub-cards rather than running benches blind. Findings from T3.8.A drove the restructure.

#### T3.8.A — Audit findings (read-only, 2026-05-22)

**Baseline provenance (verified via grep + ledger reads):**

- **27.73 t/s** at `llama-batched-bench -npp 200 -ntg 64 -npl 8` dual RTX 6000 — `data/ralph-nstream-kv-ledger.md` row 6, captured 2026-05-20 02:55Z, **pre-PSKV-Bug-C-closure**. Production server with Hadamard at same NP captured 27.73 also (PHASE doc ref).
- **23.96 t/s** at the *exact same bench config* — `data/ralph-nstream-kv-ledger.md` row 21, captured 2026-05-21 18:45Z (T2 graphs-on, no T3 yet). Row 21 note: "possibly clock state or session noise."
- **154.77 t/s** vLLM measured on same hardware (Q4 same quant, no Hadamard) — `data/gate0-np1-np8.json` 2026-05-12; project memory `project_continuous_batching_vs_perslot_dispatch`.
- **33.5 t/s** NP=1 single-slot baseline (no contention, no spec-decode).

**A1 — Day-to-day variance is ~14% at identical bench config.** Ledger rows 6 vs 21 disagree by 13.6% on what should be a deterministic measurement. Root cause not characterised — clock state, thermal, system load, or other. **Single-shot T3.8 benches cannot bind the GP3.i gates** (100 t/s and 130 t/s are within the noise band of the baseline). N≥3 runs per config required; report mean ± stddev.

**A2 — `llama-batched-bench` bypasses T3.5 server dispatch.** Source at `examples/batched-bench/batched-bench.cpp:175-186` builds the multi-seq TG batch directly via `common_batch_add(batch, 0, pp+i, {j}, true)` then calls `llama_decode`. T3.5's server-side `process_batch_tokens` split_equal grouping is NOT exercised; `dispatch_multi_seq_count` will not increment under this bench. The bench DOES exercise T3.3 (per-(token,head) row indices), T3.4 (n_stream demands_subbatch gate drop), T3.6 (K-shift/defrag multi-stream support) because `llama_decode` is the entry point those changes modified. **For T3.5 attribution, an HTTP-driven server bench is required separately.**

**A3 — Gate framing inconsistency.** Current gates state "≥ 100 t/s (3.6× over 27.7 t/s baseline)" but the bench config (no Hadamard) measures against the **23.96 t/s baseline**, not 27.73. Corrected framing:
- Conservative ≥ 100 t/s = **4.17× over batched-bench baseline** (23.96 t/s)
- Stretch ≥ 130 t/s = **5.43× over batched-bench baseline**, ~85% of vLLM (154.77)
- The "3.6× / 4.7×" framing only applies if anchored against the Hadamard-on production-server number, which the bench does not measure.

**A4 — DFlash spec-decode is orthogonal.** Production profile (`profiles/active.sh`) runs `--spec-type dflash --draft-max 4 --parallel 2`. `llama-batched-bench` runs no spec-decode at `-npl 8`. Mixing the numbers misleads. Three distinct measurements are valid: (a) batched-bench no-spec-decode (vLLM-comparable), (b) HTTP-driven server with Hadamard + DFlash (production-realistic), (c) NP=1 single-slot reference. T3.8 reports all three.

**A5 — Graph-pool VRAM probe must be captured per-bench.** Now that T3.6.M is permanent, each bench run emits `~ggml_backend_cuda_context: have N graphs (...)`. Capture this to inform whether graph reuse is meaningfully exercised under each config — feeds the future I.b.2 revisit decision.

**A6 — Tier 4 (chunked-prefill admission) contingency clarity.** PHASE doc says "if conservative misses, contingency is Tier 4." But the conservative target ≥ 100 t/s is roughly **2× the row-21 baseline + the kernel-level uplift T3.3/T3.4/T3.6 deliver**. If we miss conservative, it could be (i) the kernel uplift wasn't as large as hoped, (ii) prefill stalls bound at NP=8, (iii) variance dominated. **The Tier 4 decision needs prefill-stall % data**, not just an aggregate t/s miss. T3.8.E should compute prefill-stall fraction from `T_PP / (T_PP + T_TG)` per run.

**A7 — Concurrency hygiene.** Per `feedback_no_overlapping_benchmarks` (one-strike rule) and `feedback_no_concurrent_verify_runs`: every bench launch must `pgrep -f "verify-production|llama-server|llama-batched"` first and abort if any other GPU consumer is running. The `coord/gpu-*.state` flow used by T3.6.M is the right pattern; reuse it.

#### T3.8.S — Methodology checklist

Lightweight (not full Allium — bench methodology, not behaviour). Locked here for the M sub-card to bind against.

- **Per-config gate binding:**
  - C1 (vLLM-comparable, batched-bench no-Hadamard): TG mean ≥ 100 t/s (conservative) / ≥ 130 t/s (stretch), with stddev/mean < 5%.
  - C2 (production-realistic, HTTP-driven server with Hadamard, NP=8): TG aggregate mean ≥ 90 t/s.
  - C3 (NP=1 reference, batched-bench `-npl 1`): TG mean ≈ 33.5 t/s ± 10% (sanity floor — if not, the bench environment is wrong).
- **N per config:** ≥ 3 back-to-back runs, same shell session, no other GPU consumers, same `~/.cache/` warm-state.
- **Capture per run:** TG t/s (S_TG and total), T_PP / T_TG (for stall %), VRAM probe line, `dispatch_multi_seq_count` (server runs only), NPC quick-check.
- **Bench environment freeze:** `pgrep -af "llama-|verify-"` snapshot pre-launch; `coord/gpu-*.state` claimed for entire bench window; commit clean (`git status` clean for tracked files).
- **What kills a run:** any concurrent GPU consumer detected, OOM, NPC quick-check failure, `llama_decode rc != 0`.
- **Variance handling:** if stddev/mean ≥ 5%, increase N to 5 and report; if still high, flag clock-state instability and re-measure under `nvidia-smi -lgc` lock if necessary.

#### T3.8.M — Measurements (the bench bundle)

Four configs (run in this order):

- **M1 — C3 NP=1 reference (5 min):** `llama-batched-bench -npp 200 -ntg 64 -npl 1` × 3 runs. Validates environment + isolates kernel cost per slot without contention.
- **M2 — C1 vLLM-comparable (15 min):** `llama-batched-bench -npp 200 -ntg 64 -npl 8` no Hadamard × 3 runs. Binds GP3.i conservative/stretch gates.
- **M3 — C2 production-realistic (20 min):** HTTP-driven `llama-server` NP=8 with `--k/v-cache-hadamard`, concurrent completion harness (extend `verify-production-determinism.sh` with t/s capture) × 3 runs. Captures full T3 stack (incl. T3.5 dispatch).
- **M4 — C1 graphs-off A/B (optional, 5 min):** `GGML_CUDA_DISABLE_GRAPHS=1` × 1 run for sanity vs ledger row 21 graphs-on/off Δ.

Each run captures the row format already in `data/ralph-nstream-kv-ledger.md`. Append to a new ledger `data/t3.8-perf-gate-ledger.md`.

#### T3.8.E — Evaluation

- **Gate verdict:** binary pass/fail vs C1 mean against conservative + stretch.
- **Uplift attribution:** (C1 - C3) tells kernel-side throughput delta; (C2 server - C1 bench) tells T3.5 server-scheduler delta.
- **Prefill-stall %:** `T_PP / (T_PP + T_TG)` per run. If > 30% AND C1 misses conservative, Tier 4 contingency is justified. If < 15% AND C1 misses, Tier 4 won't help.
- **Graph-pool data:** node-counts and host-bytes per teardown, informs I.b.2 revisit.
- **Variance characterization:** stddev/mean across N=3+ — sets the floor for what gate-band claims are credible.

#### T3.8.C — Closure

- PHASE doc with the numbers + variance bands + gate verdict.
- MEMORY entry: result, variance, surprises, Tier 4 decision.
- Ledger committed.
- Branch left clean.

**Token budget:** ~80–120k across audit + methodology + 4 configs × N runs + evaluation + writeup.

---

### T3.8 — Perf gate GP3.i (original spec, superseded by audit-grade restructure above)

**Bench config — production-realistic:** `llama-server` at NP=8 with `--k-cache-hadamard --v-cache-hadamard`, HTTP-driven `verify-production-determinism.sh`-style concurrent completion harness. Captures TG t/s aggregate including Hadamard overhead.

**Bench config — vLLM-comparable:** `llama-batched-bench -npp 200 -ntg 64 -npl 8` dual RTX 6000 (no Hadamard — apples-to-apples with vLLM's 154.77 t/s measurement).

**Gates:**
- **GP3.i conservative**: TG NP=8 aggregate **≥ 100 t/s** (3.6× current 27.7 t/s baseline) on the vLLM-comparable config.
- **GP3.i stretch**: **≥ 130 t/s** (4.7× current, ~85% of vLLM).
- Production-realistic: TG NP=8 aggregate ≥ 90 t/s under Hadamard (allowing ~10% Hadamard tax).

If conservative misses: the contingency is **Tier 4 (chunked-prefill admission)** — Sarathi-Serve splices new-request prefill into running-decode ubatches, eliminating prefill stalls. Closer to vLLM's continuous-batching scheduler. Defer to post-Tier-3 measurement; only worth it if prefill stalls measurably bound throughput.

If T3 closes between conservative and stretch: declare GP3.i GREEN, evaluate further levers post-closure.

---

## Tier 4 — chunked-prefill admission (Sarathi-Serve) — landed, gate-measured 2026-05-22

T4 lifts the pre-T4 `active_pp_slot_id` PrefillSerialisationGate from the server scheduler and replaces it with multi-slot chunked-prefill admission. New per-tick token budget K (CLI `--prefill-chunk-budget K`, default `n_ubatch`) bounds total prefill admitted in any tick; fair per-slot quota `ceil(K / n_eligible_load_nonembedding)` distributes the budget across LOAD_PROMPT slots. Decode tokens for PROCESSING slots are admitted FIRST by `add_sampled_tokens` (DecodePriorityAdmission); prefill chunks are admitted afterwards. Slots not finishing prefill in one tick carry forward (PrefillCarryProgressesMonotonically). Specifications: `specs/scheduler/batch_composition.allium`, `specs/multislot/BatchComposition.tla`. Stub property test: `ik_llama.cpp/tests/spec/test-chunked-prefill-admission.cpp` (420 swept configs PASS). Trace producer + validator: per-tick `TickDispatch` emission moved from `process_batch_tokens` to `update_slots`; validator at `scripts/validate-batch-composition-trace.py`.

### T4.A — Audit findings (2026-05-22)

**A1 — Bug C non-regression argument holds structurally.** 4D KV layout (per-stream slabs at `ne[3] = n_stream`, uniform `mul_mat` shape per tick) closes Bug C regardless of admission policy. T4 only changes WHICH tokens are in the batch, not the per-tick `mul_mat` call shape. Empirically validated: `r5-probe-c4.sh ITERS=20 = 0/20 violations` under T4.

**A2 — Three defensive-scaffold patterns called out + dropped pre-coherent-flip** (per `[[feedback_bake_measurement_env_gates]]`):
- Bypass bool in original Bundle B plan — dropped, Bundle B collapsed to a single coherent flip.
- TLA+ legacy regression mode (`DecodeHoldGateOn` constant + `LegacyAdmissionOK` action + third MC config) — dropped.
- Trace validator backward-compat (legacy `prefill_slots` field, absent `budget_k` skip) — dropped; all TickDispatch fields now required.
- `LLAMA_FATTN_STRICT_SEQUENTIAL_DECODE` env knob in `batch_pending_prompt` — dropped (NP-determinism debug mode it gated is closed structurally now via T3 + 4D KV).

**A3 — Trace semantics shift from per-dispatch to per-tick.** Under T3.5 split_equal grouping a tick is typically split into prefill-only and decode-only dispatch slices; per-slice tracing would falsely violate `DecodePriorityAdmission`. Per-tick tracing in `update_slots` captures the full batch composition once per tick and binds the validator's invariants correctly. New `tick_trace_state` struct on `server_context` carries the snapshot.

### T4.M — Measurements

**T4.6 — Correctness gate sweep:**

- **GP4.j Bug C absence** (`r5-probe-c4.sh ITERS=20`): **0/20 PASS**.
- **GP4.k NPC byte-identity** (`verify-production-determinism.sh NP={1,2,4,8}`): **PASS** — cross-NP byte-identity AND batch-shape invariance both verified. T3 FRAMING B closure re-confirmed under T4 (dispatch_multi_seq_count 64/64 = 100% during NP=8 segment).
- **GP4.l DFlash composition** (test-dflash-np-multislot / test-dflash-closure / test-dflash-np-invariance): **3/3 PASS** — slot-0 byte-identical across N ∈ {1,2,4,8}, argmax-equivalent on all 8 prompts vs vLLM PR #40898, drafter_forward np-invariant across 4 seeds.
- **GP4.m Trace invariants** (NP=8 + K=256 + 8-prompt 5s staggered, 481 records): **PASS** — TokenBudgetRespected, DecodePriorityAdmission, PerTokenFlagExclusivity, PrefillCarryProgresses all hold. 7 mixed ticks (decode + prefill same batch) exercised the new admission semantics empirically; max batch 110/256 tokens.
- **GP4.n Kernel-level NPC** (test-fattn-per-slot-kv-dispatch-np-invariance): **PASS**.
- **T4.1 unit test** (test-chunked-prefill-admission, 420 swept configs): **PASS**.

**T4.7 — Perf gate** (locked clocks 1455 MHz, N=3 per config, ledger `data/t4-perf-gate-ledger.md`):

| Config | Mean t/s | σ | CV | Δ vs C0 |
|---|---|---|---|---|
| C0 (T3.8 M3 pre-T4 baseline, steady arrival) | 26.49 | 0.037 | 0.14% | — |
| C1-steady (T4 + same steady arrival) | 26.49 | 0.014 | 0.05% | 0.0% |
| C1-staggered (T4 + 5s arrival offsets) | 21.62 | 0.016 | 0.07% | −18.4% |

- **GP4.i.a (regression band, C1-steady ≥ C0×0.98 = 25.96):** **PASS** — zero regression.
- **GP4.i.b (uplift binding, C1-staggered ≥ C0×1.20 = 31.79):** **FAIL** — 21.62 t/s, 18.4% below C0.
- **GP4.i.c (variance, CV ≤ 1%):** **PASS** — 0.05% / 0.07%.

### T4.E — Evaluation

**Verdict: T4.6 GREEN, T4.7 FAIL on GP4.i.b. T4 correctness layer landed; perf justification gate measured negative.**

**Why GP4.i.b structurally fails.** The gate target was aggregate t/s ≥ steady baseline × 1.20. Staggered arrival has a longer wall floor than steady on this metric — C0 = 26.49 t/s **is** the multi-slot kernel saturation throughput at NP=8 on RTX 6000 sm_75 for Qwen 3.6 27B Q4_0 with Hadamard. Staggered arrival under-utilises the multi-slot kernel during the ramp-up window (only slots 0–2 active for t = 0–10 s), so aggregate t/s is mechanically lower than steady. The gate as specified asked staggered to exceed steady on aggregate t/s, which is unreachable on this kernel at this NP — independent of T4 admission policy.

**Where T4 admission DOES deliver:** workloads with **high prefill rate** (many short prompts in burst arrival) or **long prompts** (chunked admission keeps decode going during the prefill tail). M3-staggered with 200-token prompts at 5 s gaps is neither: prefill at PP ~60 t/s finishes in ~3.5 s, leaving 1.5 s slack before the next prompt arrives, so pre-T4 serialised prefill doesn't bottleneck the staggered workload either. T4 admission's value is structural (correctness gates GREEN; spec layer + admission scaffold + trace producer load-bearing for future work) but does not surface as a throughput uplift at this workload + model + hardware shape.

**Hardware constraint reaffirmed.** vLLM's measured 154.77 t/s on the same hardware (single shared model) is the per-paged-KV ceiling. T4 does not close that gap; the next lever (Tier 5 paged KV, listed in Open questions) would target it but is out of scope for this PHASE doc. Per `[[project_continuous_batching_vs_perslot_dispatch]]` the vLLM uplift is paged-KV plus continuous batching combined; we have continuous batching now (T4) but not paged KV.

**Per `[[feedback_oneshot_then_evaluate]]`:** "negative results land cheap when honest". GP4.i.b is closed FAIL with measurement of record. No follow-up cover. Per `[[feedback_no_workarounds]]` the perf gate target is not re-defined post-measurement to manufacture a PASS — the structural reason for the miss is documented and the deliverable scope stands.

### T4.C — Closure

- Spec layer: `specs/scheduler/batch_composition.allium`, `specs/multislot/BatchComposition.tla` (T4 form), `scripts/validate-batch-composition-trace.py` (T4 form), `ik_llama.cpp/tests/spec/test-chunked-prefill-admission.cpp`. All landed in commits `fa935eb`, `fc7d7f3`, `6ad6140`, submodule `0759c01c` + parent `9fb4e6e`, with cleanup commits `7635d04` + `35632f2`.
- Coherent flip: submodule `e282d229` (T4 admission in `batch_pending_prompt`, CLI `--prefill-chunk-budget K`, per-tick TickDispatch trace), parent `eb426e0` (submodule bump).
- T4.6 correctness gates all GREEN — see `data/t4-perf-gate-ledger.md` and per-test outputs.
- T4.7 perf gate FAIL on GP4.i.b — see `data/t4-perf-gate-ledger.md`. Harness `scripts/bench-t4-m3-staggered.sh` (new) + `scripts/bench-t3.8-m3.sh` (reused for C1-steady).
- MEMORY entry: `2026-05-22 — T4 chunked-prefill admission (Sarathi-Serve) coherent flip landed` (this file's MEMORY.md).
- Auto-memory: `~/.claude/projects/-home-llm-yarn-agentic/memory/project_t4_bundle_a_landed.md` updated with the T4.6 + T4.7 results and lessons.

**Production impact.** Production profile (`profiles/qwen36-27b-x2-dflash.sh`, NP=2 + DFlash) is **UNCHANGED**. T4 admission code is in-tree but its behaviour at default `--prefill-chunk-budget 0` (= n_ubatch = legacy chunk size) is byte-identical to pre-T4 admission policy under verify-production-determinism at NP={1,2,4,8}. No production action required from T4 closure.

**Recommendation for next workstream.** T4.7 FAIL on aggregate t/s does NOT invalidate the admission scaffold — the spec layer and admission code are correct and ready for the workloads they're designed for. If a future workstream targets burst short-prompt arrival or long-prompt prefill (e.g., RAG with long retrieved context), T4 admission is the foundation. Otherwise paged KV (Tier 5) is the next lever for catching vLLM's 154 t/s ceiling.

---

## Open questions and refinements

These need closure during T3 execution; not blocking the bundle starts but should be answered as the relevant card lands.

**OpenQ-A: Token ordering convention in the unified ubatch.** Tokens-per-seq contiguous (seq 0's all tokens, then seq 1's, etc.) or interleaved (token 0 of all seqs, then token 1, etc.)? **RESOLVED 2026-05-21 at T3.2 start: contiguous-per-seq with `split_equal` shape-uniform ubatches per tick.** Convention evidence converges: upstream PR #14363 `split_equal` semantics; `src/llama-delta-net.cpp:253-380` already consumes `[..., n_seq_tokens, n_seqs]` packing; PSKV kernel grid (`blockIdx.x = tok`, `blockIdx.z = seq`) is built for seq-outer 4D layout; T3.1 `nb33` mask addressing assumes `[n_kv, n_tok_per_seq, 1, n_stream]`. Variable `n_tok_per_seq` across streams within one tick is handled by splitting the batch into K shape-uniform ubatches (split_equal) — one `llama_decode` per ubatch. For the common all-decode case K=1.

**OpenQ-B: Bench-with-Hadamard for production-realistic perf.** `llama-batched-bench` has no Hadamard flag (silently runs without). Production uses it. **Resolves at T3.8.** Options: (a) drive bench via `llama-server` HTTP for production parity, (b) extend `llama-batched-bench` with Hadamard flags, (c) measure both and report (we publish two numbers: vLLM-comparable + production-realistic). Recommend (a)+(c).

**OpenQ-C: `defrag` and `ctx_shift` at n_stream>1.** Per-stream defrag is a per-stream cell shuffle; per-stream ctx_shift is a per-stream window slide. Effort may be larger than "lift the assertion". **Resolves at T3.6.** Scope: confirm the existing `build_k_shift` / `build_defrag` graph builders compose with the 4D layout, OR write per-stream variants. Token budget: 20-30 k if straightforward, +30-50 k if per-stream variants needed.

**OpenQ-D: MMQ I=8 fixes under unified shapes.** PHASE_MMQ_Q4_0_AR16 closure baked the I=8 split-K disable + col-j > 0 fix. Those were validated at per-stream dispatch shapes (n_tokens=1). Under T3 unified at n_tokens = n_stream × n_tok_per_seq, mul_mat sees larger shapes. **Resolves at T3.3.** Test gate: P0.A.3 batch-shape invariance harness (`scripts/test-batch-shape-invariance.sh` or equivalent) at the unified shapes. If the fixes don't compose, the fallback is to keep per-stream dispatch ONLY for the multi-seq mul_mat (a partial unification that captures most of the win without the mul_mat risk).

**OpenQ-E: T3 perf contingency if conservative target misses.** **Resolves at T3.8.** Levers identified in priority order:
1. Tier 4 — chunked-prefill admission (Sarathi-Serve). Estimated +10-30% on prefill-heavy workloads.
2. Re-open the PSKV singlewarp FA optimisation ralph loop (cancelled per `feedback_no_workarounds` history). At ne[3]>1 the kernel's ILP-2026-05-18 ratchet may have additional headroom.
3. Profile via `nsys` + `ncu` to find the new dominant kernel under unified dispatch. Could be: output projection, FFN, KV cache CPY.
4. Investigate whether vLLM's specific kernel fusions (RMSNorm + Q/K/V projection, etc.) close the remaining gap.

**OpenQ-F: `mtp_op_type == MTP_OP_DRAFT_GEN_FUSED` composition at NP>1.** Phase 37 #5 in `can_reuse_graph` allows n_tokens>1 reuse under MTP fused. Under T3.6 the bailout drops; MTP fused at NP>1 enters the multi-seq unified path. `specs/composition/mtp_fused_x_n_stream.allium` covers the contract; `test-mtp-x-n-stream.cpp` is the gate. **Resolves at T3.6.** Track via GP3.m.

**OpenQ-G: Sched intermediate-buffer aliasing concern from Tier 2 diagnostic.** Tier 2 found stream-interleaving-order-dependent K/V cache corruption (interleaving-order-dependent garbled outputs, hypothesised to be sched buffer aliasing across cross-tick stream switches). Under T3 the dispatch is ONE graph eval per tick covering all streams — no cross-tick switching. **The class of bug should not exist by construction.** But: confirm with the same FA-probe diagnostic re-run under T3.5 + T3.6 to byte-verify K/V cache contents match per-stream-correct expectations. Captured in the post-T3.6 verification (GP3.g warm-up determinism + GP3.f Bug C absence).

**OpenQ-H: Stream-mid switching during a tick.** If two consecutive ticks process different sets of slots (slot A's request completes, slot B's starts mid-stream), does the second tick's graph reuse work cleanly? The graph topology depends on `n_stream` and per-stream `n_kv` bucketing; tick-to-tick changes in active-slot membership may trigger graph rebuild. **Resolves at T3.5.** This is a normal graph-cache hit/miss — should compose with the existing topology-keyed cache; no special handling needed.

**OpenQ-I: When does Phase 0.B close formally?** 5 .allium specs + 7 property tests are in tree. Some of the property tests are RED (binding RED for T3 — they FAIL on HEAD and PASS after the matching T3 card lands). **Resolves at T3.4** (Bundle A close): all spec property tests GREEN at n_stream=1 production shape AND at synthetic multi-seq batches via the new code paths. **Closes formally at T3.6** (Bundle B close): all property tests GREEN at production n_stream=8 with full unified dispatch.

---

## What's NOT in scope

- New correctness gates (PHASE_NSTREAM_KV's six gates remain the preservation checks).
- N-stream **layout** changes (the 4D port itself is closed).
- DFlash perf (separate workstream).
- PSKV singlewarp FA optimisation (separate ralph loop, currently cancelled).
- vLLM/SGLang/LMDeploy pivot — ruled out on evidence above.
- env-gated experimental knobs (per `feedback_bake_measurement_env_gates`).

## Token estimate (refined 2026-05-21)

Per CLAUDE.md §8:

**Already spent (closed):**
- **Phase 0.A** — DFlash server CLI verify-on-post-fold: ~110 k spent on landed fixes (MAL cap + stage end-trim) + falsification matrix. P0.A.3 deferred (~30-100 k, parallel workstream, does NOT gate T3 since production runs MTP not DFlash CLI).
- **Phase 0.B** — spec/TLA+/test surface expansion: ~105-170 k spent. 4 .allium specs in tree post-T3.0 (`scheduler/batch_composition`, `kv-cache/n_stream_layer`, `dispatch/unified_stream_dispatch`, `composition/mtp_fused_x_n_stream`; `kv-cache/per_stream_read_view_patching` removed by T3.0 with its companion test as the Tier 2 read-view-patching contract no longer exists). 6 property tests in `tests/spec/` (one removed in T3.0). Some tests RED-bound to specific T3 cards.
- **Tier 1 / Tier 2 closure** — diagnostic + perf A/B: ~50-80 k spent (FA-probe build/run/analyse, graphs ON-vs-OFF A/B, ledger rows 20-21, PHASE retrospective, mdBook update).

**Remaining (Tier 3 bundles):**
- **Bundle A (T3.0–T3.4)** — scaffolding removal + find_slot multi-seq + 4D build + multi-seq decode path: **80-120 k**. Each step adds a property-test gate; production NPC stays GREEN throughout.
- **Bundle B (T3.5–T3.8)** — server-context unified dispatch + bailout drop + n_stream==1 guard lifts + DFlash composition + perf gate: **60-100 k**.
- **Open-question resolution** (OpenQ-A through OpenQ-I) — surfaced inline at the relevant cards; budgeted within the bundle estimates above.
- **Diagnosis budget** if any card hits unexpected NPC regression: 20-30 k per round, budget 2-3 rounds.

**Total remaining scope: 140-220 k tokens to Tier 3 closure.** Significantly less than the original 305-510 k estimate because (1) PSKV per-slot kernel is already production, (2) T3.1 nb33 foundation already landed, (3) spec layer substantially in place, (4) Tier 1 falsification frees us from engineering for graph reuse retention.

Per `feedback_oneshot_then_evaluate`: Bundle A lands coherently (single commit, or logical sub-commits per card if NPC verification requires staged checkpoints). Bundle B lands coherently. No partial intermediate landings of the dispatch flip.

## Primary research sources

CUDA-graph mechanism:
- [NVIDIA — Optimizing llama.cpp with CUDA Graphs](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/)
- [NVIDIA — CUDA Graphs in a Dynamic Environment](https://developer.nvidia.com/blog/employing-cuda-graphs-in-a-dynamic-environment/)
- [NVIDIA — Constructing CUDA Graphs with Dynamic Parameters](https://developer.nvidia.com/blog/constructing-cuda-graphs-with-dynamic-parameters/)

Upstream context:
- [llama.cpp PR #14363 — high-throughput mode (the n_stream port)](https://github.com/ggml-org/llama.cpp/pull/14363)
- [llama.cpp issue #14863 — Blackwell multi-GPU TG regression](https://github.com/ggml-org/llama.cpp/issues/14863)
- [llama.cpp PR #7302 — avoid disabling CUDA graphs](https://github.com/ggml-org/llama.cpp/pull/7302)
- [llama.cpp discussion #4130 — parallelisation roadmap](https://github.com/ggml-org/llama.cpp/discussions/4130)
- [llama.cpp release notes April 2026 — Walsh-Hadamard KV](https://fazm.ai/blog/llama-cpp-release-april-2026)

ik_llama.cpp ground truth:
- [PR #1080 — graph parallel: next generation](https://github.com/ikawrakow/ik_llama.cpp/pull/1080)
- [Discussion #1247 — ik_llama.cpp vs mainline (33 % TG / 6-9× PP advantage)](https://github.com/ikawrakow/ik_llama.cpp/discussions/1247)

Continuous batching (Tier 3 anchor):
- [Orca (OSDI '22)](https://www.usenix.org/system/files/osdi22-yu.pdf)
- [vLLM / PagedAttention (SOSP '23)](https://arxiv.org/abs/2309.06180)
- [Sarathi-Serve chunked prefill (OSDI '24)](https://arxiv.org/abs/2403.02310)
- [vLLM Anatomy 2025](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)
- [SGLang scheduler internals](https://deepwiki.com/sgl-project/sglang/4.2-token-sampling-and-generation)

sm_75 / Turing constraints:
- [vLLM #38918 — no working attention backend on Turing](https://github.com/vllm-project/vllm/issues/38918)
- [vLLM #29743 — Qwen3-VL Turing closed-not-planned](https://github.com/vllm-project/vllm/issues/29743)
- [vLLM #8669 — GGUF Q4 at 8.7 t/s on A100](https://github.com/vllm-project/vllm/issues/8669)
- [Dao-AILab/flash-attention #542, #720 — FA2 Turing backlog stalled](https://github.com/Dao-AILab/flash-attention/issues/720)
- [Turing Tuning Guide (CUDA docs)](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html)

Internal prior work (load-bearing for triangulation):
- `data/phase45-d10b-bench.md` — multi-slot batched draft +27 % aggregate
- `data/pskv-ilp-recovery-2026-05-18.md` — PSKV +2.95 % TG / +9.17 % PP, NPC preserved
- `data/gate0-np1-np8.json` — vLLM 4.75× measurement on our hardware (2026-05-12)
- Memory: [[project_continuous_batching_vs_perslot_dispatch]] — overturns PHASE45 D10.e abandonment for continuous-batching case
- Memory: [[feedback_n_stream_byte_compat_tradeoff]] — axis-order constraint (heads outer per stream)
- Memory: [[project_fattn_per_slot_kv_p2_landed_kernel_only]] — PSKV kernel landed, NPC preserved at NP={1,2,4,8}
- Memory: [[project_pskv_ilp_recovery_landed]] — recent perf win pattern (NPC preserved)
- Memory: [[feedback_bake_measurement_env_gates]] — no LLAMA_*_ENABLE knobs left around
