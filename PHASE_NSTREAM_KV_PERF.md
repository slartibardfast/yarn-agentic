# PHASE_NSTREAM_KV_PERF — recover the regression *and* unlock the dispatch ceiling

**Branch**: `production/2026-q2-next` (off submodule HEAD `16b608d1`).
**Predecessor**: `PHASE_NSTREAM_KV.md` — closed 2026-05-20. Bug C structurally closed; decode-side prefill gate removed; 6 correctness gates green; **-6.2 % TG NP=8 regression carried over**.
**Status**: Open. **Phase 0 prerequisites must land before Tier 2 work begins.** Direction tree below, post-prereq starting hypothesis locked to **Tier 2 refined** (extend existing `update_cache_copies` per-stream patching to the attention-read views).
**Triangulated 2026-05-20** against prior CUDA-graph work, PSKV per-slot landing, P3.X NPC failures, PHASE45 D10 multi-slot work, and `feedback_n_stream_byte_compat_tradeoff`.

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

#### P0.A.3 — DFlash output divergence from spec-none baseline [OPEN — fundamental, needs dedicated diagnostic phase]

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

#### P0.A.4 — Multi-slot DFlash SEGV in `llama_sampling_prepare` [OPEN — bounded scope]

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

Tier 2 entry condition therefore relaxes to:
- ✅ P0.A.1 (MAL cap) — landed.
- ✅ P0.A.2 (stage end-trim) — landed.
- ⏸ P0.A.3 (output divergence) — documented; deferred to dedicated diagnostic phase.
- ⏸ P0.A.4 (multi-slot SEGV) — documented; deferred (likely subsumed by P0.A.3 root cause).
- ✅ P0.A.5 (production server boots clean on post-fold for the non-DFlash production profile) — verified.

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

## Direction tree (refined and primary-sourced)

| Tier | What | Expected gain | Effort | Risk |
|------|------|---------------|--------|------|
| 1 | Drop `can_reuse_graph` n_stream > 1 bailout + accept rebuild storm at first stream-switch | +6.2 % recovery (avoids the cudaGraphInstantiate, hits cudaGraphExecUpdate path) | ~1 day | Low |
| **2** | **Patch attention-read view offsets per-stream in `update_cache_copies`** (mirroring the K/V write CPY patching that already ships) + drop the bailout | **+15-30 % beyond Tier 1** at NP=8 (full reuse, only update cost per stream) | **~1 week** | Low-Medium |
| 3 | Unified-stream dispatch: one `llama_decode` per tick with N_stream tokens packed; use existing PSKV per-slot kernel | **Approaches vLLM's measured 154.77 t/s aggregate at NP=8** (4.75× over NP=1) | 3-5 weeks | Medium-High |
| 4 | + chunked-prefill admission ([Sarathi-Serve](https://arxiv.org/abs/2403.02310)) | small additional gain on prefill-heavy workloads | +2 weeks | Low after Tier 3 |
| 5 | Full paged-KV port (V1 vLLM kernel, sm_70+) | Marginal beyond Tier 3 for our workload | 6+ months | Very high |

**SKIP**: vLLM pivot (loses Q4_0 + Hadamard + Q4_0 KV — uniquely SoTA on sm_75; nothing else in the ecosystem can consume this weight stack); persistent-kernel megakernels (Luce sm_75 batch=1 only; Mirage Hopper-first).

### Tier 1 — Drop the bailout

Simplest possible change: delete `if (transformer_kv.n_stream > 1) { ... return false; }` at `src/llama.cpp:616`. First decode for each stream still triggers a cudaGraphInstantiate (because topology-hash hits an unpopulated entry), but subsequent decodes for the same stream hit the cache and route through `cudaGraphExecUpdate`. Net: ~N graph instantiations at warm-up (one per stream), then steady-state reuse.

**Why this alone may not work**: the existing graph cache check compares `node->src[i]->data` against captured `src_address[i]`. For the **attention-read** view tensors in `llm_build_kqv`, the `data` pointer encodes the stream's offset and changes per stream. The check at `ggml-cuda.cu:4711-4717` returns false for src_address mismatch *unless* the op is `GGML_OP_CPY` or `GGML_OP_VIEW`. Whether the attention-read tensors register as `GGML_OP_VIEW` (legitimately tolerated) or as e.g. `GGML_OP_FLASH_ATTN_EXT_PER_SLOT_KV` (which would invalidate) needs source confirmation before locking the design. **T1.a is that probe.**

### Tier 2 — Patch attention-read views in `update_cache_copies` (LOCKED STARTING HYPOTHESIS)

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

## Implementation cards (Tier 3 — sketch, locked at Tier 2 close)

T3.a — **Probe**: read upstream PR #14363 `split_equal` semantics and unified ubatch graph builder changes. Map to ik_llama.cpp equivalents. Specifically identify how upstream handles the (Q/K/V projection, RMSNorm, MLP, output proj) shape-invariance across the unified-batch dimension.
T3.b — **Server-side fusion** in `process_batch_tokens` (`examples/server/server-context.cpp:4610`). Replace the seq_id-run split with a unified ubatch builder. Preserve sub-batching for INTERLEAVED patterns at n_stream>1 (the bench-path fix from `16b608d1`).
T3.c — **KQ mask builder**: emit `[n_kv, n_tokens/n_stream, 1, n_stream]` shape. Per-stream row filtering via existing seq_id mask logic.
T3.d — **FA dispatch verification**: confirm `GGML_OP_FLASH_ATTN_EXT_PER_SLOT_KV` is emitted by `llm_build_kqv` for the unified-ubatch shape and routes to the PSKV pb1 wrapper. The kernel handles n_stream at ne[3] via the per-slot mask + KV pointer arrays per slot. `per_row_k_bound` is plumbed (currently a no-op for correctness — mask handles bound enforcement; future perf opt for loop-tail trim).
T3.e — **Non-FA shape-invariance audit**: Q/K/V projection matmuls, RoPE, RMSNorm, MLP, output projection. P2 memory flagged these as divergence sources at NP>1 pre-PHASE_NSTREAM_KV ("E2 NP=4 sequential showed run2 differing from run1+run3"). PHASE_NSTREAM_KV's per-stream dispatch resolved this empirically. Tier 3 puts these ops back at ne[1]>1 within a tick. Verify via GP3.j (all Tier-2 gates remain GREEN including NPC byte-identity).
T3.f — **Drop `n_stream==1` guards** on `build_k_shift` / `build_defrag` / `v_trans` per PHASE_NSTREAM_KV's open-follow-up list. Required for full multi-slot ctx_shift + cache compaction; previously gated because of per-stream-dispatch impedance mismatch.
T3.g — **Bench gate GP3.i** against vLLM's 154.77 t/s measurement.

## What's NOT in scope

- New correctness gates (PHASE_NSTREAM_KV's six gates remain the preservation checks).
- N-stream **layout** changes (the 4D port itself is closed).
- DFlash perf (separate workstream).
- PSKV singlewarp FA optimisation (separate ralph loop, currently cancelled).
- vLLM/SGLang/LMDeploy pivot — ruled out on evidence above.
- env-gated experimental knobs (per `feedback_bake_measurement_env_gates`).

## Token estimate

Per CLAUDE.md §8:

- **Phase 0.A** — DFlash server CLI verify-on-post-fold (wiring + profile already landed in `61a7e874`). **Actual cost ~110 k** for the two fixes that landed (MAL cap + stage end-trim) PLUS the falsification matrix that confirmed T5 as the root cause (cb_eval hook perturbs target's forward pass; ngram-simple at same verify-batch shape produces clean output byte-identical to spec-none). **Plus deferred:** 30-100 k for the P0.A.3 fix (preferred path: re-architect hidden-state capture without cb_eval, ~30-50 k). Tier 2 entry does NOT block on P0.A.3 — production runs MTP, not DFlash CLI.
- **Phase 0.B** — Radical spec/TLA+/test surface expansion (5 Allium + 3 TLA+ + 5 property tests + trace harness ext.): **105-170 k tokens**. Must precede T2.a.
- **Tier 2 refined** — read-view probe + cache_copies extension + bailout drop + warm-up gate harness + GP3.a-h verification: **60-100 k tokens**.
- **Tier 3 refined** — server fusion + KQ-mask shape + graph builder uniform-batch + non-FA shape-invariance verify + GP3.i-m: **120-180 k tokens** (significantly less than the original "fresh FA kernel port" framing because the PSKV per-slot kernel is the prerequisite and it's already production).
- Diagnosis if Tier 2 hits unexpected invalidation: 20-30 k per round, budget 2-3.

Total scope: **305-510 k tokens** for this phase's required path to Tier 3 closure (excludes the deferred P0.A.3 dedicated diagnostic phase, which is its own 100-300 k workstream and can run in parallel). The structural foundation (PHASE_NSTREAM_KV's 4D layout + the existing CUDA-graph patching infra + the PSKV per-slot FA kernel) carries most of the engineering risk; this phase is the dispatch refit *plus* the surface expansion that should have shipped with PHASE_NSTREAM_KV but didn't.

Per `feedback_oneshot_then_evaluate`: Phase 0.A and 0.B can run in parallel as separate bundles. Tier 2 implements coherently after both close. Tier 3 implements coherently after Tier 2 closes. No partial intermediate landings.

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
