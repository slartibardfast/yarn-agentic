# PHASE_DFLASH — sm_75 implementation design

Target: implement DFlash speculative decoding on `2026-q2-next` (branched from `2026-q2`) of `yarn-agentic` and `ik_llama.cpp`. This document synthesises three reference implementations (vLLM, SGLang, upstream llama.cpp) against our local fork state and the hardware constraints of 2× Quadro RTX 6000 (Turing, sm_75), and picks the design that maximises shipped speedup on this hardware.

The companion Allium behavioural contract lives at `dflash_speculative.allium`. This doc commits the implementation shape; the spec commits the correctness obligations.

---

## 1. The single most consequential decision

**Inject target features via per-anchor K/V cache writes in the draft, not via T5-style cross-attention with per-step encoder rebuilds.**

Both vLLM (`qwen3_dflash.py::precompute_and_store_context_kv`) and SGLang (`dflash_worker.py::_append_target_hidden_to_draft_kv*`) do this. Upstream `llama.cpp` PR #22105 does the opposite — uses `llama_cross` machinery with an encoder context that rebuilds every step. The upstream PR's author flags this as the dominant cost in the "Future Performance Work" section ("graph reuse = 0 because `cross.n_enc` grows monotonically"), and a community datapoint on RTX 3090 (architecturally close to our sm_75) measured **−44.6%** vs baseline on 35B-A3B MoE.

Concretely:

- Projection runs **once** per accepted bonus token, not once per draft denoising step.
- Draft forward becomes plain self-attention against pre-written K/V cache slots — no encoder/decoder dual-context, no `cross.v_embd` monotonic growth, no graph-cache invalidation per cycle.
- The drafter's self-attention path on sm_75 routes to **`mma_new`** (per PHASE5 unit tests) which is byte-deterministic at production shape — we keep the existing kernel determinism guarantees.

The cost is one extra `K_proj + V_proj + RMSNorm(K) + RoPE(K) + cache_write` per draft layer per accepted token. This is small (one position per layer, scalar by comparison to the drafter's own forward). vLLM and SGLang have fused Triton kernels for it; on sm_75 we do per-layer scalar dispatch and accept the perf cost. The fused kernel is a Hopper latency optimization, not a correctness requirement.

## 2. Reuse, don't port

ik_llama.cpp's `common/speculative.cpp` is 1646 lines (vs upstream's ~500), mostly MTP-IR plumbing. The MTP-IR API surface (`llama_mtp_op_type`, `llama_mtp_fused_result`, top-2 cache, draft-argmax cache) is **dead weight under DFlash** — DFlash is an external-draft method, not a head-on-target. We keep it (removing it is a separate cleanup arc) but route the DFlash path through `examples/speculative/` instead of through the MTP API. This matches PHASE_DFLASH_SCOPING.md Option C, modified by §1 above.

What we explicitly reuse from the existing fork:

- The **accept-decision** in `common/speculative.cpp` (argmax_match, longest-prefix, bonus token). The Allium spec `AcceptPrefixDecision` covers both MTP and DFlash with the same semantics; the implementation is one function.
- The **target-side verify forward** — verify is just a multi-position decode of `[bonus, draft_0, …, draft_{N-1}]`. Standard target forward at ne[1] = block_size+1. No new kernels.
- The **slot allocator** for per-seq KV partitioning. np=1 only, so this is trivial; but the existing infrastructure already handles draft-vs-target slot mapping.
- The **GGUF loader** for shared tensors. ik_llama.cpp's loader supports tensor aliases/views; the drafter's `tok_embeddings` and `output` weights point to the target's via `target.*` keys.

What we explicitly do NOT take from upstream's PR #22105:

- **No `llama_cross` / cross-attention plumbing.** Replaced by KV-cache-write (see §1).
- **No encoder/decoder dual-context pattern.** A single draft context with its own KV cache suffices.
- **No EAGLE3 base port.** Upstream's #22105 sits atop EAGLE3 PR #18039 (also draft). The EAGLE3 base contributes the dual-context machinery, which we don't need. We implement DFlash directly.
- **No `ENCODER → DECODER` graph-type override.** Single graph build per draft forward.

## 3. Architecture summary

```
  ┌──────────────────────────────────────────────────────────────┐
  │ Target (Qwen3.6-27B, IQ4_KS, ik_llama.cpp existing kernels)  │
  │                                                              │
  │   ┌────────────┐                          ┌──────────────┐   │
  │   │ verify(    │                          │ extract_     │   │
  │   │   bonus,   │ → logits per position →  │ residual_    │   │
  │   │   block)   │                          │ stream(L_i,  │   │
  │   └────────────┘                          │   anchor_pos)│   │
  │                                           └──────┬───────┘   │
  └──────────────────────────────────────────────────┼───────────┘
                                                     │
                              hidden features at K source layers
                              (one position: the new anchor)
                                                     │
                                                     ▼
       ┌─────────────────────────────────────────────────────────┐
       │ Projection (loaded from drafter GGUF)                   │
       │                                                         │
       │   concat([h_L1, h_L2, …, h_LK]) → fc → hidden_norm      │
       │                                                         │
       │              fused feature ∈ ℝ^(draft_hidden)           │
       └────────────────────────┬────────────────────────────────┘
                                │
                  per draft layer i:
                                │
                                ▼
       ┌─────────────────────────────────────────────────────────┐
       │ Inject (drafter's k_proj_i, v_proj_i)                   │
       │                                                         │
       │   K_i = RoPE(RMSNorm_k(k_proj_i(fused)))                │
       │   V_i = v_proj_i(fused)                                 │
       │   write (K_i, V_i) → draft_kv_cache[layer=i, pos=anchor]│
       └─────────────────────────────────────────────────────────┘

                                │
            for each new accepted anchor (every cycle)
                                │
                                ▼
       ┌─────────────────────────────────────────────────────────┐
       │ Drafter forward (single pass, block_size positions)     │
       │                                                         │
       │  input = [anchor_token, MASK, MASK, …, MASK]            │
       │  positions = [anchor_pos, +1, +2, …, +block_size-1]     │
       │                                                         │
       │  self-attention, layer-type-dependent mask:             │
       │    sliding layers: causal SWA, window = 2048            │
       │    full layer    : bidirectional (non-causal)           │
       │  K/V at anchor_pos come from prewritten cache slots     │
       │  K/V at anchor_pos+k for k≥1 are computed from MASK     │
       │                                                         │
       │  output: logits at positions 1..block_size-1            │
       │  draft_tokens = greedy_argmax(logits)                   │
       └────────────────────────┬────────────────────────────────┘
                                │
                                ▼
       ┌─────────────────────────────────────────────────────────┐
       │ Target verify ([bonus, draft_0, …, draft_{N-1}])        │
       │ Accept longest-prefix; commit bonus.                    │
       │ Loop.                                                   │
       └─────────────────────────────────────────────────────────┘
```

Match this to the Allium contracts: `ExtractFeatures` (top box), `ProjectAndFuse` (projection box), `InjectKV` plus the inline `K_i/V_i` math (inject box, satisfies `InjectionConsumedAtEveryLayer`), `DraftBlockEmit` (drafter forward), `TargetVerifyBlock` + `AcceptPrefixDecision` + `AdvanceState` (verify + accept + commit).

## 4. File-level change plan

For ik_llama.cpp on branch `production/2026-q2-next`. Numbers are rough estimates; the gates in §6 bind the actual work.

| File | Change | Rough LOC |
|---|---|---|
| `src/llama-arch.h` / `src/llama-arch.cpp` | Register `LLM_ARCH_DFLASH`; tensor name mappings (`LLM_TENSOR_DFLASH_FC`, `LLM_TENSOR_DFLASH_HIDDEN_NORM`); GGUF metadata keys (`dflash.target_layer_ids`, `dflash.block_size`, `dflash.mask_token_id`, `dflash.swa_window`, `dflash.target_arch`, `dflash.target_n_embd`). | +60 |
| `src/llama-model.cpp` | DFlash arch dispatch; drafter weight loading (fc, hidden_norm, intermediate layers); shared embed/lm_head from target_model. | +200 |
| `src/llama-build-context.cpp` | New `build_dflash_drafter()` graph (standard transformer with causal-SWA mask). New `build_dflash_inject_kv()` graph (per-layer K_proj/V_proj/norm/RoPE on the fused feature, write to draft KV). Hook the existing qwen3_5_text builder to capture residual-stream snapshots at the K source-layer indices via `ggml_set_output`-style markers. | +250 |
| `src/llama-context.cpp` | `llama_set_dflash(ctx_tgt, model_dft)` API; `extract_dflash_features(ubatch)` async D→H copy at target post-step; `apply_inject_kv(ctx_dft, features, anchor_pos)` runs inject graph and writes to draft KV; per-step orchestration hook. | +250 |
| `include/llama.h` | Public C API surface: `llama_set_dflash`, `llama_get_dflash_block_size`, `llama_get_dflash_mask_token_id`, `llama_get_dflash_swa_window`, `llama_dflash_extract_features`, `llama_dflash_draft_block`. No new types beyond `llama_dflash_state` opaque handle. | +50 |
| `common/speculative.cpp` | `common_speculative_dflash_init/free/draft/accept`. The draft step runs the block-emit, the accept step reuses existing `argmax_match` longest-prefix logic. | +180 |
| `convert_hf_to_gguf.py` | `class DFlashModel(Qwen3Model)` for ik_llama (port from upstream PR's #22105 converter); `--target-model-dir` flag; write fc, hidden_norm, metadata; share tokenizer/vocab from target. | +200 |
| `examples/speculative-simple/speculative-simple.cpp` | Add `--dflash` flag; wire `common_speculative_dflash_*` into the existing speculative loop. | +50 |
| `tools/server/server-context.cpp` | 20-line gate: `if (params.speculative.dflash && params.n_parallel == 1) llama_set_dflash(...)`. Error out at np>1. | +20 |
| Tests | `tests/test-dflash-extract.sh` (residual-stream capture binding test); `tests/test-dflash-inject.sh` (per-layer K/V write binding); `tests/test-dflash-block.cpp` (drafter forward + accept under known prompt); reuse `test_flash_attn_ext_batched_det` at the verify shape (ne[1] = block_size+1). | +250 |

**Total estimated:** ~1500 lines new + adapter glue. Well under PR #22105's +1900 because we skip the encoder/decoder dual-context, the cross-attention plumbing, and the EAGLE3 base.

## 5. Where we differ from each reference impl on hot details

### Block-emit mask
- **vLLM:** layer-type-dependent. sliding_attention layers get `causal=True` (causal SWA); full_attention layers get `causal=False` (bidirectional). Asserted at `vllm/v1/spec_decode/dflash.py:435-444`.
- **SGLang:** `custom_mask=None` defaults to causal across all layers — does NOT match the trained drafter's full_attention layer.
- **Upstream llama.cpp:** no mask (full attention, every position attends to all) — wrong for the SWA layers.
- **z-lab transformers reference:** `is_causal=False` on the drafter forward (`dflash/model.py`), enabling the trained model's internal layer-type-dependent masking.
- **Us:** **layer-type-dependent**, matching vLLM. For Qwen3.6-27B-DFlash specifically: layers 0–3 causal-SWA window=2048, layer 4 bidirectional. Read per-layer from drafter GGUF metadata (`dflash.layer_types`). The block-diffusion bidirectional mixing happens at layer 4 — that's where every mask position attends to every other mask position in the block.

### Sampling at draft step
- **vLLM:** sampler config.
- **SGLang:** `topk=1` always.
- **Upstream:** forces greedy top_k=1 inside DFlash state regardless of CLI.
- **Us:** **greedy only at first landing.** Argmax over draft logits at each position. Probabilistic sampling adds a rejection-sampling layer to AcceptPrefixDecision that's currently scoped out (`ProbabilisticVerifyOutOfScope` invariant in the spec). Add later.

### Projection fan-out
- **vLLM:** one fused `F.linear` emitting `[L * 2 * kv_size, hidden]` for all draft layers at once + fused norm+RoPE Triton kernel.
- **SGLang:** same fused path on supported hardware, per-layer fallback otherwise.
- **Upstream:** the encoder graph IS the projection (one mul_mat + RMSNorm), then per-layer K/V are computed inside the decoder graph via standard wk/wv on the projected feature.
- **Us:** **per-layer scalar dispatch** for the initial landing. The fused path is a perf knob, not a correctness one; ik_llama.cpp's `iqk_mul_mat` doesn't help here (BF16 drafter weights, not low-bit), and writing a Triton equivalent is out of scope. Per-layer cost is bounded — one K_proj + one V_proj + one norm + one RoPE per layer per accepted token. Negligible against drafter forward cost.

### Source-layer indices
- **vLLM:** from drafter checkpoint config (`dflash_config.target_layer_ids`).
- **SGLang:** `build_target_layer_ids(num_target, num_draft)` — evenly spaced in `[1, num_target-3]`, overridable.
- **Upstream:** GGUF metadata key `dflash.target_layer_ids` with `+1` offset from the HF checkpoint config.
- **Us:** **GGUF metadata, read at load time, no CLI override.** The drafter was trained against specific indices; using different indices at inference is wrong. The Allium spec's `FeatureSourceFixedPerDeployment` binds this.

### Block size
- **vLLM:** default 16, configurable.
- **SGLang:** default 16, configurable.
- **Upstream:** GGUF metadata, default 16.
- **Us:** **block_size=16 from the start, matching the trained model.** GGUF metadata is the source of truth; CLI override `--dflash-block-size N` exists for experimentation but defaults to whatever the drafter declared (16 for Qwen3.6-27B-DFlash). Earlier plan was to start at 8 for kernel-determinism safety, but the drafter was trained at 16 — running at 8 means the drafter sees a mask pattern it never saw in training, and acceptance is unpredictable. Better to absorb the verify-shape (`ne[1]=17`) uncertainty as a Gate-5 binding test than to mis-match training. If Gate-5 shows non-determinism at `ne[1]=17`, that becomes a kernel-side problem to solve, not a parameter to dodge.

### Multi-slot
- **vLLM:** supports through standard speculative path.
- **SGLang:** supports.
- **Upstream:** **hard gated to np=1**.
- **Us:** **hard gated to np=1.** The multi-slot determinism bug captured in `project_mtp_multislot_determinism_investigation_failed` memory entry is orthogonal to DFlash. DFlash on np>1 would amplify the same surface (more positions per verify batch). np=1 only.

### Determinism guarantees
- **vLLM:** silent.
- **SGLang:** silent.
- **Upstream:** silent.
- **Us:** **per-deployment determinism is a closure gate.** Under fixed (block_size, target_layer_ids, draft GGUF), runs are byte-identical. Cross-deployment is not bit-stable for the same kernel-batch-shape reason we know from multi-slot. Make the binding test before merging.

### CLI surface
- **Upstream:** `--dflash` flag in `llama-speculative-simple`.
- **Us:** `--dflash` plus `--dflash-block-size N` plus `--dflash-think on|off` (the equivalent of `LLAMA_SPEC_NO_THINK` for Qwen3-family — major accept-rate lever per upstream PR notes). The server picks these up via the existing `params_base.speculative.*` plumbing.

## 6. Gates — order of work and falsifiable stop conditions

These are gates in the §5-of-CLAUDE.md sense: each must close before the next opens. Skipping a gate is forbidden per `feedback_no_skip_tests.md`.

### Gate 0 — M0: reproduce upstream on this hardware

Before any ik_llama.cpp work. The local `/home/llm/llama.cpp` checkout is already on PR-22105's branch (per the upstream research agent's report). Build with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release`. Convert `Qwen/Qwen3-8B` + `z-lab/Qwen3-8B-DFlash` per upstream's recipe. Run upstream's "quicksort, thinking off" benchmark on a single RTX 6000.

- **PASS:** ≥3x speedup over Qwen3-8B baseline. Premise holds — proceed to Gate 1.
- **FAIL:** <3x speedup. The premise that "DFlash gives big wins on hybrid-attention targets on Turing" is broken. Stop. Re-examine before any porting.

Token cost: 10-20k.

### Gate 1 — converter binding

Port `class DFlashModel(Qwen3Model)` into ik_llama.cpp's `convert_hf_to_gguf.py`. Convert `z-lab/Qwen3-8B-DFlash` (target = `Qwen/Qwen3-8B`) into a `.gguf` that has all expected tensors (fc, hidden_norm, drafter layers) and metadata (target_layer_ids, block_size, mask_token_id, swa_window, target_arch, target_n_embd). Verify by `gguf_dump` and tensor-name match against the upstream-converted version.

- **PASS:** Tensors and metadata match.
- **FAIL:** Missing or mismatched. Diagnose; don't proceed.

Token cost: 15-25k.

### Gate 2 — extract hook

Implement `dflash_extract_features` against Qwen3.5 / Qwen3.6 build graphs. Hook the residual-stream snapshot at K source-layer indices. Build a binding test that runs a known prompt through `Qwen3-8B` target and captures hidden states at the source layers; compare against the upstream-converted reference output (which the upstream PR's extract emits on the same prompt).

- **PASS:** Byte-identical hidden-state snapshots at all source-layer indices.
- **FAIL:** Either the hook fires at the wrong tensor or the snapshot mis-orders the layer/position dimensions. Fix; re-run.

Token cost: 15-25k.

### Gate 3 — inject + drafter forward

Implement `apply_inject_kv` + `build_dflash_drafter`. Build a binding test: given fixed target features and a known anchor token, run drafter forward at `block_size=4` and compare draft logits against the upstream impl on identical inputs.

- **PASS:** Byte-identical (or within 1e-5 NMSE) draft logits at block_size=4.
- **FAIL:** Logits differ. Diagnose at the per-op level; the inject path or the SWA mask or the K_proj/V_proj weights are off.

Token cost: 25-40k.

### Gate 4 — full block-emit + accept loop

Plumb `common_speculative_dflash_*` into `examples/speculative-simple`. Run the "quicksort, thinking off" benchmark on Qwen3-8B target + DFlash drafter on a single RTX 6000.

- **PASS:** Within 10% of the Gate-0 baseline measurement (we now have ik_llama.cpp's kernel advantage on the target side, which should help, not hurt).
- **FAIL:** Substantial regression. Drafter is producing low-acceptance blocks, or the accept loop is mis-counting. Diagnose.

Token cost: 30-50k.

### Gate 5 — 27B determinism

Run the determinism fixture (`scripts/test-mtp-multislot-determinism.sh` adapted to DFlash) at np=1 on Qwen3.6-27B + Qwen3.6-27B-DFlash. Compare to non-DFlash np=1 production baseline for byte-equivalence after a fixed-temperature greedy run.

- **PASS:** Within-deployment byte-identical across 3 runs with same prompt + fixed block_size.
- **FAIL:** Variance across runs. Diagnose (likely DFlash KV cache state isn't fully deterministic across cycles — check anchor advance, slot eviction).

Token cost: 10-20k.

### Gate 6 — Qwen3.6-27B speedup

Run the production-prompt set on Qwen3.6-27B + DFlash at np=1, IQ4_KS target.

- **PASS:** ≥1.5x speedup over current production `--draft 3` MTP. Ship to a new profile (`qwen36-27b-x1-dflash.sh`); leave existing MTP profile in place as fallback.
- **FAIL (<1.5x):** DFlash isn't the win we hoped for on this hardware. Stay on MTP. Document the negative result; abandon the workstream.

Token cost: 15-25k.

**Total budget if everything passes:** ~120-200k. Roughly half PR #22105's footprint because we skip the cross-attention plumbing and the EAGLE3 base.

## 7. Risk surface

### R1 — Drafter still under training
The HF card warns: *"This model is still under training, and inference engine support may not be fully available yet due to architectural changes, including causal SWA layers."* The drafter weights may change. Mitigation: prove the pipeline on `Qwen3-8B-DFlash` (the published 6x model) at Gate 4, and treat `Qwen3.6-27B-DFlash` as a Gate 6 measurement against a moving target.

### R2 — Per-layer projection cost
Per-layer scalar K_proj+V_proj+norm+RoPE on every accepted anchor token. At 2B drafter with ~30 layers, that's 60 small mat-vecs + 30 norms + 30 RoPEs per anchor. Each is on the order of 100-200µs on sm_75. Per-cycle injection cost ~5-10ms — small against ~30-50ms target forward, but not free. If Gate 4 shows the perf hit is meaningful, implement the fused path before Gate 6.

### R3 — `--target-model-dir` converter coupling
The drafter GGUF references the target's vocab + tokenizer at convert time. If the user later swaps the target's vocab (e.g., adding tool tokens), the drafter must be re-converted. This is an operational hazard but matches upstream's design — flag in the launch profile.

### R4 — Multi-slot drift
If anyone enables np>1 with DFlash on, target features extracted from a multi-slot batch are entangled across slots (same as the multi-slot determinism bug). Hard gate at np=1 in the server is essential; an env override that bypasses it would silently corrupt outputs. Don't add an override.

### R5 — Kernel batch-shape sensitivity at verify
Verify batch is `ne[1] = block_size + 1`. At block_size=8 we know `ne[1]=9` shape is not in the PHASE5 unit-tested set. The kernels test as deterministic at `ne[1] ∈ {2, 4, 8}` and route to `mma_new` at those shapes; `ne[1]=9` may route differently. Add Gate-5's binding test at the exact verify shape we'll ship before Gate 6 runs.

### R6 — Drafter SWA on Turing
We have not stress-tested causal-SWA self-attention on sm_75 at the drafter's layer count. Gate 3 binds the drafter forward correctness; Gate 5 binds determinism. If a kernel dispatch falls back to a non-deterministic path under SWA, the gates catch it.

## 8. What success looks like at end of Gate 6

- New profile `/home/llm/profiles/qwen36-27b-x1-dflash.sh` (alongside existing `qwen36-27b-x1-mtp.sh`)
- Drafter GGUF on `/mnt/archive` or `/opt/models/recast-out/`
- Production server can swap between MTP and DFlash by symlink flip on `/home/llm/profiles/active.sh`
- Branch `production/2026-q2-next` carries the implementation; merge to `production/2026-q3` (next release branch) once verified
- Measured speedup ≥1.5x over MTP at np=1 on production prompts
- Per-deployment byte determinism preserved
- Binding tests live under `tests/` in ik_llama.cpp + harness scripts in yarn-agentic's `scripts/`

## 9. What success doesn't try to achieve

- **np>1 concurrency.** Same wall as multi-slot MTP. Separate workstream.
- **Multimodal image-text-to-text.** Separate workstream (`project_qwen36_27b_multimodal_exploration` memory entry).
- **Fused projection / inject kernels.** Perf optimization. Defer until Gate 4 measures whether per-layer scalar path is acceptable.
- **Replacing MTP entirely.** MTP stays as the fallback profile; DFlash adds an alternative. The user can flip between them.
- **Tree drafting / multi-branch DFlash.** Block-emit only. Tree drafting on hybrid recurrent attention is unsolved (`project_tree_fanout_hybrid_recurrent_blocker` memory entry).

## 10. Anchor on the existing memory

The four most load-bearing existing memory entries for this workstream:

- `project_mtp_multislot_determinism_investigation_failed.md` — Why np>1 stays gated. The same surface bites DFlash if anyone touches it.
- `project_production_2026q2_landing.md` — Current production tip. The new branch forks from this; the running service is unaffected.
- `feedback_anchor_to_measured_baselines.md` — Don't compute uplift % against estimates. Gate 6 must compare against a freshly-measured MTP baseline on the same hardware.
- `feedback_probe_before_implementing.md` — Gate 0 (M0) is the probe. Skipping it would repeat the PHASE4 mistake of building 100k+ tokens of scaffolding on an untested premise.

## 11. Open implementation choices that the spec leaves to this design

Resolved from `dflash_speculative.allium`'s open questions, where this doc commits:

- **OQ-SOURCE-LAYERS:** GGUF metadata, no CLI override. (§5 "Source-layer indices")
- **OQ-DENOISE-SCHEDULE:** single_step only at first landing. (§5 "Sampling at draft step")
- **OQ-BLOCK-SIZE-FOR-TURING:** start at 8, move to 16 after Gate-5 binds determinism at the larger verify shape. (§5 "Block size")
- **OQ-QUANT-MIX:** Target IQ4_KS, drafter BF16. Shared embed/lm_head materialized from the target's IQ4_KS tensor — drafter never re-quantizes. (Allium `SharedEmbedAndLMHead` invariant.)
- **OQ-SWA-WINDOW:** read from drafter GGUF metadata `dflash.swa_window`. Surfaced via `llama_get_dflash_swa_window`.
- **OQ-THINK-MODE-EQUIVALENT:** new CLI flag `--dflash-think on|off` controls the Qwen3 thinking sentinel in the prompt. Default `off` per upstream's accept-rate measurements.
- **OQ-MULTI-SLOT:** np=1 only. Hard gate.
- **OQ-VLLM-PR-MERGE-RISK:** we are not anchoring on upstream's PR for the implementation; we anchor on the published paper + the design here. Upstream PR is reference only.
- **OQ-FAILURE-MODE-ON-MISSING-DRAFT:** at first landing, hard fail (matches MTP-IR's current pattern). Move to graceful-fallback after the path is shipped and measured.
