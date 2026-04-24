# Phase 26: TURBO_KV_4B — SOTA Gap Audit (Allium Weed Pass)

## Status

**Lifecycle artefact.** This phase captures the output of `/allium:weed` on the TURBO_KV family of specifications against the llama.cpp port. It is intentionally a read-only deliverable: it names the divergences between what the specs describe (paper-aligned SOTA TurboQuant with community improvements) and what the port implements (quant.cpp Variant F — paper-minus-QJL, per-128-block rotation, max_abs rescaling, no community improvements layered on top). No code is changed here.

The purpose is to set up the follow-on implementation phases with a clear, prioritised gap list tied directly to measurement evidence.

## Scope

Ten `.allium` files under `yarn-agentic/`:

- `turbo-kv-4b.allium` — core 4-bit algorithm
- `turbo-kv-3b.allium` — core 3-bit algorithm
- `turbo_kv_residual_window.allium` — fp16 rolling window
- `turbo_kv_asymmetric.allium` — separate K/V bit-widths
- `turbo_kv_layer_adaptive.allium` — per-layer bit-width protection
- `turbo_kv_outliers.allium` — outlier channel handling
- `turbo_kv_4b_backend.allium` — cross-backend equivalence
- `turbo_kv_4b_attention.allium` — flash-attention equivalence
- `nearest_centroid.allium` — argmin primitive
- `mul_mat_cpu.allium` — CPU matmul contract

Compared against `llama.cpp/ggml/src/ggml-turbo-kv.c` plus the CPU dispatch (`ggml-cpu/ops.cpp`) and the Vulkan backend under `ggml-vulkan/`.

Four divergences already named as `open question` declarations in `turbo-kv-4b.allium` are excluded from the findings list — they were known going in and the weed pass focused on surfacing additional divergences:

- QJL drop (paper §3 prescribes two-stage MSE+QJL; port drops QJL per [QUANT.CPP] + [TONBI-V3] Karpathy-loop ablation)
- Rotation granularity (paper uses d×d per-head; port inherits Variant F's fixed 128-element sub-blocks)
- Codebook parameterisation (paper centroids in [-1, 1]; port uses Lloyd-Max Gaussian with max_abs rescaling)
- Asymmetric K/V bit allocation (paper §4 + [TONBI-V3] measure K6/V4 or K4/V2; port is symmetric K=V=4-bit)

## Findings — prioritised by PPL impact

The 9B Qwen3.5 IQ3_XXS PPL data (`reference/ppl/results/Qwen3.5-9B-UD-IQ3_XXS`) shows `turbo_kv_4b` +0.23 PPL above F16 baseline. `turbo_kv_4b_attention.allium`'s propagated CPU-vs-Vulkan flash-attention PBT at head_dim=256 passes within fp32 tolerance, so the gap is algorithmic rather than a GPU FA bug. The prioritisation below follows each finding's citable evidence for quality impact.

### Tier 1 — direct drivers of the 9B PPL regression

**T1.1. Residual window unimplemented.** `turbo_kv_residual_window.allium` is a fully-formed spec; `grep residual_window` across the port returns zero hits. The spec cites [TONBI-V3]'s measured needle-in-haystack result: "K4/V4 rw=128 EXACT at 2K and 4K context; K4/V4 rw=0 MISS (garbage output)". Port exactly matches the `rw=0` failing configuration. Classification (c) aspirational — the cheapest first improvement.

**T1.2. Asymmetric K/V unimplemented.** `turbo_kv_asymmetric.allium` is fully-formed; `grep key_bits\|value_bits` returns zero hits. K and V both use `GGML_TYPE_TURBO_KV_4B`. In the 9B Qwen3.5 run V is still F16 (opposite direction of [TONBI-V3]'s K4/V2 prescription — V was never quantised at 9B). [TONBI-V3] 8K attention-score table reports K4/V2 + protected layers = cosine 0.9997, 99% top-1 agreement, 3.6× compression. Classification (c) aspirational.

### Tier 2 — known deficiencies not yet implemented

**T2.1. Layer-adaptive bit-widths unimplemented.** `turbo_kv_layer_adaptive.allium` describes `protected_layers = 4` / `protected_bits = 8`. `grep protected_layers` returns zero hits. Per [TONBI-V3]'s table this lifts K4/V2 top-1 agreement from 94 % → 99 %. Classification (c) aspirational.

**T2.2. Outlier handling unimplemented.** `turbo_kv_outliers.allium` describes `TURBO_KV_4BO` / `TURBO_KV_3BO` with 8 outlier channels per 128-element block (K largest-magnitude indices stored verbatim as fp16). No `TURBO_KV_4BO` in `ggml.h`. High implementation cost — the spec itself calls it "about the same effort as adding a whole new precision tier". Classification (c) aspirational.

**T2.3. 3-bit KV cache ggml type missing.** `turbo-kv-3b.allium` describes the Variant F 3-bit KV wire format (56 B / 128 elements). No `GGML_TYPE_TURBO_KV_3B` exists in `ggml.h:437-442`. The `GGML_TYPE_TURBO_3B = 48` that does exist is a *weight* quantisation type with a different block layout (fp16 scales, 52 B) and no KV-cache wiring or FA dispatch. Classification (c) aspirational. Likely best delivered together with T1.2 (asymmetric), since K4/V3 is a natural pairing.

### Tier 3 — active implementation-level bugs surfaced by weed

**T3.1. Batched FA disabled during prompt eval.** `turbo_kv_4b_attention.allium`'s `flash_attn_ext` contract implicitly covers `n_tokens ∈ {1, 8, 64}` per its `@guidance`. In `ops.cpp:8329` the batched kernel is gated by `!write_partials` and its own N==1 guard: "Batched turbo_kv_4b: only in single-token generation (N==1). During PP (N>1), the non-FA vec_dot path works correctly at all context lengths. Use that for now." A wdata-layout collision in the multi-token path is the cited reason. Correct but slow at prompt-eval time. Classification (b) code gap. Tracks alongside the unresolved debug printf at `ops.cpp:8440-8454` which fires on the same path.

**T3.2. Stale `@guidance` in attention spec.** `turbo_kv_4b_attention.allium:267-269` says "NO existing coverage compares CPU FA to GPU FA directly. That is what this spec's propagated PBT will add." — but `tests/test-turbo-kv-attention-pbt.cpp` already implements exactly that at head_dim=256. Classification (a) spec bug. Two-line fix: "will add" → "adds"; remove the "NO existing coverage" sentence.

### Tier 4 — documented or intentional divergences (no action)

**T4.1. Port's per-block `inv_std` field.** `turbo_kv_4b_backend.allium:64-67` declares the external `QuantizedHead` without `inv_std`. The same file's scope paragraph (lines 46-52) self-flags the divergence and explicitly invites `/allium:weed` to surface it. Port's `block_turbo_kv_4b` at `ggml-turbo-kv.h:64-68` carries `inv_std` as fp32. Classification (c) aspirational — self-flagged; no change until project direction shifts toward paper-aligned wire format.

**T4.2. Defensive clamps beyond spec preconditions.** `QuantizeHead` / `normalize` / `DequantizeHead` declare positive-norm and positive-`inv_std` preconditions; the port has defensive branches (`ggml-turbo-kv.c:492`, `533`, `590`, `744`) that substitute bounded sentinels on out-of-contract inputs. Comments cite the spec and mark the branches as dead on spec-valid input. Classification (d) intentional gap. Worth one spec-side line acknowledging the defensive clamps exist and are unreachable under the stated preconditions.

**T4.3. `LargeHeadDimParity` is backend-parity only.** `turbo_kv_4b_attention.allium:174-193` demands tight fp32 bounds between CPU FA and GPU FA at head_dim=256. Both backends use the same multi-block split, so satisfying this invariant says nothing about whether multi-block is algorithmically right (covered by the existing rotation-granularity open question). Classification (d) intentional gap — the invariant is correctly scoped. A one-line clarification in the invariant body would prevent misreading.

## Full report

See conversation transcript 2026-04-24 for the structured weed output with per-finding `(spec.file:line) → (code.file:line)` citations. The key table from that output:

| Tier | Spec file | Impact on 9B PPL |
|---|---|---|
| T1.1 | `turbo_kv_residual_window.allium` | **direct — [TONBI-V3] says `rw=0` is garbage** |
| T1.2 | `turbo_kv_asymmetric.allium` | **direct — [TONBI-V3] K4/V2 is 99 % top-1** |
| T2.1 | `turbo_kv_layer_adaptive.allium` | secondary — stacks on T1.1 + T1.2 |
| T2.2 | `turbo_kv_outliers.allium` | secondary — high impl cost |
| T2.3 | `turbo-kv-3b.allium` | N/A — memory budget, not quality |
| T3.1 | `turbo_kv_4b_attention.allium` | PP throughput, not PPL |
| T3.2 | `turbo_kv_4b_attention.allium` | doc hygiene |
| T4.1-3 | various | none — documented |

## Next steps

The Tier 1 findings are the actionable SOTA gap. The Tier 1 implementation programme is best sequenced as:

1. **Residual window (T1.1) first.** Smallest structural change (one fp16 side-buffer, rolling-write at append, two-pass read at attention). Direct measurement evidence it closes a large share of the 9B gap. Propagate PBT from `turbo_kv_residual_window.allium` first, then implement, then re-measure 9B IQ3_XXS.

2. **Asymmetric K/V (T1.2) second.** Requires either a new ggml type (T2.3 is a natural companion — 3-bit KV) or per-tensor bit-width metadata. Higher-effort but citation-supported. Re-measure after landing.

3. **Layer-adaptive (T2.1) third.** Composes on top of (1) and (2) as a metadata-only change per-tensor. Sibling measurement suggests it lifts top-1 from 94 → 99 %.

4. **Outlier handling (T2.2) last — or not at all.** Highest implementation cost, smallest incremental quality lift given (1) through (3) already landed.

5. **T3.1 wdata layout fix.** Unblocks batched FA at PP. Diagnose the wdata offset collision, fix, remove the N==1 guard, remove the debug printf. Orthogonal to T1–T2; can land any time.

6. **T3.2 spec doc fix.** Two-line prose correction. Can land alongside the next `tend` pass.

Each of those steps is its own phase. PHASE26 stops here — a gap audit, not an implementation plan.

## Files touched

- `PHASE26.md` (this file) — new
- `SUMMARY.md` — add PHASE26 navigation entry
