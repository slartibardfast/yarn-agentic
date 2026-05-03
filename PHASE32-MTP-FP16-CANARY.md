# MTP FP16-Recasting Canary Study (sm_75)

**One-line:** Test whether `mtp.fc.weight` cast to FP16 preserves draft
acceptance on Qwen3-Next-family models, then ship 1-3 SoTA quants per target
under `slartibardfast0/...`.

## Research Hypotheses

| ID | Statement | Falsification gate |
|----|-----------|--------------------:|
| **H1** (primary, **canary**) | Casting `mtp.fc.weight` from BF16 → FP16 preserves draft acceptance ≥ 0.7 on Qwen3.5-0.8B (MTP head fusion projection has bounded inputs; 10-bit FP16 mantissa has ~64× the resolution of INT4 codes that gave the published 0% data) | V-F1a accept < 0.5 |
| H2 | If H1, the entire MTP block (mtp.fc + mtp.layers.0.* + mtp.{e,h,shared_head}_norm) can be FP16-cast | V-F1b accept ≥ 0.7 |
| H3 | GatedDeltaNet `linear_attn.in_proj_*` / `out_proj` (stateless matmuls inside SSM block) can be FP16-cast | V-F1c accept ≥ 0.7 AND wikitext-2 PPL within +1% of V-F1 |
| H4 | `linear_attn.conv1d` + SSM scalars (`A_log`, `dt_bias`) can be FP16-cast (recurrence compute is FP32 per `mamba_ssm_dtype` regardless of weight storage) | V-F1d accept ≥ 0.7 AND PPL +1% |
| H5 (tertiary) | The Pareto-optimal recipe across {KLD vs BF16, draft accept, gen-tg ratio, file size} for sm_75 production exists in {V-F1, V-F1a, V-S1, V-S1a, V-Q1, V-Q1a} | empirical Pareto plot |

**Why H1 is the canary:** all published "0% draft accept" data is for **INT4** mtp.fc
([Lorbus / sakamakismile / AEON-7](#sources) recipes). FP16 has not been tested
upstream. mtp.fc is the most precision-sensitive tensor by published evidence; if
it survives FP16, the broader BF16-preservation list almost certainly does too.

## Targets

| Target | Quant source | MTP source |
|--------|---------------|-------------|
| **Qwen3.5-0.8B** (canary bed; iter-7 baseline 1.282× / 0.848 accept) | `/opt/models/qwen3.5-0.8b/Qwen3.5-0.8B-BF16.gguf` (intact) | already in source |
| **Qwen3.6-35B-A3B** (MoE; existing BF16 GGUF has MTP) | `/opt/models/Qwen3.6-35B-A3B-bf16.gguf` (intact) | already in source |
| **Qwen3.6-27B** (dense; existing AutoRound has no MTP) | `/opt/models/hf-cache/.../Intel--Qwen3.6-27B-int4-AutoRound/` | partial `hf download Qwen/Qwen3.6-27B --include "*mtp*"` (~280 MiB) |

## Tools (5 to write)

| # | Path | Purpose |
|---|------|---------|
| 1 | `scripts/autoround_to_q4_0_gguf.py` (saved) | AutoRound INT4 sym W4G128 → Q4_0 GGUF (lossless repack) |
| 2 | `scripts/splice_mtp_tensors.py` (NEW) | Append BF16 MTP tensors from source GGUF/safetensors into a target GGUF; update `nextn_predict_layers` |
| 3 | `scripts/recast_bf16_to_fp16.py` (NEW) | Per-tensor selective cast: clamp ±65504 + numpy RNE; takes a YAML/JSON precision policy |
| 4 | `scripts/mixed_quant_synthesis.py` (NEW) | Compose Q4_0 trunk (Tool 1) + selective-uplifted tensors (Q5_K/Q6_K via llama-quantize / FP16 via Tool 3) + BF16-preserved tensors → single mixed-precision GGUF |
| 5 | `scripts/kld_compare.sh` (NEW) | Wrap `llama-perplexity --kl-divergence-base/--kl-divergence` against wikitext-2 |
| 6 | `scripts/validate_gguf_mtp.sh` (NEW) | Smoke: load test, `nextn_predict_layers=1` check, accept ≥ 0.5, coherent output, deterministic |

## Variant matrix (Stage A — 0.8B canary + sweep)

12 variants. **Bold** = key research-question variants.

| Variant | trunk | mtp.fc | rest of MTP | linear_attn.in_proj_* | linear_attn.conv1d/A_log/dt_bias | norms | size |
|---------|-------|--------|-------------|------------------------|----------------------------------|-------|-----:|
| **V0** | BF16 | BF16 | BF16 | BF16 | BF16 | BF16 | 1.5 GiB |
| **V-F1** | FP16 | BF16 | BF16 | BF16 | BF16 | BF16 | ~780 MiB |
| **V-F1a** | FP16 | **FP16** | BF16 | BF16 | BF16 | BF16 | ~780 MiB |
| V-F1b | FP16 | FP16 | **FP16** | BF16 | BF16 | BF16 | ~780 MiB |
| V-F1c | FP16 | FP16 | FP16 | **FP16** | BF16 | BF16 | ~770 MiB |
| V-F1d | FP16 | FP16 | FP16 | FP16 | **FP16** | BF16 | ~770 MiB |
| V-F1e | FP16 | FP16 | FP16 | FP16 | FP16 | **FP16** | ~770 MiB |
| **V-S1** | Q4_0 (AutoRound) | BF16 | BF16 | BF16 | BF16 | BF16 | ~540 MiB |
| **V-S1a** | Q4_0 | **FP16** | BF16 | BF16 | BF16 | BF16 | ~540 MiB |
| V-S1b | Q4_0 trunk + FP16 embed/lm_head + Q6_K attn_v + Q5_K attn_k | BF16 | BF16 | BF16 | BF16 | BF16 | ~550 MiB |
| **V-Q1** | Q4_0 + selective Q5_K/Q6_K + FP16 embed/lm_head | BF16 | BF16 | BF16 | BF16 | BF16 | ~470 MiB |
| **V-Q1a** | (V-Q1 trunk) | **FP16** | BF16 | BF16 | BF16 | BF16 | ~470 MiB |

## Per-variant measurements

| Metric | Tool | Pass gate |
|--------|------|----------:|
| Smoke (loads, coherent, deterministic, `nextn_predict_layers=1`) | Tool 6 | binary pass/fail |
| Draft acceptance | server stats | **≥ 0.7** for "viable", **≥ 0.5** for "graduated" |
| nomtp_tg / mtp_tg / ratio | `bench-mtp-0.8b.sh` (5 runs each) | ratio ≥ 1.0× |
| KLD vs V0 BF16 (mean, p99, max) | Tool 5 (wikitext-2) | mean < 0.05 |
| File size on disk | `du -sh` | informational |

## Phase A.1 — Canary execution (~45 min wall)

| Step | Action | Gate |
|------|--------|------|
| A.1.1 | `hf download Intel/Qwen3.5-0.8B-int4-AutoRound` | exists |
| A.1.2 | Write Tools 2-6 (splice, recast, synthesis, kld_compare, validate). Verify each loads cleanly via `python -c "import …"` / `bash -n …` | syntax clean |
| A.1.3 | Build V0 reference KLD logits dump: `llama-perplexity --kl-divergence-base /tmp/v0.kld -m V0.gguf -f wikitext-2-raw/wiki.test.raw` | file produced |
| A.1.4 | Build V-F1 (Tool 3 with policy: trunk FP16, MTP+GDN+norms BF16). Tool 6 smoke. | smoke pass |
| A.1.5 | Bench V-F1 (`bench-mtp-0.8b.sh`). KLD V-F1 vs V0 (`kld_compare.sh`). | accept ≥ 0.65, ratio ≥ 1.0×, KLD < 0.05 |
| A.1.6 | Build V-F1a (V-F1 + FP16 mtp.fc.weight). Tool 6 smoke. | smoke pass |
| A.1.7 | **CANARY:** Bench V-F1a + KLD vs V0. Compare to V-F1 numbers. | see decision rule |

## Phase A.1 decision rule

| V-F1a accept | Verdict | Phase A.2 path | Stage B implication |
|--------------|---------|----------------|----------------------|
| **≥ 0.7** AND KLD ≈ V-F1 | **GREEN** — H1 confirmed | Full sweep (build V-F1b/c/d/e + V-S1/a/b + V-Q1/a, ~2 hr) | Drop BF16 mtp.fc preservation; ship FP16 mtp.fc variants |
| 0.5 ≤ accept < 0.7 | **YELLOW** — H1 partial | Reduced sweep (V-S1, V-S1a, V-Q1, V-Q1a; ~1 hr) | Ship both BF16 and FP16 mtp.fc options |
| < 0.5 | **RED** — H1 falsified | Reduced sweep (BF16-mtp.fc-only: V-S1, V-Q1, V-S1b; ~45 min) | Validate published recipe; preserve BF16 mtp.fc across all |

## Phase A.2 — Full sweep (executed per A.1 verdict)

For each variant in the chosen path:

| Step | Action | Gate |
|------|--------|------|
| Build | Tool 4 with the recipe spec | builds cleanly |
| Smoke | Tool 6 | passes |
| Bench | `bench-mtp-0.8b.sh` 5 runs each mode | ratio + accept captured |
| KLD | Tool 5 vs V0 | KLD captured |
| Append row to `/tmp/iter8-stageA-0.8b-results.md` | — | — |

## Phase A.3 — Recipe selection

User reviews `/tmp/iter8-stageA-0.8b-results.md` (Pareto frontier table + plot)
and picks **1-3 recipes** from the frontier to apply at scale. Anticipated
shortlist:
- F-family pure-FP16 (highest quality, biggest)
- S-family synthesis (Pareto-optimal balance)
- Q-family Q-trunk (smallest)

Each chosen recipe fans out to {35B-A3B, 27B} in Stage B.

## Stage B — Apply top recipes to 35B-A3B + 27B

Disk-aware ordering (start ~82 GiB free, before Stage A leaves ~80 GiB).

| Step | Action | Output | Free after |
|------|--------|-------:|-----------:|
| B.1 | Stage A picks recipe → produce 35B-A3B variant 1 (using Tool 4 from BF16 source) | ~22-36 GiB | 44-60 GiB |
| B.2 | Stage A picks recipe → produce 35B-A3B variant 2 (if 2+ recipes chosen) | ~22-36 GiB | tight; may delete BF16 source first |
| B.3 | Delete `/opt/models/Qwen3.6-35B-A3B-bf16.gguf` (only after all 35B variants done; can re-download from HF for future re-quants) | -67 GiB | +55-80 |
| B.4 | `hf download Qwen/Qwen3.6-27B --include "*mtp*"` (partial; ~280 MiB MTP-only) | ~280 MiB | ~same |
| B.5 | Tool 1 (AutoRound→Q4_0) + Tool 2 (splice MTP) + Tool 3/4 (synthesis) per recipe → 27B variants | ~10-28 GiB each | varies |
| B.6 | Tool 5 (KLD) + Tool 6 (smoke) + bench per variant. KLD on 27B requires reference dump from BF16 source which we don't have on disk; **defer KLD** to use the 0.8B finding as the precision proxy | logs | — |
| B.7 | Append all to `/tmp/iter8-stageB-results.md` | — | — |

## Stage B gates (per produced GGUF)

| Gate | Pass |
|------|------|
| Tool 6 smoke | binary pass |
| draft accept | ≥ 0.65 |
| ratio | ≥ 1.0× (35B-A3B); ≥ 1.0× (27B) |
| KLD vs BF16 reference | mean < 0.05 (35B-A3B only; 27B deferred) |

## Stage C — HF release prep (slartibardfast0/)

For each shipped variant:

| Item | Detail |
|------|--------|
| Repo name | `slartibardfast0/Qwen3.6-{27B,35B-A3B}-<scheme>-MTP-sm75` (e.g., `Qwen3.6-35B-A3B-FP16-MTP-sm75`, `Qwen3.6-27B-Q4_0-Q6K-FP16-MTP-sm75`) |
| Model card | per-tensor precision policy table; reproducibility script (exact tool invocations); sm_75 caveat; KLD + accept + tg numbers; **headline FP16 mtp.fc finding** if H1 confirmed |
| `build.sh` | the exact recipe-build script copied alongside the GGUF |
| imatrix | preserve in GGUF when used; document in card |
| License | inherit Apache-2.0 from upstream |

## Stage D — iter-8 lever-suite (deferred)

Once production GGUFs exist, run iter-8 lever-branch matrix on them. The
`iter8/probes` branch (commit `edc1f6a3`) has all instrumentation. Not
on this study's critical path.

## Out of scope

- Vulkan/HIP backends — sm_75 CUDA only
- FP8 — sm_75 has no native FP8
- Vision tower bench — text-only inference (vision tensors held BF16 for completeness)
- σ-anomaly repair on late GDN conv1d blocks — informational profile only ([AEON-7 finding](#sources))
- Training-aware re-quantization — post-training only

## <a name="sources"></a>Source citations (web-verified 2026-05-03)

| Claim | Source |
|-------|--------|
| Official Qwen3.6 releases | [HF Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B), [HF Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |
| `mtp.fc.weight` is canonical Qwen3-Next MTP fusion key; INT4 → 0% accept | [Medium "Overnight Stack"](https://medium.com/@fzbcwvv/an-overnight-stack-for-qwen3-6-27b-85-tps-125k-context-vision-on-one-rtx-3090-0d95c6291914) |
| BF16-preservation list (community-validated) | [AEON-7](https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-DFlash), [sakamakismile NVFP4-MTP](https://huggingface.co/sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP), [kaitchup Qwen3.5 quant](https://kaitchup.substack.com/p/qwen35-quantization-similar-accuracy) |
| `linear_attn.conv1d` recurrence-critical | AEON-7, sakamakismile recipes |
| Intel AutoRound INT4 reference | [Intel/Qwen3.5-0.8B-int4-AutoRound](https://huggingface.co/Intel/Qwen3.5-0.8B-int4-AutoRound) |
| QK-Norm protects FP16 attention overflow | [LangCopilot QK-Norm](https://langcopilot.com/posts/2025-06-26-qwen3-qk-norm-improved-on-device-ai-stability) |
| `mamba_ssm_dtype=float32` runtime FP32 | [state-spaces/mamba README](https://github.com/state-spaces/mamba) |
| `attn_output_gate=true` (NeurIPS 2025 Best Paper) | [Qwen3-Next blog](https://qwen3-next.com/), [vLLM Qwen3-Next](https://blog.vllm.ai/2025/09/11/qwen3-next.html) |
| FP16 max = 65,504 — clamp required | [Towards AI quantization explainer](https://pub.towardsai.net/understanding-llm-quantization-why-fp32-fp16-bf16-and-int8-matter-for-modern-ai-systems-076ea6eb9ca6), [TensorRT docs](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html) |
| sm_75 has FP16 tensor cores, no native BF16 | [vLLM Turing issue](https://github.com/vllm-project/vllm/issues/29743) |
| BF16→FP16 gains mantissa (7→10 bits), loses exponent (8→5) | TensorRT, multiple LLM quantization explainers |
