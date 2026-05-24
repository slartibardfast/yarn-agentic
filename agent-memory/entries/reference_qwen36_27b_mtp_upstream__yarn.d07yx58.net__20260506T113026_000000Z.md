---
name: Qwen 3.6 27B MTP upstream PR + froggeric GGUFs
description: Primary source — upstream llama.cpp PR #22673 adds MTP for Qwen 3.6 27B; froggeric/Qwen3.6-27B-MTP-GGUF ships pre-converted quants with fixed chat template; recommended 48 GB GPU configs
type: reference
originSessionId: 9730b98b-a48a-46ed-a147-f48c8cb9810f
---
Source: community post (2026-05-06) on Qwen 3.6 27B MTP support landing in upstream llama.cpp.

## Key facts

- **Upstream PR:** ggml-org/llama.cpp#22673 — adds MTP (Multi-Token Prediction) for Qwen 3.6 27B. Uses model's built-in tensor layers for speculative decoding. Author claims 2.5× speedup, 28 tok/s on M2 Max 96GB.
- **GGUFs:** huggingface.co/froggeric/Qwen3.6-27B-MTP-GGUF — pre-converted with the PR's converter (existing GGUFs lack the MTP tensors).
- **Chat template fixes:** huggingface.co/froggeric/Qwen-Fixed-Chat-Templates — 7 fixes vs. the vLLM-specific original jinja.
- **Build:** `git fetch origin pull/22673/head:mtp-pr && git checkout mtp-pr` against upstream llama.cpp (NOT ik_llama.cpp).
- **Server flags:** `--spec-type mtp --spec-draft-n-max 5 --cache-type-k turbo4 --cache-type-v turbo4 -c 262144`.

## 48 GB GPU recommendations (matches our 2× Quadro RTX 6000)

| Quant | KV | Max ctx | Memory | Vision |
|---|---|---|---|---|
| Q8_0 | q8_0 | 128K | 39.8 GB | ✓ |
| Q6_K | turbo4 | 262K | 35.8 GB | ✓ |

`turbo4` = 4.25-bit upstream KV quant (distinct from our ik_llama.cpp `q4_0 + RHT` aka TURBO_KV_4B).

## Gap vs. our current production

- We run **ik_llama.cpp fork** with a custom recast (`qwen3.6-27b-V-F1.T1.qq-tool1lossless.gguf`), q4_0+RHT KV (Hadamard pre-quant), MTP off.
- PR #22673 is against **upstream llama.cpp**. Porting MTP-for-27B into ik_llama.cpp is non-trivial (existing ik_llama MTP work targeted 35B-A3B MoE, see `project_ik_llama_mtp_35b_moe.md`).
- `turbo4` and our `TURBO_KV_4B` are different quant schemes despite similar branding.

## Why this matters

- 2.5× speedup on 27B is exactly the kind of win the iter-7 lever investigation (`project_mtp_iter7_post_mortem.md`) concluded was unreachable for our 0.8B/35B-A3B targets. Different model class — built-in MTP head, not external draft.
- Q6_K + turbo4 → 262K on 48 GB is a credible alternative production config if we accept moving back to upstream, accepting loss of our RHT KV work and the ik_llama.cpp MTP-IR scaffolding.

## How to apply

When discussing 27B MTP options, distinguish: upstream PR-22673 path (proven 2.5×, requires fork-flip) vs. porting into ik_llama.cpp (preserves stack, unknown cost). Don't conflate `turbo4` with `TURBO_KV_4B`.

## Chat template aspect — landed 2026-05-06

The fixed chat template (`froggeric/Qwen-Fixed-Chat-Templates/qwen3.6/chat_template.jinja`) is in production via `--chat-template-file /home/llm/profiles/qwen36-fixed-template.jinja` in `qwen36-27b-x1.sh`. All seven fixes apply except #1 (cosmetic on our jinja — `|items` is registered).

Required ik_llama.cpp change to render the fixed template: add Python-semantics `find` and `rfind` string methods to `common/jinja/value.cpp`. Submodule commit `06b3b88a` on `phase33-concat-probe`. Parent pin bumped on `phase32-q4_0_ar16-integration`. All 318 tests / 1439 assertions pass (incl. fuzzing). 4/4 live smoke probes pass.

The MTP-via-built-in-head + turbo4 KV path is *not* taken — it's upstream-only and would cost the RHT KV / Phase B / MTP-IR work. See yarn-agentic/MEMORY.md 2026-05-06 entry for full record.
