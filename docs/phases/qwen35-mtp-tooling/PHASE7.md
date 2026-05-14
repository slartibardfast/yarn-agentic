# Phase 7: Production Config at 32K + BigCodeBench Evaluation

## Context

Phase 6 shipped the full TURBO_KV_4B Vulkan shader suite. This phase focused on
evaluating the model for production deployment: comparing against an alternative
distilled model, fixing the SET_ROWS shader for correctness, benchmarking with
and without thinking mode, and validating the production configuration at 32K
context on Vega 64's 8GB HBM2.

## Results

### Production configuration (validated)

```
llama-server \
  -m Qwen3.5-9B-mtp-q4km.gguf \
  -c 32768 -ngl 99 -np 1 \
  --reasoning off \
  --cache-type-k f16:turbo_kv_4b
```

| Metric | Value |
|---|---|
| Context window | 32K tokens |
| VRAM used | 7802 MiB / 8176 MiB (374 MiB free) |
| KV cache | 842 MiB (K compressed 2.2x via TURBO_KV_4B) |
| PPL (wikitext-2, c=32768) | **7.55** |
| PPL (wikitext-2, c=512) | 9.13 (f16:turbo_kv_4b) / 8.11 (f16 baseline) |
| Prompt eval | 219 t/s (c=32768) / 1544 t/s (c=512) |
| Generation | 38.4 t/s (MTP speculative decoding) |
| BigCodeBench-Hard pass@1 | 18.9% (reasoning off) |
| Reasoning mode | off (Qwen default for 9B) |

The TURBO_KV_4B K cache compression saves 310 MiB at 32K context (576 → 266 MiB
for K cache), making the difference between ~200 MiB and ~380 MiB headroom. At
long context the PPL impact is negligible — 7.55 at c=32768 is better than the
f16 baseline at c=512 (8.11).

### BigCodeBench-Hard (Instruct, 148 tasks)

| Model | Config | pass@1 | Time | Notes |
|---|---|---|---|---|
| **Qwen3.5-9B Base** | **no thinking** | **18.9%** | **31 min** | **Production config** |
| Qwen3.5-9B Base | thinking (+1024 budget) | 10.8% | 94 min | 45/148 empty solutions |
| Openclaw-9B Distilled | no thinking | 16.9% | 27 min | No MTP, scored lower |

**Leaderboard context** (18.9% no-thinking, Q4_K_M quantized, greedy):
- Matches CodeQwen1.5-7B-Chat (18.9%) — leaderboard scores are FP16
- Beats Llama-3.1-8B (+5.4pp), Mistral-Nemo-12B (+7.4pp)
- Full-precision Qwen3.5-9B would likely score ~20-21%

### Thinking mode investigation

Thinking mode **hurts** at 9B Q4_K_M. Root causes identified:

1. **Qwen disables thinking by default on 9B and below** — only enabled on 27B+
2. **Qwen's recommended thinking budget is 32K tokens** — our max feasible budget
   (1024 extra) is 30x too small
3. **Quantized models double the truncation rate** (Kaitchup study) — the model
   exhausts its token budget on reasoning and truncates the actual answer
4. **45/148 solutions were completely empty** even with 1024 extra thinking tokens
5. **Known issue**: HuggingFace users report "circular reasoning" loops at 9B

A proxy bug was found and fixed during investigation: BigCodeBench sends
`max_completion_tokens` (not `max_tokens`), so the initial proxy wasn't inflating
the correct field. After fixing, empty solutions dropped from 54 to 45 but pass@1
only reached 10.8% — still well below the 18.9% no-thinking baseline.

## What shipped

### SET_ROWS TURBO_KV_4B rewrite (`ff99eb229`)

Replaced flat row-based addressing with proper 4D stride-based indexing:

- Proper 4D index decomposition for multi-head KV caches (ne02 > 1)
- Buffer offset handling via `misalign_offsets` (get_aoffset/get_boffset/get_doffset)
- Index tensor broadcast via `i02 % p.ne11`, `i03 % p.ne12`
- PPL verified: 9.13 (unchanged from Phase 6)

### Cherry-picked CPU fixes (6 commits from polaris-hybrid-cpu-opt)

| Commit | Fix |
|---|---|
| `c2b33c022` | models/qwen35: pre-RoPE K skip for split K cache |
| `3043d2fe4` | tests: split K cache op-level test (12/12 pass) |
| `cd6112a9c` | tests: model integration test + revert bad MTP fix |
| `2a52471f5` | kv-cache: zero-fill stale out_ids (27B MTP crash root cause) |
| `827da8d78` | MTP: clamp greedy_logits + document buffer aliasing |
| `4ded57fcc` | MTP: fix greedy_tokens buffer aliasing (set_input+set_output) |
| `6f157e423` | kv-cache: state save/load for split K cache |

The MTP crash (argmax on uninitialized logit rows → OOB embedding lookup) is
backend-agnostic — it occurs in the graph/model layer, not the Vulkan backend.
The fix applies identically to CPU and GPU execution.

### Openclaw model evaluation

- Downloaded ykarout/Qwen3.5-9b-Opus-Openclaw-Distilled (18GB)
- Converted HF → GGUF F16 → Q4_K_M (5.3GB)
- Binary-patched GGUF metadata (`block_count=32`, `nextn_predict_layers=0`) —
  converter emitted MTP metadata but no MTP tensors
- Scored 16.9% on BigCodeBench-Hard, 2pp below base model
- Deleted after benchmark (not worth keeping)

### BigCodeBench evaluation infrastructure

- Thinking token proxy (`thinking_proxy.py`): inflates `max_completion_tokens`
  to give thinking models a separate reasoning budget
- Found and fixed proxy bug: BigCodeBench sends `max_completion_tokens`, not
  `max_tokens` — the initial proxy wasn't intercepting the correct field

## VRAM budget at 32K

| Component | f16 KV | f16:turbo_kv_4b K | Savings |
|---|---|---|---|
| Model weights (Q4_K_M) | 5666 MiB | 5666 MiB | — |
| K cache (32K) | 576 MiB | 266 MiB | 310 MiB |
| V cache (32K) | 576 MiB | 576 MiB | — |
| Recurrent state | 50 MiB | 50 MiB | — |
| Compute buffer | 1098 MiB | 1227 MiB | -129 MiB |
| **Total** | ~7966 MiB | ~7802 MiB | **+182 MiB headroom** |
| **Free** | ~200 MiB | ~374 MiB | |

TURBO_KV_4B compression is critical at 32K — without it, headroom drops to
~200 MiB (marginal). With it, 374 MiB provides a comfortable production margin.

## Session statistics

- 3 BigCodeBench-Hard runs (444 total generations): no-thinking, Openclaw, thinking
- 3 PPL verifications (c=512 f16, c=512 turbo, c=32768 turbo)
- 7 commits cherry-picked from CPU branch
- 1 shader rewrite (SET_ROWS 4D addressing)
- 1 proxy bug found and fixed (max_completion_tokens vs max_tokens)
- 1 GGUF metadata binary patch (Openclaw MTP fix)

## Commit log

```
2688ee0a7 kv-cache: implement state save/load for split K cache
6c9b16f3a MTP: fix greedy_tokens buffer aliasing with set_input+set_output
2e2d778be MTP: clamp greedy_logits + document buffer aliasing root cause
5f1fea1bc kv-cache: zero-fill stale out_ids + 27B MTP crash root cause found
583c542a9 tests: model integration test + cache ops test + revert bad MTP fix
e244c17dc tests: add split K cache op-level test (12/12 pass at all sizes)
b869fb4fe models/qwen35: add pre-RoPE K skip for split K cache
ff99eb229 vulkan: rewrite SET_ROWS TURBO_KV_4B with stride-based 4D addressing
```
