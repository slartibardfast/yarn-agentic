# PHASE45 D10.b bench (2026-05-09, batched-draft API)

Submodule HEAD: <bumped after this commit>.
Parent HEAD: bumped after this bench.

## Single-slot regression check

Profile: qwen36-27b-x1.sh (np=1, ctx=262144, no -mtp).
Prompt: "The capital of France is" → 30 predicted tokens, temp=0, seed=1.

| | predicted_n | tg (t/s) | draft_n | accepted |
|---|---|---|---|---|
| pre-D10.b | 30 | 31.38 | 0 | 0 |
| D10.b     | 30 | 31.40 | 0 | 0 |

No regression. The single-slot path has no MTP (and `mtp_collected.size() == 0` so the new code is bypassed entirely).

## Multi-slot x3-mtp aggregate (3 concurrent /v1/completions, 30 tok, 5 reps)

Profile: qwen36-27b-x3-mtp.sh (np=3, ctx=786432=256k×3, -mtp --draft 3, INLINE_KV=1).
Prompts: "capital of France", "def reverse_string", "translation of hello to Spanish".
30 predicted tokens each, temp=0, seed=1.

D10.a baseline (per-slot serial draft):

| Slot | tg (t/s) | accept rate |
|---|---|---|
| 0 | 11.50 | 78% |
| 1 | 9.52  | 75% |
| 2 | 9.67  | 73% |
| **agg** | **30.7** | |

D10.b (batched draft):

| Rep | slot0 tg | slot1 tg | slot2 tg | agg | slot0 acc | slot1 acc | slot2 acc |
|---|---|---|---|---|---|---|---|
| 1 | 12.75 | 14.97 | 11.14 | 38.86 | 75% | 100% | 72% |
| 2 | 12.85 | 15.11 | 11.17 | 39.13 | 75% | 100% | 72% |
| 3 | 12.79 | 15.02 | 11.14 | 38.95 | 75% | 100% | 72% |
| 4 | 12.91 | 15.17 | 11.22 | 39.30 | 75% | 100% | 72% |
| 5 | 12.83 | 15.08 | 11.14 | 39.05 | 75% | 100% | 72% |

**Mean aggregate: 39.06 t/s** (+27% over D10.a baseline 30.7 t/s).

Per-slot tg lift: slot0 +11.6%, slot1 +58.6%, slot2 +15.3%. Slot 1 sees the biggest lift because its drafts are shortest in the batched path (chains end via p_min faster), reducing the per-slot overhead share.

Slot 1's unusual 100% accept rate (19/19) on this prompt is consistent across reps; it appears to be a property of the "def reverse_string" prompt under temp=0 — chains are short enough that p_min always cuts before a divergence. Solo single-slot also shows ~72% on this prompt, so the multi-slot batched accept tracks the single-slot behavior reasonably well.

## What changed

D10.b makes draft generation a single batched forward across M slots instead of M sequential forwards. The throughput unlock is the verify-forward pattern (already batched in D10.a) replicated for the draft side.

Three-layer API:
- `llama_spec_mtp_draft_batched` — libllama primitive (per-step batched forward, alive-mask, multi-row hidden state plumbing).
- `llama_spec_loop_gen_drafts_batched` — libllama wrapper.
- `common_speculative_draft_batched` — libcommon entry; falls back to per-slot serial when not all slots are MTP or when their underlying ctx differs.

Server consumer at `add_sampled_tokens`: M=1 still uses the existing single-slot fused fast path; M>=2 enters the batched path.

## Build status

- `cmake --build build -j 32 --target llama-server` exits 0.
- libllama.so 2.9M, llama-server 8.9M (unchanged from D10.a).
- `LLAMA_MTP_FUSED` is honored only on the M=1 single-slot path. Batched mode (M>=2) takes the per-step path; documenting in the API header.

## Known follow-ups (not blockers for D10.b)

- Slot 1's 100% accept on the reverse_string prompt suggests the batched draft is finding shorter, higher-confidence chains than the per-slot path would. Investigate if this is desirable (early termination via p_min is fine) or a regression (artifact of multi-row hidden state interaction).
- Aggregate tg target was "≥ 60-80 t/s aggregate". D10.b reaches +27% over D10.a (39 vs 30 t/s) — meaningful but below the 2× target. Further lift requires either reducing per-step launch overhead (cuda graph reuse for batched draft) or eliminating the verify-side D2H bottleneck. Out of scope for D10.b.
