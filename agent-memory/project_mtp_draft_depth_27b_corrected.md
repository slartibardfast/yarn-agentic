---
name: MTP --draft depth on Qwen 3.6 27B (MTP-IR, corrected)
description: Empirical 2026-05-09 finding that --draft 3 wins on Qwen 3.6 27B with MTP-IR on 2× RTX 6000, contradicting the older Phase 36/37/38 projection
type: project
originSessionId: 9b52ee2b-d969-442c-83ba-98396525f6fa
---
On Qwen 3.6 27B (V-F1.T1.qq Q-loose vocab-fix), MTP-IR enabled, ctx 256K with q4_0+hadamard KV cache, np=1, on the production 2× Quadro RTX 6000 (TU102) split-graph topology, the empirical 3-run-median TG at n_predict=128 greedy (T=0):

| --draft | tg (median t/s) | within-depth determinism |
|---|---|---|
| 1 | 31.88 | bit-stable across runs |
| 2 | 31.50 | bit-stable across runs |
| 3 | **33.23** | bit-stable across runs |

Depth 3 is +4% over depth 1. Depth 2 is slightly worse than depth 1 (the "rollout overhead without enough accept-rate amortisation" point of the old memory does hold at depth 2, but is overturned by depth 3).

**Why:** likely MTP-IR's verify-step amortisation behaves differently than the pre-MTP-IR head implementation the older memory was based on. Higher draft depth puts more tokens in the verify batch which improves arithmetic intensity at the verify pass; this beats the per-draft compute cost at depth 3 even at typical 0.63 accept rates. Depth 2 sits in a saddle point where the rollout overhead exceeds the verify amortisation benefit.

**Cross-depth output divergence (worth noting):** outputs differ across `--draft` values for the same prompt:
- draft=1 ≠ draft=2
- draft=1 ≠ draft=3
- draft=2 == draft=3

Same kernel batch-shape sensitivity surface as PHASE45 D10.e Bug A/B (DeltaNet, FA mma_f16) — the verify batch shape is `(1 + N drafts) tokens`, so different N produces different float-sum order, occasionally flipping greedy argmax. Within a fixed `--draft N` deployment, runs are bit-deterministic. Cross-deployment with different N values, outputs differ.

**How to apply.** Don't cite the older Phase 36/37/38 chain-rollout-regression memory as a depth recommendation for current MTP-IR builds. Re-measure on the actual code path being shipped. If shipping non-deterministic across config changes is a concern, fix `--draft` at deployment time and don't switch.

**Production state (2026-05-09):** ships `--draft 3` in `qwen36-27b-x1-mtp.sh` on `production/2026-q2` branch in both yarn-agentic and ik_llama.cpp.
