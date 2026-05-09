# PHASE45 D10.a baselines (2026-05-09, post-D10.a fixes, pre-D10.b)

Submodule HEAD: eef509d2 (PHASE45 D10.a fixes).
Parent HEAD: a4c2a4c → next bump after D10.a.
Tag: phase45-d10.a-multislot-boot.

## Single-slot x1 (no MTP)

Profile: qwen36-27b-x1.sh (np=1, ctx=262144, no -mtp).
Prompt: "The capital of France is" → 100 predicted tokens, temp=0, seed=1.

| Rep | predicted_n | tg (t/s) | draft_n | accepted |
|---|---|---|---|---|
| 1 | 100 | 30.82 | 0 | 0 |
| 2 | 100 | 31.00 | 0 | 0 |
| 3 | 100 | 30.92 | 0 | 0 |

Mean: 30.91 t/s. This is the base no-MTP single-slot baseline.

## Multi-slot x3-mtp (3 concurrent /v1/completions)

Profile: qwen36-27b-x3-mtp.sh (np=3, ctx=786432=256k×3, -mtp --draft 3, INLINE_KV=1).
Prompts (different per slot): "capital of France", "def reverse_string", "translation of hello to Spanish".
30 predicted tokens each, temp=0, seed=1.

| Slot | predicted_n | tg (t/s) | draft_n | accepted | accept rate |
|---|---|---|---|---|---|
| 0 | 30 | 11.50 | 23 | 18 | 78.3% |
| 1 | 30 | 9.522 | 20 | 15 | 75.0% |
| 2 | 30 | 9.672 | 22 | 16 | 72.7% |

Aggregate tg: ~30.7 t/s. **Flat vs single-slot's 30.91 t/s** — the multi-slot architecture provides correctness/isolation but not throughput at this stage. Per-slot draft generation is serial (each slot's `common_speculative_draft` is a separate forward), and on a bandwidth-bound 2× RTX 6000 setup, three serial draft forwards consume the same bandwidth budget as one slot's full decode.

D10.b's batched-draft API is the throughput unlock: collapse N per-slot serial draft forwards into ONE forward batched across slots' tokens.
