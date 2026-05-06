# Phase 36 Step 0 — profile pass(es) (2026-05-06)

Three runs at increasing realism. Raw `.log` files are gitignored
(`*.log`); reproduce via `scripts/profile-mtp-draft-cycle.sh` after
stopping the production llama-server.

## Runs

| Run             | Prompt                                                 | `-c`     | Sampler | Output dir                              |
|-----------------|--------------------------------------------------------|---------:|---------|-----------------------------------------|
| Synthetic 4K    | 30-token historic synthetic                            |   4 096  | greedy  | `data/profile-step0/`                   |
| X02 64K         | `scripts/agentic-prompt-corpus.jsonl#X02` (~1.3K tok)  |  65 536  | greedy  | `data/profile-step0-x02-c64k/`          |
| **X02 256K**    | `agentic-prompt-corpus.jsonl#X02`                      | 262 144  | greedy  | `data/profile-step0-x02-c256k/`         |

X02 256K matches production allocation (`profiles/qwen36-27b-x1.sh`
runs `--ctx-size 262144`). Greedy `temp=0` matches the fused MTP
path's trivial-sampler constraint but **not** production's chat
sampler (`temp=0.6 --top-p 0.95`); that regime would route through
the per-step fallback and is unprofiled.

## Reproduce

```sh
systemctl --user stop llama-server

# Synthetic 4K (the historic baseline):
bash scripts/profile-mtp-draft-cycle.sh

# X02 at 64K context:
CTX_SIZE=65536 PROMPT_ID=X02 OUTDIR_SUFFIX="-x02-c64k" \
  bash scripts/profile-mtp-draft-cycle.sh

# X02 at production 256K context:
CTX_SIZE=262144 PROMPT_ID=X02 OUTDIR_SUFFIX="-x02-c256k" \
  bash scripts/profile-mtp-draft-cycle.sh

systemctl --user start llama-server
```

## Throughput summary

| Regime         | nomtp  | d=1    | d=3    | d=5    | d=1 acc | d=5 acc |
|----------------|-------:|-------:|-------:|-------:|--------:|--------:|
| Synthetic 4K   |  33.21 |  34.25 |  30.46 |  30.00 |     81% |     54% |
| X02 64K        |  ~32   |  34.36 |  30.43 |  30.36 |     88% |     58% |
| **X02 256K**   |  31.13 |  32.14 |  28.76 |  28.65 |     88% |     58% |

The d=1 > d=3 > d=5 ranking is **invariant across all three regimes**.
Absolute throughput drops ~6% from 4K → 256K (verify graph carries
more attention work over the bigger KV).

## Per-component timing (microseconds, X02 256K)

| component               | d=0   | d=1   | d=3   | d=5   |
|-------------------------|------:|------:|------:|------:|
| build_graph             |  929  |  362  |  360  |  357  |
| sched_alloc_graph       | 1997  |  505  |  518  |  548  |
| graph_compute (all)     | 37109 | 20512 | 19662 | 19623 |
| set_inputs              |   18  |   19  |   25  |   46  |
| mtp_draft_step_decode   |   —   | 4980  | 5092  | 5044  |
| mtp_draft_step_emb_d2h  |   —   |    2  |    2  |    2  |
| mtp_draft_step_hidden_h2d| —    |    0  |    0  |    0  |

## Per-draft-step distribution at d=5

| step | Synthetic 4K (cycles=116) | X02 64K / 256K (cycles=197) |
|------|--------------------------:|----------------------------:|
| 0    | 116                       | 197                         |
| 1    |  59                       | 145                         |
| 2    |   2                       |   4                         |
| 3    |   1                       |   0                         |
| 4    |   1                       |   0                         |
| **avg drafts/cycle** | **1.54**          | **1.76**                    |

## `can_reuse_graph` HIT/MISS final summary

| Regime       | HIT  | MISS  | r1 (no_prev) | r2 (multi_token) | r7 (n_kv_changed) | r9 (mtp_op_changed) |
|--------------|-----:|------:|-------------:|-----------------:|------------------:|--------------------:|
| Synthetic 4K |   61 |   339 |    201 (59%) |        82 (24%)  |               0   |             56 (17%) |
| X02 64K      |  137 |   563 |    355 (63%) |       159 (28%)  |               1   |             48 ( 9%) |

r7 (`kv_self.n` changed) is 0–0.2% across both regimes — Step 5's
KQ_mask bucketing premise is invalidated in both.

## Key findings

1. Step 0 perf gate (`build+alloc ≥ 40% of per-step cost`) FAILED.
   build+alloc is 16–22% of per-draft-step cost on populated steps.
   Compute dominates.
2. Effective draft depth at d=5 averages ~1.5–1.8, not 5. The
   `prob < p_min` early-exit clamps depth aggressively in every
   regime tested.
3. r7 (n_kv changed) MISS reason fires 0–0.2% across regimes.
   Step 5 (KQ_mask bucketing) premise invalidated.
4. emb_d2h + hidden_h2d = 2 µs per step in every regime. Step 4
   (D2D relay) premise invalidated by 3 orders of magnitude.
5. d=1 > d=3 > d=5 throughput ranking is invariant from 4K to 256K
   context. Synthetic profile correctly captured the regression.

See `PHASE36-MULTI-GPU-MTP-DRAFT-THROUGHPUT.md` "Step 0 results"
and `PHASE36-PLAN.md` "Performance gate — RESULT: FAILED" for the
revised step ordering and revised cumulative throughput model
(realistic 256K-anchored ceiling: ~70 t/s, 2.2× baseline).
