# Phase 36 Step 0 — first profile pass (2026-05-06)

Raw `.log` files are gitignored (`*.log` in repo root). To reproduce:

```sh
# From yarn-agentic root, with production llama-server stopped:
systemctl --user stop llama-server
bash scripts/profile-mtp-draft-cycle.sh
systemctl --user start llama-server
```

Outputs land back in this directory.

## Setup

- Submodule: `ik_llama.cpp/` at `phase36-mtp-throughput` @ `d15dd96`
  (per-step component timing + `can_reuse_graph` HIT/MISS counters,
  `IK_PRINT_TIMING` CLI-override fix).
- Build: `build-profile/` configured with
  `-DCMAKE_CXX_FLAGS=-DIK_PRINT_TIMING=1`.
- Model: `/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless.gguf`
  (production INT4 AutoRound, ~18 GiB).
- Hardware: 2× Quadro RTX 6000 (TU102, sm_75, 24 GiB each), CUDA 13.2.
- Server flags: `--device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1
  -ngl 999 -fa on -c 4096 --batch-size 2048 --ubatch-size 512
  --cache-type-k q4_0 --cache-type-v q4_0 --k-cache-hadamard --v-cache-hadamard`.
- Workload: 200 greedy tokens (`temperature=0`) on the
  "history of artificial intelligence" prompt, after a 16-token warmup.
- Profile env: `LLAMA_PROFILE_DECODE=1`.

## Results

### Throughput

| draft | t/s    | acceptance     |
|-------|--------|----------------|
| d=0   | 33.21  | —              |
| d=1   | 34.25  | 81% (96/118)   |
| d=3   | 30.46  | 55% (98/177)   |
| d=5   | 30.00  | 54% (97/179)   |

### Per-component timing (microseconds, mean across calls)

| component             | d=0   | d=1   | d=3   | d=5   |
|-----------------------|-------|-------|-------|-------|
| build_graph           |  903  |  353  |  342  |  354  |
| sched_alloc_graph     | 1891  |  476  |  453  |  473  |
| graph_compute (all)   | 31142 | 16726 | 16388 | 16381 |
| set_inputs            |    2  |    3  |   27  |   15  |
| mtp_draft_step_decode |   —   | 4898  | 5152  | 4934  |
| mtp_draft_step_emb_d2h|   —   |    2  |    2  |    2  |
| mtp_draft_step_hidden_h2d| —  |    0  |    0  |    0  |

### Per-draft-step distribution at d=5

| step | n   | mean decode (µs) |
|------|-----|------------------|
| 0    | 116 | 4915             |
| 1    | 59  | 4970             |
| 2    | 2   | 4951             |
| 3    | 1   | 4949             |
| 4    | 1   | 4956             |

Effective draft depth = 179 / 116 = **1.54** (vs requested 5).

### `can_reuse_graph` HIT/MISS at d=5 (final summary)

HIT=61, MISS=339:
- r1 (no_prev): 201 (59%)
- r2 (multi_token): 82 (24%)
- r9 (mtp_op_changed): 56 (17%)
- r7 (n_kv_changed): **0**

## Key findings

1. Step 0 performance gate FAILED — build+alloc is 16–22% of
   per-draft-step cost on populated steps, not ≥40%. Compute
   dominates.
2. Draft depth at d=5 averages 1.54, not 5. The
   `prob < p_min` early-exit clamps depth aggressively.
3. `r7` (n_kv changed) = 0 % of misses — Step 5 (KQ_mask
   bucketing) premise invalidated.
4. emb_d2h + hidden_h2d = 2 µs total per step — Step 4
   (D2D relay) premise invalidated by 3 orders of magnitude.

See `PHASE36-MULTI-GPU-MTP-DRAFT-THROUGHPUT.md` "Step 0 results"
and `PHASE36-PLAN.md` "Performance gate — RESULT: FAILED" for the
revised step ordering and revised cumulative throughput model.
