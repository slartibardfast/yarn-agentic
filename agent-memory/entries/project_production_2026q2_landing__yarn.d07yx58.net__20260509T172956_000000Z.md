---
name: 2026 Q2 Production landing — Qwen 3.6 27B np=1 MTP draft=3
description: What shipped on production/2026-q2 branch on 2026-05-09 after PHASE45 D10.e per-slot dispatch was abandoned
type: project
originSessionId: 9b52ee2b-d969-442c-83ba-98396525f6fa
---
Following the PHASE45 D10.e per-slot dispatch abandonment (see `project_phase45_d10e_perslot_abandoned.md`), production landed at np=1 with MTP at draft depth 3.

**Profile:** `/home/llm/profiles/qwen36-27b-x1-mtp.sh` (active via `/home/llm/profiles/active.sh` symlink).

**Config:**
- Qwen 3.6 27B V-F1.T1.qq Q-loose vocab-fix GGUF
- 2× Quadro RTX 6000 (TU102), `--device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1`
- ctx 262144 (model native n_ctx_train; no YaRN)
- `-mtp --draft 3` (empirically beats draft=1 by +4%; see `project_mtp_draft_depth_27b_corrected.md`)
- `--parallel 1` (single slot — deterministic by construction; |B|=1 in every transformer compute)
- KV: q4_0 + hadamard
- Host-RSS budget: `--cache-ram 16384 --ctx-checkpoints 16` (post-2026-05-05 host-hang incident defaults)
- `LLAMA_MTP_INLINE_KV=1` (load-bearing per PHASE45 D9.9a)
- Sampler defaults: `--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.00 --repeat-penalty 1.0`

**Empirical perf (2026-05-09):**
- TG ≈ 33.5 t/s np=1 (smoke test 33.47 t/s, bench median 33.23 t/s)
- VRAM: 27.7 GiB used / 19.2 GiB free across 48 GiB total (40% headroom)
- Asymmetric split: GPU 0 = 11.4 GiB, GPU 1 = 17.0 GiB (graph-split bias; could rebalance but no need at current load)

**Branches pushed to GitHub:**
- yarn-agentic `production/2026-q2` @ 3b92dd3 — https://github.com/slartibardfast/yarn-agentic/tree/production/2026-q2
- ik_llama.cpp `production/2026-q2` @ b07d0bbe — https://github.com/slartibardfast/ik_llama.cpp/tree/production/2026-q2

**Tradeoff being shipped:**
- ✓ Byte-deterministic outputs (greedy stable across runs, no kernel batch-shape divergence at np=1)
- ✓ Full 256K context per conversation
- ✓ +4% TG vs `--draft 1`, ~+15% TG vs no-MTP single-slot
- ✗ One conversation in flight at a time — no multi-slot concurrency
- ✗ Per-deployment determinism only — changing `--draft N` flag changes output (kernel batch-shape sensitivity at verify-batch (1+N))

**If multi-slot concurrency becomes a hard requirement:** the recovery path is the hybrid fork/join workstream — keep RMSNorm/FFN/main matmul batched, fork per-slot only at DeltaNet (Bug A) and FA mma_f16 (Bug B), join after. Requires a `ggml_cuda_concurrent_event`-style infrastructure that ik_llama.cpp does not have (it lives upstream and was not ported). 1–2k LOC of new infra. Not on the current roadmap.

**Test fixture preserved:** `scripts/test-mtp-multislot-determinism.sh` (committed at yarn-agentic 54f6974) with negative-control divergence signature in `data/phase45-t0-negative-control/`. Runs np=1 vs np=3 byte-equivalence under MTP greedy. On bare phase45-decompose HEAD (D10.b, b07d0bbe) the test FAILS as expected — np=3 outputs diverge from np=1. Bind for any future revisit of multi-slot determinism work.

**Path 1 stash:** submodule branch `d10e0-llama-layer-perslot-wip` preserves the failed `LLAMA_BATCH_INVARIANT=multi_slot` env-gated llama-layer attempt + LLAMA_LAYER_TRACE diagnostics — for reference if anyone pursues the hybrid fork/join workstream.
