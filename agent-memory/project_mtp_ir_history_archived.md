---
name: MTP-IR history (archived) — from first port to production np=1
description: Consolidated history of MTP / MTP-IR work on ik_llama.cpp up to the 2026 Q2 production landing. Distilled from ~14 step-snapshot entries that are individually superseded by the terminal entries.
type: project
originSessionId: 9b52ee2b-d969-442c-83ba-98396525f6fa
---
This is the consolidated archive of the MTP / MTP-IR work on `ik_llama.cpp` between the initial single-pass port and the 2026 Q2 production landing. The individual step-snapshots that this entry distills were deleted during 2026-05-11 memory consolidation. Three terminal entries remain authoritative for the current state — see "Surviving entries" below.

## Lineage at a glance

1. **Initial port (single-pass MTP)** — working at 12% overhead, 1 split, cross-device fixed. Throughput-positive at np=1 on 27B Qwen 3.6.
2. **MTP spec decode on Qwen 3.5 35B-A3B** — end-to-end implemented; throughput-negative on CPU; GPU batch-parallel was identified as the only path to a win.
3. **MTP-IR (intermediate-state rollback)** — end-to-end correct on 0.8B + 35B, matched MTP-SEQ acceptance; slower than MTP-SEQ on MoE 35B under memory-spilling multi-GPU split; clean architectural verdict needed ≥26 GiB single GPU.
4. **MTP-IR port to ik_llama.cpp** — Steps 1–9 scaffolded; server built+ran coherent under `LLAMA_MTP_IR=1`; Step 6 rollback had a state-layout bug that was later resolved; no immediate speedup on the single-draft variant.
5. **35B-A3B MoE inline MTP** — drafts worked; server corruption isolated to seq_rm residue after batch=2 decode; fix landed via scheduler/rollback upgrade. Tests preserved under `tests/mtp-matrix/35b/`.
6. **Qwen 3.6 MTP on 6800 XT HIP** — verified correct (72% accept) but throughput-negative on 35B-A3B with `-ncmoe`. Master kept correctness-safe on HIP after `rw=0` default + FA LSE refusal.
7. **MTP verify-step fast-argmax cache (8a95fcb4)** — closed parity at MTP=1.282× baseline. The real cost wasn't bandwidth; it was per-verify D2H sync.
8. **Iter-7 lever post-mortem** — all 7 candidate optimisations died or parked via pre-step probes. The iter-7 cost-sim conflated prompt-eval vs gen costs. 0.8B was at the software ceiling of 1.282× MTP gain.
9. **MTP pipelining track (2-GPU optimisation)** — per-step checkpoint for split state (36ms, 7×) + MTP/re-decode interleaving (5ms, 1×). Sketched but not the production path.
10. **Phase 36 Step 0 measurements** — invalidated three plan premises: effective draft depth = 1.54 not 5; r7 KV-mismatch = 0% of misses; emb_d2h + hidden_h2d ≈ free. Reset the chain-extension reasoning.
11. **Phase 39 — upstream collapsed-context MTP port** — replaced separate ctx_mtp + chain_residual-seed with single-context inline MTP rollout. Primary-source: upstream measured 2.5×. Actual outcome on this fork: ported and working, but `tg=0.94× baseline at rollout=1` — coherent text + 39% accept but not yet a positive uplift.
12. **PHASE45 D9.5 milestone (tag `phase45-d9.5`)** — np=3 multi-slot first works, +29.8% lift over np=1 short-context baseline. KV drift was the hidden Phase 36/37/38-era cost; D9.5 closed it.

## What the chain produced as durable knowledge

- **Per-verify D2H sync was the dominant MTP cost**, not bandwidth. The fast-argmax cache addressing this was the single biggest perf delta in the chain.
- **0.8B is a yardstick only**; its 1.282× ceiling didn't translate to the 35B-A3B target. Workflow rule: exhaust 0.8B before moving to 35B-A3B; parked items unpark as a batch.
- **Phase-36 effective draft depth ≈ 1.54** is the real number, not the planning estimates of 5.
- **HIP path is correctness-safe but throughput-negative** on 35B-A3B; production decided on CUDA RTX 6000 + Qwen 3.6 27B.
- **MoE 35B-A3B needs ≥26 GiB single GPU** for a clean architectural verdict on MTP-IR; multi-GPU split adds memory-spill artifacts.
- **Tree fan-out is incompatible with hybrid recurrent attention** (DeltaNet/Mamba): the recurrent-state slot budget is `n_parallel`; parallel seq_id branching fails on Qwen 3.5/3.6. Reusable on pure-transformer models.

## Surviving entries (current authoritative state)

- **`project_mtp_draft_depth_27b_corrected.md`** — empirical 2026-05-09: `--draft 3` wins on Qwen 3.6 27B MTP-IR (33.23 t/s vs 31.88 at `--draft 1`). Supersedes Phase 36/37/38 "chain rollout > 1 regresses" projection.
- **`project_production_2026q2_landing.md`** — what shipped: np=1 + MTP `--draft 3` + ctx 256K on `production/2026-q2` in both yarn-agentic and ik_llama.cpp. Deterministic by construction.
- **`project_mtp_multislot_determinism_investigation_failed.md`** — terminal dead end on multi-slot determinism; production-shape kernels are already deterministic in unit tests; real bug is in an unaccounted surface (KV coord / RoPE / SSM state / scheduler / CUDA Graphs).

## Branches and artifacts preserved

- `production/2026-q2` branch in `slartibardfast/ik_llama.cpp` at `b07d0bbe`.
- `mtp-multislot-investigation-failed` branch at `ac994b7d` (15+ commits, all non-binding for the production same-prompt bug).
- T0 determinism fixture: `scripts/test-mtp-multislot-determinism.sh` in yarn-agentic at 54f6974 with `data/phase45-t0-negative-control/`.
- Tag `phase45-d9.5` marks the multi-slot-first-works milestone (now non-deterministic by the multi-slot bug).
- `tests/mtp-matrix/35b/` under ik_llama.cpp for the 35B-A3B inline MTP test set.
