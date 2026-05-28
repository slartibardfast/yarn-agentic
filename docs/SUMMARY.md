# Summary

[Home](home.md)

# Active

- [PHASE_CUDA_NATIVE_DISPATCH](active/PHASE_CUDA_NATIVE_DISPATCH.md)
- [PHASE_HYBRID_CHECKPOINT](active/PHASE_HYBRID_CHECKPOINT.md)
- [PHASE_TU102_SPECIALIZATION](active/PHASE_TU102_SPECIALIZATION.md)

# Archive — phases by topic

## NP determinism

- [PHASE_NP_DETERMINISM_CLOSED — canonical writeup](archive/phases/10-np-determinism/PHASE_NP_DETERMINISM_CLOSED.md)
- [PHASE_NP8_FLAKE — NP=8 single-slot flake (governor-localized)](archive/phases/10-np-determinism/PHASE_NP8_FLAKE.md)
- [PHASE_NSTREAM_KV — 4D per-stream layout + Bug C closure](archive/phases/10-np-determinism/PHASE_NSTREAM_KV.md)
- [PHASE_NSTREAM_KV_PERF — Tier 1–5 perf recovery (superseded)](archive/phases/10-np-determinism/PHASE_NSTREAM_KV_PERF.md)
- [PHASE_T6_CHARACTERISATION — measurement-only tier (closed at T6.0.a)](archive/phases/10-np-determinism/PHASE_T6_CHARACTERISATION.md)
- [PHASE_PERF_F4_1 — F.4.1' kernel rewrite (closed 2026-05-17)](archive/phases/10-np-determinism/PHASE_PERF_F4_1.md)
- [BUNDLE_B_DESIGN — paged KV read path + kernel block_table indirection](archive/phases/10-np-determinism/BUNDLE_B_DESIGN.md)
- [PHASE_CY_F18_PROPER_FIX](archive/phases/10-np-determinism/PHASE_CY_F18_PROPER_FIX.md)
- [PHASE_NP_CLOSURE](archive/phases/10-np-determinism/PHASE_NP_CLOSURE.md)
- [PHASE_NPC4_FIX_AUDIT](archive/phases/10-np-determinism/PHASE_NPC4_FIX_AUDIT.md)
- [FIX_C_V5_FINAL_REPORT](archive/phases/10-np-determinism/FIX_C_V5_FINAL_REPORT.md)
- [PLAN_DELTANET](archive/phases/10-np-determinism/PLAN_DELTANET.md)
- [PLAN_DETERMINISM_AUDIT](archive/phases/10-np-determinism/PLAN_DETERMINISM_AUDIT.md)
- [PLAN_FIX_A](archive/phases/10-np-determinism/PLAN_FIX_A.md)
- [PLAN_FIX_C](archive/phases/10-np-determinism/PLAN_FIX_C.md)
- [PLAN_NP_CLOSURE](archive/phases/10-np-determinism/PLAN_NP_CLOSURE.md)

## Perf R-series

- [PHASE_PERF_R2_NP1 — kernel re-rank at np=1 × 256k vanilla shape](archive/phases/11-perf-r-series/PHASE_PERF_R2_NP1.md)
- [PHASE_PERF_R3_NP1 — nsys-driven characterization post-RT-hardening](archive/phases/11-perf-r-series/PHASE_PERF_R3_NP1.md)
- [PHASE_PERF_R3_FOLLOWUP — R1 -25.9% → -7.3% (closed + deployed)](archive/phases/11-perf-r-series/PHASE_PERF_R3_FOLLOWUP.md)

## Kernels

- [PHASE_MMQ_Q4_0_AR16 — full shape-invariant dispatch](archive/phases/12-kernels/PHASE_MMQ_Q4_0_AR16.md)
- [PHASE_ASYNC_REDUCE — Option B (planning)](archive/phases/12-kernels/PHASE_ASYNC_REDUCE.md)

## Scheduler

- [PHASE_GGML_SCHED_DYNSPLITS — adopt upstream dynamic-splits policy](archive/phases/13-scheduler/PHASE_GGML_SCHED_DYNSPLITS.md)

## Vulkan multi-GPU split-mode graph

- [Workstream plan](archive/phases/00-vulkan-multigpu/PHASE_VULKAN_MULTIGPU_PLAN.md)
- [Phase 0 — Backend-ops test failure fixes](archive/phases/00-vulkan-multigpu/PHASE0.md)
- [Phase 1 — Async interface and events](archive/phases/00-vulkan-multigpu/PHASE1.md)
- [Phase 2 — Vulkan split buffer type](archive/phases/00-vulkan-multigpu/PHASE2.md)
- [Phase 3 — Cross-device copies via host staging](archive/phases/00-vulkan-multigpu/PHASE3.md)
- [Phase 4 — Async execution and pipeline parallelism](archive/phases/00-vulkan-multigpu/PHASE4.md)
- [Phase 5 — Topology discovery and performance tuning](archive/phases/00-vulkan-multigpu/PHASE5.md)
- [Phase 6 — Runtime testing, IQK fallbacks, lavapipe fixes](archive/phases/00-vulkan-multigpu/PHASE6.md)
- [Phase 7 — Async cross-device copy pipeline](archive/phases/00-vulkan-multigpu/PHASE7.md)
- [Phase 8 — Double-buffered staging](archive/phases/00-vulkan-multigpu/PHASE8.md)
- [Phase 9 — Parallel split buffer uploads](archive/phases/00-vulkan-multigpu/PHASE9.md)
- [Phase 10 — True async graph compute](archive/phases/00-vulkan-multigpu/PHASE10.md)
- [Phase 11 — Transfer queue utilization during graph execution](archive/phases/00-vulkan-multigpu/PHASE11.md)
- [Phase 12 — dmabuf zero-copy cross-device transfer](archive/phases/00-vulkan-multigpu/PHASE12.md)
- [Phase 13 — Vulkan FUSED_UP_GATE](archive/phases/00-vulkan-multigpu/PHASE13.md)
- [Phase 14 — Vulkan support for Qwen3.5 recurrent layers](archive/phases/00-vulkan-multigpu/PHASE14.md)
- [Phase 15 — Vulkan support for GLM-4.7-Flash](archive/phases/00-vulkan-multigpu/PHASE15.md)
- [Phase 16 — Backend-ops testing and bug fixes](archive/phases/00-vulkan-multigpu/PHASE16.md)
- [Phase 17 — Vulkan op trace for Nemotron](archive/phases/00-vulkan-multigpu/PHASE17.md)
- [Phase 18 — GPU-accelerated REDUCE via dmabuf](archive/phases/00-vulkan-multigpu/PHASE18.md)
- [Phase 19 — Graph-split correctness](archive/phases/00-vulkan-multigpu/PHASE19.md)
- [Phase 20 — Vulkan token generation performance](archive/phases/00-vulkan-multigpu/PHASE20.md)
- [Phase 21 — Dispatch reduction for hybrid Mamba TG](archive/phases/00-vulkan-multigpu/PHASE21.md)
- [Phase 22 — Fix get_tensor_async race on rBAR devices](archive/phases/00-vulkan-multigpu/PHASE22.md)

## TURBO_KV_4B

- [Phase 23 — TURBO_4B weight quantization](archive/phases/20-turbo-kv-4b/PHASE23.md)
- [Phase 24 — Formal spec → test obligations](archive/phases/20-turbo-kv-4b/PHASE24.md)
- [Phase 25 — AVX2 kernel design](archive/phases/20-turbo-kv-4b/PHASE25.md)
- [Phase 26 — SOTA gap audit](archive/phases/20-turbo-kv-4b/PHASE26.md)
- [Phase 27 — Residual window implementation design](archive/phases/20-turbo-kv-4b/PHASE27.md)
- [Phase 28 — Residual window implementation](archive/phases/20-turbo-kv-4b/PHASE28.md)
- [Phase 28 Step 5 — Handoff](archive/phases/20-turbo-kv-4b/PHASE28_STEP5_HANDOFF.md)
- [Phase 30 — Cross-architecture Vulkan regression debug](archive/phases/20-turbo-kv-4b/PHASE30.md)

## MTP production

- [Phase 29 — CUDA/HIP backend gaps for tight Qwen3.5 MTP](archive/phases/30-mtp-production/PHASE29.md)
- [Phase 31 — MTP production on ik_llama.cpp](archive/phases/30-mtp-production/PHASE31.md)
- [Phase 32 — MTP FP16-recasting canary study (sm_75)](archive/phases/30-mtp-production/PHASE32-MTP-FP16-CANARY.md)
- [Phase 32 — MTP multislot theory](archive/phases/30-mtp-production/PHASE32-MTP-MULTISLOT-THEORY.md)
- [Phase 32 — V-F1-T1-QQ multislot results](archive/phases/30-mtp-production/PHASE32-V-F1-T1-QQ-MULTISLOT-RESULTS.md)
- [Phase 33 — Production cache tuning and multi-slot stability](archive/phases/30-mtp-production/PHASE33.md)
- [Phase 33 — Baseline](archive/phases/30-mtp-production/PHASE33-BASELINE.md)
- [Phase 33 — Cache tuning plan](archive/phases/30-mtp-production/PHASE33-CACHE-TUNING-PLAN.md)
- [Phase 33 — Snoop findings](archive/phases/30-mtp-production/PHASE33-SNOOP-FINDINGS.md)
- [Phase 33 — Step 2 results](archive/phases/30-mtp-production/PHASE33-STEP2-RESULTS.md)
- [Phase 33 — Step 6 mitigation and diagnosis plan](archive/phases/30-mtp-production/PHASE33-STEP6-MITIGATION-AND-DIAGNOSIS-PLAN.md)
- [Phase 34 — Production CUDA-OOM root-cause analysis](archive/phases/30-mtp-production/PHASE34-LEAK-RCA.md)

## Graph cache

- [Phase 35 — CUDA graph cache redesign](archive/phases/40-graph-cache/PHASE35-GRAPH-CACHE-REDESIGN.md)

## Multi-GPU MTP

- [Phase 36 — Multi-GPU MTP draft throughput (design)](archive/phases/50-mtp-multigpu/PHASE36-MULTI-GPU-MTP-DRAFT-THROUGHPUT.md)
- [Phase 36 — MTP audit (prior art)](archive/phases/50-mtp-multigpu/PHASE36-MTP-AUDIT.md)
- [Phase 36 — Fused design trace](archive/phases/50-mtp-multigpu/PHASE36-FUSED-DESIGN-TRACE.md)
- [Phase 36 — Implementation plan](archive/phases/50-mtp-multigpu/PHASE36-PLAN.md)
- [Phase 36 — First closure (premature)](archive/phases/50-mtp-multigpu/PHASE36-CLOSURE.md)
- [Phase 37 — Re-opening Phase 36 — chain-residual gap](archive/phases/50-mtp-multigpu/PHASE37.md)
- [Phase 38 — Full #2 dual-stream speculative dispatch](archive/phases/50-mtp-multigpu/PHASE38.md)
- [Phase 38 — D plan](archive/phases/50-mtp-multigpu/PHASE38_D_PLAN.md)
- [Phase 39 — Adopt upstream's collapsed-context chained-rollout MTP](archive/phases/50-mtp-multigpu/PHASE39.md)
- [Phase 39 — Tree drafting](archive/phases/50-mtp-multigpu/PHASE39-TREE-DRAFTING.md)
- [Phase 40 — Top-K fan-out tree MTP drafting](archive/phases/50-mtp-multigpu/PHASE40.md)
- [Phase 41 — Tree MTP foundation](archive/phases/50-mtp-multigpu/PHASE41.md)
- [Phase 43 — NCCL capture threadlocal](archive/phases/50-mtp-multigpu/PHASE43.md)
- [Phase 44 — CUDA graph capture investigation](archive/phases/50-mtp-multigpu/PHASE44.md)
- [Phase 44 — Synthesis](archive/phases/50-mtp-multigpu/PHASE44_SYNTHESIS.md)

## llama_context decompose

- [Phase 45 — Decompose llama_context](archive/phases/60-llama-context-decompose/PHASE45.md)
- [Phase 45 — Accretions](archive/phases/60-llama-context-decompose/PHASE45_ACCRETIONS.md)
- [Phase 45 — Callsites](archive/phases/60-llama-context-decompose/PHASE45_CALLSITES.md)
- [Phase 45 — Common port](archive/phases/60-llama-context-decompose/PHASE45_COMMON_PORT.md)
- [Phase 45 — D6 split](archive/phases/60-llama-context-decompose/PHASE45_D6_SPLIT.md)
- [Phase 45 — Decode deep map](archive/phases/60-llama-context-decompose/PHASE45_DECODE_DEEP_MAP.md)
- [Phase 45 — Field audit](archive/phases/60-llama-context-decompose/PHASE45_FIELD_AUDIT.md)
- [Phase 45 — KV paths](archive/phases/60-llama-context-decompose/PHASE45_KV_PATHS.md)
- [Phase 45 — ↔ Phase 39 integration](archive/phases/60-llama-context-decompose/PHASE45_PHASE39_INTEGRATION.md)
- [Phase 45 — Server port](archive/phases/60-llama-context-decompose/PHASE45_SERVER_PORT.md)

## DFlash speculative decoding

- [Qwen3.6-27B DFlash on sm_75](archive/phases/70-dflash/PHASE_DFLASH.md)
- [DFlash Qwen 3.6 27B port](archive/phases/70-dflash/DFLASH_QWEN36_PORT.md)
- [Multi-slot orchestrator (closed 2026-05-18)](archive/phases/70-dflash/PHASE_DFLASH_MULTISLOT.md)
- [Batched-pinned dispatch — perf collapse (closed 2026-05-19)](archive/phases/70-dflash/PHASE_DFLASH_BATCHED_PINNED.md)

## Multimodal

- [Phase 46 — Multi-GPU CLIP via tensor-split](archive/phases/80-multimodal/PHASE46-MULTIGPU-CLIP-TENSOR-SPLIT.md)
- [PHASE_R1_CLIP_RACE — CLIP cross-encode race (Phase A closed 2026-05-28)](archive/phases/80-multimodal/PHASE_R1_CLIP_RACE.md)

## Qwen3.5 MTP tool calling

- [Phase 1 — Peer host quickstart](archive/phases/qwen35-mtp-tooling/PHASE1.md)
- [Phase 2 — Tool-calling accuracy harness](archive/phases/qwen35-mtp-tooling/PHASE2.md)
- [Phase 3 — Closing the 5 t/s gap (Vulkan GGML_OP_FUSED)](archive/phases/qwen35-mtp-tooling/PHASE3.md)
- [Phase 4 — Zero-split MTP + TQ_V_4B scaffolding](archive/phases/qwen35-mtp-tooling/PHASE4.md)
- [Phase 5 — FP16 RPM research + mul_mat_vec infrastructure](archive/phases/qwen35-mtp-tooling/PHASE5.md)
- [Phase 6 — Vulkan TURBO_KV_4B — first Vulkan Walsh-Hadamard transform](archive/phases/qwen35-mtp-tooling/PHASE6.md)
- [Phase 7 — SET_ROWS correctness + Openclaw BigCodeBench comparison](archive/phases/qwen35-mtp-tooling/PHASE7.md)
- [Phase 8 — TURBO_KV_4B flash attention — multi-head stride bug](archive/phases/qwen35-mtp-tooling/PHASE8.md)

# Reference

- [Benchmark expectations](reference/BENCHMARK_EXPECTATIONS.md)
- [Benchmarks](reference/BENCHMARKS.md)
- [HARP_2B research log (DATA.md)](reference/DATA.md)
- [Dispatch comparison](reference/DISPATCH_COMPARISON.md)
- [MTP instructions](reference/MTP_INSTRUCTIONS.md)
- [Multi-GPU per-step checkpoint](reference/MULTI-GPU-PER-STEP-CHECKPOINT.md)
- [PHASE31 handoff (2026-05-01)](reference/handoffs/HANDOFF.md)
- [Transfer snapshot (2026-04-26)](reference/handoffs/TRANSFER.md)
- [CX.A aftermath audit (2026-05-16)](archive/refs-2026-05-16/AUDIT_2026-05-16_CX_A_AFTERMATH.md)
- [Research log (2026-05-16)](archive/refs-2026-05-16/RESEARCH_2026-05-16.md)

# Memory

- [Project memory (MEMORY.md)](MEMORY.md)
