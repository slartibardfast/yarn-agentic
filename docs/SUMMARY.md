# Summary

[Introduction](README.md)
[Project status (start here)](STATUS.md)

# Determinism — Qwen 3.6 27B on dual sm_75

- [Master phase tracker (A, B, C, CX, CY, D, E, F)](PHASE_MMQ_Q4_0_AR16.md)
- [Closed: NP-determinism — canonical writeup](PHASE_NP_DETERMINISM_CLOSED.md)
- [Open: F.4.1' perf-recovery phase](PHASE_PERF_F4_1.md)
## Archive (closed sub-phases + superseded plans)

- [Archive: PLAN_DETERMINISM_AUDIT](archive/np-determinism/PLAN_DETERMINISM_AUDIT.md)
- [Archive: PHASE_CY_F18_PROPER_FIX](archive/np-determinism/PHASE_CY_F18_PROPER_FIX.md)
- [Archive: PHASE_NP_CLOSURE](archive/np-determinism/PHASE_NP_CLOSURE.md)
- [Archive: PLAN_NP_CLOSURE](archive/np-determinism/PLAN_NP_CLOSURE.md)
- [Archive: PHASE_NPC4_FIX_AUDIT](archive/np-determinism/PHASE_NPC4_FIX_AUDIT.md)
- [Archive: FIX_C_V5_FINAL_REPORT](archive/np-determinism/FIX_C_V5_FINAL_REPORT.md)
- [Archive: PLAN_FIX_A](archive/np-determinism/PLAN_FIX_A.md)
- [Archive: PLAN_FIX_C](archive/np-determinism/PLAN_FIX_C.md)
- [Archive: PLAN_DELTANET (superseded by NSTREAM closure)](archive/np-determinism/PLAN_DELTANET.md)
- [Archive: CX.A aftermath audit (2026-05-16)](archive/2026-05-16-refs/AUDIT_2026-05-16_CX_A_AFTERMATH.md)
- [Archive: Research log (2026-05-16)](archive/2026-05-16-refs/RESEARCH_2026-05-16.md)

# N-stream KV cache (Bug C closure)

- [Closed: PHASE_NSTREAM_KV — 4D per-stream layout + Bug C structural closure](PHASE_NSTREAM_KV.md)
- [Closed: PHASE_NSTREAM_KV_PERF — Tier 3 dispatch + Tier 4 admission + Tier 5 paged KV + Tier 5.9 paged BACKING & defrag](PHASE_NSTREAM_KV_PERF.md)

# Characterisation (Tier 6)

- [Open: PHASE_T6_CHARACTERISATION — measurement-only tier; ablation matrix + per-feature deep-dives](PHASE_T6_CHARACTERISATION.md)

# DFlash speculative decoding

- [Qwen3.6-27B DFlash on sm_75](phases/70-dflash/PHASE_DFLASH.md)
- [DFlash Qwen 3.6 27B port](phases/70-dflash/DFLASH_QWEN36_PORT.md)
- [Archive: Multi-slot orchestrator (closed 2026-05-18)](archive/dflash/PHASE_DFLASH_MULTISLOT.md)
- [Archive: Batched-pinned dispatch — perf collapse (closed 2026-05-19)](archive/dflash/PHASE_DFLASH_BATCHED_PINNED.md)

# TU102 specialization

- [Qwen 3.6 27B kernel ranking (opened 2026-05-19)](PHASE_TU102_SPECIALIZATION.md)
- [Kernel re-rank at np=1 × 256k vanilla shape (opened 2026-05-24)](PHASE_PERF_R2_NP1.md)

# MTP production

- [Phase 29: CUDA/HIP backend gaps for tight Qwen3.5 MTP](phases/30-mtp-production/PHASE29.md)
- [Phase 31: MTP production on ik_llama.cpp](phases/30-mtp-production/PHASE31.md)
- [Phase 32: MTP FP16-recasting canary study (sm_75)](phases/30-mtp-production/PHASE32-MTP-FP16-CANARY.md)
- [Phase 32: MTP multislot theory](phases/30-mtp-production/PHASE32-MTP-MULTISLOT-THEORY.md)
- [Phase 32: V-F1-T1-QQ multislot results](phases/30-mtp-production/PHASE32-V-F1-T1-QQ-MULTISLOT-RESULTS.md)
- [Phase 33: Production cache tuning and multi-slot stability](phases/30-mtp-production/PHASE33.md)
- [Phase 33: Baseline](phases/30-mtp-production/PHASE33-BASELINE.md)
- [Phase 33: Cache tuning plan](phases/30-mtp-production/PHASE33-CACHE-TUNING-PLAN.md)
- [Phase 33: Snoop findings](phases/30-mtp-production/PHASE33-SNOOP-FINDINGS.md)
- [Phase 33: Step 2 results](phases/30-mtp-production/PHASE33-STEP2-RESULTS.md)
- [Phase 33: Step 6 mitigation and diagnosis plan](phases/30-mtp-production/PHASE33-STEP6-MITIGATION-AND-DIAGNOSIS-PLAN.md)
- [Phase 34: Production CUDA-OOM root-cause analysis](phases/30-mtp-production/PHASE34-LEAK-RCA.md)

# Multi-GPU MTP

- [Phase 36: Multi-GPU MTP draft throughput (design)](phases/50-mtp-multigpu/PHASE36-MULTI-GPU-MTP-DRAFT-THROUGHPUT.md)
- [Phase 36: MTP audit (prior art)](phases/50-mtp-multigpu/PHASE36-MTP-AUDIT.md)
- [Phase 36: Fused design trace](phases/50-mtp-multigpu/PHASE36-FUSED-DESIGN-TRACE.md)
- [Phase 36: Implementation plan](phases/50-mtp-multigpu/PHASE36-PLAN.md)
- [Phase 36: First closure (premature — see Phase 37)](phases/50-mtp-multigpu/PHASE36-CLOSURE.md)
- [Phase 37: Re-opening Phase 36 — chain-residual gap](phases/50-mtp-multigpu/PHASE37.md)
- [Phase 38: Full #2 dual-stream speculative dispatch](phases/50-mtp-multigpu/PHASE38.md)
- [Phase 38: D plan](phases/50-mtp-multigpu/PHASE38_D_PLAN.md)
- [Phase 39: Adopt upstream's collapsed-context chained-rollout MTP](phases/50-mtp-multigpu/PHASE39.md)
- [Phase 39: Tree drafting](phases/50-mtp-multigpu/PHASE39-TREE-DRAFTING.md)
- [Phase 40: Top-K fan-out tree MTP drafting (our novel work)](phases/50-mtp-multigpu/PHASE40.md)
- [Phase 41: Tree MTP foundation](phases/50-mtp-multigpu/PHASE41.md)
- [Phase 43: NCCL capture threadlocal](phases/50-mtp-multigpu/PHASE43.md)
- [Phase 44: CUDA graph capture investigation](phases/50-mtp-multigpu/PHASE44.md)
- [Phase 44: Synthesis](phases/50-mtp-multigpu/PHASE44_SYNTHESIS.md)

# Vulkan multi-GPU split-mode graph

- [Workstream plan](phases/00-vulkan-multigpu/PHASE_VULKAN_MULTIGPU_PLAN.md)
- [Phase 0: Backend-ops test failure fixes](phases/00-vulkan-multigpu/PHASE0.md)
- [Phase 1: Async interface and events](phases/00-vulkan-multigpu/PHASE1.md)
- [Phase 2: Vulkan split buffer type](phases/00-vulkan-multigpu/PHASE2.md)
- [Phase 3: Cross-device copies via host staging](phases/00-vulkan-multigpu/PHASE3.md)
- [Phase 4: Async execution and pipeline parallelism](phases/00-vulkan-multigpu/PHASE4.md)
- [Phase 5: Topology discovery and performance tuning](phases/00-vulkan-multigpu/PHASE5.md)
- [Phase 6: Runtime testing, IQK fallbacks, and lavapipe fixes](phases/00-vulkan-multigpu/PHASE6.md)
- [Phase 7: Async cross-device copy pipeline](phases/00-vulkan-multigpu/PHASE7.md)
- [Phase 8: Double-buffered staging](phases/00-vulkan-multigpu/PHASE8.md)
- [Phase 9: Parallel split buffer uploads](phases/00-vulkan-multigpu/PHASE9.md)
- [Phase 10: True async graph compute](phases/00-vulkan-multigpu/PHASE10.md)
- [Phase 11: Transfer queue utilization during graph execution](phases/00-vulkan-multigpu/PHASE11.md)
- [Phase 12: dmabuf zero-copy cross-device transfer](phases/00-vulkan-multigpu/PHASE12.md)
- [Phase 13: Vulkan FUSED_UP_GATE support](phases/00-vulkan-multigpu/PHASE13.md)
- [Phase 14: Vulkan support for Qwen3.5 recurrent layers](phases/00-vulkan-multigpu/PHASE14.md)
- [Phase 15: Vulkan support for GLM-4.7-Flash](phases/00-vulkan-multigpu/PHASE15.md)
- [Phase 16: Backend-ops testing and bug fixes](phases/00-vulkan-multigpu/PHASE16.md)
- [Phase 17: Vulkan op trace for Nemotron](phases/00-vulkan-multigpu/PHASE17.md)
- [Phase 18: GPU-accelerated REDUCE via dmabuf](phases/00-vulkan-multigpu/PHASE18.md)
- [Phase 19: Graph-split correctness](phases/00-vulkan-multigpu/PHASE19.md)
- [Phase 20: Vulkan token generation performance](phases/00-vulkan-multigpu/PHASE20.md)
- [Phase 21: Dispatch reduction for hybrid Mamba token generation](phases/00-vulkan-multigpu/PHASE21.md)
- [Phase 22: Fix get_tensor_async race on rBAR devices](phases/00-vulkan-multigpu/PHASE22.md)

# TURBO_KV_4B

- [Phase 23: TURBO_4B weight quantization](phases/20-turbo-kv-4b/PHASE23.md)
- [Phase 24: Formal spec → test obligations](phases/20-turbo-kv-4b/PHASE24.md)
- [Phase 25: AVX2 kernel design — non-AVX-512 microarchitecture targets](phases/20-turbo-kv-4b/PHASE25.md)
- [Phase 26: SOTA gap audit (Allium weed pass)](phases/20-turbo-kv-4b/PHASE26.md)
- [Phase 27: Residual window implementation design](phases/20-turbo-kv-4b/PHASE27.md)
- [Phase 28: Residual window implementation](phases/20-turbo-kv-4b/PHASE28.md)
- [Phase 28 Step 5: Handoff](phases/20-turbo-kv-4b/PHASE28_STEP5_HANDOFF.md)
- [Phase 30: Cross-architecture Vulkan regression debug](phases/20-turbo-kv-4b/PHASE30.md)

# Graph cache

- [Phase 35: CUDA graph cache redesign — test-first plan](phases/40-graph-cache/PHASE35-GRAPH-CACHE-REDESIGN.md)

# llama_context decompose

- [Phase 45: Decompose llama_context](phases/60-llama-context-decompose/PHASE45.md)
- [Phase 45: Accretions](phases/60-llama-context-decompose/PHASE45_ACCRETIONS.md)
- [Phase 45: Callsites](phases/60-llama-context-decompose/PHASE45_CALLSITES.md)
- [Phase 45: Common port](phases/60-llama-context-decompose/PHASE45_COMMON_PORT.md)
- [Phase 45: D6 split](phases/60-llama-context-decompose/PHASE45_D6_SPLIT.md)
- [Phase 45: Decode deep map](phases/60-llama-context-decompose/PHASE45_DECODE_DEEP_MAP.md)
- [Phase 45: Field audit](phases/60-llama-context-decompose/PHASE45_FIELD_AUDIT.md)
- [Phase 45: KV paths](phases/60-llama-context-decompose/PHASE45_KV_PATHS.md)
- [Phase 45: ↔ Phase 39 integration](phases/60-llama-context-decompose/PHASE45_PHASE39_INTEGRATION.md)
- [Phase 45: Server port](phases/60-llama-context-decompose/PHASE45_SERVER_PORT.md)

# Qwen3.5 MTP tool calling

- [Phase 1: Peer host quickstart](phases/qwen35-mtp-tooling/PHASE1.md)
- [Phase 2: Tool-calling accuracy harness](phases/qwen35-mtp-tooling/PHASE2.md)
- [Phase 3: Closing the 5 t/s gap (Vulkan GGML_OP_FUSED)](phases/qwen35-mtp-tooling/PHASE3.md)
- [Phase 4: Zero-split MTP + TQ_V_4B scaffolding](phases/qwen35-mtp-tooling/PHASE4.md)
- [Phase 5: FP16 RPM research + mul_mat_vec infrastructure](phases/qwen35-mtp-tooling/PHASE5.md)
- [Phase 6: Vulkan TURBO_KV_4B — first Vulkan Walsh-Hadamard transform](phases/qwen35-mtp-tooling/PHASE6.md)
- [Phase 7: SET_ROWS correctness + Openclaw BigCodeBench comparison](phases/qwen35-mtp-tooling/PHASE7.md)
- [Phase 8: TURBO_KV_4B flash attention — multi-head stride bug](phases/qwen35-mtp-tooling/PHASE8.md)

# Reference

- [Benchmark expectations](reference/BENCHMARK_EXPECTATIONS.md)
- [Benchmarks](reference/BENCHMARKS.md)
- [HARP_2B research log (DATA.md)](reference/DATA.md)
- [Dispatch comparison](reference/DISPATCH_COMPARISON.md)
- [MTP instructions](reference/MTP_INSTRUCTIONS.md)
- [Multi-GPU per-step checkpoint](reference/MULTI-GPU-PER-STEP-CHECKPOINT.md)
- [PHASE31 handoff (2026-05-01)](reference/handoffs/HANDOFF.md)
- [Transfer snapshot (2026-04-26)](reference/handoffs/TRANSFER.md)

# Memory

- [Project memory (MEMORY.md)](MEMORY.md)
