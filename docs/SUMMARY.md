# Summary

[Introduction](README.md)

# Vulkan multi-GPU split-mode graph

- [Workstream plan](phases/00-vulkan-multigpu/PHASE_VULKAN_MULTIGPU_PLAN.md)
- [Phase 0: Backend-Ops Test Failure Fixes](phases/00-vulkan-multigpu/PHASE0.md)
- [Phase 1: Async Interface and Events](phases/00-vulkan-multigpu/PHASE1.md)
- [Phase 2: Vulkan Split Buffer Type](phases/00-vulkan-multigpu/PHASE2.md)
- [Phase 3: Cross-Device Copies via Host Staging](phases/00-vulkan-multigpu/PHASE3.md)
- [Phase 4: Async Execution and Pipeline Parallelism](phases/00-vulkan-multigpu/PHASE4.md)
- [Phase 5: Topology Discovery and Performance Tuning](phases/00-vulkan-multigpu/PHASE5.md)
- [Phase 6: Runtime Testing, IQK Fallbacks, and lavapipe Fixes](phases/00-vulkan-multigpu/PHASE6.md)
- [Phase 7: Async Cross-Device Copy Pipeline](phases/00-vulkan-multigpu/PHASE7.md)
- [Phase 8: Double-Buffered Staging](phases/00-vulkan-multigpu/PHASE8.md)
- [Phase 9: Parallel Split Buffer Uploads](phases/00-vulkan-multigpu/PHASE9.md)
- [Phase 10: True Async Graph Compute](phases/00-vulkan-multigpu/PHASE10.md)
- [Phase 11: Transfer Queue Utilization During Graph Execution](phases/00-vulkan-multigpu/PHASE11.md)
- [Phase 12: dmabuf Zero-Copy Cross-Device Transfer](phases/00-vulkan-multigpu/PHASE12.md)
- [Phase 13: Vulkan FUSED_UP_GATE Support](phases/00-vulkan-multigpu/PHASE13.md)
- [Phase 14: Vulkan Support for Qwen3.5 Recurrent Layers](phases/00-vulkan-multigpu/PHASE14.md)
- [Phase 15: Vulkan Support for GLM-4.7-Flash](phases/00-vulkan-multigpu/PHASE15.md)
- [Phase 16: Backend-Ops Testing and Bug Fixes](phases/00-vulkan-multigpu/PHASE16.md)
- [Phase 17: Vulkan Op Trace for Nemotron](phases/00-vulkan-multigpu/PHASE17.md)
- [Phase 18: GPU-Accelerated REDUCE via dmabuf](phases/00-vulkan-multigpu/PHASE18.md)
- [Phase 19: Graph-Split Correctness](phases/00-vulkan-multigpu/PHASE19.md)
- [Phase 20: Vulkan Token Generation Performance](phases/00-vulkan-multigpu/PHASE20.md)
- [Phase 21: Dispatch Reduction for Hybrid Mamba Token Generation](phases/00-vulkan-multigpu/PHASE21.md)
- [Phase 22: Fix get_tensor_async Race on rBAR Devices](phases/00-vulkan-multigpu/PHASE22.md)

# TURBO_KV_4B

- [Phase 23: TURBO_4B Weight Quantization](phases/20-turbo-kv-4b/PHASE23.md)
- [Phase 24: TURBO_KV_4B — Formal Spec → Test Obligations](phases/20-turbo-kv-4b/PHASE24.md)
- [Phase 25: TURBO_KV_4B AVX2 Kernel Design — Non-AVX-512 Microarchitecture Targets](phases/20-turbo-kv-4b/PHASE25.md)
- [Phase 26: TURBO_KV_4B — SOTA Gap Audit (Allium Weed Pass)](phases/20-turbo-kv-4b/PHASE26.md)
- [Phase 27: TURBO_KV_4B — Residual Window Implementation Design](phases/20-turbo-kv-4b/PHASE27.md)
- [Phase 28: TURBO_KV_4B — Residual Window Implementation](phases/20-turbo-kv-4b/PHASE28.md)
- [Phase 28 Step 5: Handoff](phases/20-turbo-kv-4b/PHASE28_STEP5_HANDOFF.md)
- [Phase 30: turbo_kv_4b Cross-Architecture Vulkan Regression Debug](phases/20-turbo-kv-4b/PHASE30.md)

# MTP Production

- [Phase 29: CUDA/HIP Backend Gaps for Tight Qwen3.5 MTP](phases/30-mtp-production/PHASE29.md)
- [Phase 31: MTP Production on ik_llama.cpp](phases/30-mtp-production/PHASE31.md)
- [Phase 32: MTP FP16-Recasting Canary Study (sm_75)](phases/30-mtp-production/PHASE32-MTP-FP16-CANARY.md)
- [Phase 32: MTP Multislot Theory](phases/30-mtp-production/PHASE32-MTP-MULTISLOT-THEORY.md)
- [Phase 32: V-F1-T1-QQ Multislot Results](phases/30-mtp-production/PHASE32-V-F1-T1-QQ-MULTISLOT-RESULTS.md)
- [Phase 33: Production Cache Tuning and Multi-Slot Stability](phases/30-mtp-production/PHASE33.md)
- [Phase 33: Baseline](phases/30-mtp-production/PHASE33-BASELINE.md)
- [Phase 33: Cache Tuning Plan](phases/30-mtp-production/PHASE33-CACHE-TUNING-PLAN.md)
- [Phase 33: Snoop Findings](phases/30-mtp-production/PHASE33-SNOOP-FINDINGS.md)
- [Phase 33: Step 2 Results](phases/30-mtp-production/PHASE33-STEP2-RESULTS.md)
- [Phase 33: Step 6 Mitigation and Diagnosis Plan](phases/30-mtp-production/PHASE33-STEP6-MITIGATION-AND-DIAGNOSIS-PLAN.md)
- [Phase 34: Production CUDA-OOM Root-Cause Analysis](phases/30-mtp-production/PHASE34-LEAK-RCA.md)

# Graph Cache

- [Phase 35: CUDA Graph Cache Redesign — Test-First Plan](phases/40-graph-cache/PHASE35-GRAPH-CACHE-REDESIGN.md)

# Multi-GPU MTP

- [Phase 36: Multi-GPU MTP Draft Throughput (design)](phases/50-mtp-multigpu/PHASE36-MULTI-GPU-MTP-DRAFT-THROUGHPUT.md)
- [Phase 36: MTP Audit (prior art)](phases/50-mtp-multigpu/PHASE36-MTP-AUDIT.md)
- [Phase 36: Fused Design Trace](phases/50-mtp-multigpu/PHASE36-FUSED-DESIGN-TRACE.md)
- [Phase 36: Implementation Plan](phases/50-mtp-multigpu/PHASE36-PLAN.md)
- [Phase 36: First Closure (premature — see Phase 37)](phases/50-mtp-multigpu/PHASE36-CLOSURE.md)
- [Phase 37: Re-opening Phase 36 — chain-residual gap](phases/50-mtp-multigpu/PHASE37.md)
- [Phase 38: Full #2 dual-stream speculative dispatch](phases/50-mtp-multigpu/PHASE38.md)
- [Phase 38: D Plan](phases/50-mtp-multigpu/PHASE38_D_PLAN.md)
- [Phase 39: Adopt upstream's collapsed-context chained-rollout MTP](phases/50-mtp-multigpu/PHASE39.md)
- [Phase 39: Tree Drafting](phases/50-mtp-multigpu/PHASE39-TREE-DRAFTING.md)
- [Phase 40: Top-K Fan-Out Tree MTP Drafting (our novel work)](phases/50-mtp-multigpu/PHASE40.md)
- [Phase 41: Tree MTP Foundation](phases/50-mtp-multigpu/PHASE41.md)
- [Phase 43: NCCL Capture Threadlocal](phases/50-mtp-multigpu/PHASE43.md)
- [Phase 44: CUDA Graph Capture Investigation](phases/50-mtp-multigpu/PHASE44.md)
- [Phase 44: Synthesis](phases/50-mtp-multigpu/PHASE44_SYNTHESIS.md)

# llama_context decompose

- [Phase 45: Decompose llama_context](phases/60-llama-context-decompose/PHASE45.md)
- [Phase 45: Accretions](phases/60-llama-context-decompose/PHASE45_ACCRETIONS.md)
- [Phase 45: Callsites](phases/60-llama-context-decompose/PHASE45_CALLSITES.md)
- [Phase 45: Common Port](phases/60-llama-context-decompose/PHASE45_COMMON_PORT.md)
- [Phase 45: D6 Split](phases/60-llama-context-decompose/PHASE45_D6_SPLIT.md)
- [Phase 45: Decode Deep Map](phases/60-llama-context-decompose/PHASE45_DECODE_DEEP_MAP.md)
- [Phase 45: Field Audit](phases/60-llama-context-decompose/PHASE45_FIELD_AUDIT.md)
- [Phase 45: KV Paths](phases/60-llama-context-decompose/PHASE45_KV_PATHS.md)
- [Phase 45: ↔ Phase 39 Integration](phases/60-llama-context-decompose/PHASE45_PHASE39_INTEGRATION.md)
- [Phase 45: Server Port](phases/60-llama-context-decompose/PHASE45_SERVER_PORT.md)

# DFlash

- [Qwen3.6-27B DFlash Speculative Decoding (sm_75)](phases/70-dflash/PHASE_DFLASH.md)
- [Phase 46: DFlash speculative decoding port](phases/70-dflash/PHASE46.md)

# Qwen3.5 MTP Tool Calling

- [Phase 1: Peer Host Quickstart](phases/qwen35-mtp-tooling/PHASE1.md)
- [Phase 2: Tool-Calling Accuracy Harness](phases/qwen35-mtp-tooling/PHASE2.md)
- [Phase 3: Closing the 5 t/s Gap (Vulkan GGML_OP_FUSED)](phases/qwen35-mtp-tooling/PHASE3.md)
- [Phase 4: Zero-split MTP + TQ_V_4B Scaffolding](phases/qwen35-mtp-tooling/PHASE4.md)
- [Phase 5: FP16 RPM Research + mul_mat_vec Infrastructure](phases/qwen35-mtp-tooling/PHASE5.md)
- [Phase 6: Vulkan TURBO_KV_4B — First Vulkan Walsh-Hadamard Transform](phases/qwen35-mtp-tooling/PHASE6.md)
- [Phase 7: SET_ROWS Correctness + Openclaw BigCodeBench Comparison](phases/qwen35-mtp-tooling/PHASE7.md)
- [Phase 8: TURBO_KV_4B Flash Attention — Multi-Head Stride Bug](phases/qwen35-mtp-tooling/PHASE8.md)

# Reference

- [Benchmark Expectations](reference/BENCHMARK_EXPECTATIONS.md)
- [Benchmarks](reference/BENCHMARKS.md)
- [HARP_2B Research Log (DATA.md)](reference/DATA.md)
- [Dispatch Comparison](reference/DISPATCH_COMPARISON.md)
- [MTP Instructions](reference/MTP_INSTRUCTIONS.md)
- [Multi-GPU Per-Step Checkpoint](reference/MULTI-GPU-PER-STEP-CHECKPOINT.md)
- [PHASE31 Handoff (2026-05-01)](reference/handoffs/HANDOFF.md)
- [Transfer Snapshot (2026-04-26)](reference/handoffs/TRANSFER.md)

# Memory

- [Project Memory (MEMORY.md)](MEMORY.md)
