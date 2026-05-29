# Summary

[Home](home.md)

# Active

- [PHASE_HYBRID_CHECKPOINT](active/PHASE_HYBRID_CHECKPOINT.md)
- [PHASE_TU102_SPECIALIZATION](active/PHASE_TU102_SPECIALIZATION.md)

# Archive

## Determinism

- [PHASE_NP_DETERMINISM_CLOSED — canonical writeup](archive/phases/determinism/PHASE_NP_DETERMINISM_CLOSED.md)
- [PHASE_NP8_FLAKE — NP=8 single-slot flake (governor-localized)](archive/phases/determinism/PHASE_NP8_FLAKE.md)
- [PHASE_NSTREAM_KV — 4D per-stream layout + Bug C closure](archive/phases/determinism/PHASE_NSTREAM_KV.md)
- [PHASE_NSTREAM_KV_PERF — Tier 1–5 perf recovery (superseded)](archive/phases/determinism/PHASE_NSTREAM_KV_PERF.md)
- [PHASE_T6_CHARACTERISATION — measurement-only tier (closed at T6.0.a)](archive/phases/determinism/PHASE_T6_CHARACTERISATION.md)
- [PHASE_PERF_F4_1 — F.4.1' kernel rewrite (closed 2026-05-17)](archive/phases/determinism/PHASE_PERF_F4_1.md)
- [PHASE_PERF_R2_NP1 — kernel re-rank at np=1 × 256k vanilla shape](archive/phases/determinism/PHASE_PERF_R2_NP1.md)
- [PHASE_PERF_R3_NP1 — nsys-driven characterization post-RT-hardening](archive/phases/determinism/PHASE_PERF_R3_NP1.md)
- [PHASE_PERF_R3_FOLLOWUP — R1 -25.9% → -7.3% (closed + deployed)](archive/phases/determinism/PHASE_PERF_R3_FOLLOWUP.md)
- [PHASE_R1_CLIP_RACE — CLIP cross-encode race (Phase A closed 2026-05-28)](archive/phases/determinism/PHASE_R1_CLIP_RACE.md)
- [BUNDLE_B_DESIGN — paged KV read path + kernel block_table indirection](archive/phases/determinism/BUNDLE_B_DESIGN.md)
- [PHASE_CY_F18_PROPER_FIX](archive/phases/determinism/PHASE_CY_F18_PROPER_FIX.md)
- [PHASE_NP_CLOSURE](archive/phases/determinism/PHASE_NP_CLOSURE.md)
- [PHASE_NPC4_FIX_AUDIT](archive/phases/determinism/PHASE_NPC4_FIX_AUDIT.md)
- [FIX_C_V5_FINAL_REPORT](archive/phases/determinism/FIX_C_V5_FINAL_REPORT.md)
- [PLAN_DELTANET](archive/phases/determinism/PLAN_DELTANET.md)
- [PLAN_DETERMINISM_AUDIT](archive/phases/determinism/PLAN_DETERMINISM_AUDIT.md)
- [PLAN_FIX_A](archive/phases/determinism/PLAN_FIX_A.md)
- [PLAN_FIX_C](archive/phases/determinism/PLAN_FIX_C.md)
- [PLAN_NP_CLOSURE](archive/phases/determinism/PLAN_NP_CLOSURE.md)

## Multi-GPU

- [PHASE_CLIP_CAPTURE_SYNC — CLIP encode 28% faster (decoupled events); capture functional but not a CLIP win (closed + deployed 2026-05-29)](archive/phases/multi-gpu/PHASE_CLIP_CAPTURE_SYNC.md)
- [Vulkan multi-GPU workstream plan](archive/phases/multi-gpu/PHASE_VULKAN_MULTIGPU_PLAN.md)
- [Phase 0 — Backend-ops test failure fixes](archive/phases/multi-gpu/PHASE0.md)
- [Phase 1 — Async interface and events](archive/phases/multi-gpu/PHASE1.md)
- [Phase 2 — Vulkan split buffer type](archive/phases/multi-gpu/PHASE2.md)
- [Phase 3 — Cross-device copies via host staging](archive/phases/multi-gpu/PHASE3.md)
- [Phase 4 — Async execution and pipeline parallelism](archive/phases/multi-gpu/PHASE4.md)
- [Phase 5 — Topology discovery and performance tuning](archive/phases/multi-gpu/PHASE5.md)
- [Phase 6 — Runtime testing, IQK fallbacks, lavapipe fixes](archive/phases/multi-gpu/PHASE6.md)
- [Phase 7 — Async cross-device copy pipeline](archive/phases/multi-gpu/PHASE7.md)
- [Phase 8 — Double-buffered staging](archive/phases/multi-gpu/PHASE8.md)
- [Phase 9 — Parallel split buffer uploads](archive/phases/multi-gpu/PHASE9.md)
- [Phase 10 — True async graph compute](archive/phases/multi-gpu/PHASE10.md)
- [Phase 11 — Transfer queue utilization during graph execution](archive/phases/multi-gpu/PHASE11.md)
- [Phase 12 — dmabuf zero-copy cross-device transfer](archive/phases/multi-gpu/PHASE12.md)
- [Phase 13 — Vulkan FUSED_UP_GATE](archive/phases/multi-gpu/PHASE13.md)
- [Phase 14 — Vulkan support for Qwen3.5 recurrent layers](archive/phases/multi-gpu/PHASE14.md)
- [Phase 15 — Vulkan support for GLM-4.7-Flash](archive/phases/multi-gpu/PHASE15.md)
- [Phase 16 — Backend-ops testing and bug fixes](archive/phases/multi-gpu/PHASE16.md)
- [Phase 17 — Vulkan op trace for Nemotron](archive/phases/multi-gpu/PHASE17.md)
- [Phase 18 — GPU-accelerated REDUCE via dmabuf](archive/phases/multi-gpu/PHASE18.md)
- [Phase 19 — Graph-split correctness](archive/phases/multi-gpu/PHASE19.md)
- [Phase 20 — Vulkan token generation performance](archive/phases/multi-gpu/PHASE20.md)
- [Phase 21 — Dispatch reduction for hybrid Mamba TG](archive/phases/multi-gpu/PHASE21.md)
- [Phase 22 — Fix get_tensor_async race on rBAR devices](archive/phases/multi-gpu/PHASE22.md)
- [Phase 36 — Multi-GPU MTP draft throughput (design)](archive/phases/multi-gpu/PHASE36-MULTI-GPU-MTP-DRAFT-THROUGHPUT.md)
- [Phase 36 — MTP audit (prior art)](archive/phases/multi-gpu/PHASE36-MTP-AUDIT.md)
- [Phase 36 — Fused design trace](archive/phases/multi-gpu/PHASE36-FUSED-DESIGN-TRACE.md)
- [Phase 36 — Implementation plan](archive/phases/multi-gpu/PHASE36-PLAN.md)
- [Phase 36 — First closure (premature)](archive/phases/multi-gpu/PHASE36-CLOSURE.md)
- [Phase 37 — Re-opening Phase 36 — chain-residual gap](archive/phases/multi-gpu/PHASE37.md)
- [Phase 38 — Full #2 dual-stream speculative dispatch](archive/phases/multi-gpu/PHASE38.md)
- [Phase 38 — D plan](archive/phases/multi-gpu/PHASE38_D_PLAN.md)
- [Phase 39 — Adopt upstream's collapsed-context chained-rollout MTP](archive/phases/multi-gpu/PHASE39.md)
- [Phase 39 — Tree drafting](archive/phases/multi-gpu/PHASE39-TREE-DRAFTING.md)
- [Phase 40 — Top-K fan-out tree MTP drafting](archive/phases/multi-gpu/PHASE40.md)
- [Phase 41 — Tree MTP foundation](archive/phases/multi-gpu/PHASE41.md)
- [Phase 43 — NCCL capture threadlocal](archive/phases/multi-gpu/PHASE43.md)
- [Phase 44 — CUDA graph capture investigation](archive/phases/multi-gpu/PHASE44.md)
- [Phase 44 — Synthesis](archive/phases/multi-gpu/PHASE44_SYNTHESIS.md)
- [Phase 46 — Multi-GPU CLIP via tensor-split](archive/phases/multi-gpu/PHASE46-MULTIGPU-CLIP-TENSOR-SPLIT.md)

## Speculative decoding

- [Phase 29 — CUDA/HIP backend gaps for tight Qwen3.5 MTP](archive/phases/speculative-decoding/PHASE29.md)
- [Phase 31 — MTP production on ik_llama.cpp](archive/phases/speculative-decoding/PHASE31.md)
- [Phase 32 — MTP FP16-recasting canary study (sm_75)](archive/phases/speculative-decoding/PHASE32-MTP-FP16-CANARY.md)
- [Phase 32 — MTP multislot theory](archive/phases/speculative-decoding/PHASE32-MTP-MULTISLOT-THEORY.md)
- [Phase 32 — V-F1-T1-QQ multislot results](archive/phases/speculative-decoding/PHASE32-V-F1-T1-QQ-MULTISLOT-RESULTS.md)
- [Phase 33 — Production cache tuning and multi-slot stability](archive/phases/speculative-decoding/PHASE33.md)
- [Phase 33 — Baseline](archive/phases/speculative-decoding/PHASE33-BASELINE.md)
- [Phase 33 — Cache tuning plan](archive/phases/speculative-decoding/PHASE33-CACHE-TUNING-PLAN.md)
- [Phase 33 — Snoop findings](archive/phases/speculative-decoding/PHASE33-SNOOP-FINDINGS.md)
- [Phase 33 — Step 2 results](archive/phases/speculative-decoding/PHASE33-STEP2-RESULTS.md)
- [Phase 33 — Step 6 mitigation and diagnosis plan](archive/phases/speculative-decoding/PHASE33-STEP6-MITIGATION-AND-DIAGNOSIS-PLAN.md)
- [Phase 34 — Production CUDA-OOM root-cause analysis](archive/phases/speculative-decoding/PHASE34-LEAK-RCA.md)
- [Qwen3.6-27B DFlash on sm_75](archive/phases/speculative-decoding/PHASE_DFLASH.md)
- [DFlash Qwen 3.6 27B port](archive/phases/speculative-decoding/DFLASH_QWEN36_PORT.md)
- [DFlash multi-slot orchestrator (closed 2026-05-18)](archive/phases/speculative-decoding/PHASE_DFLASH_MULTISLOT.md)
- [DFlash batched-pinned dispatch — perf collapse (closed 2026-05-19)](archive/phases/speculative-decoding/PHASE_DFLASH_BATCHED_PINNED.md)

## Kernels and scheduling

- [PHASE_CUDA_NATIVE_DISPATCH — single-threaded CUDA-native dispatch (C0–C14), 6 Allium + 4 TLA+ gated; closed + deployed 2026-05-29](archive/phases/kernels-and-scheduling/PHASE_CUDA_NATIVE_DISPATCH.md)
- [PHASE_MMQ_Q4_0_AR16 — full shape-invariant dispatch](archive/phases/kernels-and-scheduling/PHASE_MMQ_Q4_0_AR16.md)
- [PHASE_ASYNC_REDUCE — Option B (planning)](archive/phases/kernels-and-scheduling/PHASE_ASYNC_REDUCE.md)
- [PHASE_GGML_SCHED_DYNSPLITS — adopt upstream dynamic-splits policy](archive/phases/kernels-and-scheduling/PHASE_GGML_SCHED_DYNSPLITS.md)
- [Phase 35 — CUDA graph cache redesign](archive/phases/kernels-and-scheduling/PHASE35-GRAPH-CACHE-REDESIGN.md)

## Quantization

- [Phase 23 — TURBO_4B weight quantization](archive/phases/quantization/PHASE23.md)
- [Phase 24 — Formal spec → test obligations](archive/phases/quantization/PHASE24.md)
- [Phase 25 — AVX2 kernel design](archive/phases/quantization/PHASE25.md)
- [Phase 26 — SOTA gap audit](archive/phases/quantization/PHASE26.md)
- [Phase 27 — Residual window implementation design](archive/phases/quantization/PHASE27.md)
- [Phase 28 — Residual window implementation](archive/phases/quantization/PHASE28.md)
- [Phase 28 Step 5 — Handoff](archive/phases/quantization/PHASE28_STEP5_HANDOFF.md)
- [Phase 30 — Cross-architecture Vulkan regression debug](archive/phases/quantization/PHASE30.md)

## Infrastructure

- [Phase 45 — Decompose llama_context](archive/phases/infrastructure/PHASE45.md)
- [Phase 45 — Accretions](archive/phases/infrastructure/PHASE45_ACCRETIONS.md)
- [Phase 45 — Callsites](archive/phases/infrastructure/PHASE45_CALLSITES.md)
- [Phase 45 — Common port](archive/phases/infrastructure/PHASE45_COMMON_PORT.md)
- [Phase 45 — D6 split](archive/phases/infrastructure/PHASE45_D6_SPLIT.md)
- [Phase 45 — Decode deep map](archive/phases/infrastructure/PHASE45_DECODE_DEEP_MAP.md)
- [Phase 45 — Field audit](archive/phases/infrastructure/PHASE45_FIELD_AUDIT.md)
- [Phase 45 — KV paths](archive/phases/infrastructure/PHASE45_KV_PATHS.md)
- [Phase 45 — ↔ Phase 39 integration](archive/phases/infrastructure/PHASE45_PHASE39_INTEGRATION.md)
- [Phase 45 — Server port](archive/phases/infrastructure/PHASE45_SERVER_PORT.md)

## Qwen3.5 tool calling

- [Phase 1 — Peer host quickstart](archive/phases/qwen35-tool-calling/PHASE1.md)
- [Phase 2 — Tool-calling accuracy harness](archive/phases/qwen35-tool-calling/PHASE2.md)
- [Phase 3 — Closing the 5 t/s gap (Vulkan GGML_OP_FUSED)](archive/phases/qwen35-tool-calling/PHASE3.md)
- [Phase 4 — Zero-split MTP + TQ_V_4B scaffolding](archive/phases/qwen35-tool-calling/PHASE4.md)
- [Phase 5 — FP16 RPM research + mul_mat_vec infrastructure](archive/phases/qwen35-tool-calling/PHASE5.md)
- [Phase 6 — Vulkan TURBO_KV_4B — first Vulkan Walsh-Hadamard transform](archive/phases/qwen35-tool-calling/PHASE6.md)
- [Phase 7 — SET_ROWS correctness + Openclaw BigCodeBench comparison](archive/phases/qwen35-tool-calling/PHASE7.md)
- [Phase 8 — TURBO_KV_4B flash attention — multi-head stride bug](archive/phases/qwen35-tool-calling/PHASE8.md)

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
