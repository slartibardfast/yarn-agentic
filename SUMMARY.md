# Summary

[Introduction](README.md)

- [Plan](PLAN.md)

# Phases

- [Phase 0: Backend-Ops Test Failure Fixes](PHASE0.md)
- [Phase 1: Async Interface and Events](PHASE1.md)
- [Phase 2: Vulkan Split Buffer Type](PHASE2.md)
- [Phase 3: Cross-Device Copies via Host Staging](PHASE3.md)
- [Phase 4: Async Execution and Pipeline Parallelism](PHASE4.md)
- [Phase 5: Topology Discovery and Performance Tuning](PHASE5.md)
- [Phase 6: Runtime Testing, IQK Fallbacks, and lavapipe Fixes](PHASE6.md)
- [Phase 7: Async Cross-Device Copy Pipeline](PHASE7.md)
- [Phase 8: Double-Buffered Staging](PHASE8.md)
- [Phase 9: Parallel Split Buffer Uploads](PHASE9.md)
- [Phase 10: True Async Graph Compute](PHASE10.md)
- [Phase 11: Transfer Queue Utilization During Graph Execution](PHASE11.md)
- [Phase 12: dmabuf Zero-Copy Cross-Device Transfer](PHASE12.md)
- [Phase 13: Vulkan FUSED_UP_GATE Support](PHASE13.md)
- [Phase 14: Vulkan Support for Qwen3.5 Recurrent Layers](PHASE14.md)
- [Phase 15: Vulkan Support for GLM-4.7-Flash](PHASE15.md)
- [Phase 16: Backend-Ops Testing and Bug Fixes](PHASE16.md)
- [Phase 17: Vulkan Op Trace for Nemotron](PHASE17.md)
- [Phase 18: GPU-Accelerated REDUCE via dmabuf](PHASE18.md)
- [Phase 19: Graph-Split Correctness](PHASE19.md)
- [Phase 20: Vulkan Token Generation Performance](PHASE20.md)
- [Phase 21: Dispatch Reduction for Hybrid Mamba Token Generation](PHASE21.md)
- [Phase 22: Fix get_tensor_async Race on rBAR Devices](PHASE22.md)
- [Phase 23: TURBO_4B Weight Quantization](PHASE23.md)
- [Phase 24: TURBO_KV_4B — Formal Spec → Test Obligations](PHASE24.md)
- [Phase 25: TURBO_KV_4B AVX2 Kernel Design — Non-AVX-512 Microarchitecture Targets](PHASE25.md)
- [Phase 26: TURBO_KV_4B — SOTA Gap Audit (Allium Weed Pass)](PHASE26.md)
- [Phase 27: TURBO_KV_4B — Residual Window Implementation Design](PHASE27.md)

# Qwen3.5 MTP Tool Calling

- [Phase 1: Peer Host Quickstart](phases/qwen35-mtp/PHASE1.md)
- [Phase 2: Tool-Calling Accuracy Harness](phases/qwen35-mtp/PHASE2.md)
- [Phase 3: Closing the 5 t/s Gap (Vulkan GGML_OP_FUSED)](phases/qwen35-mtp/PHASE3.md)
- [Phase 4: Zero-split MTP + TQ_V_4B Scaffolding](phases/qwen35-mtp/PHASE4.md)
- [Phase 5: FP16 RPM Research + mul_mat_vec Infrastructure](phases/qwen35-mtp/PHASE5.md)
- [Phase 6: Vulkan TURBO_KV_4B — First Vulkan Walsh-Hadamard Transform](phases/qwen35-mtp/PHASE6.md)
- [Phase 7: SET_ROWS Correctness + Openclaw BigCodeBench Comparison](phases/qwen35-mtp/PHASE7.md)
- [Phase 8: TURBO_KV_4B Flash Attention — Multi-Head Stride Bug](phases/qwen35-mtp/PHASE8.md)

# Performance

- [Benchmark Expectations](BENCHMARK_EXPECTATIONS.md)
- [Benchmarks](BENCHMARKS.md)

# Reference

- [Memory](MEMORY.md)
