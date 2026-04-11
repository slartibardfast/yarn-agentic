# Vulkan Multi-GPU Split Mode Graph

## Design Principles

- **Vendor-neutral**: All changes use Vulkan spec features only (no driver-specific extensions). Works on RADV, NVIDIA proprietary, ANV (Intel), lavapipe, etc.
- **NVIDIA-safe**: No changes to CUDA/NCCL paths. Vulkan backend changes use standard Vulkan 1.2+ APIs (timeline semaphores, host-visible staging). NVIDIA's Vulkan driver supports all of these.
- **Upstreamable**: Minimal, surgical changes. Phase 1 (async+events) is a standalone improvement that benefits single-GPU Vulkan too. Each phase can be upstreamed independently.
- **Host-staging model**: Cross-device transfers go through host-visible memory (GPU A → host → GPU B). This is the only portable approach in the Vulkan spec. Phase 12 adds optional dmabuf P2P for RADV.

## File Layout

- `ggml/src/ggml-vulkan.cpp` — direct submodule edits for vtable wiring, event functions, async paths
- `ggml/src/ggml-vulkan-multigpu.cpp` — **new file** for split buffer type, cross-device staging, topology
- `ggml/include/ggml-vulkan.h` — public API additions (split buffer type)
- `src/llama.cpp` — integration points for split mode graph

## Phases

### Foundation (complete)

1. [PHASE 1: Enable Async Interface and Events](PHASE1.md) — **COMPLETE**
2. [PHASE 2: Implement Vulkan Split Buffer Type](PHASE2.md) — **COMPLETE**
3. [PHASE 3: Cross-Device Tensor Copies via Host Staging](PHASE3.md) — **COMPLETE**
4. [PHASE 4: Async Execution and Pipeline Parallelism](PHASE4.md) — **COMPLETE**
5. [PHASE 5: Topology Discovery and Performance Tuning](PHASE5.md) — **COMPLETE**
6. [PHASE 6: Runtime Testing, IQK Fallbacks, and lavapipe Fixes](PHASE6.md) — **COMPLETE**

### Performance (complete)

7. [PHASE 7: Async Cross-Device Copy Pipeline](PHASE7.md) — **COMPLETE**
8. [PHASE 8: Double-Buffered Staging](PHASE8.md) — **COMPLETE**
9. [PHASE 9: Parallel Split Buffer Uploads](PHASE9.md) — **COMPLETE**
10. [PHASE 10: True Async Graph Compute](PHASE10.md) — **COMPLETE**
11. [PHASE 11: Transfer Queue Utilization During Graph Execution](PHASE11.md) — SKIPPED (cross-device copies already use transfer queues; no DMA/compute overlap opportunity with 3 graph splits)
12. [PHASE 12: dmabuf Zero-Copy Cross-Device Transfer (RADV)](PHASE12.md) — **COMPLETE**

### Ops, Testing, and Bug Fixes (complete)

13. [PHASE 13: Vulkan FUSED_UP_GATE Support](PHASE13.md) — **COMPLETE**
14. [PHASE 14: Vulkan Support for Qwen3.5 Recurrent Layers](PHASE14.md) — **COMPLETE**
15. [PHASE 15: Vulkan Support for GLM-4.7-Flash](PHASE15.md) — **COMPLETE**
16. [PHASE 16: Backend-Ops Testing and Bug Fixes](PHASE16.md) — **COMPLETE**
17. [PHASE 17: Vulkan Op Trace for Nemotron](PHASE17.md) — **COMPLETE**
18. [PHASE 18: GPU-Accelerated REDUCE via dmabuf](PHASE18.md) — **COMPLETE**
19. [PHASE 19: Graph-Split Correctness](PHASE19.md) — **COMPLETE** (fixed by Phase 22)
20. [PHASE 20: Vulkan Token Generation Performance](PHASE20.md) — **IN PROGRESS**
21. [PHASE 21: nemotron_h_moe Architecture Support](PHASE21.md) — PLANNED
22. [PHASE 22: Fix get_tensor_async Race on rBAR Devices](PHASE22.md) — **COMPLETE**

### Upstream Alignment

0. [PHASE 0: Backend-Ops Test Failure Fixes](PHASE0.md) — **COMPLETE** (926/926 pass)

## Upstreamability Notes

| Phase | Upstream Risk | Notes |
|-------|---------------|-------|
| 1 | Low | Enables existing disabled code + adds standard event API |
| 2 | Medium | New split buffer type, mirrors CUDA pattern exactly |
| 3 | Medium | Cross-device copy via staging, clean fallback path |
| 4 | Medium | Async pipeline, needs careful testing on multiple vendors |
| 5 | Low | Read-only topology query, no behavioral change |
| 6 | Low-Medium | Scalar fallbacks and graceful stubs improve non-AVX2 builds; lavapipe fixes are additive |
| 7 | Medium | Replaces sync fence with timeline semaphore in copy path |
| 8 | Low | Internal buffer management change, no API impact |
| 9 | Low | Parallel uploads, independent per-device operations |
| 10 | Medium-High | Changes graph_compute contract from sync to async; scheduler must call synchronize |
| 11 | Medium-High | Adds cross-queue dependencies within a device; complex synchronization |
| 12 | Low (optional) | Driver-specific, behind capability probe, clean fallback |

## Performance Impact Summary

| Phase | Primary Benefit | Workload |
|-------|----------------|----------|
| 7 | Eliminates fence stall per cross-device copy | All multi-GPU |
| 8 | Overlaps staging with compute | Prompt processing |
| 9 | Parallel model loading | Startup time |
| 10 | Overlaps graph execution across GPUs | Prompt processing |
| 11 | Overlaps DMA with compute on same GPU | Bandwidth-bound ops |
| 12 | 2x cross-device bandwidth (RADV only) | All multi-GPU |
