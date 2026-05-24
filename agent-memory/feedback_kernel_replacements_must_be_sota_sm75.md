---
name: Kernel replacements must be SoTA for sm_75 with worked register allocations + profiling
description: When designing replacement CUDA kernels for the dual-TU102 sm_75 production target, the spec/design must include state-of-the-art per-shape geometry, fully worked-out register allocations, and committed-to-repo profiling data (nsys/ncu) — not just "TML batch-invariance recipe applied." Spec without these is incomplete.
type: feedback
originSessionId: 9b52ee2b-d969-442c-83ba-98396525f6fa
---
Rule: any CUDA kernel replacement for the dual-Quadro-RTX-6000 sm_75 / TU102 production target must include:

1. **State-of-the-art kernel design for sm_75**, citing the specific architecture features used (Turing `mma.sync.m16n8k8`, `ldmatrix.sync` swizzle patterns, 64 KiB SMEM per SM, 65536-register file per SM, 32-thread warps).
2. **Fully worked-out register allocations**: explicit register-per-thread budget, occupancy calculation, SMEM budget per CTA, expected blocks-per-SM. Use `--ptxas-options=-v` data committed to the design spec, not eyeballed.
3. **Profiling data baseline**: nsys + ncu profile of the kernel at the target shape committed to the repo as part of the design. Without measured throughput / SM utilization / memory-bandwidth %, the design is unanchored.
4. **Citation of the architecture's published peaks**: HBM 624 GB/s per-GPU (~1.25 TB/s aggregate with NVLINK), FP16 tensor-core 130.5 TFLOPs per-GPU (261 TFLOPs aggregate), NVLINK ~100 GB/s aggregate. Performance contract = % of these peaks.

**Why:** User has stated this explicitly multiple times across the workstream and confirmed again on 2026-05-14: "please note the kernel replacement must be SoTA for sm_75 with fully worked register allocations / profiling data". Also stated: "deterministic-at-any-cost is rejected" (see `feedback_determinism_must_co_optimize_perf`); also "100% CPU, 100% memory bandwidth utilization, not one wasted or repeated byte" (`feedback_zero_waste_mantra`). The DFlash kernel-design spec (`specs/dflash/kernel-design.md`) is the existing template: §6.x layouts have full register budgets, SMEM allocations, occupancy targets, and binding to architecture peaks. New kernels in the deltanet workstream must follow the same template.

**How to apply:**

- When writing a replacement kernel spec, include sections for: per-thread register budget (with `--ptxas-options=-v` output), SMEM layout, expected occupancy, target % of the relevant ceiling (HBM bandwidth for memory-bound ops, tensor-core peak for compute-bound ops).
- Reference DFlash kernel-design §6.1, §6.2, §6.3, §6.6 as templates. Each kernel there has these sections.
- During implementation: gather actual nsys + ncu profiles at the target shape, commit them to `data/deltanet/`, cite in the spec as the empirical bind.
- D5 / S2.4 spec writing in the staged-NP plan is incomplete without these sections.
- Reject "we'll worry about perf at D8" framing — the perf contract is part of the SPEC, not a post-implementation check. See also `feedback_determinism_must_co_optimize_perf`.

Companion to `feedback_determinism_must_co_optimize_perf` (perf is co-equal with correctness), `feedback_zero_waste_mantra` (full utilization mandate), and `feedback_anchor_to_measured_baselines` (cite measured numbers, not estimates).
