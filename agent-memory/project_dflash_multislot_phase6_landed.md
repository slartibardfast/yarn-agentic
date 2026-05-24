---
name: dflash-multislot-phase6-landed
description: DFlash multi-slot Phase 6 (test-dflash-np-multislot.cpp full-pipeline harness) landed on production/2026-q2-next 2026-05-18; all 3 Phase 6 gates green; per-cycle latency bounded by known lm_head/GEMM bandwidth-bound kernel state (future kernel-optimization work)
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

DFlash multi-slot libllama API extension Phase 6 on `production/2026-q2-next` on 2026-05-18.

**Plan:** `data/dflash-multi-slot-impl-plan-2026-05-18.md` §Phase 6.
**Prior phases:** [[dflash-multislot-phase5-landed]].

## Phase 6 landed (commits `f9f6a284` submodule, parent bump)

### New harness `tests/dflash-speculative/test-dflash-np-multislot.cpp`
- Drives `llama_dflash_draft_batch` at NP ∈ {1,2,4,8} with identical
  per-seq prefill content.
- Captures slot-0 candidates at every NP, asserts byte-identity vs NP=1.
- Times `n_cycles=16` back-to-back draft cycles per NP and reports
  per-cycle latency, aggregate tokens/sec, per-slot tokens/sec.

### Measurement of record (locked clocks 1455 MHz, dual Quadro RTX 6000, n_cycles=16)

| NP | per_cycle ms | aggregate tok/s | per-slot tok/s | scaling vs N=1 |
|----|--------------|------------------|-----------------|-----------------|
| 1  | 3139         | 1.3              | 1.3             | 1.00×           |
| 2  | 3341         | 2.4              | 1.2             | 1.85×           |
| 4  | 3717         | 4.3              | 1.1             | 3.31×           |
| 8  | 5367         | 6.0              | 0.7             | 4.62×           |

Slot 0 byte-identical across all NPs.

## Phase 6 gates — all met
- Gate 1: NPC harness PASS NP={1,2,4,8} multi-GPU (verified at every prior phase).
- Gate 2: slot-0 byte-identity across NP ∈ {1,2,4,8} — new harness pins this at the driver layer (T7 already pinned it at the kernel layer).
- Gate 3: non-degenerate aggregate t/s, scaling 4.6× from N=1 to N=8.

## Notes
- Absolute tok/s is bounded by the DFlash lm_head + GEMM kernel state (~1% of TU102 peak — see [[dflash-t8-closed]]). Kernel optimization is future work; Phase 6's gate is multi-slot correctness + non-degenerate scaling, not absolute speed.
- Comparison against `data/phase_dflash_t8/bench-spec-{none,mtp}.json` is not apples-to-apples: T8 references are llama-bench single-slot (n_parallel=1) numbers; Phase 6 is multi-slot dispatch through the orchestrator. Different harnesses, different measurement units.
- Per-cycle latency growth is sublinear (1.71× cost for 8× slots), showing the shared drafter_forward + lm_head amortize across slots; per-slot drop to 47% retention at N=8 reflects the shared-kernel-pipeline serialization.

## What multi-slot DFlash now enables on this branch
1. C API: `llama_dflash_draft_batch(ctx, n_slots, ...)` — Phases 1-4.
2. Orchestrator: `common_speculative_draft_batched` all-DFlash fan-out — Phase 5.
3. Server: `llama-server --parallel N --spec-type dflash --model-draft <sidecar>` — server-side wiring fix landed alongside Phase 5.
4. Test pinning: T7 (kernel) + Phase 5 batch-vs-serial / spec-batched-fanout (C-API + orchestrator) + Phase 6 multi-slot harness (driver + throughput) — three layers of regression coverage.
