---
name: Tree-K speculative workstream (Phase 40/41/42) — abandoned, superseded by DFlash
description: Consolidated history of the tree-K fan-out speculative decoding workstream. Closed for two independent reasons; superseded by the DFlash workstream on production/2026-q2-next.
type: project
originSessionId: 9b52ee2b-d969-442c-83ba-98396525f6fa
---

Consolidates two prior entries (deleted in this dream-flow pass):
- `project_tree_fanout_hybrid_recurrent_blocker.md` — DeltaNet incompatibility
- `project_phase41_42_workstream.md` — approved Phase 41+42 plan

## Lineage at a glance

1. **Phase 39** — measured K=1 d=1 single-draft baseline = 32.58 tok/s at 256K X02 context. Plain MTP draft=3 = 33.5 tok/s. Margin tight.
2. **Phase 40** — implemented top-K=2 d=1 fan-out tree drafting end-to-end. Hit `llama-delta-net.cpp:70 GGML_ASSERT(s < qnext_state_slots)` on first decode. Two independent close reasons:
   - Probe Δ = 0.06 t/s vs single-draft — too small to be worth the complexity.
   - DeltaNet recurrent state is sized to `--parallel` slot count (=1 in production); transient per-branch seq_ids (1024 + slot.id*8 + i) blow the slot allocator.
3. **Phase 41+42 plan** (`~/.claude/plans/wild-sleeping-storm.md`) — approved 18-task workstream combining option (A) DeltaNet K-slot extension with option (B) custom-mask tree structure. Never started; superseded.
4. **DFlash pivot** (current) — production target migrated to DFlash speculative on `production/2026-q2-next`. Tree-K abandoned as a direction on hybrid recurrent models.

## Durable knowledge produced

**Hybrid-recurrent + parallel seq_id branching is structurally incompatible.** DeltaNet (and Mamba/RWKV-class recurrent attention) keep per-seq_id recurrent state. Tree branches at the verify position diverge (different input tokens → different states), so each branch needs its own state slot. The codebase's recurrent-state arrays don't scale past `n_parallel`. Options to unblock are all expensive:
- (a) Skip tree mode for hybrid models.
- (b) Extend recurrent-state arrays to accommodate K extra slots per main seq_id.
- (c) Position-only branching with custom KQ_mask — bypasses seq_ids but adds substantial mask complexity.

When designing tree-shaped speculation for any model, first check whether it has recurrent layers. Pure-transformer models (no recurrent layers) work fine with seq_id branching.

## Preserved artifacts

- Branch: `phase40-tree-fanout` on both parent + submodule (Phase 40 implementation lives there as reusable code for any pure-transformer model port).
- Plan: `~/.claude/plans/wild-sleeping-storm.md` (approved but never executed; ignore unless reviving tree-K).

## What remains authoritative

- DFlash workstream on `production/2026-q2-next` — current direction.
- `project_continuous_batching_vs_perslot_dispatch.md` — separately addresses the multi-slot dispatch question.
- `project_production_2026q2_landing.md` — what shipped (MTP --draft 3 single-slot).
