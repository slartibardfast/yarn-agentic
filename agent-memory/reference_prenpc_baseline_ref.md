---
name: prenpc-baseline-ref
description: "Canonical pre-NPC perf baseline ref for ik_llama.cpp determinism-perf comparisons is `production/2026-q2` (merge-base with `production/2026-q2-next`). Do NOT pick intermediate commits on q2-next — all 50+ commits ahead are NPC/determinism work and dirty the baseline."
metadata: 
  node_type: memory
  type: reference
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

When running perf comparisons for the NP-determinism work on `ik_llama.cpp`:

- **HEAD** = `production/2026-q2-next` (current production binary with all six NPC fixes + F.4.1' kernel rewrite baked in).
- **Pre-NPC baseline** = `production/2026-q2` (the branch immediately before NPC.4 closure landed).

These two branches diverged at the NPC.4 closure work; `production/2026-q2` is the canonical pre-NPC ref. Build it in a separate worktree (e.g. `ik_llama-prenpc/`) with its own `build-prenpc/` so the HEAD `build/` is preserved.

**Do not** pick commits in the middle of `production/2026-q2-next` (e.g. `eb93b39f`, `746d19b0`) as a "pre-NPC" baseline. Those are mid-stream determinism / capture-tooling commits and dirty the comparison. The full NPC stack is 50+ commits ahead of `production/2026-q2`; only the merge-base is clean.

**Provenance:** confirmed by user during F.4.1'-followup nsys/ncu workflow on 2026-05-17.

Related: `[[project_np_determinism_complete_closure]]`, `[[project_fattn_per_slot_kv_p2_landed_kernel_only]]`.
