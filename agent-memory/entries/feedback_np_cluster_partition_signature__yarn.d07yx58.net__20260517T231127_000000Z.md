---
name: np-cluster-partition-signature
description: "When NP byte-identity tests fail in a \"cluster of NPs ≡ each other but differ from another cluster\" pattern, suspect an ncols_y/ne2/batch-shape-derived dispatch decision (not random ULP drift). Single-GPU all-tensors-in-layer capture localizes cheaply."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

When the production NP-determinism harness fails with this pattern:

- `NP=A ≡ NP=B` mutually byte-identical, AND
- `NP=C ≡ NP=D` mutually byte-identical, BUT
- `{A,B}` cluster ≢ `{C,D}` cluster

the divergence is almost certainly a **dispatch decision** that
branches on a batch-shape-derived parameter (e.g. `args.ne2`,
`args.ncols_y`, `n_tokens`), not random ULP drift.

**Why:** Heuristics in `mmvq-templates.cuh` and similar dispatchers
pick `nwarps`, `rows_per_cuda_block`, kernel template instance,
etc. based on these dimensions. Different choices produce different
reduction trees → different ULP outcomes that are stable *within*
a cluster (deterministic for fixed shape) but differ *across*
clusters (different shapes cross a branch boundary).

**How to apply:**

1. Read the cluster partition. Which NP values agree with each
   other? The boundary is the cue.
2. Grep the relevant dispatcher (`mmvq-templates.cuh`,
   `mul_mat_vec_q_cuda`, etc.) for conditionals that branch on
   the shape variable matching that boundary.
3. The fix is usually pinning the dispatch decision so the
   reduction-order is constant across NP (e.g. `nwarps=4`
   regardless of `ncols_y`; `rpcb=1` regardless of `ncols_y`).
4. If the boundary doesn't map to a known dispatcher, single-GPU
   `llama-state-capture --all-in-layer --decode-only --layers
   <N1>,<N2>` localizes the first divergent tensor cheaply.

**Provenance (confirmed twice on this codebase):**
- 2026-05-17 F.4.1 (`feedback_simd_needs_independent_reference`-adjacent):
  `NP=4 ≡ NP=8` mutually identical, differ from `NP=1`. Boundary at
  `ne2 >= 2` triggers `nwarps=1` in the mmvq dispatcher. Fix:
  pin `nwarps=4` when `ids_data==nullptr`.
- 2026-05-17 F.4.1' (this entry): `NP={1,2,4}` mutually identical,
  `NP=8` differs. Boundary at `ncols_y <= 4 ? 4 : 2` in the
  `mul_mat_vec_q_cuda` nwarps selector. Fix: pin `nwarps=4`
  when `force_rpcb1`.

Same diagnostic signature, different dispatch dimension. Faster
than re-running intra-layer capture each time.

Related: `[[feedback_simd_needs_independent_reference]]` — when
SIMD/CPU dispatch decisions go wrong, the production-vs-reference
comparison is the surfaced; here the cross-NP byte comparison is
the corresponding production-stack surface.
