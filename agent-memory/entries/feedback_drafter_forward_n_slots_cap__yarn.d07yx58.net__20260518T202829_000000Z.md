---
name: drafter-forward-n-slots-cap
description: "Kernel storage strides must use bind-time capacity (n_slots_cap), never the dispatch count (N_slots). Conflating them produces wrong pointers when N_slots < storage capacity. Pattern was hidden in dflash_drafter_forward until Phase 4 exercised it."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

**Rule:** When a CUDA kernel takes a "slot count" parameter, that parameter should not double as the per-layer or per-slot storage stride if the storage was allocated with a different capacity at bind time. Keep them as separate parameters.

**Why:** `dflash_drafter_forward` used `N_slots` (the dispatch count) for both grid iteration AND per-layer K/V cache base offset: `layer * N_slots * SeqLen * H_kv * D_h`. The K/V cache was allocated by `alloc_ctx_scratch` with stride `n_slots_cap` (from `cparams.n_seq_max`). When a caller dispatches at `N_slots < n_slots_cap`, the kernel reads layer L's K/V from `layer * N_slots * (per-slot bytes)` but the data lives at `layer * n_slots_cap * (per-slot bytes)` — wrong offset, garbage results for any layer > 0. The bug was invisible at `n_seq_max=1` (where `N_slots == n_slots_cap == 1` always) and only surfaced when Phase 4's batch-vs-serial test set `n_seq_max=2` and the serial leg called with `N_slots=1`.

**How to apply:**
- When writing a kernel that uses caller-supplied slot/batch counts in pointer arithmetic, ask: "is this multiplying an offset into storage I didn't allocate myself?" If yes, take the storage stride as a separate parameter (or compute it from a separate cap parameter).
- When auditing existing kernels with multi-slot/multi-batch support, grep for `N_slots * ...` or `n_batch * ...` in pointer math and check whether `N_slots` is also the storage stride for the buffer being indexed. If the buffer's allocation site uses a different value (e.g., `n_seq_max`, a `_cap` field, or a bind-time constant), the kernel has the same bug.
- Document the distinction in the launcher's `.cuh` comment: "dispatch count (≤ storage capacity)" vs "storage stride (bind-time capacity)".
- A single test at `n_seq_max=1` is insufficient for multi-slot dispatch coverage — always include a test that runs at `n_seq_max > 1` with `N_slots < n_seq_max` to catch this class of bug.

See [[dflash-multislot-phase4-landed]] for the fix and [[no-host-concerns-in-code]] for the rule on keeping kernel parameter names content-descriptive (we called the new field `n_slots_cap`, not e.g. `phase4_capacity`).
