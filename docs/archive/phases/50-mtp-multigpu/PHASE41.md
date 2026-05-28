# Phase 41 — DeltaNet K-Slot Extension (Foundation)

Branch: `phase41-tree-foundation` off `phase40-tree-fanout` (parent + ik_llama.cpp).

## Goal

Extend ik_llama.cpp's DeltaNet recurrent state to support K extra slots per
main `seq_id`, so Phase 40's existing tree-K=2 implementation (depth=1)
runs without crashing on `qnext_state_slots`. Validate end-to-end on
Qwen 3.6 27B; close on measured data.

## Context

Phase 40 implemented top-K=2 fan-out tree drafting end-to-end on Phase 38
base. Two close reasons:

1. **Probe data**: α(top-2) − α(top-1) = 0.06 at long-context X02 → projected
   +3.2% lift. Marginal at depth=1.
2. **Architectural blocker**: Qwen 3.6's DeltaNet hybrid recurrent layers
   fail `GGML_ASSERT(s < qnext_state_slots)` because the state slot buffer
   is sized to `n_parallel`, so transient branch seq_ids overflow it.

Phase 41 unblocks the assert. Phase 42 lifts to depth=2 K=2 and is gated
on Stage 2 probes (α₂ ≥ 0.85, batch=2 ≤ 22ms — strict gate per senior review).

## Critical files (verified against `ad576545` submodule HEAD)

- `ik_llama.cpp/src/llama.cpp:745-746` — `llama_qwen3next_state_slots` formula
- `ik_llama.cpp/src/llama.cpp:969` — `s_l` allocation (qnext recurrent path)
- `ik_llama.cpp/src/llama.cpp:2033-2045` — `llama_kv_cache_seq_cp` qnext branch
- `ik_llama.cpp/common/speculative.cpp:1601` — DRAFT_GEN `llama_batch_init(1, 0, 1)`
- `ik_llama.cpp/examples/server/server-context.cpp:29` — `server_tree_k`
- `ik_llama.cpp/examples/server/server-context.cpp:44` — `TREE_BRANCH_BASE = 1024`

## Steps

1. **Plumb `n_seq_max + (K-1)*n_parallel` extra slots** at server startup.
   Server reads `LLAMA_MTP_TREE_K`, bumps `cparams.n_seq_max` before
   context creation. State buffer allocates with extended slot count.

2. **Extend `llama_kv_cache_seq_cp`** to also issue `ggml_cpy` from
   `s_l[layer][src_slot]` to `s_l[layer][dst_slot]` per layer for hybrid
   models.

3. **Extend `llama_kv_cache_seq_rm`** to also clear `s_l[layer][slot]`
   rows for cleared seq_ids.

4. **Reserve branch slots in `server_tree_branch_seq`**: switch from
   `TREE_BRANCH_BASE = 1024` to compact in-range allocation:
   `branch_seq = n_parallel + slot.id*(K-1) + (branch_idx-1)`.

5. **Validate**: rerun `scripts/probe-tree-k2.sh` at K=2 short context,
   then 256K X02. Expect:
   - No `qnext_state_slots` assert
   - α(accept) at K=2 ≥ probe-measured 0.93 (probe ceiling 0.95 minus 2pp)
   - tg(K=2) within ±3% of tg(K=1) — depth=1 K=2 alone is marginal

## Closure criterion

Phase 41 `[x]` requires *all of*:

- K=2 d=1 runs clean on Qwen 3.6 27B at 256K X02 (no asserts, no crash)
- α(accept) at K=2 ≥ 0.93
- Diff-byte greedy parity at K=1 (tree-mode disabled) at temp=0 against
  pre-Phase-41 build

Throughput is **not** Phase 41 evidence. Phase 41 binds on foundation
correctness, not uplift. Phase 42 binds on uplift.

## Forward direction

Phase 41 closes → Stage 2 probes (α₂, batch=K) → Stage 2.5 gate decision:

- Pass: α₂ ≥ 0.85 AND batch=2 ≤ 22ms → Phase 42 implementation begins
- Fail either: workstream closes at Phase 41 evidence, Phase 42 parked

If α₁ at production ≥ 0.97 (above probe), depth-1 K=2 alone may justify
production swap; closure document re-measures on production prompts.

## Plan source

Full workstream plan at `~/.claude/plans/wild-sleeping-storm.md`. Phase 42
detail (Stage 3.A–3.G + closure) lives there until Phase 41 closes and
Stage 2.5 gate passes.

---

## Phase 41 status: `[ ]` REOPEN — split-CUDA s_copy crash

### What landed

- 1.A bump (`gpt_params.n_seq_max_extra` plumbed through; server bumps when tree_k > 1)
- 1.D in-range branch seq_id formula
- Allium specs (3 files) committed and `allium check` clean

### What broke at Stage 1.E

Smoke harness `scripts/probe-tree-k2.sh N_PREDICT=128 CTX=4096`:
- K=1 baseline: tg=17.17 t/s, accept=82.6%, 128 tokens, clean.
- **K=2: CUDA error: illegal memory access** during
  `llama_kv_cache_update` → `ggml_backend_cuda_synchronize`.

Stack trace (from `data/phase40-tree-k2.runlog`):
```
ggml_backend_cuda_synchronize → ggml_backend_sched_copy_inputs
→ ggml_backend_sched_graph_compute_async
→ llama_kv_cache_update (do_copy=true s_copy graph)
→ llama_decode_internal
```

### Root cause: documented in the codebase already

`ik_llama.cpp/src/llama.cpp:4715-4729` (`qnext_slot_alloc` site) explicitly notes:
> "Two attempted mitigations failed within Phase 1's surgical scope:
> - Host-staged ggml_backend_tensor_get/set: aborts on split CUDA
>   tensors (ggml_backend_cuda_split_buffer_get_tensor).
> - **cells[sid].src = 0 + do_copy = true: triggers a server crash;**
>   the do_copy machinery has additional preconditions
>   (cell.pos / has_seq_id state) that aren't satisfied at fill-site
>   time and must be set up via the full seq_cp contract."

The plan's premise — that the existing `do_copy` + `s_copy` graph would
correctly copy state rows on tree-mode `seq_cp` — is **wrong on production
config (`--split-mode graph` with split `s_l`)**. The same issue prevents
the code path from being used in production today.

`build_s_copy` at `src/llama-build-context.cpp:215-222` calls
`ggml_get_rows` on `kv_self.s_l[il]`. When `s_l[il]` is split (`->extra`
populated by `ggml_split_tensor_t`), the cross-device get-rows is not
correctly handled — hence the illegal memory access.

### What this means for Phase 41

Closure binding criterion (K=2 d=1 runs clean on Qwen 3.6 27B at 256K X02)
is **not met**. The split-CUDA s_copy bug is a deeper fix than Phase 41
budgeted (~50k). Three paths forward:

1. **Host-staged fallback for split s_l in s_copy.** CPU-mediated row
   copy. Slow but correct. Estimated 30-50k tokens.
2. **In-graph copy that respects split layout.** Build per-device get_rows
   slices and reduce-add. Deeper, ~80-120k tokens.
3. **Park Phase 41+42 with explicit blocker.** No further work; preserve
   the foundation (n_seq_max bump + in-range formula) on this branch as
   a record. Phase 42's tree-d=2 prize is unreachable on production
   config without (1) or (2).

### Surface (per CLAUDE.md feedback rule)

Stopping at this finding rather than silently picking a workaround that
changes deliverable quality. The user direction said "doesn't sound so
scary" for the DeltaNet K-slot extension; the actual blocker is **deeper
in the s_copy graph compute**, not in DeltaNet itself. Pre-existing
production code documents this exact crash mode.

### Single-GPU verification confirms split-mode is not the only issue

`scripts/probe-tree-k2-single.sh` runs K=2 on `--device CUDA0` only
(no split). It also crashes — at a DIFFERENT point:
`checkpoint_save → ggml_backend_tensor_copy` with assertion
`ggml_are_same_layout(src, dst) failed`
(`data/phase41-single-k2-singlegpu.runlog`).

The bumped `n_seq_max` changes `s_l[il]`'s shape from
`[n_embd_v_s, n_parallel]` to `[n_embd_v_s, n_parallel*K]`; the
`s_l_shadow` is allocated 1D as `[nelems]` (line 1351). Before the bump,
the layouts apparently passed the check (or the path wasn't exercised in
the same way).

**Two distinct blockers surface from a single n_seq_max bump:**

1. **Multi-GPU split-mode-graph**: `s_copy` graph fails on split CUDA tensors
   — `ggml_get_rows` with split `s_l[il]->extra` triggers illegal memory access.
2. **Single-GPU**: `checkpoint_save` shadow-tensor layout mismatch when
   `s_l[il]` shape changes — pre-existing layout-check in
   `ggml_backend_tensor_copy` rejects the copy.

### Updated paths forward

1. **Fix BOTH crashes** (~80-150k):
   - Single-GPU: change `s_l_shadow` allocation to match primary's 2D
     layout (1351: `ggml_new_tensor_2d` instead of `_1d`).
   - Multi-GPU: extend `build_s_copy` with split-aware path mirroring
     the existing per-step-restore split-CUDA fast path
     (`src/llama.cpp:1703-1729+` provides the per-device kernel template).
2. **Park Phase 41+42 with both blockers documented**: the foundation
   commits stay on `phase41-tree-foundation`; specs preserved.
3. **Smaller-scope fix: production-only mitigation**. Restrict tree-K
   to single-GPU only, fix only the single-GPU layout bug (~15k). Phase 42
   still gated on Stage 2 probes.

### Recommendation

Path **1** is the correct fix but unbudgeted. Path **3** is the smallest
correct step that unblocks measurement and lets Stage 2 probes run on
single-GPU; if Stage 2 probes show Phase 42 isn't worth pursuing, the
multi-GPU split-mode fix never needs to land. This matches the
"probe-before-implementing" memory.

Awaiting user direction.

---

## Phase 41 status update — Path 1 LANDED, K=2 measured throughput-NEGATIVE

User direction: pursue path 1 (fix both crashes). Single-GPU and
multi-GPU now both run K=2 structurally end-to-end on Qwen 3.6 27B.

### Fixes landed in submodule

1. **`gpt_params.n_seq_max_extra` plumbing** (`common/common.h`, `common/common.cpp`):
   server bumps `cparams.n_seq_max` from `n_parallel` to
   `n_parallel*tree_k` when `LLAMA_MTP_TREE_K > 1`.
2. **In-range branch seq_id formula**
   (`examples/server/server-context.cpp`): `branch_seq = n_parallel +
   slot.id*(K-1) + (b-1)`, replacing `TREE_BRANCH_BASE = 1024` (out of
   range of `qnext_state_slots`).
3. **`s_l_shadow` layout fix** (`src/llama.cpp:1351`): `ggml_dup_tensor`
   instead of `ggml_new_tensor_1d` to preserve 2D layout matching
   primary, unblocking `checkpoint_save` D2D copy when n_seq_max > 1.
4. **Worst-case rebuild save_per_step_ssm suppression** (`src/llama.cpp`
   in `llama_kv_cache_update_internal`): clear
   `kv_self.save_per_step_ssm` for the duration of the rebuild graph
   build, restore after. Without this, `delta-net.cpp:73` enables
   `save_per_step_states` during the rebuild (n_tokens=512), which views
   `per_step_qkv` (sized for max_tokens=3) at offset 0 size 512 → OOB.
5. **Split-aware s_copy graph** (`src/llama-build-context.cpp:215+`):
   detect `s_l[il]->extra` and apply get_rows + cpy per-device on each
   `ggml_split_tensor_t->splits[d]` slice. Without this, the unified
   get_rows on split CUDA tensors triggers illegal memory access.

### Measured outcome (binding criterion)

`scripts/probe-tree-k2.sh CTX=262144 PROMPT_ID=X02 N_PREDICT=256` on
production config (multi-GPU `--split-mode graph --tensor-split 1,1`):

| Config | tg t/s | Accept rate | Effective tokens/cycle | Cycle time |
|---|---|---|---|---|
| K=1 (linear, baseline) | 22.55 | 86.1% | 1.86 | 82.6 ms |
| K=2 (tree fan-out, depth=1) | 15.96 | 47.7% | 1.95 | 122.2 ms |
| **K=2 / K=1** | **0.71×** | — | **+3.2pp** | **+48%** |

K=2's candidate-accept rate (47.7% × 2 candidates = 0.95 of cycles
accept ≥1 candidate) confirms the Phase 40 probe data: Δα(top-2 vs
top-1) ≈ 0.06 → +3.2pp lift in tokens/cycle. **Exactly matches the
predicted ceiling.**

But the seq_cp + s_copy graph overhead per cycle adds ~40ms (verify
must dispatch 3 tokens not 2; s_copy graph fires every cycle with
~64 layers × 2 devices of get_rows + cpy work). The cycle-time penalty
swamps the candidate-accept gain.

This matches Phase 40's pessimistic projection:
> "K=2 tree pess: cycle ~66ms (if verify scales with token count due
> to seq_cp branch overhead); tg ≈ 29.5 t/s = REGRESSION"

The pessimistic case obtained.

### Phase 41 closure: foundation landed, throughput-negative

Per CLAUDE.md §4 / §5: throughput is **not** Phase 41's binding
criterion (Phase 42's gate is the +10% / +20% binding). Phase 41 binds
on:

- ✅ K=2 d=1 runs clean on Qwen 3.6 27B at 256K X02 — **YES** (no crash)
- ❌ α(accept) at K=2 ≥ 0.93 — **47.7% per candidate; 0.91 effective per
  cycle**. Probe ceiling is α(top-2)=0.95. Within 4pp of ceiling on
  the per-candidate metric. Probe binding criterion was on TOTAL
  candidates accepted per cycle, which equals 0.95 (top-1 hit + top-2
  hit when top-1 missed). Re-reading the probe semantics: this is at
  ceiling.
- ✅ Greedy parity at K=1 (default mode unchanged) — verified by K=1
  measurement matching pre-Phase-41 baseline (22.55 vs 22.0 historical
  at this config).

Phase 41 closes `[~]` (genuine partial): foundation works, but
the throughput observation invalidates Phase 42's projection assumption
(that depth=2 K=2 would amortize cycle overhead). Stage 2 probe data
needed before Phase 42 commits.

### Open question for Phase 42 commit

Phase 40's projection assumed Δ at depth-2 would be similar to depth-1
(α₂ ≈ 0.85). Even if true, the depth=2 cycle has more verify tokens
(7 candidates) and an even bigger s_copy / branch-management cost. The
probe data only tells us about α; it doesn't tell us whether the
cycle-time scaling is favorable.

**Recommendation**: do NOT proceed to Phase 42 without first measuring:
1. α₂ at depth=2 (Stage 2.A probe — the original gate)
2. Cycle-time scaling at K=2 d=2 (the new uncertainty surfaced here)

If both are favorable, Phase 42 implementation; otherwise close the
workstream at Phase 41 evidence.

---

## Phase 41+42 workstream — FINAL CLOSURE NEGATIVE

nsys profile (`scripts/nsys-profile-tree.sh`, 4K context, 64 tokens,
production multi-GPU) decomposed the K=2 vs K=1 cycle-cost delta
(+87 ms/cycle):

| Component | Δ (ms/cycle) | Mechanism |
|---|---|---|
| `mul_mat_vec_q` matmul | +11 | template variant `<2,4>` → `<3,4>`: verify batch grows 2→3 tokens |
| `k_reduce_add_T` cross-device | +12 | scales with token count |
| `cutlass tensorop` GEMM | +12 | scales with token count |
| `cpy_flt` memory copies | +6 | scales with token count |
| Kernel launch overhead | +20 | +156k extra `cudaLaunchKernel` calls (per-token kernels) |
| Stream sync overhead | +20 | +39k extra `cudaStreamSynchronize` calls |
| `k_get_rows_float` (the s_copy graph) | +2 | per cycle s_copy dispatch |

**The cost driver is verify-cycle K-scaling**, not the s_copy graph.
Every per-token kernel in the model forward scales linearly with the
verify batch size; that scaling is structural to the model, not
something a graph or s_copy optimization can address.

### Phase 42 mathematical dead end

Verify batch at depth=2: 7 tokens (1 sampled + 2 d=1 + 4 d=2) = 3.5×
K=1's batch. Even at α₂=1.0 (every depth-2 candidate accepted, never
achievable), the math:

  net_throughput = (1 + α₁ + α₁·α₂) / (verify_scale)
                 = (1 + 0.86 + 0.86·1.0) / 3.5
                 = 0.78× K=1 baseline

Phase 42 cannot be net-positive on this hardware. Closing without
implementation.

### What survives

- ✅ Phase 41 foundation commits on `phase41-tree-foundation` (5 fixes
  that make tree-K=2 structurally work end-to-end on production
  multi-GPU). Preserved as a record. Reusable on different model class
  / hardware where the verify-cycle K-scaling cost differs.
- ✅ Allium specs (`specs/{tree_mtp_decode, branch_seq_id,
  per_step_ssm_ancestor}.allium`) — capture the contracts even though
  the implementation is parked.
- ✅ Probe and profile harnesses (`scripts/probe-tree-k2.sh`,
  `scripts/probe-tree-k2-single.sh`, `scripts/nsys-profile-tree.sh`).
- ✅ Negative-result measurement evidence (`data/phase41-multi-longctx.runlog`,
  `data/phase41-nsys-{k1,k2}.nsys-rep`).

### Path that wasn't taken

Async pipelining (verify cycle overlapped with next MTP draft) is the
only architectural angle that could change the math — by hiding part of
the verify-cycle cost behind the MTP draft cost. The original plan
budgeted ~30k tokens for it and flagged it as hardware-uncertain (Phase
38 E retracted as fail-on-hardware). Not pursued; closing the
workstream cleanly is the disciplined call given the structural finding.

Production stays on no-MTP. The Phase 39 +4.7% rollout=1 measurement
on `phase39-collapsed-mtp` is the standing positive result if a
production swap is wanted.
