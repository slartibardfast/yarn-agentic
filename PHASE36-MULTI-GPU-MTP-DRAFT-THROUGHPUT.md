# Phase 36: Multi-GPU MTP Draft Throughput

Make draft depth > 1 faster than draft depth 1 on 2-GPU graph split.

## Status

| Step | State | Summary |
|------|-------|---------|
| 1. Profile draft overhead | [ ] | Enable `IK_PRINT_TIMING` + `LLAMA_PROFILE_DECODE`, measure per-component cost across 100+ draft steps |
| 2. KQ_mask bucketing | [ ] | Pad `n_kv` to bucket boundaries so consecutive draft steps reuse the same graph shape |
| 3. Fused multi-draft graph | [ ] | Single cgraph chaining N draft steps — one build+alloc instead of N |
| 4. MTP tail / DeltaNet pipeline overlap | [ ] | Overlap `mtp_accept_tokens` (~5 ms) with DeltaNet state re-advancement on separate GPU contexts |
| 5. Acceptance rate investigation | [ ] | Understand why d≥2 acceptance drops from 86% to 59–63% on this model/quant |

## Context

Per-step checkpoint for split DeltaNet state landed (see
[Multi-GPU Per-Step Checkpoint](MULTI-GPU-PER-STEP-CHECKPOINT.md)).
The ~36 ms re-decode penalty is eliminated. But d=1 still wins:

| Config | Throughput | Accept |
|--------|-----------|--------|
| No MTP | 33.5 t/s | — |
| d=1 | 35.3 t/s | 86% |
| d=3 | 32.5 t/s | 63% |
| d=5 | 32.4 t/s | 59% |

Two bottlenecks remain: per-draft scheduling overhead and
acceptance rate decay.

## Step 1: Profile draft overhead

The codebase has `#if IK_PRINT_TIMING` instrumentation around
`build_graph`, `sched_alloc_graph`, and `compute` in `llama.cpp`.
Also `LLAMA_PROFILE_DECODE` in `speculative.cpp`.

Build with `-DIK_PRINT_TIMING=1`, run with
`LLAMA_PROFILE_DECODE=1`. Collect timing for 100+ draft steps to
validate the estimated cost breakdown:

| Component | Estimated | Source |
|-----------|-----------|--------|
| build_graph | ~1 ms | CPU: walk layers, create ggml nodes |
| sched_alloc_graph | ~3 ms | CPU: split graph, assign backends, allocate |
| compute (output matmul) | ~3 ms | GPU: bandwidth-limited [1,3584]×[3584,152K] |
| compute (attn+FFN) | ~1 ms | GPU: small matmuls, local tensors |
| scheduling overhead | ~2 ms | CPU: dispatch, sync |
| **Total per draft** | **~10 ms** | |

For d=5: 5 × 10 ms = 50 ms draft overhead vs 30 ms baseline step.

## Step 2: KQ_mask bucketing for graph reuse

Pad `n_kv` to the nearest bucket (e.g., 64) when constructing the
KQ_mask tensor. Consecutive draft steps in the same bucket produce
identical graph shapes, enabling `can_reuse_graph()`.

**File:** `src/llama-build-context.cpp` (`build_inp_KQ_mask`)

```cpp
// Before:
lctx.inp_KQ_mask = ggml_new_tensor_2d(ctx0, type, n_kv, ...);
// After:
int64_t n_kv_bucketed = GGML_PAD(n_kv, 64);
lctx.inp_KQ_mask = ggml_new_tensor_2d(ctx0, type, n_kv_bucketed, ...);
```

Mask values for `[n_kv, n_kv_bucketed)` set to `-inf` in
`set_inputs()`. Over-read cost: ~63 extra KV rows × n_heads ×
head_dim ≈ 200 KB — negligible.

**Expected impact:** ~4 out of 5 draft steps at d=5 skip the ~4 ms
build+alloc cost → ~16 ms saved per speculative batch.

## Step 3: Fused multi-draft graph

Build a single `ggml_cgraph` that chains N draft steps. One
`build_graph()` + `alloc_graph()` call for all N drafts. Each
step's `ggml_argmax` output feeds the next step's
`ggml_get_rows(tok_embd, ...)`.

```
inp_mtp_states → [step_0: embed→norm→eh_proj→attn→FFN→output→argmax]
                  → [step_1: embed(argmax_0)→...→argmax_1]
                  → ...
                  → [step_4: ...→argmax_4]
output: 5 token IDs
```

KV positions pre-allocated. Each attention step reads up to
`n_kv_start + step_index` entries.

**Cost reduction:** 1 × (build + alloc) instead of N ×
(build + alloc). For N=5: save ~20 ms per speculative batch.

**Files:** `src/graphs/build_qwen35.cpp` (fused builder),
`src/llama.cpp` (`MTP_OP_DRAFT_GEN_FUSED`),
`common/speculative.cpp` (use fused path for trivial sampler).

## Step 4: MTP tail / DeltaNet pipeline overlap

After verify+accept, `mtp_accept_tokens` (~5 ms) and DeltaNet
state re-advancement are sequential. They use separate contexts
and separate GPUs — overlap is possible.

**Prerequisite:** Steps 2–3 first (higher payoff, simpler).

## Step 5: Acceptance rate investigation

d=1 acceptance (86%) drops to 59–63% at d≥2. Possible causes:

- Quantization noise amplified by autoregressive chaining
  (IQ4_XS + q4_0 Hadamard KV)
- Model-intrinsic: Qwen3.6 27B MTP head may not predict well
  beyond 1 step on this distribution
- Temperature/sampling interaction with speculative path

Compare acceptance at different quant levels (Q6_K, Q8_0) and
against upstream single-GPU numbers to isolate.

## Target

d=5 throughput exceeding d=1 (35.3 t/s) on 2× RTX 6000 graph
split. Upstream single-GPU reports 2.5× with d=5 — achieving
1.5× (50 t/s) on 2-GPU would be a strong production win.

## Hardware

2× Quadro RTX 6000 (TU102, sm_75, 24 GiB each), CUDA 13.2,
`--split-mode graph --tensor-split 1,1`, 262K context.

Model: Qwen3.6 27B IQ4_XS with q4_0 Hadamard KV cache.
