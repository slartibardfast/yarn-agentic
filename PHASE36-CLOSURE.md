# Phase 36 Closure

## Outcome

| Component | Status | Production-context measurement |
|---|---|---|
| Step 0 — Profile draft cycle | **Closed** | Tagged decode_op + per-step component timing committed; baseline measured at d=0/1/3/5 across 4K, 64K, 256K. |
| Step 1 — Fused multi-draft cgraph | **Builds, runs, produces wrong tokens at d≥3** | Issue G unresolved; +0% at d=1 (gate is n_draft>1, gate doesn't fire), -36% at d=3 (acceptance collapses to 18% vs per-step 58%). |
| Step 2 — Async dual-stream pipeline | **Blocked on Step 1** | Step 2.1 (per-device draft streams) committed but unused. Step 2.2/2.4 (sched_draft + pipeline cycle) require a working fused dispatch. |
| Step 3 — Eliminate UPDATE_ACCEPTED | **Closed and validated** | At d=1 with `LLAMA_MTP_INLINE_KV=1`: 34.93 t/s vs nomtp 31.10 t/s = **+12% MTP-vs-nomtp**, the largest measured Phase 36 win. |
| Step 4 — Device-resident hidden-state relay | **Detector landed; D2D path disabled** | The host-stage fallback is in place; the same-backend partial D2D path defers to set_draft_input_hidden_state until a row-selection contract is locked down. |
| Step 5 — KQ_mask bucketing | **Comparator landed** | `can_reuse_graph` condition 7 buckets to multiples of 64. Has zero measured benefit in the X02 256K workload (r7=0% of misses) but no regression either. |
| Step 6 — MTP head F16 | **Pre-existing in production GGUF** | `blk.64.nextn.eh_proj.weight` is 100 MiB = 5120×10240×2 (F16). No further leverage available from this step. |

## Key shipping result

**Step 3 (`LLAMA_MTP_INLINE_KV=1`)** is the validated win. At production 256K context with the X02 1.3K-token agentic prompt and greedy sampling:

```
nomtp baseline:        31.10 t/s
d=1 + inline-KV hook:  34.93 t/s   (+12.3% MTP-vs-nomtp)
d=3 + inline-KV hook:  29.53 t/s   (-5%, MTP regresses below nomtp)
d=5 + inline-KV hook:  29.59 t/s   (-5%, same)
```

Production default should keep `LLAMA_MTP_INLINE_KV` unset until soak validates correctness across longer agentic workloads. Once flipped on, the d=1 path becomes the production MTP configuration.

## What was built (the synthesis)

End-state architecture diagram:

```
┌─ verify decode (mtp_op=NONE, n=N+1) ─────────────────────────────┐
│  main 64 layers → h_pre_norm tagged                  [Step 3.1]  │
│  + kv-only MTP for ALL N+1 positions                 [Step 3.2]  │
│  + lm_head → verify logits                                       │
└──────────────────────────────────────────────────────────────────┘
        ↓ accept decision
        ↓ seq_rm trims rejected on main+MTP            [Step 3.3]
        ↓
┌─ fused draft (mtp_op=DRAFT_GEN_FUSED, n=M) ──────────────────────┐
│  inp_mtp_states ← host-stage of h_pre_norm           [Step 4]    │
│  step 0..M-1: token + residual → attn (write KV)     [Step 1]    │
│  per-device argmax + in-graph reduction              [Step 1]    │
│  bucketed n_kv comparator for graph reuse            [Step 5]    │
│  draft stream (low-priority, lazy-init)              [Step 2.1]  │
│  output: argmax_0 .. argmax_{M-1}                                │
└──────────────────────────────────────────────────────────────────┘
        ↓ verify cycle k+1 reads MTP KV
```

Components landed (env-gated for A/B):
- `LLAMA_MTP_INLINE_KV=1`: enables Steps 3.1+3.2+3.3.
- `LLAMA_MTP_FUSED=1`: enables Step 1's fused draft path. *Currently produces low-acceptance output due to Issue G — do not enable in production.*
- Always-on: Step 5's `can_reuse_graph` bucketing, Step 4's host-stage Issue F detector + fallback, Step 2.1's draft streams.

The `n_tokens` keystone refactor (driving `n_tokens` from `cur->ne[1]` / `input->ne[1]` at the start of `build_std_attention`, `llm_build_mul_mat_qkv_gated`, `llm_build_mul_mat_qkv`) is what lets the same primitive serve verify, fused-chain, and kv-only hook without forking. That refactor is committed and benign for all existing call sites.

## Issue F (original blocker — split-mode lm_head + on-device argmax)

**What it is:** for the (untested-here) split-mode-graph lm_head case, `ggml_concat` of per-device matmul outputs followed by `ggml_argmax` was suspected of returning out-of-vocab indices because the argmax kernel reads only its own device's slice.

**Status:** detector + fallback path in place (see `prepare_mtp_graph_inputs` + the per-device argmax + reduction path in `build_qwen35_mtp_fused`). The production model has `model.output->extra == nullptr` (single-device output even with split-mode-graph), so the per-device path is dormant.

**Granular test scaffolds:** `tests/mtp-fused/test-mtp-fused-split-argmax.cpp` documents the expected harness; allocator preconditions block direct execution without the model-loader's split-buft path. Listed in CMakeLists; will turn green once a model-free split harness exists.

## Issue G (the actual blocker for fused d≥3)

**What it is:** the production AutoRound GGUF has both `hparams.n_vocab` and `model.vocab.n_tokens()` set to **248320** (the matmul-aligned padded width), but the *underlying tokenizer's trained vocabulary* is **152064** (Qwen 3.6's real range). The padded range [152064, 248320) maps to "Invalid token" when fed back through the tokenizer.

**Why per-step survives:** per-step's argmax goes through `common_sampler_sample_speculative` which sits behind the sampler chain (top_k/top_p in production; the speculative-cache fast path in MTP). The chain implicitly suppresses padded entries because their trained-logit magnitudes are below real-vocab tops on calibrated workloads.

**Why fused breaks:** fused uses raw on-device `ggml_argmax` over the full 248320-wide logits with no sampler chain. Whenever quantization noise or chain drift produces a high-magnitude logit at a padded index, fused picks it; the resulting token id is unreachable text and the chain feeds a corrupt residual into the next step.

**Evidence:**

```
[mtp-fused] step=0 tok=248045 prob=0.2698  -- padded; in-vocab=hparams.n_vocab so detector misses
[mtp-fused] step=0 tok=409 prob=0.1533     -- real
```

~25% of fused step 0 outputs are in the padded range; cascades to 18% accept at d=3, 13% at d=5.

**The vocab-slice attempt** (slice logits to `model.vocab.n_tokens()`) is in the code but is a no-op for this GGUF because both fields equal 248320. The "real" boundary lives in the tokenizer's trained vocab metadata, not in either of the fields the runtime reads.

## Fix paths for Issue G (next-session work)

In rough order of effort:

1. **Per-step sampling for the chain tokens.** D2H each step's logits, run the same `common_sampler_sample_speculative` chain that per-step uses, set the next step's `inp_tokens` host-side. Cost: ~5MB D2H per fused decode (5 steps × 248320 × 4B), negligible at PCIe 16GB/s. Defeats some "fused = single compute" benefit (need to interleave host ops), but produces correct tokens by construction.
2. **In-graph sampler-equivalent filtering.** Build top_k or top_p (or just a "mask out padded indices" op) into the graph before argmax. Requires either a custom CUDA kernel for top_k or sourcing the real-vocab boundary from the tokenizer model. Keeps single-compute.
3. **GGUF-side fix.** Re-quantize with the lm_head shrunk to the real vocab dim, or pad the lm_head weights at [152064, 248320) with -INF logits so they never win argmax. Ships once; works for all on-device argmax paths.

Path 3 is the cleanest if the GGUF can be regenerated.

## Tests committed

| File | Status | Drives |
|---|---|---|
| `tests/mtp-fused/test-mtp-fused-symbols.cpp` | GREEN | API surface (Step 1.1) |
| `tests/mtp-fused/test-mtp-fused-split-argmax.cpp` | RED | Issue F regression check (split lm_head + argmax) |
| `tests/mtp-fused/test-mtp-fused-chain-residual.cpp` | RED stub | Fused vs per-step token parity at step k |
| `tests/mtp-ubatch-hook/test-hook-acceptance-vs-update-accepted.cpp` | RED stub | Step 3 hook accepted-token sequence parity |
| `tests/mtp-fused/test-mtp-fused-{step-count-bound,single-compute,argmax-correctness,determinism,kv-coverage,kv-chop-rewrite,prob-populated}.cpp` | RED scaffolding (b34e661b) | Fused contract invariants |
| `tests/mtp-ubatch-hook/test-hook-{tag-tensor,fires-once,no-secondary-decode,cross-ubatch-pairing,idempotent-chop}.cpp` | RED scaffolding (b34e661b) | Hook contract invariants |
| `tests/mtp-verify-accept/...` | RED scaffolding (58009a77) | Verify-accept decision contract |

The RED scaffolding documents the contracts the implementation must respect; many require a test fixture that can drive `LLAMA_MTP_INLINE_KV` per-process and compare token sequences across ctx lifetimes. That fixture is the next major test-infrastructure task.

## Cumulative measurement table (X02 256K, greedy)

| Config | nomtp | d=1 | d=3 | d=5 |
|---|---:|---:|---:|---:|
| Baseline (no env) | 31.10 | 32.04 | 28.78 | 28.77 |
| `LLAMA_MTP_INLINE_KV=1` (Step 3) | 31.11 | **34.93** | 29.53 | 29.59 |
| `LLAMA_MTP_FUSED=1 + LLAMA_MTP_INLINE_KV=1` | 31.16 | 34.86 | 20.85 | 17.83 |

**Production default:** both env vars unset. The Step 3 hook can flip to default-on once a soak validates the d=1 win on real chat traffic. Fused stays gated until Issue G is resolved.

## Plan vs reality

The original plan projected ~70 t/s ceiling at production context (2.2×). Realistic measured ceiling with the synthesized stack:

- Step 3 alone gives the +12% (d=1) baseline win.
- Steps 1+2 unblock another +14% projected, *contingent on Issue G fix*.
- Step 6 was already done.

Without Issue G fixed, Phase 36 closes at +12% (d=1) — a real but smaller win than the plan's headline. The d=5 path is structurally close to d=1 in this workload because the early-exit on `prob < p_min` clamps effective draft depth to 1.5–1.8.

The work is *unblockable* from this branch — the keystone n_tokens refactor, the synthesized graph builders, the per-device argmax + reduction infrastructure, the bucketed comparator, the host-stage relay, and the test scaffolds are all committed. The next session that picks this up has all six steps' code paths already wired; only Issue G's fix and Step 2.2/2.4's pipeline-cycle plumbing remain.

---

## Superseded by PHASE37 (2026-05-07)

This closure document recorded the state at the moment Phase 36 was first marked done. Phase 37 then reopened it: the binding claim "fused beats per-step at default settings" did not hold against measurement at production context. PHASE37.md captures the full reopen → repair → recalibrate cycle and supersedes the closure characterisation above. The work above (Steps 0–6) is still landed; Phase 37 added four bug fixes (vocab emit, KV cell offset, seed copy-on-set, seed source context) and four schedule items (#3a F32 attention precision, #3b KV cpy dependency anchor, #4 adaptive chain depth, #5 fused graph reuse). Phase 37 #2 (pipelining) closed as infrastructure-only when measurement showed the projected lift was an order of magnitude too high.

**The recalibrated binding claim, as of Phase 37 closure**: at deployed settings (`LLAMA_MTP_FUSED=1 LLAMA_MTP_INLINE_KV=1 LLAMA_MTP_CHAIN_MIN_PROB=0.5`), the fused chain achieves effective output **parity** with per-step at d=3 on Quadro RTX 6000 sm_75 — within ±5% noise on `effective_output_ratio = accept_ratio × tg_ratio`. **Parity, not beat.** The dependency-bounded ceiling on this hardware is at unity; the over-aspirational tg thresholds in the original plan reflected projections that did not survive measurement.

The fused implementation's ship value is in optionality, not throughput: lower verify-side overhead at long context, simpler integration with KV scheduling, simpler control flow for future graph-batching across slots.

Read PHASE37.md for the full re-open narrative, the four bug fixes, the schedule progress, and the Path 3 measurement-driven recalibration. Read `tests/mtp-fused/gate.yaml` for the binding gate that this closure now points at.

The earlier closure text above is preserved as historical record (per CLAUDE.md §6 append-only on facts). The original "fused beats per-step" framing was wrong; this superseding section is the correction.
