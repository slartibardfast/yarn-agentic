---
name: DFlash T8 CLOSED — bench infra + measurement of record; throughput bounded by two named kernels at ~1% TU102 peak
description: T8 closed bench infrastructure for apples-to-apples spec-method comparison (none / mtp / dflash) with PPL-of-output, plus the canonical Phase 3 measurement on Qwen 3.6 27B. DFlash tg geomean 1.14 t/s = 3.4% of MTP, attributed to two specific kernels running at ~1% of dual-TU102 peak.
type: project
originSessionId: phase_dflash_t8_2026-05-13_14
---
DFlash T8 closed `[x]` on production/2026-q2-next 2026-05-14.

## What landed (infra)

`examples/llama-bench/llama-bench.cpp` extended with `--spec {none,mtp,dflash,...}` + `--draft N` + `--spec-model PATH` + `--prompt-file PATH` + `--ppl-of-output`. Bench-validated MTP wiring (+13.0% over the production memory baseline 33.23 t/s, within ±20% sanity band) confirms apples-to-apples comparison reads the same path production does. Smoke test in `tests/dflash-speculative/test-llama-bench-spec-init.cpp`.

## Phase 3 measurement of record

8 prompts × 3 spec methods × 3 reps at Qwen 3.6 27B INT4, ctx 4096:

|  | none | mtp draft=3 | dflash draft=4 |
|---|---:|---:|---:|
| tg t/s geomean | 29.31 | **33.86** | 1.139 |
| accept_rate geomean | — | 43.6% | **54.1%** |
| PPL of output geomean | 1.160 | 3.091 | **1.158** |

Data: `data/phase_dflash_t8/gate6-spec-*.json`, aggregated to `gate6-summary.json` via `aggregate-gate6.py`.

## Two findings the data binds

1. **DFlash output quality ≈ vanilla greedy.** PPL geomean 1.158 (DFlash) vs 1.160 (none) — within noise. MTP PPL geomean 3.091 sits ~3× higher (batch-shape effects in MTP's verify push the output away from the strict greedy-target trajectory). At 54% acceptance, DFlash's per-cycle output is *more* greedy-faithful than MTP's 43.6%. The T1–T7 byte-identity-to-vLLM kernel correctness work is what makes this happen.

2. **DFlash throughput is bounded by two named kernels.** Per Phase 2 nsys (`data/phase_dflash_t8/dflash-nsys-1cycle.{nsys-rep,sqlite}` dropped from HEAD 2026-05-14 — see history if needed) + source inspection:
   - `dflash_drafter_lm_head_kernel` (`ggml/src/ggml-cuda/dflash/dflash-drafter-lm-head.cu:37-64`): 41.3% of GPU time, 1228 ms/call. **600× off the TU102 BF16 memory-bandwidth ceiling** (~2 ms for the 1.2 GiB weight scan at 624 GB/s). Root cause: one CTA per row, scalar fp32 dot per thread over D_emb=5120, no tensor cores, no SMEM weight tile.
   - `gemm_row_x_col_kernel` (`ggml/src/ggml-cuda/dflash/dflash-drafter-forward.cu:121-141`): 34.0% of GPU time, 29 ms/call. Root cause: 5 CTAs on 72 SMs = 7% SM utilization, scalar fp32, no tensor cores.

   Both are reference impls that closed T3/T4's byte-identity bindings against scalar fp32 oracles — correct, deliberately un-tuned.

## TU102 + NVLINK optimization envelope (named, future work)

| Resource | Per-GPU | Dual w/ NVLINK | Relevant to |
|---|---:|---:|---|
| HBM bandwidth | 624 GB/s | 1.25 TB/s aggregate | lm_head BF16 weight scan |
| FP16 tensor core peak | 130.5 TFLOPs | 261 TFLOPs | GEMM, lm_head rewrites |
| NVLINK aggregate | — | ~100 GB/s | logits allreduce after V-shard |

lm_head path: V-shard 248320 across both GPUs, F16 tensor-core GEMV per-GPU (Turing `mma.sync.m16n8k8`), multi-CTA grid hitting ~all 72 SMs, NCCL allreduce 248k floats (~1 ms). Projected: lm_head from 1228 ms → ~2 ms (memory-bandwidth floor). GEMM rewrite follows same shape at smaller scale.

Possibly also: cooperative WMMA mega-kernel (`kernel-design.md §6.1`) — T7's source-read showed the implementation deviates from spec here (per-step `__global__` launches instead of `cudaLaunchCooperativeKernel` covering 5 layers + lm_head). Eliminates per-step launch overhead.

## Non-obvious wiring notes for future sessions

Documented in PHASE_DFLASH.md and the bench code, repeated here for memory:

- `cparams.mtp = true` must go on the **target context** (not just `cparams_dft`) for bench MTP to work. Plus `cparams.pooling_type = LLAMA_POOLING_TYPE_NONE`. Plus `mparams.mtp = true` to load NextN layers.
- Embeddings flag must stay TRUE permanently during the MTP spec loop. Qwen 3.6 + MTP emits BOTH logits and embeddings when nextn_predict_layers > 0 (`src/llama.cpp:5411,5427`); flipping it off between verify and draft breaks MTP recurrence. Per-cycle MTP re-seed needs the embedding at row `n_acc` of the verify decode (mirrors `server-context.cpp:4019-4030`).
- `llama_get_logits_ith(ctx, i)` takes a BATCH POSITION, not a "need-slot" counter — the PPL re-decode initially returned 0 PPL because of this indexing error.
- DFlash uses `cb_eval` extract hook with F16/F32 dtype split — see `feedback_cb_eval_dtype_split.md`.
- DFlash needs `llama_spec_ckpt_discard(ctx)` before any post-spec-loop long-batch decode (rep boundary, PPL second pass); otherwise the `save_per_step_ssm` flag stays on and prefill-shape data misroutes into the 5-sized per_step buffer (`ggml.c:5391` view assertion).
- No EOG exit in bench TG: bench measures throughput; PPL-of-output bounds quality separately.

## Artifacts kept

- `examples/llama-bench/llama-bench.cpp` (spec extension + PPL + DFLASH_TIMING env-gated chrono ledger)
- `tests/dflash-speculative/test-llama-bench-spec-init.cpp` (smoke test for 3 init paths)
- `common/perplexity.{h,cpp}` (extracted shared NLL/PPL kernel)
- `data/phase_dflash_t8/gate6-*.json`, `aggregate-gate6.py`, `cycle-timing-{dflash,mtp}.runlog`
- 8 ship-gate prompts at `data/phase_dflash_t8/prompts/p[0-7].txt`
- Env-gated `DFLASH_DIAG=1` (cb_eval + stage + trim) and `DFLASH_TIMING=1` (per-cycle chrono) in source for future regression triage.
