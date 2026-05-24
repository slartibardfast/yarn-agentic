---
name: project-t6-3-j-1m-ctx-ceiling
description: "production/2026-q2-next 2026-05-24: T6.3.j — bandwidth math kills 1M-ctx production target under the user-stated 100+ t/s constraint. Two MTP bugs fixed (a69f19de per-step gate; 711212a6 WARMUP _multi setter) — both stand. But Phase 2 overnight prefill at 1M+YaRN ran at 9.7 t/s. Bandwidth ceiling for Qwen3.6-27B hybrid (16/64 full-attn layers × Q4_0): 80 t/s peak at 1M, 32-38 t/s realistic. NVLink helps decode (AllReduce 26.5% → ~5%) but not long-ctx prefill (DRAM-bound). Recalibration: 262K native or 524K YaRN-factor-2 are the achievable parking destinations; 1M needs NVLink + Q4_0 tensor-core matmul (T7 territory)."
metadata:
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

Follows `[[project-t6-3-mtp-swap-validation-failed]]`. After fixing both MTP bugs surfaced by the first overnight, the second overnight (a69f19de + 711212a6 build) cleared Phase 1 boot smoke (128 tokens, 9.5s wall — fix works), but Phase 2 cold 1M prefill ran at 9.7 tokens/sec and hit the 1-hour client timeout repeatedly. This memory captures the resulting ceiling analysis and architectural recalibration.

## Two MTP bugs fixed (both stand)

| commit | location | symptom | fix |
|---|---|---|---|
| `a69f19de` | `src/llama-delta-net.cpp:73` | GGML_ASSERT view overflow at MTP prefill | gate `save_per_step_states` on `batch.n_tokens <= per_step_max_allocated`. Complements PHASE45 D10 multi-slot guard. |
| `711212a6` | `examples/server/server-context.cpp:5063` | SIGSEGV in libc memcpy at first decode after MTP setup (100% repro) | use `_multi` variant of hidden-state setter; the WARMUP path's `batch_mtp_hidden_state` is `n_toks × n_embd` floats, the single-row setter only stored `n_embd`. |

Verified by 5/5 successive requests with previously-crashing prompt at `profiles/qwen36-27b-x1-yarn-1m-mtp.sh`.

## The 9.7 t/s prefill observation

Phase 2 overnight: 853K-token War-and-Peace prefill at MTP+1M+YaRN. Server alive, GPU at 100% util, advancing position by 512 every 53 seconds = **9.7 tok/s**. Each 512-token chunk wrote + evicted a 152 MiB context checkpoint (`--ctx-checkpoints 64 --ctx-checkpoints-interval 512` defaults). Checkpoint overhead ~340 ms / chunk = 0.66 ms/token; rest is the actual forward cost.

## Bandwidth ceiling (Qwen3.6-27B, 2× sm_75, Q4_0+Hadamard KV)

Per GGUF: `qwen35.full_attention_interval = 4` → 16 of 64 layers full-attn, 48 DeltaNet (no growth with ctx).

Per-token KV bandwidth (full-attn only): `n_head_kv(4) × head_dim(128) × 2 (K+V) × 0.5 bytes × 16 layers = 8.4 KB × ctx`.

| ctx | DRAM peak ceiling | pre-NVLink real (40%) | post-NVLink real (45-50%) |
|---:|---:|---:|---:|
| 262K | 322 t/s | 130 t/s | 150 t/s |
| 524K | 156 t/s | 60 t/s | 75 t/s |
| 786K | 106 t/s | 42 t/s | 50 t/s |
| 1M | **80 t/s** | **32 t/s** | **38 t/s** |

NVLink (50 GB/s vs PCIe 12 GB/s) cuts small-message AllReduce ~10×. T6.2 measured AllReduce at 26.5% of GPU time at production NP=2 decode. Decode wins ~20% wall. Long-ctx prefill is DRAM-bound; NVLink contribution <1%.

## Production constraint: "1M is predicated on 100+ t/s"

User-locked 2026-05-24. The bandwidth math shows **100+ t/s at 1M is not on the table for this hardware** — even with NVLink, peak 80 t/s ceiling falls below 100, and realistic 32-38 t/s. The 1M target requires either:
- Q4_0 tensor-core matmul path (T6.2.b candidate; weeks of kernel work, T7 territory)
- Different hardware (Ampere+ with Marlin int4; or B200-class with 8 TB/s memory)

## Recalibrated parking ladder

| ctx | YaRN factor | 100+ t/s feasible? | architecture impact |
|---:|---:|---|---|
| **262144 (native)** | none | Yes, comfortable (130 t/s pre-NVLink) | Same ctx as production; single-slot+queue+cache+MTP architecture is the gain |
| **524288** | 2.0 | Marginal pre-NVLink (60 t/s), likely post-NVLink (75 t/s) | 2× per-slot ctx vs production; still YaRN territory |
| 1048576 | 4.0 | **No on current hardware** | Gated on T7 tensor-core matmul + NVLink |

## Honest verdict

The "1M Yarn + MTP single-slot + transparent queue + cache" production candidate **as originally scoped is not achievable** on this hardware under the 100+ t/s constraint. The architectural intent is sound; the **1M** part isn't.

Recommended next move: recalibrate to 524K (YaRN factor=2.0) and measure prefill with the bug-fixed build. Validate against the 100+ t/s gate. If 524K passes, that's the parking target. If it doesn't, fall back to 262K native.

## T6.3.j subtasks opened (named, not deferred-as-cover)

- **T6.3.k** — measure prefill t/s at 262K and 524K with the (a69f19de + 711212a6)-fixed build. ubatch sweep at chosen target.
- **T6.3.l** — post-NVLink re-measure (2026-05-24 install pending per nvidia-smi nvlink --status showing "inActive").
- **T6.3.m** — characterise prefill bottleneck at long ctx (nsys at ~350K-position ubatch). Identify dominant kernels.
- **T6.3.n** — investigate ctx-checkpoint overhead at long ctx. Tunable via `--ctx-checkpoints-interval`; default 512 is wrong for long-prompt prefill.

## Discipline notes

CLAUDE.md §1 (Think Before Coding): the user's "100+ t/s" constraint was load-bearing and should have been surfaced before scoping 1M. Surface assumptions earlier next time.

CLAUDE.md §8 (estimate in tokens, not days): "1M ctx production candidate" carried implicit perf assumptions that weren't checked against bandwidth math at design time. Cost: spent the session validating a non-viable target. The two bug fixes that landed are real value, but the architectural completion was blocked by a constraint we could have anchored at the start.

CLAUDE.md §4 (no follow-up cover): the 1M target is named as not achievable on current hardware. T6.3.k-n capture the remaining work toward an achievable parking destination.

Related: `[[project-t6-3-mtp-swap-validation-failed]]` (the first overnight failure), `[[project-t6-3-dflash-deep-dive-closed]]` (T6.3 verdict that motivated parking), `[[feedback-anchor-to-measured-baselines]]` (the bandwidth math discipline — should have anchored 100+ t/s expectation against measured T6.2 baseline at design time, not after running the overnight).
