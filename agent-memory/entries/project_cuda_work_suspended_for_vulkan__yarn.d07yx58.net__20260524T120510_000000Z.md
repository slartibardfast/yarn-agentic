---
name: project-cuda-work-suspended-for-vulkan
description: "production/2026-q2-next 2026-05-24: CUDA-backend work on Qwen 3.6 27B production suspended at session close; switching to Vulkan backend on same host. Three submodule bug fixes ship: 3ee7816f (T6.6 SEGV — n_layer truncation), a69f19de (delta_net per-step save gate), 711212a6 (server-context WARMUP _multi setter). Production stable on DFlash. CUDA_HANDOFF.md committed to repo as peer-pickup synthesis; covers all CUDA findings including what carries to Vulkan (bandwidth ceilings, DFlash 27B drafter quality, the three bug fixes — all backend-agnostic) vs what doesn't (NCCL AllReduce 26.5%, mul_mat_q_split_k shared-mem analysis, PSKV singlewarp, nsys/ncu attribution data — all CUDA-specific)."
metadata:
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

Closes the CUDA-backend chapter on `production/2026-q2-next`. Vulkan transition follows; this memory anchors what was learned + what to re-verify post-transition.

## Session-close state under CUDA

- **Parent**: `production/2026-q2-next` at `e0f2090` (CUDA_HANDOFF.md commit)
- **ik_llama.cpp submodule**: `711212a6` (server-context WARMUP _multi fix)
- **Production runs**: `profiles/qwen36-27b-x2-dflash.sh` (DFlash, --parallel 2, ctx 524288, Q4_0+Hadamard KV)
- **NPC contract**: bound at submodule via `scripts/verify-production-determinism.sh`; PASS at NP={1,2,4,8} last verified

## Three submodule bug fixes that ship (carry to Vulkan)

All three are in backend-agnostic code (graph builder + server-context, not CUDA kernels). They should apply unchanged under Vulkan.

| commit | location | bug |
|---|---|---|
| `3ee7816f` | `src/llama-build-context.cpp:332,539` | `kv_cache_init`'s loop truncates to `hparams.n_layer - nextn_predict_layers` on `!model.mtp` path (= 64 for Qwen3.6 27B); `llm_build_context::n_layer = 65`; K-shift + defrag loops read past `k_l` vector end. Under DFlash heap layout: stable 0x1ea30 garbage past end → survived nullptr skip → SEGV on `->extra`. Fix: bound by `std::min(n_layer, k_l.size())`. |
| `a69f19de` | `src/llama-delta-net.cpp:73` | `save_per_step_states = save_per_step_ssm && batch.n_tokens > 1`; per-step buffers sized at spec-ckpt-init for `max_tokens = drafted.size() + 1 = 2`; prefill ubatches far exceed that → `ggml_view_2d` overflow at `build_layer_attn_linear_core` line 631 → GGML_ABORT. Fix: gate on `batch.n_tokens <= per_step_max_allocated`. Complements PHASE45 D10 multi-slot guard. |
| `711212a6` | `examples/server/server-context.cpp:5063` | WARMUP path called single-row `llama_set_draft_input_hidden_state`, but dst tensor `inp_mtp_states` shape is `(n_embd, n_tokens)`; `prepare_mtp_graph_inputs` memcpy'd `n_embd × n_tokens × sizeof(float)` bytes from `n_embd`-only buffer → SIGSEGV in libc AVX2 `vmovdqu`. Fix: use `_multi` variant. |

## CUDA-specific findings (do NOT carry to Vulkan; re-derive there)

- T6.2 nsys: `mul_mat_q_split_k<Q4_0>` 31% of decode time; NCCL AllReduce 26.5%; PSKV singlewarp 3.2%
- T6.2.b ncu: dominant matmul kernel DRAM 44.3%, compute 17.5%, occupancy-bound at 25% (shared-mem 40-45 KiB/block of 64 KiB cap)
- T6.3 axis 4 nsys (under DFlash): `mul_mat_f16_pinned_kernel_wmma` (drafter forward) 17.8% — the new dominant cost when DFlash is on
- PSKV singlewarp ILP recovery (T3.5): TG +2.95%, PP +9.17%, ncu per-CTA −32.7% on sm_75. CUDA-only kernel.
- ALGO0 cuBLAS algo pin for cross-NP byte-identity. CUDA-only.
- Bandwidth ceilings derived against TU102 DRAM 672 GB/s peak — hardware-level, **DO carry**. See `[[project-t6-3-j-1m-ctx-ceiling]]`.

## Workload-level findings (DO carry; model + workload are backend-independent)

- DFlash net-negative at every measured axis on Qwen3.6-27B at gate0 mixed-prompt workload (T6.3 4-axis deep-dive). Independent confirmation from Paterson (RTX 3090 Ti, single GPU, sm_86) + Allen Kuo Medium community reports. The 27B-drafter pair has known-low acceptance.
- Per-prompt DFlash acceptance is content-dominated: range 0.392 (King Lear prose) → 0.808 (haiku), mean 0.529.
- 1M ctx + MTP + YaRN architecture is bandwidth-infeasible at 100+ t/s on this hardware: peak 80 t/s, realistic 32-38 t/s.
- vLLM measured DFlash too: 154 → 35 t/s = 4.4× penalty (worse than ik_llama's 1.85×). DFlash hurts vLLM MORE.

## Open subtasks that pause under CUDA but resume under Vulkan with re-verification

All bandwidth-math-derived recommendations still apply but the measured numbers need Vulkan re-derivation.

- **T6.3.b** — DFlash post-NVLink re-measure (was scheduled for 2026-05-24; NVLink not yet active per session-close `nvidia-smi nvlink --status`)
- **T6.3.c** — DFlash on bench-t3.8-m3-shape upper-bound acceptance
- **T6.3.k** — prefill t/s at 262K + 524K with the three-fix build. UNDER VULKAN this becomes the load-bearing first measurement. ubatch sweep at chosen target. Decides production parking destination.
- **T6.3.l** — post-NVLink T6.2 + T6.3.k re-measure. Under Vulkan, NVLink benefit may be different; AllReduce is NCCL-only.
- **T6.3.m** — long-ctx prefill characterisation. Vulkan profiling needs different tools (RenderDoc / Nsight Graphics / radv perfetto), not nsys.
- **T6.3.n** — `--ctx-checkpoints-interval` overhead. Backend-agnostic checkpoint logic; investigation carries.
- T6.4-T6.10 — pre-existing Tier 6 follow-ons.

## Vulkan workstream entry suggestion

Per `CUDA_HANDOFF.md` §11:

1. Build with `-DGGML_VULKAN=ON`; smoke `llama-server` at small model first
2. NPC contract under Vulkan: single-GPU first, then multi-GPU
3. Multi-GPU cross-sync design (NCCL is CUDA-only; Vulkan equivalent needs explicit choice)
4. Re-derive T6.1-equivalent matrix to confirm DFlash net-negative direction holds
5. Re-derive T6.2-equivalent kernel attribution with Vulkan profilers
6. Don't forecast — measure. The 1M-ctx ceiling may rise or fall under Vulkan.

## Discipline notes for next session

- The auto-memory `MEMORY.md` index is at 128 entries; **129 files in the dir**. Audit at session close: 1:1 file ↔ entry, 0 orphans, 3 explicit SUPERSEDED entries preserved as audit records.
- The repo-level `MEMORY.md` (489 KB at repo root) is a different, complementary log of project decisions. Don't merge them.
- `CUDA_HANDOFF.md` at repo root is the peer-pickup synthesis; trust it as the navigator but trust PHASE docs + memory entries + bench data when they disagree.
- `TRANSFER.md` exists for host migration mechanics. Moot for the current Vulkan-on-same-host switch but still valid for any future host change.

Related: `[[project-t6-3-j-1m-ctx-ceiling]]` (bandwidth math), `[[project-t6-3-mtp-swap-validation-failed]]` (the parking attempt history), `[[project-t6-3-dflash-deep-dive-closed]]` (the T6.3 verdict that motivated parking), `[[project-vulkan-bi-mmv-root-cause]]` (prior Vulkan investigation memory — may be relevant for the upcoming backend switch).
