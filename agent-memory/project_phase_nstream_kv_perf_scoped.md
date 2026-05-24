---
name: phase-nstream-kv-perf-scoped
description: "PHASE_NSTREAM_KV_PERF design-locked 2026-05-20 as a four-layer phase: Phase 0.A (DFlash CLI fix) + Phase 0.B (radical Allium/TLA+/test surface expansion) MUST precede Tier 2 (per-stream attention-read-view patching via existing in-tree cudaGraphExecUpdate) → Tier 3 (unified-stream dispatch via existing PSKV per-slot FA kernel). Aims at vLLM's measured 154.77 t/s NP=8 ceiling on same hardware. Total 305-490k tokens."
metadata:
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

PHASE_NSTREAM_KV_PERF is scoped as a four-layer phase on
`production/2026-q2-next` (submodule HEAD `16b608d1`).

**Why scope changed from the original "per-stream graph cache" stub:**
Two parallel deep-research agents + triangulation against existing
in-tree work revealed the original hypothesis was under-ambitious.
ik_llama.cpp's downstream `cudaGraphExecUpdate` machinery (Phase
36/37/38, `ggml-cuda.cu:4500-4830`) already supports per-call
patching of `src_address`/`ne`/`nb` and explicitly tolerates
`src_address` change for `GGML_OP_VIEW` + `GGML_OP_CPY` nodes.
`update_cache_copies()` (`src/llama.cpp:630`) already patches K/V
*write* CPY view offsets per-stream (PHASE_NSTREAM_KV_4D N2.b). The
only thing missing is per-stream patching of K/V *read* views in
`llm_build_kqv` — Tier 2 is a mechanical extension of N2.b, not
novel CUDA-graph engineering. The PSKV per-slot FA kernel
(`GGML_OP_FLASH_ATTN_EXT_PER_SLOT_KV`) is already production and
NPC-verified — Tier 3's "ragged FA on sm_75" is the kernel we
already wrote, just routed through unified dispatch. vLLM's 4.75×
at NP=8 was MEASURED on our hardware 2026-05-12
(`data/gate0-np1-np8.json`); Tier 3 aims to approach it.

**Phase 0 prereqs (must close before Tier 2 work begins):**

- **P0.A — DFlash server CLI**. Cascade-discovered four-layer scope
  during 2026-05-20 verify-on-post-fold session. Two fixes LANDED
  (submodule production/2026-q2-next), two issues DEFERRED:

  - **P0.A.1 LANDED** — drafter K/V VRAM cap. `seq_len_cap = ctx_tgt
    n_ctx` allocated 21.5 GB on production --ctx-size 524288
    --parallel 2 → OOM at boot. Fixed by capping at MAL_max =
    swa_window + block_size + 16 (= 2080) → 85 MB. 4 SWA + 1 full-attn
    drafter layers all bounded by this cap; full-attn becomes
    effectively SWA-bounded as an explicit trade-off.

  - **P0.A.2 LANDED** — stage_target_hiddens end-trim restore. MAL-cap
    commit accidentally dropped `buf.resize(mal_anchors * D_emb)`;
    stale rejected-draft hiddens accumulated; acceptance fell to 8 %.
    Restoring end-trim → 54 % at temp=0.

  - **P0.A.3 DEFERRED** — DFlash output diverges from spec-none
    baseline at the first multi-token verify-batch decode. Same
    prompt + seed + temp produces "Here's a thinking process..."
    (spec-none) vs " - The **UserUser**...quick quick quick..."
    (DFlash). Survives Q4_0/Hadamard/temp variation; not the
    GEMV-vs-GEMM ULP theory. MTP on same target produces clean
    output → DFlash-specific bug. Six theories T1-T6 captured in
    PHASE_NSTREAM_KV_PERF.md with ASCII illustrations:
      T1 KQ-mask off-by-one in multi-token verify batch
      T2 buf-row vs cache_tokens cardinality mismatch
      T3 cb_eval rejected-position residual (off-by-one trim)
      T4 drafter K/V RoPE position semantics
      T5 cb_eval sync breaking async K/V writes
      T6 cudaGraphExecUpdate stale src_address
    Estimated 100-300k dedicated phase to root-cause; runs in parallel
    to Tier 2.

  - **P0.A.4 DEFERRED** — multi-slot SEGV (`batch.logits[5] != true`).
    Likely subsumed by P0.A.3 root cause.

  **Tier 2 entry condition RELAXED.** P0.A.3/4 do NOT block Tier 2.
  Production runs MTP `--draft 3` at np=1, not DFlash CLI. Tier 2's
  gate GP3.n binds on MTP NP=1 production smoke; DFlash gates GP3.e
  bind at libllama layer only (existing test-dflash-np-multislot
  harness passes).
- **P0.B — Spec/TLA+/test surface expansion** (105-170k tokens).
  5 new Allium + 3 new TLA+ + 5 property tests + trace-harness
  extension. Targets surface S1-S5 doesn't cover: CUDA-graph
  reuse semantics, unified-stream dispatch, MTP × n_stream
  composition, DFlash × unified-verify composition, server-CLI
  contracts, warm-up determinism. Historical justification:
  tasks #37/38/39 (cudaMallocAsync NPC stochastic 1/8) were not
  catchable by S1-S5 because no spec existed for CUDA runtime
  state ordering. Five gates GP0.B.a-e.

P0.A and P0.B run in parallel. Tier 2 begins only when BOTH close.

**Tier 2 (post-prereqs, 60-100k tokens):** extend
`update_cache_copies` to patch K/V read view offsets per-stream
(mirror N2.b CPY patching); drop `n_stream > 1` bailout at
`src/llama.cpp:616`. Six implementation cards T2.a-g. Nine gates
GP3.a-h + GP3.n (MTP NP=1 production smoke).

**Tier 3 (post-Tier-2, 120-180k tokens):** server-side fusion in
`process_batch_tokens`; KQ-mask shape `[n_kv, n_tokens/n_stream,
1, n_stream]`; verify FA dispatch routes through PSKV; non-FA
shape-invariance audit; drop n_stream==1 guards on
K-shift/defrag/v_trans. Five gates GP3.i-m.

**Workload coverage explicit per gate:**
- Standard Qwen 3.6 27B vanilla NP=8: GP3.a-c, f-h
- Qwen 3.6 27B MTP NP=1 (current production): GP3.n
- Qwen 3.6 27B MTP NP>1 (future): GP3.m
- DFlash multi-slot: GP3.e (full test suite), GP3.k (kernel
  NPC at ne[1]>1), GP3.l (orchestrator symmetric+asymmetric
  under unified verify)

**Hardware ground truth (probed 2026-05-20):** no NVLink bridge
installed; GPU0-GPU1 PHB-connected via PCIe Gen3 (~13 GB/s).
Closest published fabric analogue: ik_llama.cpp PR #1080 on
4× RTX 3090 PCIe — Llama-3-70B Q4_0 at ~50 t/s gen on split-mode
graph. MoE all-reduce volume scales with active params (3B) not
total params (35B) — graph-split remains the right multi-GPU
mode.

**vLLM pivot ruled out on evidence:** weight format
incompatibility (Q4_0+Hadamard+Q4_0 KV uniquely SoTA on sm_75 —
Marlin-AWQ comparable not better, QuaRot PR #15162 closed); FP8
KV impossible on TU102; FA2/FA3 don't backport; NVFP4 produces
"meaningless outputs" on Turing (vLLM #33461); vLLM Gemma4 has
zero working attention backends on sm_75 (#38918); GGUF path
ran 8.7 t/s on A100 for Llama-3.1 70B (#8669). ik_llama.cpp's
split-mode-graph already beats mainline 33% TG / 6-9× PP on our
weight class — kernels SoTA, only dispatch behind.

**How to apply:** when resuming PHASE_NSTREAM_KV_PERF work, read
[`PHASE_NSTREAM_KV_PERF.md`](/home/llm/yarn-agentic/PHASE_NSTREAM_KV_PERF.md)
+ [`STATUS.md`](/home/llm/yarn-agentic/STATUS.md) +
[`PLAN.md`](/home/llm/yarn-agentic/PLAN.md). Start with Phase 0;
do NOT skip to Tier 2. Per `feedback_oneshot_then_evaluate`, no
partial intermediate landings — each tier is one coherent bundle
evaluated against its gates.

**Related:** [[project_continuous_batching_vs_perslot_dispatch]]
(4.75× anchor measured), [[project_pskv_ilp_recovery_landed]]
(NPC-preserving perf pattern), [[project_fattn_per_slot_kv_p2_landed_kernel_only]]
(PSKV kernel that Tier 3 leverages),
[[feedback_n_stream_byte_compat_tradeoff]] (axis order locked),
[[feedback_bake_measurement_env_gates]] (no LLAMA_*_ENABLE knobs),
[[feedback_drafter_forward_n_slots_cap]] (DFlash kernel
bind-time vs dispatch-time stride distinction),
[[project_dflash_multislot_phase5_landed]] (DFlash CLI gap that
P0.A fixes), [[project_dflash_multislot_phase6_landed]] (test
harness GP3.e binds).

**Parent commits this entry:** `0adfe9f` (initial Tier 2 lock),
`07bb7b9` (triangulation against in-tree), `14a60f6` (workload
coverage), `d9fb279` (Phase 0 prereqs added), and the doc-refresh
commit (PLAN/STATUS/README/MEMORY/SUMMARY).
