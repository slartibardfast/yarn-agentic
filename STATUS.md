# Project status — Qwen 3.6 27B determinism on dual sm_75

Last updated: 2026-05-20

## The goal

Get a production multi-slot inference server that returns **byte-identical output at NP={1,2,4,8}** across the prompts a real user would type. The hardware is two Quadro RTX 6000s (TU102, sm_75); the software is a fork of llama.cpp called `ik_llama.cpp`.

Byte-identity matters because the alternative is "deterministic up to fp32-ULP noise" — which means argmax can flip, tokens can diverge, and an experiment that returns one answer at NP=1 returns a different answer at NP=2. For research reproducibility, regression debugging, and any system that stores or compares model responses, that's unacceptable.

We started thinking it was a kernel-level math problem. It turned out to be a scheduler problem masquerading as one.

## Where we are now (2026-05-20)

**The NP-determinism work is structurally closed on `production/2026-q2-next`** at submodule HEAD `16b608d1`. The path:

1. **Kernel-level audit landed four fixes** (Phase A/B custom Q4_0_AR16 MMQ/MMVQ, Phase C cuBLAS algo pinning, CY.F.17 stream_K disable, CY.F.18 scheduler `needs_sync` race). These are real, valid, shipped fixes — but none of them closed the production-state NP-determinism gap on their own.
2. **The bisection of 2026-05-17** found the residual failure was content-dependent (PASS–FAIL–PASS across prompt sizes), single-GPU-symmetric, and lived below the conditionals the audit was looking at. The "Today's bisection" notes in earlier git history record the falsified facts (mla_attn force-reset, ne[1]>32 cast dead, harness bash bug that masked half a session's probes).
3. **Bug C mechanism confirmed (2026-05-20).** `LLAMA_KV_CONCURRENT_TRACE` instrumentation showed the failing iteration had a **mixed prefill+decode batch**: slot 0's first decode token batched together with slot 1's full 210-token prefill. CUDA `mul_mat` picks GEMV (single-thread sequential accumulator) for `[d, 1]` and tiled GEMM (parallel partial sums + reduction tree) for `[d, N>1]`. Same math, bit-different accumulation order, ~1 ULP per element propagating to ~3e-3 absolute by layer 0. No kernel-level patch without rewriting all `mul_mat` to GEMV-order at every batch size — major perf loss, ruled out by the user.
4. **Policy-level closure landed first** as v1 PP-serialisation + decode-side prefill gate (`cef533ac` + follow-on). Prevents mixed batches from forming. Correct fix but loses TG-overlap (~6s wall-time penalty in `test-pp-serialization.sh`).
5. **Structural closure landed today** as `PHASE_NSTREAM_KV` — the upstream 4D per-stream KV layout port. K/V tensors become `[head_dim, kv_size_per_stream, n_head_kv, n_stream]`; each session owns its slice by construction; the server's `process_batch_tokens` splits multi-seq batches into per-stream sub-batches before any `llama_decode` so each call sees a single-stream batch. **Mixed batches cannot form by construction; the decode-side gate is removed.**

Six correctness gates green on the production 27B (single + multi-GPU NP-determinism, r5-probe-c4 0/20 without gate, DFlash multi-slot byte-identical, spec tests, PP serialisation above threshold). See `PHASE_NSTREAM_KV.md` for the full closure table.

## What's open — and why

`PHASE_NSTREAM_KV_PERF.md` (carries from `PHASE_NSTREAM_KV` closure, scoped 2026-05-20 after deep triangulation):

The 4D port disables graph reuse at `n_stream > 1` because stream-aware view offsets in `llm_build_kqv`'s K/V read views bake into the graph; reusing a graph across streams would read from the wrong slice. Every per-stream sub-batch rebuilds the graph. On `llama-batched-bench` TG NP=8 that's a **-6.2 % regression**. The G3.h perf gate failed by that margin; user-selected override merged the bundle because all six correctness gates were green and the structural goal was delivered.

The plan is now four-layered (Phase 0 prereqs → Tier 1/2 → Tier 3 → optional Tier 4):

- **Phase 0.A** — DFlash server CLI verify-on-post-fold. Wiring + profile already landed in `61a7e874` (2026-05-18). 2026-05-20 session FIXED two bugs (MAL cap drafter K/V OOM 21.5 GB → 85 MB; stage end-trim restore 8 % → 54 % acceptance). The earlier "T5 CONFIRMED" / "cb_eval root cause" diagnosis was REOPENED later that day after a libllama-level binding observational test (`ik_llama.cpp/tests/dflash-speculative/test-dflash-extract-observational.cpp`, held on disk uncommitted) showed byte-identical argmax across three shapes (single prefill / 64-tok autoregress / 160-tok verify-style 5-wide) with cb_eval armed vs disarmed. The previous matrix was correlation, not causation: ngram-simple ≡ spec-none ≠ DFlash, but the architectural diff between ngram-simple and DFlash is NOT only cb_eval — it is the entire DFlash pipeline. cb_eval install is now empirically exonerated. P0.A.3 root cause is OPEN; candidates are combine_features GEMM ordering, inject_kv_fused async sync, drafter_forward kernel state, common_speculative_draft position math, post-fold 4D KV ↔ drafter KV alias. Next experiment is an A/B on `examples/dflash-speculative-simple` with cb_eval force-disabled. P0.A.4 multi-slot SEGV remains deferred. Tier 2 does NOT block on P0.A.3/4.
- **Phase 0.B** — Radical Allium / TLA+ / test surface expansion. Existing S1–S5 covers what PHASE_NSTREAM_KV preserved (Bug C absence, per-stream allocator, mask isolation). Tier 2/3 introduce surface (CUDA-graph reuse semantics, unified-stream dispatch, MTP/DFlash composition under graph reuse) that is not yet bound by spec. Historical justification: tasks #37/38/39 (cudaMallocAsync NPC stochastic 1/8) were not catchable by S1–S5 because no spec existed for CUDA runtime state ordering.
- **Tier 1 / 2** — Patch attention-read view offsets per-stream in `update_cache_copies` (mirroring the K/V write CPY patching that already ships in N2.b); drop the `n_stream > 1` bailout in `can_reuse_graph`. Existing in-tree `cudaGraphExecUpdate` infra (Phase 36/37/38) carries the downstream patching. Recovers regression + 15-30 % on top.
- **Tier 3** — Unified-stream dispatch (one `llama_decode` per tick spanning N streams via the ne[3] axis). Uses the existing production PSKV per-slot FA kernel — no novel FA kernel port. Approaches vLLM's measured 154.77 t/s aggregate at NP=8.

Adjacent K-shift / defrag / v_trans paths are guarded at `n_stream == 1` and lifted by Tier 3. MLA (DeepSeek) stays out of scope.

Total scope: 290-510 k tokens phase-wide (Phase 0 ≈ 110-230 k depending on P0.A smoke outcome; Tier 2 ≈ 60-100 k; Tier 3 ≈ 120-180 k).

## What this changes vs the 2026-05-17 framing

- **The race isn't kernel-level.** It's a scheduler problem: the kernel just exposed the underlying mixed-batch composition issue. Once batch composition is enforced single-stream, `mul_mat` sees a uniform shape per call and the GEMV-vs-GEMM accumulation question goes away.
- **The `PLAN_DETERMINISM_AUDIT.md` strict capture-harness approach was superseded** by the spec-led S1–S5 layer (Allium + TLA+ + property tests + NDJSON trace harness) which served the same role at one-tenth the token cost. That plan is now under `docs/archive/np-determinism/`.
- **Hadamard's "second effect"** observation (keeping cache values uniform enough that drift stays below argmax-flip margin) was a useful diagnostic but isn't load-bearing once batches are single-stream by construction.

## Where to start the next session

1. Read this file + `PHASE_NSTREAM_KV.md` (closure detail) + `PHASE_NSTREAM_KV_PERF.md` (current scope).
2. Read MEMORY.md entries from 2026-05-20 — the override decision, the byte-compat axis-order tradeoff, the G3.h root cause, the Phase 0 prereq scoping entry.
3. Start Phase 0 — P0.A (DFlash server CLI fix) and P0.B (Allium / TLA+ / test surface expansion) can run in parallel. Tier 2 begins only after both close (GP0.A.a-d and GP0.B.a-e all GREEN).

Note: production `llama-server.service` is currently FAILED (stopped during gate runs 2026-05-20 ~09:59). Restart manually with `systemctl --user restart llama-server` after deciding whether to ship the post-fold submodule (currently selected) or pin earlier.

## Other live workstreams

These are independent of the NP-determinism work and not blocked by it:

- **DFlash speculative decoding** — Qwen 3.6 27B with bespoke sm_75 kernels. Kernel-pipeline closed at T1-T9; multi-slot orchestrator closed 2026-05-18; batched-pinned dispatch closed 2026-05-19 (both archived under `docs/archive/dflash/`).
- **MTP production** — Multi-token prediction for Qwen 3.6 27B, shipped Q2 2026.
- **Vulkan multi-GPU split-mode graph** — RDNA2 + Vega multi-GPU production stack, 22 phases of work landed.
- **TU102 specialization** — open kernel-ranking work (`PHASE_TU102_SPECIALIZATION.md`).
- **F.4.1 perf recovery** — open (`PHASE_PERF_F4_1.md`).
- **AsyncReduce** — planning (`PHASE_ASYNC_REDUCE.md`).

See the workstream-specific docs in the mdBook nav.
