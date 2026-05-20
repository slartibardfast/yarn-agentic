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

`PHASE_NSTREAM_KV_PERF.md` (carries from today's `PHASE_NSTREAM_KV` closure):

The 4D port disables graph reuse at `n_stream > 1` because stream-aware view offsets bake into the graph; reusing a graph across streams would read from the wrong slice. Every per-stream sub-batch rebuilds the graph (~2–3 ms each). On `llama-batched-bench` TG NP=8 that's a **-6.2 % regression** vs the pre-port 27.73 t/s baseline. The G3.h perf gate failed by that margin; user-selected override merged the bundle because all six correctness gates were green and the structural goal of the phase was delivered.

The next phase is per-stream graph cache: one `prev->graph` per `stream_id`, so each per-stream sub-batch reuses its own stream's graph. Adjacent K-shift / defrag / v_trans paths are guarded at `n_stream == 1` and need lifting for full multi-slot coverage. MLA (DeepSeek) stays out of scope.

## What this changes vs the 2026-05-17 framing

- **The race isn't kernel-level.** It's a scheduler problem: the kernel just exposed the underlying mixed-batch composition issue. Once batch composition is enforced single-stream, `mul_mat` sees a uniform shape per call and the GEMV-vs-GEMM accumulation question goes away.
- **The `PLAN_DETERMINISM_AUDIT.md` strict capture-harness approach was superseded** by the spec-led S1–S5 layer (Allium + TLA+ + property tests + NDJSON trace harness) which served the same role at one-tenth the token cost. That plan is now under `docs/archive/np-determinism/`.
- **Hadamard's "second effect"** observation (keeping cache values uniform enough that drift stays below argmax-flip margin) was a useful diagnostic but isn't load-bearing once batches are single-stream by construction.

## Where to start the next session

1. Read this file + `PHASE_NSTREAM_KV.md` (closure detail) + `PHASE_NSTREAM_KV_PERF.md` (next phase scope).
2. Read MEMORY.md entries from 2026-05-20 — the override decision, the byte-compat axis-order tradeoff, the G3.h root cause.
3. Begin the per-stream graph cache design lock (`PHASE_NSTREAM_KV_PERF.md` "Starting hypothesis" section). Open questions there are intentionally unresolved — measure before committing.

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
