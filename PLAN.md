# PLAN — active workstream

Per the README convention, the **single currently-active workstream** lives in this file. Closed workstreams move to `docs/archive/<topic>/`. The full design + implementation cards for the active phase live in its own `PHASE_*.md`.

## Active phase: `PHASE_NSTREAM_KV_PERF.md`

**Goal**: Recover the -6.2 % TG NP=8 regression carried over from PHASE_NSTREAM_KV closure (2026-05-20) *and* unlock the dispatch-layer ceiling. vLLM's measured 154.77 t/s aggregate at NP=8 on the same hardware (2026-05-12) anchors the ceiling at ~4.75× over current production.

**Structure** (four-layered — full detail in [`PHASE_NSTREAM_KV_PERF.md`](PHASE_NSTREAM_KV_PERF.md)):

- **Phase 0.A** — DFlash server CLI verify-on-post-fold. Wiring + production profile already landed in `61a7e874` (2026-05-18). Verification surfaced two real bugs that LANDED as fixes (P0.A.1 MAL cap, P0.A.2 stage end-trim) plus **two open issues that needed dedicated diagnostic scope**: P0.A.3 DFlash output divergence from spec-none baseline (target's logits in multi-token verify batch produce different argmax than single-token spec-none decode — survives Q4_0/Hadamard/temp variation; six testable theories T1-T6 in PHASE doc), and P0.A.4 multi-slot SEGV (`batch.logits[5] != true`, likely subsumed by P0.A.3 root cause). **Tier 2 does NOT block on P0.A.3/4** — production runs MTP not DFlash; Tier 2's gates bind on MTP NP=1.
- **Phase 0.B** — Radical Allium / TLA+ / test surface expansion. 5 new Allium specs + 3 new TLA+ specs + 5 property tests + trace-harness extension. Required because existing S1–S5 covers what PHASE_NSTREAM_KV preserved (Bug C absence, per-stream allocator, mask isolation) but NOT the surface Tier 2/3 introduce (CUDA-graph reuse, unified-stream dispatch, MTP/DFlash composition under graph reuse). Historical justification: tasks #37/38/39 (cudaMallocAsync NPC stochastic 1/8) were uncatchable by S1–S5. 105–170 k tokens. Five gates GP0.B.a–e.
- **Tier 1 / 2** — Extend `update_cache_copies` per-stream patching to the K/V *read* views in `llm_build_kqv` (mirrors the write-CPY patching that already ships from PHASE_NSTREAM_KV_4D N2.b); drop the `n_stream > 1` bailout in `can_reuse_graph` (`src/llama.cpp:616`). Existing in-tree `cudaGraphExecUpdate` infrastructure (Phase 36/37/38) carries the downstream patching. 60–100 k tokens. Eight gates GP3.a–h + GP3.n.
- **Tier 3** — Unified-stream dispatch (one `llama_decode` per tick spanning N streams via the ne[3] axis). Uses the existing production PSKV per-slot FA kernel (`GGML_OP_FLASH_ATTN_EXT_PER_SLOT_KV`) — no novel FA kernel port. 120–180 k tokens. Five gates GP3.i–m.

P0.A and P0.B run in parallel. Tier 2 begins only after both close. Tier 3 begins only after Tier 2 closes. Total phase: **305–490 k tokens**.

## What just closed

[`PHASE_NSTREAM_KV.md`](PHASE_NSTREAM_KV.md) — 2026-05-20. 4D per-stream KV layout port; Bug C structurally closed; decode-side prefill gate removed; six correctness gates green on Qwen 3.6 27B production. One perf gate (G3.h) failed by -6.2 %; user-selected override merged the bundle, perf recovery handed to this PERF phase.

## Where to start the next session

1. Read [`STATUS.md`](STATUS.md), [`PHASE_NSTREAM_KV.md`](PHASE_NSTREAM_KV.md), [`PHASE_NSTREAM_KV_PERF.md`](PHASE_NSTREAM_KV_PERF.md) in that order.
2. Read MEMORY.md entries from 2026-05-20 — override decision, byte-compat axis-order tradeoff, Phase 0 prereq scoping.
3. Begin Phase 0 — P0.A (server CLI wiring) and P0.B (spec/test surface) in parallel.

## Other tracked workstreams

Not blocked by this phase, not active in any session right now:

- DFlash speculative decoding — Qwen 3.6 27B with bespoke sm_75 kernels. Kernel pipeline T1–T9 closed; multi-slot orchestrator Phases 1–6 closed; batched-pinned dispatch landed. CLI wiring is Phase 0.A of this PERF phase.
- MTP production — Qwen 3.6 27B native MTP weights, np=1 --draft 3 in production at ~33.5 t/s.
- Vulkan multi-GPU split-mode graph — RDNA2 + Vega production stack.
- TU102 specialisation — open kernel-ranking work in `PHASE_TU102_SPECIALIZATION.md`.
- F.4.1' perf recovery — open in `PHASE_PERF_F4_1.md`.
- AsyncReduce — planning in `PHASE_ASYNC_REDUCE.md`.

See the mdBook nav (`docs/SUMMARY.md`) for the full tree.
