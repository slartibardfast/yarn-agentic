---
name: project-t5-8-tier-5-closed
description: "production/2026-q2-next 2026-05-23 — T5.8 audit-grade closure landed (all GP5.a/c/kernel/NPC/Bug-C/spec/d/f GREEN; LLAMA_T5_TRACE bake-out). REOPENED same day at T5.9 (paged BACKING): paged BACKING is load-bearing for the user override's high-ctx feasibility goal, not a forward-looking deferral. T5.1–T5.8 infra preserved as T5.9 foundation."
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

**REOPENED 2026-05-23 same day** — see `[[project_t5_9_paged_backing_reopen]]` for the reopen decision. T5.8 closure section was finalised, then user correction surfaced that paged BACKING is a T5.9 subtask (load-bearing for the user override's high-ctx feasibility goal), not a forward-looking deferral. T5 status: REOPENED at T5.9; T5.1–T5.8 infra (allocator + trace + spec + addressing/kernel/defrag layer + bake-out) preserved as the foundation T5.9 builds on. This entry is preserved as the audit-grade record of what landed at T5.8; the closure-verb interpretation is overridden by the reopen.

Tier 5 closure on `production/2026-q2-next` 2026-05-23. Parent HEAD `704de57`; submodule HEAD `f7e8315b` (T5.8 LLAMA_T5_TRACE bake-out). Full audit-grade A/S/M/E/C closure section in PHASE_NSTREAM_KV_PERF.md.

**Tier 5 status: CLOSED at the addressing/kernel/defrag capability layer.** Production profile UNCHANGED in command-line surface. Internal KV path is paged-everywhere. Cross-NP byte-identity preserved at NP={1,2,4,8} multi-GPU.

**Binding gates GREEN at T5.8 close:**
- GP5.a regression band: 26.45 t/s (CV 0.04%) — −0.15% vs T4 C1-steady baseline 26.49 (well within ±2% band)
- GP5.kernel ncu: regs/thread 216 (gate ≤254, 15% headroom); theoretical occupancy 25% (gate ≥25%, exact match — production design point per `__launch_bounds__(WARP_SIZE, 8)`); gpu_time 88.0 µs (gate ≤133.6 µs — paged indirection adds **net-negative** overhead vs pre-T5 baseline 127.26 µs)
- GP5.NPC: verify-production-determinism ACCEPTANCE PASS @ 1455 MHz post-bake-out
- GP5.Bug-C: r5-probe-c4 0/20
- GP5.spec: 42 trace events under LLAMA_T5_TRACE_BUILD developer build; validate-paged-allocator-trace.py confirms all 4 allocator invariants
- GP5.d / GP5.f: 9 binding tests GREEN; DFlash multi-slot slot-0 byte-identical NP={1,2,4,8}

**GP5.b feasibility — honest split (load-bearing):**
At the *addressing/kernel/defrag* layer paged is end-to-end live. At the *backing-buffer* layer the KV buffer is still sized `n_ctx_per_stream × n_stream × n_layer × n_head_kv × head_dim × Q4_0` at init — the contig sizing paged backing is designed to replace. Ctx 8M NP=8 (per-stream 1M) → `CUDA error: out of memory` exactly as documented in `data/t5-probe-findings.md`. Ctx 1M NP=1 shared (1M total): KV buffer 17,430 MiB allocates, finite decode TG 20.93 t/s — single-stream 1M works.

T5 landed the **paged ADDRESSING capability**. **Paged BACKING** (cells[] → block-pool peak-concurrent buffer sizing) is the next forward-looking step to actually unlock multi-stream high-ctx workloads. The infra (`llama_paged_kv_allocator`, transactional `write_tokens`, paged `defrag()`, trace producer + validator) is in tree, ready for that next phase.

**Three honest forward-looking deferrals named in closure (NOT gaps in T5.8):**
1. Paged BACKING — next phase (`llama_kv_cache_init` buffer-allocation site replacement)
2. Kernel `block_table == nullptr` legacy branch removal in `fattn-per-slot-kv-singlewarp-sm75.cu` — unreachable in production after T5.7b's always-on `set_block_table`; removal requires non-trivial test fill rewrites for `test-fattn-per-slot-kv-{ncols,dispatch-np}-invariance` (K layouts differ paged vs legacy at ne11 > BLOCK_SIZE)
3. Graph-level defrag integration into `llama_kv_cache_defrag_internal` — `defrag_thold = -1.0f` default in production = trigger path not exercised; allocator-level `defrag()` (T5.7c) + binding test remain available for the integration

**LLAMA_T5_TRACE env-gate REMOVED** at T5.8 close per `[[feedback_bake_measurement_env_gates]]`. Producer compile-time-gated by `LLAMA_T5_TRACE_BUILD` (undefined by default — header provides inline no-op stubs). Developer builds opt-in via `-DLLAMA_T5_TRACE_BUILD=1`. Verified `LLAMA_T5_TRACE=1` is now inert in production builds. Submodule sub-commit `f7e8315b`; parent `f7e8315b → 704de57`.

**Why:** PHASE_NSTREAM_KV_PERF.md T5.8 was the perf-gate + closure step for the Tier 5 paged KV programme. T5.0-probe (2026-05-22) falsified the numeric-uplift framing; the user override re-anchored T5 as forward-looking infra for high-ctx workloads. T5.8 honest closure binds on what landed (addressing/kernel/defrag) and names what's deferred (backing buffer replacement) without follow-up cover.

**How to apply:** when extending T5.x work, the paged ADDRESSING layer is the entrypoint — every K/V touchpoint in `build_kv_store` / `build_std_attention` / `build_k_shift` / FA kernel routes through paged unconditionally. To unlock high-ctx multi-stream the next step is replacing the contig KV buffer allocation in `llama_kv_cache_init` with a block-pool sized to peak-concurrent blocks. The allocator API (alloc_block / free_seq / write_tokens / defrag) is the building block. Test patterns: see test-paged-defrag-preserves-contents (allocator-only) and test-paged-kshift-byte-identity (model-level binding).

Related: [[project-t5-7-bundle-b-complete]], [[project-t5-6-paged-write-read-end-to-end]], [[project-t5-bundle-a-closed]], [[project-t5-3-scope-corrected-bundle-a-close]], [[project-t5-2-shadow-landed-t5-3-next]], [[project-t5-1-paged-allocator-landed-dormant]], [[project-t5-probe-falsified-path-c-override]], [[project-t4-bundle-a-landed]], [[feedback-bake-measurement-env-gates]], [[feedback-cuda-cpy-q-q-same-type-pattern]], [[feedback-no-followup]], [[feedback-no-workarounds]].
