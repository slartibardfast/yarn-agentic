# yarn-agentic

This site is the public record of [David Connolly](https://blog.david.connol.ly), a developer plus agentic AI assistants, forking [`ik_llama.cpp`](https://github.com/ikawrakow/ik_llama.cpp) (Iwan Kawrakow's fork of [`llama.cpp`](https://github.com/ggml-org/llama.cpp)) into a byte-deterministic, multi-GPU LLM inference server on two consumer Quadro RTX 6000s (sm_75 / TU102, PCIe Gen3 PHB until 2026-05-24 when NVLink NV2 was confirmed at 91% of theoretical). The parent repo (`yarn-agentic`) orchestrates two C++ submodules (`ik_llama.cpp`, `llama.cpp`); development happens in twelve-hour sessions against this ledger. What you are reading is not a writeup of the work, it is the work.

**Agentic Host folder.** The parent repo is the state machine that keeps long-running agentic-AI sessions coherent across context-window breaks. `docs/active/` is read at session start — an agent resumes work without re-priming. `MEMORY.md` (11 028 lines, append-only) is the cross-session journal: every closure, every incident, every diagnosis lands here. `git mv docs/active/PHASE_X.md docs/archive/phases/<topic>/` is the closure signal; the move is what makes the closure real. The C++ submodules (`ik_llama.cpp`, `llama.cpp`) hold the code; the parent holds everything else — plans, memory, specs, scripts, gates, the mdBook ledger you are reading.

**[Allium](https://juxt.github.io/allium/v2/) plugin.** 45 `.allium` files (30 under `specs/`, 15 at repo root) encode behavioural contracts (entities, enumerations, invariants); 4 are wired into the `spec-tla-gate` workflow (`dflash.allium`, `batch-invariance.allium`, `MgpuSplitConfig.allium`, `CrossCodepathConsistency.allium`). `allium check` (v3.2.3) gates each; `allium plan` emits an obligation ledger that's diffed against the committed `data/*_test_obligations.json` on every push — edit a spec without regenerating the ledger and the gate trips. `scripts/check-bindings.py` enforces three-way correspondence between Allium invariants, TLA+ definitions, and C++ test citations.

**TLA+ plugin.** 50 `.tla` files under `specs/`; 6 are gated by 14 TLC runs in the workflow ([Specula](https://github.com/specula-org/Specula)-authored, [TLC](https://github.com/tlaplus/tlaplus) `tla2tools.jar` v1.8.0). 10 positive `.cfg`s must PASS — current design satisfies invariants. 4 negative-control `.cfg`s must FAIL with a named violation (`NumRejectedTokensFlowsBackToProposer`, `AnchorPosPreserved`, `PerSlotVerifyDispatchAtMultiSlot`, `NoCrossSlotRegionOverlap`). A negative-control run that unexpectedly passes means the invariant is vacuous; CI fails. This is how Phase 41's `branch_seq_id` buffer overflow was caught via the `AllBranchSeqIdsFitInSlotBuffer` invariant before code touched a GPU. The remaining 41 `.allium` and 44 `.tla` files are work-in-progress contracts not yet wired into the gate.

Production model: `qwen3.6-27b-V-F1.T1.lm_head-f16.gguf` — BF16 weights, F16 lm_head, Q4_0 KV cache, Hadamard transform.

## Active

- [`PHASE_CUDA_NATIVE_DISPATCH`](active/PHASE_CUDA_NATIVE_DISPATCH.md) — ground-up CUDA-idiomatic dispatch replacing the openmp parallel multi-backend path. C0–C14 code arc complete (`b86931c`); live-verification follow-on open.
- [`PHASE_HYBRID_CHECKPOINT`](active/PHASE_HYBRID_CHECKPOINT.md) — stabilise → diagnose → Phase 45-aligned decomposition. Phase 1 in flight.
- [`PHASE_TU102_SPECIALIZATION`](active/PHASE_TU102_SPECIALIZATION.md) — kernel ranking for production inference on sm_75.

## Work

Reverse-chronological. One link per line.

- 2026-05-28 — `PHASE_R1_CLIP_RACE` Phase A closed and deployed. CLIP cross-encode non-determinism localized to two disjoint failure modes; per-node sync fence (`CLIP_DISABLE_SYNC_FENCE=1` to revert) and B.5e activation-buffer-clear both required for 10/10 byte-identical encodes. [archive](archive/phases/determinism/PHASE_R1_CLIP_RACE.md)
- 2026-05-28 — `PHASE_PERF_R3_FOLLOWUP` closed. R1 ctx-allocation tax narrow-fixed via `ggml_backend_sched_set_zero_on_reset` opt-out: `-25.9%` → `-7.3%` at ctx=256k. [archive](archive/phases/determinism/PHASE_PERF_R3_FOLLOWUP.md)
- 2026-05-27 — `PHASE_CUDA_NATIVE_DISPATCH` C0–C14 code arc complete; C14 live-verification artifact landed. [active](active/PHASE_CUDA_NATIVE_DISPATCH.md)
- 2026-05-27 — NP=8 single-slot flake root-caused to CPU governor=powersave (12-hour code-level chase that day-1 governor lock would have closed). [archive](archive/phases/determinism/PHASE_NP8_FLAKE.md)
- 2026-05-27 — `PHASE_PERF_R3` 95-min characterisation run executed: no PD4 regression, RT chain `+24%`, ubatch=256 free `+4.7%`, NP=2 deadlock no longer reachable, NCCL active at 14.8%. [archive](archive/phases/determinism/PHASE_PERF_R3_NP1.md)
- 2026-05-27 — Phase 46 closed; production live on the multi-GPU CLIP build. Median CLIP encode `14 421 ms`; ten encodes byte-identical at the response sha256. [archive](archive/phases/multi-gpu/PHASE46-MULTIGPU-CLIP-TENSOR-SPLIT.md)
- 2026-05-26 — Phase 46 B.5e test sequence (capture bisect → libmgpu validation → NPC.4 six-test → GPU clock-lock → tests K/L/M/N → state-leak → node-73 localization) closed via combined fix: `cudaDeviceSynchronize` default + per-split drain + gallocr zero-on-reset between encodes.
- 2026-05-25 — Stale `libggml.so` deploy regression. Patched binary installed; runtime loaded build-tree library via RUNPATH. Atomic-install + sha256-verify guard added to `scripts/deploy-llama-server.sh`.
- 2026-05-25 — Hybrid checkpoint Phase 1 landed: restore disabled for recurrent models (SEGV mitigation).
- 2026-05-25 — Phase 35 Step B alloc-aware CUDA graph eviction landed; Phase 46 opened for multi-GPU CLIP.
- 2026-05-25 — `GGML_SCHED_MAX_SPLITS` legacy assert removed (`ik_llama.cpp` `252217d8`) — three-line surgical patch matching upstream PR #9047 policy.
- 2026-05-24 — NVLink NV2 verified on `xeon`: `91%` of theoretical ceiling. Production LLM stack migrated `yarn` → `xeon` (nginx + three backends + cert pipeline).
- 2026-05-24 — End-to-end NVLink bench on `xeon`: graph split-mode beats layer by `18–44%`.
- 2026-05-23 — T5 reopened same day; paged BACKING shipped as T5.9 (defrag default on, ~0.3% overhead, 100+ fires per 60s run). Production aggregate `26.65 t/s` at NP=8 on `bench-t3.8-m3`.
- 2026-05-22 — Tier 3 unified-stream dispatch T3.6 closed (full grid); T3.8 perf gate FAIL — saturates at multi-slot kernel ceiling `26.49 t/s`. Theory falsified, Tier 4 justified.
- 2026-05-22 — Tier 4 chunked-prefill admission (Sarathi-Serve) coherent flip landed. T4.7 perf gate honest FAIL — staggered ≤ steady on aggregate t/s, structurally.
- 2026-05-21 — P0.A.4 multi-slot DFlash SEGV closed (per-stream-scoped `speculative_decoding_accept` + local-frame index translation).
- 2026-05-21 — P0.A.3 (DFlash output divergence) closed via MMQ I=8 disable + cross-mmq_x dispatch uniformity.
- 2026-05-20 — `PHASE_NSTREAM_KV` closed on `production/2026-q2-next` (submodule `16b608d1`). Bug C structurally closed; decode-side prefill gate removed. -6.2% TG NP=8 regression carried over.
- 2026-05-20 — N1 4D structural landed; N2/N3 require non-byte-compatible axis order. Bug C spec layer (S1–S5) landed pre-N1 4D port.
- 2026-05-19 — DFlash Path A closed (pinned-HMMA dispatch on drafter forward + lm_head). Production GGUF switched to F16-lm_head recast target.
- 2026-05-19 — I=8 MMQ ground-up rewrite shipped (Lever D). Target #1 split-K MMQ for decode shapes lands. Lever A stacked +1% incremental.
- 2026-05-19 — Option C lands: cuBLAS `ALGO0` pin closes silent NPC regression.
- 2026-05-19 — R5 NPC investigation: kernel layer innocent; bug bisected to MMQ I=8 + a second latent kernel via compute-sanitizer initcheck on `process_tile`.
- 2026-05-17 — `PHASE_NP_DETERMINISM_CLOSED`: NPC.6 closed, full NP-determinism shipped. `active.sh` flipped to multi-slot deterministic.
- 2026-05-17 — NPC.5 multi-GPU closed; NPC.4 production harness closed (ctx-checkpoint tolerance was splitting prefill); NPC.2 root cause `ssm_conv` NP-divergence; NPC.1/NPC.3 localize prefill shape-dep on `n_tokens-per-ubatch`.
- 2026-05-17 — F.4.1' closed — non-packed `up_gate` launch + `force_rpcb1`. F.4.1 root cause localized + fixed (F.4 perf gap remains).
- 2026-05-17 — Phase CY closed; CY.F.18 proper fix landed (`has_reduce`-gated persistence). Audit A.1' closed (FA prefill 256-tok shape-dep baked out).
- 2026-05-16 — CY.F.18 root cause + fix: scheduler sync lifecycle race closed.
- 2026-05-16 — CY.F.17 root cause: MMQ `stream_K`, not singlewarp FA.
- 2026-05-16 — CY.F.16 Option A: arch-force F32 reduce closes NP={4,8} determinism.
- 2026-05-16 — TRACE-1..6 + research dive: root cause is WMMA k-chunk decomp; FIX-C v4 lands.
- 2026-05-15 — `PHASE_MMQ_Q4_0_AR16` Phase A and Phase B closed. Phase C F16-pinned WMMA GEMM landed; production NP-determinism partial.
- 2026-05-14 — Phase 2 `fattn_per_slot_kv_sm75` kernel work complete.
- 2026-05-13 — DFlash T4 closed (argmax-equivalent across 8 prompts vs vLLM). T6 closed (probe-before-implementing path).
- 2026-05-12 — DFlash T1 closed: drafter GGUF converter landed. vLLM measured at `154.77 t/s` NP=8 aggregate on the same hardware.
- 2026-05-09 — `PHASE45` D8 closed (`spec_loop` extraction validated end-to-end). D9 design (server+common port + extract + delete + rename). D9.5 milestone (tag `phase45-d9.5`).
- 2026-05-08 — `PHASE45` D6 closed (Option A wrapper validates byte-identical) and D7 closed (wrapper has zero CUDA cost). `PHASE45` supersedes PHASE38 D/E/F via decomposition.
- 2026-05-08 — `PHASE 44` capture-indirection diagnosis correction.
- 2026-05-05/06 — Two host hangs under `--parallel 2` (driver-class, not OOM). NP=2 deadlock unreachable as of 2026-05-27 R3 reproducer (driver bump 580.x → 595.71.05, CUDA 12.x → 13.2).
- 2026-05-04 — `PHASE32` Stage B closed at 27B + 35B-A3B (Tool 1 lossless fix).
- 2026-05-03 — `PHASE32` reframe: FP16 trunk wins; `mtp.fc` cast is a tie. T2-T5 KLD proof + name-dedup post-mortem.
- 2026-05-02 — `PHASE31` Step 5 closed `[~]` (binding negative: 27B has no MTP heads). Quadro replication of `PHASE31` MTP.
- 2026-05-01 — MTP works correctness-good on 3060 Ti; throughput negative under `--cpu-moe`. TurboQuant work abandoned in this tree; MTP becomes sole focus.
- 2026-04-26 — Qwen3.6-35B-A3B BF16 GGUF smoke-verified on CPU. Ampere Vulkan turbo_kv_4b dequant regression vs Vega. Coopmat2-capable Ampere refuses LSE at `supports_op`; cm2 LSE port aborted, dispatcher fallback to cm1 chosen.
- 2026-04-26 — Host hard-hung; replaced zram with disk swap + systemd-oomd.
- 2026-04-19 — t/s Pareto bench closed on qwen35-0.8b. HARP_2B throughput ceiling + `vec_dot_type` discipline. Strategic pivot: abandon trellis 2-bit, compare to Unsloth. 2-bit area abandoned after Unsloth dominance benchmark.
- 2026-04-18 — Per-layer sensitivity sweep on qwen35-0.8b. Stale-gguf masquerading as a Vega shader bug.
