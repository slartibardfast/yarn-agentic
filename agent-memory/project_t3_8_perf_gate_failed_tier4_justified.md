---
name: project-t3-8-perf-gate-failed-tier4-justified
description: "PHASE_NSTREAM_KV_PERF T3.8 closed FAIL on production/2026-q2-next 2026-05-22. T3.5 multi-seq dispatch fires at 93% rate but delivers ~0% throughput uplift at decode shape on PSKV-singlewarp + Q4_0 KV + sm_75. Gate theory was wrong: the throughput is locked away by prefill stalls (Tier 4 / continuous batching), not by dispatch packing."
metadata:
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

# T3.8 perf gate GP3.i — FAILED, theory falsified, Tier 4 justified

**Fact:** GP3.i fails on all three sub-gates. Measurement of record at `data/t3.8-perf-gate-ledger.md`. All measurements taken under locked GPU clocks (nvidia-smi `clocks.current.sm` = 1455 MHz on both GPUs); CV across N=3 runs ≤ 0.43% per config — single-shot bench numbers ARE reliable under this regime. M2 NP=8 batched-bench no-Hadamard = 27.73 t/s aggregate (gate ≥ 100). M3 NP=8 server + Hadamard = 26.49 t/s aggregate (gate ≥ 90). T3.5 dispatch counter `total=192 multi_seq=179` byte-identical across M3 runs — 93% multi-seq firing rate, scheduler deterministic.

**Why:** The PHASE doc's direction tree estimated +3.6×–4.7× from "unified-stream multi-seq dispatch — kernel batching, NOT graph reuse" (PHASE_NSTREAM_KV_PERF.md ~line 1918). That assumption did not hold for our kernel + shape. At decode n_tokens=1-per-seq on PSKV-singlewarp + Q4_0 KV + sm_75:

1. The kernel work is grid-parallel per-seq either way (PSKV per-slot fattn-singlewarp).
2. Launch + dispatch overhead amortises below per-kernel runtime (PSKV singlewarp is ~127 µs/call per ledger row 19; cudaLaunchKernel is ~5-10 µs).
3. Memory traffic dominates and isn't reduced by batching seqs into one tick.

So sub-batching 8 single-seq decodes (M2's QNEXT_SEQ_INTERLEAVED fallback path) ≈ unified multi-seq dispatch (M3's T3.5 active path) within the Hadamard tax (~5%). This **generalises** the PHASE doc's Tier 2 finding (CUDA graph reuse ≈ 0% at our shape) to dispatch packing — both rest on the same kernel-cost-dominates-launch-cost mechanism.

The vLLM 154.77 t/s reference number on the same hardware comes from continuous batching / chunked-prefill admission (PHASE doc OpenQ-C / Tier 4), splicing new-request prefill into running-decode ubatches. This is a different mechanism that addresses prefill stalls, NOT decode throughput at fixed-NP. M2's stall fraction (T_PP/(T_PP+T_TG)) is **77.9%** at the bench shape — the throughput is locked away by prefill stalls, not by anything in T3 scope.

**How to apply:**

- **Don't propose dispatch-packing or scheduler-grouping work as a throughput lever for Qwen 3.6 27B Q4_0 KV PSKV-singlewarp at sm_75.** It costs ~zero relative to per-kernel runtime. T3 proves this empirically.
- **The actual throughput lever for the vLLM gap is Tier 4 (continuous batching / chunked-prefill admission).** A fresh phase should plan this: splice new-request prefill ubatches into running-decode ticks so 8 slots' prefills overlap with each other's decode rather than stalling sequentially. Closer to vLLM's continuous-batching scheduler.
- **Locked GPU clocks (1455 MHz) are required for binding perf measurements.** Without lock, the prior ledger rows 6 and 21 differ by 14% on the same config. Under lock, CV < 0.5%. Run `nvidia-smi --query-gpu=clocks.current.sm` before each binding measurement and abort if not 1455.
- **`llama-batched-bench` doesn't exercise the server scheduler.** Its token-major batch layout triggers QNEXT_SEQ_INTERLEAVED fallback at `src/llama.cpp:5789`. To bind server-scheduler perf measurements, use the HTTP-driven `scripts/bench-t3.8-m3.sh` pattern instead (8 concurrent /v1/completions, n_predict fixed, deterministic prompt).
- **The T3 correctness deliverables stay.** Multi-stream KV is necessary for future ctx + parallelism scaling regardless of which throughput lever is pursued next. Don't revert anything from T3.0–T3.7 in service of "perf didn't materialise."

**Stall % and Tier 4 trigger:** M2 stall 77.9%, M3 is more complex (concurrent prefills via HTTP queue) but qualitatively similar. PHASE doc Tier 4 trigger ("if conservative misses AND stall > 30%, Tier 4 justified") fires unambiguously.

**Graph-pool VRAM data (per the T3.6.M probe, captured during all T3.8 configs):** pool grows with parallelism but bounded across all four M-configs: ≤ 25 graphs / ≤ 400 nodes / ≤ 110 KB host per device. **No runaway under multi-stream + Hadamard.** I.b.2 bailout-drop revisit therefore doesn't have a VRAM-pressure justification.

**Established:** T3.8 closure 2026-05-22 with parent commits for ledger + PHASE doc. Production profile unchanged. Next throughput lever is Tier 4, separate phase.
