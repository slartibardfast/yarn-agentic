=== PD4 SUMMARY ===

## PD4 Baseline — 2026-05-27 12:19Z

Build: 1db6c2eb (post-Phase 46 closure). GPU clocks locked 1455 MHz.

### LM perf (test-production-np-determinism, N_PREDICT=128, prompt~210t)

| NP | Aggregate TG | Per-slot TG | PP per-slot | Scaling vs NP=1 |
|---|---|---|---|---|
| 1 | 17.9 t/s | 17.9 t/s | 241.5 t/s | 1.00× |
| 2 | 25.2 t/s | 12.6 t/s | 121.5 t/s | 1.41× |
| 4 | 28.9 t/s | 7.2 t/s | 101.5 t/s | 1.61× |
| 8 | 30.8 t/s | 3.8 t/s | 64.0 t/s | 1.72× |

vLLM reference: 154.77 t/s aggregate at NP=8 on same hardware → 5.0× headroom to ceiling.

### Dispatch counter (first 64 dispatches)

| NP | total | multi_seq | multi-seq ratio |
|---|---|---|---|
| 1 | 64 | 0 | 0% (no concurrent slots, expected) |
| 2 | 64 | 64 | 100% |
| 4 | 64 | 55 | 86% |
| 8 | 64 | 38 | 59% |

Multi-seq ratio drops as NP rises — slot arrival order breaks down (seq_first==0 condition met less often).

### CLIP encode (verify-multigpu-clip LATENCY_N=10)

- median: **14450 ms** (Phase 46 closure was 14421 ms — within noise)
- p95: 14746 ms
- raw: 15900, 14460, 14504, 14522, 14416, 14450, 14444, 14511, 14445, 14746
- warm-up (sample 1): 15900 ms = +1.4 s vs steady-state
- steady-state variance: 330 ms (samples 2-10)
- baseline ceiling (1.3× CPU vision 42000): 54600 ms → 73.6% headroom

### Determinism at NP=8 (incidental)

This PD4 run reproduced the NP=8 flake at **slot 5** (new — prior reps hit slot 6 or 7). Failing-slot histogram now: slot 5, 6, 7 (any of the three high-indexed slots).

### Bound perf targets for §4.3 of PHASE_CUDA_NATIVE_DISPATCH

- **LM NP=1 TG**: 17.9 t/s baseline; ±5% target → 17.0-18.8 t/s post-phase
- **LM NP=8 aggregate TG**: 30.8 t/s baseline; conservative 1.5× target → 46 t/s; stretch 3× → 92 t/s
- **CLIP encode median**: ≤ 14450 ms; stretch (with C8+C9 active) → ≤ 8000 ms (45% reduction)
- **CLIP encode hard gate**: ≤ 1.3× CPU vision = 54600 ms

### Files

- np-determinism logs: $OUT/np-determinism/run-20260527T122019/
- CLIP baseline: $OUT/clip-baseline/run-20260527T122326/latency.json
- This summary: $OUT/SUMMARY.md
