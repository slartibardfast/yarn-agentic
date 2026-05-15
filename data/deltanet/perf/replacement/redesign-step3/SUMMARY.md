# Task #136 Redesign — Approach A + split-K perf at decode NP=1

Date: 2026-05-15
Branch: production/2026-q2-next
Capture: `tests/dflash-speculative/test-np-validity-vanilla.cpp` with
`LLAMA_FATTN_PER_SLOT_KV_ENABLE=1`, n_seqs=1, n_gen=32
Config: Qwen 3.6 27B production target, Q4_0 + Hadamard KV

## Per-kernel timing (nsys)

| Kernel | Calls | Avg time |
|---|---:|---:|
| `fattn_per_slot_kv_sm75_single_head_split_k_kernel` | 992 | **63.4 µs** |
| `fattn_per_slot_kv_sm75_combine_kernel` | 992 | 1.6 µs |
| **wmma_f16 baseline (from baseline-prod)** | — | **26 µs** |

**Ratio: 63.4 / 26 ≈ 2.4× slower than wmma_f16 at decode shape (n_kv ≈ 24-56).**

Improvement over Approach C variant (12× slower at decode per the prior data
in `nsys-vs-wmma-long-ctx-12x-slower.SUMMARY`) but still doesn't meet the
task goal ("beat wmma_f16 per-call time").

## Likely overhead sources at small n_kv

1. **MAX_PB_HARD_CAP=16 always**, regardless of actual n_kv.
   - At n_kv=40 (early decode): true pb=1 (n_kv < 256). But grid.z=16 forces 16x
     CTA launches, 15 of which return early after writing a sentinel meta. The
     early-return CTAs still incur launch overhead and SMEM allocation.
   - Effective work-to-launch ratio: 1/16 of CTAs do real work.
2. **Combine kernel always launched** even when pb=1 (pass-through).
   - Adds 1.6 µs/call kernel-launch overhead.
3. **16 m-axis rows always**, even at n_tokens=1 (1 real + 15 padded).
   - Same wasted compute as wmma_f16's cols_per_block=8 path — not a regression
     but doesn't gain on this axis.

## NP-validity (regression check)

PASS at NP={1, 2, 4} × n_gen=16 — full slots terminate, no NaN, decode_ok,
in-vocab, PPL within band. Determinism contract holds.

## What's still on the table

- Trim MAX_PB_HARD_CAP at runtime based on n_kv (D2H sync acceptable if used
  ONCE per decode step, OR use a sentinel + first-CTA-of-each-stripe
  early-exit pattern that avoids the launch entirely)
- Skip combine kernel launch when max_pb is effectively 1 (need pre-launch
  decision — same D2H question)
- ncu detailed metrics to identify the actual bottleneck (DRAM throughput,
  occupancy, cycles per iteration) — would let us point at a specific cause
  rather than guess
- Approach: split-K at LONG n_kv only, single-pass at short n_kv (n_kv < 512
  threshold) — needs runtime branch in dispatch fn
