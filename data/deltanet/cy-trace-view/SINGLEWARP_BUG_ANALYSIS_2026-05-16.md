# Singlewarp NP=2 multi-step bug — analytical framework (2026-05-16)

## What we know empirically

**Repro conditions** (test-cy-np2-multi-step-decode):
- singlewarp kernel (LLAMA_PSKV_MODE=singlewarp)
- n_seq_max=2 context
- 12-token prompt
- 32-step batched decode (n_seqs=2 in FA grid per step)
- Decode step 11 onward: slot 0 deterministically diverges from NP=1 baseline (same alt tokens every run)
- Steps 0-10: byte-identical to NP=1

**Clean conditions** (no divergence):
- singlewarp at NP=1 (n_seqs=1 grid)
- wmma at NP=2 (any decode length, in unit test)

**Production data** (V4 server, cont_batching enabled):
- singlewarp NP={1,4,8}: byte-deterministic across 5+ runs
- singlewarp NP=2: ~40% pass rate, intermittent slot-1 (or slot-0) divergence

## The puzzle

The unit test shows a **deterministic** singlewarp bug at NP=2 multi-step.
Production shows **intermittent** failure at NP=2 only — NP=4/8 are perfectly clean.

If singlewarp deterministically broke at n_seqs≥2, NP=4 and NP=8 should ALSO fail (they have n_seqs=4 and n_seqs=8 grids). They don't.

Possible reconciliations:
1. **Production cont_batching almost-never co-batches multiple decode slots in one FA call.** The server schedules decodes serially most of the time. NP=4/8 production then runs at n_seqs=1 grid (clean). NP=2 occasionally co-batches (intermittent fail). Unit test ALWAYS co-batches (deterministic fail).
2. **The bug fires only at n_seqs=2 specifically**, not at n_seqs=4 or 8. Some n_seqs-mod-N pattern.
3. **The bug is in the unit test setup**, not in singlewarp. The unit test's batched prefill + batched decode doesn't match what production does.

## Hypothesis space (ranked by plausibility given evidence)

### H1 — `cont_batching` serializes decode at NP=4/8 in production, exposing the bug only at NP=2

If production's `update_slots()` for some reason batches decode slots independently at NP=2 but not NP=4/8, the FA grid sees:
- NP=1 production: n_seqs=1 always → singlewarp clean
- NP=4 production: most decodes at n_seqs=1, occasional n_seqs=4 → singlewarp would fail if bug fires at n_seqs≥2 but we see clean
- NP=2 production: often n_seqs=2 → bug fires

Doesn't quite fit because NP=4/8 are CLEAN. Either the bug doesn't fire at n_seqs≥4 (some specific n_seqs=2 trigger), OR NP=4/8 are run with n_seqs=1 grids.

### H2 — The bug is n_seqs=2 specific (not n_seqs≥2)

What's special about exactly 2 concurrent CTAs vs 4 or 8?
- 2 CTAs on TU102: probably co-scheduled on the same SM (since SM has many warps and singlewarp uses 1 warp/CTA, 2 CTAs = 2 warps on the same SM)
- 4+ CTAs: spread across multiple SMs

If the bug requires two CTAs on the SAME SM (shared register file partitioning, shared L1, shared scheduler), n_seqs=2 hits it more reliably than n_seqs=4/8 (which spread).

**Probe**: force CTAs onto different SMs via launch config. Or run nsys to confirm SM occupancy.

### H3 — Unit test's batched prefill creates a different cache state than production

Unit test: prefills both seqs in ONE llama_decode (24 tokens, FA at n_seqs=2 during prefill).
Production: cont_batching may serialize prefills (one seq per batch, n_seqs=1).

If singlewarp PREFILL at n_seqs=2 writes K/V correctly but FA output (which feeds the next layer's residual) has a tiny error, that error propagates through 64 layers into the KV cache for ALL slots. Then step 11 diverges because the KV cache has wrong values written during prefill.

**Probe**: switch unit test to serial prefill mode (already supported via LLAMA_TEST_SERIAL_PREFILL=1). Earlier test confirmed serial prefill ALSO triggers the same divergence at step 11. So this hypothesis is FALSE.

### H4 — Step 11 is where some quantized softmax precision threshold is crossed

The Welford online softmax has a piecewise function:
```c
scale_corr = diff_old >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_old) : 0.0f;
```
With SOFTMAX_FTZ_THRESHOLD = -20. At step 11, n_kv=24, the accumulated `kqmax` may sit such that one specific k's `diff_old` crosses -20 in NP=2 but not NP=1.

If precision in `kqmax` accumulation differs minutely between n_seqs=1 and n_seqs=2 grids (e.g., due to compiler reordering FMA operations), the threshold can fire differently at step 11 deterministically per run.

**Probe**: change SOFTMAX_FTZ_THRESHOLD to -∞ (no flush). If divergence disappears, this is the cause.

### H5 — KV cache write at multi-step decode at n_seqs=2 has a race or wrong addressing

The K/V cache is updated each decode step by a write kernel. If at n_seqs=2 the write kernel has a race or addressing bug, slot 0's K cache content drifts from what it would be at n_seqs=1.

After 11 steps of slightly-wrong K writes, the cache content for slot 0 differs enough that FA at step 11 produces a different argmax token.

**Probe**: capture K cache content at step 11 for NP=1 and NP=2 slot 0. Compare byte-by-byte. If different, this is the cause. If identical, the cache is fine and singlewarp's FA compute is the bug.

### H6 — Singlewarp uses an uninitialized register / state that happens to be 0 at n_seqs=1 grid but garbage at n_seqs=2

Reading the kernel: VKQ, kqmax, kqsum, Q_reg are all initialized before use. No obvious uninit reads. But the **compiler may pre-load values into specific registers**, and the register allocation could differ between n_seqs=1 and n_seqs=2 grids (different launch_bounds usage, occupancy hint).

**Probe**: inspect ptxas output for both. Or force_launch_bounds(1) to make register allocation consistent.

### H7 — Cross-CTA register file partitioning at TU102 sm_75

When 2 CTAs run on the same SM, the register file is partitioned. The compiler may have a bug where partitioned-register access reads from neighbor CTA's bank.

**Probe**: change `__launch_bounds__(WARP_SIZE, 2)` to `__launch_bounds__(WARP_SIZE, 1)` (only 1 block per SM). Forces CTAs to different SMs.

## Discriminating tests (in order)

1. **H4 (softmax FTZ)**: change threshold to FLT_MAX_NEG (effectively no flush). Build, rerun unit test. If clean, H4 is confirmed.

2. **H5 (KV cache write)**: add cb_eval capture of K cache contents at step 11 in NP=1 and NP=2. Diff. If different, H5; if identical, exclude.

3. **H7 (register partitioning)**: change launch_bounds to (WARP_SIZE, 1). Build, rerun. If clean, H7.

4. **H2 (n_seqs=2 specific)**: force a 4-slot grid even for NP=2 (use dummy slot data). If divergence disappears at 4-slot grid, H2 narrows.

## What the analysis tells us about the fix

- **If H4 (softmax)**: simple — remove or relax the FTZ threshold. Trivial fix.
- **If H5 (KV write)**: harder — find the KV write kernel's n_seqs=2 bug.
- **If H7 (register partitioning)**: change launch_bounds. Simple but reduces occupancy.
- **If H2 (n_seqs=2 specific)**: requires per-grid-size dispatch.

The user's note that "this is likely just unit test's small prompt" suggests they suspect the SOFTMAX FTZ threshold could be triggered by small-prompt-specific kq magnitudes. With longer prompts, kqmax would grow larger and FTZ might never fire — explaining why production may not see the bug.

**Recommended next probe**: H4 (softmax FTZ). Single-line kernel change, fastest test, matches the "step 11" deterministic timing (which would be exactly when the FTZ threshold first crosses for a specific decode token).
