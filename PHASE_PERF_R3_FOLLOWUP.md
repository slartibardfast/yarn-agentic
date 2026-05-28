# PHASE_PERF_R3_FOLLOWUP — perf regressions + NP=2 production-scale validation

**Opened:** 2026-05-28 08:20Z
**Branch:** `production/2026-q2-next`
**Parent:** `PHASE_PERF_R3_NP1.md` (executed 2026-05-28 00:45Z, REPORT in
`data/perf-r3-np1/REPORT.md`)
**Status:** PD pack. No experiments run.

## Why this phase exists

The PHASE_PERF_R3 execution surfaced three things that require follow-up
work before any production change beyond the trivial `--ubatch-size 256`
ship can land:

1. **The Phase I NP=2 deadlock proof was at `--ctx-size 32768` only.** The
   2026-05-06 hang was at `--ctx-size 524288` + NP=2 — exactly the
   configuration we'd want to ship to preserve 256k-per-slot capacity at
   NP=2. We have no valid evidence that the production-scale NP=2 config
   is hang-free on the current stack.

2. **R1 — ctx-size allocation tax.** At identical prompt depth (~200t),
   `--ctx-size 8k` gave 18.21 t/s (A1) but `--ctx-size 262144` gave 11.54
   t/s (E1) — a **-37% TG** penalty from allocating the production-size
   KV pool, even when most of it is unused. This is the biggest single
   perf gap exposed by R3 — bigger than any other lever pulled.

3. **R2 — sharp 3k→12k TG inflection.** At `--ctx-size 262144`, TG peaks
   at 15.13 t/s (2901t prompt) then collapses to 7.52 t/s by 12081t
   prompt. **-50% TG over ~4× prompt depth** is too sharp for natural
   KV-attention scaling and suggests a kernel/cache threshold.

R1 + R2 are real findings. If R1 closes ~half its gap, that's a +10-15%
production TG win — substantially more than NP=2 throughput gain or the
`--ubatch-size 256` lever. Investigation order should prioritize R1 and
R2 over NP=2 validation, because:

- R1 fix benefits every workload on the existing NP=1 production
  deployment (no risk surface change).
- R2 may be a kernel-tunable that helps regardless of NP.
- NP=2 helps multi-user throughput but adds risk surface; less impactful
  if R1 closes 37% on its own.

## Production state going into this phase

| Component | State |
|---|---|
| `llama-server.service` | STOPPED (user directive, 2026-05-28 08:13Z) |
| `/home/llm/profiles/qwen36-27b-x1-vanilla.sh` | **rolled back to pre-np2 safe config**: `--parallel 1 --ubatch-size 512 --ctx-size 262144 --cache-ram 40960 --ctx-checkpoints 64` + RT chain. Known-good. |
| Repo wrapper at `scripts/systemd/llm-rt-tuning/qwen36-27b-x1-vanilla.sh` | Mirrors deployed (safe pre-np2) |
| systemd drop-ins | `00-lib-path.conf`, `02-cuda-graph-probe.conf`, `03-rt-deps.conf`, `04-rt-flags.conf` — all in place |
| `llm-rt-prep.service` | enabled (governor=performance, IRQs 0-3) |
| Build binary at `/opt/llm-server/bin/llama-server` | `b2cf8fbf` (C-arc + RT) |

A user restart of the service via `sudo systemctl start llama-server.service`
boots the known-good production-baseline config. **No NP=2 risk.**

## Published-curve calibration (2026-05-28 web review)

Anchoring R1/R2 expectations against the public literature before running
experiments — so that "surprising" and "expected" are pre-declared:

### R1 (ctx-allocation tax) — architecturally expected; the question is T5.9's payback

- The unified KV pre-allocates `n_ctx × n_seq_max` per layer regardless of
  usage (ggml-org Discussion #21961). Pre-paged dispatch is known to pay
  this tax in full.
- FA kernels in llama.cpp lazily reserve a dequantized f16 scratch buffer
  at full ctx size (TurboQuant Discussion #20969). Q4_0 + ctx=256k → an
  FA dequant scratch ~16× larger than at ctx=8k, hit on every step even
  with a 200t prompt.
- **Magnitude band:** 37% from 16× more allocated KV is within the
  expected range for the pre-paged dispatch shape. The R1 finding is
  not anomalous in size.
- **What is testable:** ik_llama's T5.9 paged-KV is *supposed* to
  eliminate the allocation tax for sparse usage. Phase 3 must measure
  whether T5.9 is actually paying back. If our gap is identical with
  vs. without `--cache-ram` + `--ctx-checkpoints`, T5.9 is a no-op for
  short-prompt workloads — that's the load-bearing finding.

### R2 (3k→12k cliff) — position is anomalous; magnitude is in range

Reference benchmark (Nemotron-3-Nano-30B-A3B, 128K ctx, Q4_0 vs F16, from
TurboQuant Discussion #20969):

| Depth | F16 t/s | Q4_0 t/s | Delta |
|---|---|---|---|
| ~6K  | 44.7 | 45.0 | +0.7% |
| ~24K | 44.6 | 39.3 | -11.9% |
| ~110K| 38.0 | 24.0 | -36.8% |

- Published Q4_0 dequant-overhead curve is **smooth and monotonic**, not
  cliff-shaped. Material degradation starts above 24K.
- Our cliff is at **3K→12K** — well below the published threshold. That
  position is not explained by standard Q4_0 dequant overhead.
- Our config has **RHT/Hadamard pre-quant**
  (`--k-cache-hadamard --v-cache-hadamard`), a custom transform layer the
  upstream Q4_0 benches do not have. Leading hypothesis: a Hadamard-path
  kernel selection or block-count threshold fires somewhere in the 3K-8K
  range.
- Flash-Decoding (PyTorch / Stanford CRFM, 2023) shows that long-context
  cliffs are usually artifacts of not having split-K decoding at that
  depth. ik_llama's GRAPH-mode FA may inherit the same limitation, but
  the published cliff position is at 32K+, not 3K-8K.
- **Verdict:** R2 cliff position is genuinely worth a kernel-level diff.
  Phase 2 should specifically scan for Hadamard transform kernels in
  addition to FA / cuBLAS.

### What the calibration changes

- **Don't be surprised by R1 magnitude.** -37% from 16× wider KV is in
  the expected band. The phase deliverable is "did T5.9 close it for
  sparse usage", not "why so big."
- **Be surprised by R2 cliff position.** Below 24K is not the published
  shape — there is a real engine-specific cost center to find.
- **Phase 1's "smooth decay" decision branch is now legitimate.** If the
  3K→12K sweep curve is smooth, it matches the published Q4_0 shape and
  R2 closes as misframed. That outcome is now an expected case, not a
  failure of the phase.

Sources captured in MEMORY.md / `data/perf-r3-followup/REPORT.md`.

## Investigation plan

### Phase 1 — R2: localize the 3k→12k TG inflection (cheap, no risk)

Hypothesis: a specific kernel selection or cache-spill threshold fires
between prompt depth 3k and 12k that costs ~50% TG.

**Method:**

1. Start single-slot server `--ctx-size 262144 --parallel 1 --ubatch-size 256`
   + full RT chain (matches production target with G2a lever applied).
2. Sweep prompt depths: 3000, 4000, 5000, 6000, 8000, 10000, 12000 — finer
   granularity than Phase E.
3. 3 reps × 7 depths × N_PREDICT=128 = 21 requests, ~10-15 min wall time.
4. Plot TG vs prompt depth. Find inflection point.

**Decision:**
- Smooth decay across the range → R2 is misframed; not actually an
  inflection, just observation variance at Phase E sparse points.
  Close R2.
- Clear step-function at some depth N → captured. Move to Phase 2 with
  nsys traces at N-500 and N+500.

### Phase 2 — R2: nsys kernel diff across the inflection (medium cost)

If Phase 1 finds the inflection at depth N:

1. nsys trace at prompt depth N-500 (above the cliff) — fast TG side.
2. nsys trace at prompt depth N+500 (below the cliff) — slow TG side.
3. Diff `cuda_gpu_kern_sum` — identify which kernel(s) regressed.

**Decision tree (in priority order — Hadamard first per calibration):**
- **Hadamard transform kernel cost growing disproportionately** → our
  config has `--k-cache-hadamard --v-cache-hadamard`; upstream Q4_0
  benches do not. A kernel selection or block-count threshold in the
  RHT path firing between 3K-8K would land here and is the calibration's
  leading hypothesis. Localize in `ggml-cuda/hadamard*.cu` or wherever
  the RHT path lives in the fork.
- FA kernel change → kernel-level fix (likely a tile-size or warp-count
  choice that's wrong for large KV). Lever in successor phase. Note:
  published FA-decode cliffs are at 32K+, not 3K-8K, so this is a
  secondary hypothesis.
- cuBLAS algo change → cuBLAS algo pin or workspace tuning.
- New kernel appearing → indicates ggml dispatched to a different code
  path. Localize via source review.
- No clear kernel-level cause → the cost is in the host-dispatch or
  memory-bandwidth side. Different investigation.

### Phase 3 — R1: ctx-size allocation tax decomposition (medium cost)

Hypothesis: allocating ctx=256k vs ctx=8k pays a per-step cost that scales
with the pool size, NOT with actual KV usage. The pre-paged dispatch
shape's allocation tax is architecturally expected (Discussion #21961,
TurboQuant FA dequant-scratch finding). The actionable question is
whether T5.9 paged-KV is paying it back for sparse usage.

**Method — primary sweep (shape of the tax):**

1. Start identical servers at `--ctx-size ∈ {8192, 32768, 131072, 262144}`,
   `--parallel 1`, `--ubatch-size 256`, full RT chain.
2. Run identical 200t prompt + N_PREDICT=128 bench on each, 3 reps.
3. Plot TG vs allocated ctx-size at the same workload.

**Expected outcomes (shape):**
- Linear decline → tax is proportional to pool size; suggests something
  scans the whole pool per step.
- Step-function → there are kernel thresholds at specific ctx values
  (e.g. 64k = block-count crossing).
- Flat with small ctx, drop only at large ctx → cache-spill (allocated
  pool no longer fits in L2/HBM working set).

**Method — T5.9 paged-KV effectiveness sub-test (~12 min):**

The above sweep measures the *current* shape; this sub-test measures
*how much T5.9 is buying us*. At the two endpoints of the sweep
(ctx=8192 and ctx=262144), run a second config with the paged-allocator
backing disabled or minimal:

1. ctx=8192 + `--cache-ram 0 --ctx-checkpoints 0` vs production
   `--cache-ram 40960 --ctx-checkpoints 64` → expected near-equal
   (small pool, T5.9 has nothing to page).
2. ctx=262144 + `--cache-ram 0 --ctx-checkpoints 0` vs production
   settings → the discriminator. If TG is the same in both, T5.9 is
   not active in our profile for sparse usage. If TG with checkpoints
   is materially higher than without, T5.9 is paying back the
   allocation tax and the size of the win quantifies the recovery.

3 reps each at both ctx points = 12 requests, ~6 min server-up time
plus warmup overhead. Adds ~12 min to Phase 3.

**Decision branch on T5.9 sub-test:**
- T5.9 is recovering meaningful TG (>10% delta at ctx=262144) → keep
  the allocator settings; the remaining R1 gap is structural and
  Phase 4 must find it in the kernel diff.
- T5.9 is a near no-op at sparse usage (<5% delta) → load-bearing
  finding. Open a successor phase to investigate why
  `paged.seed_identity_per_stream()` and the block-table indirection
  aren't activating for short prompts. This may be the biggest single
  lever in the followup.

### Phase 4 — R1: nsys diff ctx=8k vs ctx=256k at same workload (high info)

Take the two extreme ctx points (8k and 256k) at the same 200t prompt
workload. nsys trace each. Diff `kern-sum`.

The 37% gap has to land in some kernel — find which.

### Phase 5 — R3: NP=2 + ctx=524288 production-scale reproducer

**Run only after R1/R2 are characterized.** R1 may close the gap that
makes NP=2 attractive in the first place. Defer until that's known.

If R1/R2 don't close enough, then:

1. Pre-test:
   - Confirm uptime + driver version (will compare against 2026-05-05/06)
   - SSH alternate-host recovery path verified (current SSH listening
     on :22 — but no second machine on the LAN tested yet — open
     question for ops)
   - Set RSS watchdog at 24 GB (vs 20 GB used in earlier Phase I — go
     wider since 524k ctx is at higher legitimate memory)
   - Time-box: 10 min per rep (longer than Phase I's 5 min because
     deep prefills take longer)

2. Test geometry (matches 2026-05-05 incident as closely as possible):
   - Server: `--ctx-size 524288 --parallel 2 --ubatch-size 256 --cache-ram 16384 --ctx-checkpoints 16` + full RT chain
   - Slot 0: 131k token prefill + 128 predict (the 2026-05-05 incident
     was at slot 0 position ~157k; 131k is the closest match we have a
     prebuilt prompt for)
   - Slot 1 enters at t+30s (mid-prefill, when slot 0 is at ~3-4 min of
     its expected ~8-10 min prefill)
   - Monitor: nvidia-smi dmon, host RSS, journalctl

3. Reps:
   - Rep 1: slot 0 131k + slot 1 short, 30s offset
   - Rep 2: same (confirmation)
   - Rep 3: both slots concurrent 65k prefill (sustained two-slot
     pressure)

4. Outcomes:
   - All clean → ship `--parallel 2 --ctx-size 524288` to production
     under a controlled soak (1h → 24h before flipping).
   - Hang on any rep → MEMORY.md entry updating the
     2026-05-05/06 constraint with the new finding. NP=2 stays off.
   - Partial degradation (one slot fails, host stays up) → characterize
     what fails and reconsider.

## Time + token budget

| Phase | Wall | Tokens | Notes |
|---|---|---|---|
| Phase 1 (R2 sweep) | 15 min | 6k | bench-only |
| Phase 2 (R2 nsys diff) | 30 min | 10k | depends on Phase 1 result |
| Phase 3 (R1 ctx sweep + T5.9 sub-test) | 37 min | 10k | bench-only; sub-test discriminates paged-KV payback |
| Phase 4 (R1 nsys diff) | 35 min | 10k | extends Phase 3 |
| Phase 5 (NP=2 524k reproducer) | 45 min | 15k | high risk, last |
| Report aggregation | 15 min | 6k | |
| **Total** | **~177 min** | **~57k** | Fits comfortably in any reasonable window |

Phases 1+2 are independent of 3+4. Could run in parallel if appetite for
two simultaneous servers exists (different ports, alternate by phase).

## Status (2026-05-28)

- **Phase 1 — DONE (R2 closed as misframed).** Fine-grained sweep at
  ctx=262144 NP=1 ubatch=256 + RT chain shows smooth concave-down decay
  from 16.53 t/s @ n_pp=2221 to 9.25 t/s @ n_pp=8851. Per-step slope
  decreases monotonically — no cliff. Phase E's "peak then drop" was
  sparse-sampling artifact. Calibration prediction held (published Q4_0
  cliffs are at 32K+, not 3K-12K). See
  `data/perf-r3-followup/phase1-r2-sweep/FINDINGS.md`.
- **Phase 2 — SKIPPED.** No R2 cliff to diff.
- **Phase 3** — next; load-bearing R1 investigation.
- **Phase 4** — dependent on Phase 3 outcome.
- **Phase 5** — gated on Phase 3/4.

## Acceptance — phase closes when

- [x] Phase 1 sweep curve plotted; R2 inflection localized or rejected
      → **rejected, smooth decay**
- [x] If R2 real: Phase 2 kernel diff identifies the cost center
      → **R2 not real, Phase 2 not needed**
- [ ] Phase 3 sweep curve plotted; R1 allocation-tax shape characterized
- [ ] Phase 3 T5.9 sub-test recorded: paged-KV payback quantified at
      ctx=8k and ctx=262144; "is T5.9 active for sparse usage?" answered
- [ ] Phase 4 kernel diff identifies the cost center for R1
- [ ] Decisions recorded on ship candidates per finding:
      - R1 fixable → open successor phase to ship the fix
      - R1 inherent → document, close
      - R2 fixable → open successor phase
      - R2 inherent → document, close
- [ ] Phase 5 (only if reached): NP=2 outcome A/B/C captured, MEMORY.md
      updated, ship decision recorded
- [ ] REPORT.md committed to `data/perf-r3-followup/`

## Standing notes for this phase

- The "deployed wrapper" is the pre-np2 safe config (NP=1, ubatch=512,
  ctx=262144 + RT chain). If anyone restarts production via `systemctl
  start llama-server.service`, it boots in this known-good state.
- The `--ubatch-size 256` lever (G2a +4.7% TG) is a separate trivial
  ship candidate. Can land independently with a 1-line wrapper edit
  once R1/R2 close or when the user wants to take that win.
- `/tmp/perf-r3-prompts/` has the 210/1k/4k/16k/65k/131k token-target
  prompts ready for re-use. No regeneration needed.
- The harness `scripts/test-production-np-determinism.sh` has the
  `EXTRA_ARGS` and `THREAD_COUNT` env hooks already added (R3 prep).

## Supersedes / informs

- Supersedes the Phase I outcome A claim "NP=2 is unlocked" — that
  claim is only true at ctx=32k. Production-scale validation still
  pending. The PHASE_PERF_R3_NP1.md headline #5 stays as a "no longer
  reachable at small ctx" finding pending Phase 5 here.
- MEMORY.md note `2026-05-28 00:45Z` PHASE_PERF_R3 finding (5) "NP=2
  deadlock no longer reachable" should be read as "no longer
  reachable at ctx=32k NP=2"; production-scale NP=2 + ctx=524k is
  the Phase 5 target.
- Auto-memory `project_perf_r3_executed.md` headline #6 should be
  similarly qualified.
