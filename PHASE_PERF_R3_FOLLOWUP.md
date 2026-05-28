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

**Decision tree:**
- FA kernel change → kernel-level fix (likely a tile-size or warp-count
  choice that's wrong for large KV). Lever in successor phase.
- cuBLAS algo change → cuBLAS algo pin or workspace tuning.
- New kernel appearing → indicates ggml dispatched to a different code
  path. Localize via source review.
- No clear kernel-level cause → the cost is in the host-dispatch or
  memory-bandwidth side. Different investigation.

### Phase 3 — R1: ctx-size allocation tax decomposition (medium cost)

Hypothesis: allocating ctx=256k vs ctx=8k pays a per-step cost that scales
with the pool size, NOT with actual KV usage.

**Method:**

1. Start identical servers at `--ctx-size ∈ {8192, 32768, 131072, 262144}`,
   `--parallel 1`, `--ubatch-size 256`, full RT chain.
2. Run identical 200t prompt + N_PREDICT=128 bench on each, 3 reps.
3. Plot TG vs allocated ctx-size at the same workload.

**Expected outcomes:**
- Linear decline → tax is proportional to pool size; suggests something
  scans the whole pool per step.
- Step-function → there are kernel thresholds at specific ctx values
  (e.g. 64k = block-count crossing).
- Flat with small ctx, drop only at large ctx → cache-spill (allocated
  pool no longer fits in L2/HBM working set).

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
| Phase 3 (R1 ctx sweep) | 25 min | 8k | bench-only |
| Phase 4 (R1 nsys diff) | 35 min | 10k | extends Phase 3 |
| Phase 5 (NP=2 524k reproducer) | 45 min | 15k | high risk, last |
| Report aggregation | 15 min | 6k | |
| **Total** | **~165 min** | **~55k** | Fits comfortably in any reasonable window |

Phases 1+2 are independent of 3+4. Could run in parallel if appetite for
two simultaneous servers exists (different ports, alternate by phase).

## Acceptance — phase closes when

- [ ] Phase 1 sweep curve plotted; R2 inflection localized or rejected
- [ ] If R2 real: Phase 2 kernel diff identifies the cost center
- [ ] Phase 3 sweep curve plotted; R1 allocation-tax shape characterized
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
