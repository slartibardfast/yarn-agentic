# Phase 33 — production cache tuning

Operational tuning of the running 27B production server. Goal: turn
observed checkpoint churn into faster turn transitions for OpenCode
agentic workloads, without re-quantizing or rebuilding.

## Observed signals (live system, post-restart)

- Slot 0 emits a checkpoint every ~2048 tokens during prefill
  (cadence ≡ `--batch-size`, NOT the documented 512-token default of
  `--ctx-checkpoints-interval`). 13 checkpoints in 65 s, 132 ms each,
  150 MiB each → ~6.5% prefill tax on a long prompt.
- Pre-restart slot 1 task 25198 emitted an erasure cascade: 15
  consecutive checkpoints across pos 69631→94693 (~2.3 GiB of cache)
  invalidated and dropped in a single second. This is the hybrid /
  prefix-rewrite invalidation pattern documented in upstream issue
  #19794.
- Active load: 1 of 4 slots used. `--parallel 4` capacity dormant.
- VRAM: GPU0 5.2 GiB free, GPU1 0.5 GiB free. No headroom for
  more aggressive on-GPU cache.

## ik_llama flags relevant here (NOT same as upstream)

| Flag | Default | What it does |
|------|---------|--------------|
| `--ctx-checkpoints` | 32 | Cap of checkpoints per slot |
| `--ctx-checkpoints-interval` | 512 | Min token gap between checkpoints |
| `--ctx-checkpoints-tolerance` | 5 | Edge-case end-of-prompt grace |
| `--cache-ram` (`-cram`) | 8192 MiB | Host-RAM pool for checkpoints |
| `--cache-ram-similarity` (`-crs`) | 0.50 | Prefix-similarity threshold to trigger restore (ik_llama-only) |
| `--cache-ram-n-min` | 0 | Min cached tokens before restore triggers |
| `--recurrent-ckpt-mode` | auto | SSM state strategy for MTP draft (not used here) |

Note: `--ctx-checkpoints-interval 512` is documented but observed
cadence is 2048. Likely the effective minimum is `max(interval,
batch_size)`. Worth confirming during step 1.

## Decisions

1. **Don't blindly raise the interval.** Web research (PR #17428
   commentary, Issue #22218 production agent configs) consistently
   recommends FINER granularity for coding agents — `1024` interval
   with 256 checkpoints — because mid-prefix edits invalidate
   suffixes anyway. Coarser checkpoints just increase recompute on
   each invalidation.

2. **Raise the checkpoint ceiling AND the RAM budget proportionally.**
   At 150 MiB/checkpoint × 4 slots × 32 cap = 19 GiB used. We have
   32 GiB budget. Headroom exists. Bumping cap to 64 → 38 GiB
   needs cache-ram bumped to 40 GiB. Host RAM has 46 GiB available;
   tight but workable.

3. **Tune `--cache-ram-similarity`.** This ik_llama-specific knob is
   exactly what mitigates the OpenCode "tool-envelope rewrite breaks
   prefix" pattern. Default 0.50 may already be doing work. Raise
   first (try 0.7) — should prevent partial-match restores that lead
   to slow re-eval. If hit-rate drops too far, lower to 0.3.

4. **Defer interval reduction.** Cutting interval below 2048
   doubles checkpoint creation rate AND each checkpoint is fixed-cost
   ~132 ms; net throughput tax could double. Only worth it if step
   2/3 don't fix the user-observed slow turns. Bench-then-decide.

5. **Acknowledge #5 may be plugin-introduced.** OpenCode's plugin
   system (oh-my-opencode, custom hooks) can rewrite system prompts
   or tool envelopes between turns. The slot-1 erasure cascade
   timing fits a plugin transaction. We'll surface this as a
   diagnostic step before throwing more RAM at the symptom.

## Plan steps (ordered)

### Step 1 — measure current baseline (one OpenCode session)

Tail journal for 10 minutes during a real OpenCode multi-turn coding
task. Capture for each turn:

- N checkpoints created
- N invalidated/erased
- Restore-on-resume hit rate (look for "restored context checkpoint"
  vs "forcing full prompt re-processing")
- Effective interval (delta between consecutive `pos_min` in create
  log lines) — confirm it's 2048 not 512

Write findings to `PHASE33-BASELINE.md`. No code/profile changes.
This is the bind for "did the tuning help?"

### Step 2 — raise the ceiling (low-risk, reversible)

Edit `profiles/qwen36-27b-x4.sh`:

- Add `--ctx-checkpoints 64` (was implicit 32)
- Bump `--cache-ram 40960` (was 32768)

Restart, verify host RAM still has ≥4 GiB free buff/cache after warmup.
Re-run an equivalent OpenCode session, re-measure. Compare.

### Step 3 — tune similarity threshold

If step 2 helped, hold and tune `-crs`:
  - First try `-crs 0.70`. Re-measure restore hit rate.
  - If hit rate drops, try `-crs 0.30`.
  - Pick winner.

### Step 4 — reduce interval (only if needed)

If steps 2+3 didn't fix slow turns:

- Add `--ctx-checkpoints-interval 1024`
- This roughly doubles checkpoint creation rate; verify --cache-ram
  budget still covers it (64 cap × 150 MiB = 9.6 GiB per slot).
- Re-measure. If throughput tax exceeds turn-time savings,
  revert.

### Step 5 — diagnose plugin invalidation pattern

Independent of throughput tuning. Compare two sessions side-by-side:

- Session A: OpenCode with default plugin set
- Session B: OpenCode with plugins disabled

If A shows the erasure cascade and B doesn't → confirmed plugin
cause; the fix lives in plugin config (stable prompt envelope), not
server tuning. Document, file an issue against the plugin or
configure prompt template to be stable across turns.

## Risk register

| Risk | Mitigation |
|------|------------|
| Bumping cache-ram to 40 GiB starves OS buffer cache; disk I/O slows | Watch `free -h` post-restart; if buff/cache drops <2 GiB, revert |
| Higher checkpoint count amplifies invalidation cost | Step 5 addresses root; tuning is a band-aid |
| Restart drops in-flight slot 0 work | Coordinate with user before each restart; do during idle window |

## Out of scope

- Building with metrics endpoint (separate task; nice-to-have)
- Sub-agent / oh-my-opencode wiring (separate strand)
- NVLink/PCIe x16 hardware moves
- Recurrent-ckpt-mode tuning (only matters with MTP on; current
  profile is nomtp)

## Sources

- [llama.cpp PR #17428](https://github.com/ggml-org/llama.cpp/pull/17428) — fixed-token-interval checkpoint design rationale
- [llama.cpp issue #19794](https://github.com/ggml-org/llama.cpp/issues/19794) — Qwen3-Coder-Next hybrid invalidation, the canonical example of our slot-1 cascade
- [llama.cpp issue #22218](https://github.com/ggml-org/llama.cpp/issues/22218) — production coding-agent config: `--checkpoint-every-n-tokens 1024 --ctx-checkpoints 256 --swa-full`
- [llama.cpp issue #19977](https://github.com/ggml-org/llama.cpp/issues/19977) — Qwen3.5-122B real-world `--cache-ram 20000 --ctx-checkpoints 32`
- [llama.cpp discussion #21480](https://github.com/ggml-org/llama.cpp/discussions/21480) — checkpoint sizes vary 9× by architecture
- [ik_llama.cpp discussion #1434](https://github.com/ikawrakow/ik_llama.cpp/discussions/1434) — ctx-checkpoints perf impact (caveat: bench-sweep doesn't trigger checkpoints)
- [llama.cpp tools/server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) — flag reference
