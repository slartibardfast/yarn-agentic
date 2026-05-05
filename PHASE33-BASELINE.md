# Phase 33 — Step 1 baseline measurement

Captured ~12 min into a live OpenCode session against the 27B
production server (commit b4cb862, post-PHASE33 plan). Single user,
slot 0 only.

## Counts

| Event                | Count |
|----------------------|-------|
| Checkpoints created  | 89    |
| Erased invalidated   | 5     |
| Restored             | 2     |
| Full re-process      | 0     |

## Cadence

Effective interval between consecutive create-checkpoint `pos_min`
values: **2048 tokens**, matching `--batch-size`. The configured
`--ctx-checkpoints-interval 512` (ik_llama default) is overridden by
the batch boundary — checkpoints can only land on a logical-batch
edge.

## Latency

`create_check` per-checkpoint cost (n=60 lines with explicit
timing): **min 132.2 ms, mean 139.0 ms, max 151.3 ms**. Each
checkpoint is **~150 MiB** in host RAM.

Total checkpoint creation time over the 12-min window:
89 × 139 ms ≈ 12.4 s, ~1.7% wallclock tax. Acceptable in
absolute terms; partly redundant due to the cap-eviction issue
below.

## Restore path is working

Both restore events landed at `pos_min = 96945` from tasks 129 and
236, ~133 ms each. That's the system intercepting an OpenCode
resume of an existing conversation and skipping ~95K tokens of
prefill — without it, full re-eval would be many seconds.

## Invalidations are bounded, not cascading

All 5 invalidations occurred within a ~190-token window (96951,
97017, 97024, 97030, 97133) immediately after the restore at
96945. Pattern: OpenCode resumes at 96945 then advances; the next
checkpoint (96951+) belongs to a slightly-different prefix and is
dropped. This is "trim-to-checkpoint then drift" — small, expected,
not the pre-restart slot-1 cascade.

## The real bottleneck is the cap

The journal log shows the count progression hit `26 → 27 → … →
32 of 32` and then wrapped to `7 → 8 → 9 of 32`. LRU eviction
kicked in. With slot 0's ~85K-token conversation (`/slots` reports
prompt_chars=341173 ≈ 85K tokens) and the 2048-token interval,
covering the full conversation requires ~42 checkpoints. We
have 32. The earliest ~14K tokens have been evicted from the cache;
if OpenCode references that range, it hits partial restore.

## Verdict

- Restore mechanism functional. Don't break it.
- Invalidation rate healthy under current usage; Step 5 (plugin
  diagnostic) deferred — not a problem at this scale.
- Step 2 (raise `--ctx-checkpoints` to 64, `--cache-ram` to 40 GiB)
  directly addresses the eviction. With 64 cap × 150 MiB × 4 slots
  = 38.4 GiB max, 40 GiB budget covers it. Host has 42 GiB
  "available" before the bump; tight but workable, drops to ~22
  GiB after fill — still enough for OS buff/cache.
- Step 4 (interval reduction) deferred — coarser interval would
  *worsen* eviction at our cap. Finer would create more eviction
  pressure. The right move is more cap, same interval.

## Not done in this step

- `--cache-ram-similarity` (Step 3) — has no effect until Step 2's
  larger cache pool exists for it to consult. Will measure after
  Step 2 lands.

## Snapshot data sources

- `journalctl --user -u llama-server --since "$SINCE"` from
  10:49:54 UTC (server start) to ~11:02 UTC.
- `curl /slots` snapshot at 11:00 UTC.
- `free -h` at 11:00 UTC: 20 GiB used, 14 GiB free, 39 GiB
  buff/cache, **42 GiB available**.
