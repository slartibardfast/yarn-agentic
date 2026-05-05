# Phase 33 — Step 2: raise the ceiling

## Change

`profiles/qwen36-27b-x4.sh` (not a git-tracked dir; recorded here):

```diff
     --cache-type-k q4_0 --cache-type-v q4_0 \
     --k-cache-hadamard --v-cache-hadamard \
-    --cache-ram 32768 \
+    --cache-ram 40960 \
+    --ctx-checkpoints 64 \
     --no-context-shift \
```

`--cache-ram` 32 GiB → 40 GiB. `--ctx-checkpoints` 32 → 64.

## Rationale

Step 1 baseline showed slot 0 hit `32 of 32` checkpoints during a
single OpenCode session, then started LRU-evicting the earliest
~14 K tokens. With 64 cap and 2048-token interval, one slot now
covers up to ~131 K tokens of conversation before any eviction.
Budget math:

- Worst case: 4 slots × 64 checkpoints × 150 MiB = 38.4 GiB
- Allowed:   40 GiB
- Single-slot reality: most sessions use only slot 0, so worst-case
  is 64 × 150 MiB = 9.6 GiB

Host RAM headroom check: `available` was 47 GiB right after
restart (cache pool starts empty). Will refill as slot 0 builds
up checkpoints. Even at the 38.4 GiB ceiling, host has headroom
left for OS buff/cache.

## Verification

Boot log confirmed:
```
prompt cache is enabled, size limit: 40960 MiB
```

Server health: `{"status":"ok","slots_idle":4,"slots_processing":0}`.

VRAM unchanged (cache lives in host RAM): GPU0 18.7 / 5.3 free,
GPU1 23.4 / 0.6 free. Same razor-thin GPU1 margin as before.
Host RAM 47 GiB available pre-fill.

`--ctx-checkpoints 64` is not echoed at boot but it took effect —
will be visible in journal as `X of 64` once slot 0 emits its
first checkpoint of the new session.

## Next

Step 3 (`--cache-ram-similarity` tuning) needs an active session
with cache history to consult, so we wait for the new pool to
re-fill before measuring it. Step 4 (interval) deferred — Step 1
showed eviction, not creation overhead, was the real bottleneck.
