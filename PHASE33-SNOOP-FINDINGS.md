# Phase 33 — live OpenCode snoop findings

Run: `snoop-20260505T111847`, ~38 min, 2 concurrent OpenCode
sessions against the 27B production server (Step 2 profile:
`--parallel 4`, cap=64, cache-ram=40 GiB).

## Headline: production multi-slot is broken under real OpenCode traffic

Five `GGML_ASSERT` crashes in 38 min:

```
src/ggml-cuda/concat.cu:202:
  GGML_ASSERT(src0->type == src1->type && src0->type == dst->type) failed
```

| Crash | Trigger | Notes |
|-------|---------|-------|
| 11:17:52 | (slot init) | pre-session |
| 11:18:16 | (slot init) | pre-session |
| 11:36:50 | slot 1 task 12966 launch | slot 1 first real activation |
| 11:47:42 | slot 1 task 9088 launch (p0=0) | slot 1 second real activation |

systemd auto-restarted the service each time (visible in
`journalctl: Started llama.cpp inference server` after each
ABRT). nginx returns **502** during the restart window: 13 of 71
requests = **18% production failure rate**.

The Phase 32 multi-slot bench (PHASE32-V-F1-T1-QQ-MULTISLOT-RESULTS.md)
passed at np=2 mtp 30.86 t/s, np=4 27.73 t/s, np=8 28.79 t/s — but
that bench used 8 *uniform* prompts × 40 predicted tokens. Real
OpenCode traffic with heterogeneous per-slot prompt shapes
hits a path the bench didn't exercise.

## Slot scheduling: serialised in practice, not by intent

| Metric | Value |
|--------|-------|
| Total polls | 1326 |
| Polls with slot 0 active | 1033 (78%) |
| Polls with slot 1 active | **0 (0%)** |
| Polls with slots 2/3 active | 0/0 |
| Polls with 2+ slots concurrently active | 0 |

Slot 1 was *assigned* tasks (3 launches in journal: 12966 @ 11:36,
10 @ 11:37, 9088 @ 11:47) but the server **crashed every time
slot 1 reached `launch_slot_with_task`**. Net effect: every
request was handled by slot 0 — strict serialisation.

Per-slot prompt arrival was correct (slot 1 received the second
session's 84 K-token prompt), so the scheduler routes properly;
the kernel can't survive the second slot's first prompt batch.

## Cache behaviour

| Metric | Step 1 baseline | Step 2 (this run) |
|--------|-----------------|-------------------|
| Creates | 89 / 12 min | 339 / 38 min (~similar rate) |
| Restores | 2 | 13 |
| Erased invalidated | 5 | **193** |
| Hit rate (restores / POSTs) | n/a | **18.3 %** |
| Cap usage seen | 32/32 | **64/64** |

- Cap-eviction recurred at the new ceiling: max usage hit `64 of
  64`. Slot 0's conversation grew to 109 K tokens — Step 2's
  ~131 K headroom was consumed.
- Invalidation **pos distribution** confirms the plugin-rewrite
  hypothesis: median 39307, max 108925, with **181 of 193 (94 %)
  outside the tail zone**. These are mid-prefix invalidations.
- The 11:36:17 slot-1 lifecycle is the cleanest evidence. OpenCode
  session 2 connected with a 54618-token prompt, server restored
  pos=10239, then **immediately erased ~12 contiguous checkpoints
  from pos 12287 onward** — OpenCode's prompt diverged from the
  cached prompt right at the 10K → 12K boundary. Every checkpoint
  past the divergence point had to be thrown away.

## Capacity reality

- Slot 0 prompt grew to 87,495 tokens (~350 KB string).
- Slot 1 prompt grew to 84,129 tokens before the crash — proves
  multi-session is working *up to* the concat point.
- 1326 polls × ~150 MiB/checkpoint × ~50 retained checkpoints
  pushed `--cache-ram` toward the 40 GiB ceiling without OOM.

## Immediate production mitigation

Drop `--parallel` to 1. Reasons:

1. The crash bug means parallel slots are **net-negative** for
   uptime — every concurrent attempt is a coin-flip on a
   server-wide ABRT.
2. Per Step 1 + this run, slot 0 handles ~99 % of real traffic
   anyway; we don't lose meaningful concurrency by serialising.
3. Single-slot path is well-tested (Phase 32 single-slot mtp
   bench: 35.36 t/s solid).
4. With `--parallel 1` at `--ctx-size 1048576`, slot 0 gets the
   full 1 M-token context budget — strictly more than the current
   per-slot 256 K cap.
5. Reversible: re-enabling `--parallel 4` is a one-line edit when
   the concat bug is fixed.

## Diagnostic backlog

1. **Reproduce the concat.cu:202 assert in isolation** — script
   that drives two slots through OpenCode-shaped prompts of
   different sizes simultaneously, captures the failing tensor
   types. (Prior Phase 32 bench is uniform; need a heterogeneous
   variant.)
2. **Locate the concat call site** — concat.cu:202 is the kernel
   guard; need to walk back to the cgraph op that issued it.
   Likely candidates: hybrid-arch attn-residual paths, KV
   accumulator joins.
3. **Plugin-rewrite verification** — examine the
   `prompts/slot-0.diff` output for the actual diff hunks at
   invalidation timestamps. (Not done yet; Layer-2 capture
   succeeded, just not analysed.)
4. **Step 3 (`-crs`) deferred** — meaningless until parallelism
   is stable; without uptime there's no cache to consult.

## Files

| Path | Bytes | Content |
|------|-------|---------|
| `~/snoop-runs/snoop-20260505T111847/journal.log` | 1.45 MB | full server log |
| `~/snoop-runs/snoop-20260505T111847/nginx.log` | 16 KB | OpenCode HTTP requests |
| `~/snoop-runs/snoop-20260505T111847/slots.jsonl` | 743 KB | 1326 polls × 4 slots |
| `~/snoop-runs/snoop-20260505T111847/gpu.tsv` | 535 KB | dmon per-second |
| `~/snoop-runs/snoop-20260505T111847/prompts/` | 59 snapshots | slot-0×58, slot-1×1 |
| `~/snoop-runs/snoop-20260505T111847/findings.md` | — | summary by `snoop-summarise.py` |
