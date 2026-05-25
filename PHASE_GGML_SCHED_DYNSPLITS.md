# PHASE GGML_SCHED — adopt upstream dynamic-splits policy

**Status**: planning. Pre-implementation. Triggered by the 2026-05-25 production assert wedge on `llama-server.service` (host `xeon`, qwen3.6-27b np=1 vanilla profile). Diagnosis below; fix is a three-line surgical patch matching upstream PR #9047 policy.

**Branch**: `production/2026-q2-next` (top-level), `production/2026-q2-next` in the `ik_llama.cpp` submodule.

## 1. Trigger — production wedge (2026-05-25 ~07:52 UTC)

`llama-server.service` (MainPID 59875, started 2026-05-24 22:31:07 UTC) entered a "listening but not serving" state at 07:52:46 UTC. Symptoms:

- `/health` and `/slots` time out at 10 s+; port `:8080` accepts TCP but the HTTP layer never responds.
- Every subsequent `/v1/chat/completions` request returns `status=200` with an immediate `srv stop: cancel task` log line — the HTTP layer ACKs but no tokens are produced.
- GPU utilization 0 % despite the model still resident (~23 GiB across 2× RTX 6000).
- Systemd unit shows `active (running)` — `Restart=on-failure` never triggered because the process is still alive.

Trigger sequence in the journal (one-shot, after 9 h idle):

```
07:52:45  Prompt cache: cache size: 8997, n_keep: 0, cache_ram_similarity: 0.50
07:52:46  cache state: 1 prompts, 3600.579 MiB (limits: 40960.000 MiB)
07:52:46  apply_checkp: restored context checkpoint (149.631 MiB)
07:52:46  apply_checkp: erased invalidated context checkpoint (149.634 MiB)
07:52:46  kv cache rm [p0=512, end)
07:52:46  fragmentation: 0.94
07:52:46  /home/dconnolly/yarn-agentic/ik_llama.cpp/ggml/src/ggml-backend.cpp:1768:
              GGML_ASSERT(i_split < GGML_SCHED_MAX_SPLITS) failed
```

The assert fired on a short "hi" prompt routed through the prompt-cache restore path against a saved 8997-token prompt with `cache_ram_similarity: 0.50` and `f_keep: 0.00`. The combination — long idle, mismatched prompt, `apply_checkp` + `kv cache rm`, KV fragmentation 0.94 — produced an inference graph that required more splits than the compile-time cap of `GGML_SCHED_MAX_SPLITS = 4096`.

For reference, the healthy unfragmented split count for this model on this hardware is **387** (from the init line `llama_init_from_model: graph splits = 387`). Fragmentation 0.94 produced a >10× multiplier.

## 2. Codebase evidence — ik fork is at the pre-#9047 band-aid

Current `ik_llama.cpp/ggml/src/ggml-backend.cpp`:

| Line | What it says | Status |
|------|--------------|--------|
| 1106 | `#define GGML_SCHED_MAX_SPLITS 4096` | Compile-time cap. Predates upstream PR #9047 dynamic-splits fix. |
| 1158–1160 | `splits` is a `*` pointer; `splits_capacity` field exists in struct | **Dynamic machinery already present** — partially backported. |
| 1762–1766 | `if (i_split >= sched->splits_capacity) { splits_capacity *= 2; realloc(...); }` | Grow logic wired up and correct. |
| **1768** | **`GGML_ASSERT(i_split < GGML_SCHED_MAX_SPLITS);`** | **Stale legacy assert.** Fires *after* the realloc-grow loop has already successfully expanded the array. This is the line that crashed production. |
| 2451 | `nodes_size = graph_size + GGML_SCHED_MAX_SPLITS*GGML_SCHED_MAX_SPLIT_INPUTS*2` | Sizes auxiliary arrays against the fixed constant rather than `graph_size`. |
| 2457 | `context_buffer_size = GGML_SCHED_MAX_SPLITS*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(...)` | Same. |
| 2467 | `const int initial_splits_capacity = 16;` | Upstream's initial-capacity pattern already present. |

`git log -S "GGML_SCHED_MAX_SPLITS" -- ggml/` in the ik fork shows the only commit touching the constant is the initial mainline merge (commit `0ceeb117`). The ik fork has never bumped or replaced it.

Upstream `llama.cpp/ggml/src/ggml-backend.cpp` (post PR #9047, merged Aug 2024):

```c
const size_t ggml_sched_max_splits = graph_size;
// at most there is one split for each node in the graph
const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
...
sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(struct ggml_tensor) + ...;
```

Upstream's `GGML_SCHED_MAX_SPLITS` survives only inside `#if 0`-gated debug code (`GGML_SCHED_MAX_SPLITS_DEBUG` for the `causes[]` logging array).

## 3. Fix — the three-line surgical patch

```diff
--- a/ggml/src/ggml-backend.cpp
+++ b/ggml/src/ggml-backend.cpp
@@ ~1768 @@   (inside ggml_backend_sched_split_graph)
                 if (i_split >= sched->splits_capacity) {
                     sched->splits_capacity *= 2;
                     sched->splits = (ggml_backend_sched_split *)realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
                     GGML_ASSERT(sched->splits != NULL);
                 }
-                GGML_ASSERT(i_split < GGML_SCHED_MAX_SPLITS);
                 split = &sched->splits[i_split];

@@ ~2451 @@   (inside ggml_backend_sched_new)
-    const size_t nodes_size = graph_size + GGML_SCHED_MAX_SPLITS*GGML_SCHED_MAX_SPLIT_INPUTS*2;
+    // Upstream PR #9047: scheduler split count is bounded by graph_size
+    // (at most one split per node); the splits array grows dynamically.
+    const size_t ggml_sched_max_splits = graph_size;
+    const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
     sched->node_backend_ids = (int *)calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
     ...
-    sched->context_buffer_size = GGML_SCHED_MAX_SPLITS*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(struct ggml_tensor) + ggml_graph_overhead_custom(graph_size, false);
+    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(struct ggml_tensor) + ggml_graph_overhead_custom(graph_size, false);
```

What this does:

1. **Delete the stale assert at line 1768.** The realloc-on-grow loop already handles capacity. Without this line removed, the assert defeats the entire dynamic-splits machinery the ik fork already ships.
2. **Scale auxiliary array sizes with `graph_size`** instead of the constant. Matches upstream's "at most one split per node" bound. Memory cost grows with model complexity rather than being capped at the fixed 4096-splits worst case.
3. **Leave `#define GGML_SCHED_MAX_SPLITS 4096` in place.** It now only affects the debug `causes[]` array at line 1277 (used when `GGML_SCHED_DEBUG=1`). Cleaning up that path is a separate follow-up, not part of this fix.

Why not bump to a larger constant instead (e.g. 32768):
- A larger constant is a band-aid that re-occurs the moment a more fragmented graph or a bigger model appears. The upstream policy is the architectural fix.
- The dynamic machinery is already present in the ik fork — the constant is the only thing standing in its way. The minimal change that activates it is also the architecturally correct one.
- Memory cost difference is negligible on a 128 GiB host. Both options are cheap; one is right.

## 4. Tasks

### T1 — Author this PHASE doc + SUMMARY entry

- [x] Write `PHASE_GGML_SCHED_DYNSPLITS.md` at repo root.
- [x] Symlink into `docs/PHASE_GGML_SCHED_DYNSPLITS.md` so mdbook picks it up.
- [x] Add entry to `docs/SUMMARY.md` under a new "Scheduler stability" section.
- [x] Commit + push (per CLAUDE.md §5).

**Closure**: doc exists at root, symlink resolves, SUMMARY entry renders in the published site.

### T2 — Apply the patch

- [ ] Edit `ik_llama.cpp/ggml/src/ggml-backend.cpp`:
  - [ ] Delete line ~1768 (`GGML_ASSERT(i_split < GGML_SCHED_MAX_SPLITS);`).
  - [ ] Replace `nodes_size` formula at ~line 2451 with `graph_size`-based local.
  - [ ] Replace `context_buffer_size` formula at ~line 2457 with the same local.
- [ ] Commit in the `ik_llama.cpp` submodule with message referencing upstream PR #9047 and the 2026-05-25 assert. Do NOT push (user owns the upstream destination; push is a separate decision).

**Closure**: `grep -n "GGML_ASSERT(i_split < GGML_SCHED_MAX_SPLITS)" ik_llama.cpp/ggml/src/ggml-backend.cpp` returns nothing; both formulas reference the new `ggml_sched_max_splits` local.

### T3 — Rebuild + redeploy

- [ ] Rebuild llama-server: `cmake --build /home/dconnolly/yarn-agentic/ik_llama.cpp/build -j --target llama-server`.
- [ ] Install: `sudo install -m 0755 -o root -g llm /home/dconnolly/yarn-agentic/ik_llama.cpp/build/bin/llama-server /opt/llm-server/bin/llama-server`.
- [ ] Restart: `sudo systemctl restart llama-server.service`.
- [ ] Verify `/health` returns 200 within 2 s.

**Closure**: service is `active (running)` with a fresh start timestamp, `/health` HTTP 200, `/v1/models` returns the loaded model, a "hi" completion succeeds.

### T4 — Reproduce the original crash scenario (binding verification)

The step is not closed by "service comes back". The crash binds on the prompt-cache restore path against a high-fragmentation cache. Reproducing the failure-inducing sequence and confirming success is what binds this fix per CLAUDE.md §4.

- [ ] Send a long prompt (~9k tokens, mimicking the saved cache from the original incident) to seed a prompt cache.
- [ ] Wait — or force checkpoint accumulation — until `cache state` shows multiple checkpoints with non-trivial size.
- [ ] Send a short, dissimilar prompt ("hi" or similar) to trigger the `apply_checkp` + `kv cache rm` + high-fragmentation path.
- [ ] Confirm no `GGML_ASSERT` in the journal; confirm the second prompt completes normally with token output.

**Closure**: the exact sequence that crashed production at 07:52:46 completes a normal inference cycle on the new binary.

### T5 — Follow-up (separate phase)

- [ ] Gate the `causes[]` static array at `ggml-backend.cpp:1277` behind `#if 0` to match upstream's debug-only pattern, and rename the constant to `GGML_SCHED_MAX_SPLITS_DEBUG`. Removes the last user of the legacy constant.
- [ ] Audit the systemd unit: the wedge survived `Restart=on-failure` because the process didn't exit. Add either `Type=notify` + `WatchdogSec=` if llama-server supports `sd_notify`, or a sibling `llama-server-healthcheck.timer` that probes `/health` and restarts on N consecutive failures.

Tracked here for visibility; out of scope for this fix.

## 5. Open questions

None as of 2026-05-25. The codebase evidence and the upstream reference are unambiguous.

## 6. Cross-references

- Upstream fix: [PR #9047 — ggml : dynamic ggml_sched_max_splits based on graph_size](https://github.com/ggml-org/llama.cpp/pull/9047).
- Originating upstream report: [Issue #9044](https://github.com/ggml-org/llama.cpp/issues/9044).
- Service journal evidence: `journalctl -u llama-server.service --since '2026-05-24 22:31:00'` on host `xeon`.
- Profile: `/home/llm/profiles/qwen36-27b-x1-vanilla.sh` (256k ctx, `--parallel 1`, q4_0 K/V with hadamard rotations, `--cache-ram 40960`, `--ctx-checkpoints 64`, `--no-context-shift`).
