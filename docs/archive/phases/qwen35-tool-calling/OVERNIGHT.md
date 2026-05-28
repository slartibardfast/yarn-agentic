# Overnight Autoresearch Plan — qwen35-mtp profiling sweep

One-shot plan for a ~12-hour autonomous run. Profiling-first: the goal is
**decision data for where to point future optimization effort**, not more
Phase 2 tool-calling accuracy numbers.

The Phase 2 harness produces a clean ranking with two models (Qwen3.5-9B
wins on abstention, Hermes-3 hallucinates tool names on negatives) and
nothing a third or fourth model adds is going to flip that. So I'm
demoting accuracy work from the overnight plan and putting compute-side
measurements in its place.

## Overall goal

Answer four questions the user will want on waking:

1. **Where does the time actually go?** Per-op GPU time distribution
   during a realistic Qwen3.5-9B q4km forward pass on Vega 64, split
   across prompt eval and generation. Should point clearly at the top 3
   hotspots that future optimization should target.
2. **Was Phase 3/4 fusion worth it?** A quantitative A/B of the
   `GGML_VK_DISABLE_FUSION` runtime toggle — tokens/sec delta, per-op
   time delta, graph split count delta, for both q4km and f16-cache
   configs.
3. **Does MTP survive realistic context?** A context-length sweep of
   the MTP draft acceptance rate and effective speedup for Qwen3.5-9B
   q4km at n_ctx = 1K, 4K, 8K, (16K if VRAM permits).
4. **Which KV-V quant is actually worth it on this hardware?** An 8-row
   matrix (f16 + 7 quants) of throughput, VRAM, and output similarity
   to f16 at a fixed seed.

The profiling data is **cold** in the sense that it doesn't ship a fix,
but it is the input to every future Phase 5/6/7 decision. That's the
overnight trade: burn compute, produce a prioritization signal.

## What I already know I will NOT do tonight

- No shader work. Phase 4 is frozen.
- No rebuilds of `/home/llm/src/qwen35-mtp/build-vk/`. `test-backend-ops`
  and `llama-bench` are not built; tonight's plan works around that.
- No polaris fork edits.
- No model downloads beyond what's already on disk (`Qwen3.5-9B-mtp-q4km`,
  `Qwen3.5-9B-mtp-f16`, `Hermes-3-Llama-3.1-8B.Q4_K_M`).
- No destructive git operations (`reset --hard`, force push, `clean -fd`).
- No `rm -rf` outside `harness/results/` and the HF cache dir.
- No blocking polls longer than the Bash tool's 2-second rule.

## Sequence

Nine steps. Each has a go/no-go criterion, an expected artifact, a rough
time budget, and a fallback. If a step fails the go criterion twice in a
row, log the failure and move to the next step — do not spin or retry.

All outputs go under:

```
phases/qwen35-mtp/profile/                 # new directory tonight
├── overnight.log                          # append-only run log, created step 0
├── per-op/                                # category A
├── fusion-ab/                             # category B
├── mtp-sweep/                             # category D
├── kv-matrix/                             # category C
├── vram-budget/                           # category G
├── pipeline-stats/                        # bonus
└── vega-vs-navi/                          # optional category F
```

Each subdirectory holds raw stderr dumps, parsed JSON summaries, and
where applicable the harness that drove the run.

### Step 0 — Environment sanity (< 5 min)

- `df -h /` — abort the whole run if `/` has < 10 GB free. Current is 23 GB.
- `pgrep -af llama-server` — `pkill -9 -f build-vk/bin/llama-server`.
- `cd /home/llm/yarn-agentic/.claude/worktrees/llama && git status` —
  must be clean. Abort otherwise.
- `cd /home/llm/src/qwen35-mtp && git status` — must be clean on
  `vulkan-phase4`. Abort otherwise.
- `git log --oneline -1` on both repos → record to `overnight.log`.
- Create `phases/qwen35-mtp/profile/` and the subdirectories above.
- Write the abort-on-regression baseline: the known-good Qwen3.5-9B q4km
  startup with `-fa on --cache-type-v tq_v_4b` should give `graph splits = 1`.
  Record this to `overnight.log` as a reference to compare morning-side.

**Go:** all checks pass, log file created, `profile/` skeleton in place.
**No-go:** any check fails → abort the whole run, log to overnight.log,
leave a `WAKE_USER` note at the top of the log.

### Step 1 — Per-op profiling, Qwen3.5-9B q4km, Vega 64 (~45 min)

**Purpose:** category A — answer "where does the time actually go?".

Use `GGML_VK_PERF_LOGGER=1` + `GGML_VK_PERF_LOGGER_CONCURRENT=1` +
`GGML_VK_PERF_LOGGER_FREQUENCY=5` (print every fifth graph for denoising).
Start llama-server on Vega 64 with `--cache-type-v f16` (baseline config
that matches the Phase 2 harness baseline). Drive it with a fixed 512-token
prompt via `/completion` and let it generate 256 tokens, three times.

Then kill the server, parse the stderr into a JSON summary (per-op total
time, per-op call count, per-op avg time, GFLOPS where reported), and
emit:

```
profile/per-op/qwen35-9b-q4km-f16kv-vega-<UTC>.stderr
profile/per-op/qwen35-9b-q4km-f16kv-vega-<UTC>.json
```

Top 10 ops by total time go into `overnight.log` as a preview.

**Go:** JSON summary has ≥ 5 distinct op entries, total time > 0.
**No-go:** perf logger produced no output → check env var was set in
server environment, try once more with `GGML_VK_PERF_LOGGER_CONCURRENT=0`
(synchronous mode), log result either way.
**Time:** 45 min.

### Step 2 — Fusion A/B for the same workload (~60 min)

**Purpose:** category B — quantify Phase 3/4.

Run the exact same workload from step 1 twice more:

- Fusion ON (default — already captured in step 1, no re-run needed).
- Fusion OFF via `GGML_VK_DISABLE_FUSION=1`.

Also A/B the two adjacent fusion axes separately if there's time:

- `GGML_VK_DISABLE_MULTI_ADD=1` — the multi-add rms fusion.
- `GGML_VK_DISABLE_GRAPH_OPTIMIZE=1` — the graph-optimize pass.

Artifacts per run:

```
profile/fusion-ab/fusion-off-<UTC>.stderr
profile/fusion-ab/fusion-off-<UTC>.json
profile/fusion-ab/multi-add-off-<UTC>.{stderr,json}
profile/fusion-ab/graph-opt-off-<UTC>.{stderr,json}
```

Plus a derived comparison file:

```
profile/fusion-ab/delta-<UTC>.json
```

…with top-10 ops where fusion-off saw the biggest regression, and
top-10 ops where fusion actually hurt (if any — surprises are the
interesting signal).

**Go:** fusion-off report has strictly higher per-op time on at least
RMS_NORM_MUL or SILU_MUL (if those are still getting fused on q4km).
**No-go:** no meaningful delta → log and move on. This either means
fusion isn't kicking in (config problem) or Phase 3/4 isn't touching
this particular workload's hot ops.
**Time:** 60 min for all three A/Bs.

### Step 3 — MTP draft acceptance context sweep (~60 min)

**Purpose:** category D — answer "does MTP survive long context?".

Extend `run_eval.py` (or write a new small script) to:

- Accept a `--context-length` parameter.
- Send a synthetic prompt padded to approximately that many tokens via
  `/completion` (not `/v1/chat/completions` — we want the `timings`
  dict which includes `draft_n` and `draft_n_accepted`).
- Run for a fixed 256-token generation.
- Record: `prompt_per_second`, `predicted_per_second`, `draft_n`,
  `draft_n_accepted`, acceptance ratio.

Run the sweep at:

- n_ctx = 1024
- n_ctx = 2048
- n_ctx = 4096
- n_ctx = 8192 (if VRAM permits — Qwen3.5-9B q4km + f16 KV should fit)
- n_ctx = 16384 (if VRAM permits)

Always use Qwen3.5-9B q4km on Vega 64 (the only MTP-enabled model
we have). Temperature = 0, seed = 42, draft settings unchanged.

Artifacts:

```
profile/mtp-sweep/mtp-ctx-sweep-<UTC>.json
profile/mtp-sweep/mtp-ctx-sweep-<UTC>.md     # human-readable table
```

**Go:** at least 3 rows of data collected, acceptance > 0 on all of them.
**No-go:** acceptance = 0 across all rows → the server is running
without MTP (sanity check: `-fa on` doesn't affect MTP, but maybe the
server args are wrong). Log and move on.
**Time:** 60 min.

### Step 4 — KV-V quant throughput + fingerprint matrix (~3 hours)

**Purpose:** category C — answer "which `--cache-type-v` should I default
to for agent work on this hardware?".

For each of:

```
f16 q4_0 q4_1 q5_0 q5_1 q8_0 iq4_nl tq_v_4b
```

and **only** the Qwen3.5-9B q4km model on Vega 64:

1. Start the server with `--flash-attn on --cache-type-v <type>` and
   also `GGML_VK_PERF_LOGGER=1` for per-op capture.
2. Grep `graph splits` from startup — must be 1, else mark as REGRESSION
   and skip.
3. Record from the startup log: `Vulkan0 KV buffer size` lines (K size
   and V size).
4. Drive a fixed prompt (same one used in step 1) via `/completion`
   with `n_predict=128`, `temperature=0`, `seed=42`. Record:
   - `prompt_per_second`
   - `predicted_per_second`
   - first 32 output tokens (as a "fingerprint" hash — if any row
     differs from f16 by more than ~25% of tokens we know the quant is
     hurting quality)
5. Kill the server. Parse the perf-logger output and keep only the top
   10 ops.
6. Repeat 3× per type for median stability, re-using the server instance
   for speed (no restart between the 3 runs).

Artifacts:

```
profile/kv-matrix/kv-<type>-<UTC>.stderr
profile/kv-matrix/kv-<type>-<UTC>.json   # k/v sizes, t/s, fingerprint, top-10 ops
profile/kv-matrix/kv-matrix-summary-<UTC>.json
profile/kv-matrix/kv-matrix-summary-<UTC>.md   # markdown table for RESULTS.md
```

**Go:** all 8 rows populated OR ≥ 6 rows populated. The f16 row must
exist (it's the reference fingerprint).
**No-go:** `graph splits != 1` on any row → that's a Phase 4 regression
and should wake the user. Log with `WAKE_USER` marker in overnight.log,
but still continue collecting the other rows.
**Time:** 3 hours is pessimistic (24 min/row ×  8 rows); realistic is
closer to 90 min.

### Step 5 — VRAM budget at scaling contexts (~45 min)

**Purpose:** category G — answer "where does Qwen3.5-9B OOM on Vega 64?".

Use `GGML_VK_MEMORY_LOGGER=1` (newly discovered knob, gives detailed
VRAM alloc/free traces) plus `llama_memory_breakdown_print` output.

For Qwen3.5-9B q4km + `-fa on` + `--cache-type-v f16`, attempt server
startup at:

- n_ctx = 4096
- n_ctx = 8192
- n_ctx = 16384
- n_ctx = 32768
- n_ctx = 65536

For each: record whether startup succeeded, the `llama_memory_breakdown_print`
output, peak VRAM, headroom, and where the OOM line falls.

Then do the same sweep with `--cache-type-v q4_0` and `--cache-type-v tq_v_4b`
to measure how much further the quant pushes the cliff.

Artifacts:

```
profile/vram-budget/ctx-<n>-v-<type>-<UTC>.stderr
profile/vram-budget/vram-budget-summary-<UTC>.json
profile/vram-budget/vram-budget-summary-<UTC>.md
```

**Go:** at least 3 rows of the f16 sweep landed.
**No-go:** server refuses to start even at n_ctx=4096 → environment
regression, log with WAKE_USER.
**Time:** 45 min.

### Step 6 — Pipeline stats bonus (~30 min, optional)

**Purpose:** bonus — learn which kernels are register-pressured or
shmem-pressured on Vega.

Single server startup on Vega 64 with
`GGML_VK_PIPELINE_STATS=flash_attn` then kill. Capture the register /
shared memory usage stats for all flash-attention pipeline variants
(there are many after Phase 4). Then repeat with filter
`mul_mat`, `rms_norm`, `rope`.

Artifacts:

```
profile/pipeline-stats/pipeline-stats-<filter>-<UTC>.stderr
profile/pipeline-stats/pipeline-stats-summary-<UTC>.md
```

**Go:** at least one pipeline entry with non-zero register count.
**No-go:** unsupported on Vega (VK_KHR_pipeline_executable_properties
may not be present) → log and skip.
**Time:** 30 min.

### Step 7 — Vega vs Navi head-to-head (~60 min, optional)

**Purpose:** category F — Phase 4 correctness regression check across
GPUs + a cross-architecture timing comparison.

Repeat step 1 (per-op profile, same workload) but with
`GGML_VK_VISIBLE_DEVICES=0` (Navi 21) instead of `=1` (Vega). Same
model, same prompt, same seeds.

Artifacts:

```
profile/vega-vs-navi/navi-per-op-<UTC>.stderr
profile/vega-vs-navi/navi-per-op-<UTC>.json
profile/vega-vs-navi/vega-vs-navi-diff-<UTC>.md
```

**Go:** Navi run completes, top-10 ops table comparable to step 1's.
**No-go:** Navi produces different output bytes than Vega for
identical inputs → log as a CORRECTNESS issue and continue.
**Time:** 60 min (Navi warmup is slower than Vega for reasons).

### Step 8 — Op coverage analysis (fallback for category E, ~45 min)

**Purpose:** answer "which ops are still landing on CPU?".

With `test-backend-ops` unavailable, use the scheduler's own debug
output instead. Start the Vega server with
`GGML_SCHED_DEBUG=2 GGML_VK_VISIBLE_DEVICES=1` and one fixed
completion request. Grep the scheduler dump for:

- Every `split` entry's source and destination backend.
- Every tensor that's being copied between Vulkan and CPU buffers
  during a forward pass.
- Any `supports_op = false` log line from `ggml_vk_supports_op`.

Emit a summary:

```
profile/op-coverage/qwen35-9b-q4km-coverage-<UTC>.json
profile/op-coverage/qwen35-9b-q4km-coverage-<UTC>.md
```

…listing the total op count in the forward pass, how many ran on
Vulkan, how many landed elsewhere, and which ones they were. After
Phase 4 I expect the answer to be "0 ops fall off Vulkan", but it's
worth confirming.

**Go:** at least one split entry parsed from the debug output.
**No-go:** GGML_SCHED_DEBUG=2 produces no extra info vs `=1` → fall
back to level 1 and best-effort parse. Log and move on.
**Time:** 45 min.

### Step 9 — RESULTS.md synthesis + handoff (~60 min)

**Purpose:** make the data readable by a human.

Extend `RESULTS.md` (or rename it if the focus is now profiling-centric
— tentatively `PROFILING.md`) with new sections:

1. **Per-op time distribution** — table from step 1, top 15 ops.
2. **Fusion A/B delta** — table from step 2, with per-op deltas.
3. **MTP context sweep** — table from step 3.
4. **KV-V quant matrix** — table from step 4.
5. **VRAM budget curve** — table from step 5.
6. **Pipeline stats** — bullet list from step 6 (if any).
7. **Vega vs Navi delta** — table from step 7 (if any).
8. **Op coverage** — bullet list from step 8.
9. **Observations** — 4-6 bullets on what surprised us, what validates
   Phase 4, what the data says about Phase 5 priorities.
10. **Future work prioritization** — 3-5 bullet proposals for Phase 5+
    based on the data, not guesses.

Plus a morning handoff section at the top of `overnight.log`:

```
=== WAKE-UP SUMMARY ===
Steps completed: N/9
Steps skipped:   M (with reasons)
Artifacts:       profile/per-op/*.json, profile/fusion-ab/*.json, …
Warnings:        (any WAKE_USER markers propagated here)
Next action:     (one sentence)
```

Commit the RESULTS.md (or PROFILING.md), push to origin/main, kill
any lingering server, and exit.

**Go:** a non-trivial RESULTS update is committed + pushed.
**No-go:** nothing gathered in steps 1-8 → commit a `FAILED: see
overnight.log` marker and exit.
**Time:** 60 min.

## Failure contract

- **Server crashes mid-step** → kill cleanly, log the crash + stderr
  tail, retry **once** with a smaller ctx size, then skip the step.
- **Harness script exception** → log traceback, move to next step.
  Do not edit `run_eval.py` except in step 3 (MTP context sweep).
- **Disk fills below 10 GB** → stop all background work, cleanup HF
  cache download dir, resume.
- **Uncommitted changes accumulate** → commit with `wip: overnight
  step N` message on the correct branch before moving on. Never lose
  data.
- **A `WAKE_USER` marker is produced** → still continue the rest of
  the plan, but prepend the marker to `overnight.log`'s WAKE-UP
  SUMMARY so the user sees it first on waking.

## Safety rules (unchanged from v1 of this plan)

- No `git reset --hard`, no `git clean -fd`, no force-push.
- No `rm -rf` outside `harness/results/` or the HF download cache.
- No rebuilds of `build-vk/`.
- No `--no-verify`, `--no-edit`, or signing bypass on any git action.
- Every step logs BEFORE it starts and AFTER it ends so the morning
  handoff can reconstruct what happened.

## Rough time budget

| Step | Budget | Cumulative |
|------|-------:|-----------:|
| 0 — sanity | 5 min | 5 min |
| 1 — per-op profile | 45 min | 50 min |
| 2 — fusion A/B | 60 min | 1h 50 |
| 3 — MTP ctx sweep | 60 min | 2h 50 |
| 4 — KV-V quant matrix | 180 min | 5h 50 |
| 5 — VRAM budget | 45 min | 6h 35 |
| 6 — pipeline stats | 30 min | 7h 05 |
| 7 — Vega vs Navi | 60 min | 8h 05 |
| 8 — op coverage | 45 min | 8h 50 |
| 9 — RESULTS writeup | 60 min | 9h 50 |
| **total** | **~9h 50** | |

Leaves ~2h of slack inside the 12-hour window. Overflow plan if I
finish early:

- Rerun step 2 with `GGML_VK_DISABLE_ASYNC=1` to see whether async
  submit is buying anything.
- Rerun step 2 with `GGML_VK_DISABLE_MMVQ=1` to see whether quantised
  matmul-vec is the right default for Qwen3.5-9B on Vega.
- Rerun step 4 with Hermes-3-8B q4km to see how the quant matrix
  shifts for a different architecture.
- Re-run the Phase 2 harness at temperature=0.7 on Qwen3.5-9B to check
  whether the accuracy numbers hold under sampling.

Stretch goals only — not promises.

## What this plan explicitly deprioritizes

- **More comparison models in the Phase 2 harness.** Two models give
  enough ranking signal to commit a 2-model RESULTS.md note and move
  on. A third model would not change the "over-eager vs. conservative"
  story we already have, and the night's compute is better spent on
  profiling.
- **New hard cases.** The 92-case set is sufficient for now. If a
  profiling step reveals a specific prompt shape that stresses the
  server, we can add test cases in the morning.
- **New features in `run_eval.py`.** Only the MTP-instrumentation
  change in step 3 is in scope; every other harness edit is off-limits.

## Success criteria on waking

1. `phases/qwen35-mtp/profile/` exists and contains per-op JSON data
   for at least 2 of {step 1, step 2, step 4}.
2. The commit history on `main` shows `wip: overnight step N` commits
   for every step that ran.
3. `git status` is clean on both repos.
4. `llama-server --version` still returns build `b8783-71ba1ed4a`.
5. Regression check: `Qwen3.5-9B-mtp-q4km.gguf` with `--cache-type-v
   tq_v_4b --flash-attn on` still starts with `graph splits = 1`.
6. `phases/qwen35-mtp/RESULTS.md` (or `PROFILING.md`) has a new
   "Per-op time distribution" table at minimum.

If all 6 hold, the run is a success regardless of how many of the
9 steps individually completed.
