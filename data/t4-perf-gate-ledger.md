# T4.7 perf-gate ledger

PHASE_NSTREAM_KV_PERF T4.7 — measurement of record for the GP4.i
gate after the T4 coherent flip (chunked-prefill admission, Sarathi-Serve).

**Bench commit:**
- Parent `eb426e0` (T4 coherent flip submodule bump)
- Submodule `e282d229` (T4 chunked-prefill admission landing)

**Common bench config (all rows unless noted):**
- Model: `/opt/models/recast-out/qwen3.6-27b-V-F1.T1.lm_head-f16.gguf`
- Hardware: 2× Quadro RTX 6000 (TU102, sm_75, 24 GiB each), CUDA 13.2,
  driver 595.58.03
- Split: `--device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1`
- KV: `--cache-type-k q4_0 --cache-type-v q4_0 --k-cache-hadamard
  --v-cache-hadamard`
- FA on, `-ngl 999`, `--threads 16`, `--batch-size 2048
  --ubatch-size 512`, `--no-context-shift`
- CTX_PER_SLOT = 4096 (CTX_TOTAL = 32768 at NP=8) — same as T3.8 M3
  number of record
- N_PARALLEL = 8, N_PREDICT = 200, N_RUNS = 3 per config
- Locked clocks: 1455 MHz SM on both GPUs (verified via
  `nvidia-smi --query-gpu=clocks.current.sm`)
- `--prefill-chunk-budget 0` (= n_ubatch = 512) for all C1 configs

**Capture format per row:**

| ID | Config | TG t/s mean | σ | CV % | wall s | Notes |

---

## C0 — pre-T4 baseline (T3.8 M3 number of record)

**Bench code state:** pre-T4 coherent flip (submodule HEAD at
`0759c01c` and earlier — `active_pp_slot_id`
PrefillSerialisationGate active).

**Bench CLI:** identical to `scripts/bench-t3.8-m3.sh` defaults.

| Run | Aggregate t/s | wall_s | total_tokens |
|---|---|---|---|
| 1 | 26.50 | 60.38 | 1600 |
| 2 | 26.46 | 60.47 | 1600 |
| 3 | 26.49 | 60.40 | 1600 |

**Mean = 26.49 t/s aggregate, σ = 0.037, CV = 0.14%.**

Source: `data/t3.8-perf-gate-ledger.md` M3 row.

## C1-steady — T4 at M3-steady workload (post coherent flip)

**Bench code state:** post-T4 coherent flip (submodule `e282d229`).
Chunked admission active; `--prefill-chunk-budget 0` defaults
K = n_ubatch = 512.

**Bench CLI:** identical to `scripts/bench-t3.8-m3.sh` defaults (8
concurrent prompts started simultaneously, fixed n_predict=200).

**Output dir:** `data/t4-c1-steady-20260522-194853/`

| Run | Aggregate t/s | dispatch counter |
|---|---|---|
| 1 | 26.481 | (not emitted at default 64-tick interval over this run) |
| 2 | 26.503 | same |
| 3 | 26.477 | same |

**Mean = 26.49 t/s aggregate, σ = 0.014, CV = 0.05%.**

VRAM probe lines indicate 12–16 graphs per device, 373–377 nodes,
96–97 KB host bookkeeping — pool stable run-to-run (no leak).

## C1-staggered — T4 at M3-staggered workload (5s arrival offsets)

**Bench code state:** post-T4 coherent flip (submodule `e282d229`).

**Bench CLI:** `scripts/bench-t4-m3-staggered.sh`. 8 prompts fire at
arrival offsets t = 0, 5, 10, 15, 20, 25, 30, 35 seconds. Same model,
ctx, --k/v-cache-hadamard, K_BUDGET=0 (= n_ubatch).

**Output dir:** `data/t4-c1-staggered-20260522-195252/`

| Run | Aggregate t/s | dispatch counter |
|---|---|---|
| 1 | 21.622 | total=1600 multi_seq=0 |
| 2 | 21.630 | total=1600 multi_seq=0 |
| 3 | 21.599 | total=1600 multi_seq=0 |

**Mean = 21.62 t/s aggregate, σ = 0.016, CV = 0.07%.**

Per-tick batches under staggered admission are predominantly
single-seq (split_equal does not group because per-seq token counts
diverge — slots in PROCESSING contribute 1 decode token, slots in
LOAD_PROMPT contribute K-or-fewer prefill tokens, the shapes don't
match for multi-seq grouping). This is expected behaviour under T4 +
T3.5 (multi-seq dispatch fires when slots agree on per-seq count;
mixed prefill+decode batches do not satisfy that condition).

VRAM probe: 8–11 graphs per device, 227–276 nodes, 58–71 KB host
bookkeeping. Smaller pool than C1-steady because fewer unique batch
shapes are exercised.

---

## T4.7 — Gate verdict (binding numbers)

**Gates:**
- **GP4.i.a (regression band):** C1-steady mean ≥ C0 mean × 0.98 =
  **25.96 t/s** (hard binding) — **PASS** (26.49 ≥ 25.96; zero
  regression, identical to C0 within noise).
- **GP4.i.b (uplift binding):** C1-staggered mean ≥ C0 mean × 1.20 =
  **31.79 t/s** (hard binding for declaring T4 GREEN) — **FAIL**
  (21.62 < 31.79; in fact 18.4% **below** C0 baseline).
- **GP4.i.c (variance):** CV ≤ 1% per config — **PASS** (0.05% for
  C1-steady, 0.07% for C1-staggered; well within the locked-clocks
  bench discipline).

**Verdict: FAIL on GP4.i.b. T4.7 closes as honest negative.**

**Why GP4.i.b fails — structural, not implementation.** The gate as
specified targets aggregate t/s. Under staggered arrival the wall
clock is bounded below by the arrival window (35s) plus the last
prompt's decode time (~10s when only it is still running), giving
a minimum wall of ~45s for 1600 tokens → max throughput ~35.6 t/s
**if** the system perfectly saturated every multi-slot moment.
Actual is 21.62 t/s because only the late-tick window has all 8
slots in decode concurrently; the early ticks (only slots 0–2
active) run at near-single-slot throughput (24 t/s) which is the
multi-slot kernel's gross throughput divided by the active slot
count. The steady-arrival baseline (C0 = 26.49) is the **maximum**
sustained throughput on this hardware at this NP — staggered
arrival cannot exceed it on aggregate-t/s because steady
**already** saturates the multi-slot kernel.

T4 admission would deliver staggered-uplift only if pre-T4 admission
had held idle slots while a single slot prefilled. Pre-T4 (post-N3)
already runs decode concurrently with prefill (DecodeHoldGate
removed), and prefill is fast enough (~60 t/s prefill) that the
serialized single-slot-at-a-time prefill of pre-T4 doesn't bottleneck
the staggered workload either. T4 admission's value
is in workloads where prefill demand is HIGH RATE (many short
prompts in burst arrival) or where prompts are LONG (chunked
admission keeps decode going during the prefill tail) — neither
condition is exercised by 200-token-prompt steady or staggered M3.

**Correctness layer (T4.6 GP4.j / k / l / m / n + T4.1 unit) stays
GREEN** — the chunked-prefill admission, spec layer, trace producer,
and unit test are all load-bearing for future work even though this
phase's perf measurement is negative. Per
`[[feedback_oneshot_then_evaluate]]`: "negative results land cheap
when honest". Per `[[feedback_no_followup_cover]]` /
`[[feedback_no_workarounds]]`: the gate is recorded as FAIL with
measurement of record; no follow-up cover; the work as-defined is
closed.

**What this means for production.** Production profile
(`qwen36-27b-x2-dflash.sh`, NP=2 + DFlash) is UNCHANGED. The
T4 admission code is in the tree but gated behind the
`--prefill-chunk-budget` CLI flag — when unset/0 it defaults to
n_ubatch which is the legacy n_batch chunk size, so existing
production runs are byte-identical (verified at NP=1248 by GP4.k
under T4.6). Future work that targets the workloads where T4
actually delivers (burst short-prompt, long-prompt prefill) can
build on this scaffold.

---

## VRAM and dispatch counter (informational)

VRAM probe lines and dispatch counter from server logs are captured
per-run in `data/t4-c1-steady-*/server-run*.log` and
`data/t4-c1-staggered-*/server-run*.log`. T4 admission keeps the
graph pool bounded (T3.6.M finding); the dispatch counter should fire
at 90%+ multi-seq ratio under T4 just as it did under T3.5 (T4 only
changes WHICH tokens are in the batch, not the per-tick dispatch
shape).
