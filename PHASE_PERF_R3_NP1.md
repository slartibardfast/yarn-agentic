# PHASE_PERF_R3_NP1 — nsys-driven characterization post-RT-hardening

**Opened:** 2026-05-27 23:30Z
**Branch:** `production/2026-q2-next`
**Scope:** Detailed kernel + dispatch-gap characterization of the
production binary at the post-RT-hardening shape (submodule `b2cf8fbf`,
flags `--mlockall --rt-prio 50 --cpu-mask 0xF0 --threads 4`,
governor=performance, IRQs pinned 0-3). Establish a fresh nsys-anchored
baseline, identify the actionable bottleneck behind the live-observed
~8 t/s in deep-context thinking-mode vs the PD4 17.9 t/s clean baseline.

**State:** PD pack — no runs yet. Awaits maintenance-window authorization.

**Relation to prior phases:**

- Lifts methodology from `PHASE_PERF_R2_NP1.md` (kernel re-rank, 2026-05-24).
- Lifts harness from `data/cuda-native-dispatch/predesign/pd4-baseline-20260527T121951/`.
- This phase **characterizes**; lever-pulling lands in successor phases.

---

## 0. Goal

**Find the actionable bottleneck** that explains the 17.9 → 8.2 t/s gap
(or prove there isn't one and the difference is workload-shape), with
nsys traces as the primary evidence and per-token kernel + gap
breakdowns as the deliverable.

"Actionable" means: each row in the final report names a specific cost
center (a kernel, a host-side stall, a peer-copy hot spot, a cuBLAS
algo regression), gives the per-token µs cost, and connects to a
candidate fix.

---

## 1. What we know

### Data points

| Source | Build | Workload | Result | Date |
|---|---|---|---|---|
| PD4 baseline | `1db6c2eb` (Phase 46 closure) | NP=1, `N_PREDICT=128`, prompt ~210t, default flags | **17.9 t/s TG** | 2026-05-27 12:19Z |
| Live snoop | `b2cf8fbf` (C-arc + RT) | NP=1, agentic deep-context, thinking-mode | **~8.2 t/s TG** (server slot n_decoded delta) | 2026-05-27 23:01Z |

### What changed between them

| Knob | PD4 | Current |
|---|---|---|
| Submodule | `1db6c2eb` | `b2cf8fbf` (~50 commits later: C-arc C1-C12 + RT additions) |
| `--threads` | 16 | **4** |
| `--cpu-mask` | (none) | **0xF0** (pinned to logical 4-7) |
| `--rt-prio` | (none) | **50** (SCHED_FIFO) |
| `--mlockall` | (off) | **on** |
| CPU governor | powersave | performance |
| IRQ affinity | 0-15 | 30/106/110 → 0-3 |
| Prompt size | ~210 tokens | unknown, agentic (likely 5k-100k) |
| Generation depth | 128 | 1700+ tokens deep |
| Mode | greedy non-thinking | thinking-mode (`reasoning_content` active) |
| Concurrent slots | 0 (clean bench) | (live user load) |

**The two data points are not apples-to-apples in *any* of those rows.**
That alone could explain the gap — we don't know whether the regression
is real until we re-bench at the PD4 shape on the current binary.

---

## 2. Questions to answer

In priority order (each subsequent question is only worth asking if the
prior one comes back ambiguous or "yes, regression"):

**Q1.** On the current binary (`b2cf8fbf` + RT flags), under PD4-identical
workload (NP=1, N_PREDICT=128, ~210t prompt, greedy), what is the TG
rate? *If ≈17.9 t/s, the gap is purely workload-shape; close this phase.*

**Q2.** If Q1 shows regression: which RT flag(s) caused it? Toggle each
off one at a time and re-bench.

**Q3.** What is the per-token kernel breakdown at the PD4 shape? Does it
match the PHASE_PERF_R2_NP1 ranking? *Identifies kernel-level regressions
in the C-arc work.*

**Q4.** What is the per-token kernel breakdown at the deep-context shape?
*Identifies what scales poorly with KV depth + thinking-mode.*

**Q5.** What is the host-side dispatch-gap distribution per token? Are
inter-kernel gaps small (compute-bound) or large (host-bound)? *Decides
whether the bottleneck is on the GPU or on the dispatch thread.*

---

## 3. Variables in flight

The cartesian is intractable. Test plan below picks the cheapest paths
through this space that distinguish the candidate causes.

| Variable | Levels | Cost to flip |
|---|---|---|
| Build | `1db6c2eb` (PD4 ref) ↔ `b2cf8fbf` (current) | medium (rebuild + redeploy a separate binary into a non-prod path) |
| `--threads` | {1, 2, 4, 8, 16} | nil (CLI arg only) |
| `--cpu-mask` | {none, 0xF0} | nil |
| `--rt-prio` | {0, 50} | nil |
| `--mlockall` | {off, on} | nil |
| Governor | {performance, powersave} | nil (sysfs toggle, reversible) |
| Prompt size | {short ~210t, medium ~2k, deep ~50k} | small (prompt file) |
| N_PREDICT | {128, 1024} | nil |
| Mode | {greedy non-thinking, thinking-mode chat} | small (different request body) |
| NP | **1 only** | (NP>1 deferred per 2026-05-05/06 hang risk discussion) |

---

## 4. Methodology

### 4.1 Harness

Two harnesses cover the matrix:

- **Bench harness**: `scripts/test-production-np-determinism.sh` with
  NP_LIST="1" and a fixed PROMPT — this is the exact PD4 harness; lifts
  identically. Captures wall-time TG via the server's `usage` field and
  per-token timing via server stderr `IK_PRINT_TIMING` if enabled.
- **nsys harness**: `scripts/nsys-revisit-pre-port.sh` + `_run-nsys-graphcache.sh`
  pattern (existing). Starts the build-tree binary under
  `nsys profile -t cuda,nvtx,osrt --gpu-metrics-device=all`, runs a
  fixed bench against it, terminates. Output: `.nsys-rep` per run.

### 4.2 nsys capture command (canonical for this phase)

```bash
nsys profile \
    -t cuda,nvtx,osrt \
    --gpu-metrics-device=all \
    --gpu-metrics-frequency=1000 \
    --sample=none \
    --cpuctxsw=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --duration=30 \
    -o "$OUT/trace-${RUN_ID}.nsys-rep" \
    "$BIN" <flags...>
```

`--sample=none` and `--cpuctxsw=none` keep nsys overhead minimal so we
don't deform what we measure. `--gpu-metrics-frequency=1000` (1 kHz) is
enough resolution for per-token granularity at 8-18 t/s.

### 4.3 nsys post-processing — actionable extractions

For every captured trace, generate four reports:

```bash
nsys stats --report cuda_gpu_kern_sum     trace.nsys-rep > kern-sum.txt
nsys stats --report cuda_gpu_kern_exec_sum trace.nsys-rep > kern-exec-sum.txt
nsys stats --report cuda_api_sum          trace.nsys-rep > api-sum.txt
nsys stats --report cuda_gpu_mem_size_sum trace.nsys-rep > mem-sum.txt
```

**Derived metrics per trace:**

| Metric | Formula | Action threshold |
|---|---|---|
| Compute fraction | `Σ(kernel_time) / wall_time` | <60% → host-bound; >90% → compute-bound |
| Per-token wall | `wall_time / n_decoded_tokens` | compare to PD4 baseline µs/token (TBD from PD4 logs) |
| Per-token kernel | `Σ(kernel_time) / n_decoded_tokens` | regression vs PD4 → kernel-level cause |
| Per-token gap | `(wall - kernel) / n_decoded_tokens` | regression vs PD4 → host-dispatch cause |
| Top-N kernels | `kern-sum.txt` top 10 by total ms | new entries or shifts → C-arc regressed something |
| Peer copy bytes/token | `cudaMemcpyPeer*` from `api-sum.txt` ÷ n_decoded | rising → cross-GPU graph-split costing more |

These four numbers per trace are the actionable signal. Each run
appends a row to a table; the diff across rows answers the questions
in §2.

### 4.4 Captured-context controls

For every test in the matrix:

- GPU clocks locked at 1455 MHz (verify pre-run)
- CPU governor = performance (verify pre-run)
- CUBLAS_WORKSPACE_CONFIG=:4096:8 (verify in service env)
- Service stopped before the test; restored after
- Same model file, same prompt file, same seed (where applicable),
  same N_PREDICT
- Each test ×3 reps minimum for variance
- nsys overhead is real — for the t/s-bench column, also capture a
  matching non-nsys run to confirm the nsys-overhead delta

---

## 5. Test matrix (ordered cheapest-first)

Tests are designed so each one is roughly self-contained and produces
its own actionable cell.

### Phase A — re-baseline (answers Q1)

| # | Build | Threads | Mask | Prio | mlock | Prompt | N_PREDICT | Reps | Output |
|---|---|---|---|---|---|---|---|---|---|
| A1 | b2cf8fbf | 4 | 0xF0 | 50 | on | PD4 (~210t) | 128 | 3 | t/s, no nsys |
| A2 | b2cf8fbf | 4 | 0xF0 | 50 | on | PD4 | 128 | 1 | t/s + nsys trace |

**Decision after A1+A2:** if A1 ≈ 17.9 ± 5%, regression is workload-shape;
skip to Phase D. Otherwise proceed to Phase B.

### Phase B — RT-flag bisection (answers Q2)

Each row toggles ONE flag relative to A. Same prompt, same N_PREDICT.

| # | Threads | Mask | Prio | mlock | What it isolates |
|---|---|---|---|---|---|
| B1 | 16 | (none) | 0 | off | "no RT" reference (closest to PD4) |
| B2 | 4 | (none) | 0 | off | --threads 4 alone |
| B3 | 4 | 0xF0 | 0 | off | + cpu-mask |
| B4 | 4 | 0xF0 | 50 | off | + rt-prio |
| B5 | 4 | 0xF0 | 50 | on | + mlockall (= A1 shape) |

3 reps each. The t/s should monotonically degrade (or improve) row-to-
row if the flag is causal; flat row means the prior flag was the cause.

### Phase C — kernel re-rank at PD4 shape (answers Q3)

Single nsys trace at config B1 (no-RT reference) and at A2 (with-RT).
Diff `kern-sum.txt` between them. Top-10 differences are the actionable
cells.

### Phase D — workload-shape characterization (answers Q4 + Q5)

| # | Prompt | N_PREDICT | Mode | Output |
|---|---|---|---|---|
| D1 | short (PD4) | 128 | greedy | nsys (matches A2) |
| D2 | medium (~2k) | 256 | greedy | nsys |
| D3 | deep (~50k synthetic) | 256 | greedy | nsys |
| D4 | thinking-mode chat | 1024 | thinking | nsys via attached PID against live service (short window) |

Diff D1 → D2 → D3: identify what scales linearly vs super-linearly
with KV depth. D4 vs D3 isolates the thinking-mode-specific cost (if any).

---

## 6. Run cards (concrete commands)

Each test gets a unique RUN_ID timestamp + dir under `data/perf-r3-np1/`.

### Phase A skeleton

```bash
RUN_ID=$(date -u +%Y%m%dT%H%M%S)
OUT=/home/dconnolly/yarn-agentic/data/perf-r3-np1/A-baseline-$RUN_ID
mkdir -p $OUT

# A1 — t/s only
sudo systemctl stop llama-server.service
LLAMA_SERVER_BIN=/home/dconnolly/yarn-agentic/ik_llama.cpp/build/bin/llama-server \
NP_LIST="1" N_PREDICT=128 \
DEVICE=CUDA0,CUDA1 TENSOR_SPLIT=1,1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 PORT=18292 \
EXTRA_ARGS="--threads 4 --cpu-mask 0xF0 --rt-prio 50 --mlockall" \
  bash scripts/test-production-np-determinism.sh \
  > $OUT/A1-rep1.log 2>&1
# ... repeat for rep2, rep3

# A2 — same flags + nsys
... see §4.2 capture command, output to $OUT/A2.nsys-rep
```

The `EXTRA_ARGS` plumb-through requires a tiny harness change — the
existing `test-production-np-determinism.sh` doesn't pass arbitrary
extra args. Add an `${EXTRA_ARGS:-}` env hook (one-line change). Lands
as a sub-commit before Phase A starts.

### nsys attach to running production (Phase D4)

```bash
PID=$(systemctl show llama-server.service -p MainPID --value)
nsys profile \
    --attach $PID \
    -t cuda,nvtx,osrt \
    --gpu-metrics-device=all \
    --gpu-metrics-frequency=1000 \
    -d 30s \
    -o $OUT/D4-live-trace.nsys-rep
```

This is the only test that runs against the live production service
(rather than stopping it). Bounded 30s window. Non-disruptive (nsys
attach is sampling-only at the configured rate).

---

## 7. Output format

`data/perf-r3-np1/<phase>-<RUN_ID>/` per run, each with:

- `run-card.md` — what was run, exact flags, host state at start
- `*.log` — raw stderr from llama-server + the bench harness
- `*.nsys-rep` — nsys trace (binary)
- `kern-sum.txt`, `api-sum.txt`, `mem-sum.txt` — nsys post-process outputs
- `derived-metrics.json` — the 6 actionable numbers from §4.3

After all phases complete, top-level `data/perf-r3-np1/REPORT.md` aggregates:

- Table of t/s across the test matrix
- Per-token kernel-cost diff PD4 ↔ current
- Per-token gap diff PD4 ↔ current
- Top-N kernel regressions (if any)
- Workload-shape scaling chart (D1→D2→D3)
- Recommendation: which lever to pull next (becomes the seed for a
  successor phase, e.g. PHASE_PERF_R3_LEVER_X)

---

## 8. Decision tree

```
A1 result:
├── A1 ≈ 17.9 ± 5%        → no regression on clean bench. The 8 t/s in live
│                            is workload-shape (thinking-mode + deep ctx).
│                            Skip to Phase D directly.
│
├── A1 ≪ 17.9             → regression exists.
│                            Run Phase B to isolate the flag.
│                            Then Phase C to get the kernel-level cause.
│                            Then Phase D for full workload picture.
│
└── A1 ≈ partial loss    → ambiguous; run both Phase B and Phase D.

Phase B result:
├── B1 ≈ 17.9             → "no-RT" matches PD4. Bisect onwards: which
│                            flag introduced the loss?
│
└── B1 ≪ 17.9             → loss is NOT in the RT flags. Loss is in the
                              C-arc itself (build delta from 1db6c2eb to
                              b2cf8fbf). Phase C identifies the kernel.
```

---

## 9. Time + token budget

| Phase | Wall (window) | Tokens |
|---|---|---|
| Harness change + dry-run | 15 min | ~5k |
| Phase A (4 runs incl. nsys) | 25 min | ~10k |
| Phase B (15 runs, 3 reps × 5 configs) | 60 min | ~15k |
| Phase C (nsys diff + report) | 30 min | ~10k |
| Phase D (4 nsys captures incl. live-attach) | 30 min | ~15k |
| REPORT.md aggregation + commit | 20 min | ~10k |
| **Total** | **~180 min** | **~65k** |

The Phase A→B→C→D path is **sequential** because each phase's decision
depends on the prior phase's result. Optimistic short-circuit: if
A1 ≈ 17.9, Phases B and C are skipped, total drops to ~90 min.

---

## 10. Risks

1. **nsys overhead deforms the measurement.** Mitigation: every nsys
   trace is paired with a non-nsys reference run; the actionable
   numbers are nsys-derived BUT the t/s comparison uses the non-nsys
   run. The trace shape (relative kernel costs) is more
   important than absolute timing.

2. **Phase D (live attach) interferes with production.** Mitigation:
   30s window only, `--sample=none`. Worst case: the live request
   sees a few ms of extra latency during the window. Tested OK in
   prior nsys-graphcache runs.

3. **The deep-context synthetic prompt (D3) needs to be valid for the
   Qwen 3.6 chat template.** Mitigation: build it from real chat
   history + repeated padding to target token count; verify it parses
   through the jinja template before profiling.

4. **B-series bisect creates a config that hits the 2026-05-05/06
   `--parallel 2` issue.** Mitigation: NP=1 only in this phase.
   `--parallel 2` is explicitly out of scope per the project-level
   discussion.

5. **GPU clocks unlock mid-run.** Mitigation: re-verify before every
   test row. `gpu-clocks.sh status` cheap.

---

## 11. Acceptance — phase closes when

- [ ] Phase A results table populated with 3-rep mean+stddev t/s
- [ ] If applicable, Phase B isolation row identifying the causal flag
- [ ] Phase C kernel-diff table with top-10 regressions or "no diff"
- [ ] Phase D scaling chart D1→D2→D3 + thinking-mode delta from D4
- [ ] REPORT.md committed to `data/perf-r3-np1/`
- [ ] Decision recorded: either (a) "no perf regression, workload-shape
      explains the live obs" and close, or (b) "regression localized
      to <X>, next phase is <Y>" and open the successor phase

---

## 12. Out of scope

- NP>1 anything (per 2026-05-05/06 host-hang risk discussion)
- Kernel-level optimization work (this phase characterizes; levers go
  in successor phases per the PHASE_PERF_R2_NP1 pattern)
- DFlash speculative decoding (production runs vanilla; bringing back
  DFlash is its own phase)
- Cross-engine vs vLLM comparison (PHASE_T6_CHARACTERISATION T6.0
  covered this; not duplicating here)
- Multi-host or multi-machine perf (single-host xeon characterization
  only)
