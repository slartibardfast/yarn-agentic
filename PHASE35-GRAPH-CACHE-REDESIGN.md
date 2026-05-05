# PHASE 35 — CUDA Graph Cache: Instrumentation, Topology-Class Keying, Allocation-Aware Eviction

> **Status (working doc):** Plan landed; no code phases shipped yet.
> Phase ordering: A (instrumentation) → A.gate (decision) → B + E (the
> actual redesign) → C/D (conditional) → F (post-mortem amendments).
> Each phase opens with **test contracts (RED)** that must be written
> and failing before any of that phase's implementation is touched.

## 1. Context

PHASE 34 (`PHASE34-LEAK-RCA.md`, AMENDED 2026-05-05) attributed the
production CUDA OOM to the `cuda_graphs` cache cap of 128 — but the
attribution overreached. The "~25–30 MiB per cached
`cudaGraphExec_t`" figure was fabricated to fit the observed 3.5 GiB
GPU drift, not measured. The 128-entry cap itself is uncommitted local
work added in this same session as a stop-gap; pre-our-MTP-work the
cache had no cap. Other plausible drivers (`ggml_cuda_pool` HWM,
cuBLAS workspace, driver-side per-launch state) were not ruled out.

This phase does the work right, in order:

1. **Land instrumentation that actually measures the relevant
   numbers.** Decide whether the graph cache is even the leak source
   before redesigning it.
2. **Redesign around what the existing infrastructure already
   supports** — `cudaGraphExecUpdate` is wired; `cpy_dest_ptrs` is a
   working precedent for per-launch parameter patching.
3. **Replace the count-based cap with allocation-aware eviction**
   driven by `cudaMemGetInfo` headroom.
4. **Re-audit and amend PHASE34** once measurement confirms or
   refutes its causal claim.

Test-first throughout: every phase writes its assertion harness before
its production code.

## 2. Critical files

| Path | Role |
|------|------|
| `ik_llama.cpp/ggml/src/ggml-cuda.cu` | cache control flow, key derivation, capture/instantiate/update sites |
| `ik_llama.cpp/ggml/src/ggml-cuda/graph.cuh` | `ggml_cuda_graph` struct (instance, graph, node properties, indirection state) |
| `ik_llama.cpp/ggml/src/ggml-cuda/common.cuh:862` | `cuda_graphs` map declaration on `ggml_backend_cuda_context` |
| `ik_llama.cpp/ggml/include/ggml-cuda.h` | accessor surface: extend `ggml_backend_cuda_graph_cache_size` family with new probes |
| `ik_llama.cpp/tests/test-cuda-graph-cache-bounded.cpp` | prior-art test harness; templates the new tests |
| `ik_llama.cpp/tests/CMakeLists.txt` | register new `test-cuda-graph-*` binaries with CTest |
| `yarn-agentic/scripts/cuda-graph-probe/` | new dir: integration test harnesses + analysis scripts |
| `yarn-agentic/PHASE34-LEAK-RCA.md` | needs second amendment after Phase A measurement (Phase F1) |
| `yarn-agentic/PHASE35-GRAPH-CACHE-REDESIGN.md` | this doc; updated as findings land |
| `yarn-agentic/SUMMARY.md` | add Phase 35 entry |
| `profiles/qwen36-27b-x1.sh` | env knobs for probes + new safety margin |
| `~/.config/systemd/user/llama-server.service` | `Environment=` for the probe knobs |

Specific edit sites in `ggml-cuda.cu`:

- `:541` — `ggml_backend_cuda_graph_cache_size`: extend with sibling
  accessors (see §3.4 Accessor surface).
- `:4241-4256` — `ggml_cuda_graph_get_key`: split into `_shape_key`
  (current behaviour; preserved) and new `_topology_key` (skip ne hash).
- `:4280-4297` — `ggml_cuda_get_graph`: replace count-cap eviction with
  allocation-aware eviction (Phase E); add hit-count and last-use
  bookkeeping.
- `:4364-4387` — `cpy_dest_ptrs` / `dest_ptrs_d` precedent (Phase C
  generalises this; do not modify in B).
- `:4393-4404` — `set_ggml_graph_node_properties`: capture op_params
  for the comparator's strict-on-op_params rule (Phase B).
- `:4406-4442` — `ggml_graph_node_has_matching_properties`: relax for
  topology-class comparison (ne mismatch OK iff op_params match).
- `:4444-4485` — `is_cuda_graph_update_required`: under topology-key
  routing, ne diffs are now Update-payload, not "needs new entry."
- `:4487-4512` — `update_cuda_graph_executable`: instrument with
  chrono+cudaEvent timing (Phase A).
- `:4657` — call site of `ggml_cuda_get_graph` from
  `ggml_backend_cuda_graph_compute`: switch to topology-key probe
  with shape-update fallback. Verify nullptr-eager fallback exists
  here as a precondition for Phase E.
- `:4665, 4672-4677, 4687-4694` — `disable_due_to_*` flag block; new
  `disable_due_to_vram_pressure` joins the family (Phase E).

Specific edits in `graph.cuh`:

- `:14-42` — `ggml_cuda_graph` struct gets new fields: `hits`,
  `last_use_us`, `topology_key`, `current_shape_key`,
  `disable_due_to_vram_pressure`, `failed_update_op_params_signatures`
  (the per-(topology, op_params) blacklist from Phase B5).

## 3. Cross-cutting test architecture

### 3.1 Test directory layout

```
ik_llama.cpp/tests/test-cuda-graph-probe-*.cpp         (CTest-integrated unit tests)
ik_llama.cpp/tests/test-cuda-graph-update-*.cpp
ik_llama.cpp/tests/test-cuda-graph-eviction-*.cpp
yarn-agentic/scripts/cuda-graph-probe/run-instrumentation-gate.sh      (integration harnesses)
yarn-agentic/scripts/cuda-graph-probe/run-update-path-gate.sh
yarn-agentic/scripts/cuda-graph-probe/run-soak.sh
yarn-agentic/scripts/cuda-graph-probe/parse-probe-dump.py (decision-criteria evaluator)
yarn-agentic/scripts/cuda-graph-probe/agentic-replay.sh   (np=1 + np=4 driver)
```

Naming convention: `test-cuda-graph-<area>-<scenario>.cpp` for unit tests,
`run-<phase>-<purpose>.sh` for integration harnesses. The split mirrors
the existing pattern (`test-cuda-graph-cache-bounded.cpp` is unit-test
in the harness; `test-mtp-multislot*.sh` etc. are integration scripts).

### 3.2 Naming convention for tests in this phase

| Prefix | Meaning |
|--------|---------|
| `A.T*`  | Phase A unit/integration test contract |
| `A.D*`  | Phase A decision criterion (post-soak gate) |
| `B.T*`  | Phase B contract |
| `E.T*`  | Phase E contract |
| `C.T*`, `D.T*` | conditional phases |

Each contract has **fail mode**, **pass mode**, **how to run**, and
**which implementation step makes it GREEN**.

### 3.3 Probe dump format

JSONL (newline-delimited JSON), one record per probe-event or per
flush-snapshot.

```
/mnt/archive/cuda-graph-probe/<runid>/<backend_id>-<probe>.jsonl
```

`<runid>` = `YYYYMMDDTHHMMSS-<random6>` set at backend init.
`<backend_id>` = `cuda0`, `cuda1`, …
`<probe>` ∈ `{hit_counter, timing, vram_delta, update_failures,
disable_too_many, destroy_free_delta}`.

Schema is per-probe; common fields are always:

```json
{"ts_ns": <int64>, "runid": "<str>", "backend": "<str>", "probe": "<str>", "..." }
```

Per-probe payload (the `...`):

- `hit_counter`: `{"topology_key": "0x...", "shape_key": "0x...", "hits_total": <int>, "last_use_us": <int64>, "n_distinct_shapes_in_class": <int>}`. Snapshot dumped on flush, not per-event.
- `timing`: `{"event": "capture|instantiate|update|launch_submit|launch_device", "duration_us": <float>, "n_nodes": <int>, "topology_key": "0x..."}`. Per-event line; histograms computed offline by `parse-probe-dump.py`.
- `vram_delta`: `{"event": "insert|destroy", "topology_key": "0x...", "free_before_bytes": <int64>, "free_after_bytes": <int64>, "delta_bytes": <int64>, "synced": true, "note": "..."}`. Per-event. Confound noted in `note` (cuBLAS workspace, neighbor allocations).
- `update_failures`: `{"topology_key": "0x...", "shape_key_old": "0x...", "shape_key_new": "0x...", "fallback": "reinstantiate"}`. Per-event.
- `disable_too_many`: `{"topology_key": "0x...", "consecutive_updates": <int>}`. Per-event.
- `destroy_free_delta`: same as `vram_delta` with `event="destroy"`. Phase E precondition probe.

In-memory accumulators (fixed-size where possible: timings as
ring-buffer of last 16k samples per event, hits in a `unordered_map`
bounded by `GGML_CUDA_GRAPH_PROBE_MAX_KEYS` default 32k). Flushed:

1. On `SIGUSR1`.
2. Every `GGML_CUDA_GRAPH_PROBE_FLUSH_SEC` seconds (default 30) via a
   detached background thread.
3. On `~ggml_backend_cuda_context()` (existing lifetime hook).

Each flush opens the file in append mode, writes records, `fsync`s,
closes. Atomic from a parser's perspective.

### 3.4 Accessor surface (extends `ggml-cuda.h`)

```c
// existing
GGML_API GGML_CALL size_t ggml_backend_cuda_graph_cache_size(ggml_backend_t backend);

// new — for tests and decision-criteria scripts
GGML_API GGML_CALL size_t ggml_backend_cuda_graph_topology_class_count(ggml_backend_t backend);
GGML_API GGML_CALL size_t ggml_backend_cuda_graph_disable_vram_pressure_count(ggml_backend_t backend);
GGML_API GGML_CALL size_t ggml_backend_cuda_graph_update_failure_count(ggml_backend_t backend);
GGML_API GGML_CALL int    ggml_backend_cuda_graph_probe_flush(ggml_backend_t backend); // returns 0 on success
GGML_API GGML_CALL int    ggml_backend_cuda_graph_probe_active(void);                  // 1 if GGML_CUDA_GRAPH_PROBE=1
```

Tests use these; production code never depends on them.

### 3.5 Build/run pattern

Unit tests: register in `tests/CMakeLists.txt` mirroring the existing
`test-cuda-graph-cache-bounded` block. Build with the project's
existing target (`cmake --build build --target test-cuda-graph-*`).

Integration harnesses: `bash scripts/cuda-graph-probe/run-<phase>-gate.sh`.
Each harness:

1. Confirms required artefacts (model GGUFs, snoop replay corpus).
2. Sets the env knobs (`GGML_CUDA_GRAPH_PROBE`,
   `GGML_CUDA_GRAPH_PROBE_FLUSH_SEC`, `GGML_CUDA_GRAPH_VRAM_MARGIN`).
3. Runs the workload (llama-bench, llama-perplexity, llama-server +
   replay, etc.).
4. Calls `parse-probe-dump.py --gate <phase>` to evaluate the
   per-phase decision criteria.
5. Exits 0 on GREEN, non-zero with a structured failure report on RED.

### 3.6 Rule for the test-first discipline

Per `feedback_no_skip_tests.md`: every contract listed below gets
written *and committed RED* before its implementation lands. Per
`feedback_test_first_negative_claims.md`: the test must actually
exercise the code path it claims to validate. Per CLAUDE.md §4: every
step has a verification check; "binds on the step's actual claim,"
not on an adjacent easier case.

---

## 4. Phase A — Instrumentation (mandatory; gates everything else)

Goal: produce evidence-backed answers to the questions PHASE34
guessed at.

### 4.1 Test contracts (RED first)

| ID | Test | File | Fail mode | Pass mode | Made GREEN by |
|----|------|------|-----------|-----------|---------------|
| A.T1 | Probe-disabled overhead | `scripts/cuda-graph-probe/run-overhead-canary.sh` | mean throughput delta > ±2% across 3 runs | mean delta within ±2%, stddev reported | A7 dump infra (must short-circuit when env unset) |
| A.T2 | Dump schema parses | `tests/test-cuda-graph-probe-schema.cpp` | malformed JSONL or missing required fields | every record validates against §3.3 schema | A1+A7 |
| A.T3 | Hit-counter monotonicity | `tests/test-cuda-graph-probe-hit-monotonic.cpp` | counter post < counter pre, or skipped lookups uncounted | counter post ≥ counter pre across 1k driven graph submits | A1 |
| A.T4 | Flush triggers | `scripts/cuda-graph-probe/run-flush-trigger.sh` | SIGUSR1 produces no dump, or process kill mid-run produces empty file | SIGUSR1 produces non-empty dump < 1 s after signal; SIGKILL during run leaves at least one timer-flushed dump on disk | A7 |
| A.T5 | `cudaGraphExecDestroy` free-delta probe | `tests/test-cuda-graph-probe-destroy-frees.cpp` | probe records `delta_bytes` = 0 (means destroy doesn't release), AND no diagnostic recorded | probe records sane delta OR records confound + `note` flagging cuda_pool reuse | A6 |
| A.T6 | Comparator-baseline dump | `tests/test-cuda-graph-probe-distinct-shapes.cpp` | running 100 distinct shapes produces 0 hit-counter records | dump contains ≥ 100 distinct shape_key entries, ≥ 1 topology_key | A1+A7 |

A.T1 is special: it's the regression canary that protects production
from instrumentation cost. Run as **3 runs × (PROBE=0 baseline,
PROBE=1 active)** with `llama-bench` on Qwen3.6-27B-V-F1.T1.qq, single
prompt 128t generation. Mean throughput delta must be within ±2% with
the probe **disabled** (the probe-on path can be slower; we just need
zero-overhead-when-off to confirm the gating is correct).

### 4.2 Decision criteria (post-soak gate)

After A's soak (§4.4), `parse-probe-dump.py --gate A` evaluates:

| ID | Criterion | Threshold | Decides |
|----|-----------|-----------|---------|
| A.D1 | `update_failures / total_updates` | < 5% | B viable as primary fix |
| A.D2 | distinct topology-class count under realistic workload | ≤ 10 | B's collapse worth the relaxation |
| A.D3 | `cudaGraphExecUpdate` host-side latency, P95 | < 100 µs | C (kernel-arg indirection) unnecessary |
| A.D4 | per-entry VRAM cost, mean | record value; **no threshold — informational** | sizes future eviction decisions |
| A.D5 | `cudaGraphExecDestroy` free-delta, mean | > 0 bytes | E's eviction loop can actually relieve pressure |
| A.D-ABORT | A.D1 < 80% **AND** A.D2 > 50 | both | **abort B+E**; pivot to PHASE 35.5 cuda_pool / cuBLAS instrumentation |

A.D-ABORT is the explicit "if Phase A says it's NOT the graph cache"
escape hatch. If hit: open PHASE 35.5, do not implement B or E.

### 4.3 Implementation (GREEN — small, focused commits)

Each step lands as one commit, with the matching A.T* test going
GREEN at landing. Tests written first (RED) in step A.0.

**A.0 — Land all A.T* tests RED.** Single commit. Verify they all fail
on the current binary (reasonable failures: missing accessor,
unimplemented probe, no dump file). Push.

**A.1 — Per-(topology_key, shape_key) hit counter** with `last_use_us`.

In `graph.cuh`, add to `ggml_cuda_graph`: `uint64_t topology_key`,
`uint64_t current_shape_key` (= ne fingerprint last seen), `uint64_t
hits`, `uint64_t last_use_us`. Bump on every successful lookup hit.
On miss, the new entry's fields initialise from creation context.

In `ggml-cuda.cu:4241-4256`, add a sibling `ggml_cuda_graph_get_topology_key`
that hashes only `n_nodes` and `n->op` per node (no ne loop). Existing
`_get_key` becomes the shape key, unchanged in semantics.

Snapshot dump on flush: walk `ctx.cuda_graphs`, emit one
`hit_counter` record per entry with `n_distinct_shapes_in_class`
counted from a parallel
`unordered_map<topology_key, set<shape_key>>` if that view is needed
for parser convenience (otherwise compute offline).

→ A.T3 and A.T6 GREEN.

**A.2 — Capture/instantiate/update/launch timings.**

Use `std::chrono::steady_clock` for the host-side calls
(`cudaStreamEndCapture`, `cudaGraphInstantiate`,
`cudaGraphExecUpdate`, the host-submit portion of `cudaGraphLaunch`).
Use **`cudaEvent_t` paired before/after a `cudaGraphLaunch`** for the
device-side wall time of the captured graph. Both timings emit to
the `timing` probe with distinct `event` strings:
`launch_submit` (chrono) vs `launch_device` (cudaEvent).

Important review fix: PHASE34's earlier draft conflated these.
`cudaEvent` on a host-only API is meaningless. Test A.T2 checks the
schema; a manual review of the dump confirms units are sensible.

→ part of A.T2 GREEN; A.T1 still pending until A.7 lands.

**A.3 — Per-entry VRAM delta probe** with paired sync.

At insert time (`ggml-cuda.cu:4280-4297` cache-insert region):

```
cudaDeviceSynchronize();
size_t free_before, total;
cudaMemGetInfo(&free_before, &total);
// existing instantiate path
cudaDeviceSynchronize();
size_t free_after;
cudaMemGetInfo(&free_after, &total);
emit_vram_delta(topology_key, free_before, free_after, "insert");
```

`note` field documents the confound: "cudaMemGetInfo reports
device-global free; concurrent cuBLAS workspace and pool activity may
mask the per-entry cost. Reported as distribution; treat single
samples as upper bound."

The `parse-probe-dump.py` aggregator reports mean, P50, P95, max
across all `vram_delta:insert` records — the distribution exposes the
noise floor. A.D4 reads the mean as informational.

→ part of A.T2 GREEN; A.D4 collectible after soak.

**A.4 — Update failure rate.**

Counter incremented in
`update_cuda_graph_executable:4495-4501` on the
`cudaErrorGraphExecUpdateFailure` branch. Total updates counter
incremented at function entry. Both surface via the new accessor
`ggml_backend_cuda_graph_update_failure_count`. Emit one
`update_failures` JSONL record per failure.

→ part of A.T2 GREEN; A.D1 collectible.

**A.5 — `disable_due_to_too_many_updates` rate.**

Counter incremented at `:4687-4694` set site. Per-event JSONL record
with `consecutive_updates` value at trip time.

→ part of A.T2 GREEN.

**A.6 — `cudaGraphExecDestroy` free-delta probe.**

In `~ggml_cuda_graph` (and any other destroy site — grep
`cudaGraphExecDestroy`), pair sync + `cudaMemGetInfo` exactly as A.3
but with `event="destroy"`. This is the precondition probe for Phase
E: if destroy returns memory to the OS/driver, eviction can work; if
not (cuda_pool retains), E needs to also call `cudaDeviceSynchronize`
+ `cudaCtxResetPersistingL2Cache` or accept that eviction is a
no-op for VRAM-pressure purposes.

→ A.T5 GREEN; A.D5 collectible.

**A.7 — Dump infrastructure: signal handler + timer thread + teardown
flush.**

In `ggml_init_cublas` or backend init (already a once-per-process
hook), spawn a detached `std::thread` that:

```
while (!shutdown) {
  std::this_thread::sleep_for(flush_sec);
  flush_all_backends();
}
```

Register `SIGUSR1` handler that sets a `flush_pending` atomic; timer
thread drains.

`flush_all_backends()` iterates the global `info->all_ctx` list
(already maintained at `:533-539`), takes each context's lock, walks
the in-memory accumulators, appends to the JSONL files, fsyncs.

If `GGML_CUDA_GRAPH_PROBE=0` (default), the entire infra is `#ifdef`'d
or runtime-short-circuited:

```c
if (!ggml_backend_cuda_graph_probe_active()) return;
```

The accessor reads the cached env once.

→ A.T1, A.T2, A.T4 GREEN.

### 4.4 Soak protocol

**Workload:** snoop replay corpus (`scripts/agentic-prompt-corpus.jsonl`)
+ a synthetic shape-shuffler injected for prefill-burst diversity.
Driven via `llama-server` on the qwen36-27b-x1 profile, replay tool
issues requests until **100 000 generated tokens** total (≈ 30 min
wallclock at ~50 t/s, but token-count makes the gate reproducible
regardless of machine speed).

**Configurations to soak:** np=1 and np=4 separately (the production
crash was multi-slot; np=1 is the single-tenant gate). Two
independent runs each, totaling four runs. Soak for 100k tokens per
run → 400k tokens across the gate.

**Env:**

```
GGML_CUDA_GRAPH_PROBE=1
GGML_CUDA_GRAPH_PROBE_FLUSH_SEC=30
GGML_CUDA_GRAPH_VRAM_MARGIN=512  # used only after E lands; harmless here
```

**Output:** `/mnt/archive/cuda-graph-probe/<runid>/`. Each run gets a
distinct runid; the parser handles cross-run aggregation.

**Gate:** `parse-probe-dump.py --gate A
/mnt/archive/cuda-graph-probe/<runid-glob>/` exits 0 if all A.D*
thresholds pass and A.D-ABORT does not trigger. Phase B starts only
on green.

### 4.5 Phase A commits

| Commit | Content |
|--------|---------|
| A.0 | All A.T* tests landed RED + CMakeLists registration |
| A.1-A.7 | One commit per step; tests transition RED→GREEN incrementally |
| A.gate | Soak results dump committed (analysis + JSONL excerpts) under `yarn-agentic/data/phase35/A-soak-<runid>/` |

---

## 5. Phase B — Topology-class keying (the main redesign)

**Pre-condition:** A.D1 ≥ 95%, A.D2 ≤ 10, A.D-ABORT not triggered.

### 5.1 Test contracts (RED first)

| ID | Test | File | Fail mode | Pass mode | Made GREEN by |
|----|------|------|-----------|-----------|---------------|
| B.T1 | PPL bit-identity | `scripts/cuda-graph-probe/run-ppl-identity.sh` | PPL diff ≠ 0.0 between pre-B and post-B | exact 0.0 across the wikitext-2 slice | B1-B5 |
| B.T2 | Cache-collapse assertion | `scripts/cuda-graph-probe/run-topology-class-count.sh` | distinct topology classes > 10 after replay | ≤ 10 classes | B1-B3 |
| B.T3 | State-mutation under alternating Updates | `tests/test-cuda-graph-update-state-mutation.cpp` | output of A→B→A→B→A sequence diverges from eager-path baseline | element-wise exact match across all 5 outputs | B3-B5 |
| B.T4 | Thrash bound | `scripts/cuda-graph-probe/run-thrash-bound.sh` | `disable_due_to_too_many_updates` > 0 events on the agentic replay | 0 trips on np=1, ≤ 1% trip-rate on np=4 | B5 + (possibly) widening detector |
| B.T5 | Comparator strict-on-op_params | `tests/test-cuda-graph-comparator-op-params.cpp` | op_params change within same topology key passes Update silently and produces wrong output | comparator forces re-instantiate OR produces correct output via Update | B4 |
| B.T6 | HIP build + bit-identity | `scripts/cuda-graph-probe/run-hip-bit-identity.sh` (gated on host availability) | HIP build red, OR PPL diff ≠ 0.0 on HIP | both green | B1-B5 + any `#ifdef __HIP_PLATFORM_AMD__` workarounds |
| B.T7 | `test-cuda-graph-cache-bounded` still GREEN | existing | regression in count-cap behaviour | unchanged | B preserves existing cap pathway |

Notes on individual tests:

- **B.T3 specifically targets the "Update(A) → Update(B) → Update(A)"
  ordering** that PPL-identity (B.T1) doesn't probe. The test
  constructs three distinct same-topology shape variants in a loop,
  each with deterministic input, drives them sequentially through the
  graph backend, and compares to the same graph run via
  `ggml_backend_cpu_compute` (eager). Failure mode is subtle: the
  cached executable might leak state from a prior shape's input
  pointer. PPL averages across many samples and would mask this.

- **B.T5 strict-on-op_params** constructs two graphs with same
  topology + ne but different `op_params` (e.g., `GGML_OP_SCALE`
  alpha = 0.5 vs 1.5). The relaxed comparator must NOT pass Update
  on this pair; it must either flag re-instantiate or — if Update
  succeeds — produce numerically correct output for the new alpha.
  Failure means the relaxation went too far.

- **B.T4 thrash bound** runs the same replay corpus as Phase A's
  soak, parses the `disable_too_many` JSONL records. Pre-condition
  for B5 sufficiency: A.D1 ≥ 95%. If B.T4 fails, the detector at
  `:4687-4694` is too tight for the new traffic pattern; widen from
  4 to 16 consecutive in a sub-step.

### 5.2 Implementation

**B.0 — Tests RED.** Single commit. All B.T* fail on current binary
because `topology_class_count` accessor returns 0, comparator hasn't
been touched, etc. Push.

**B.1 — `ggml_cuda_graph_get_topology_key`** (new function in
`ggml-cuda.cu:~4241-4256`). FNV-1a 64-bit over `(n_nodes, op[i] for i
in nodes)`. Distinct from `_get_key` which still hashes ne. Both
coexist: shape_key still tracked per entry, topology_key is the new
map key.

**B.2 — Cache map switch.** `ggml_backend_cuda_context::cuda_graphs`
keyed by `topology_key` (in `common.cuh:862`). Each cached
`ggml_cuda_graph` carries `current_shape_key` (= shape_key last applied
via successful instantiate or update) and `failed_update_op_params_signatures`
(unordered_set of signatures that have failed Update; see B5).

**B.3 — `ggml_cuda_get_graph` routing.**

```
auto topology_key = ...;
auto shape_key    = ...;
auto it = cuda_graphs.find(topology_key);

if (it == cuda_graphs.end()) {
    // miss → existing capture+instantiate path, then insert with
    //         current_shape_key = shape_key.
}

auto & entry = it->second;
entry->hits++;
entry->last_use_us = now_us();

if (entry->current_shape_key == shape_key) {
    // exact hit — fast path, cudaGraphLaunch directly.
} else if (entry->failed_update_op_params_signatures.contains(signature_of(cgraph))) {
    // known-bad combo for this class → re-instantiate (no Update attempt).
} else {
    // shape change within class → call update_cuda_graph_executable.
    // On success: entry->current_shape_key = shape_key.
    // On cudaErrorGraphExecUpdateFailure: existing fallback re-instantiates;
    //   record signature in failed_update_op_params_signatures so future
    //   same-(topology, op_params-signature) lookups skip the Update attempt.
}
```

**B.4 — Comparator relaxation** in
`ggml_graph_node_has_matching_properties:4406-4442`.

Strictness rules:

- ne[0..3] difference: now **OK** (handled by Update path; was a
  cache-miss trigger before).
- nb[0..3] difference: still **strict** unless the per-op rules in
  `is_cuda_graph_update_required:4444-4485` already deemed it
  Update-safe (e.g., for CPY/VIEW). Preserve those rules; do not
  generalise.
- src address difference: still **strict** for non-CPY ops; CPY uses
  `cpy_dest_ptrs` indirection.
- op_params difference: **strict** — these go to kernel constants or
  affect kernel selection. Triggers re-instantiate. Tested by B.T5.

The op_params-signature is the hash of `(node_idx, op_params_byte_view)`
across the cgraph. Cheap to compute; recorded in `ggml_graph_node_properties`
alongside existing fields.

**B.5 — Per-(topology, op_params-signature) blacklist.**

Resolved scope decision (was an open Q in the review): **per-signature
within class**, NOT whole-class. When Update fails for a given
op_params signature, we cache that fact in
`failed_update_op_params_signatures` on the entry. Subsequent lookups
with the same signature skip the Update call and go straight to
re-instantiate. Other signatures in the same class are unaffected
and still attempt Update. This minimises wasted Update calls without
collapsing the class.

→ B.T1, B.T3, B.T5 GREEN incrementally as B1-B5 land.

**B.6 — Verify nullptr-eager fallback at `:4657`.** Read the
dispatcher; confirm a path exists where `ggml_cuda_get_graph`
returning `nullptr` causes `ggml_backend_cuda_graph_compute` to fall
back to the eager submit. If absent, add a one-liner. Document in this
plan when verified. (Also a precondition for E.)

→ B.T7 must remain GREEN: existing cap-based test still passes
because the cap path is preserved (see Phase E for its retirement).

### 5.3 Phase B commits

One commit per step (B.0 tests, then B.1-B.6). Final commit:
`PHASE35: Phase B post-soak measurements`.

---

## 6. Phase E — Allocation-aware eviction

**Pre-condition:** A.D5 confirms `cudaGraphExecDestroy` returns memory
(or implementation note documents a workaround); B.6 confirms
nullptr-eager fallback works.

### 6.1 Test contracts (RED first)

| ID | Test | File | Fail mode | Pass mode | Made GREEN by |
|----|------|------|-----------|-----------|---------------|
| E.T1 | OOM-resistance synthetic | `tests/test-cuda-graph-eviction-oom-resistance.cpp` | abort/OOM during the 50-class drive, OR `disable_due_to_vram_pressure` count = 0 (eviction never engaged) | no abort + ≥ 1 vram-pressure event | E1-E4 |
| E.T2 | nullptr-eager fallback | `tests/test-cuda-graph-eviction-eager-fallback.cpp` | output incorrect when flag forced ON | element-wise exact match vs forced-OFF baseline | E2 + B.6 |
| E.T3 | Eviction actually relieves pressure | `tests/test-cuda-graph-eviction-frees.cpp` | post-eviction `cudaMemGetInfo` free is unchanged | free increases by ≥ 0.5 × evicted-entry recorded VRAM cost | E2 + A.D5 confirmation |
| E.T4 | Production soak np=1 + np=4 | `scripts/cuda-graph-probe/run-soak.sh` | any of: GPU0/1 free < 1 GiB at any point; host RSS > 16 GiB; abort/OOM; throughput delta > ±2% from pre-B baseline (3 runs) | all hold for full soak (100k tokens, np=1 + np=4) | E1-E5 + B |
| E.T5 | `test-cuda-graph-cache-bounded` semantics preserved | existing | env-driven hard cap stops working | env still applies as optional ceiling on top of vram-driven cap | E5 |

### 6.2 Implementation

**E.0 — Tests RED.** All E.T* fail (vram-pressure flag unimplemented,
eviction loop unimplemented, etc.). Push.

**E.1 — `cuda_safety_margin_bytes()` helper.** Reads
`GGML_CUDA_GRAPH_VRAM_MARGIN` (MiB; default 512). Optional: derive a
floor from compute-buffer size via `ggml_backend_cuda_get_buffer_size`
hooks if obtainable; otherwise just the env value.

**E.2 — Eviction loop in `ggml_cuda_get_graph` cache-miss branch**
(`ggml-cuda.cu:4280-4297` rewrite):

```
size_t free_bytes;
cudaMemGetInfo(&free_bytes, &total_bytes);

while (cuda_graphs.size() > 1 && free_bytes < safety_margin) {
    auto victim = pick_lfu(cuda_graphs);  // by hits, ties on last_use_us
    cuda_graphs.erase(victim);             // dtor calls cudaGraphExecDestroy
    cudaMemGetInfo(&free_bytes, &total_bytes);
}

if (free_bytes < safety_margin) {
    // single-entry-too-big: cannot fit even a fresh entry.
    // Mark a context-level disable_due_to_vram_pressure tally bump
    // and return nullptr → eager path.
    ctx->disable_due_to_vram_pressure_count++;
    return nullptr;
}
```

**E.3 — `last_use_us` field on `ggml_cuda_graph`** (already added in
A.1 for hit-counter).

**E.4 — `disable_due_to_vram_pressure_count`** on the backend
context, exposed via `ggml_backend_cuda_graph_disable_vram_pressure_count`.

**E.5 — `GGML_CUDA_GRAPH_MAX` retained as optional ceiling.**

Revised from review feedback: do NOT remove the env. New semantics:
the count-cap is no longer the default mechanism (vram-driven is),
but if the env is set, apply it as `min(env_cap, ∞)` on top of the
vram-driven cap. Default unset = no count cap. This keeps debugging
and bisection ergonomic, and preserves the existing
`test-cuda-graph-cache-bounded` test (B.T7 / E.T5).

→ E.T1 through E.T5 GREEN.

### 6.3 Phase E commits

One per step. Final: `PHASE35: Phase E production soak results`.

---

## 7. Phase C — cpy-indirection generalisation (conditional)

**Pre-condition:** A.D3 reports P95 update latency ≥ 100 µs **AND**
B has shipped and post-B steady-state still shows update-driven host
overhead in nsys.

If neither holds, skip C entirely.

### 7.1 Test contracts (sketch — fully fleshed when triggered)

- C.T1 — kernel-arg indirection produces correct output across N ne
  variants without re-Update.
- C.T2 — patch-on-launch is faster than `cudaGraphExecUpdate` for
  the same shape change (compare to A.2 timings).

### 7.2 Implementation outline

C1. Audit kernel-arg sites where ne flows in (FA n_kv extent,
mat-mul row counts, …).
C2. Mirror `cpy_dest_ptrs` for ne values:
`ne_indirection_table` (host) + `ne_indirection_d` (device).
Captured kernels read `ne_indirection_d[node_idx][dim]` instead of
constant ne.
C3. Patch host table + cudaMemcpy to device on launch — no Update
needed for ne-only changes.

This is non-trivial scope. Defer past the initial B+E ship.

---

## 8. Phase D — heat-tier admission (conditional)

**Pre-condition:** Post-B, `parse-probe-dump.py` shows ≥ 25% of
distinct shape keys are single-occurrence under realistic traffic
**AND** their aggregate VRAM cost (from A.3 distribution) exceeds
the safety margin under typical contention.

If B's topology collapse already absorbs them, D is unnecessary.

### 8.1 Test contracts (sketch)

- D.T1 — cold shapes (hits < HEAT_MIN) take the eager path; output
  matches graph-cache baseline.
- D.T2 — once a shape promotes (hits ≥ HEAT_MIN), it lands in the
  cache and subsequent calls hit the graph path.
- D.T3 — counter map LRU eviction (default cap 4096) doesn't unbound
  host RSS.

### 8.2 Implementation outline

D1. `hits` already added in A.1.
D2. Defer capture+instantiate until `hits >= GGML_CUDA_GRAPH_HEAT_MIN`
(env, default 2). Below that, return `nullptr` → eager path.
D3. Counter map separately bounded (LRU on `last_use_us`, default
4096 entries).
D4. Phase doc captures the per-shape latency penalty: HEAT_MIN-1
non-graph runs per genuinely cold shape.

---

## 9. Phase F — docs

F.1 — Open `PHASE34-LEAK-RCA.md`, prepend a second AMENDED block
summarising what Phase A measured. Strike or annotate whatever
Phase A refuted/confirmed. **Reference this doc** for the redesign.
Single commit, push immediately (CLAUDE.md §5).

F.2 — Update **this doc** (`PHASE35-GRAPH-CACHE-REDESIGN.md`) with
Phase A findings, Phase B/E rationale and final shipped design,
soak results, `cudaGraphExecUpdate` measurements. Append-only at
bottom in a "Findings" section; do not rewrite the plan above so
the test-first reasoning stays auditable.

F.3 — `SUMMARY.md` already updated in this commit with PHASE35 entry;
no further change unless C/D/E ship as separate phases.

F.4 — Commit hygiene per CLAUDE.md §5: PHASE34 amendment one commit
+ push, this doc updates separate commits + push, SUMMARY.md
addition bundled with this doc's creation (allowed for additions).

---

## 10. Reused infrastructure (do not duplicate)

- `update_cuda_graph_executable` (`ggml-cuda.cu:4487`) — the
  `cudaGraphExecUpdate` wrapper with re-instantiate fallback.
  Phase B invokes it more often; does not replace it.
- `ggml_graph_node_has_matching_properties` (`:4406`) — the per-node
  comparator. Phase B relaxes ne strictness only; nb / src / op_params
  rules unchanged for non-CPY/VIEW.
- `set_ggml_graph_node_properties` (`:4393`) — captures node state.
  Phase B extends it to record op_params signature.
- `cpy_dest_ptrs` / `dest_ptrs_d` (`:4364-4387`, `graph.cuh:34-40`) —
  precedent and working example for Phase C generalisation.
- `disable_due_to_too_many_updates` flag (`graph.cuh:30`) — kept;
  Phase B may widen the threshold via a sub-step if B.T4 fails.
- `ggml_backend_cuda_graph_cache_size` accessor (`:541`) — kept;
  new accessors (§3.4) join it.
- `test-cuda-graph-cache-bounded.cpp` — kept GREEN throughout;
  serves as B.T7 / E.T5 regression canary.
- `info->all_ctx[device]` global registry (`:533-539`) — used by the
  flush thread to walk all backend contexts.

---

## 11. Out of scope

- Multi-GPU coordination of graph caches — each backend context has
  its own `cuda_graphs` map; the redesign keeps that.
- Upstreaming to llama.cpp — separate phase once stable.
- Host-side `--cache-ram` and `--ctx-checkpoints` — separate hygiene
  work (PHASE34 M2). Not part of this GPU-side redesign.
- Vulkan backend graph caching — different code path entirely.
- cuda_pool HWM and cuBLAS workspace — only instrumented if Phase A
  triggers ABORT to PHASE 35.5.

---

## 12. Risk register

| Risk | Mitigation |
|------|------------|
| Phase A probe overhead changes scheduler behaviour under timed runs | A.T1 explicit ±2% gate with PROBE=0; abort if regressed |
| `cudaMemGetInfo` deltas contaminated by neighbour allocations | Paired-sync sampling; report distribution not single sample; document confound in `note` field |
| Update success rate workload-dependent: many same-topology shapes failing Update | A.D1 measures it; B5 per-signature blacklist contains the damage; B falls back to current shape-keyed behaviour for blacklisted signatures |
| Phase E `safety_margin` empirical | Default 512 MiB; expose env knob; tune from T4/T5 results |
| PPL bit-identity (B.T1) fails on first attempt due to comparator over-relaxation | B.T5 is the fast canary; fix the comparator, not the test |
| HIP path divergence on `hipGraphExecUpdate` | T6 canary; if red, gate new path behind `#ifndef __HIP_PLATFORM_AMD__` and keep current behaviour on HIP |
| Probe dump file corruption mid-write | Per-flush `open(O_APPEND) → write → fsync → close`; parser tolerates trailing partial line |
| `cudaGraphExecDestroy` doesn't return memory to driver (cuda_pool) | A.D5 detects; E gets pivot path: keep cap-based cap as primary, vram-driven as advisory; document in F.2 |
| Background flush thread lifecycle vs `cudaDeviceReset` / process teardown | Detached thread sets shutdown flag; teardown drains-then-joins with 1 s timeout; force-flushes on `~ggml_backend_cuda_context()` regardless |
| Probe data exfiltration concern (`/mnt/archive` is the network share) | Dump contains only counters and timings; no model data, no tokens, no prompts. Document in commit message |

---

## 13. Verification summary

**Phase A shippable on its own** (instrumentation-only commit) once:
- A.0 commit RED, A.1-A.7 commits each GREEN against their test;
- A.T1 ±2% overhead gate green;
- 100k-token × 4-run soak completed and JSONL dumps committed under
  `yarn-agentic/data/phase35/A-soak-<runid>/`;
- `parse-probe-dump.py --gate A` exits 0 (or ABORT triggered, in
  which case Phase 35.5 opens instead of B).

**Phase B + E shippable as a unit** when **all** of:
- B.T1 PPL diff = 0.0 across the wikitext slice;
- B.T2 topology-class count ≤ 10 confirmed in fresh dump;
- B.T3 state-mutation alternating-shape sequence GREEN;
- B.T4 thrash bound met on production replay;
- B.T5 comparator strict-on-op_params GREEN;
- B.T6 HIP build green + bit-identity (or `#ifdef`'d off);
- B.T7 / E.T5 existing cap-based test still green;
- E.T1 OOM-resistance synthetic GREEN;
- E.T2 nullptr-eager fallback GREEN;
- E.T3 eviction-actually-frees GREEN (or A.D5-driven workaround);
- E.T4 production soak np=1 + np=4 GREEN, perf within ±2% of pre-B
  baseline (3 runs each), no aborts, free VRAM ≥ 1 GiB throughout,
  host RSS ≤ 16 GiB.

**Phase C deferred** unless A.D3 + post-B nsys both flag
update-host-overhead as material.

**Phase D deferred** unless post-B distribution shows single-occurrence
shapes consuming material VRAM.

**Phase F (doc amendments) lands alongside** the corresponding code
phase per CLAUDE.md §5.

---

## 14. Findings (appended as work lands)

### 14.1 Phase A.0 + A.1–A.7 instrumentation landed

**Submodule commits** (branch `phase33-concat-probe`):
- `8febfc41` — initial stub accessor surface + four RED test binaries.
- `1e46cd41` — descriptive renames (probe35.* → cuda_graph_probe.*,
  test-phase35-A-* → test-cuda-graph-probe-*); probe accumulator + JSONL
  flush infra; ggml_cuda_graph dtor records destroy vram_delta event.
- `0a119a76` — wires capture / instantiate / update / launch_submit
  timing call sites; insert vram_delta paired-sync sample;
  update_failures + disable_too_many recorders; lifetime hardening
  (being_destroyed atomic, lock-order matching, on_context_destroyed
  before stream/event teardown). Stale phase35-named tests deleted.

**Parent commits** (branch `phase32-q4_0_ar16-integration`):
- `8d95308` — landed this doc + SUMMARY entry.
- `89d5677` — first submodule pin bump + run-overhead-canary.sh /
  run-flush-trigger.sh in scripts/cuda-graph-probe/.
- `314a0a0` — pin bump after the rename + accumulator commit.
- `060a482` — pin bump after the timing/vram_delta wiring commit.

**Test status (synthetic):** ctest 5/5 GREEN locally. Five probe types
all emit records under the synthetic test workloads:

| probe | records observed |
|-------|------------------|
| hit_counter      | per cache entry; bumped twice per submit (compatibility-check + main-compute call sites both invoke `ggml_cuda_get_graph`) |
| timing/capture   | once per cache miss (after `cudaStreamEndCapture`) |
| timing/instantiate | once per cache miss + once per Update-fail re-instantiate |
| timing/update    | once per `cudaGraphExecUpdate` invocation |
| timing/launch_submit | once per `cudaGraphLaunch` (host-submit overhead, not device wall time) |
| vram_delta/insert | once per cache miss (paired-sync `cudaMemGetInfo`) |
| vram_delta/destroy | once per `~ggml_cuda_graph` (paired-sync `cudaMemGetInfo`) |
| update_failures  | not yet observed in synthetic tests (synthetic shapes don't trip `cudaErrorGraphExecUpdateFailure`) |
| disable_too_many | not yet observed in synthetic tests |

### 14.2 Early signal: `cudaGraphExecDestroy` returns no VRAM (synthetic)

In the destroy-frees test (cap=2, 8 distinct shapes → ~6 forced
evictions), every `vram_delta event=destroy` record reports
`delta_bytes = 0` (free_before == free_after, both with paired
`cudaDeviceSynchronize`). Across the 6 destroys:

```
mean delta_bytes  = 0
nonzero deltas    = 0/6
```

This is the precondition signal that Phase E was waiting on. **If the
A.gate soak under realistic 27B traffic confirms this pattern,
Phase E's allocation-aware eviction loop will not relieve VRAM
pressure** — the freed `cudaGraphExec_t` memory stays in
`cuda_pool` (or somewhere else accounted as "global allocator")
rather than returning to the OS-visible free pool that
`cudaMemGetInfo` reports.

Implications, if confirmed at scale:
- Phase E's `disable_due_to_vram_pressure` flag still useful (we can
  *detect* low headroom and refuse new entries), but its eviction
  loop shrinks to "no-op then return nullptr" rather than "evict to
  make room."
- Alternative mechanisms to actually release: investigate
  `cudaDeviceSetLimit(cudaLimitGraphMemAlloc*)`,
  `cudaDeviceGraphMemTrim`, or recreating the cuda_pool. Out of scope
  for the immediate redesign; logged as an open Phase E sub-question.

This finding is **synthetic only** at this commit. The A.gate soak
on real 27B agentic traffic is what makes it actionable; until then
treat as a hypothesis to verify, not a settled result.

### 14.3 Open follow-ups within Phase A scope

- **Production smoke pending:** harness `run-overhead-canary.sh` and
  `run-flush-trigger.sh` exist but haven't been driven against the
  active production server — overlap rules forbid concurrent
  GPU-bound bench while llama-server is up.
- **`update_failures` and `disable_too_many` need realistic
  workload coverage:** synthetic add-graphs don't trigger them. The
  A.gate soak is the natural binding test.
- **`parse-probe-dump.py` decision evaluator:** stub committed in
  `scripts/cuda-graph-probe/`; computes per-probe summary
  statistics. The PASS/FAIL/ABORT verdict logic for A.D1–A.D5 lands
  during A.gate when we have real data to wire it against.

