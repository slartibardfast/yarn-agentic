# PHASE_NP_CLOSURE — Production NP-determinism, real-server closure

## TL;DR

NP-determinism for Qwen 3.6 27B on dual TU102 (Q4_0 KV + Hadamard) is
**NOT closed**. Three known kernel/dispatch bugs are baked into source
defaults this session (CY.F.17 stream_K, A.1' FA per-slot-kv route,
singlewarp + delta-net + cuBLAS knobs). The production server still
fails NP-cross byte-identity at the slot level on the existing harness
(`scripts/test-production-np-determinism.sh`) — a separate slot-position
bug surfaces in the real continuous-batching path that the prior
capture-based work (CY.F.7) did NOT bind on. Closing this requires its
own audit cycle.

## Goal

Production server `profiles/active.sh` produces byte-identical logits
at any `--parallel N` for N ∈ {1, 2, 4, 8} on the same prompt, with
no LLAMA_* / GGML_* / CUBLAS_* environment variables required.

Why: unlock multi-slot serving without violating the implicit user
contract that the same prompt yields the same output regardless of
co-tenants. Today's `--parallel 1` exists precisely to dodge this gap.

## State of evidence (post-2026-05-17 session)

### Closed (with binding evidence)

- **CY.F.17** — MMQ stream_K shape-dep. Baked at `mmq.cuh`; deterministic
  by default. Verified at vlong NP=1 vs NP=2 slot-0, l_out all 63
  layers byte-identical (`bake-pskv-vlong-np{1,2}/`).
- **A.1'** — FA prefill 256-tok shape-dep at slot-0 prefill output.
  Baked in `llama-build-context.cpp` — per-slot-kv route is always-on
  for the Qwen 3.5/3.6 shape (Dq=Dv=256, gqa≤16, no attn_sinks). The
  `wmma_f16_case<256,256,32,half>` heuristic-driven prefill route is
  no longer reachable for this shape.
- **Singlewarp as the default per-slot-kv mode** (`fattn-per-slot-kv-sm75.cu`).
  The prior `LLAMA_PSKV_MODE` env-gate is removed. Singlewarp is the
  FIX-C v5 path — per-row single-warp CTA, fp32 canonical k-loop, no
  cross-warp reduction.
- **delta-net.cu `use_256` always-on** — `LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH`
  env-gate removed; recurrent kernel pinned to threads_per_block=256.
- **cuBLAS determinism** — `CUBLAS_WORKSPACE_CONFIG=:4096:8` set via
  `setenv(... ,0)` before first `cublasCreate`. TF32 math mode disabled
  (`CUBLAS_DEFAULT_MATH`). Both baked, env-gate removed.

### Open

- **Production-stack NP-cross byte-identity** — `test-production-np-determinism.sh`
  fails at NP={2,4,8} vs NP=1 baseline, even on its default ~200-token
  prompt and with all the bakings above in place.
- **Slot-position dependence at NP=8** — outputs cluster in groups
  (e.g. on vlong: 4 slots one answer, 4 slots another). Reproducible.
- **Audit A.1** (#214 in TaskList) — singlewarp-specific binding at the
  production K/V distribution, still pending and now the natural
  successor to A.1'.

## Today's harness failure signature (binding evidence for the open bug)

`scripts/test-production-np-determinism.sh`, default prompt (~200 tok),
no env overrides:

| NP pair | slot 0 | bytes |
|---|---|---|
| NP=1 vs NP=2 | DIFFERS | 340 vs 380 |
| NP=1 vs NP=4 | DIFFERS | 340 vs 380 |
| NP=1 vs NP=8 | DIFFERS | 340 vs 356 |
| NP=2 vs NP=4 | **IDENTICAL** | 380 vs 380 |
| NP=2 vs NP=8 | DIFFERS | 380 vs 356 |
| NP=4 vs NP=8 | DIFFERS | 380 vs 356 |

Interpretation:
- NP=1 differs from every NP>1 → there's a path that fires only at
  multi-slot, OR NP=1 takes a different code path (likely `Q->ne[1]==1`
  decode special-case).
- NP=2 == NP=4 slot-0 → the multi-slot path is *internally* stable for
  small NP.
- NP=8 differs → something at NP=8 specifically (ubatch split? cross-
  device shard pattern? slot-id addressing?).

`/opt/models/yarn-audit-data/closure-final/default/run-20260517T162725/`
holds the raw outputs + per-slot diffs.

vlong (502 tok) has the same pattern with worse intra-NP clustering.

## What this is NOT

- Not a regression from the bakings — the failure reproduces under the
  prior env-stack too (we tested both).
- Not the A.1' bug — that's the prefill 256-tok shape-dep at slot-0,
  which is closed at the capture level.
- Not a cb_eval / capture artifact (CY.F.2 era) — this is the real
  production server's HTTP completion output.
- Not a singlewarp correctness bug per the unit-test suite — DATA-1/3/4
  passed singlewarp at the unit-test level. The bug is at server +
  scheduler integration, not in the kernel proper.

## Recommended starting points for the next phase

1. **Pin the multi-slot vs single-slot path divergence**. NP=1's
   different output suggests there is a path that fires only when
   `cparams.n_parallel > 1`. Candidates: KV cache layout (per-slot vs
   continuous), `Q->ne[1]==1` decode special-case in `fattn.cu` line
   113, the `inp_per_row_k_bound` tensor that's only set up under
   `use_per_slot_kv`, the `only_active_experts` host-buffer path that
   only fires under specific scheduling.

2. **Capture decode-step intra-NP at production server** — extend the
   F.1 capture tool to drive a server-style request (prefill + N
   decode steps), not just a one-shot prefill. The existing harness
   has only proven prefill-output identity; the failure mode lives in
   decode + KV cache scheduling.

3. **Bisect on N**. NP=2 == NP=4 slot-0 but NP=8 differs. The boundary
   between 4 and 8 is informative — KV cache page size, ubatch split
   threshold, or warp-count heuristic.

4. **Re-read CY.F.7/8/9 carefully**. Memory of CY.F.7 says cross-NP
   slot-0 was byte-identical on the V4 *capture* path; CY.F.2 noted a
   "capture-mechanism discrepancy". The capture path likely held
   invariants that production does not. The real binding has to be
   server-driven, not capture-tool-driven.

## Anti-patterns (do not repeat)

- **Don't trust DATA-1's "production harness PASS"** without
  re-running. It either ran under a different state or on a different
  harness. The current harness in the current build does not pass.
- **Don't conflate per-slot-kv prefill identity with NP-determinism**.
  A.1' fixed one specific bug at the prefill stage; the broader bug is
  somewhere else (decode + scheduler).
- **Don't bake an env-var as "fix" if it doesn't actually close the
  harness end-to-end**. Bakings are correctness cleanups; closure
  requires the harness pass.
- **Don't relocalize known-fixed bugs**. CY.F.17 / A.1' / singlewarp /
  cuBLAS workspace / TF32 / delta-net thread-count are all sealed.
  Don't waste tokens re-verifying them.

## Critical files

**Production engine** (source-baked):
- `ik_llama.cpp/ggml/src/ggml-cuda/mmq.cuh` — CY.F.17 stream_K off.
- `ik_llama.cpp/src/llama-build-context.cpp` lines 2697–2725 — A.1'
  per-slot-kv always-on.
- `ik_llama.cpp/ggml/src/ggml-cuda/fattn-per-slot-kv-sm75.cu` line
  2454 area — singlewarp hardcoded.
- `ik_llama.cpp/ggml/src/ggml-cuda/delta-net.cu` line 219 area —
  `use_256 = true`.
- `ik_llama.cpp/ggml/src/ggml-cuda/common.cuh` line 931 area —
  CUBLAS_WORKSPACE_CONFIG + TF32 off.

**Harness**:
- `scripts/test-production-np-determinism.sh` — the binding contract.
  Already cleared of harness-side env requirements after this session's
  bakings (those env vars are no-ops now).

**Capture tooling**:
- `ik_llama.cpp/examples/llama-state-capture/llama-state-capture.cpp`
  — F.1 prefill-only capture. Needs extension for server-style
  decode-step capture (Recommendation #2 above).

**Data left for the next session**:
- `/opt/models/yarn-audit-data/closure-final/default/run-20260517T162725/`
  — full failed harness run on default prompt, NP={1,2,4,8}.
- `/opt/models/yarn-audit-data/closure-final/vlong/run-20260517T162319/`
  — same on vlong.
- `/opt/models/yarn-audit-data/bake-pskv-vlong-np{1,2}/` — A.1' closure
  evidence (slot-0 prefill identity).
- `/opt/models/yarn-audit-data/intra-layer3-vlong-{pskv-,}np{1,2}/` —
  intra-layer-3 walkdown captures that localized A.1'.

**Related TaskList items**:
- #214 — Audit A.1 Singlewarp FA at production K/V distribution
  (now the natural next ticket).
- #178 — RESEARCH-5 reference oracle gathering plan (still pending).
- #155 / #156 — Phase E/F perf + closure binding (gated on this
  phase landing).

## Update 2026-05-19 — R5 session

### R5.1 — Reproduce + characterise (CLOSED)

Production harness re-run at HEAD (commit `5b6605d8`, post-MMQ-I=8,
post-v1-revert). Failure reproduces matching the table above. Artifacts
in `data/npc-r5-baseline/run-20260519T182906/`.

### R5.2 / R5.3 — Bisect on first divergent decode tensor (CLOSED — kernel layer exonerated)

`llama-state-capture --autoregress {prefill, auto-0, auto-1} --np {1,2}`
across layers {0, 1, 2, 3, 30, 31, 32, 33, 60, 61, 62, 63} reports ALL
captured tensors slot-0 byte-identical at NP=1 vs NP=2. Every kernel
dispatched in the capture-tool path is NPC-clean.

Caveat per `feedback_verify_test_mechanism_before_trusting`: the capture
tool's own NP=1 generated text does NOT match the server's NP=1 text
("code generation. The development of..." vs "code generation, though
they remain..."). Same prefix to token 2 of the completion, then
diverges. So the capture path is not bit-identical to the server's
decode path. The "kernel-clean" conclusion holds STRONGLY for the
capture-tool path, WEAKLY for the server's path until server-side
instrumented capture confirms it.

### R5.4* — Server-side bisection via cb_eval (CLOSED — bug C localised)

Added cb_eval tensor-dump hook to `llama-server` (submodule `3ad8934e`)
gated on `LLAMA_SERVER_CAPTURE_DIR` env. Used it to fire `N` identical
concurrent requests against the production server at NP=N, capturing
`{l_out, Vcur, Kcur_hadamard, Qcur, kqv_out, attn_out}` at slots `0`
and per layer `{0, 1, 3, 31, 62, 63}`.

Findings:
- Single-request capture under the hook PASSES. The hook itself does a
  synchronous device-to-host copy and a blob write, which perturbs
  timing enough to hide concurrent races. Per
  `feedback_verify_test_mechanism_before_trusting` the single PASS is
  NOT proof of NPC.
- Under concurrent multi-request fire, two distinct bug classes
  separated by symptom:
  - **Bug B — deterministic NP=4 failure.** OOB read at
    `mmq.cuh:4358` (process_tile_i8 tile_y load) pinned by
    `compute-sanitizer initcheck`. Closed by submodule `1f83f681`
    (Z2-narrow: bound `l` to `tile_y_extent` so threadIdx.y=1 lanes
    in the final `l0=256` iter cannot overshoot the allocation).
    Production harness NP=4 is deterministic-clean post-fix.
  - **Bug C — NP=2 stochastic ~10% drift.** Persists post-Z2-narrow.
    Same prompt, identical seeds, same Z2-narrow build. Reproduces
    ~1 in 10 runs of the harness. Probe `r5-probe-np4.sh PROBE=2`
    (NP=8 fire 4) also fails ~20% — failure rate is NOT correlated
    purely with slot-pool == in-flight count.

### R5 kernel-coverage sweep (CLOSED — kernel layer exonerated at production dims)

Every Qwen 3.6 27B production-shape kernel now has a shape-invariance
test at production dims:

| Kernel                     | Test                                              | Result |
|----------------------------|---------------------------------------------------|--------|
| MMQ Q4_0_AR16              | test-mmq-q4-0-ar16-shape-invariance-prod-dim      | PASS   |
| MMVQ Q4_0_AR16             | test-mmvq-q4-0-ar16-shape-invariance-prod-dim     | PASS   |
| FA per-slot KV (singlewarp)| test-fattn-per-slot-kv-dispatch-np-invariance     | PASS   |
| RMSNorm                    | test-rmsnorm-batch-shape-invariance               | PASS   |
| RoPE                       | test-rope-batch-shape-invariance                  | PASS   |
| ggml_reduce                | test-ggml-reduce-shape-invariance                 | PASS   |
| cuBLAS pinned-HMMA         | test-cublas-pinned-shape-invariant                | PASS   |
| DeltaNet (linear-attn)     | test-deltanet-shape-invariance                    | PASS   |

Each test instantiates the kernel via `ggml-backend` CUDA at the
production geometry, fixes slot-0 inputs while leaving other-slot
inputs reproducible-random per slot index (not per `n_seqs`), and
asserts slot-0 output is byte-identical across `n_seqs ∈ {1, 2, 4, 8}`.
The DeltaNet test additionally asserts byte-identity of slot-0 new-state.

These tests are unit-test-cheap (each runs in seconds) and bind for
future regressions.

**Conclusion.** Bug C (NP=2 stochastic ~10%) is NOT a single-kernel
shape-invariance bug at production geometry. The remaining candidate
surface is integration-level: continuous-batching scheduler / slot
allocator / batch composition / cb_eval dispatch ordering / multi-GPU
cudaEvent timing on the inter-device shard boundary.

### Subtasks remaining (Bug C is OPEN)

Named as concrete subtasks per `feedback_no_risks_only_tasks` and
`feedback_no_followup_cover` — Bug C is not "deferred follow-up", it
is the unfinished part of this phase.

- **C.1 Slot-allocator audit.** Read `llama_kv_cache_find_slot` and
  the seq_id → slot mapping under concurrent fire. Specifically:
  does the slot assigned to a given seq_id depend on the order in
  which the two requests' tokens land in the scheduler queue? If
  yes, that's the integration-level non-determinism.
- **C.2 Batch composition under concurrent prefill.** When two
  requests arrive within one tick, does the resulting ubatch's
  token interleaving depend on completion order of the per-request
  prefill kernels (which is timing-dependent)?
- **C.3 cb_eval dispatch ordering.** Within a single forward, are
  intra-batch tensor copies, the `transformer_kv` write-back, and
  the per-slot KV gather always ordered the same way regardless of
  arrival timing? If any of these uses a `cudaStreamWaitEvent` with
  a timing-dependent event, that's the race.
- **C.4 Multi-GPU shard timing.** Test bug C with single-GPU (no
  inter-device split). If single-GPU reproduces the stochastic
  failure, C.4 is closed (not the boundary); if single-GPU PASSes,
  the bug lives in the cross-device cudaEvent path.

C.4 is the cheapest probe and a strong signal in either direction —
do it first.

### C.4 result (CLOSED — NOT the multi-GPU path)

`scripts/r5-probe-c4.sh ITERS=10` on `--device CUDA0` only, NP=2, fire
2 concurrent identical requests, same prompt/seed as the harness.

Result: **1 / 10 FAIL** — same ~10% rate as multi-GPU. The bug is NOT
the multi-GPU cudaEvent path / inter-device split.

Failure signature on the one failing iter (artefacts in
`/tmp/r5-probe-c4-234248/`):
- Slot 0 emits ` code` (correct first token) then immediately collapses
  to incoherent fragments (`, , and,,,,/,, and the,,, and,,.`) and
  finally attempts to **re-emit the input prompt** mid-output (`The AI,
  of artificial intelligence began in earnest with the work of Alan
  Turing...`).
- Slot 1 emits coherent text but a different valid completion than the
  NP=1 baseline (`probabilistic systems that can produce inaccurate
  outputs` vs the baseline's `fundamentally statistical pattern
  matchers`). Length 356 vs baseline 367.
- Divergence between slot 0 and the NP=1 baseline at character 5 — one
  token into the completion.

Interpretation: slot 0's attempt to re-emit the input prompt strongly
suggests its KV cache region was overwritten or mis-addressed by slot
1's concurrent prefill — slot 0 reads cells that contain slot 1's
prompt tokens instead of its own decoded state. This is a
**slot-KV cross-contamination at concurrent prefill**, not a kernel
non-determinism.

Implication for the next subtasks:
- C.1 (slot allocator) and C.2 (batch composition under concurrent
  prefill) are now the highest-priority probes.
- C.3 (cb_eval dispatch ordering) drops in priority — cb_eval is fine
  on single-GPU and the symptom is a KV-address bug not an op-order
  bug.

### Commits this session

- Submodule `1f83f681` — `ggml-cuda/mmq`: bound I=8 tile_y load to
  `tile_y_extent` (Bug B fix).
- Submodule `3ad8934e` — `server`: tensor-dump cb_eval hook for R5
  NPC bisection (diagnostic; delete after closure per
  `feedback_bake_measurement_env_gates`).
- Submodule `a6f8b198` — `tests`: production-dim MMVQ shape-invariance
  test.
- Submodule `b54a905c` — `tests`: DeltaNet shape-invariance at
  production geometry.
- Parent: submodule bumps + MEMORY appends + `scripts/r5-sanitize.sh` +
  `scripts/r5-probe-np4.sh` + `scripts/r5-capture-bisect.sh`.

## Estimated token cost

Per CLAUDE.md §8, in tokens not days:

- Localize the multi-slot path divergence — 30–60k (depends on
  whether decode-step capture is needed).
- Implement the fix (kernel? scheduler? cache layout?) — 20–80k
  (high variance; structural fixes are more expensive).
- Verify (harness + cross-prompt) — 15–25k.
- MEMORY + PHASE doc + commit — 5–10k.
- **Total**: 70–175k for a clean closure cycle.
