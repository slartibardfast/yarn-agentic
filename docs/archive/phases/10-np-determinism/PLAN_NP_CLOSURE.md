# PLAN_NP_CLOSURE — Single + Multi-GPU NP-determinism closure

**Branch**: `production/2026-q2-next`
**Predecessor**: `PHASE_NP_CLOSURE.md` (handover, no forward plan)
**Closure binding**: `scripts/test-production-np-determinism.sh`

## Goal

Production `profiles/active.sh` produces byte-identical greedy-decode output at
`--parallel N` for N ∈ {1, 2, 4, 8} on `F.3` 100-prompt corpus, with **no
LLAMA_* / GGML_* / CUBLAS_* env vars required**. Single-GPU first, then
multi-GPU.

## Premise (what we know from PHASE_NP_CLOSURE handover)

Three kernel-level fixes are baked into source and verified individually:

- CY.F.17 — MMQ stream_K off by default (`mmq.cuh`).
- A.1' — FA per-slot-kv route always-on for Qwen3.5/3.6 shape (`llama-build-context.cpp` 2697–2722).
- Singlewarp as default per-slot-kv mode (`fattn-per-slot-kv-sm75.cu`).
- delta-net `use_256` always-on (`delta-net.cu`).
- cuBLAS workspace + TF32-off baked at `common.cuh`.

Production harness `test-production-np-determinism.sh` on the realistic
~200-token default prompt FAILS with this signature:

| pair | slot 0 |
|---|---|
| NP=1 vs NP=2 | DIFFERS (340 vs 380 bytes) |
| NP=1 vs NP=4 | DIFFERS (340 vs 380) |
| NP=1 vs NP=8 | DIFFERS (340 vs 356) |
| **NP=2 vs NP=4** | **IDENTICAL** (380 vs 380) |
| NP=2 vs NP=8 | DIFFERS |
| NP=4 vs NP=8 | DIFFERS |

Two divergence boundaries to close:

- **D-α** — NP=1 differs from every NP>1. Suggests NP=1 takes a code path the
  multi-slot continuous-batching path does not (decode special-case, or
  scheduling/cache-layout that only activates when `n_parallel > 1`).
- **D-β** — NP=2 ≡ NP=4 slot-0, but NP=8 ≠ NP=4. Suggests an ubatch/page/
  scheduler boundary that crosses between 4 and 8.

The capture-tool (F.1 / CY.F.7) bound prefill-output identity only. **No
prior binding exercises the real continuous-batching decode loop.** Closure
needs a server-driven decode-step binding.

## Closure criterion

`scripts/test-production-np-determinism.sh`, `--device CUDA0` (single-GPU)
and `--device CUDA0,CUDA1` (multi-GPU), `q4_0` K/V + Hadamard, no env stack:

- F.3 100-prompt corpus × NP ∈ {1, 2, 4, 8} × 3 sweeps = 1200 runs / device
  count = 2400 runs total
- All NP>1 slot outputs byte-identical to that prompt's NP=1 baseline
- Zero divergences across the run

A pre-closure milestone (smoke binding) is the existing default+vlong
prompts × NP={1,2,4,8} × 5 sweeps with the same identity check. Useful as a
cheap regression gate while iterating; not sufficient for closure.

## Phases

### NPC.1 — Single-GPU baseline (cheap reproduction)

**Goal**: confirm the harness fails on `--device CUDA0` alone.

**Why**: eliminates multi-GPU peer-write noise. If the bug fires single-GPU
(which CY.F.17 / A.1' history says it does), single-GPU is the cheaper
diagnostic surface for NPC.2–NPC.4. If multi-GPU is *strictly worse* than
single-GPU, that becomes its own subtask in NPC.5.

**Steps**:
1. Run `DEVICE=CUDA0 bash scripts/test-production-np-determinism.sh` on the
   default ~200-tok and vlong ~500-tok prompts at NP={1,2,4,8}.
2. Record per-NP slot outputs and cross-NP diffs.
3. Verify D-α (NP=1 vs NP>1) and D-β (NP=8 boundary) reproduce on a single
   device.

**Verify**: matrix at single-GPU shows the same D-α / D-β signature as the
existing multi-GPU run (`closure-final/default/run-20260517T162725/`). If
single-GPU PASSES, the bug is multi-GPU peer-write only — skip to NPC.5.

**Budget**: ~10k tokens (no new code, just harness re-runs).

### NPC.2 — Localize D-α (NP=1 vs NP>1 path divergence)

**Goal**: identify the first layer/op where NP=1 slot-0 decode-step state
diverges from NP=2 slot-0.

**Why**: NP=1 fires a path the multi-slot continuous-batching path doesn't.
Candidates worth ruling out in order:

1. **Decode bs=1 special-case** in `fattn.cu` line 95–116. Q3.6 K/V is q4_0
   so line 113's `!ggml_is_quantized(K->type)` should be FALSE — vector
   kernel branch is *not* taken at decode. But lines 95/101 are
   `Q->ne[1]==1 && Q->ne[2]/K->ne[2]==6` for head_dim=256, *unconditionally*
   routing to `ggml_cuda_flash_attn_ext_mma_new` regardless of K/V quant.
   Q3.6 GQA = 40/8 = 5, not 6 → those gates don't fire. **Verify by grep:
   does any FA path branch on `Q->ne[1]==1` for Q3.6?**
2. **`inp_per_row_k_bound`** values at NP=1 (single sequence) vs NP=2 (two
   sequences) — is the per-row bound tensor populated differently? Same
   per-slot-kv kernel called, but bound input might force different
   K-masking behaviour.
3. **KV cache scheduling**: `cparams.n_parallel == 1` may take a different
   continuous-batching scheduler path (e.g., skip the slot-allocation
   logic, write KV to a different region, batch differently).
4. **ubatch threshold**: at NP=1 the entire ~200-tok prompt fits in one
   ubatch (size 512). At NP=2 it likely still fits, but slot-position
   indexing changes.

**Steps**:
1. Extend `llama-state-capture` (or add an in-server capture hook) to dump
   per-step l_out at slot-0 during an HTTP completion. Current F.1 is
   prefill-only (one-shot); the failure is in the decode-step loop.
2. Drive the harness with the extended capture enabled at NP=1 and NP=2,
   slot-0, same prompt + seed.
3. Bisect: find first divergent layer (binary-search by layer pair).
4. Within that layer, bisect by op (attention input, qcur/kcur/vcur,
   attn_out, mlp_out, etc.) using cb_eval names.
5. Categorize: kernel-internal vs scheduler/cache layout.

**Verify**: produces a named tag `{layer:02d}/{op}` where NP=1 slot-0 differs
from NP=2 slot-0 by at least one bit. Identifies whether the divergence is
a kernel call site that gets different *inputs* (scheduling bug) or the
same inputs producing different *outputs* (kernel shape-dep we missed).

**Budget**: 40–80k. The capture extension is the expensive piece.

### NPC.3 — Localize D-β (NP=4 → NP=8 boundary)

**Goal**: identify what scheduler/cache knob differs between NP=4 (PASS)
and NP=8 (FAIL) at slot-0.

**Why**: NP=2 ≡ NP=4 slot-0 means the multi-slot decode path is internally
stable for small N. NP=8 specifically introduces a new behaviour. Cheap
candidates:

- `-ub 512` ubatch split — at NP=8 the per-step decode batch is 8 tokens;
  unlikely to cross the 512 threshold, but prefill at 8 seqs × ~200 tok =
  1600 tokens > 512 → prefill spans 4 ubatches. NP=4 spans 2 ubatches; NP=2
  spans 1.
- KV cache page size (`--ctx-checkpoints 3`).
- Cross-device sharding only firing at NP ≥ 8 (unlikely on single-GPU).
- A scheduler heuristic that switches when n_seqs > 4.

**Steps**:
1. Sweep `-ub` ∈ {128, 256, 512, 1024} at fixed NP=8. Does the boundary
   between NP=4 (PASS) and NP=8 (FAIL) shift?
2. Sweep `--batch-size` similarly.
3. Sweep N ∈ {5, 6, 7, 8} to find the exact integer boundary.

**Verify**: produces a named knob whose change moves the PASS↔FAIL boundary,
and a hypothesis for what code path it gates.

**Budget**: 20–40k. Mostly harness re-runs.

### NPC.4 — Fix or pin the divergent path(s)

**Goal**: source-bake a fix that closes the D-α and D-β bindings.

The shape of the fix is unknown until NPC.2 + NPC.3 land. Three patterns:

- **If the divergence is kernel-call-site inputs differing** (scheduler /
  cache layout / bound-tensor populated differently): pin the input.
  E.g., force the same KV slot layout regardless of N, or set
  `inp_per_row_k_bound` to a content-independent value.
- **If the divergence is a different kernel being dispatched**: route both
  paths through the same kernel. E.g., disable the NP=1 fast-path; always
  go through the per-slot-kv route. Note: this may regress decode perf.
- **If the divergence is the same kernel producing different outputs at
  different shapes** (a sixth shape-dep bug like CY.F.17 / A.1'): bake the
  shape-invariant path.

**Steps**:
1. Implement candidate fix in `ik_llama.cpp/...`, gate behind an env-var
   first for measurement.
2. Re-run NPC.1's smoke binding (default + vlong × NP={1,2,4,8}).
3. If smoke PASSES, run the F.3 100-prompt corpus × NP × 3 sweeps,
   single-GPU.
4. If full corpus PASSES, remove the env-gate (bake into source). Per the
   CY.F.17 / A.1' pattern: ensure deterministic path is the default, opt-in
   for old behaviour if perf needs it.
5. Update `MEMORY.md` with the named fix + closure evidence.

**Verify**: zero divergences across F.3 corpus × NP × 3 sweeps,
**single-GPU**.

**Budget**: 30–80k (high variance; structural fixes are more expensive
than tactical ones).

### NPC.5 — Multi-GPU closure

**Goal**: re-run F.3 100-prompt × NP × 3 sweeps at `--device CUDA0,CUDA1
--split-mode graph --tensor-split 1,1`.

**Why**: if NPC.4 closed single-GPU, multi-GPU may close trivially or may
expose a separate peer-write divergence (Phase D's original framing).

**Steps**:
1. Run the F.3 corpus binding at multi-GPU.
2. If PASS: closure achieved. Update profiles/active.sh to remove
   `--parallel 1` constraint if applicable.
3. If FAIL: capture multi-GPU slot-0 vs single-GPU slot-0 at the first
   divergent layer. This is a separate audit (CY.F.18 territory:
   `sched->has_reduce` lifecycle, peer-stream sync).

**Verify**: zero divergences at multi-GPU.

**Budget**: 15–40k (15k if PASS, up to 40k if a multi-GPU-specific bug
surfaces and needs a new fix).

### NPC.6 — Ship

**Goal**: profiles/active.sh + harness wiring + MEMORY closure.

**Steps**:
1. Update `profiles/active.sh` to remove any remaining env-stack and to
   enable the deterministic default multi-slot setting (e.g., `--parallel 4`
   or whatever the production target is).
2. Update `scripts/test-production-np-determinism.sh` to default-run the
   F.3 corpus (not just the default prompt). Add a smoke-test mode for
   fast regression checking.
3. MEMORY.md entry: closure landed, fix name, evidence dir.
4. `PHASE_NP_CLOSURE.md` and this plan: append closure section, mark
   `[x]` boxes inline.

**Verify**: a fresh checkout passes the F.3 corpus end-to-end with no env
overrides.

**Budget**: 5–10k.

## Token budget summary

| Phase | Tokens |
|---|---|
| NPC.1 single-GPU baseline | 10k |
| NPC.2 D-α localization | 40–80k |
| NPC.3 D-β localization | 20–40k |
| NPC.4 fix + bake | 30–80k |
| NPC.5 multi-GPU closure | 15–40k |
| NPC.6 ship | 5–10k |
| **Total** | **120–260k** |

Multi-session. Pickup points are well-defined (each NPC.x has its own data
dir under `data/audit-npc-{name}/` or `/opt/models/yarn-audit-data/npc-{name}/`).

## Risks

- **R1**: NPC.4's "pin the divergent path" regresses decode perf (e.g.,
  routing NP=1 through per-slot-kv adds overhead). Mitigation: measure
  before/after with `llama-bench` at NP=1; if regression > 5%, the fix
  becomes opt-in for determinism-required deployments.
- **R2**: D-α and D-β are *independent* bugs requiring two fixes.
  Mitigation: NPC.2 and NPC.3 are independent — run them in parallel if
  budget allows.
- **R3**: NPC.5 surfaces a peer-write determinism bug beyond CY.F.18's
  `has_reduce` fix. Mitigation: don't gate NPC.6 on a "clean closure";
  treat any multi-GPU-specific finding as a follow-up sub-phase.
- **R4**: NPC.2's server-driven capture extension exceeds budget. Mitigation:
  fall back to a less expensive proxy — drive llama-cli with NP-equivalent
  flags and instrument cb_eval at a fixed layer; not a perfect substitute
  for HTTP-completion but cheaper.
- **R5**: F.3 corpus run-time at NP × sweeps × 100 prompts may exceed
  practical session limits. Mitigation: shard the run; preserve per-prompt
  results to `/opt/models/yarn-audit-data/npc-closure-final/`.

## Anti-patterns (do not repeat from PHASE_NP_CLOSURE)

- Don't relocalize known-fixed bugs. CY.F.17 / A.1' / singlewarp / cuBLAS
  workspace / TF32 / delta-net thread-count are sealed.
- Don't conflate per-slot-kv prefill identity with NP-determinism. A.1'
  bound prefill; the open bug is in decode + scheduler.
- Don't trust capture-tool identity as the closure binding. The capture
  tool's V4 path held invariants the real server does not (per
  PHASE_NP_CLOSURE §"Anti-patterns").
- Don't bake an env-var as a "fix" if it doesn't close the F.3 corpus
  binding. Bakings are correctness cleanups; closure requires the
  100-prompt PASS.
- Don't close NPC.5 on single-GPU evidence alone. The closure criterion
  is *both* device configurations.

## Critical files

**Production engine** (already source-baked, don't re-touch unless evidence
contradicts):
- `ik_llama.cpp/ggml/src/ggml-cuda/mmq.cuh` — CY.F.17.
- `ik_llama.cpp/src/llama-build-context.cpp` 2697–2722 — A.1'.
- `ik_llama.cpp/ggml/src/ggml-cuda/fattn-per-slot-kv-sm75.cu` — singlewarp.
- `ik_llama.cpp/ggml/src/ggml-cuda/delta-net.cu` — `use_256`.
- `ik_llama.cpp/ggml/src/ggml-cuda/common.cuh` — cuBLAS workspace + TF32.

**Suspect for NPC.2/3** (touch with evidence):
- `ik_llama.cpp/ggml/src/ggml-cuda/fattn.cu` — dispatch decision tree.
- `ik_llama.cpp/src/llama-build-context.cpp` — `inp_per_row_k_bound` setup.
- `ik_llama.cpp/src/llama.cpp` server slot-scheduling / KV cache layout.
- `ik_llama.cpp/ggml/src/ggml-backend.cpp` — scheduler graph splitting.

**Harness**:
- `scripts/test-production-np-determinism.sh` — binding contract.
- `data/diverse-prompts/` (F.3 corpus, 100 prompts at 5 size buckets).

**Capture tooling** (to be extended):
- `ik_llama.cpp/examples/llama-state-capture/llama-state-capture.cpp` —
  prefill-only today; NPC.2 needs a decode-step-aware variant or an
  in-server cb_eval hook driven by a real HTTP request.

**Data dirs**:
- `/opt/models/yarn-audit-data/closure-final/` — current failure evidence.
- `/opt/models/yarn-audit-data/npc-{phase}/` — new captures per NPC.x.

## Where to start

1. Read this plan + PHASE_NP_CLOSURE.md.
2. Read MEMORY.md 2026-05-17 entries (CY.F.17 bake, A.1' bake, harness
   failure, anti-patterns).
3. Start NPC.1: single-GPU reproduction. ~10k tokens to confirm the bug
   surface before any new code.
4. Branch into NPC.2 and NPC.3 in parallel once single-GPU is confirmed.

---

## Closure (2026-05-17)

**Status**: NPC.1–NPC.6 all closed.

| Phase | State | Evidence |
|---|---|---|
| NPC.1 single-GPU baseline | CLOSED | reproduced D-α / D-β |
| NPC.2 D-α localization | CLOSED | ssm_conv ruled out; ffn_up_gate-2 localized |
| NPC.3 D-β localization | CLOSED | subsumed by F.3 fixes |
| NPC.4 fix + bake | CLOSED | six default-on fixes; `PHASE_NPC4_FIX_AUDIT.md` |
| NPC.5 multi-GPU closure | CLOSED | `DEVICE=CUDA0,CUDA1` harness PASS |
| NPC.6 ship | CLOSED | `profiles/qwen36-27b-x8-deterministic.sh` + `scripts/verify-production-determinism.sh` |

**Closure binding satisfied**: `scripts/test-production-np-determinism.sh`
at `DEVICE=CUDA0,CUDA1`, default `CTX_CHECKPOINTS=3`, NP={1,2,4,8} —
all slots byte-identical to NP=1 baseline, all cross-NP slot-0 pairs
byte-identical. No env stack required (six fixes baked default-on).

**Canonical writeup**: `PHASE_NP_DETERMINISM_CLOSED.md`.

**R1 outcome (perf regression)**: realized.

  | NP | PP HEAD | PP pre-NPC | TG HEAD | TG pre-NPC |
  |---|---|---|---|---|
  | 1 | 95.74   | 176.98     | 17.95   | 18.00      |
  | 2 | 17.97   | 17.97      | 17.11   | 19.80      |
  | 4 | 17.33   | 17.32      | 18.21   | 23.49      |
  | 8 | 16.15   | 16.10      | 18.70   | 25.26      |

The ≤3% budget set in `feedback_determinism_must_co_optimize_perf.md`
is overrun (-45% NP=1 PP, -26% NP=8 aggregate TG). User accepted
2026-05-17 given the volume of work required to land F.4.1' (a new
`ncols_y>=2`, `rows_per_cuda_block=1` kernel that combines bandwidth
amortization with NP-invariance). F.4.1' is tracked in
`PHASE_NPC4_FIX_AUDIT.md` as non-blocking future work.

**What's still open (non-blocking)**:

- ~~F.4.1' kernel rewrite to close the perf gap.~~ **CLOSED 2026-05-17**
  — see `PHASE_PERF_F4_1.md`. Delivered TG +5–8% over HEAD at NP≥2 with
  NP-determinism preserved; remaining ~10–20% TG gap vs pre-NPC is
  owed by fixes #2 (PSKV) / #4 (cuBLAS per-slot loop), now the
  next named subtask.
- Evidence-dir prune: `/opt/models/yarn-audit-data/npc4-*` (~50 GB)
  and `/tmp/npc4-f41-*` (~130 MB). Salient signatures captured in
  MEMORY; the byte dumps are reproducible from harness if needed.
- Clangd-flagged unused includes from this iteration
  (`mtmd-helper.h` in `server-context.cpp`; `llama-delta-net.h` +
  `unordered_set` in `llama-build-context.cpp`).
