# PHASE 46 — Multi-GPU CLIP via Tensor-Split

> **Status (2026-05-25):** Plan only. Production is currently on
> CPU vision (`--no-mmproj-offload`) as the interim measure until
> this work lands. See `PHASE35-GRAPH-CACHE-REDESIGN.md` §15 for the
> motivating diagnosis.

## 1. Context

Qwen 3.6 27B + a 1024-token vision encoder (`mmproj-Qwen3.6-27B-Q8_0.gguf`,
334 tensors, ~700 MiB on disk) on 2× Quadro RTX 6000 (24 GiB each)
runs into a structural budget problem:

| Resource | CUDA0 | CUDA1 |
|----------|-------|-------|
| LM weights (split 1,1) | ~6.6 GiB | ~6.6 GiB |
| KV cache q4_0 @ 256k (split 1,1) | ~5 GiB | ~5 GiB |
| cuBLAS workspace + scratch | ~1.5 GiB | ~1.5 GiB |
| **Headroom (free)** | **~11.6 GiB** | **~9.0 GiB** |
| Vision encoder `cudaGraphInstantiate` working set @ 1024 tokens | **~9–11 GiB** | **~9–11 GiB** |

The vision encoder, as currently structured, has to live entirely
on **one** device. Whichever device hosts CLIP gets the full
~10 GiB working set on top of the LM+KV+scratch already there —
which fits on neither side once the vision compute graph is
instantiated.

Phase 35 Step B's allocation-aware eviction (commit `606ce62b`)
cannot help here: there is nothing useful to evict; the
single-graph-too-big constraint is the binding wall.

**The fix is structural: split the vision encoder across both
GPUs**, the same way the LM is split with `--split-mode graph
--tensor-split 1,1`. With 1:1 split, each device absorbs roughly
half the vision working set (~5 GiB), which both have headroom for.

## 2. Why this isn't already possible

`examples/mtmd/clip.cpp:488-516` (read 2026-05-25):

```cpp
if (ctx_params.use_gpu) {
    auto backend_name = std::getenv("MTMD_BACKEND_DEVICE");
    if (backend_name != nullptr) {
        backend = ggml_backend_reg_init_backend_from_str(backend_name);
    }
    if (!backend && n_backend > 1) {
        backend = ggml_backend_reg_init_backend(1, nullptr);
    }
}
if (backend) {
    LOG_INF("%s: CLIP using %s backend\n", __func__, ggml_backend_name(backend));
    backend_ptrs.push_back(backend);
    backend_buft.push_back(ggml_backend_get_default_buffer_type(backend));
}
```

Two structural limitations:

1. **One backend.** A single `ggml_backend_t` is selected and
   pushed into `backend_ptrs` / `backend_buft`. The
   `ggml_backend_sched_t` infrastructure accepts a vector of N
   backends and will partition graph nodes across them — but only
   one is ever supplied here.
2. **Default buffer type.** `ggml_backend_get_default_buffer_type`
   returns a per-device buffer type (e.g. `cuda_buffer_type_0`).
   Even if we supplied two backends, tensor weights would still be
   allocated on a single device. The LM uses
   `ggml_backend_cuda_split_buffer_type(tensor_split)` to make a
   single logical tensor physically reside on multiple devices;
   `clip.cpp` does not.

Both must change for a true multi-GPU CLIP.

## 3. Goal — the verifiable target

After this phase lands, the following must all hold:

1. `MTMD_BACKEND_DEVICE=CUDA0,CUDA1` (comma list) makes CLIP register
   both backends with its scheduler.
2. `MTMD_TENSOR_SPLIT=1,1` (or `--mmproj-tensor-split 1,1` after
   §11.2 lands) makes the mmproj weights **row-chunked** across
   CUDA0 and CUDA1 via the shared `mgpu_split_config` infrastructure
   that Path B (§12) extracts from `llama_model`. Per-device VRAM
   usage measurable to within ~5% of the requested ratio.
3. A 1024-token vision encode succeeds on the production
   configuration (27B LM + 256k context + 1024-token vision) with
   no OOM, no `cudaErrorMemoryAllocation`, no fallback to CPU.
4. Vision-encoder output for a fixed test image is **byte-identical**
   to the single-device (CUDA0) baseline at the same `--image-min-tokens`
   / `--image-max-tokens`, modulo documented reduce-order f32 epsilon
   on non-determinism-controlled paths.
5. No regression on text-decode determinism, verified by re-running
   the existing binding harnesses against the deployed build:
   - **G3.a** — `scripts/test-production-np-determinism.sh` PASS at
     NP ∈ {1, 2, 4, 8} (byte-identity).
   - **G3.c** — `scripts/r5-probe-c4.sh` 0 / 20 divergences on
     Qwen 3.6 27B.
   These are unmodified — Phase 46 only consumes them.
6. CPU-vision fallback path (current production state) still works
   if `MTMD_BACKEND_DEVICE` is unset or set to a single device.

7. **Peer-access verified at init (HARD, P2).**
   `cudaDeviceCanAccessPeer(0,1)` and `cudaDeviceCanAccessPeer(1,0)`
   both return true; `cudaDeviceEnablePeerAccess` succeeds both
   directions. If either fails, refuse to start with a clear
   non-cryptic error. See §12.3 P2.

8. **Encode latency ≤ 1.3× single-GPU baseline (HARD, P7).**
   §11.1 captures the single-GPU baseline; Phase 46 closure binds
   on this gate. Promoted from "stretch" to "binding" after the
   user named "maximum possible speed" as a constraint on
   2026-05-25. See §12.3 P7.

(Original §3 had a separate "Stretch" criterion at #7 — within
1.5×. That has been replaced by the binding #8 above at 1.3×;
the original criterion is preserved in git history.)

## 4. Implementation plan

### Step 1 — Multi-backend parsing in clip.cpp (~30 LoC)

`examples/mtmd/clip.cpp:488-516`:

- Tokenise `MTMD_BACKEND_DEVICE` on `,`.
- For each token, call `ggml_backend_reg_init_backend_from_str` and
  push the result to `backend_ptrs` / `backend_buft`.
- If multiple backends were pushed, the first becomes the "primary"
  (`backend`) for legacy single-backend code paths.
- If zero CUDA backends were initialised, fall through to the
  existing CPU fallback (no regression).

Acceptance for Step 1: `MTMD_BACKEND_DEVICE=CUDA0,CUDA1` produces
the journal line `clip_ctx: have 2 back-ends:` showing both CUDA0
and CUDA1 registered, plus a CPU backend at the end. Vision encode
still works because by default all tensors live on the first
backend (no split-buffer yet).

### Step 2 — Row-chunked split via shared `mgpu_split_config` infra (Path B)

**Superseded by §12.4 sub-steps B.0-B.8.** Brief sketch retained
here; binding plan is §12.

What Step 2 actually is, under Path B:

- The LM's GRAPH-mode split infrastructure (`create_split`,
  `prepare_split_tensors`, `model.splits`, `model.split_buft`,
  `ctx_for_layer_split`) is extracted to a shared header and
  generalized into an `mgpu_split_config` struct.
- Both `llama_model` and `clip_ctx` consume the same struct.
- `clip.cpp` populates one from `MTMD_TENSOR_SPLIT` /
  `--mmproj-tensor-split`, calls the shared `create_split` to
  row-chunk mmproj weights across CUDA0/CUDA1, lets
  `ggml_backend_sched_t` partition the compute graph as it
  already does for the LM under `--split-mode graph`.
- Before any of this lands, the 5 formal specs in §12.2 must
  PASS (B.0-full).
- After it lands, LM gates G3.a, G3.c,
  `test-n-stream-kv-layout`, Phase 45 D10.a all re-cert (B.6).
- Encode latency must be ≤ 1.3× single-GPU baseline (B.7, HARD).

Acceptance for Step 2 (delegated to §12.4 B.0-B.7): see §12.4.

### Step 3 — Test contracts

Land RED first (per CLAUDE.md §4, test-first discipline). All in
`ik_llama.cpp/tests/`:

- `test-clip-multi-backend-init.cpp` — RED: assert 2 backends in
  `backend_ptrs` after init with `MTMD_BACKEND_DEVICE=CUDA0,CUDA1`.
  Will fail until Step 1 lands.
- `test-clip-weight-split.cpp` — RED: load mmproj with
  `MTMD_TENSOR_SPLIT=1,1`, query per-device buffer sizes, assert
  both > 0 and within 5 % of each other. Will fail until Step 2.
- `test-clip-encode-equivalence.cpp` — RED: encode the existing
  in-tree fixture `examples/mtmd/test-1.jpeg` (MIT-licensed under the
  repo root `LICENSE`, 640×488 JPEG, SFW — the canonical upstream
  mtmd test image, already used by `llama-mtmd-cli`) with
  (a) CUDA0-only and (b) CUDA0+CUDA1 split; assert outputs equal to
  f32 epsilon. **No new image bytes are added to the tree.** Will
  fail until Step 2 is correct.
- Integration: `scripts/verify-multigpu-clip.sh` — end-to-end
  vision-encode test against a running `llama-server` with the
  production profile; assert no OOM, /health stays 200, response
  contains the expected output. After the multi-GPU encode passes,
  the script also invokes
  `scripts/test-production-np-determinism.sh` and
  `scripts/r5-probe-c4.sh` so a single binding run covers both the
  vision goal and the text-decode regression guards (Goal #5).

### Step 4 — Production rollout

1. Land Steps 1+2 on a dev branch
   (`dev/2026-q2-multigpu-clip`) with all four tests GREEN.
2. Cherry-pick to `production/2026-q2-next`. Build with
   `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15`.
3. Edit the profile script to drop `--no-mmproj-offload` and add
   `MTMD_TENSOR_SPLIT=1,1` to the systemd drop-in. Set
   `MTMD_BACKEND_DEVICE=CUDA0,CUDA1`.
4. `scripts/deploy-llama-server.sh`. Verify `clip_ctx: have N
   back-ends:` shows N ≥ 3 (CUDA0, CUDA1, CPU).
5. Run `scripts/verify-multigpu-clip.sh` against production.
6. If anything fails: unset `MTMD_BACKEND_DEVICE` and
   `MTMD_TENSOR_SPLIT`, re-add `--no-mmproj-offload` to the profile,
   restart. Production is back to CPU vision — same state as the
   2026-05-25 interim.

## 5. Orphan audit — **superseded by §12.5** (Path B touches LM hot path)

The original §5 audit covered only the CLIP-side patch sites and
declared "no work orphaned." That holds only for Path A/C, not for
Path B. Under Path B (chosen 2026-05-25), the LM-side touch surface
listed in §12.5 is material, and the binding mitigation is the LM
gate re-cert at B.6.

Path B's LM-side touch surface in summary (full table at §12.5):

- `create_split`, `prepare_split_tensors`: extracted to shared
  header; 13 call-site touches in `llama-load-tensors.cpp`.
- `ctx_for_layer_split`: refactored to take `mgpu_split_config &`;
  ~20 caller sites in arch-specific tensor loaders.
- `model.splits`, `model.split_buft`, `model.devices`: migrated
  into `mgpu_split_config`; ~30 read sites updated.
- Buft-setup at `llama.cpp:4168-4198`: refactored to populate the
  new struct.

None of these is in the prior K/V family HIGH-risk list, but the
buft-setup family is adjacent and was not previously formally
specified. B.0's `BuftSetupLoop.tla` closes that gap before code.

## 6. Risks

1. **Cross-device peer access overhead.** TU102 + NV2 NVLink is
   measured working between CUDA0 and CUDA1 (per `project_xeon_host_hardware.md`),
   but the vision encoder's activations crossing device boundaries
   will add latency. Mitigated by P1 (f16 exchange default, §12.3)
   which halves cross-device transfer bytes. Bounded by P7
   (≤ 1.3× single-GPU baseline, §12.3, HARD).
2. **cuBLAS workspace duplication.** Each device that hosts vision
   ops allocates its own workspace. Per-device cost increases by
   ~256 MiB. Headroom budget at the top of this doc accounts for
   this. P6 default (no trim) per §12.3.
3. **Determinism.** Row-chunked split via shared infra goes through
   the same `ggml_backend_cuda_split_buffer_type` reduction path the
   LM uses in production under `--split-mode graph`. The reduction
   order is fixed at the kernel level (parity envelope spec'd by
   `specs/cuda_nccl_allreduce.allium`). f16 exchange (P1) adds
   epsilon at the cross-device boundary; vision-encoder output feeds
   high-temp autoregressive decode, which tolerates this.
4. **Upstream divergence.** `clip.cpp` in mainline llama.cpp may
   evolve. We will rebase rather than maintain divergent state once
   a stable design is in place. Path B's shared-header extraction
   *increases* upstream-divergence surface slightly because the
   shared infra is ik-fork-specific; this is accepted scope.
5. **LM regression under B.3–B.4 refactor (NEW, Path B-specific).**
   Refactoring `create_tensors_helper` and the buft-setup loop is
   semantics-preserving in design but touches LM hot-path code with
   ~50 read sites across architecture-specific tensor loaders.
   Mitigated by B.6 LM gate re-cert (G3.a NP∈{1,2,4,8}, G3.c r5
   probe, `test-n-stream-kv-layout`, Phase 45 D10.a 3-slot smoke).
   Without all four PASS, Path B is incomplete and the phase stays
   OPEN.
6. **Spec failure under B.0 (NEW, B.0-full-specific).** Any of the
   five Phase-46 specs (§12.2) failing TLC / Alloy Analyzer blocks
   Path B entirely — by design. Spec failure is not a "fix code and
   re-spec" loop; it's a "the design is wrong, redesign before
   coding" signal.
7. **Perf gate failure under B.7 (NEW, P7-specific).** If encode
   latency at production load exceeds 1.3× the §11.1 single-GPU
   baseline, Phase 46 stays OPEN. Possible follow-ups: increase the
   exchange precision (P1 f16 → f32 round-trip), tune the split
   ratio (P4 away from strict 1:1), or accept the gap and re-open
   the perf budget with explicit user authorization.

## 7. Out of scope

- Optimising vision-encoder latency. The goal is to fit, not to be
  fast.
- Multi-GPU CLIP for **non-CUDA** backends (Vulkan, Metal, ROCm).
  Out of scope; current production is CUDA-only.
- Rewriting the vision encoder graph for better cross-device
  locality. Step 2 uses the existing graph layout with split
  weights; scheduler decides ops based on data residency.
- Backporting upstream's vision changes. Independent track.

## 8. Estimated cost (in tokens, per CLAUDE.md §8) — revised for Path B + B.0-full + perf gate

Original (Path A/C, no formal specs): ~105-125k.
§11 additions (baseline capture, CLI parity, observation, deploy
guard): +20-25k → ~125-150k.
**Path B + B.0-full + perf gate (current):** ~225-275k. Full
breakdown in §12.4 table.

Cost drivers vs original:
- B.0 formal specs (5 specs, write + model-check): +60-95k
- B.1–B.4 LM-side refactor (extract + struct + helper + buft-setup): +35-40k
- B.5 clip.cpp wiring with P1+P2 gates: +25k (similar to original Step 2)
- B.6 LM gate re-cert: +15k
- B.7 perf gate (capture + verify ≤ 1.3×): +10k

This is a **~4-day phase, not a 1-day phase.** Justified by the
three constraints named on 2026-05-25: do not split, full formal
coverage, maximum possible speed.

## 9. Files

| Path | Role |
|------|------|
| `ik_llama.cpp/examples/mtmd/clip.cpp:488-516` | Backend init + P2 peer-access gate (Step 1 / B.5) |
| `ik_llama.cpp/examples/mtmd/clip.cpp` (weight load) | Shared `mgpu_split_config` consumer (Step 2 / B.5) |
| `ik_llama.cpp/src/ggml-mgpu-split.h` | **NEW** — shared header (B.1, B.2) |
| `ik_llama.cpp/src/ggml-mgpu-split.cpp` | **NEW** — extracted `create_split` + `prepare_split_tensors` |
| `ik_llama.cpp/src/llama-load-tensors.cpp` | Call sites migrated to shared header (B.1, B.3) |
| `ik_llama.cpp/src/llama-model.h` | `model.splits` etc. migrate into `mgpu_split_config` member (B.3) |
| `ik_llama.cpp/src/llama.cpp:4168-4198` | Buft-setup refactored to populate struct (B.4) |
| `yarn-agentic/specs/mgpu-split/MgpuSplitConfig.allium` | **NEW** (B.0 spec #1) |
| `yarn-agentic/specs/mgpu-split/BuftSetupLoop.tla` | **NEW** (B.0 spec #2) |
| `yarn-agentic/specs/mgpu-split/CreateSplitBalance.tla` | **NEW** (B.0 spec #3) |
| `yarn-agentic/specs/mgpu-split/ClipCrossDeviceFlow.tla` | **NEW** (B.0 spec #4, extends `AsyncReduce.tla`) |
| `yarn-agentic/specs/mgpu-split/CrossCodepathConsistency.allium` | **NEW** (B.0 spec #5) |
| `ik_llama.cpp/tests/test-clip-multi-backend-init.cpp` | NEW (Step 3) |
| `ik_llama.cpp/tests/test-clip-weight-split.cpp` | NEW (Step 3) |
| `ik_llama.cpp/tests/test-clip-encode-equivalence.cpp` | NEW (Step 3) |
| `ik_llama.cpp/tests/test-clip-encode-latency.cpp` | **NEW** (B.7 perf gate, ≤ 1.3× single-GPU) |
| `ik_llama.cpp/examples/mtmd/test-1.jpeg` | **Reused fixture** — already MIT, already in tree |
| `ik_llama.cpp/tests/CMakeLists.txt` | Register tests |
| `yarn-agentic/scripts/verify-multigpu-clip.sh` | NEW integration harness (wraps G3.a + G3.c at the end) |
| `yarn-agentic/scripts/test-production-np-determinism.sh` | **Reused** (G3.a) — invoked by the integration harness |
| `yarn-agentic/scripts/r5-probe-c4.sh` | **Reused** (G3.c) — invoked by the integration harness |
| `yarn-agentic/PHASE46-MULTIGPU-CLIP-TENSOR-SPLIT.md` | This doc |
| `yarn-agentic/docs/SUMMARY.md` | Add Phase 46 entry |
| `/home/llm/profiles/qwen36-27b-x1-vanilla.sh` | Drop `--no-mmproj-offload` at deploy |
| `/etc/systemd/system/llama-server.service.d/02-cuda-graph-probe.conf` | Add `MTMD_BACKEND_DEVICE` + `MTMD_TENSOR_SPLIT` at deploy |

## 10. Checkboxes

- [x] Step 1: multi-backend parsing in clip.cpp (landed in B.5 part 1 — see below)
- [~] **Step 2: Path B sub-steps (§12.4) — partial; see status below**
  - [x] **B.0** — Five formal specs land in `specs/mgpu-split/` (commit `34b7151`)
    - [x] `MgpuSplitConfig.allium` — `allium check` PASS, 0 errors (16 invariants)
    - [x] `BuftSetupLoop.tla` — TLC PASS on all 4 mode .cfgs (graph/layer/attn/none)
    - [x] `CreateSplitBalance.tla` — TLC PASS, 106 distinct states (termination + balance + sum)
    - [x] `ClipCrossDeviceFlow.tla` — TLC PASS at N_LAYERS=2, 11,238 distinct states
    - [x] `CrossCodepathConsistency.allium` — `allium check` PASS, 0 errors (12 invariants)
    - [x] `docs/SUMMARY.md` "Multi-GPU split formal specs" section added
  - [x] **B.1** — `create_split` extracted to `ggml/{include,src}/ggml-mgpu-split.{h,cpp}` (submodule commit `f2704241`); 13 LM-side call sites continue via thin wrapper
  - [x] **B.2** — `ggml_mgpu_split_config` struct + `ggml_mgpu_split_config_check` runtime invariant verifier (submodule `4ce3e51f`)
  - [x] **B.3** — `create_tensors_helper` reads via `cfg.buft_layer[i].first/.second`; struct mirrors model fields (submodule `ffaa94c3`)
  - [x] **B.4** — `model.mgpu_split_config` member added; populated immediately after the buft-setup loop at `llama.cpp:4198`; check runs and logs PASS / first failure (submodule `69d7ffe7`)
  - [~] **B.5** — partial (submodule `ba186fdb` for part 1)
    - [x] Multi-backend init from comma-separated `MTMD_BACKEND_DEVICE`
    - [x] P2 peer-access gate (`ggml_backend_cuda_can_access_peer`) — refuses to start on failure
    - [ ] **B.5b — multi-device weight residency** (OPEN). Requires per-tensor `ggml_split_tensor_t` decoration on mmproj weights before `alloc_ctx_tensors_from_buft` at `clip.cpp:3804`. Mirrors the LM-side `llama_layer::split_*` pattern at `llama-model.h:201-210` and the per-arch `prepare_split_tensors` calls at `llama-load-tensors.cpp:3931+`. Estimated ~80-150 LoC plus clip-model-struct additions. **Phase 46 closure binds on this.**
    - [ ] B.5c CLI flags: `--mmproj-devices`, `--mmproj-tensor-split`, `--mmproj-split-mode`, `--mmproj-smf16/smf32`, `--mmproj-smgs`
    - [ ] B.5d P1 f16 cross-device exchange default
    - [ ] `test-clip-multi-backend-init.cpp` GREEN
    - [ ] `test-clip-weight-split.cpp` GREEN
    - [ ] `test-clip-encode-equivalence.cpp` GREEN
  - [ ] **B.6** — LM gate re-cert (HARD; deferred to maintenance window — production service uses both GPUs at capacity, test binary cannot allocate concurrent VRAM)
    - [ ] G3.a `test-production-np-determinism.sh` PASS NP∈{1,2,4,8}
    - [ ] G3.c `r5-probe-c4.sh` 0/20 divergences
    - [ ] `test-n-stream-kv-layout` PASS
    - [ ] Phase 45 D10.a 3-slot smoke PASS
    - **Note:** B.1-B.4 are semantics-preserving by construction (cfg mirrors model fields; reads through cfg are equivalent to reads through model fields). Empirical bit-identity verification remains as the binding closure step.
  - [ ] **B.7** — CLIP perf gate (HARD, P7): encode latency ≤ 1.3× §11.1 single-GPU baseline. Cannot run until B.5b lands (current state: weights single-device; multi-backend init has no observable benefit).
    - [ ] `test-clip-encode-latency.cpp` GREEN
  - [ ] **B.8** — Production rollout via deploy script + rollback drill
- [ ] Step 4: production rollout
  - [ ] Cherry-pick to `production/2026-q2-next`
  - [ ] Deploy via `scripts/deploy-llama-server.sh`
  - [ ] `scripts/verify-multigpu-clip.sh` PASS against production
  - [ ] `scripts/test-production-np-determinism.sh` (G3.a) PASS at NP ∈ {1, 2, 4, 8}
  - [ ] `scripts/r5-probe-c4.sh` (G3.c) 0 / 20 divergences
  - [ ] CPU-vision fallback path still works (rollback drill)
- [ ] Close: this PHASE doc updated with final implementation
      notes; MEMORY.md entry appended.

## 11. Scope changes (2026-05-25 — pre-execution review)

After §1–10 were written, a review identified four adjacent items
that strengthen Phase 46 without changing its structural claim.
Folded in here rather than rewritten into §4 so the original scoping
decision stays visible in doc history.

Budget impact: original ~105–125k tokens → revised **~125–150k
tokens** (+20-25%). Justified per item below.

### 11.1 — Baseline latency capture (precondition for §3 stretch #7)

§3 stretch criterion 7 ("within 1.5× single-GPU baseline") is
currently unverifiable: production is on CPU vision, so no
single-GPU encode latency is on record. The Phase 35 work didn't
pin it either.

Action — *before* Step 1 lands:

1. Stop the service.
2. Temporarily flip the profile to single-device GPU vision at a
   reduced image budget that fits on CUDA0 alone (suggest
   `--image-max-tokens 256` and `--image-min-tokens 256` — small
   enough to dodge the OOM that drove §1's headroom table).
3. Send one vision request with `examples/mtmd/test-1.jpeg`.
4. Record `mtmd: image encoded in N ms` from the journal.
5. Restore the CPU-vision profile, restart.

Acceptance: the recorded number is written into §6 as the reference
for stretch criterion 7. Estimated **+3k tokens, ~10 min wallclock**.

### 11.2 — CLI flag parity with LM `--split-mode graph` family

Original plan routed config through `MTMD_BACKEND_DEVICE` and
`MTMD_TENSOR_SPLIT` env vars. The LM uses a richer CLI surface
under `--split-mode graph`: `--tensor-split`, `-smf16/-smf32`,
`-smgs`. Mirror it for CLIP:

- `--mmproj-devices CUDA0,CUDA1` — peer of `--tensor-split`'s
  device-selection role.
- `--mmproj-tensor-split 1,1` — peer of `--tensor-split` itself.
- `--mmproj-split-mode graph` — explicit (default; only `graph` is
  supported in Phase 46. Other LM modes — `layer`, `attn` — are
  out-of-scope for CLIP).
- `--mmproj-smf16` / `--mmproj-smf32` — exchange precision
  (**P1, §12.3: f16 default ON** for maximum speed).
- `--mmproj-smgs` — graph-scheduling control, passthrough of the
  LM's `-smgs` knob for tuning.

Plumb in `common/common.cpp` → `common_params.mmproj_*` fields,
then pass to clip.cpp during init.

The env vars stay as zero-cost compat fallback (env-var route read
only if the CLI flag is unset). Profile script and systemd drop-in
migrate to flags at deploy time.

Acceptance: `llama-server --help` lists all five flags;
`--mmproj-devices CUDA0,CUDA1 --mmproj-tensor-split 1,1
--mmproj-split-mode graph --mmproj-smf16` produces the same
multi-backend init + row-chunked split + f16 exchange as the
env-var route. Estimated **+50-70 LoC across `common/common.cpp`
and `clip.cpp`, +12-18k tokens** (was +8-13k for the smaller flag
set).

### 11.3 — Phase 35 §15.7 closure observation (free, passive)

PHASE35-GRAPH-CACHE-REDESIGN.md §15.7 has one open closure
criterion: capture at least one `evict_pressure` event under real
workload. Phase 46's `verify-multigpu-clip.sh` will exercise large
vision + LM graphs concurrently — exactly the load Step B was
designed for.

Add ~5 lines to `verify-multigpu-clip.sh`:

```bash
EVICT_BEFORE=$(journalctl -u llama-server.service --since="-1m" | grep -c "evict_pressure")
# ... run vision encode ...
EVICT_AFTER=$(journalctl -u llama-server.service --since="-1m" | grep -c "evict_pressure")
echo "PHASE35 §15.7 observation: evict_pressure events during this run: $((EVICT_AFTER - EVICT_BEFORE))"
```

Acceptance: harness emits the count line. If ≥ 1, propagate to
PHASE35 §15.7 closure with the journal entry. If 0 across the
harness run, that's also a real signal (the 4096 MiB headroom may
be too generous; potential follow-up but **not** a Phase 46
blocker). Estimated **+2k tokens**.

### 11.4 — Deploy-script regression guard for multi-backend path

`scripts/deploy-llama-server.sh` already refuses to install a
`libggml.so` containing the legacy `GGML_SCHED_MAX_SPLITS` assert
string (CLAUDE.md §9). Add a symmetric guard for clip.cpp's
multi-backend path:

1. Step 1 emits a distinctive `LOG_INF` from the multi-backend
   parse path — e.g. `clip_ctx: multi-backend init: N devices`.
2. B.1 introduces a shared-infra symbol — e.g.
   `ggml_mgpu_create_split`. A Path B-complete binary contains this
   symbol; a pre-Path-B binary does not.
3. Deploy script does **both** checks:
   ```bash
   if ! strings build/bin/llama-server | grep -q "multi-backend init"; then
       echo "FATAL: build missing multi-backend CLIP — refusing to deploy" >&2
       exit 1
   fi
   if ! nm -D build/lib/libggml.so | grep -q "ggml_mgpu_create_split"; then
       echo "FATAL: build missing shared mgpu_split_config infra — refusing to deploy" >&2
       exit 1
   fi
   ```
4. Both checks fire on the build-tree binary *before* the install
   step, so a bad build can't reach `/opt/llm-server/`.

Acceptance: deploy script refuses to install a `llama-server`
binary that doesn't contain the multi-backend init string. The
guard is opt-out-able with an explicit `--allow-no-mmproj-mgpu`
flag for emergency rollback to a pre-Phase-46 binary. Estimated
**+10-15 LoC in deploy script, +3k tokens**.

### 11.5 — CLIP perf gate (HARD, P7, §12.3)

Encode latency for the 1024-token vision encode against
`examples/mtmd/test-1.jpeg` (the in-tree fixture) on the
production configuration must be **≤ 1.3× the single-GPU
baseline captured in §11.1**.

Measurement: `test-clip-encode-latency.cpp` runs N=10 encodes,
discards first 2 (warm-up), reports median + p95. Median ≤ 1.3×
baseline is the gate. p95 ≤ 1.5× baseline is a soft check (logs
a warning, doesn't fail the gate).

If the gate fails:
- Phase 46 stays OPEN.
- Possible follow-up paths: tune P1 (f16 → f32 round-trip);
  adjust P4 split ratio; instrument layer-by-layer transfer time
  to identify the bottleneck; verify P3 compute/transfer overlap
  holds in practice (matches the spec).
- Do **not** accept the gap silently. "Maximum possible speed"
  per user 2026-05-25.

Acceptance: `test-clip-encode-latency.cpp` GREEN. Estimated
**+10k tokens** (already in B.7).

### Checkboxes for §11

- [ ] §11.1 — single-GPU baseline latency captured and pasted into §6
- [ ] §11.2 — full CLI flag family implemented and tested
  - [ ] `--mmproj-devices`, `--mmproj-tensor-split`,
        `--mmproj-split-mode graph`, `--mmproj-smf16/--mmproj-smf32`,
        `--mmproj-smgs` all in `llama-server --help`
  - [ ] Env-var fallback still works (compat)
- [ ] §11.3 — `verify-multigpu-clip.sh` emits `evict_pressure` event
      count; if ≥ 1, PHASE35 §15.7 closed in the same commit
- [ ] §11.4 — Deploy-script regression guard checks both the
      multi-backend log string AND the shared-infra symbol; verified
      by attempting to deploy a built-without-Path-B binary (must refuse)
- [ ] §11.5 — CLIP perf gate `test-clip-encode-latency.cpp` GREEN
      (median ≤ 1.3× §11.1 baseline)

## 12. Design pivot — Path B (extract LM split infra), B.0-full formal-spec-first, perf-gated (2026-05-25, post-grep)

Supersedes the LAYER-flavored §12 draft that briefly appeared in
this session and was never committed. That draft was based on an
incorrect understanding of `LLAMA_SPLIT_MODE_GRAPH` semantics in
the ik fork; corrected by reading `llama-load-tensors.cpp` +
`llama.cpp:4168-4198` end-to-end. See §12.0.

### 12.0 — Ground-truth correction

In the ik fork, `LLAMA_SPLIT_MODE_GRAPH` is **not** a per-layer
pinning mode. It is **row-chunked tensor split** via
`ggml_backend_cuda_split_buffer_type` (constructed at
`llama.cpp:4168-4198` via `llama_default_buffer_type_split`), with
the fork's own row-chunking infrastructure layered on top:
`model.splits`, `model.split_buft`, `create_split` (the
mem_used-balanced row-chunk algorithm at
`llama-load-tensors.cpp:351`), `prepare_split_tensors`, and the
`ctx_for_layer_split` indirection used at ~20 sites across the
architecture-specific tensor loaders.

The "graph" in the mode name refers to the
`ggml_backend_sched_t` graph-partitioning that happens **after**
weights are physically distributed — not to layer-pinning.
**Production LM runs `--split-mode graph` today.**

The earlier §12 draft confused this with upstream's
`LLAMA_SPLIT_MODE_LAYER`. The corrected understanding makes the
original Step 2 design (using `ggml_backend_cuda_split_buffer_type`)
directionally right — it just needed to be expressed inside the
ik fork's GRAPH-mode framework, not as a one-off call from
clip.cpp.

### 12.1 — Path B chosen (user-authorized 2026-05-25)

**Phase 46 will extract the LM's GRAPH-mode split infrastructure
to a shared header and consume it from both `llama_model` and
`clip_ctx`.** Path B (LM-extraction) chosen over Path A (raw
primitive in clip.cpp) and Path C (mtmd-only mirror) on 2026-05-25
with explicit user authorization, knowing the larger scope.

Why B over A/C: long-term codebase health. One implementation,
two consumers. Avoids drift. Strongest reading of "authentic to
the codebase."

### 12.2 — B.0-full: formal-spec-first (test-first taken to its strongest reading)

All five Phase-46 formal specs must be **written and machine-checked**
before any C++ touches Path B. This is stronger than CLAUDE.md §4's
RED-tests-first discipline.

Specs to land under `specs/mgpu-split/`:

1. **`MgpuSplitConfig.allium`** — Alloy-style model of the
   `mgpu_split_config` struct invariants:
   - Every weight tensor has exactly one residency
   - Residency consistent with `buft_layer`
   - Sum of per-device allocations ≤ per-device capacity
   - `splits[i]` ordering valid (monotonic, normalized)
   - No orphan layers (every layer has both `split_buft` and
     `buft_layer` assigned)

2. **`BuftSetupLoop.tla`** — TLA+ model of `llama.cpp:4168-4198`
   buft-assignment loop:
   - For every `i ∈ [i_gpu_start, n_layer)`: `buft_layer[i]`
     assigned `{split_buft, buft_layer_offload}`
   - `main_gpu` selection consistent with `splits[]`
   - Loop terminates
   - Behavior under `split_mode ∈ {NONE, LAYER, ATTN, GRAPH}` all
     distinct and verified

3. **`CreateSplitBalance.tla`** — TLA+ model of `create_split()`
   at `llama-load-tensors.cpp:351`:
   - Termination: the two `while` loops at `:374-395, :396-414`
     terminate in finite steps for all valid inputs
   - Balance: `|result[i] - splits[i] * nchunk| ≤ 1` for all `i`
     post-condition
   - Sum invariant: `sum(result) == nchunk`
   - No negative values

4. **`ClipCrossDeviceFlow.tla`** — extends `AsyncReduce.tla`.
   Models per-layer cross-device transfer for the CLIP encoder
   topology:
   - Deadlock-freedom (inherited)
   - Layer-i output available on both devices before
     layer-(i+1) starts on either
   - Compute/transfer overlap holds:
     `comm_state[d,i] = REDUCE_KERNEL` can be concurrent with
     `compute_state[d,i+1] = COMPUTING`
   - Encode latency bound: `max{compute_d + transfer_d}` per
     layer (binds on perf gate P3)

5. **`CrossCodepathConsistency.allium`** — Alloy-style model
   asserting that any state legal for the LM under the new
   `mgpu_split_config` is also legal for CLIP, and vice versa:
   - Invariant set of `MgpuSplitConfig` holds independently of
     consumer
   - Refactoring the LM to use the struct preserves all LM-side
     invariants from `BuftSetupLoop` and `CreateSplitBalance`

**Acceptance for B.0:**
- All 5 specs land in `specs/mgpu-split/`
- TLC runs (BuftSetupLoop, CreateSplitBalance, ClipCrossDeviceFlow)
  PASS within bounded state space; state-count budget documented
  in each `.cfg`
- Alloy Analyzer (MgpuSplitConfig, CrossCodepathConsistency) finds
  no counterexample within scope: ≥ 8 for residency-bound, ≥ 4 for
  cross-codepath
- `docs/SUMMARY.md` gets a new "Multi-GPU split formal specs"
  subsection linking each
- **No C++ code in Path B is permitted until B.0 PASS.**

### 12.3 — Performance baked in (maximum possible speed, user-named 2026-05-25)

The "maximum possible speed" constraint promotes these from
optional design considerations to **load-bearing acceptance
criteria**:

| ID | Decision | Default | Mechanism |
|---|---|---|---|
| **P1** | f16 cross-device exchange | f16 ON | `--mmproj-smf16` / `--mmproj-smf32` flag pair, mirrors LM's `-smf16/-smf32` |
| **P2** | Peer-access verification at init (HARD) | required | `cudaDeviceCanAccessPeer(0,1)` AND `cudaDeviceEnablePeerAccess(1,0)` both directions; refuse to start if either fails |
| **P3** | Compute/transfer overlap | required | Verified by `ClipCrossDeviceFlow.tla`; checked empirically via encode-latency gate |
| **P4** | Tensor split ratio | `1,1` static | Not exposed as a knob; default suffices given §1 headroom |
| **P5** | Phase 35 × Phase 46 graph cache interaction | uncached for multi-backend | Multi-backend graphs run uncached; Phase 35 cache stays single-backend. ~5-10ms instantiate cost is negligible vs ~30-50ms encode |
| **P6** | cuBLAS workspace | per-device default, not trimmed | §1 headroom accounts for it |
| **P7** | Encode latency (HARD gate) | ≤ 1.3× single-GPU baseline | §11.1 captures baseline; B.7 binds on this. Was a stretch in original §3, now binding |

### 12.4 — Sub-step breakdown (replaces §4 Step 2)

| Step | Type | LoC | Tokens |
|---|---|---|---|
| **B.0** | Write 5 formal specs (§12.2) | ~165-200 spec | ~50-80k |
| **B.0.1** | Model-check all 5 until PASS | — | ~10-15k |
| **B.1** | Extract `create_split` + `prepare_split_tensors` to shared header (`src/ggml-mgpu-split.{h,cpp}`); update 13 call sites in llama-load-tensors.cpp | ~100 | ~15k |
| **B.2** | Define `mgpu_split_config` struct per the verified `MgpuSplitConfig.allium` spec | ~50 | ~8k |
| **B.3** | Refactor `create_tensors_helper` to take `mgpu_split_config &` instead of reading `model.splits` / `model.split_buft` directly | ~50 | ~10k |
| **B.4** | Refactor buft-setup at `llama.cpp:4168-4198` to populate the new struct | ~30 | ~5k |
| **B.5** | Wire `clip.cpp`: multi-backend init + populate `mgpu_split_config` + use shared `create_split` + P2 peer-access gate + P1 f16 default | ~150 | ~25k |
| **B.6** | LM gate re-cert: G3.a NP∈{1,2,4,8}, G3.c r5-probe-c4, `test-n-stream-kv-layout`, Phase 45 D10.a 3-slot smoke; **all PASS** | test runs | ~15k |
| **B.7** | CLIP perf gate: encode latency ≤ 1.3× §11.1 baseline (P7, HARD). If fails, phase stays OPEN. | test + measure | ~10k |
| **B.8** | Production rollout via `scripts/deploy-llama-server.sh` + rollback drill | ops | ~5k |
| **Total** | | ~165-200 spec LoC + ~380 code LoC + 13-20 call-site touches | **~225-275k** |

### 12.5 — Path B orphan audit (revises §5)

| Region | File:Line | Touch | Risk |
|---|---|---|---|
| `create_split`, `prepare_split_tensors` | `src/llama-load-tensors.cpp:351` (def); 13 calls at :3906, :4230, :4255-56, :4397, :4417, :4458, :4481, :4498, :4504, :4510, :4516, :4537 | Extracted to shared header | LOW — semantics-preserving move; B.6 binds |
| `ctx_for_layer_split()` | `src/llama-load-tensors.cpp:173` | Refactor to take `mgpu_split_config &`; ~20 caller sites in arch-specific tensor loaders | LOW-MEDIUM — many call sites but mechanical |
| `model.splits`, `model.split_buft`, `model.devices` | `src/llama-model.h:496` + ~30 read sites | Migrate fields into `mgpu_split_config`; model holds one struct | MEDIUM — refactor of model state; **not** in prior K/V HIGH-risk list, but adjacent |
| Buft-setup at `llama.cpp:4168-4198` | `src/llama.cpp:4168-4198` | Inline → populate `mgpu_split_config` | LOW |
| `clip.cpp` backend init | `examples/mtmd/clip.cpp:488-516` | Multi-backend, peer-access gate, f16 default | NEW additive code path |
| `clip.cpp` weight loader | `examples/mtmd/clip.cpp` (weight load) | Use shared `mgpu_split_config` + `create_split` | NEW additive code path |

**B.6 LM gate re-cert is the binding mitigation** for the
LM-side touch surface. Without all four gates PASS, B is
incomplete and the phase stays OPEN.

### 12.6 — Changes to other sections

Earlier sections updated in place (this is the authoritative
list; original wording in git history):

- **§3 Goal #2** — row-chunked across CUDA0/CUDA1 via shared
  `mgpu_split_config` (not generic "split 1:1").
- **§3 Goal #4 (NEW)** — encode latency ≤ 1.3× single-GPU
  baseline (HARD, replaces original §3 stretch #7).
- **§3 Goal #5 (NEW)** — peer-access verified at init (`cudaDeviceCanAccessPeer` + `EnablePeerAccess` both directions); refuse to start otherwise.
- **§4 Step 2** — redirects to §12.4. Steps 1, 3, 4 still hold
  at a sketch level; B.0-B.8 in §12.4 is the binding plan.
- **§5** — superseded by §12.5.
- **§6 Risks** — adds risk #5 (LM regression under B.3-B.4
  refactor; mitigated by B.6), #6 (spec failure; gates Path B
  entirely), #7 (perf gate failure under B.7; phase stays open).
- **§8 Cost** — revised: ~225-275k tokens (was 125-150k).
- **§10 Step 2 sub-items** — replaced by B.0-B.8 hierarchy.
- **§11.2** — CLI flag list expanded with `--mmproj-split-mode graph`
  (default; mirrors LM), `--mmproj-smf16` (P1 default ON),
  `--mmproj-smgs` (passthrough). Env-var path retained as compat.
- **§11.4** — deploy guard extends to check the **shared-infra
  symbol** (e.g. `ggml_mgpu_create_split` or equivalent), not
  just the clip-side log string. A binary built before B.1 lands
  must be refused.
- **§11.5 (NEW)** — CLIP perf gate (P7, HARD; see §11.5 below).

### 12.7 — Authenticity rationale (final, post-grep)

What "authentic to the codebase" means here, in final form:

- Same primitive (`ggml_backend_cuda_split_buffer_type`)
- Same algorithm (`create_split` with `mem_used` balancing)
- Same struct (`mgpu_split_config`, generalized from inline
  `llama_model` fields)
- Same CLI naming (`--split-mode graph`, `--tensor-split`,
  `-smf16` → `--mmproj-*` equivalents)
- Same gates re-certified post-refactor (G3.a, G3.c,
  test-n-stream-kv-layout, Phase 45 D10.a)
- Different consumer (`clip.cpp` instead of `llama_model`)

**One implementation. Two consumers. Five formal specs. Hard
perf gate. No phase split.** Maximum-rigor reading of the user's
2026-05-25 constraints.


