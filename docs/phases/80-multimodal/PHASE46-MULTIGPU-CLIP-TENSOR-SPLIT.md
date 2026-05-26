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

> **2026-05-26 maintenance-run finding (RED).** End-to-end empirical
> verification revealed two stacked structural gaps:
>
> **Run 1 (07:11 UTC, ~4 min downtime).** B.5b weights distribute but
> `alloc_compute_meta: graph splits = 1` for the 3739-node encode →
> entire compute graph single-devices → OOM on `cudaGraphInstantiate`
> at 1024 and 256 image-token budgets. Evidence:
> `/tmp/phase46-multigpu-clip/run-20260526T071133/server.stderr`.
>
> **Run 2 (07:47 UTC, ~25 min downtime, after B.5e partial: 0xff marker
> + fused-QKV exclusion landed in submodule `d8242a71`).** Partitioning
> now works: `graph splits = 55` (perfectly matches prediction of 27
> layers × 2 markers + 1). But encode hits `CUDA error: an illegal
> memory access was encountered` at `cudaStreamSynchronize` AFTER
> partitioning succeeds. The 0xff marker forces sched partition
> boundaries but the data dependencies between splits aren't honored
> without LM's complementary cross-device helpers (`do_split_norm`,
> `get_input_tensor_sm_graph`, explicit `GGML_OP_REDUCE`, per-device
> matmul decomposition `split_u`/`split_g`/`split_d`).
> Evidence: `/tmp/phase46-b5e-v3.stderr`.
>
> Phase 46 remains OPEN. New build is NOT deployed. Production is back
> on CPU-vision, `/health = 200` confirmed at 07:14 + 07:52 UTC.
> Strict GRAPH-mode approach per user ("we will not switch to layer").

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
    - [~] **B.5b — multi-device weight residency** (submodule `79f359d6`). Two-ctx pattern: `ctx_data_split` (multi-device, large matmul weights, pre-decorated with `clip_split_tensor.ggml` extras) + `ctx_data` (single-device, norms/biases/small tensors). Mirrors LM-side `llama_layer::split_*`. ~194 LoC + the new `clip_split_tensor` struct + 4 new clip_ctx fields.
          - [x] Build clean; loader fires "B.5b multi-device weight residency enabled (n_cuda=2)" + "B.5b split-buf allocated, 111 split tensors" on real boot (verified 2026-05-26 maintenance run).
          - [ ] **REOPENED 2026-05-26** — weight distribution alone is INSUFFICIENT. Real-host encode OOMs on `cudaGraphInstantiate` at device 0 because the ggml backend scheduler partitions the 3739-node CLIP graph into `graph splits = 1` (all on backend 0). Both backends register, P2 peer-access gate PASSes, 111 split tensors are placed, but the encode compute graph still single-devices. New subtask: B.5e graph-partitioning — make the input/activation buffer types multi-device-aware OR add explicit cross-device boundary markers so the sched produces > 1 split. Evidence: `/tmp/phase46-multigpu-clip/run-20260526T071133/server.stderr`.
    - [x] B.5c CLI flags: `--mmproj-devices`, `--mmproj-tensor-split`, `--mmproj-split-mode`, `--mmproj-smf16/smf32`, `--mmproj-smgs` (submodule `c648b624`); CLI→env-var bridge in `server-context.cpp` keeps clip.cpp's reader unchanged
    - [x] B.5d P1 f16 default ON — `mmproj_smf16 = true` in `common_params` (landed in B.5c)
    - [x] `test-clip-multi-backend-init.cpp` GREEN (9 cases PASS on dev host; submodule `ef7c41a4`)
    - [x] `test-clip-weight-split.cpp` GREEN (8 cases PASS on dev host; submodule `ef7c41a4`)
    - [~] `test-clip-encode-equivalence.cpp` — built, SKIPs cleanly until
          `scripts/verify-multigpu-clip.sh` produces
          `/tmp/phase46-multigpu-clip/equivalence.json` during maintenance window.
          Also: §3 acceptance #4 (byte-identity vs single-GPU baseline) is
          structurally unverifiable since single-GPU CLIP encode OOMs at
          the production residency — reframe the test to inter-run
          determinism (same image → same output bytes across N multi-GPU runs).
    - [~] **B.5e (NEW, 2026-05-26 partial, submodule `d8242a71`)** — graph
          partitioning across both backends. Two empirically-verified-correct
          improvements landed; both necessary, both insufficient on their own.
          - [x] `mark_split()` helper + calls in `build_attn` (clip.cpp:2727)
                and `build_ffn` (clip.cpp:2645). Sets the `0xff` op_params
                marker read by `ggml-backend.cpp:1727`. Mirrors LM at
                `llama-build-context.cpp:1261, 2444`. **Empirically lifted
                `alloc_compute_meta: graph splits` from 1 → 55** (27 layers
                × 2 markers + 1; matches the prediction in the approved
                plan). Verified `/tmp/phase46-b5e-256.stderr`.
          - [x] Fused-QKV exclusion in `is_splittable` (clip.cpp:3541+).
                qwen3vl packs QKV into one weight then view-3d slices at
                fixed offsets; row-chunking misaligns those views → IMA.
                Filter excludes any tensor name containing `qkv`.
                Empirically dropped 111 → 84 split tensors.
          - [ ] **STILL FAILING (2026-05-26 second maintenance run):** with
                the above two changes, multi-GPU CLIP encode at both 1024
                and 256 image tokens hits `CUDA error: an illegal memory
                access was encountered` at `cudaStreamSynchronize`
                (`ggml-cuda.cu:4499`), AFTER the 55-split partition is
                established. Production restarted; new binary NOT deployed.
                Evidence: `/tmp/phase46-b5e-v3.stderr`.
          - [ ] **Remaining work — port LM's cross-device graph
                infrastructure.** The 0xff marker forces sched partition
                boundaries; without the LM's complementary helpers the
                data dependencies between splits aren't honored, yielding
                IMA. Required helpers (search llama-build-context.cpp):
                `do_split_norm` (per-device norm of split input),
                `get_input_tensor_sm_graph` (cross-device input fetch),
                explicit `GGML_OP_REDUCE` insertions to consolidate
                row-chunked matmul outputs, per-device matmul decomposition
                (LM's `split_u` / `split_g` / `split_d` convention at
                `llama-build-context.cpp:1252-1268`). This is multi-day
                work touching `clip_graph` infrastructure. Strict GRAPH-mode
                approach per user 2026-05-26 ("we will not switch to layer").
          - [ ] **Why a small CLIP-side patch cannot close this
                (code-level confirmation, 2026-05-26).** The ik fork's
                `ggml/src/ggml-cuda.cu:2126-2127` hardcodes
                `dev[id].row_low = 0; dev[id].row_high = ne01` in
                `ggml_cuda_op_mul_mat`'s per-device loop — there is no
                code path that consults a split partition. The probe
                `ggml_backend_buffer_is_cuda_split` exists at line 998
                but the line-1000 comment confirms it is "only used in
                debug builds currently" — the ik fork stripped upstream
                llama.cpp's split-buft-aware mul_mat dispatch. Therefore
                the LM's explicit per-device decomposition pattern is
                the **only** mechanism in this fork that makes a
                row-chunked weight work; CLIP must mirror it. A
                hypothetical alternative — reintroducing upstream's
                transparent split-buft mul_mat path in ggml-cuda.cu —
                is also multi-day work and would diverge from the LM's
                established approach.
          - [x] **Diagnosis pinned (2026-05-26 Channel A capture, submodule
                pending).** Instrumented `clip_ctx` with an env-gated
                `CLIP_DEBUG_SCHED` per-node eval callback that forces
                `ggml_backend_synchronize` after each sched node, so a
                CUDA IMA surfaces at the offending node rather than at
                the global sync fence. Maintenance run captured the
                following at `/tmp/phase46-b5e-debug/run-20260526T101234/`:

                ```
                [CLIP_DBG NODE    21 op=RESHAPE  name=v.position_embd.weight (reshaped)
                                  ne=[1152, 48, 48, 1]  src0=v.position_embd.weight
                                  src0_op=NONE  src0_buft=CUDA_Split]
                [CLIP_DBG OK      21]
                [CLIP_DBG NODE    22 op=PERMUTE  name=v.position_embd.weight (reshaped) (permuted)
                                  ne=[48, 48, 1152, 1]  src0_op=RESHAPE  src0_buft=CUDA_Split]
                [CLIP_DBG OK      22]
                [CLIP_DBG NODE    23 op=UPSCALE  name=node_23
                                  ne=[36, 26, 1152, 1]  src0_op=PERMUTE  src0_buft=CUDA_Split]
                CUDA error: an illegal memory access was encountered
                  in function ggml_backend_cuda_synchronize at ggml-cuda.cu:4499
                ```

                **Offending op: `GGML_OP_UPSCALE` (node 23), src0 chain
                `v.position_embd.weight → RESHAPE → PERMUTE → UPSCALE`,
                src0 storage `CUDA_Split` (row-chunked across both
                GPUs).** This is the qwen3vl encoder's positional-embed
                upscale to match the actual image-grid (the 48×48
                positional grid is upscaled to 36×26 for this image
                budget). It happens at node 23 of the encode — far
                BEFORE any layer loop.

                **The prior multi-day "per-device matmul decomposition
                port" diagnosis was wrong.** The IMA is not in any
                MUL_MAT — it is in an UPSCALE of a row-chunked
                positional embedding that B.5b should never have
                row-chunked in the first place.

                **Implied fix shape — small, targeted patch to
                `is_splittable` at clip.cpp:3541+.** Extend the
                existing `qkv` exclusion to also exclude positional
                embeddings (and likely any small embedding-like
                tensor that flows into non-matmul ops such as
                UPSCALE / INTERPOLATE / IM2COL). Pattern:

                ```cpp
                // PHASE 46 B.5e — exclude positional embeddings.
                // They are reshaped/permuted/upscaled before use; the
                // CUDA UPSCALE kernel reads src0->data contiguously
                // and IMAs on row-chunked storage. Empirically caught
                // at clip.cpp encode node 23 on 2026-05-26.
                if (strstr(n, "position_embd") != nullptr) {
                    return false;
                }
                ```

                Position embeddings are small (~5–10 MiB at f16/f32);
                duplicating them on each device is essentially free
                (single-digit MiB of redundant residency).

                Whether this exclusion alone closes B.5e or whether
                additional small exclusions are needed is unknown until
                the next debug capture under the patch.

                Evidence: `/tmp/phase46-b5e-debug/run-20260526T101234/server.stderr`
                (3.7 MB; full CLIP_DBG NODE trace through node 23 + IMA).

          - [x] **Channel A re-run with `position_embd` exclusion landed
                (2026-05-26 run-20260526T103906).** Encode now advances
                past nodes 21-23 (positional grid OK). New IMA fires at
                **node 50, op=MUL_MAT, src0=`v.blk.0.attn_out.weight`,
                src0_buft=CUDA_Split**. This IS a matmul on a row-chunked
                weight — the case the prior pattern-matching diagnosis
                identified. The position_embd UPSCALE was a non-matmul
                anomaly that fired *before* the first matmul; with it
                excluded, the first matmul fires the IMA the prior
                speculation predicted.

          - [x] **Static audit of remaining 83 split tensors
                (2026-05-26, after position_embd exclusion).** Every
                CUDA_Split-bound tensor in qwen3vl (and every other
                clip_graph architecture) is consumed exclusively by
                `ggml_mul_mat` in shared helpers:
                - `layer.o_w` (attn_out) → `build_attn` mul_mat
                - `layer.ff_up_w`, `ff_gate_w`, `ff_down_w` → `build_ffn`
                  mul_mats
                - `model.mm_{0,1}_w` (merger) → direct mul_mat in
                  `build_qwen3vl`
                - `layer.deepstack_fc{1,2}_w` → `build_ffn` mul_mat
                  (deepstack branch)

                `ggml_cuda_op_mul_mat` cannot read row-chunked src0
                (`ggml-cuda.cu:2126-2127` hardcodes `row_low=0,
                row_high=ne01`). **Therefore: no further small
                `is_splittable` exclusions exist that would close B.5e.**
                The empirical iteration would just hit the next
                attn_out / ffn_up matmul. The static audit confirms
                this without further maintenance windows.

          - [~] **libmgpu architectural validation (2026-05-26).**
                User-directed pivot: extracted the Megatron-TP
                graph builders into a new `libmgpu` library (sibling
                of `ggml/`, `src/`, `examples/mtmd/`), then ported
                CLIP's loader, `build_ffn`, and `build_qwen3vl`
                attention block to use it. Architecture details in
                the approved plan file
                `/home/dconnolly/.claude/plans/examine-how-we-do-serene-liskov.md`
                under "libmgpu — folder structure and relationships".

                Submodule commits (production/2026-q2-next):
                - `7a213294` — Phases 0-2: libmgpu skeleton, CMake
                  wiring, classifier (mgpu_classify_weight), helpers
                  (mgpu_get_input_split, mgpu_norm_split).
                - `8aa3aa21` — Phase 3: mgpu_build_ffn_megatron with
                  fused-fast-path + biased-general-path.
                - `fe3cb270` — Phase 4: mgpu_build_attn_megatron_fused_qkv
                  with M-RoPE support; classifier refined (qkv →
                  REPLICATE because col-parallel split crosses Q/K/V
                  concat boundaries).
                - `242af825` — Phases 5-7: CLIP loader uses
                  mgpu_classify_weight; build_ffn calls
                  mgpu_build_ffn_megatron; build_qwen3vl attention
                  calls mgpu_build_attn_megatron_fused_qkv.
                - `473cdbc4` — three bug fixes from maintenance
                  runs: RoPE bit-mask dispatch (VISION=24 has the
                  MROPE bit), per-device bias slicing for COL-
                  parallel matmuls, row-parallel bias add moved
                  AFTER reduce, classifier scoped to REPLICATE
                  only attn_qkv (everything else NONE — REPLICATE
                  with non-libmgpu consumers IMAs).

                **Empirical state after Phase 7:**
                - **PASS:** Multi-GPU CLIP encode COMPLETES at
                  --image-min-tokens 256 (single run; 1713 nodes
                  through libmgpu); non-empty assistant response.
                  Evidence: /tmp/phase46-b5e-debug/run-20260526T133955/.
                - **PASS:** Multi-GPU CLIP encode COMPLETES at
                  --image-min-tokens 1024 (production budget, 3
                  consecutive samples). Median request time ~14.0s
                  (prompt processing ~9.05s for 1058 tokens =
                  117 t/s). Evidence:
                  /tmp/phase46-b5e-debug/run-20260526T135500/.
                - **FAIL: Determinism — output non-reproducible
                  across runs at temp=0, seed=42.** Three consecutive
                  samples at 1024 tokens produced THREE different
                  image descriptions for the same JPEG. The model
                  generates plausible text from each garbage
                  embedding without knowing the embedding differs.
                  Either ggml_reduce sum-order varies across runs
                  (most likely), or cuBLAS algo selection varies
                  per per-device matmul, or FA non-determinism
                  surfaces under the per-device dispatch. The LM's
                  NPC.4 ALGO0 cuBLAS algo pin is matmul-shape-
                  specific and may not apply to CLIP's shapes.

                **B.5e is NOT CLOSED** despite Phase 7 wire-up.
                Determinism is a hard gate (§3 acceptance #4
                reframed: "same image → same output bytes across N
                multi-GPU runs"). Output non-determinism = encode is
                effectively producing random garbage that the LM
                rationalizes; deploying this would make Qwen 3.6
                vision unpredictable in production.

          - [ ] **Remaining work for B.5e CLOSURE (after Phase 7):**

                **NPC.B5e.3 LOCALIZED (2026-05-26):** Capture-bisect
                via extended `clip_debug_eval_cb` (submodule
                `e850fe0a`, FNV-1a per-node hash stream gated by
                `CLIP_CAPTURE_HASH`) proves the libmgpu graph is
                **bit-deterministic** when each node has a
                `ggml_backend_tensor_get` invoked after it.

                Maintenance window 2026-05-26 14:38-14:48: two
                encodes of `test-1.jpeg` at 1024-token budget under
                `CLIP_DEBUG_SCHED=1 CLIP_CAPTURE_HASH=...` produced
                bit-identical 1714/1714 hash streams + identical
                text output ("brown pillow"). Evidence:
                `/tmp/phase46-b5e-capture/20260526T143751/`.

                **NPC.B5e.4 SIX-TEST DIAGNOSTIC (2026-05-26
                15:08-16:18):** added `CLIP_LOG_FINAL_HASH` and
                ran a series of env-gated structural-sync variants
                in production mode (no per-node sync). Evidence:
                `/tmp/phase46-b5e-tests/20260526T152848/`.

                | Test | Sync                                     | Result |
                |------|------------------------------------------|--------|
                | C    | production async (no extra sync)         | FAIL   |
                | B    | per-split stream sync (split's backend)  | FAIL   |
                | E    | per-node stream sync (callback path)     | FAIL   |
                | F    | full-device sync after every reduce      | FAIL   |
                | G    | full-device sync after cpy_tensor_async  | FAIL   |
                | H    | full-device sync per node (replaces E)   | FAIL   |
                | I    | per-thread stream sync per node          | IMA on LM init |
                | Cap  | per-node tensor_get (pageable DtoH)      | PASS   |

                All targeted-sync variants fail. Only the
                comprehensive per-node `ggml_backend_tensor_get`
                pattern produces determinism.

                **Key discovery: per-backend streams are created
                with `cudaStreamNonBlocking`**
                (`ggml/src/ggml-cuda/common.cuh:906`). This flag
                disables implicit synchronization with default
                streams. So Test I (sync via cudaStreamPerThread)
                does NOT sync the named non-blocking streams →
                LM init IMA confirms.

                **Open question:** what specifically does
                `ggml_backend_cuda_buffer_get_tensor` (which uses
                `cudaMemcpyAsync` to pageable host + cudaStream
                Synchronize on cudaStreamPerThread) do that all
                the targeted syncs don't? Hypotheses for next
                session:
                - The DtoH memcpy reads the destination side's
                  per-device storage from each device through
                  peer access; the read itself may force a
                  device-fence that cudaDeviceSynchronize lacks
                  on this hw (Quadro RTX 6000, CUDA 13.2).
                - There may be a kernel-level race in a specific
                  op (FA, mul_mat, soft_max) that ONLY high-
                  frequency per-tensor reads happen to mask via
                  read-side serialization.
                - The capture's tensor_get is invoked per F32/F16
                  /BF16 node (~1714 nodes); maybe the determinism
                  requires sync at SPECIFIC nodes outside the
                  reduce/cpy boundaries.

                Diagnostic env knobs left in source (all OFF by
                default, no production impact):
                - `CLIP_LOG_FINAL_HASH=1`
                - `GGML_SCHED_PER_SPLIT_SYNC=1`
                - `GGML_REDUCE_POST_DEVICE_SYNC=1`
                - `GGML_CPY_POST_DEVICE_SYNC=1`
                - `GGML_CUDA_FULL_DEVICE_SYNC=1`
                - `GGML_CUDA_PER_THREAD_SYNC=1` (causes IMA — do not set)

                **DISCOVERY (2026-05-26 ~16:25):** The entire NPC.4
                six-test diagnostic round was conducted with
                **UNLOCKED GPU clocks**. At test time, SM clocks
                were 300 MHz (idle ramping under load). The
                LM-blessed production determinism harness
                (`scripts/verify-production-determinism.sh`)
                explicitly REQUIRES locked clocks at 1455 MHz and
                fails the pre-check otherwise:

                ```
                FAIL: GPU ${idx} SM clock is ${cur_sm} MHz,
                      expected 1455 MHz.
                Run `sudo bash scripts/gpu-clocks.sh lock` to
                lock clocks at 1455 MHz.
                ```

                Per `scripts/gpu-clocks.sh:7-9`:
                > Unlocked clocks let SM frequency vary with
                > thermal/power state, which makes concurrent
                > multi-slot timing non-deterministic.

                So the LM determinism contract holds under locked
                clocks, NOT under unlocked clocks. Our six-test
                CLIP investigation under unlocked clocks could
                ALL be operational variance rather than a code-
                level race.

                The simplest hypothesis was: **CLIP encode is
                deterministic under locked clocks, same as LM**.

                **Test J (2026-05-26 16:30, NEGATIVE):** ran
                `sudo bash scripts/gpu-clocks.sh lock` (confirmed
                1455 MHz on both GPUs), then 3 samples in pure
                production async mode (no extra env knobs except
                `CLIP_LOG_FINAL_HASH=1`). Result:
                - J1: hash `e344298495d432ff` ("brown pillow")
                - J2: hash `f71fc3b1672bb0f3` ("person, woman")
                - J3: hash `75211050486650d1` (different)

                **Three different hashes. Clock-lock alone is
                NOT sufficient for CLIP determinism.** Evidence:
                `/tmp/phase46-b5e-tests-J/20260526T163051/`.

                The CLIP non-determinism is therefore NOT pure
                operational variance — there IS a code-level race
                that's independent of timing.

                **Test K (2026-05-26 16:40, PASS):** modified
                capture to use `cudaMallocHost` pinned host memory.
                K1 + K2 produced bit-identical hash streams (final
                hash `2554e340101807ab` both). Pinned vs pageable
                does NOT distinguish — the READ itself is the
                fence.

                **Test L (2026-05-26 16:47-16:55, LOCALIZED):**
                selective op-class skip in capture mode:
                - `CLIP_CAPTURE_SKIP_OPS=MUL_MAT` (L_NOMM): final
                  hash `2554e340101807ab` → MATCHES baseline.
                  MUL_MAT-output reads are NOT needed.
                - `CLIP_CAPTURE_SKIP_OPS=REDUCE` (L_NORE): final
                  hash `8ab8037be27b05c3` → DIFFERS. **REDUCE-
                  output reads ARE the load-bearing fence.**

                **ROOT CAUSE:** REDUCE output tensors need a host
                readback to enforce peer-access memory consistency.
                The reduce kernels (NCCL or ring) issue cross-
                device peer-access writes; subsequent reads see
                those writes only if a peer-access fence is
                enforced. `cudaDeviceSynchronize` drains compute
                streams but does NOT enforce peer-access fence for
                writes from OTHER devices. `cudaMemcpyAsync` DtoH
                from device memory DOES enforce it (the DMA engine
                needs consistent memory state).

                **Structural fix candidate (Test M for next
                session):** insert a tiny ~4-byte `cudaMemcpyAsync`
                DtoH readback after each reduce code path in
                `ggml_cuda_op_reduce` (gated behind a default-on
                knob). The tiny size keeps the cost negligible;
                the readback acts as the peer-access fence.
                Estimated 10-15 LoC.

                If Test M restores determinism in production async
                mode, **B.5e closes** with a small surgical fix
                that does NOT require:
                - Eval-callback path (avoids ~3× encode latency)
                - libmgpu restructure (avoids large refactor)
                - Operational-only clock lock (Test J disproved
                  sufficiency, but it's still good hygiene)

                Evidence: `/tmp/phase46-b5e-tests-KL/20260526T163912/`.

                Production state after Test L: restored on CPU-
                vision, /health=200. Clocks remain locked at
                1455 MHz from Test J.

                1. Next session: bisect WITHIN the capture's
                   tensor_get behavior — separate the cudaStream
                   Sync call from the cudaMemcpyAsync. Try sync
                   without memcpy (already tested as H = fails).
                   Try memcpy without sync (would test if the
                   readback itself causes a fence). Or instrument
                   to print exact stream contents.
                2. Or pursue option 1a from the prior plan: force
                   eval-callback path with no capture (matches
                   Test E semantically; fails). Pursue option 1d
                   instead: restructure libmgpu reduce to gather
                   on one device. Large refactor.
                3. Re-run 3-sample test; verify sha256 match.
                4. Then close B.5e [x].

                Performance reframe note (pre-determinism): one
                1024-token sample is ~14.0s wall vs ~10+s CPU
                baseline. Per-device-decomp overhead is real and
                non-trivial at this scale; the original "30-50×
                speedup vs CPU" framing in the plan was
                overoptimistic.
          HARD prerequisite for B.7.
  - [ ] **B.6** — LM gate re-cert (HARD; deferred to maintenance window — production service uses both GPUs at capacity, test binary cannot allocate concurrent VRAM)
    - [ ] G3.a `test-production-np-determinism.sh` PASS NP∈{1,2,4,8}
    - [ ] G3.c `r5-probe-c4.sh` 0/20 divergences
    - [ ] `test-n-stream-kv-layout` PASS
    - [ ] Phase 45 D10.a 3-slot smoke PASS
    - **Note:** B.1-B.4 are semantics-preserving by construction (cfg mirrors model fields; reads through cfg are equivalent to reads through model fields). Empirical bit-identity verification remains as the binding closure step.
  - [ ] **B.7** — CLIP perf gate (HARD, P7).
    - **Gate reframe (2026-05-26):** §11.1 single-GPU baseline is **structurally unobtainable** — even a single-GPU CLIP encode at the production LM+KV+scratch residency OOMs on `cudaGraphInstantiate`. There is no single-GPU latency reading to multiply by 1.3×. Once B.5e closes the graph-partitioning gap, B.7's gate is reframed to: encode completes successfully under the production LM+KV+scratch config at the target image-token budget, no OOM, median latency recorded as the new reference number (no ratio).
    - [ ] Empirical: maintenance run 2026-05-26 OOM'd on `cudaGraphInstantiate` at 1024 AND 256 image tokens. Phase 46 cannot deploy until B.5e graph-partitioning lands. `scripts/verify-multigpu-clip.sh` (commit `b347398`) is the binding driver.
    - [~] `test-clip-encode-latency.cpp` — built, SKIPs cleanly until harness produces input JSON
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

- [~] §11.1 — STRUCTURALLY UNOBTAINABLE (user confirmation + empirical
      check 2026-05-26): even single-GPU CLIP at the production LM+KV
      residency OOMs on `cudaGraphInstantiate`. There is no
      "single-GPU baseline" number to capture; B.7 reframed accordingly
      (see §10 B.7 gate reframe note).
- [x] §11.2 — full CLI flag family implemented (landed in B.5c, submodule `c648b624`)
  - [x] `--mmproj-devices`, `--mmproj-tensor-split`,
        `--mmproj-split-mode graph`, `--mmproj-smf16/--mmproj-smf32`,
        `--mmproj-smgs` all in `common/common.cpp` arg handlers + `--help`
  - [x] Env-var fallback works via CLI→env-var bridge in
        `examples/server/server-context.cpp:159+`
- [~] §11.3 — `scripts/verify-multigpu-clip.sh` (commit `b347398`) emits
      `evict_pressure` event count during the harness run; promotion to
      PHASE35 §15.7 closure is conditional on count ≥ 1 in real run
- [x] §11.4 — Deploy-script regression guard (commit `149ad76`): checks
      `multi-backend init` string in libmtmd.so AND
      `ggml_mgpu_create_split` symbol in libggml.so; `--allow-no-mmproj-mgpu`
      opt-out for emergency rollback. Verified both positive case (current
      build passes) AND negative case (synthetic Phase-46-stripped build:
      guard ABORT with exit 1, expected message printed; 2026-05-26).
- [~] §11.5 — `test-clip-encode-latency.cpp` (submodule `ef7c41a4`) built,
      SKIPs cleanly until §11.1 baseline + harness run produce its input

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


