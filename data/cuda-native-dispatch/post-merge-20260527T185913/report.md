# PHASE_CUDA_NATIVE_DISPATCH — C14 verification

**RUN_ID:** 20260527T185913
**Date:** 2026-05-27
**Host:** xeon (2× Quadro RTX 6000, NV2 NVLink)
**Submodule HEAD:** `4465a7d1` (ik_llama.cpp `production/2026-q2-next`)
**Parent HEAD:** `e97d965` (yarn-agentic `main`)

## Arc commits

| # | Commit | Submodule | Parent | Status |
|---|---|---|---|---|
| C0 | Calibration framework | `148d5ac5` | `8ce192c` | ✅ shipped |
| C1 | Single-threaded `compute_splits` + event chain | `31b1ccd7` | `f19390e` | ✅ shipped |
| C2 | Eager pre-allocation of CUDA-context lazy fields | `127d0dd6` | `e6c5c4e` | ✅ shipped |
| C3 | Per-backend `graph_compute` in-capture gate | `99c662ed` | `6bbf37d` | ✅ shipped |
| C4 | Outer `cudaStreamBeginCapture(Relaxed)` wraps dispatch | `fcfc034f` | `e2c526f` | ✅ shipped |
| C5 | Multi-device graph cache | `62060434` | `f57f72e` | ✅ shipped |
| C6 | CPU split hoist-out | `7d1dd368` | `5a58922` | ✅ shipped |
| C7 | libmgpu port to captured subgraphs (no source change) | `7242a803` | `ccd6841` | ✅ shipped |
| C8-C11 | Register four calibrated ops | `9ed48cbd` | `bfae445` | ✅ shipped |
| C12 | Delete obsolete env knobs + dead state | `4465a7d1` | `e97d965` | ✅ shipped |
| C13 | (absorbed into C1 + C12) | — | — | folded |
| C14 | Verification commit (this report) | — | (next) | in progress |

## Unit test sweep

All 7 binding unit tests PASS on the build-tree binary:

- `test-single-threaded-dispatch`: `dispatch_thread_count == 1` over 100 iters (C1 + C2 invariants)
- `test-multi-device-graph-capture`: 20 captured round-trips, hash `0847900abf072325 × 20` (C4)
- `test-multi-device-graph-cache`: 50 cached replays, hash `0847900abf072325 × 50` (C5)
- `test-libmgpu-subgraph-capture`: 30 iters, hash `040a6e2a9adb6325 × 30` (C7)
- `test-cuda-calibration-framework`: T4-T9 PASS (C0)
- `test-calibration-ops-registered`: T1-T3 PASS (C8-C11; all 4 ops calibrate to SIZE_MAX on xeon)
- `test-paged-allocator-determinism`: PASS (regression smoke)

Raw output: `unit-test-sweep.txt`.

## Specs LIVE

- `calibrated_dispatch_framework.allium` (C0)
- `single_threaded_dispatch.allium` (C1+C2+C3+C6) — 11 contracts
- `cross_device_event_chain.allium` (C4) — 6 contracts
- `multi_device_graph_cache.allium` (C5) — 6 contracts
- `libmgpu_subgraph_capture.allium` (C7) — 4 contracts
- `calibrated_op_equivalence.allium` (C8-C11) — 4 contracts
- TLA+: `CalibrationFramework`, `CUDANativeDispatch`, `CUDAGraphCacheConsistency`, `CalibratedOpEquivalence`

## Production-deploy gate status

**NOT YET DEPLOYED to `/opt/llm-server/`.**

Production currently runs the Phase-46 closure build (`1db6c2eb`). The C14 deploy requires:

1. Maintenance window (production stopped, ~25 min):
   - `sudo systemctl stop llama-server.service`
   - Run `scripts/test-production-np-determinism.sh` with build-tree binary (NP={1,2,4,8} × 2 reps for NP=8 flake histogram)
   - Run `scripts/r5-probe-c4.sh` (G3.c single-GPU NP=2 ×20 iters)
   - Run `scripts/verify-multigpu-clip.sh LATENCY_N=10` — assert all 10 CLIP encodes byte-identical AND median latency ≤ 14450 ms (Phase-46 baseline 14421)
   - Run `./build/bin/llama-server --help` sanity for the embedded `commit=` stamp
2. Deploy via `scripts/deploy-llama-server.sh` (the canonical install path).
3. Restart service, poll `/health` to 200.
4. Vision smoke against live `:8085`.

**G3.a expectation under C1-C12:**
- NP={1,2,4}: byte-identical to NP=1 baseline (C1 dispatch is deterministic by construction; PD1 racing fields gone after C2; capture+replay deterministic).
- NP=8 single-slot flake: the localized race surface (host-side CUDA driver state under openmp parallel dispatch) is removed by C1. The flake **should** be gone. Confirmation requires the live test.

**A/B controls:**
- Against `GGML_SCHED_EVAL_SERIALIZE=1`: this knob is deleted at C12; the A/B is no longer runnable. Equivalent property: the C1 dispatch is structurally serialized (`HostThreadIsExactlyOne`), so the eval-serialize and dispatch paths are now the same code path.
- Against Phase-46 closure (`1db6c2eb` at `/opt/llm-server/`): full CLIP byte-identity to sha `fb5167dbc1e7f95b` is the C7 contract `LibmgpuCachedReplayMatchesPhase46Closure`. Requires the maintenance window to verify.

**Calibration cache sanity:**
- Cache file: `$XDG_CACHE_HOME/ggml/cuda-calibration-{fnv1a64}.json`
- Cache key on xeon: `v1-cuda130-drvNNN-gpus:<uuid>...` (hashed to a 16-hex filename)
- Cache contains all 4 op thresholds at SIZE_MAX (default-wins stubs).
- `FORCE_RECALIBRATE=1` bypass works (T9 PASS).

## Performance plan (deferred to live deploy)

PD4 baseline (RUN_ID=20260527T121951, recorded at C0 prep):
- LM TG NP=1: 17.9 t/s
- LM TG NP=8 aggregate: 31.4 t/s
- CLIP encode median: 14421 ms

Post-merge expectations (PHASE_CUDA_NATIVE_DISPATCH.md §4.3):
- LM TG NP=1: 17.9 t/s ± 5% (no change — single-stream dispatch unchanged)
- LM TG NP=8: conservative 46 t/s, stretch 60, aspirational 92 (vLLM ceiling 154.77). Improvement comes from the captured cudaGraph reducing per-iter host-dispatch overhead.
- CLIP encode median: ≤ 14450 ms (within Phase-46 baseline; with C5 cache the first encode pays Instantiate cost but subsequent encodes only pay GraphLaunch).

**These targets are not bound by C14's unit tests.** Live measurements happen at deploy time.

## Rollback path

If C14 deploy regresses production:
1. `sudo systemctl stop llama-server.service`
2. Restore previous binary: `sudo install -m 0755 -o root -g llm /home/dconnolly/yarn-agentic/ik_llama.cpp/build-rollback/bin/llama-server /opt/llm-server/bin/llama-server` (where `build-rollback/` is a separate checkout of `1db6c2eb` rebuilt before the deploy attempt).
3. `sudo systemctl start llama-server.service`.
4. Phase-46 closure `1db6c2eb` is the known-good rollback target. Per `project_phase46_closed.md`, the rollback drill at Phase-46 closure verified this works via `--allow-no-mmproj-mgpu`.

## Closure criteria

- [x] All 14 C-commits land (C13 absorbed into C1+C12; otherwise 1:1)
- [x] All 6 LIVE Allium specs + 4 LIVE TLA+ modules in tree
- [x] All 7 unit tests PASS on build-tree binary
- [x] Submodule pushed (commits C0 through C12)
- [x] Parent repo pushed
- [ ] Production deploy: PENDING user-authorized maintenance window
- [ ] G3.a determinism battery on live service: PENDING (post-deploy)
- [ ] B.7 CLIP latency vs Phase-46 baseline: PENDING (post-deploy)
- [ ] 24-hour soak on NP=1: PENDING (post-deploy)

## Open follow-ups (not in this phase)

1. **NCCL re-enable** for REDUCE_CROSS_DEVICE: the typo at `ggml-cuda.cu:4449` (`#ifdef GGML_USE_NCCL__` → `GGML_USE_NCCL`) is unfixed. C9's stub probe yields SIZE_MAX so production runs memcpy-peer + add (the Phase-46 closure path); enabling NCCL would benefit large reduces but needs a real probe + careful capture-compatibility testing.
2. **Real probe implementations** for C8 / C10 / C11 if a future deployment needs the alt strategies. Framework infra is in place; only the probe function bodies change.
3. **Cache key extension** to `(topology, device_layout, n_seq)` per phase doc §3.1 C5. C5 ships with topology-only key; n_seq is implicit in n_splits and device_layout is implicit in per-split backend_id, but a deployment with key collisions can extend without ABI change.

---

The PHASE_CUDA_NATIVE_DISPATCH code arc is structurally complete at C12.
C14 (this commit) records the binding evidence. Production deploy is the
final step and is gated on a user-authorized maintenance window.
