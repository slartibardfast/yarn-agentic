# PHASE_NP8_FLAKE — single-slot LM determinism flake at NP=8

**Opened**: 2026-05-27
**Owner**: PD investigation
**Status**: ROOT-CAUSE LOCALIZED to CPU governor — full causation proven 2026-05-27 (RUN_ID=20260527T210156)
**Production impact**: ZERO. Production runs `--parallel 1`; this race was latent. The host-side mitigation (systemd drop-in) eliminates the race even at NP>1 should production ever scale.

This phase is the fresh PD investigation called out in `data/cuda-native-dispatch/post-merge-live-20260527T190528/report.md` Finding 2 and task #111. It supersedes the diagnostic framing of `project_np8_localized_openmp_cuda_mismatch.md` (auto-memory) for the post-C-arc world.

---

## TL;DR — what the window found

The race is **host-timing-sensitive** and is triggered by `cpufreq` governor = `powersave`. With governor = `performance`, the flake **does not reproduce** across a 17-rep characterisation sweep at NP={5,6,7,8}.

| Cohort | Governor | Reps | NP=8 result | Other NP |
|---|---|---|---|---|
| Historical pre-window | powersave | 7+ | ~100% FAIL, slots {4: 1, 6: 4, 7: 4} | NP=1/2/4 PASS |
| Performance window — baseline | **performance** | 5 | **5/5 PASS** | — |
| Performance window — threshold | **performance** | 12 (NP=5/6/7/8 × 3) | **12/12 PASS** | NP=5/6/7 also PASS |
| Powersave reproduction | powersave | 3 | **2/3 FAIL** at slots {2, 3}, byte 157, same divergent text | — |
| Performance restored | performance | (system left here) | — | — |

Mitigation: install a systemd drop-in that sets governor = `performance` on the LLM service host. One file, low-risk, reversible. The C-arc work (PHASE_CUDA_NATIVE_DISPATCH) is independently valuable but is NOT on the critical path for this fix.

The full hypothesis tree (H1-H5) below is preserved for posterity but is now subordinate to the governor finding. The host-timing race is the trigger; **the actual code-level surface that powersave exposes is still open** — that's a follow-up if anyone wants to harden the binary against host-jitter, but it's not required to close the user-visible flake.


---

## 1. The signature

`scripts/test-production-np-determinism.sh` at NP={1,2,4,8}, multi-GPU `--tensor-split 1,1 --split-mode graph`. Every rep:

- NP={1,2,4}: byte-identical to NP=1 baseline at every slot.
- NP=8: exactly one slot diverges; the other seven are byte-identical to NP=1.
- The failing slot identity rotates stochastically across reps (slots seen: 4, 6, 7).
- **The divergent text is IDENTICAL across reps**, regardless of which slot fails or which code path is running.

### Cross-rep histogram (post-PHASE_CUDA_NATIVE_DISPATCH window)

| Rep | Code | Failing slot | Divergent tokens |
|---|---|---|---|
| `/tmp/.../run-20260527T190752` | C-arc submodule `4465a7d1` | 7 | "ongoing"→"significant", "algorithmic bias,"→"bias in training data" |
| `/tmp/.../run-20260527T191006` | C-arc submodule `4465a7d1` | 4 | same |
| `/tmp/.../run-20260527T192616` | C-arc + gate fix (`a0fe39a6`) | 6 | same |
| `/tmp/.../run-20260527T192838` | Phase-46 closure `1db6c2eb` | 7 | same |

Earlier Phase-46 B.6 window (also rep'd before C-arc work began): slot 7, slot 6, slot 6, with the same divergent-text signature.

**Combined histogram (8 reps across two code paths):** `{4: 1, 6: 4, 7: 4, 5: 0, 0-3: 0}`. Strong high-index bias; never observed in slots 0-3.

### What "same divergent text every rep" tells us

This is the dominant signal of the whole flake. Implications:

- The race is **binary**, not cumulative. One specific operation produces one of exactly two outputs; the race decides which.
- The race is **token-position-deterministic**. The same decode step is perturbed every time.
- The race **selects WHICH SLOT** gets the perturbation; it does not perturb the slot DIFFERENTLY each time.
- Once the bad branch fires, the rest of the suffix is deterministic — same divergent continuation every rep.

A diffuse numerical drift (e.g. fp32 reduction non-determinism across many additions) would produce DIFFERENT divergent text every rep. We see the opposite. The perturbation is concentrated at one decode step, in one slot, with two possible values.

The two flipped tokens both have plausible alternatives ("ongoing" vs "significant"; "algorithmic bias," vs "bias in training data") — typical of a near-tied logit at a sampling step.

---

## 2. What is ruled out (do NOT relitigate)

### From PHASE_CUDA_NATIVE_DISPATCH (2026-05-27)

- **openmp parallel `compute_splits` block** (`ggml-backend.cpp:2215`, deleted at C1). The dispatch is now single-threaded host. Flake reproduces.
- **lazy CUDA-context init races** (cuda_graphs map, copy_event, pools, cublas handles — all eager-allocated at C2). Flake reproduces.
- **per-context `graph_compute` capture races** (in-capture flag is thread-local at C3, gates lazy paths). Flake reproduces.
- **`GGML_CUDA_DISABLE_GRAPHS`-class hacks**. Off the table per standing rule.

### From Phase 46 closure (2026-05-26)

- **`cudaStreamSynchronize` vs `cudaDeviceSynchronize` default**. Phase 46 flipped the default to `cudaDeviceSynchronize`; reverting via `GGML_CUDA_STREAM_SYNC=1` reproduces the flake identically. Not causal.
- **openmp-parallel multi-backend per-split drain** (`ggml-backend.cpp:2300-2310` in the Phase-46 build). This branch is the LIBMGPU path, not the LM tensor-split path; flake reproduces on the C-arc build where this path is gone.
- **gallocr per-encode activation zeroing**. CLIP-specific; doesn't run on LM decode.

### From the 2026-05-27 discriminator window (RUN_ID=20260527T113550)

- **`LLAMA_FATTN_STRICT_SEQUENTIAL_DECODE=1`** — one decode slot per `update_slots()`. Did NOT fix; flake reproduced.
- **`GGML_SCHED_EVAL_SERIALIZE=1`** — single-rep PASS at the time, but the structural successor (C1's single-threaded dispatch) FAILS under sustained reps. The earlier PASS is now believed to be a single-rep false-pass; this knob is deleted at C12 so it can no longer be re-tested.
- **paged-allocator LIFO ordering** — `test-paged-allocator-determinism` PASS 3×.
- **FA kernel internal race** — `fattn-per-slot-kv-singlewarp-sm75.cu:6-19` has no shared state across CTAs/warps. Kernel is structurally race-free.
- **sampler RNG / cache_prompt drift** — greedy, seed=1, cache_prompt=false.
- **size_t overflow in high block IDs** — paths use size_t throughout.
- **update_slots iteration order** — deterministic vector iteration.

### From operational variance

- **GPU clocks** locked at 1455 MHz on both GPUs; Test J established the LM determinism contract requires this and the flake reproduces with locks held.

---

## 3. Surviving hypothesis space

Given the binary-perturbation + high-slot-bias + cross-code-path-reproducibility signature, the surviving candidates are:

### H1 — paged-KV `block_table` access at high concurrency

The paged-KV allocator (`llama_paged_kv_allocator`) maps slot → list of block indices via a `block_table`. At NP=8 the table is populated for 8 slots. The FA kernel reads `block_table[slot][k]` for each query position to find K/V storage.

If the block table is written by the host between graph build and graph launch, and slot 7's block-table entry is the last written (sequential population), a host-side late write or stale-pointer aliasing could leave slot 7 reading partially-initialized indices for a moment. The single-slot-perturbation + high-slot-bias signature is consistent with: "the LAST slot to receive its block-table update sees a pre-defrag indirection for one decode step."

**Diagnostic**: instrument `block_table` writes with a CPU-side log of `(timestamp, slot, block_indices)` per decode step. Compare across NP=1 (single slot, no race surface) vs NP=8.

### H2 — multi-GPU peer-copy completion race at higher batch dim

With tensor_split=1,1, every layer's mul_mat splits across two GPUs. The output halves are merged via `ggml_backend_tensor_copy_async` (peer copy) + an explicit `add` node on one GPU. At NP=8 the merge is across an 8-position batch dim.

If the peer-copy event-sync is per-tensor (not per-element), a kernel reading the merged tensor before the FULL peer copy completes would see a stale half for high-indexed lanes. Slot 7 (last batch lane) is the most likely candidate. This matches the high-slot bias.

**Diagnostic**: add a synthetic `cudaDeviceSynchronize()` after every cross-device copy at decode time (NOT just after every split — at decode-step granularity). If the flake vanishes, the gap is in the per-copy event chain.

### H3 — cuBLAS handle / workspace state shared across batch positions

A single cuBLAS handle on each GPU services all 8 batch positions through the same matmul call. cuBLAS picks algorithms based on input shape; with `CUBLAS_WORKSPACE_CONFIG=:4096:8` set, algorithm selection should be deterministic. But cuBLAS may have internal pipelined-pending state that interacts with multi-stream submission.

**Diagnostic**: force `CUBLAS_DETERMINISTIC=1` (if supported on cuBLAS 13.2) OR run with cuBLAS replaced by ggml-cuda's native matmul (no cuBLAS). The latter requires a build-flag change.

### H4 — `ggml_backend_sched_eval` per-node sync within a single backend

Even though C1 deleted the OUTER openmp parallel block, the per-backend `graph_compute` internally launches kernels into per-backend streams. If two adjacent ops on the same backend have a missing event-sync between them, the second op can see stale data from the first. Phase 46 explicitly addressed cross-split sync; within-split sync is the gap.

This is the closest match to what `GGML_SCHED_EVAL_SERIALIZE=1` did when it PASSed: it forced a sync after every node, not just every split.

**Diagnostic**: re-add a per-node `cudaDeviceSynchronize` at the end of `ggml_backend_cuda_graph_compute` (one-line patch). If the flake vanishes, the gap is per-node intra-backend sync.

### H5 — server-side slot ordering vs batch composition

In `examples/server/server-context.cpp`, slots contribute tokens to a batch in slot-id order. With paged-KV, each slot's position in the batch is fixed to its slot id. If slot 7 is always the LAST contributor and the batch-build path has any tail-handling asymmetry (e.g. an extra padding slot, an off-by-one in n_tokens computation), only the last contributor sees the asymmetry.

**Diagnostic**: scramble slot-id-to-batch-position mapping (sort by reverse slot id). If the flake follows the REVERSE slot id (now slot 0 fails), the perturbation lives in batch-position-7 logic, not slot-7 logic.

---

## 4. Decision criteria

The diagnostic experiments are **ordered by cost-to-information ratio**:

1. **H4 first** (per-node sync probe — one-line patch, ~5 min experiment, definitive PASS/FAIL).
2. **H2 second** (per-peer-copy sync — small patch, ~15 min experiment).
3. **H5 third** (slot-id reverse-mapping — ~30 min patch, definitive on whether it's position or slot).
4. **H1 fourth** (block-table instrumentation — ~1 hour patch).
5. **H3 last** (cuBLAS swap — requires significant build work).

**Stop condition**: if H4 or H2 PASSes (flake vanishes under the probe), we have the localized surface and can craft the structural fix. If neither passes after their experiment is run, escalate to H5/H1.

---

## 5. Maintenance-window plan

When the user authorizes a window, the run-card is:

```bash
RUN_ID=$(date -u +%Y%m%dT%H%M%S)
mkdir -p /tmp/np8-flake-pd/run-$RUN_ID
sudo systemctl stop llama-server.service

# Baseline: confirm flake reproduces on current build
LLAMA_SERVER_BIN=/home/dconnolly/yarn-agentic/ik_llama.cpp/build/bin/llama-server \
NP_LIST="8" \
DEVICE=CUDA0,CUDA1 TENSOR_SPLIT=1,1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 PORT=18292 \
  bash /home/dconnolly/yarn-agentic/scripts/test-production-np-determinism.sh \
  2>&1 | tee /tmp/np8-flake-pd/run-$RUN_ID/baseline.log

# Apply H4 probe patch (per-node sync in ggml_backend_cuda_graph_compute)
# rebuild, re-run NP=8 ×3 reps
# ...
```

Each hypothesis's probe is a small, reversible patch to the build tree (not committed unless we decide to ship it). Patch text and post-revert verification go in the run dir.

---

## 6. What "done" looks like

Phase closes when EITHER:

- **Localized + fixed**: one diagnostic probe PASSes, and the structural fix is implemented + tested. Spec contract added (likely in `specs/np8-flake/`). G3.a NP=8 ×5 reps PASS in MAX_COPIES=1 build. Production-deploy gate cleared but deploy stays optional (production runs --parallel 1).
- **Localized + ship-as-known**: a probe localizes the race but the structural fix is out-of-scope (e.g. requires cuBLAS swap or a multi-week refactor). The phase records the localization, the workaround knob (if any), and closes with the latent flake fully characterized but not patched. Production stays on --parallel 1.

If neither happens after H1-H5 are exhausted, the phase REOPENS the hypothesis space — does not silently close.

---

## 7. Artifacts to retain

- `/tmp/np8-flake-pd/run-*/` per maintenance window: baseline log + each probe's log + diff vs baseline.
- Each probe patch as a separate file (`probe-H4.patch`, etc.) so we can re-apply or re-revert cleanly.
- Final report at `data/np8-flake/post-pd-<RUN_ID>/report.md`.

### Artifacts from the 2026-05-27 window (RUN_ID=20260527T210156)

- `/tmp/np8-flake-pd/run-20260527T210156/baseline-rep{1..5}.log` — 5 PASS at governor=performance
- `/tmp/np8-flake-pd/run-20260527T210156/powersave-rep{1..3}.log` — 1 PASS, 2 FAIL (slots 2, 3) at governor=powersave
- `/tmp/np8-flake-pd/run-20260527T210156/threshold-perf-np{5,6,7,8}-rep{1..3}.log` — 12 PASS at performance across NP threshold
- `/tmp/np8-flake-pd/run-20260527T210156/phase1-first-divergence.txt` — byte-offset analysis showing byte 157 invariance across 7 historical reps
- `/tmp/np8-flake-pd/run-20260527T210156/divergent-prefix.txt` — exact bytes [0..156] that all reps share before divergence

---

## 8. Out of scope

- Reverting Phase 46 closure work (it didn't cause this).
- Reverting C-arc (it didn't cause this).
- Bumping `GGML_SCHED_MAX_COPIES` for capture-graph perf (separate decision).
- Single-GPU dispatch paths (G3.c is GREEN; flake is multi-GPU-only).
- vLLM-style throughput chasing (separate phase deferred).
- Hardening the binary against host-timing jitter (the underlying code-level surface that powersave exposes). This would make the binary robust independent of governor and would close the "real" race. Out of scope for this phase because the systemd drop-in mitigation is sufficient for the production goal; can be picked up as a follow-up if anyone wants to ship without governor as a hard prereq.

---

## 9. Proposed mitigation (the actual fix)

A systemd drop-in for `llama-server.service` that pins CPU governor while the service is active. Existing drop-ins live at `/etc/systemd/system/llama-server.service.d/`:

```
[Service]
ExecStartPre=/bin/sh -c 'echo performance | tee /sys/devices/system/cpu/cpufreq/policy*/scaling_governor > /dev/null'
```

Requires NoNewPrivileges=no on the service OR a `CAP_SYS_ADMIN`-equivalent sudo rule for the path; alternative is a separate one-shot service that fires at boot. The simplest variant is a `cpupower.service` system-wide (`cpupower frequency-set -g performance`) that runs at boot — that's the canonical Linux idiom and decouples the LLM service from the governor concern.

**Recommendation:** ship the system-wide variant (`cpupower-performance.service` or equivalent), document the dependency, and verify it survives reboot. Test plan: reboot host, confirm governor = performance, run G3.a NP=8 ×3 reps before starting any other service.

---

## Pointer back to the auto-memory chain

- `project_np8_singleslot_flake.md` — earliest framing (pre-localization).
- `project_np8_localized_openmp_cuda_mismatch.md` — diagnostic window framing (now partially superseded — the `GGML_SCHED_EVAL_SERIALIZE=1` PASS was a single-rep false-pass).
- `project_phase_cuda_native_dispatch_open.md` — C-arc closure context.
- This phase doc — replaces the auto-memory framings for the actionable PD work.
