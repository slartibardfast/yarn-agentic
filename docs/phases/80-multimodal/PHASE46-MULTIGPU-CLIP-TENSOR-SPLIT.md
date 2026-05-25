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
2. `MTMD_TENSOR_SPLIT=1,1` makes the mmproj weights physically
   split 1:1 across CUDA0 and CUDA1 (verified by inspecting per-device
   VRAM usage before vs after a vision encode).
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

Stretch:

7. Latency for 1024-token vision encode is within 1.5× the
   single-GPU baseline. (Cross-device peer transfer adds overhead;
   we are not optimising for speed, only for fit.)

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

### Step 2 — Split buffer type for mmproj weights (~80-150 LoC)

`examples/mtmd/clip.cpp`, weight allocation path (around the
`clip_model_loader` weight assignment block; exact line will be
identified during implementation):

- Read `MTMD_TENSOR_SPLIT` env (comma-separated floats, e.g. `1,1`).
- If set and multiple CUDA backends were initialised in Step 1,
  construct a `ggml_backend_cuda_split_buffer_type(tensor_split)`
  (following the pattern in `src/llama-model-loader.cpp` /
  `src/llama-model.cpp` for the LM).
- Use that buffer type for the mmproj weight tensors instead of the
  per-device default.
- The `ggml_backend_sched_t` will then distribute compute across
  devices based on weight residency.

Acceptance for Step 2: `nvidia-smi` (or `cudaMemGetInfo` from a
test harness) shows mmproj weight bytes split across both devices
within tolerance of the requested ratio. Vision encode completes;
output matches single-device baseline.

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

## 5. Orphan audit

Code that touches or assumes CLIP's backend structure today.
**None** of these would be broken by adding multi-backend support
provided the single-backend path still works (Goal #6).

| Region | File:Line | Risk |
|---|---|---|
| CLIP backend init | `examples/mtmd/clip.cpp:488-516` | **This is the patch site.** Both code paths (single env, fallback `index=1`) preserved. |
| CLIP sched creation | `examples/mtmd/clip.cpp` (after the init block) | Already vector-typed; passes whatever `backend_ptrs.size()` is. |
| mmproj weight loader | `examples/mtmd/clip.cpp` weight assignment | **The other patch site.** New `if (split_buft) ...` branch; default path unchanged. |
| `clip_image_batch_encode` | `examples/mtmd/clip.cpp` | Uses `ggml_backend_sched_graph_compute` — backend-agnostic. No change needed. |
| `MTMD_BACKEND_DEVICE` consumers | systemd drop-in only | Env name unchanged; comma-list is a strict superset. |
| llama-server multimodal request path | `examples/server/server-context.cpp:161` | `params.mmproj_use_gpu` boolean — unaffected. |

**No work is orphaned.** The change is strictly additive to clip.cpp's
init path. Compare to PHASE 39's hybrid memory backport (1500-2000
LoC at risk): this is on a different scale of structural impact.

## 6. Risks

1. **Cross-device peer access overhead.** TU102 + NV2 NVLink is
   measured working between CUDA0 and CUDA1 (per `project_xeon_host_hardware.md`),
   but the vision encoder's activations crossing device boundaries
   will add latency. Estimated 20-50 % wallclock penalty for the
   1024-token encode. Acceptable for the use case (vision is
   request-driven, not in the steady-state decode loop).
2. **cuBLAS workspace duplication.** Each device that hosts vision
   ops allocates its own workspace. Per-device cost increases by
   ~256 MiB. Headroom budget at the top of this doc accounts for
   this.
3. **Determinism.** Multi-device reductions in the vision encoder
   may introduce f32 epsilon non-determinism on paths that aren't
   batch-invariance-controlled (the LM's invariance work doesn't
   cover libmtmd). Acceptance criterion 4 measures this; if epsilon
   exceeds some threshold, document and accept (vision-encoder
   outputs feed a non-determinism-tolerant pathway: the LM
   tokenises them with high-temperature decoding).
4. **Upstream divergence.** `clip.cpp` in mainline llama.cpp may
   evolve. We will rebase rather than maintain divergent state once
   a stable design is in place.

## 7. Out of scope

- Optimising vision-encoder latency. The goal is to fit, not to be
  fast.
- Multi-GPU CLIP for **non-CUDA** backends (Vulkan, Metal, ROCm).
  Out of scope; current production is CUDA-only.
- Rewriting the vision encoder graph for better cross-device
  locality. Step 2 uses the existing graph layout with split
  weights; scheduler decides ops based on data residency.
- Backporting upstream's vision changes. Independent track.

## 8. Estimated cost (in tokens, per CLAUDE.md §8)

- Read clip.cpp + understand backend init: ~10k
- Read LM split-buffer code as reference: ~15k
- Land Step 1 + test: ~20k
- Land Step 2 + tests: ~40-60k
- Verification + production deploy + rollback drill: ~15k
- PHASE 46 closure writeup: ~5k

**Total: ~105-125k tokens.** One focused session.

## 9. Files

| Path | Role |
|------|------|
| `ik_llama.cpp/examples/mtmd/clip.cpp:488-516` | Backend init (Step 1) |
| `ik_llama.cpp/examples/mtmd/clip.cpp` (weight load) | Split buffer (Step 2) |
| `ik_llama.cpp/tests/test-clip-multi-backend-init.cpp` | NEW (Step 3) |
| `ik_llama.cpp/tests/test-clip-weight-split.cpp` | NEW (Step 3) |
| `ik_llama.cpp/tests/test-clip-encode-equivalence.cpp` | NEW (Step 3) |
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

- [ ] Step 1: multi-backend parsing in clip.cpp
  - [ ] Implementation
  - [ ] `test-clip-multi-backend-init.cpp` GREEN
  - [ ] Journal `clip_ctx: have N back-ends:` shows ≥ 3 with `MTMD_BACKEND_DEVICE=CUDA0,CUDA1`
- [ ] Step 2: split buffer for mmproj weights
  - [ ] `ggml_backend_cuda_split_buffer_type` integration
  - [ ] `test-clip-weight-split.cpp` GREEN
  - [ ] `test-clip-encode-equivalence.cpp` GREEN
- [ ] Step 3: tests landed + RED → GREEN traced
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

### 11.2 — CLI flag parity with `--tensor-split`

Original plan routes config through `MTMD_BACKEND_DEVICE` and
`MTMD_TENSOR_SPLIT` env vars because `clip.cpp` already parses the
first. The LM uses CLI flags (`--tensor-split 1,1`,
`--split-mode graph`). Add:

- `--mmproj-devices CUDA0,CUDA1` — peer of `--tensor-split`'s
  device-selection role.
- `--mmproj-tensor-split 1,1` — peer of `--tensor-split` itself.

Plumb in `common/common.cpp` → `common_params.mmproj_devices` /
`mmproj_tensor_split`, then pass to clip.cpp during init.

The two env vars stay as a zero-cost compat fallback (env-var route
read only if the CLI flag is unset). Profile script and systemd
drop-in migrate to flags at deploy time.

Acceptance: `llama-server --help` lists both flags;
`--mmproj-devices CUDA0,CUDA1 --mmproj-tensor-split 1,1` produces
the same multi-backend init as the env-var route. Estimated
**+30-50 LoC across `common/common.cpp` and `clip.cpp`, +8-13k
tokens**.

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
2. Deploy script does:
   ```bash
   if ! strings build/bin/llama-server | grep -q "multi-backend init"; then
       echo "FATAL: build missing multi-backend CLIP — refusing to deploy" >&2
       exit 1
   fi
   ```
3. The check fires on the build-tree binary *before* the install
   step, so a bad build can't reach `/opt/llm-server/`.

Acceptance: deploy script refuses to install a `llama-server`
binary that doesn't contain the multi-backend init string. The
guard is opt-out-able with an explicit `--allow-no-mmproj-mgpu`
flag for emergency rollback to a pre-Phase-46 binary. Estimated
**+10-15 LoC in deploy script, +3k tokens**.

### Checkboxes for §11

- [ ] §11.1 — single-GPU baseline latency captured and pasted into §6
- [ ] §11.2 — `--mmproj-devices` and `--mmproj-tensor-split` flags
      implemented and tested
  - [ ] `llama-server --help` lists both flags
  - [ ] Env-var fallback still works (compat)
- [ ] §11.3 — `verify-multigpu-clip.sh` emits `evict_pressure` event
      count; if ≥ 1, PHASE35 §15.7 closed in the same commit
- [ ] §11.4 — Deploy-script multi-backend regression guard landed
      and verified by attempting to deploy a built-without-Step-1
      binary (must refuse)

