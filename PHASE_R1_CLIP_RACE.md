# PHASE_R1_CLIP_RACE — localize and fix the actual CLIP cross-encode race

**Opened:** 2026-05-28 10:35Z
**Last revised:** 2026-05-28 10:50Z
**Status:** PD pack — no experiments run yet
**Parent:** [PHASE_PERF_R3_FOLLOWUP](PHASE_PERF_R3_FOLLOWUP.md) (R1 closed
with interim narrow-it; this phase removes the workaround)
**Production state:** submodule `44f81ad1` deployed; LM decoder opts out
of the B.5e buffer-clear; CLIP keeps default-true.

## What the phase has to deliver

The single binding outcome that closes this phase:

> A multi-GPU CLIP encode pipeline that produces **10/10 byte-identical
> response sha256s** across 10 chat completions of the same image with
> the same prompt at temperature=0, **without** relying on the
> `ggml_backend_sched_reset` activation-buffer clear. Once that holds,
> the `zero_on_reset` flag added in submodule 44f81ad1 is removed.

This phase is **not perf-critical** — the R1 win is already banked in
production. It's technical-debt cleanup to delete a workaround whose
true effect was overstated.

## What is empirically known (2026-05-28)

Test: 10 chat completions, same image (`examples/mtmd/test-1.jpeg`,
1024 image tokens), same text prompt, temperature=0, NP=1, ctx=262144.
The response is `choices[0].message.reasoning_content`.

| build state                          | sha256 split  | source |
|---|---|---|
| B.5e buffer-clear enabled (default)  | 8/10 vs 2/10  | run-20260528T103042 |
| B.5e buffer-clear disabled (everyone)| 7/10 vs 3/10  | run-20260528T100733 |

The buffer-clear **reduces but does not eliminate** the divergence. The
Phase 46 "bit-correct + reproducible output" closure claim is not
backed by the actual B.7 gate
(`tests/spec/test-clip-encode-latency.cpp`), which validates only
median encode latency.

The divergence is **semantic-level, not byte-noise**: encodes in the
minority cluster produce entirely different opening sentences ("identify
what is in the image" vs "a short sentence describing the image"). For
greedy decode this requires early-step logit divergence sufficient to
flip a softmax argmax.

## Hypothesis space (and the discriminator that collapses each)

| H# | Hypothesis | Discriminator |
|---|---|---|
| H1 | CLIP encoder produces bit-identical embeddings; LM-side state leaks across requests | Capture CLIP embedding tensor sha256 across 10 encodes (Phase A) |
| H2 | CLIP encoder produces non-deterministic embeddings; LM faithfully amplifies into different responses | Same as H1 — embedding sha differs across encodes |
| H3 | Both CLIP and LM contribute | Phase A shows both vary |
| H4 | A specific gallocr-reused buffer is read-before-fully-written by some kernel | Phase B bisect-by-buffer + compute-sanitizer initcheck |
| H5 | The race is in a host-side allocator state (block-table layout differs across encodes) | T5.9 paged_kv_allocator state dump pre/post encode |
| H6 | Sampling determinism (e.g. logit ordering of nearly-equal scores) is non-deterministic on the GPU | LM stand-alone test (no CLIP) with identical prompt 10× — checks whether LM is deterministic on its own |

H1-H3 are first because they require zero kernel work and collapse the
entire space.

## Phase A — discriminate CLIP-side vs LM-side (~60 min)

**Goal:** localize whether the cross-encode divergence is in CLIP
encoder output or in the LM pipeline downstream.

### A.1 — Add NVTX markers around the CLIP encode output read

In `examples/mtmd/clip.cpp` (or wherever the CLIP forward pass writes
the final image-embedding tensor), add a debug hook gated by env var
`LLAMA_DEBUG_CLIP_EMBED_HASH=1`:

- After the CLIP forward pass completes, before the embedding tensor is
  consumed by the LM input layer, `cudaMemcpy` it to host and
  `sha256sum` the bytes.
- Log to stderr: `clip_embed_sha256: <hex>`.

Same patch shape as the SIGUSR1 profiler hook from
PHASE_PERF_R3_FOLLOWUP (zero-link-dependency, env-gated). ~20 LOC.

### A.2 — Re-run the 10-encode gate with the embedding hash hook

```bash
sudo systemctl stop llama-server.service
LATENCY_N=10 LLAMA_DEBUG_CLIP_EMBED_HASH=1 \
  BIN=/home/dconnolly/yarn-agentic/ik_llama.cpp/build/bin/llama-server \
  PORT=18293 \
  bash /home/dconnolly/yarn-agentic/scripts/verify-multigpu-clip.sh \
       --image-min-tokens 1024 --image-max-tokens 1024
```

Then extract from `server.stderr`:
```
grep clip_embed_sha256: $RUN_DIR/server.stderr
```

### A.3 — Decision tree

- **All 10 embedding hashes identical** → CLIP encoder is deterministic;
  bug is LM-side. Skip to Phase D-LM (LM-side localization).
- **Embedding hashes split 8/10 vs 2/10 mirroring the response split** →
  CLIP encoder is the source. Proceed to Phase B.
- **Embedding hashes have a different split (e.g. 9/10 vs 1/10)** → both
  CLIP and LM contribute; investigate CLIP first since it's the
  upstream source.

### A.4 — Sanity: standalone LM determinism check (parallel test)

Send 10 identical text-only completions (no image) at temperature=0
through the same prompt template. Sha256 the responses. If they're all
identical, LM is deterministic on its own → reinforces "bug is CLIP-side"
in case Phase A.3 was ambiguous.

## Phase B — if CLIP-side: bisect by buffer (~2-3h)

**Goal:** identify which gallocr-allocated buffer(s) actually carry the
divergence-causing state across encodes.

### B.1 — Instrument `ggml_backend_sched_reset` to clear ONE buffer

Add a debug knob `GGML_DEBUG_BUFFER_CLEAR_INDEX=N` (env var):

- When N is -1 (default): current behavior gated on `zero_on_reset`.
- When N is 0..n_buffers-1: clear only buffer index N, regardless of
  `zero_on_reset`.

This lets us A/B "clear all buffers" vs "clear only buffer N".

### B.2 — Run bisect

For each buffer index N in [0, n_buffers), run the 10-encode gate twice:
once with N as the only cleared buffer, once with all buffers cleared
(`zero_on_reset=true`).

Per buffer: compare sha256 split.

- A buffer that, when alone cleared, gives **8/10 vs 2/10** matches
  current default → that buffer drove the masking. Other buffers'
  contents don't matter.
- A buffer that gives **3/10 vs 7/10** (the no-clear baseline) is
  irrelevant — clearing it didn't change anything.
- Multiple buffers may be load-bearing; iterate to find the minimal set.

Per buffer this is ~3 minutes of CLIP encodes; n_buffers typically
~4-12 for the CLIP sched, so ~30-60 min total bench wall time.

### B.3 — Identify the kernel(s) writing the load-bearing buffer(s)

For each load-bearing buffer, find which graph node(s) write to it. Use
`GGML_SCHED_DEBUG=1` to dump the per-node backend assignment, or grep
the sched's `node_backend_id` array against tensor names.

The write site tells us which CUDA kernel is leaving partial-init
bytes. Read that kernel.

## Phase C — compute-sanitizer initcheck (parallel to Phase B, ~45 min)

**Goal:** mechanically detect reads of uninitialized device memory.
Independent of Phase B (different methodology — if both agree, the
finding is solid).

### C.1 — Build with debug symbols, sanitizer-friendly

```bash
cmake -B build-sanitize -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CUDA_FLAGS='-g -G -lineinfo' \
    -DCMAKE_CXX_FLAGS='-g -O0' \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15 \
    [+ same flags as production build]
cmake --build build-sanitize -j --target llama-server
```

### C.2 — Run compute-sanitizer on a single CLIP encode

```bash
compute-sanitizer --tool=initcheck --check-device-heap=yes \
    --log-file=/tmp/clip-initcheck.log \
    build-sanitize/bin/llama-server \
    -m <gguf> --mmproj <mmproj> --device CUDA0,CUDA1 \
    --split-mode graph --tensor-split 1,1 -ngl 999 \
    --no-mmproj-offload --image-min-tokens 1024 \
    --port 18299 --host 127.0.0.1 &
# Send one image encode request via curl, then SIGINT the server
```

initcheck is very slow (10-100×). One encode is enough; we're looking
for any uninitialized-memory read report.

### C.3 — Parse the initcheck log

For each reported uninitialized read, capture:
- The kernel name
- The file:line of the read (CUDA `-lineinfo` gives this)
- The buffer address being read

Cross-reference with Phase B's load-bearing buffer addresses if Phase B
already ran. Convergent evidence = locked finding.

## Phase D — kernel fix (~1-3h, depends on what Phase B/C find)

Once the kernel is identified, the fix shape depends on the failure
mode:

**D-partial-write:** kernel writes only N of M bytes. Fix: extend the
write loop bounds, OR ensure the output tensor's nbytes matches what
the kernel writes.

**D-stale-view:** kernel reads through a view that includes bytes
outside the previous write region. Fix: tighten the view to the
actually-written region, OR have the upstream kernel write zeros to
the gap.

**D-allocator-aliasing:** gallocr is reusing a buffer for a SMALLER
tensor than what previously occupied that address, leaving stale bytes
the new tensor's view doesn't include but a downstream kernel reads.
Fix: clear-on-reuse in gallocr (cheaper than the per-encode global
clear), OR fix the downstream kernel's read bounds.

Verification per fix: 10-encode gate with `zero_on_reset=true` flag
deleted entirely. Must hit 10/10 sha256-identical to close.

## Phase D-LM (only if Phase A.3 points to LM-side)

Same shape as Phase B/C but for the LM sched:

- B.5e buffer-clear in LM sched is currently OFF (we opted out
  44f81ad1). So if LM has its own race, it's already exposed.
- Phase A.4's standalone LM determinism check is the discriminator. If
  it fails, the LM has independent non-determinism that the original
  buffer-clear was also masking — narrow-it broke that mask.
- Plan would be: bisect-by-buffer on the LM sched, identify the kernel.
- Fix is the same shape as Phase D.

## Phase E — flag removal + verification (~30 min)

Once the underlying race is fixed:

1. Delete the `zero_on_reset` field from `struct ggml_backend_sched`.
2. Delete `ggml_backend_sched_set_zero_on_reset` (header + impl).
3. Delete the `ggml_backend_sched_set_zero_on_reset(...)` calls in
   `src/llama.cpp`.
4. The buffer-clear block in `ggml_backend_sched_reset` can stay or
   go — if the underlying race is truly fixed, it's safety theater
   either way. Recommend deleting it too (cleaner code, no per-step
   memset cost for CLIP).
5. Rebuild.
6. Re-run BOTH gates:
   - 10-encode CLIP determinism gate (`verify-multigpu-clip.sh`):
     must be 10/10 sha256-identical
   - LM perf bench (`perf-r3-followup-phase3.sh`): must match
     post-narrow-it performance
   - LM determinism (across reps of `perf-r3-followup-phase3.sh`):
     must be 18/18 byte-identical
7. Deploy via `scripts/deploy-llama-server.sh`.
8. Update PHASE_R1_CLIP_RACE.md → CLOSED.

## Risks + mitigations

| risk | mitigation |
|---|---|
| Phase A shows BOTH CLIP and LM diverge → scope doubles | Discover early; if both, time-budget Phase B for CLIP only, schedule Phase D-LM as a follow-up phase |
| Phase B finds NO single load-bearing buffer (the race is in interaction across multiple buffers) | Fall back to Phase C initcheck as primary localization |
| Phase C initcheck reports nothing actionable (build-time compile errors with -G, sanitizer doesn't catch the issue) | Accept that the race is in HOST-side state, not device memory; pivot to host-side instrumentation (allocator state dump) |
| The fix turns out to require an upstream llama.cpp change we can't unilaterally land | Document, keep the interim flag, close phase as "blocked on upstream" |
| Phase 46 NPC localization claim ("race is in openmp parallel multi-backend path, deleted by C1") still load-bearing despite C1 → race may have re-emerged somewhere C1 didn't cover | Phase B should surface this via the bisect; no special handling needed |

## Time + token budget

| Phase | Wall | Tokens | Notes |
|---|---|---|---|
| A (CLIP-vs-LM discriminator) | 60 min | 8k | Small server-side patch + bench |
| B (bisect by buffer) | 2-3h | 15k | Patch + N bench iterations |
| C (compute-sanitizer initcheck) | 45 min | 6k | Slow tool but mechanical |
| D (kernel fix) | 1-3h | 12k | Depends on what's found |
| E (flag removal + verify) | 30 min | 4k | Delete code + re-verify both gates |
| Report aggregation | 30 min | 5k | |
| **Total** | **~6-8h** | **~50k** | Fits in a single maintenance window |

## Maintenance window requirements

- Production service must be **STOPPED** for all bench iterations
  (both GPUs at capacity under production profile, can't run a second
  binary concurrently). Same constraint as PHASE 46.
- Pre-window: announce expected downtime
- Mid-window snapshot: gpu-clocks lock at 1455 MHz (LM determinism
  contract — see `project_phase46_gpu_clock_discovery.md`)
- Post-window: restart `llama-server.service`, verify `/health=200`,
  run vision smoke

## Acceptance — phase closes when ALL of

- [ ] Phase A: CLIP-vs-LM discriminator answered with binding evidence
- [ ] Phase B: load-bearing buffer(s) identified by name + index
- [ ] Phase C: compute-sanitizer report captured (clean OR identifying
      a specific kernel)
- [ ] Phase D: actual race fixed (kernel patch or allocator fix); no
      reliance on the buffer-clear
- [ ] Phase E: 10/10 byte-identical CLIP encodes; 18/18 byte-identical
      LM TG reps; perf matches post-narrow-it state
- [ ] `zero_on_reset` flag and its callers deleted
- [ ] Buffer-clear in `ggml_backend_sched_reset` deleted (or
      explicitly kept with rationale)
- [ ] Submodule pushed; parent pointer bumped; deployed
- [ ] MEMORY.md entry + auto-memory updated

## Standing notes

- The current interim measure (LM decoder opts out, CLIP keeps default)
  is correct and shippable. This phase replaces it, doesn't fix it.
- The Phase 46 closure claim of "bit-correct + reproducible output"
  has been empirically shown to be overstated. If we don't fix this
  properly, future regressions could ship under the same false claim.
- After this phase, the `LLAMA_NSYS_PROFILE_RANGE` SIGUSR1/SIGUSR2 hook
  (submodule `f2a7ad10`) can stay — it's general profiling
  instrumentation, not race-mitigation.
