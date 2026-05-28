# PHASE_R1_CLIP_RACE — localize and fix the actual CLIP cross-encode race

**Opened:** 2026-05-28 10:35Z
**Status:** OPEN — followup to PHASE_PERF_R3_FOLLOWUP
**Parent:** [PHASE_PERF_R3_FOLLOWUP](PHASE_PERF_R3_FOLLOWUP.md) (R1 closed
with interim narrow-it measure)

## Why this phase exists

PHASE_PERF_R3_FOLLOWUP closed R1 (the ctx-allocation tax) by adding a
per-sched `zero_on_reset` flag that lets the LM decoder skip the
PHASE 46 B.5e activation-zero-on-reset path. The narrowing is correct
for the measured workload but **carries two unresolved findings that
this phase must clean up**:

### Finding 1 — the B.5e buffer-clear is interim, not a real fix

The B.5e fix lives in `ggml_backend_sched_reset` (zeroing all gallocr
activation buffers on every reset). Its design rationale is recorded
as:

> "Some kernel in the multi-device CLIP graph reads partially-
> initialized memory on subsequent encodes (view-stride / kernel-
> not-fully-overwriting issue **not localized at the kernel level**)."

The kernel was never pinned. The fix masks the symptom rather than
fixing the cause. The narrow-it preserves the mask for CLIP but the
underlying bug is unaddressed.

### Finding 2 — the B.5e fix doesn't actually achieve 100% determinism

Empirical test 2026-05-28 with the buffer-clear active (current
production behavior) showed **2/10 distinct sha256 hashes** across 10
multi-GPU CLIP encodes of the same image with the same prompt at
temperature=0. The PHASE 46 closure claim of "bit-correct +
reproducible output" was overstated. The actual Phase 46 B.7 gate
(`tests/spec/test-clip-encode-latency.cpp`) validates only median
encode latency, not output identity.

Inspecting the divergence: encodes 5 and 6 produced entirely different
response sentences ("identify what is in the image" vs "a short
sentence describing the image"). This is not byte-noise — it's the
language model producing semantically different completions, which at
temperature=0 (greedy) requires early-step logit divergence.

This implies one of:
- CLIP image embeddings DO vary slightly across encodes (downstream LM
  greedy-decode amplifies the small embedding change into a different
  response path)
- The LM's KV-cache state leaks between separate chat requests
- Some other shared host/device state mutates across requests

The buffer-clear reduces the divergence rate (from 7/10 vs 3/10
without it to 8/10 vs 2/10 with it) but does not eliminate it.

## What this phase needs to deliver

1. **Bisect by buffer.** Instead of "clear all" vs "clear none", clear
   one buffer at a time. Identify which buffer(s) actually drive the
   bimodal hash split. Most buffers will be irrelevant; the load-bearing
   one(s) localize the race.
2. **Inspect the buffer's writing kernel** for partial-write patterns
   (output tensor strides that leave gaps, tile-size mismatches with
   actual dimensions, etc.).
3. **Verify the older Phase 46 localization claims against the current
   binary.** Auto-memory has contradictions:
   - `project_phase46_test_kl_localized.md` — REDUCE-output reads
     drive the race
   - `project_phase46_node73_localized.md` — first divergence at
     libmgpu W_o matmul, hypothesis "gallocr address evolution"
   - `project_phase46_npc_localized.md` — **"race is in ggml-backend
     openmp parallel multi-backend path, not libmgpu"** (newest,
     supersedes prior)
   The openmp path was deleted by PHASE_CUDA_NATIVE_DISPATCH C1.
   So the NPC localization no longer applies to the current binary.
   Need fresh localization on the C1-deleted-openmp build.
4. **Use compute-sanitizer `--tool=initcheck`** to mechanically detect
   reads of uninitialized device memory during a CLIP encode. The
   B.5e team didn't run this; it would directly identify the kernel.
5. **Investigate whether the LM-side variation is CLIP-input-driven or
   independent.** Test: capture the CLIP embedding tensor across two
   encodes, sha256 it. If embeddings are identical but LM responses
   differ, the bug is LM-side (KV state, sampling determinism). If
   embeddings differ, the bug is CLIP-side.
6. **Fix the actual race** so the buffer-clear can be deleted entirely.
   Both LM and CLIP would benefit (CLIP encodes also pay the clear cost
   even though it's small per-encode).

## Acceptance — phase closes when

- [ ] Per-buffer bisect localizes which buffer(s) actually drive the
      cross-encode divergence
- [ ] The kernel that writes the bisected buffer is identified by name +
      file:line
- [ ] CLIP-embedding-vs-LM-response test isolates whether the variation
      is CLIP-side or LM-side
- [ ] Compute-sanitizer initcheck run captured (clean OR pointing at a
      specific kernel)
- [ ] The actual race is fixed (kernel patch or allocator fix)
- [ ] `zero_on_reset` flag is removed (or kept only for documentation
      purposes if a deliberate design choice)
- [ ] CLIP encode produces 10/10 byte-identical responses across the
      multi-GPU determinism gate

## What stays in place until this phase closes

- The interim `zero_on_reset` flag on `struct ggml_backend_sched`
- The LM decoder's opt-out call in `src/llama.cpp` at the sched init
  sites
- The PHASE 46 B.5e buffer-clear (now gated on the flag, default true
  for CLIP)

## Time estimate

- Bisect-by-buffer: ~2h (rebuild + N runs)
- Kernel reading once localized: ~1h
- initcheck run: ~30 min (slow tool, but mechanical)
- Embedding-vs-response test: ~1h
- Fix + verify: ~2h
- Writeup + flag removal: ~1h

**Total: ~7-8h scoped, single maintenance window.**

## Out of scope

- Touching the LM decoder's opt-out (it stays in place; this phase's
  win is removing the need for it AT ALL by fixing the underlying bug)
- Production deploy (the interim is shippable; this phase delivers a
  cleaner shippable state)
- Any further LM perf work (R1 is closed by the interim)

## Why now

The interim narrowing already delivered the LM TG win (+17.6pp on R1).
This phase is technical-debt cleanup, not perf-critical. Schedule
when the next maintenance window opens.
