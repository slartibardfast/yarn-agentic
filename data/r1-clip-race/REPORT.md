# PHASE_R1_CLIP_RACE Phase A — closure report

**Opened:** 2026-05-28 10:35Z
**Closed + deployed:** 2026-05-28 12:45Z
**Total elapsed:** ~2 hours
**Outcome:** Multi-GPU CLIP cross-encode non-determinism structurally
characterized + fixed. Production deployed.

## 1. Problem statement

`PHASE_PERF_R3_FOLLOWUP` closed the production R1 ctx-allocation tax
(-25.9% → -7.9% TG at ctx=256k) with an interim narrow-it: the LM
decoder opts out of the PHASE 46 B.5e activation-buffer-clear that
costs ~6.7% of per-step wall time. Multi-GPU CLIP kept the buffer-
clear active.

But the B.5e fix was always a workaround — Phase 46's closure
acknowledged that "the kernel reading partially-initialized memory
was not localized at the kernel level," and an empirical test
during the prior session showed it didn't even achieve 100%
determinism (8/10 vs 2/10 sha256 split across CLIP encodes). The
Phase 46 B.7 perf gate validated median latency, not response
identity, so the "bit-correct + reproducible" claim was never
empirically enforced by the closing acceptance test.

The job for this phase: localize the actual race and either fix the
underlying kernel (deleting the workaround entirely) or characterize
it well enough to know what we're shipping.

## 2. Hypothesis space at phase open

Six hypotheses, with discriminators:

| H | claim | discriminator |
|---|---|---|
| H1 | CLIP encoder is bit-deterministic; LM-side state leaks across requests | embedding hash across encodes |
| H2 | CLIP encoder is non-deterministic; LM faithfully amplifies | same — embedding hash diverges |
| H3 | Both contribute | both vary |
| H4 | Specific gallocr-reused buffer carries the stale state | bisect-by-buffer |
| H5 | Paged-KV / allocator address evolution between encodes | T5.9 build A/B |
| H6 | Greedy decode flips on near-equal logits (LM-internal) | standalone LM determinism test |

## 3. The chronology — investigation as it actually happened

### 3.1 First test — confounded design

I ran the 10-encode CLIP determinism gate with `CLIP_LOG_FINAL_HASH=1`
to capture the embedding tensor's FNV-64 hash per encode. Result:
**10 distinct CLIP embedding hashes** across 10 encodes; LM produced
2 distinct response sha256s (8 vs 2 split).

I confidently concluded: "CLIP encoder is non-deterministic at the
bit level." Wrote that up. Proposed Phase B as "bisect-by-buffer in
the CLIP sched" to find which buffer carries the state.

### 3.2 User pushback — interleaving hypothesis

The user pushed back: each chat request runs `CLIP encode → LM
completion → CLIP encode → LM completion …`. The LM completion
between two CLIP encodes mutates shared GPU state (CUDA allocator
free list, cuBLAS workspace, NCCL communicator state). The next
CLIP encode then operates in a slightly different environment.

The test I'd run did NOT isolate CLIP — it measured CLIP+LM
interleaved. The "CLIP-alone is non-deterministic" claim was an
unjustified leap.

This was a correct and important correction.

### 3.3 Second test — the discriminating experiment

I patched `clip.cpp:clip_image_batch_encode` to add a
`CLIP_BENCH_REPEAT_N` env knob that runs the encode N times
internally in a tight loop, with zero LM completion between
iterations. One chat request → N back-to-back CLIP encodes in the
same process, same call stack.

Result: **10/10 distinct embedding hashes still** — `da08dedb...`,
`e0b8efd9...`, `98c9f836...`, `715ce5db...`, `f305c72d...`,
`d4c05c24...`, `df827337...`, `f3b3fbf6...`, `df5e3993...`,
`ad430caa...`.

The user's interleaving hypothesis was falsified by the test the
hypothesis itself motivated. CLIP encoder IS non-deterministic
back-to-back in the same process with no LM activity between them.

### 3.4 User direction — survey upstream, then code audit

The user redirected: study how upstream llama.cpp and vLLM handle
similar mixed-kernel workloads, then audit the actual code rather
than bisecting by runtime tunables (NCCL on/off, `GGML_CUDA_DISABLE_GRAPHS`,
`CUBLAS_WORKSPACE_CONFIG` variants).

Web survey findings (saved verbatim in MEMORY.md):

- **vLLM v1 encoder disaggregation** (Nov 2025): separates the vision
  encoder into its own process. The motivation in their docs is
  throughput and scalability — not determinism. They did not document
  the cross-encode race we're seeing. Their architectural fix
  incidentally eliminates the issue as a side effect.
- **llama.cpp upstream `mtmd`** library uses the SAME coupled-in-process
  architecture as ik_llama.cpp. The mtmd context allocates its own
  `ggml_backend_sched` (separate from the LM sched) but shares the
  CUDA context, streams, cuBLAS workspace, and global allocator.
- Auto-memory `project_phase46_npc_localized.md` had concluded "race
  is in ggml-backend.cpp's openmp parallel multi-backend path, not
  libmgpu." But that openmp path was deleted by `PHASE_CUDA_NATIVE_DISPATCH`
  C1, replaced with single-threaded dispatch. So that prior
  localization no longer applies to the current binary.
- Auto-memory `project_phase46_node73_localized.md` said the first
  cross-encode divergence appeared at node 73 (libmgpu W_o matmul,
  `tp.cpp:464`). But this was on the pre-C1 binary; structural state
  has changed since.

### 3.5 Code audit — maximum effort

I audited the kernels and dispatch paths CLIP exercises, focused on
what's different from LM:

| component | finding |
|---|---|
| `reduce.cu` (libmgpu's REDUCE-ADD across devices) | Ring-reduce path (`ne[1] >= 32`) is mathematically order-invariant for nhave=2 — FP add is commutative |
| `mul_mat_q_split_k` + `mul_mat_q_split_k_fixup` | Explicitly designed for determinism — canonical-order combine, `#pragma unroll`, kernel comment confirms intent |
| `conv2d.cu` (CLIP's patch embedding) | Each thread computes one output element with fixed inner-loop accumulation order, no atomics |
| `atomicAdd` usage across `ggml-cuda/` | Only one site (`ssm-conv.cu:355`, counter atomic — not FP accumulation) |
| `cublasGemmEx` calls | Algo-pinned to `CUBLAS_GEMM_ALGO0_TENSOR_OP` with explicit comment "different algos produce different fp accumulator orders" |
| Direct `cublasLt*` calls | None — the `cublasLt::splitKreduce_kernel` seen in nsys is cuBLAS-internal via the algo-pinned path |
| `ggml-backend.cpp` dispatch | C1 single-threaded; `needs_sync` sticky when `has_reduce`; ensures every cross-backend input read syncs |
| `GGML_SCHED_MAX_COPIES` | =1 on this build → CUDA outer-graph capture is structurally unreachable |
| CLIP FA dispatch (sm_75, head=80, n_pos=1024, F32 precision) | Routes to `ggml_cuda_flash_attn_ext_wmma_f16_case<80,80,32,float>` → heuristic picks parallel_blocks=1 (no split-K combine) |

The audit cleared every kernel I looked at for fixed-shape determinism.
Yet empirically the embeddings differ 10/10.

I flagged the question to the user: "What's actually being non-
deterministic, if all the kernels are designed to be deterministic?"

### 3.6 User direction — apply hashing approaches that worked before

The user redirected me to use what Phase 46 already developed.
Specifically: per-node DtoH hashing in the eval-callback, the exact
technique that originally localized to node 73 on the pre-C1 binary.

I found `clip.cpp:511` — the existing `clip_debug_eval_cb` already
implements this. It writes one line per evaluated node to the file
named by `CLIP_CAPTURE_HASH=<path>`, format
`node_idx, op, name, shape, hash, ptrs`.

### 3.7 The bisect, finally

Driver: launch llama-server with `CLIP_DEBUG_SCHED=1 CLIP_CAPTURE_HASH=
<hash_file>`, send two chat requests in one server lifetime,
shutdown. Both encodes' per-node hashes land in the same file
(3428 lines = 1714 nodes × 2 encodes).

Result of the diff:

> **0 diffs found across 1714 nodes.**
>
> All nodes byte-identical between encodes.

Both encodes ALSO produced the same final embedding hash:
`CLIP_FINAL_HASH = 2554e340101807ab` (× 2). Bit-identical.

**With per-node sync via the eval callback active, CLIP is fully
deterministic.** Without the callback (and without buffer-clear), the
race fires.

### 3.8 The fix

I patched `clip.cpp` to install a no-op eval callback by default on
the multi-backend CLIP sched. Returns `true` on every ask, forces
`ggml_backend_sched_eval` into the per-node `compute_async +
ggml_backend_synchronize` path. Escape hatch `CLIP_DISABLE_SYNC_FENCE=1`.

Ran the 10-encode determinism gate. Result: **10/10 byte-identical
CLIP responses** (`fb5167dbc1e7f95b` × 10).

I then proposed deleting the B.5e buffer-clear entirely on the
hypothesis that per-node sync subsumed it.

### 3.9 The second falsification — buffer-clear is not redundant

The user authorized the deletion. I removed the buffer-clear, kept
the sync fence on. Ran the gate.

Result: **3 distinct sha256s across 10 encodes.**

So the sync fence and the buffer-clear cover **disjoint failure modes**.
The per-node sync fence covers cross-stream timing variance. The
buffer-clear covers stale-reads from gallocr-reused buffers where
some kernel partial-writes its output region (the original Phase 46
B.5e theory). Both are required.

I reverted the deletion, restored the per-sched `zero_on_reset` flag
with the LM opt-out, and re-ran. Result: **10/10 byte-identical**
once more.

### 3.10 Empirical matrix — the load-bearing evidence

Final state of the bisect matrix:

| sync fence | buffer-clear | response sha256s (10 encodes) |
|---|---|---|
| OFF | OFF | 10/10 distinct |
| OFF | ON  | 8/10 vs 2/10 (the prior production state) |
| ON  | OFF | 3 distinct |
| **ON**  | **ON**  | **10/10 IDENTICAL ✓** |

Per-node node-by-node confirmation with both fixes active:
**1714/1714 nodes byte-identical** across two back-to-back encodes
in the same process.

## 4. Final architecture

| sched | per-node sync fence | B.5e buffer-clear |
|---|---|---|
| CLIP multi-backend | ON (default) | ON (`zero_on_reset=true`, default) |
| LM single-stream  | not installed | OFF (`zero_on_reset=false`, opt-out) |

The CLIP sched gets both fixes because both failure modes apply to
its multi-device parallel work. The LM sched skips both: LM has no
multi-device parallel split (each layer runs sequentially on a single
backend per split-mode-graph), and LM has no cross-encode state-leak
surface (autoregressive decoding within one completion; KV cache
state evolves monotonically, never re-reads stale memory).

## 5. Performance impact

| metric | value | versus |
|---|---|---|
| CLIP encode median latency (post-fence) | 10466 ms | prior 10392 ms |
| Sync fence overhead | +0.7% | within rep noise |
| LM TG R1 tax at ctx=256k | -7.9% | unchanged (LM unaffected) |
| LM TG determinism (18 reps × 6 configs) | 18/18 identical | unchanged |
| LM absolute TG at ctx=256k | 18.79 t/s | unchanged from pre-fence |
| LM absolute TG at ctx=8k | 20.40 t/s | unchanged |

The sync fence cost was much lower than predicted (~20-30%) because
the multi-device CLIP graph already has cross-split syncs at split
boundaries. Per-node sync within a single device's split adds only
launch-overhead-class delays, mostly absorbed by what the dispatcher
already does.

## 6. Code shipped

Three submodule commits on `production/2026-q2-next` (b2cf8fbf → 4f0a045f):

- **`af41d2b0`** — `llama: O(1) per-seq max-pos cache replaces full-pool scan`
  (PHASE_PERF_R3_FOLLOWUP small lever; preserved here for completeness)
- **`44f81ad1`** — `ggml-backend: ggml_backend_sched_set_zero_on_reset opt-out`
  (PHASE_PERF_R3_FOLLOWUP main lever — LM perf win)
- **`4f0a045f`** — `clip: per-node sync fence by default + retain B.5e
  buffer-clear` (this phase)

Parent repo commit `f823872` bumps the submodule pointer.

Deploy: `scripts/deploy-llama-server.sh` clean — atomic install, hash
verify, regression guard pass, `/health` 200 within 4 s, journal
shows `per-node sync fence installed (default)`.

## 7. Lessons learned

### 7.1 Bisect methodology — what helped

- **Per-node DtoH hashing** in the eval callback (the Phase 46
  technique) was exactly right. It localized the question from "where
  in the encode does divergence happen?" to "is the kernel layer
  deterministic given sync?" in a single bench.
- The **back-to-back-in-same-process** discriminator (`CLIP_BENCH_REPEAT_N`)
  was essential to falsify the interleaving hypothesis. Without it,
  the LM-side state-mutation explanation would have remained plausible
  and we'd have chased a different fix.

### 7.2 Bisect methodology — what didn't help

- **Static code audit alone**, even at maximum effort, could not pin
  the source. Every kernel I read was designed for determinism.
  Audit cleared the question "is any kernel non-deterministic at
  fixed shape?" but couldn't answer "where is the timing variance
  introduced?"
- **Web survey of upstream / vLLM** confirmed that vLLM didn't
  document this race (their fix was throughput-motivated and
  incidentally eliminated it). Useful for ruling out "industry knows
  the answer" but didn't surface a specific patch.
- **Runtime knob bisect** (NCCL on/off, etc.) — the user explicitly
  ruled this out as "not credible." That call was right; the
  empirical fix came from a structural-instrument bisect, not knob
  flipping.

### 7.3 Hypothesis discipline

Two times in this phase I confidently asserted a conclusion and the
next experiment falsified it:

1. "CLIP encoder is non-deterministic at the bit level" (after the
   first test, before the back-to-back discriminator). The user's
   pushback caught it. The follow-up test confirmed CLIP itself
   varies, but the original assertion was based on insufficient data.

2. "Per-node sync makes the buffer-clear redundant" (after the
   sync-fence test gave 10/10 identical). The deletion experiment
   immediately falsified it (3 distinct sha256s without the clear).

In both cases, the discipline of "run the next experiment before
landing the next change" caught the error within minutes. Without
that discipline, either error would have shipped a regression.

### 7.4 Two-mode failures need two-fix solutions

The original Phase 46 B.5e closure shipped three combined fixes and
documented each as covering "a different mode." The current binary
has only two of those three structurally in place (C1 deleted the
openmp dispatch which the per-split drain was protecting). This
phase confirmed the remaining two are independent and both required.

In retrospect, the prior session's hypothesis that "B.5e overstated
its impact" was correct (the gate is perf-only) but missed that the
clear was still load-bearing — it's just covering a different
failure mode than the sync fence does.

## 8. What does NOT close with this phase

Two future-work items are noted in the phase doc but explicitly out
of scope:

### 8.1 Localize the specific partial-writing kernel

The B.5e buffer-clear is a whole-allocation hammer. The actual
kernel that partial-writes its output region was never identified —
neither in Phase 46 nor here. Identifying it would let us patch the
kernel to fully overwrite its output, after which the buffer-clear
could be deleted entirely (not just opted-out for LM).

Tools available for this work: `CLIP_CAPTURE_SKIP_OPS=<op_name>` env
knob (`clip.cpp:536`) skips hashing for specific ops; can be repurposed
to selectively SKIP the sync fence per op to find which op's lack of
sync causes divergence. Or `compute-sanitizer --tool=initcheck` to
mechanically detect uninitialized device-memory reads.

Not blocking. Future phase if production cost pressures push us.

### 8.2 Narrow the sync-fence policy

Per-node sync is conservative — it forces sync between every node,
not just nodes where a race actually exists. Phase 46 tried
"reduce-only sync" and showed it wasn't sufficient, but the minimal
set of must-sync nodes was never enumerated.

At current cost (+0.7% on CLIP encode), the perf gain from narrowing
is small. Probably not worth pursuing unless the cost grows in some
future workload.

## 9. Acceptance — closed

- [x] CLIP cross-encode race structurally characterized (two
      independently load-bearing failure modes)
- [x] Both fixes shipped and empirically proven required
- [x] 10/10 byte-identical CLIP responses with both fixes active
- [x] 18/18 byte-identical LM TG reps (LM perf preserved via opt-out)
- [x] Submodule pushed (b2cf8fbf → 4f0a045f), parent pointer bumped
- [x] Deployed via `scripts/deploy-llama-server.sh`, journal confirms
      `per-node sync fence installed (default)`
- [x] MEMORY.md closure entry committed
- [x] Phase doc reflects empirical reality (corrected from initial
      "delete the workaround" hypothesis)

## 10. Production state going forward

The LM workload (the dominant production traffic) is materially faster
than at the start of this session window:

| state | LM TG @ ctx=256k | R1 tax | CLIP determinism |
|---|---|---|---|
| Pre-session (pre-deploy) | 14.34 t/s | -25.9% | 8/10 vs 2/10 |
| Post-PHASE_PERF_R3_FOLLOWUP | 18.79 t/s | -7.9% | 8/10 vs 2/10 (interim) |
| **Post-PHASE_R1_CLIP_RACE Phase A** | **18.79 t/s** | **-7.9%** | **10/10 IDENTICAL** |

LM TG perf is unchanged from the prior step (the sync fence affects
CLIP only). CLIP determinism is now fully bound.

Both phases are CLOSED.
