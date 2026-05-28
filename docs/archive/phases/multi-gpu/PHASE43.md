# Phase 43 — NCCL All-Reduce + CUDA Graph Capture for MTP K=1 / K=2 Production

## Goal

Unlock CUDA graph capture for the production decode path on Qwen 3.6 27B
(dense), 2× Quadro RTX 6000 (TU102 sm_75) PCIe 3.0 x8 (NVLink soon).

**Co-equal targets:**
- **MTP K=1 captured ≥ 45 t/s** (linear chain, simplest config)
- **MTP K=2 captured ≥ 50 t/s** (tree fan-out, used by industry — EAGLE,
  SpecInfer, etc.)

Both at 256K X02 vs current uncaptured baselines:
- noMTP: 23.13 t/s
- MTP K=1: 22.66 t/s
- MTP K=2: 15.87 t/s

Projected captured throughput once verify is K-flat:

| Config | Verify ms (captured) | Draft ms (captured) | Cycle ms | tg t/s |
|---|---|---|---|---|
| noMTP | 28 | — | 28 | 35.7 |
| MTP K=1 | 30 | 5 | 35 | 53.1 |
| **MTP K=2 (preferred)** | **30** | **5** | **35** | **55.7** |

K=2 wins under graph capture because Δα=0.06 (more tokens/cycle) pays off
when verify cost is K-flat. Phase 41's K=2 throughput-negative finding
was specific to the UNCAPTURED regime where verify scaled super-linearly
with batch.

## Problem

`ggml-cuda.cu:4487` disables graph capture for any cgraph containing
`GGML_OP_REDUCE`. TP-split-graph emits one REDUCE per layer for
cross-device sync, so capture is silently disabled on every multi-GPU
forward. The verify cycle pays 156k extra `cudaLaunchKernel` calls per
~30 cycles of generation (Phase 41 nsys decomposition).

Microbench (Phase 42, `data/phase42-scaling-*.runlog`):

| batch | tg t/s | cycle ms | scaling vs verify(1) |
|---|---|---|---|
| 1 (noMTP) | 23.13 | 43.2 | 1.0× |
| 2 (MTP K=1) | 22.66 | 82.1 | 1.53× |
| 3 (MTP K=2 tree) | 15.87 | 122.9 | 2.47× |

Super-linear scaling 2→3 confirms launch+sync overhead dominates per
extra verify token. Graph capture flattens this because per-token
launches collapse into a single graph submission.

## Approach: NCCL all-reduce replacement

Replace the existing `ggml_cuda_op_reduce` (cudaMemcpyPeerAsync + custom
`k_reduce_add_T` kernel) with `ncclAllReduce`. Two benefits:

1. **Eliminates the `GGML_OP_REDUCE` graph-capture disable.** Either by
   (a) using a new op type the gate doesn't recognize, or (b) modifying
   the gate to allow REDUCE when NCCL is the backend. Approach (b) is
   simpler.
2. **NVLink-ready.** When the NVLink bridge arrives, NCCL automatically
   uses it (much higher cross-device BW than PCIe 3.0 x8). Code path
   needs no further change.

Validation: greedy parity at temp=0, fixed seed, byte-diff vs current
implementation.

## Scope (in this phase)

- Single-process multi-GPU NCCL (no multi-node, no MPI)
- `GGML_OP_REDUCE` as the only NCCL-routed op (broadcast/all-gather later
  if needed)
- F32, F16 dtypes (matches current reduce.cu support)
- 2-device assumption (extends naturally; no hard 2-GPU code)
- ik_llama.cpp fork only (not upstream llama.cpp/)

Out of scope:
- Multi-process / multi-node NCCL
- NCCL replacement of broadcast / all-gather / scatter (only allreduce)
- New TP layouts (we keep `--split-mode graph --tensor-split 1,1`)

## Stage breakdown (~40-60k tokens total)

### Stage 0 — Build infrastructure (~5k)

1. Verify NCCL availability: `find / -name 'nccl.h' 2>/dev/null` and
   `ldconfig -p | grep nccl`. If missing, install via package manager
   or CUDA toolkit.
2. Modify `CMakeLists.txt` to find_package(NCCL) and link the cuda
   backend library. Gate behind `GGML_CUDA_NCCL=ON` cmake flag (default
   OFF for safety, ON for our build).
3. Verify build with NCCL: header includes work, library resolves.

### Stage 1 — NCCL init in ggml_backend_cuda_context (~10k)

1. Add `ncclComm_t comm` field to `ggml_backend_cuda_context`
   (`ggml-cuda/common.cuh`).
2. Add `ggml_cuda_nccl_init()` called at backend creation when
   `n_devices > 1`:
   - `ncclCommInitAll(comms, n_devices, device_ids)` — one comm per device
   - Store comm handle on each context
3. Add `ggml_cuda_nccl_destroy()` at backend teardown.
4. Add a synthetic test (test/test-nccl-allreduce.cpp): allocate
   per-device buffers, fill with known values, ncclAllReduce, validate
   sum.

### Stage 2 — Replace ggml_cuda_op_reduce (~10k)

1. In `ggml-cuda/reduce.cu`, modify `ggml_cuda_op_reduce`:
   ```cpp
   if (use_nccl(ctx)) {
       ggml_cuda_nccl_allreduce(dst, ctx);
   } else {
       // existing peer-copy + k_reduce_add_T path (fallback)
   }
   ```
2. Add `ggml_cuda_nccl_allreduce(ggml_tensor *, ggml_backend_cuda_context &)`:
   - For each device, get the input tensor's per-device slice
   - Call `ncclAllReduce(slice, slice, nelem, ncclFloat, ncclSum, comm, stream)`
   - Result is the in-place reduced sum on every device
3. Compare-mode env var `GGML_NCCL_DISABLE=1` for falling back to old
   path during validation.

### Stage 3 — Greedy parity validation (~5-10k)

1. Build with NCCL enabled.
2. Run llama-server with current production profile + `LLAMA_NCCL_PARITY=1`.
3. Run a fixed-seed temp=0 completion against the same prompt with and
   without NCCL (toggling via env var).
4. byte-diff the responses. Must be identical.
5. If divergence: investigate. Common causes are float order-of-operation
   differences (NCCL ring vs k_reduce_add_T direct sum). Bound tolerance
   to fp32 strict-equality before allowing.

### Stage 4 — Enable graph capture (~5-10k)

1. Modify `ggml-cuda.cu:4487-4490`:
   ```cpp
   if (node->op == GGML_OP_REDUCE && !use_nccl_for_reduce(ctx_for_node)) {
       use_cuda_graph = false;
       break;
   }
   ```
2. Verify a small captured graph that includes a NCCL allreduce node
   actually runs correctly (NCCL has its own capture support; verify it
   integrates with cudaStreamBeginCapture).
3. Re-run greedy parity post-capture (capture should be transparent).

### Stage 5 — Measurement + close decision (~10-15k)

1. Re-run `scripts/probe-verify-scaling.sh` (which we just built).
2. Re-run nsys profile (`scripts/nsys-profile-tree.sh`). Verify
   `cudaLaunchKernel` count drops from ~340k to ~50k for K=1.
3. Compute deltas vs current measurements:
   - noMTP @ 256K: 23.13 → expect ≥ 30 t/s (gate: +30%)
   - MTP K=1 @ 256K: 22.66 → expect ≥ 34 t/s (gate: +50%)
   - MTP K=2 @ 256K: 15.87 → expect ≥ 40 t/s (gate: +152%, primary target
     given K=2 is the industry-standard tree pattern and projects highest
     under capture)
4. Greedy parity at production prompt set for all three.
5. Decision tree:
   - If MTP K=2 ≥ 40 t/s: propose K=2 production swap
     (`profiles/qwen36-27b-x1.sh` + `-mtp --draft 1` + `LLAMA_MTP_TREE_K=2`)
   - Else if MTP K=1 ≥ 34 t/s: propose K=1 swap (simpler config)
   - Else if noMTP captured ≥ 30 t/s: propose noMTP capture-only swap
     (smallest change)
   - Else: close negative; NCCL infrastructure stays on `phase43-nccl`
     branch as a record

If gates fail: close negative, document gap, NCCL infrastructure stays
on `phase43-nccl` branch as a record.

## Critical files

- `/home/llm/yarn-agentic/ik_llama.cpp/ggml/src/ggml-cuda/reduce.cu` —
  current GGML_OP_REDUCE impl, dispatch entry point
- `/home/llm/yarn-agentic/ik_llama.cpp/ggml/src/ggml-cuda.cu` lines
  4487-4490 — the gate to modify
- `/home/llm/yarn-agentic/ik_llama.cpp/ggml/src/ggml-cuda/common.cuh`
  — ggml_backend_cuda_context struct (add comm field here)
- `/home/llm/yarn-agentic/ik_llama.cpp/CMakeLists.txt` — build, NCCL
  link
- `/home/llm/yarn-agentic/profiles/qwen36-27b-x1.sh` — production
  profile (modified at Stage 5 if gates pass)

## Risk areas + mitigation

| Risk | Mitigation |
|---|---|
| NCCL not packaged / not available on this host | Stage 0 verifies first; if missing, escalate to user before code work |
| NCCL init takes long (multi-second) | Initialize once at process start; don't re-init |
| NCCL all-reduce slower than current on PCIe 3.0 x8 (no NVLink yet) | Bench NCCL vs old path BEFORE enabling capture; if NCCL alone is slower, keep old path until NVLink lands AND just modify the capture gate to ignore REDUCE on the existing path (simpler fallback) |
| Graph capture still fails for other reasons | Stage 4 includes greedy parity; if capture introduces subtle errors, disable and investigate |
| MTP K=1 captured doesn't reach 34 t/s | Stage 5 closure documents the actual measurement; close negative |
| Existing reduce path is actually capture-safe and we did unnecessary NCCL surgery | Stage 0 includes a quick test: just disable the REDUCE gate, see if existing path captures successfully. If yes, NCCL is optional; we can stack it later for NVLink benefit |

## Open questions for user

1. **NCCL availability**: confirm NCCL is/will-be installed on the host?
   The CUDA 13 toolkit usually ships with libnccl.so. Quick check:
   `dpkg -l | grep nccl` or equivalent. If missing, this becomes a
   pre-Stage-0 dependency.

2. **Stage 0 quick-test first**: should I empirically test whether the
   existing reduce path is actually capture-safe (just modify the
   gate, no NCCL)? If yes, this could reveal NCCL is optional for the
   capture goal. If no, we go straight to NCCL implementation.

3. **Validation tolerance**: greedy parity at strict-equality (byte-diff)
   may fail because NCCL ring-reduce has different float-add order than
   `k_reduce_add_T`'s direct sum. Should I bound parity at, e.g., NMSE
   < 1e-7 instead of strict-equality?

## Verification

End-to-end gate at Stage 5:
- Build clean: `cmake --build /home/llm/yarn-agentic/ik_llama.cpp/build -j 32 --target llama-server`
- Greedy parity passes (NMSE bound TBD per Q3)
- nsys: cudaLaunchKernel count drops by ~5–10× (340k → ~50k for K=1)
- probe-verify-scaling.sh: MTP K=1 ≥ 34 t/s @ 256K X02
- Tree-K (K=2) IS a co-equal closure criterion (industry standard pattern;
  projects best under capture). Phase 41's K=2 negative was specific to
  uncaptured regime; Phase 43 capture inverts the math.

## What this plan does NOT propose

- Re-implementing tree-K from scratch (Phase 41 foundation already
  on `phase41-tree-foundation`; Phase 43 stacks on top)
- Phase 38 E async dispatch (proven negative on this hardware)
- MoE-specific work (model is dense; no MUL_MAT_ID gate trips)
- Moving away from TP-split-graph (TP wins by 2× compute capacity per M1
  measurement)
- Multi-process / multi-node NCCL

---

## Phase 43 measurement + close — NEGATIVE on this hardware

### What landed (Stages 0-2 — already implemented in upstream code)

- `GGML_NCCL` cmake option (line 100 of ggml/CMakeLists.txt) already
  defaults ON. Just needed `pacman -S nccl` to install the library.
- `reduce.cu:136-165` already has NCCL allreduce dispatch path,
  gated `#ifdef GGML_USE_NCCL`.
- `ggml-cuda.cu:281-296` already has `ncclCommInitAll` at backend
  creation. `common.cuh:766-768` has `nccl_coms[]` + `have_nccl`.
- Build + runtime: "NCCL main communicator initialized" confirmed.

### Stage 3 measurement (NCCL alone, no capture)

`scripts/probe-verify-scaling.sh` n_predict=128 256K X02:

| Config | NCCL-on tg | Pre-NCCL baseline (Agent X) | Δ |
|---|---|---|---|
| noMTP | 16.99 t/s | 21.77 t/s | **-22%** |
| MTP K=1 | 15.94 t/s | 21.31 t/s | **-25%** |
| MTP K=2 | 12.09 t/s | 15.26 t/s | **-21%** |

NCCL on PCIe 3.0 x8 (no NVLink) is materially slower than the
hand-rolled `cudaMemcpyPeerAsync + k_reduce_add_T` path. Matches
Agent 1's research warning: "Don't expect a perf win from NCCL on
this topology pre-NVLink." Magnitude exceeded expectation (-22% vs
expected ±5%).

### Stage 4 attempt — graph capture with NCCL

Modified the `GGML_OP_REDUCE` gate to allow capture when NCCL routes
the op (`ggml-cuda.cu:4487`).

**Result: argmax drift.** Capture engaged (10-12 cached graphs), but
K=1 accept rate dropped from 0.868 → 0.716 (-15pp). Same failure mode
Agent X observed in the gate-only speculative test.

Tried `cudaStreamCaptureModeThreadLocal` (per Agent 1's recommendation
for multi-GPU): server crashes during model warmup in
`ggml_cuda_op_mul_mat_cublas`. cuBLAS does host-allocations or
non-captured-stream operations that ThreadLocal rejects.

**Two stacked failure modes:**
1. Relaxed mode: capture works structurally but produces argmax drift
   (subtle FP-add reordering through captured graph)
2. ThreadLocal mode: cuBLAS is fundamentally not capture-safe in this
   codebase

### Closure: NEGATIVE on this hardware, NCCL kept for NVLink future

Both projected upsides (~+50% via capture) are blocked:
- NCCL alone is throughput-negative on PCIe x8 (NVLink would invert this)
- Graph capture has cuBLAS-class incompatibilities beyond the REDUCE
  gate — would need ggml-cuda backend rework to address

Source reverts:
- `ggml-cuda.cu:4487` gate fix reverted
- Capture mode reverted to `Relaxed`
- Working directory clean post-revert

**Recommendations:**
1. **Production**: rebuild with `-DGGML_NCCL=OFF` to restore the prior
   peer-copy + ring-reduce path until NVLink arrives. NCCL kept
   available behind the cmake flag for the post-NVLink swap.
2. **Post-NVLink**: re-measure. NCCL ring-allreduce on NVLink at ~50-100
   GB/s likely beats current peer-copy baseline (which is PCIe x8
   limited at ~7 GB/s). Capture remains broken; orthogonal to NCCL.
3. **For graph capture upside (+50% projected)**: requires
   ggml-cuda backend work to make cuBLAS dispatch capture-safe.
   Significant scope; out of Phase 43.

### Reusable artifacts preserved

- `scripts/probe-greedy-parity.sh`, `scripts/parity-nmse.py` — parity
  test infrastructure
- `specs/cuda_nccl_allreduce.allium` — contract spec for the NCCL
  reduce path
- `data/phase43-nccl-{scaling,capture}.runlog` — measurement evidence

PHASE41 + PHASE42 + PHASE43 = three negative results on MTP/tree-K
acceleration on TU102 sm_75 PCIe x8. The diagnostic story is now
complete: this hardware is launch-overhead bound + cross-device
bandwidth limited. Both classes of fix (capture, NCCL) are blocked
by deeper compatibility issues. Architecture is ready for NVLink
upgrade; software work is gated on ggml-cuda backend modernization.
