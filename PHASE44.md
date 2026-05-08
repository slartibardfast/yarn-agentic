# Phase 44 — CUDA Graph Capture Stability for Multi-GPU TP Decode

## Goal

Make CUDA graph capture deliver actual launch-overhead reduction on the
existing multi-GPU TP setup (Qwen 3.6 27B dense, 2× RTX 6000 TU102 sm_75
PCIe 3.0 x8). Phase 43 measurement evidence:

- Capture infrastructure exists (`ggml-cuda.cu:4470-4630`, `cpy_dest_ptrs`
  indirection from prior Phase 35)
- The `GGML_OP_REDUCE` gate at line 4487 correctly keeps REDUCE splits
  out of capture (each REDUCE is its own n_nodes=1 split per Agent 2)
- Compute-only splits should capture cleanly — REDUCE is between, not
  within
- BUT: removing the gate (Agent X test, our Stage 4 with NCCL) produces
  identical -14 to -15pp K=1 accept drift regardless of whether NCCL
  is involved. Drift is NOT about REDUCE/NCCL.

## Reframed problem statement

The per-cycle update_required check (line 4632, 4977) triggers `update_required`
every cycle, causing graph rebuilds. After 4 consecutive updates the
capture self-disables (line 4977-4986). The failures we observed —
captured replay producing different output than uncaptured fresh graph
— happen during the 1-3 cycles where capture is engaged but rebuilding,
not during steady-state replay (which never reaches steady state).

Three plausible per-cycle property changes that trigger rebuilds:
1. **KV cache cell positions** (`cells[].pos`) advance each cycle
2. **Sampled token data pointer** (`slot.sampled` lives on a per-cycle batch buffer)
3. **MTP scratch tensor addresses** (per_step_ssm/qkv/shadow buffers cycle-managed)

The `cpy_dest_ptrs` indirection (line 4539) handles per-token CPY destinations
but doesn't cover other classes of per-cycle change.

## Approach

Three stages, each with empirical gating before proceeding:

### Stage 0 — Confirm the rebuild-loop hypothesis

Add diagnostic logging to `is_cuda_graph_update_required` (line 4632)
and `ggml_graph_node_has_matching_properties` (line 4579) to print:
- Which node mismatches when rebuild triggers
- Which property mismatches (data ptr, ne, nb, src ptrs)
- Cycle counter

Run probe-verify-scaling.sh with this instrumentation. Identify the
specific property pattern that flips every cycle.

Output: a list of "this property of this op changes every cycle, here's
the change pattern."

Budget: ~5-10k tokens. Simple instrumentation + analysis.

### Stage 1 — Extend indirection to cover the identified properties

For each cycle-changing property identified in Stage 0:
- Add a new indirection mechanism (modeled after `cpy_dest_ptrs`)
- Patch the captured graph at replay time with the current cycle's value
- Skip these properties in `ggml_graph_node_has_matching_properties` so
  rebuild doesn't trigger

Key likely targets:
- KV cache cell pointers (per-cycle pos advance)
- Sampled token batch entry
- per_step_ssm shadow buffer (Phase 36/37 infrastructure)

Budget: ~25-40k tokens. Touches `ggml-cuda.cu:4564-4630` (property tracking),
`set_inputs` paths in `llama.cpp` for cycle-changing inputs, and adds
new indirection-style mechanisms.

### Stage 2 — Validate stability + measure

Re-run probe-verify-scaling.sh + nsys-profile-tree.sh:
- nsys: `cudaLaunchKernel` count drops dramatically (from ~340k to <50k for K=1)
  — the actual capture-saved launches showing up
- probe: tg ≥ Phase 43 baseline + meaningful improvement
- greedy parity passes (NMSE < 1e-7 against pre-Phase-44 baseline)

Decision tree:
- If captured tg ≥ +30% over baseline AND parity holds: production swap consideration
- If captured tg ≥ +10% over baseline AND parity holds: ship as default-on
- If parity fails OR tg <+10%: close negative; document the indirection
  mechanism for future use

Budget: ~10-15k tokens.

## Critical files

- `/home/llm/yarn-agentic/ik_llama.cpp/ggml/src/ggml-cuda.cu:4470-4630` —
  capture compatibility check + property tracking
- `/home/llm/yarn-agentic/ik_llama.cpp/ggml/src/ggml-cuda/graph.cuh` —
  ggml_cuda_graph struct (where new indirection lives)
- `/home/llm/yarn-agentic/ik_llama.cpp/src/llama.cpp` set_inputs paths
  — cycle-changing input plumbing
- `/home/llm/yarn-agentic/scripts/probe-verify-scaling.sh` — measurement
  harness
- `/home/llm/yarn-agentic/scripts/nsys-profile-tree.sh` — capture
  validation via launch count

## Risks

| Risk | Mitigation |
|---|---|
| Stage 0 finds property mismatches that aren't cleanly indirectable (e.g., op_params change semantically each cycle) | Stage 0 closes negative; capture stability isn't achievable without deeper rework |
| Multiple unrelated property flips per cycle | Stage 1 budget grows; consider closing at partial coverage + measuring partial wins |
| cuBLAS-in-capture (the ThreadLocal mode crash from Phase 43) — even with stable property tracking in Relaxed mode, ThreadLocal might still fail. We'd be stuck on Relaxed which has subtly worse semantics | Stay on Relaxed mode; verify greedy parity carefully. If Relaxed produces drift even with stable properties, the ggml-cuda backend's cuBLAS path needs rework — Phase 45 territory. |
| Indirection mechanisms add per-cycle overhead that exceeds the launch-overhead savings | Measurement at Stage 2 closes the question empirically |

## What this plan does NOT propose

- Multi-device-coordinated capture (the architectural issue I'd thought
  about earlier turned out not to be the actual bottleneck — REDUCE is
  already correctly excluded)
- NCCL replacement (Phase 43 closed; NCCL on PCIe x8 is throughput-negative)
- ggml-backend.cpp scheduler changes (the per-split dispatch is already
  correct; the issue is in property tracking within ggml-cuda backend)

## Verification

Stage 0 outputs (diagnostic) → Stage 1 outputs (capture stability) →
Stage 2 outputs (production-relevant measurement).

Each stage is gated; if Stage 0 reveals the issue isn't fixable by
indirection, the whole workstream closes at Stage 0 evidence rather
than committing to Stage 1 implementation.

## Rationale for this scope

PHASE44 reframes from "multi-device capture" (the wrong diagnosis) to
"capture stability across decode cycles" (the real issue). Phase 43's
data — identical drift magnitude with and without NCCL — pins the
problem to within a single device's compute capture, not at the
cross-device boundary. This is a tractable scope (~40-65k tokens
depending on indirection complexity) vs the originally-feared
multi-device scheduler rework.

Phase 43 NCCL infrastructure stays preserved on the cmake flag for
post-NVLink revisit; orthogonal to this phase.

---

## Stage 0 results — confirmed + sharpened diagnosis

Diagnostic instrumentation (env-gated `LLAMA_PHASE44_DIAG`) added to
`ggml-cuda.cu` and ran probe with multi-GPU TP at 4K context, 16-token
generation. Output:

**Per-split shape (first 12 cycles):**
```
cycle 0  device=0  n_nodes=50  first=FUSED_RMS_NORM  has_reduce=0
cycle 1  device=1  n_nodes=49  first=FUSED_RMS_NORM  has_reduce=0
cycle 2  device=1  n_nodes=1   first=REDUCE         has_reduce=1  -> disabled
cycle 3  device=0  n_nodes=3   first=FUSED_RMS_NORM  has_reduce=0
cycle 4  device=1  n_nodes=4   first=FUSED_RMS_NORM  has_reduce=0
cycle 5  device=1  n_nodes=1   first=REDUCE         has_reduce=1  -> disabled
... (pattern repeats)
```

**Aggregate over 6555 dispatch calls:**
- has_reduce=1 (REDUCE-only splits): 2176
- has_reduce=0 (compute-only splits): 4379
- "disabled by check_node_graph_compatibility": 2176 (= REDUCE count, gate works)
- "update_required=1" on compute splits: ~50% of cycles → rebuild loop

**Property mismatch breakdown (from `LLAMA_PHASE44_DIAG_PROPS=1`):**
- 18× `FUSED_RMS_NORM` output `node_address` mismatch
- 16× `FUSED_RMS_NORM` `src[1]_data_mismatch`
- 2× `FUSED_RMS_NORM` `src[0]_data_mismatch`

All on `FUSED_RMS_NORM`. Addresses cycle predictably:
- Output: `0x7f6454000000` ↔ `0x7f6454005080` (alternating slots)
- src[1]: `0x7f646b498000` → `0x7f646b400000` → `0x7f646b49d000` …

**Pattern**: ggml's per-cycle scratch allocator gives different scratch
slots each call. The captured graph captures one set; replay at next
cycle needs a different set. Property mismatch → rebuild → after 4
consecutive rebuilds, capture self-disables permanently.

This is the production state today: capture is technically enabled but
silently disabled at runtime by the rebuild-loop self-protect.

### Stage 1 candidate fixes (ordered by cheapest first)

1. **Investigate `cudaGraphExecUpdate` failure path** (~5-15k). The
   line 4681 path already exists; if it succeeded for "only addresses
   moved" cases, we wouldn't need property-stable graphs. Check why
   it's failing in practice — could be a small fix.

2. **Allocator stability** (~10-20k). Make ggml's compute scratch
   give deterministic slots across cycles. If outputs always land at
   same address, captured graph stays valid. Smaller code change but
   touches core allocator semantics; some cycle-to-cycle variation
   may be intentional.

3. **Indirection extension** (~20-40k). Add cpy_dest_ptrs-style
   indirection for FUSED_RMS_NORM (and any other ops that get
   identified as moving). Use `cudaGraphExecKernelNodeSetParams` at
   replay time to patch addresses. Most general; biggest scope.

### Stage 1 recommendation

Try option 1 first (investigate cudaGraphExecUpdate). If the existing
update path is fixable, the rest of this phase becomes trivial.

If option 1 yields nothing actionable, fall back to option 2
(allocator stability) — that's the upstream-aligned fix.

Option 3 only if 1 and 2 both fail.

---

## Stage 1 investigation — option 1 ruled out

Read the capture-replay flow at `ggml-cuda.cu:4720+`:

When `update_required=true`:
1. **Re-execute all kernels live** (line 4734-4760) — these run inside
   the cudaStreamBeginCapture region, so they're being captured. But
   they ALSO execute as live launches.
2. End capture (line 4772). New graph object recorded.
3. `cudaGraphExecUpdate` (line 4842) tries to update existing exec.

So the "savings" from capture only kick in when `update_required=false`
and we replay via `cudaGraphLaunch` instead of executing live. With
`update_required=true` every cycle, kernels execute live regardless.

cudaGraphExecUpdate ISN'T failing — it's working fine. The problem is
that the work is already done before Update is called. Update is a
side effect, not a performance lever.

The cache architecture contributes: `topology_key` at line 4401 hashes
only `(n_nodes, op-sequence)` — the SAME for both alternating slot
states. Both states fight for one cache slot; whichever was last
captured wins; the other rebuilds.

**Option 1 (cudaGraphExecUpdate) is ruled out** — the path works, but
it's positioned after the live execution.

To get capture savings, `update_required` must be **false** on most
cycles. This requires either:
- **Option 2 (allocator stability)**: scratch allocator gives the same
  slot for the same op slot across cycles. Properties match → replay.
- **Option 3 (multi-slot cache key)**: extend `topology_key` to include
  a discriminator that distinguishes alternating slot states.
  Alternating states get separate cache entries; both stable on
  re-visit; both replay on hit.

Option 2 is upstream-aligned but invasive. Option 3 is local but
requires identifying the right discriminator (e.g., low bits of the
first node's data address) — non-obvious without understanding the
allocator's state machine.

### Honest assessment after Stage 0 + 1 investigation

Phase 44's "capture stability" is achievable, but the cheapest fix is
~10-20k tokens (Option 2 or 3) AND requires understanding ggml's
allocator semantics that are upstream-class. The phase has bottomed
out into "ggml-cuda backend modernization" territory — same conclusion
the Phase 43 close gestured at.

Cumulative: Phase 41 + 42 + 43 + 44 all converge on **the ggml-cuda
backend was designed pre-graph-capture for the multi-GPU TP path, and
modernizing it is a significant upstream-class workstream.**

### Decision point

Three paths from here:

A. **Continue — implement Option 2 or 3 in PHASE44 Stage 1.** ~10-20k
   for partial fix; ~30-50k if it cascades into more deeply-rooted
   upstream changes. Real probability of negative outcome (Stage 2
   measurement still doesn't bind +10% gate).

B. **Park PHASE44 with Stage 0+1 findings preserved.** Document the
   diagnosis; revisit when ggml-cuda backend modernization (which is
   needed for many things, not just capture) becomes a separate
   workstream.

C. **Pivot away from capture entirely.** Different optimization angle
   — e.g., reduce the per-token kernel launch count via fusion in the
   model itself, KV cache compression, draft-model spec decode (a
   separate small draft on a single GPU avoiding TP entirely).

Recommendation: **B + C**. The diagnostic story is now complete on
this axis (4 phases of negative results converging on the same
architectural conclusion). Putting more work into capture without
the broader ggml-cuda modernization is unlikely to land.
