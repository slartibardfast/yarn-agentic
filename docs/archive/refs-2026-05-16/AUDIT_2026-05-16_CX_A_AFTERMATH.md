# Audit — Phase CX.A retraction aftermath

**Date**: 2026-05-16
**Trigger**: CX.A "fp32 frag_c_VKQ promotion" implementation found to be unnecessary; the 5888/6144 row-0 divergence signal was a TEST stride misread, not kernel non-determinism.

This document audits every NP-invariance / batch-shape claim in the repo against the actual binding evidence, to identify which claims are now invalidated, which are still well-bound, and what the real remaining production-determinism gap is.

---

## 1. The bug — class summary

`test-fattn-per-slot-kv-ncols-invariance.cpp` (pre-fix) extracted row-0 with stride `head_idx * Dv * n_tok + d`, asserting fa shape `[Dv, n_tok, N_HEADS_Q]`.

Actual: `ggml_flash_attn_ext_per_slot_kv` constructs the result with `ne = {v->ne[0], q->ne[2], q->ne[1], q->ne[3]}` (ggml.c:10284) → `[Dv, N_HEADS_Q, n_tok, N_SEQS]`. Head stride is `Dv`, token stride is `Dv * N_HEADS_Q`.

For `n_tok = 1` the wrong and correct strides coincide. For `n_tok > 1` the test read different heads' data than it thought. That produced a stable 5888/6144 = 23/24-heads "diff" signature that looked like inter-row kernel contamination but was pure layout misread — head 0 (h * stride = 0 for any stride) always matched; the other 23 heads always "diverged."

This is a CLASS of bug: **test extracts output buffer with the wrong tensor-stride assumption.** Any test that:
1. Calls a ggml op whose output `ne` ordering it implicitly assumes, and
2. Reads the output back via `ggml_backend_tensor_get` + computed offsets, rather than ggml's stride-aware view

is vulnerable to this class.

---

## 2. Test inventory — vulnerability assessment

Survey of every test in `tests/dflash-speculative/` that claims to bind NP- or batch-shape invariance, against this bug class.

| Test | What it claims | Output extraction | Vulnerable? | Current state |
|---|---|---|---|---|
| `test-fattn-per-slot-kv-ncols-invariance.cpp` | row-0 byte-identical across ne[1] ∈ {1,2,4,8} via FA-per-slot-kv | computed offsets into [Dv, N_HEADS_Q, n_tok] | **YES — bug found and fixed** | PASS post-fix (ik_llama.cpp 395496d4) |
| `test-fattn-per-slot-kv-dispatch-np-invariance.cpp` | output byte-identical across n_kv_max stride at N_TOK=1 | full buffer bit-compare, N_TOK=1 fixed | NO (N_TOK=1 → strides degenerate) | PASS (verified 2026-05-16) |
| `test-fattn-per-slot-kv-sm75.cpp` | legacy bespoke kernel correctness | calls `fattn_per_slot_kv_sm75_launch` directly with explicit buffer layouts; bypasses ggml | NO (not on production dispatch path; test layout is the API contract) | Information-only — kernel is NOT what production calls |
| `test-dflash-np-invariance.cpp` | DFlash drafter NP-invariance | does not use `ggml_flash_attn_ext_per_slot_kv` (grep'd) | NO | unchanged — outside production NP-determinism scope |
| `test-rmsnorm-batch-shape-invariance.cpp` (new CX.B) | RMSNorm row-0 invariance across ne[1] | full buffer read, layout `[N_EMBD, n_tok]` matches ggml_rms_norm's shape-preserving contract trivially (row 0 = first N_EMBD floats) | NO | PASS |
| `test-rope-batch-shape-invariance.cpp` (new CX.C) | RoPE token-0 invariance across ne[2] | computed offsets into [HEAD_DIM, N_HEADS_Q, n_tok, N_SEQS] — ggml_rope_ext preserves input shape, so layout matches input layout | NO (output shape == input shape, verified by `ggml_dup_tensor` in ggml.c:9171) | PASS |
| `test-cublas-pinned-shape-invariant.cpp` | F16-pinned GEMM shape invariance | (not re-audited; outside FA/RoPE/RMSNorm scope) | TBD | claimed PASS in Phase C closure |
| `test-mmq-q4-0-ar16-shape-invariance.cpp` | MMQ shape invariance for Q4_0_AR16 | (out of immediate scope) | TBD | claimed PASS in Phase A closure |
| `test-mmvq-q4-0-ar16-shape-invariance.cpp` | MMVQ shape invariance for Q4_0_AR16 | (out of immediate scope) | TBD | claimed PASS in Phase B closure |

**Audit verdict for the FA family**: one bug found, one fixed. The other FA tests are not vulnerable (different scenario shape).

**Audit recommendation for the other Phase A/B/C tests**: explicitly verify the output-extraction stride matches the ggml op contract. Lower priority than the immediate post-CX.A work, but a real follow-up — they're written in the same pattern and the same author bias could repeat.

---

## 3. Spec claims (specs/deltanet/fattn-per-slot-kv-sm75.md §15.*) — status

| Section | Claim | Falsified by CX.A retraction? | Status |
|---|---|---|---|
| §15.7 v1 | `wmma_f16-pb1<256,256,8,half>` DIVERGES on cache-leakage AND NP-cross | partial — fp16 KQ_acc_t IS associativity-non-deterministic for softmax warp_reduce_sum (real concern, separate from CX.A) | KEEP claim, but re-bind via fresh probe |
| §15.7 v2 | `wmma_f16-pb1<256,256,8,float>` MATCHES cache-leakage but DIVERGES NP-cross due to "fp16 frag_c_VKQ" | **FALSIFIED** — CX.A retraction proved frag_c_VKQ does NOT cause inter-row leakage. The NP-cross divergence attributed to it had a different cause | RETRACT this attribution |
| §15.10 | `multi_row_kernel` (per-row CTA, fp32 throughout) DIVERGES on cache-leakage | unaffected by CX.A | keep |
| §15.12 | Production routing recommendation | unaffected | keep |
| §15.13 | "Concurrent batched-decode byte-identity: not bound. The wmma fp16 `frag_c_VKQ` accumulator makes row 0's output subtly depend on other rows' content. Argmax flips after a few decode steps when slots are batched together." | **FALSIFIED** — was based on the mis-measured ncols-invariance test. The wmma_f16-pb1<...,float> kernel IS byte-identical at ne[1]>1 with correct measurement | RETRACT this section's central conclusion |
| §15.14 | --no-cont-batching probe partial improvement | unaffected by CX.A (server-level observation, not kernel) | keep |
| §15.15 | "Production route: wmma_f16-pb1<256,256,8,float>" + practical recommendation | RECONSIDER — the listed scope-non-deliverable ("concurrent batched-decode at NP>1") was based on §15.13 which is falsified | keep route, retract the "non-deliverable" framing |
| §15.16 | cb_eval per-layer probe: ALL 60+ layers byte-identical at NP=1 vs NP=4 SINGLE-PROMPT | unaffected by CX.A; independent measurement | keep — this is the strongest evidence for kernel-level NP-invariance |
| §15.17 | LLAMA_FATTN_STRICT_SEQUENTIAL_DECODE results: NP=2 GREEN; NP>=4 has non-deterministic across-runs anomalous slots | unaffected by CX.A | keep |

**Net effect of CX.A retraction on spec**:
- §15.13's "wmma frag_c_VKQ is the residual blocker" claim falsified.
- The production wmma_f16-pb1<256,256,8,float> kernel is more capable than §15.13/§15.15 claimed.
- Concurrent batched-decode IS deliverable from the kernel's perspective. The real residual must be elsewhere.

---

## 4. What the production harness actually measures

`scripts/test-production-np-determinism.sh`:
- Starts llama-server at NP=1, fixed prompt, captures decoded text.
- Restarts llama-server at NP={2,4,8}, sends N concurrent requests of the same prompt (one per slot), captures each slot's text.
- Compares each slot's text vs NP=1 baseline byte-identically.

The harness exercises the **full inference stack**: tokenizer, embedding, attention, FFN, sampler, KV cache, multi-GPU split, continuous batching scheduler. Any divergence at any layer flips the resulting token if it crosses an argmax decision boundary.

Last recorded result (per MEMORY.md 2026-05-15): 5/14 slots byte-identical, "shifting pattern across runs" at NP=4. **Pattern shifting across runs at the same code is RUN-TO-RUN non-determinism, not deterministic shape-dependence.**

Per spec §15.17, with `LLAMA_FATTN_STRICT_SEQUENTIAL_DECODE=1 --no-cont-batching`:
- NP=2: 2/2 byte-identical ✓
- NP=4: 3/4 byte-identical, 1 anomalous slot per run, **varies across runs**
- NP=8: 4/8 byte-identical, 4 anomalous, **varies across runs**

Per spec §15.16, with cb_eval per-layer probe at NP=1 vs NP=4 SAME-PROMPT SINGLE-SLOT: ALL 60+ layers byte-identical.

---

## 5. The actual residual — re-stated after audit

Three independent findings converge:

1. **Kernel-level NP-invariance is GREEN** for the production path: FA (CX.A fixed-test PASS + dispatch-NP-invariance PASS), RMSNorm (CX.B PASS), RoPE (CX.C PASS). All per-row CTAs algebraically; tests confirm bit-exactly.

2. **Layer-level NP-invariance is GREEN** for SINGLE-PROMPT processing: §15.16 cb_eval probe shows all 60+ residual streams byte-identical between NP=1 and NP=4 server configs when only slot 0 is active.

3. **Concurrent multi-slot decode at NP>=4 has run-to-run non-determinism**: §15.17 shows 1/4 slot diverges, BUT WHICH SLOT VARIES across runs of the same code. This is the actual problem.

**The CX.A retraction strengthens this picture**: the kernel is more robust than we thought, so the residual is NOT in the per-op math. The residual is **in the cross-slot ordering of concurrent dispatch** — atomic ordering, CUDA stream interleaving, allocator address dependence, or graph cache eviction.

Per-op batch-shape unit tests (the path I was on before the audit) **cannot catch this class of bug**. They probe deterministic shape-dependence; they can't probe scheduler timing.

---

## 6. What's needed instead — a coherent test+trace plan

To localize the run-to-run NP>=4 anomaly, we need:

### 6a. Op-level traces from the concurrent harness — find first-divergent op

`scripts/probe-first-divergent-layer.sh` already exists for SAME-PROMPT comparison (NP=1 vs NP=4 single-slot). It found all layers GREEN. We need an analogous probe that captures **per-op outputs from a concurrent run that's known to diverge**, comparing the anomalous slot's per-op outputs vs the same slot in a clean run.

Workflow:
1. Run NP=4 concurrent harness 10x to identify which slot becomes anomalous (varies).
2. For the anomalous slot's run, capture cb_eval per-layer outputs.
3. For a clean run (all slots match), capture the same.
4. Diff per-layer to find first divergence.

This is a layer-granularity TRACE, not a unit test.

### 6b. Op-level intermediate capture inside FA / RMSNorm / matmul

If first-divergent-layer points to layer X, we then need to drill INSIDE layer X to find which op. The cb_eval mechanism captures only top-level named nodes (`l_out-<il>`). Internal intermediates (post-RMSNorm-1, post-QKV-proj, post-RoPE, post-FA, post-FFN-down) need additional named tags. The existing `h_pre_norm` capture is a precedent.

### 6c. Kernel-side instrumentation — atomics + reduce trees

Once an op is localized, instrument that kernel to print:
- The order in which thread blocks complete (if atomicAdd is involved).
- The values feeding each cross-warp reduction.
- Any cache state read at kernel entry.

This is `ncu`-territory or printf-instrumented kernel territory.

### 6d. CUDA graph cache state probe

Per §15.14: "NP-invariant CUDA graph topology" was flagged as needed. Confirm whether the same prompt at NP=4 (concurrent) generates the SAME cuda graph hash as at NP=1, and whether the graph cache evicts differently across runs. The `cuda-graph-probe` directory exists for this.

### 6e. Allocator state probe

`ggml_cuda_pool_alloc` returns different addresses across calls. If a kernel's behavior depends on pointer alignment (mod 128, mod 256), allocator state would inject run-to-run non-determinism. Probe by dumping allocator state at the start of each anomalous slot's compute.

---

## 7. Why my "more unit tests" approach was misdirected

After CX.A retraction I started writing batch-shape probes for RMSNorm, RoPE, CPY, Hadamard. They are all PASS by algebraic construction (per-row CTAs) — they would have all PASSED without informing the residual gap. The audit shows the residual is RUN-TO-RUN under concurrent NP>=4 — a different class of bug entirely. Unit tests of one op at fixed shape against itself across batch dim cannot catch ordering / timing / allocator effects.

The right next steps are §6a-§6e above — traces and instrumentation, not more shape probes.

---

## 8. Concrete proposed deliverables (next session)

1. **TRACE-1**: Adapt `probe-first-divergent-layer.sh` for CONCURRENT NP=4 anomalous-slot capture. Run 10×, identify run with single anomalous slot, capture per-layer outputs from anomalous slot, diff vs clean-run baseline. Output: first-divergent-layer name.

2. **TRACE-2**: At the first-divergent layer, add named-output captures for the intra-layer intermediates (`h_pre_norm`, `attn_qkv`, `attn_post_rope`, `attn_post_fa`, `attn_post_proj`, `ffn_norm_in`, `ffn_post_gate`, `ffn_post_up`, `ffn_post_down`). Re-run, find first-divergent intra-layer op.

3. **TRACE-3**: At the first-divergent op, dump (i) the input tensor, (ii) the output tensor, (iii) the launch geometry, (iv) the cuda graph hash, (v) the input tensor pointer alignment, in BOTH the anomalous and clean runs. Compare.

4. **TRACE-4**: From TRACE-3 output, identify whether the divergence is:
   - Numerical (values differ at some bit position) → some op has cross-thread non-associative reduction order
   - Geometric (launch geometry differs) → dispatch heuristic depends on global state
   - Pointer (alignment differs) → allocator-state-dependent kernel behavior
   - Graph (different cuda graph hash) → graph cache topology effect

5. **PATCH** is then targeted at the localized cause; binding test is byte-identity reproduction of the previously-anomalous concurrent run.

The /loop should pause unit-test generation until traces from TRACE-1 are in hand. The user's "be systematic" instruction is satisfied by §2 inventory above + the trace plan, not by more shape probes.

---

## 9. Status as of this audit

- Phase A/B/C CLOSED — re-verify their test extractions don't have the CX.A bug class (lower priority follow-up).
- Phase CX.A RETRACTED — kernel was correct, test was wrong; fixed.
- Phase CX.B GREEN — RMSNorm batch-shape invariant (expected, confirmed).
- Phase CX.C GREEN — RoPE batch-shape invariant (expected, confirmed).
- Phase CX.1 CLOSED — DeltaNet threads_per_block pin still relevant.
- Phase CX.D / CX.E / CX.F (cb_eval, CPY, Hadamard probes) — **DEFER**. Unlikely to catch the residual per §6.
- Production NP-determinism gap: localized to **run-to-run non-determinism at concurrent NP>=4**, per §15.17. Cause not yet identified at op level. Path forward = TRACE plan §6a-§6e.

**No actionable kernel patches until TRACE results land.**
