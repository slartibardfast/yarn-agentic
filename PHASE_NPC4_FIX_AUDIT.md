# PHASE_NPC4_FIX_AUDIT — deep audit & fix plan for `ggml_cuda_up_gate_unary`

**Branch**: `production/2026-q2-next`
**Predecessor**: `PHASE_NPC_HANDOVER.md` (NPC.4 LOCALIZED status)
**Status**: Bug located. Fix path planned. Not yet implemented.

## Section 1 — Architectural correction

Qwen 3.6 27B as produced for this host (GGUF
`qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf`) is **NOT an MoE
model**. GGUF metadata shows `general.architecture = qwen35` and layer-2
tensor inventory shows DENSE FFN weights:

```
blk.2.attn_qkv.weight           # DeltaNet attention
blk.2.attn_gate.weight
blk.2.ssm_*                     # DeltaNet recurrent state
blk.2.ffn_up.weight             # DENSE FFN (no _exps suffix)
blk.2.ffn_gate.weight
blk.2.ffn_down.weight
blk.2.attn_norm.weight
blk.2.post_attention_norm.weight
```

Graph builder is `build_qwen35()` (not `build_qwen35moe()`), defined at
`src/graphs/build_qwen35.cpp:127`. The layer loop calls `llm_build_ffn`
(dense), not `llm_build_std_moe_ffn`. The 27B param count is achieved by
a wide dense FFN, not by routed experts.

This corrects the "shared expert" framing in the prior session. The
diverging tensor `ffn_up_gate-2` is the **dense fused up+gate matmul**,
named at `src/llama-build-context.cpp:839` from
`ggml_fused_up_gate(ctx, up, gate, cur, unary_op)` under the
`cparams.fused_up_gate` branch.

Layer shapes (empirical, from captures):
- `ffn_norm-2`: 20480 bytes per slot = 5120 floats = `n_embd`.
- `ffn_up_gate-2`: 17408 floats per slot = 2 × 8704 = `2 × n_ff`. So
  `n_ff = 8704` for the dense Qwen 3.6 27B FFN.

## Section 2 — Empirical localization

Walked every cb_eval-visible tensor at layer 2, decode-step 0,
NP=1 vs NP=2 (see `scripts/compare-intra-layer.py`):

```
order  name                       sz1     szk      verdict     first   max|Δ|
0-32   <all DeltaNet path>         …       …       IDENTICAL   —       —
33     ffn_norm-2                 20480   40960    IDENTICAL   —       —
34     ffn_up_gate-2              69632  139264    DIFFERS     1       1.490e-08
35     l_out-2                    20480   40960    DIFFERS     0       6.855e-06
```

Conclusion: input to `ffn_up_gate` is bit-identical across NP=1 and
NP=2; the matmul itself injects one ULP of drift in element 1, which
then amplifies through `ffn_down` + residual + the next layers.

## Section 3 — Where the bug lives (code citation)

File: `ggml/src/ggml-cuda.cu`
Function: `ggml_cuda_up_gate_unary` (lines 3692–3763).

This is the CUDA backend handler for `GGML_OP_FUSED_UP_GATE`. It
branches on `src1->ne[1]` (= number of tokens in the ubatch = NP for a
single-token decode):

```cpp
if (src1->ne[1] <= 8) {
    quantize_row_q8_1_cuda(...);                       // src1 → Q8_1

    if (src1->ne[1] == 1 && src0_1->type == src0_2->type) {
        // === NP=1 PATH ===
        ggml_cuda_op_fused_mul_mat_vec_q_id(...);      // ONE kernel: up+gate+silu+mul fused
        return;
    }

    // === NP=2..8 PATH ===
    ggml_cuda_op_mul_mat_vec_q(ctx, src0_1, src1, …, dst_up,    …);  // up only
    ggml_cuda_op_mul_mat_vec_q(ctx, src0_2, src1, …, dst->data, …);  // gate only
    // [post-fix step at line 3759]
} else {
    // === NP=9+ PATH ===
    if (ggml_cuda_should_use_mmq(...)) {
        quantize_mmq_q8_1_cuda(...);
        ggml_cuda_op_mul_mat_q(src0_1, …);             // MMQ up
        ggml_cuda_op_mul_mat_q(src0_2, …);             // MMQ gate
    } else {
        ggml_cuda_mul_mat(src0_1, …);                  // cuBLAS up
        ggml_cuda_mul_mat(src0_2, …);                  // cuBLAS gate
    }
}

ggml_fused_mul_unary(…);                               // silu(g)*u (NP=2+ paths only)
```

**Three distinct dispatch regions** keyed on n_tokens:
| n_tokens | Path | Kernel(s) used |
|---|---|---|
| 1 | "fast-TG fused" | `k_fused_mul_mat_vec_q<type, 1, nwarps>` (single kernel) |
| 2–8 | "non-fused MMVQ" | `k_mul_mat_vec_q<type, N, nwarps>` × 2 + `fused_mul_silu_f32` |
| 9+ | "MMQ" or "cuBLAS" | `k_mul_mat_q<type, ...>` × 2 + `fused_mul_silu_f32` |

Empirically the NP=1 ↔ NP=2 boundary leaks 1 ULP. We have not yet
measured NP=2 ↔ NP=4 ↔ NP=8 or any of the 8 ↔ 9 crossover; both are
suspect under the same logic.

## Section 4 — Theoretical bit-identity check for the non-fused MMVQ path

`k_mul_mat_vec_q<type, ncols_y, nwarps>` (file
`ggml/src/ggml-cuda/mmvq-templates.cuh:70`) accumulates per-thread
partial sums:

```cpp
for (int kbx = tid/(qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
    for (int j = 0; j < ncols_y; ++j) {
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            tmp[j][i] += vec_dot_q_cuda(vx, &y[j*blocks_per_col_y + kby], (row0+i)*blocks_per_row_x+kbx, kqs);
        }
    }
}
```

For `ncols_y ∈ {1, 2}`:
- `rows_per_cuda_block = ncols_y < 4 ? 1 : 2` → both = 1.
- `blocks_per_iter = vdr * nwarps * WARP_SIZE / qi` — independent of ncols_y.
- For `j = 0`: kbx range and stride are identical → SAME partial sum
  per thread.
- Cross-warp reduction (`tmp_shared[l][j][i][threadIdx.x]`) — for j=0,
  same writes/reads regardless of `ncols_y`.
- Final `warp_reduce_sum` — same.

**In theory the j=0 (slot-0) output is bit-identical between
`ncols_y=1` and `ncols_y=2` instantiations.** Same for any `ncols_y`
that keeps `rows_per_cuda_block=1`, i.e. all `ncols_y < 4`. For
`ncols_y ∈ {4..8}` `rows_per_cuda_block=2` and the per-thread state
shape changes — slot-0 likely still bit-identical but needs verification.

This means the bit divergence we see at NP=1 vs NP=2 is **not** from
running the same kernel template at different `ncols_y` — it's from
running **two different kernel families** entirely (`k_fused_mul_mat_vec_q`
vs `k_mul_mat_vec_q × 2 + fused_mul_silu_f32`).

The fused kernel additionally applies `silu(g) * u` inside the matmul
kernel (line 252 of `mmvq-templates.cuh`):

```cpp
g = g/(1 + expf(-g));
g = min(g, limit);
r = max(-limit, min(limit, u)) * g;
```

The non-fused path stores raw `u` and `g` to memory then runs
`fused_mul_silu_f32` (file `unary.cu:64`) which does the same algebra:

```cpp
float g = x[i] / (1.0f + expf(-x[i]));
g = min(g, limit);
dst[i] = g * max(-limit, min(limit, y[i]));
```

These two snippets are **algebraically identical** but live in
**different compiled cubins**. `expf` is allowed to round
slightly differently between compilation units, especially with
PTX-level fast-math (which CUDA enables by default for `expf`). The
1-ULP drift in element 1 is a single rounding of `expf` differing
between the in-kernel call and the standalone `fused_mul_silu_f32`
call.

## Section 5 — Why this was missed by prior determinism bakes

CY.F.17 (MMQ stream_K) baked deterministic dispatch for the **MMQ**
path used in attention GEMMs at prefill and large-batch decode. It
covers the NP=9+ region's MMQ branch but not:

- The NP=1 fast-TG fused-mmvq path
- The NP=2..8 non-fused mmvq pair-of-calls
- The NP=9+ cuBLAS fallback when MMQ isn't supported

A.1′ (FA per-slot-KV) baked the attention output across slots, but the
divergence we're chasing is **post-attention** (every DeltaNet
attention through layer 2 is byte-identical for slot-0). FA bake is
correctly orthogonal.

Singlewarp / delta-net `use_256` cover DeltaNet kernels which are
demonstrably bit-identical here.

cuBLAS-workspace + TF32-off covers the standard `ggml_cuda_mul_mat`
cuBLAS calls. Doesn't help if the dispatch routes around cuBLAS into
the mmvq family.

**Gap:** there is no NP-invariance audit on the `ggml_cuda_up_gate_unary`
n_tokens-dependent dispatch.

## Section 6 — Wider audit — other ops with n_tokens-dependent dispatch

Same dispatch pattern is repeated for several ops. Each needs an
NP-invariance check.

1. **`ggml_cuda_up_gate_unary`** (lines 3692–3763, the one we caught).
   `GGML_OP_FUSED_UP_GATE`. Fires for every dense layer's fused
   up+gate. Currently `64 firings / decode step` on this model.

2. **`ggml_cuda_moe_up_gate_unary`** (called at line 4098).
   `GGML_OP_MOE_FUSED_UP_GATE`. Not on this model (dense FFN), but
   would fire for any future MoE variant. Likely has the same kind of
   shape-dependent dispatch internally.

3. **`ggml_cuda_mul_mat_id`** (line 2986). Already has at least three
   distinct paths gated on `src1->ne[1]`, `src1->ne[2]`, etc.: fast TG
   MMVQ-id, MMQ-id, fallback. Not in critical path for this dense
   model but matters for MoE.

4. **`ggml_cuda_mul_mat`** (called for non-id GEMMs throughout). For
   `n_tokens=1` decode, may go through MMVQ; for `n_tokens=2..8` may
   stay MMVQ but with different `ncols_y`; for `n_tokens>8` switches
   to MMQ or cuBLAS. Inspect for the same fused/non-fused crossover.

5. **`ggml_fused_mul_unary`** (file `unary.cu`). Standalone unary, fired
   in the non-fused path of (1). Bit-identical with respect to NP
   because it's per-element pure, but its inputs come from differently
   compiled kernels — making the COMBINATION of MMVQ+unary non-equal
   to the in-kernel fused variant.

Action: each op listed above gets an intra-layer capture pair at NP=1
vs NP=2 verifying slot-0 byte-identity, on its own input.

## Section 7 — Candidate fixes, ranked by complexity

### A. Per-slot loop over the fused fast-TG path (RECOMMENDED)

**Cost**: ~10 lines in `ggml_cuda_up_gate_unary`.

**Idea**: do what `ggml_cuda_moe_up_gate_unary` already does at line
3252 — loop over `iy = 0..src1->ne[1]-1` and call the fused single-token
kernel per slot. Drops the per-NP branch entirely; all NP values use
the SAME `k_fused_mul_mat_vec_q<type, 1, nwarps>` instantiation per slot.

```cpp
if (src1->ne[1] <= some_threshold && src0_1->type == src0_2->type) {
    quantize_row_q8_1_cuda(...);
    auto local_dst = *dst;
    local_dst.ne[1] = 1; local_dst.nb[2] = local_dst.nb[1];
    auto local_src1 = *src1;
    local_src1.ne[1] = 1; local_src1.nb[2] = local_src1.nb[1];
    for (int iy = 0; iy < src1->ne[1]; ++iy) {
        local_src1.data = src1_quantized.get() + iy * src_1_ddq_size;
        local_dst.data  = (char *)dst->data + iy * dst->nb[1];
        ggml_cuda_op_fused_mul_mat_vec_q_id(ctx, src0_1, &local_src1, /*ids=*/nullptr, &local_dst, …);
    }
    return;
}
```

**Pros**: minimal kernel change. Reuses an already-deterministic
single-token kernel for all NP values. NP=1 and NP=N take the EXACT
same execution path — single kernel launches in sequence, each
producing exactly the byte pattern NP=1 produces today.

**Cons**: N kernel launches instead of 1 for NP=N. At NP=8 that's 8
launches vs 1; latency overhead per FUSED_UP_GATE op. Production
NP=4..8 on this host may pay a few % decode latency. Measurable. To
quantify before commit.

### B. Drop the fused fast-TG path; always non-fused

**Cost**: delete the `if (src1->ne[1] == 1 && ...)` branch entirely;
fall through to the two-call mmvq path. Verify `ncols_y=1`
mmvq-non-fused matches `ncols_y=N` mmvq-non-fused for slot-0.

**Pros**: simpler than (A); same kernel template (`k_mul_mat_vec_q`)
for all NP values; no extra launches.

**Cons**: removes a known-fast NP=1 single-kernel optimization.
Decode latency at NP=1 likely regresses (one kernel → three kernels:
up, gate, fused-mul-silu). Worse for the common case.

### C. Always route through MMQ regardless of n_tokens

**Cost**: change the `if (src1->ne[1] <= 8)` gate to always call MMQ.

**Pros**: covered by CY.F.17 stream_K determinism bake; one path for
everything.

**Cons**: MMQ is slower at very small ncols_y than MMVQ; this regresses
NP=1 substantially. Not recommended unless (A) and (B) both fail.

### D. Make the fused and non-fused outputs bit-identical via a
**shared device function**

**Cost**: refactor `silu_with_limit` into a single `__device__`
function that both `k_fused_mul_mat_vec_q` and the standalone
`fused_mul_silu_f32` call. Force the compiler to use the same `expf`
PTX for both. Cubin still differs but same intrinsic call.

**Pros**: in theory keeps fused fast-path for NP=1.

**Cons**: PTX-level rounding can still differ — sharing source code
doesn't guarantee bit-identical compiled output. Speculative and
fragile.

### Decision

Land **Option A** (per-slot loop over the fused fast-TG kernel). It
trades a few % latency at NP>1 for cross-NP bit-identity. Latency cost
is bounded and measurable; correctness gain is absolute. After A
lands, measure latency cost; if intolerable, A→B is also viable.

## Section 8 — Verification plan

After implementing the fix:

1. **Per-op intra-layer-2 verify** (cheap, immediate):
   ```bash
   /tmp/run-npc4-capture.sh 1 /opt/models/yarn-audit-data/npc4-fix-np1
   /tmp/run-npc4-capture.sh 2 /opt/models/yarn-audit-data/npc4-fix-np2
   python3 scripts/compare-intra-layer.py \
     /opt/models/yarn-audit-data/npc4-fix-np1 \
     /opt/models/yarn-audit-data/npc4-fix-np2 \
     --phase decode-0 --layer 2
   ```
   PASS: `ffn_up_gate-2` reads `IDENTICAL`. `l_out-2` reads `IDENTICAL`.

2. **Whole-graph per-layer verify** (medium): run `--tensors l_out
   --layers all` at NP=1 vs NP={2,4,8}; expect every layer's slot-0
   `l_out` to be bit-identical.

3. **Wider op audit verify** (broader): with `--all-in-layer --layers
   all`, find any tensor at any layer whose slot-0 still diverges.
   Repeat fix for the next offender.

4. **Production harness** (final): rerun
   `scripts/test-production-np-determinism.sh` and confirm cross-NP
   byte-identity at the server level.

5. **Latency measurement** (must-do before declaring done): bench
   `llama-bench` at NP=1, 2, 4, 8 with the fix vs without. Document
   the delta. Acceptance criterion: ≤ 3% decode-throughput regression
   per `feedback_determinism_must_co_optimize_perf.md`.

## Section 9 — Step plan

- [ ] **F.1** Implement Option A in
  `ggml/src/ggml-cuda.cu:3692-3763`. Single edit: replace the
  `if (src1->ne[1] == 1 …)` branch with a per-slot loop calling
  `ggml_cuda_op_fused_mul_mat_vec_q_id` `Ny` times. Build.

- [ ] **F.2** Run the verify in §8.1. Expect `ffn_up_gate-2` and
  `l_out-2` to flip to `IDENTICAL`.

- [ ] **F.3** Run the verify in §8.2. Find any remaining diverging
  layer/op. If clean, proceed to F.4. If not, identify the next
  divergence-producing op and add its fix to this PHASE doc as F.3a.

- [ ] **F.4** Latency bench (§8.5). Cap regression ≤ 3%.

- [ ] **F.5** Production harness verify (§8.4). Expect cross-NP
  byte-identity at server level.

- [ ] **F.6** Bake the fix into a permanent submodule commit; bump
  parent submodule pointer; update `MEMORY.md` with closure entry.

## Section 10 — Out of scope (don't pull in)

- Multi-GPU NPC.5 closure — already separate phase.
- Future MoE-variant determinism (sections of §6 marked "future"). Audit
  those when a MoE variant ships to production; don't pre-fix.
- Compaction of the `mmvq-templates.cuh` switch ladder. Out of NPC
  scope.
- KV-cache or Hadamard rotation changes — those are fixed by production
  config and orthogonal.
