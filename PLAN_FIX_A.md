# FIX-A — full worked plan

**Date**: 2026-05-16
**Goal**: eliminate the slot-position-dependence in the production FA path (`ggml_flash_attn_ext_per_slot_kv` → `wmma_f16_case_pb1<256,256,8,float>`) when `LLAMA_FATTN_PER_SLOT_KV_ENABLE=1`, restoring NP-cross byte-identity at the cost of N× per-layer FA launches.

**Binding** (success criteria, in order):
1. TRACE-6 re-run shows `flash_attn_per_slot_kv-1003`/`-2003` byte-identical between slot 0 and slot 1 at NP=2 (was: n_diff=1042/1515; target: n_diff=0).
2. TRACE-1 re-run shows ALL slots byte-identical to slot 0 at every layer at NP={2,4,8} (was: slot-parity, odd slots differ from even at layer 3+; target: all match).
3. Production harness `scripts/test-production-np-determinism.sh` returns 14/14 byte-identical at NP={1,2,4,8} (was: 5/14; target: 14/14, modulo the §15.17 run-to-run stochastic residual which FIX-A may or may not also close).
4. No regression in NP=1 path (regression test: same prompt at NP=1 before vs after FIX-A produces byte-identical token sequence and same end-of-token logits).

## Strategy

In `src/llama-build-context.cpp` `build_std_attention()`, when `use_per_slot_kv` is true and `q->ne[1] > 1` (i.e., NP>1 decode or multi-token prefill), replace the single batched FA call with a loop that issues N FA calls, each with `ne[1]=1`. Concatenate the per-slot outputs along the n_tokens dim (output `ne[2]`).

When `q->ne[1] == 1`, fall through to the existing single-call path unchanged.

## Tensors that need per-slot views

For each per-slot iteration `s ∈ [0, N)`:

### Q

`q` is the permuted Q: `ggml_permute(Qcur, 0, 2, 1, 3)` → `ne = [head_dim, n_tokens, n_heads, n_seqs]`.

Per-slot view:
```cpp
ggml_tensor * q_s = ggml_view_4d(ctx0, q,
    q->ne[0], 1, q->ne[2], q->ne[3],
    q->nb[1], q->nb[2], q->nb[3],
    (size_t) s * q->nb[1]);
```

Result `ne = [head_dim, 1, n_heads, n_seqs]`. Stride preserved.

### KQ_mask — the difficult one

`KQ_mask` from `build_inp_KQ_mask()`: `ne = [n_kv, GGML_PAD(n_tokens, 16)]`. For n_tokens ≤ 16, the second dim is **exactly 16**, NOT n_tokens × 16. Slots occupy rows 0..n_tokens-1; rows n_tokens..15 exist as memory but have no semantic meaning.

A naive per-slot view `ne[1]=1` at offset `s * nb[1]` fails TWO things:
- `ggml_flash_attn_ext_per_slot_kv` assertion: `mask->ne[1] >= GGML_PAD(q->ne[1], 16)` = 16.
- `ggml_view_2d` size check: at slot s with `ne[1]=16`, view extends to row s+15; for s>0 this overshoots the 16-row buffer.

**Three resolution options**:

**Option B1 — Pre-pad the mask in `build_inp_KQ_mask`**: allocate `[n_kv, n_tokens * GGML_KQ_MASK_PAD]` instead of `[n_kv, GGML_PAD(n_tokens, 16)]`. Each slot gets its own 16-row block; view at offset `s*16` with ne[1]=16 fits cleanly.
- Pros: clean per-slot blocks; assertion still holds for both batched and per-slot dispatch.
- Cons: invasive — changes `build_inp_KQ_mask` for ALL graph builders (qwen35, qwen35moe, llama, etc.). Need to verify host-side `llama_set_inputs` mask-fill code adapts to the new shape (host code writes mask at row s = slot s; with new shape, host writes at row s*16). May break other graphs that rely on the existing layout.
- Surface area: 1 ggml.c function (mask alloc) + 1 host code site (mask fill) + audit of all build graphs that reference `inp_KQ_mask`.

**Option B2 — Relax the FA-op assertion to `mask->ne[1] >= q->ne[1]`**: the 16-pad is defensive over-padding; CUDA and CPU FA only access q->ne[1] rows in practice. Then per-slot mask view at `ne[1]=1, offset = s * nb[1]` works.
- Pros: minimal change (1 line in ggml_flash_attn_ext_per_slot_kv, optionally 1 in ggml_flash_attn_ext for symmetry).
- Cons: weakens a public contract; need to verify NO kernel paths actually do require the 16-pad (e.g., for SIMD alignment, wmma fragment loads, or CPU vector loads). Survey needed.
- Surface area: 1-2 asserts in ggml.c + audit of all FA kernel paths to confirm no >ne01 mask reads.

**Option B3 — Replicate the slot's mask row to a fresh 16-row tensor per per-slot call**:
```cpp
ggml_tensor * mask_s_row = ggml_view_2d(ctx0, KQ_mask, n_kv, 1, nb1, s*nb1);
ggml_tensor * mask_s     = ggml_repeat(ctx0, mask_s_row, /*shape=*/ggml_new_tensor_2d(ctx0, F16, n_kv, 16));
```
- Pros: no ggml.c changes, no mask-alloc changes.
- Cons: extra compute (replicate kernel); extra memory; possibly extra non-determinism source via ggml_repeat itself.
- Surface area: only build graph.

**Recommendation**: Option B2 (relax assert). It is the smallest correct change. The 16-pad was a defensive constant; the kernels do not actually need it. Pre-flight: audit FA kernel paths for any mask read past `ne01` rows.

#### Pre-flight audit for Option B2 (must pass before edit)

Survey items:
1. `ggml/src/ggml-cuda/fattn-wmma-f16.cuh` — mask read `maskh[j*(nb31/sizeof(half)) + k_VKQ_0 + k]`. j ranges 0..ne01-1 (verified TRACE-3/6 analysis). Safe.
2. `ggml/src/ggml-cuda/fattn-vec-*.cuh` — mask read pattern. Need to confirm.
3. `ggml/src/ggml-cuda/fattn-common.cuh` — `launch_fattn` mask handling. Confirm no extra padding requirements.
4. `ggml/src/ggml.c` `ggml_compute_forward_flash_attn_ext_f16` — CPU FA mask access via `mask->nb[1] * j` for j in [0..neq1). Safe (verified above).
5. `iqk_flash_attn_*` — IQK fallback path. Need to confirm.

If all 5 confirm "only neq1 rows accessed", B2 is safe.

If any path accesses past neq1 rows (e.g., for warp-aligned mask loads), use B1 or B3 instead.

### per_row_k_bound

`inp_per_row_k_bound` is `I32[q->ne[1]]`. Per-slot view: `ne[0]=1, offset = s * sizeof(int32_t)`.

```cpp
ggml_tensor * bound_s = ggml_view_1d(ctx0,
    lctx.default_decoder.inp_per_row_k_bound, 1,
    (size_t) s * sizeof(int32_t));
```

No assertion issues.

### Per-slot FA output and concat

Each per-slot FA returns `ne = [Dv, n_heads, 1, n_seqs]`. Concatenate N of them along `dim=2` (n_tokens):

```cpp
ggml_tensor * cur = per_slot_outs[0];
for (s = 1; s < N; ++s) cur = ggml_concat(ctx0, cur, per_slot_outs[s], 2);
```

`ggml_concat` makes the output contiguous. Subsequent `ggml_hadamard`, `ggml_reshape_2d`, `mul_mat wo` operate on it as before.

### Op_params per per-slot FA call

The existing post-branch code:
```cpp
if (n_swa > 0) ((int32_t *)cur->op_params)[4] = n_swa;
if (should_use_f32_precision) ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
```

For per-slot N>1: `cur` is now `ggml_concat`, NOT a flash_attn op. `ggml_flash_attn_ext_set_prec` asserts on op type — would abort.

**Resolution**: apply both inside the per-slot loop on each per-slot FA op; gate the post-branch block with a flag `fa_op_params_already_applied`.

## Implementation diff (commit 1, after pre-flight audit)

### Change 1 — `ggml/src/ggml.c` (only if B2 chosen)

Two asserts to relax:
- `ggml_flash_attn_ext`:
  ```c
  GGML_ASSERT(mask->ne[1] >= q->ne[1] && "FA mask must have at least q->ne[1] rows");
  ```
- `ggml_flash_attn_ext_per_slot_kv`:
  ```c
  GGML_ASSERT(mask->ne[1] >= q->ne[1] && "FA mask must have at least q->ne[1] rows");
  ```

### Change 2 — `src/llama-build-context.cpp` `build_std_attention()`

Replace the body of the `if (use_per_slot_kv) { ... }` branch with the per-slot loop + concat per the views above. Add `fa_op_params_already_applied` flag. Gate post-branch n_swa + set_prec.

## Verification plan

After change 1 + 2, build and run:

### V1 — unit-test pass-through
```bash
cmake --build build -j 32
./build/bin/test-fattn-per-slot-kv-ncols-invariance
./build/bin/test-fattn-per-slot-kv-dispatch-np-invariance
./build/bin/test-rmsnorm-batch-shape-invariance
./build/bin/test-rope-batch-shape-invariance
```
All four must pass. (These don't exercise the new per-slot dispatch path — they're regression coverage for adjacent ops.)

### V2 — TRACE-6 binding (the FIX-A gate)
```bash
mkdir -p /home/llm/yarn-agentic/data/trace-6-fixa-pass-2026-05-16
LLAMA_FATTN_PER_SLOT_KV_ENABLE=1 LLAMA_TEST_NP=2 \
LLAMA_TEST_TARGET=... LLAMA_TEST_PROMPT_DIR=... \
LLAMA_TEST_OUT_DIR=/home/llm/yarn-agentic/data/trace-6-fixa-pass-2026-05-16 \
./build/bin/test-trace-2-intra-layer-capture
python /home/llm/yarn-agentic/data/trace-6-2026-05-16/diff_v2.py
```
Expected: `flash_attn_per_slot_kv-1003` / `-2003` show **n_diff=0** between slot 0 and slot 1. `l_out-3` also n_diff=0.

If V2 passes, FIX-A is correctness-confirmed at NP=2.

### V3 — TRACE-1 binding at NP={2,4,8}
```bash
for NP in 1 2 4 8; do
  LLAMA_FATTN_PER_SLOT_KV_ENABLE=1 LLAMA_TEST_NP=$NP \
  LLAMA_TEST_TARGET=... LLAMA_TEST_PROMPT_DIR=... \
  LLAMA_TEST_OUT_DIR=/home/llm/yarn-agentic/data/trace-1-fixa-2026-05-16/np$NP \
  ./build/bin/test-deltanet-d1-capture
done
python data/trace-1-2026-05-16/diff.py  # adapt path
```
Expected: every slot byte-identical to slot 0 at every layer for NP ∈ {2,4,8}; slot 0 at NP=1 vs slot 0 at NP=N still differs at layer 0 due to the known F32-vs-F16 storage path (independent of FIX-A scope).

### V4 — Production NP-determinism harness
```bash
systemctl --user stop llama-server  # already stopped per current plan
bash scripts/test-production-np-determinism.sh
```
Expected: 14/14 byte-identical OR documented residual that maps to §15.17's run-to-run stochastic component.

### V5 — NP=1 regression
```bash
# Run llama-server at --parallel 1 with FIX-A binary; compare token output for a fixed prompt
# vs the same against pre-FIX-A binary (saved separately).
```
Expected: byte-identical token sequence. (If V5 fails, FIX-A introduced an NP=1 regression — must investigate and roll back.)

### V6 — Perf measurement
```bash
# Use llama-bench or a custom decode-rate harness.
# Measure: tokens/sec at NP=1, NP=2, NP=4, NP=8 with FIX-A on vs FIX-A off.
```
Expected slowdowns vs baseline (estimates from analysis):
- NP=1: 0% (same path; one slot loop iter).
- NP=2: ~5-15% per-slot decode rate.
- NP=4: ~10-30% per-slot, but aggregate still beats NP=1.
- NP=8: ~20-50% per-slot, aggregate ≈ 2.5× NP=1.

Capture results to `data/fixa-perf-2026-05-16.json`.

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| B2 audit reveals a kernel path that requires 16-row mask padding | Medium | Forces switch to B1 or B3 | Pre-flight audit; if fails, switch path |
| ggml_concat introduces fp32 non-determinism in some scenario | Low | Defeats fix | concat is element-wise copy, fp-deterministic; verify via V2 |
| Per-slot FA at ne[1]=1 hits some kernel path the test fixture didn't cover (e.g., parallel_blocks heuristic differs) | Low | Subtle drift | V2 + V3 should catch; if drift appears, debug |
| FIX-A breaks NP=1 path | Low | Production regression | V5 explicitly tests; falls through to existing single-call code at q->ne[1]==1 |
| Mask view contiguity check fails | Medium | Build aborts | Audit `ggml_is_contiguous` semantics on views; may need to use `ggml_cont` to materialize |
| Multi-device (split-mode graph) per-slot dispatch inside the for-id loop multiplies launches by 2× more | Already-known | Increases launch overhead | Document; perf measurement V6 quantifies |
| n_swa + set_prec applied to concat tensor instead of FA op | Already-found | Build aborts | `fa_op_params_already_applied` flag |
| ggml_repeat-style alternative (B3) introduces its own determinism issue | If used | Defeats fix | B3 is fallback only; if used, add unit test |

## Rollback plan

If any V2-V5 fails after the V1 unit tests pass:

```bash
git checkout -- src/llama-build-context.cpp ggml/src/ggml.c
cmake --build build -j 32
```

Re-baseline TRACE-6 to confirm rollback put us back at the slot-parity-diverging state. Then investigate the specific V*-failure with targeted instrumentation.

## Commit plan (post-success)

If V2-V5 all pass:
1. Commit 1: ggml.c assert relaxation (B2) alone.
2. Commit 2: build_std_attention per-slot dispatch.
3. Commit 3: data/trace-6-fixa-pass-2026-05-16/ + data/trace-1-fixa-2026-05-16/ trace bundles.
4. Commit 4: data/fixa-perf-2026-05-16.json perf results.
5. Commit 5: PHASE_MMQ_Q4_0_AR16.md update — close out the FA NP-determinism subtask under CX or new phase; reference TRACE-* findings.
6. Commit 6: spec update at `specs/deltanet/fattn-per-slot-kv-sm75.md` — retract §15.13's frag_c_VKQ framing; add §15.18 documenting the actual root cause (batched k-loop chunk decomposition) and FIX-A's per-slot dispatch.
7. Commit 7: MEMORY.md entry.

## Open questions before edit

1. **Does the host-side mask fill (`llama_set_inputs` or equivalent) work correctly when the FA op sees a per-slot mask view rather than the full mask?** The mask values are filled before graph compute; FA reads them. As long as the slot-row content is at the same memory address as before, FA reads the same bytes. Should be OK.

2. **Does `ggml_is_contiguous` return true for the per-slot mask view (ne[1]=1, offset = s * nb[1])?** Need to check. If false, the contiguity assert at `ggml_flash_attn_ext_per_slot_kv:10259` fails. May need `ggml_cont(mask_s)` which materializes a copy — defeats some of the perf goal but is correct.

3. **For NP=1 prefill (n_tokens > 1, all from slot 0)**, is `use_per_slot_kv` even true? Looking at the condition `use_per_slot_kv = ... && q->ne[2] / k->ne[2]) <= 16 && ...`, the prefill case at n_tokens=12 (our test) — `q->ne[2]` is 12 heads per device, `k->ne[2]` is 2 KV heads per device, ratio = 6. So use_per_slot_kv is true even at prefill. **FIX-A would decompose prefill into N=n_tokens FA calls** (12 calls for our test). That's a big perf hit at prefill. We may want to GATE FIX-A on "decode only" (i.e., n_tokens == n_seq_max). Verify.

4. **Concat of N tensors along ne[2]** — does ggml_concat support arbitrary count? It takes 2 at a time; chained.  N-1 chained concats per FA layer per device — small but non-zero graph node count. Verify graph builds without exceeding node count limits.

5. **The flash_attn_per_slot_kv_slot cb tags** — each per-slot FA op gets cb'd with `il_cb*100 + s`. Make sure the il_cb*100 doesn't collide with other tags. Existing il_cb = `1000*(id+1) + il`. For id=0 il=3: 1003. *100 + s = 100300 + s. Fine, unique.

## Decision points for user before edit

- **Confirm B2 (relax assert) is acceptable**, or prefer B1 (pre-pad mask) or B3 (replicate per-call)?
- **Should FIX-A be decode-only** (`q->ne[1] <= some_threshold` to avoid prefill blowup), or applied uniformly?
- **OK to proceed with V1-V5 verification once edits land**, or want a smaller intermediate gate first?
