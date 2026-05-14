# Phase 36 Fused Chain — Design Trace

## End-state synthesis (what the architecture must achieve)

```
┌─ verify decode (mtp_op=NONE, n=N+1) ─────────────────────────────┐
│  main 64 layers → h_pre_norm tagged                              │
│  + kv-only MTP for ALL N+1 positions → MTP KV slot 64 populated  │
│  + lm_head → verify logits                                       │
└──────────────────────────────────────────────────────────────────┘
        ↓ accept decision (which drafts pass)
        ↓ seq_rm(slot.n_past, -1) → trims rejected on main+MTP
        ↓ kv_self.head = pos_base + n_accepted
        ↓
┌─ fused draft (mtp_op=DRAFT_GEN_FUSED, n=M) ─── on draft_stream ──┐
│  inp_mtp_states ← D2D last row of h_pre_norm                     │
│  step 0: seed_token + inp_mtp_states → attn (write KV at p+0)    │
│  step k: argmax_{k-1} + step_{k-1}_residual → attn (write p+k)   │
│  output: argmax_0 .. argmax_{M-1}                                │
└──────────────────────────────────────────────────────────────────┘
        ↓ verify cycle k+1 reads MTP KV for verification
```

## Reference: how the per-step path works

The validated baseline (per-step DRAFT_GEN, no fused):

For each draft step k = 0..N-1:
1. `prepare_mtp_graph_inputs`: copy host buffer (`lctx.draft_input_hidden_state`) to `inp_mtp_states` device tensor.
2. `llama_decode(batch=[current_token at current_pos])`:
   - `build_qwen35_mtp(prev=inp_mtp_states, token=current_token)` builds:
     - `hidden_state_norm = norm(prev, hnorm)`
     - `token_emb = get_rows(tok_embd, current_token)`
     - `token_emb_norm = norm(token_emb, enorm)`
     - `combined = concat(token_emb_norm, hidden_state_norm, axis=0)`
     - `fused = lora_mm(eh_proj, combined)`
     - `attn_out = self_attention(fused, KV write at current_pos)`
     - `cur = ffn(attn_out)` (when MTP layer has FFN — Qwen 3.6 27B does)
     - `cur = cvec.apply_to(cur, il)`
     - **`normed = norm(cur, shared_head_norm)` — tagged "result_norm"**
     - `logits = lm_head(normed)` — tagged "result_output"
3. Post-decode: `embd` extraction picks "result_norm" → `lctx.embd[0]` = first row of `normed`.
4. `id_next = sampler.sample(logits)` (greedy → argmax).
5. `emb = llama_get_embeddings_ith(ctx, 0)` → host pointer to `lctx.embd[0]`.
6. `llama_set_draft_input_hidden_state(emb)` → `lctx.draft_input_hidden_state = emb`.
7. `current_input_id = id_next; current_n_past++`.

**Chain residuals: `result_norm` (post shared_head_norm).**

**Chain tokens: argmax of logits.**

## Reference: how the fused builder is supposed to mirror this

Single graph build with N steps chained in-graph:

- Inputs:
  - `inp_mtp_states` (n_embd, 1) — host-filled with seed_hidden (= per-step's row-0 of result_norm of previous decode)
  - `inp_tokens` (n_tokens=N) — host-filled; step 0 reads index 0
  - `inp_pos` (n_tokens × n_pos_per_embd) — positions [pos_base..pos_base+N-1]
  - `inp_KQ_mask` (n_kv, n_tokens × GGML_KQ_MASK_PAD)

- Per step k:
  - `tok_id_k = (k==0) ? inp_tokens[0:1] : argmaxes[k-1]`
  - `prev_input = (k==0) ? inp_mtp_states : prev_residual`
  - call `build_qwen35_mtp_kv_only(prev_input, tok_id_k, ..., inp_pos[k:k+1], KQ_mask[k_col:k_col+pad])`
    - Inside: hnorm + enorm + concat + eh_proj + attention (writes KV at slot[head+k])
  - FFN
  - cvec
  - `normed = norm(cur, shared_head_norm)`
  - `prev_residual = normed`
  - `logits = lm_head(normed)`
  - `argmax_k = ggml_argmax(logits)` (output tensor, named "mtp_argmax_<k>")
  - `prob_k = softmax(logits)[argmax]` (named "mtp_prob_<k>")

After compute: read N argmaxes + N probs from named output tensors.

## Measured: where the chain breaks

X02 256K, greedy, both env vars on:

| n_draft | t/s | acceptance | per-cycle accepted |
|---------|-----|-----------|--------------------|
| 1 (per-step) | 34.21 | 77% | 0.77 |
| 3 (fused)    | 20.75 | 18% | ~0.54 |
| 5 (fused)    | 17.78 | 13% | ~0.65 |

Acceptance pattern at d=5: 19 accepted out of 143 generated. Decomposing by step assuming each draft step is independent:
- If step 0 accepts at ~77% (matches d=1) and step ≥1 accepts at 0%: total = 0.77/5 = 15%. Close to measured 13%.

**Conclusion: step 0 is correct, step ≥ 1 is broken.**

## Design issues to trace (no code changes — analysis only)

### Issue A: chain residual identity

**Per-step:** step k+1's `prev_embeddings` = first row of `lctx.embd`, which is the post-`shared_head_norm` residual at row 0 of the previous DRAFT_GEN's `result_norm` tensor.

**Fused:** step k+1's `prev_embeddings` = `prev_residual` = the just-computed `normed` tensor of step k. Same `shared_head_norm` applied. Tensor shape (n_embd, 1).

**Question:** are these byte-identical when computed under identical inputs? Both apply the same RMS norm with the same weights. Both feed the same upstream cur (post-FFN, post-cvec). They *should* be byte-identical.

**Trace experiment (no build):** compare the two function call sites.

- Per-step `build_qwen35_mtp` line 250: `cur = llm_build_norm(ctx0, cur, hparams, mtp_layer.nextn.shared_head_norm, NULL, LLM_NORM_RMS, cb, il);`
- Fused `build_qwen35_mtp_fused` (after my edit): `ggml_tensor * normed = llm_build_norm(ctx0, cur, hparams, mtp_layer.nextn.shared_head_norm, NULL, LLM_NORM_RMS, cb, il_mtp);`

Same call. Same result. Issue A: **EXONERATED** by source inspection.

### Issue B: chain token identity

**Per-step:** `current_input_id = id_next = argmax(logits)` from `common_sampler_sample_speculative`. Captured to host, then re-fed via the next decode's inp_tokens[0].

**Fused:** `argmax_k = ggml_argmax(logits)` — a graph node of shape (1) I32. Step k+1 reads `tok_id = argmaxes[k-1]` and does `get_rows(tok_embd, tok_id)`.

**Question:** is `ggml_argmax(logits)` semantically equivalent to host-side argmax of D2H'd logits? Both should pick the index of the maximum value along dim 0.

**Trace experiment:** check `ggml_argmax` operates over dim 0 returning shape (a->ne[1]). For our (n_vocab, 1) input, returns shape (1) — single index. Same result as host-side.

**Caveat:** the previous debug print showed step 0's argmax value = 248045 — outside Qwen 3.6's n_vocab of ~152064. **This is the smoking gun** — argmax is returning an out-of-range value.

**Hypothesis B1:** `ggml_argmax` on the device output of split-mode `build_output` is reading garbage memory.
- `build_output` split path concat's per-device matmuls with `ggml_concat(ctx, o[0], o[1], 0)`.
- The concat output tensor's data layout: contiguous (n_vocab, 1) F32.
- `ggml_argmax` reads dim-0 elements from data buffer.
- If concat doesn't materialize properly (e.g., view aliasing across devices), argmax reads memory outside the logical n_vocab range.

**Hypothesis B2:** the multi-device concat's output is NOT contiguous on a single device — argmax CUDA kernel can't follow the cross-device split.

**Trace experiment:** check `ggml_concat` produces a contiguous output, or whether it's a view that argmax can't traverse correctly across device boundaries. The split-tensor path in `build_output` (lines 2030-2050) uses `ggml_concat` after `lora_mm`. If the `lora_mm` outputs are on different devices, the concat creates a logical concat but the physical data might not be contiguous on either device.

### Issue C: KV write/read causality

**Per-step:** each step is its own cgraph. K/V cpy's complete in their own decode before kv_self.head advances. Next decode's K_view reads finalized cache.

**Fused:** all steps in one cgraph. K/V cpy's are graph nodes. K_view of step k+1 spans cells [0..head+k+1]. For cells [head..head+k] to read step 0..k's writes, the cpy nodes must execute BEFORE the K_view's downstream reads.

**Trace by gf->nodes ordering:**
- Step k's `llm_build_kv` calls `ggml_build_forward_expand(graph, k_cur_cpy)` explicitly — adds `k_cur_cpy` to the graph at insertion time.
- Step k's attn_out (with K_view inside) is added later via `ggml_build_forward_expand(graph, attn_out_k)`.
- Step k+1's K_view is inside step k+1's attention, which is added later still.

In `gf->nodes` order: `k_cur_cpy_0` < `K_view_0` < `attn_out_0` < ... < `k_cur_cpy_1` < `K_view_1` < ...

So `K_view_1` reads slot[head] AFTER `k_cur_cpy_0` has run (slot[head] was written). Correct for slot[head].

For slot[head+1]: `K_view_1` reads it AFTER `k_cur_cpy_1`? Both are in step 1's sub-graph. Within build_std_attention call, the cpy is added BEFORE the FA op (which uses K_view). So gf order: cpy_1, K_view_1, FA_1. Cpy runs before view. Correct.

Issue C: **EXONERATED** by gf->nodes ordering analysis.

### Issue D: KQ_mask layout mismatch with FA's per-query reading

**My layout:** mask shape (n_kv, n_tokens × GGML_KQ_MASK_PAD). Step k's view: (n_kv, GGML_KQ_MASK_PAD) at row offset k × GGML_KQ_MASK_PAD.

**FA contract:** mask[i][j] = mask for key position i attending from query position j. For n_queries=1, only mask[i][0] is read by the kernel.

**Issue:** my view of step k starts at row offset k × KQ_MASK_PAD in the source. The view's "row 0" corresponds to source's row k × KQ_MASK_PAD. set_inputs fills source row `j × KQ_MASK_PAD` with step j's visibility. So step k's view-row-0 = source row k × KQ_MASK_PAD = step k's visibility. Correct.

**But:** when ggml_view is created with offset, the resulting tensor's `data` pointer is `source->data + offset`. The view's stride (`nb[0]`, `nb[1]`) determines how it reads. For a view of (n_kv, KQ_MASK_PAD) at row offset:
- `nb[0]` = element_size
- `nb[1]` = source's `nb[1]` = `n_kv * element_size`
- offset = `k × KQ_MASK_PAD × n_kv × element_size` bytes from source start

Hmm the offset is in BYTES from source start. Source layout: `data[col * n_kv + row]` where col = column index in (n_kv, n_pad_total). For col = k × KQ_MASK_PAD, offset bytes = k × KQ_MASK_PAD × n_kv × element_size. ✓

But... `ggml_view_2d(ctx, src, ne0, ne1, nb1, offset)` — let me re-check the signature.

Actually `ggml_view_2d` parameters: `ctx, source, ne0, ne1, nb1, offset`. `nb1` is the "row stride" of the result view. For our case, nb1 = `source->nb[1]` = `n_kv * element_size`. The offset is in bytes.

So my view has: ne[0] = n_kv, ne[1] = KQ_MASK_PAD, nb[0] = source->nb[0] = element_size, nb[1] = source->nb[1] = n_kv * element_size. Offset = k × KQ_MASK_PAD × source->nb[1].

Hmm — my code passes `(size_t) k * GGML_KQ_MASK_PAD * lctx.inp_KQ_mask->nb[1]`.

`source->nb[1] = n_kv * element_size`. So offset bytes = k × KQ_MASK_PAD × n_kv × element_size. ✓

The view points at source byte offset = k × KQ_MASK_PAD × n_kv × element_size. In source's element terms (where data[idx] = idx * element_size bytes from start), the view starts at element index `k × KQ_MASK_PAD × n_kv`. That's source column `k × KQ_MASK_PAD` (since column j starts at element index j × n_kv).

Step k's view "column 0" (in the view) corresponds to source column `k × KQ_MASK_PAD`. set_inputs writes step j's visibility to source column `j × KQ_MASK_PAD`. Match: step k's view column 0 = step k's visibility.

Issue D: **EXONERATED** by stride math.

### Issue E: inp_pos for MROPE

Verified: inp_pos is sized `n_tokens × n_pos_per_embd`. Step k's view: `n_pos_per_embd` elements at offset `k × n_pos_per_embd`. Correct for MROPE 4-axis.

Issue E: **EXONERATED**.

### Issue F: lm_head output across multi-device split

**The smoking gun candidate:** `build_output` for split lm_head uses:
```cpp
for (int id = 0; id < split_output->n_device; ++id) {
    auto split = split_output->splits[id];
    o.push_back(llm_build_lora_mm(lctx, ctx, split, cur));  // shape (n_vocab/n_dev, 1) per device
}
cur = ggml_concat(ctx, o[0], o[1], 0);  // (n_vocab, 1) — but where does this concat live?
```

`ggml_concat` along axis 0 of two tensors on different devices — is the result a contiguous tensor on a single device, or a virtual view spanning devices?

For our purposes, `ggml_argmax` runs on the CUDA backend on ONE device. If the concat result spans two devices, argmax can only see the data on its own device — reading garbage past the boundary.

**Hypothesis F:** `ggml_argmax` on the split-mode `build_output` result returns indices that may exceed the per-device n_vocab/2 because the kernel reads across the concat's device boundary. The 248045 = 152064 + 95981 — possibly = first device's n_vocab/2 + an offset within the second device's range that the kernel mis-indexed.

Actually 152064/2 = 76032. 248045 - 76032 = 172013. Doesn't decompose cleanly. But the value being out-of-vocab is real.

**Trace experiment:** for the per-step path, argmax happens HOST-SIDE after D2H of logits. So per-step never exercises ggml_argmax on split outputs. My fused path DOES — and the test fixture would expose this.

**Conjecture:** ggml_argmax on a ggml_concat of split-device tensors is the broken primitive. Per-step doesn't hit it because it does host-side argmax.

## Next experiments (no builds — just measurements/reads)

1. **Run d=1 fused with the gate lowered to n_draft >= 1.** If d=1 fused matches per-step d=1 exactly → step 0 fused works → chain is the issue. If d=1 fused doesn't match per-step → step 0 itself is broken (likely Issue F: argmax on split lm_head).

2. **Read step 0's argmax value when running fused at d=2 with debug.** Compare to host-side argmax of step 0's logits (would need a separate observe-only path). If argmax value is out-of-vocab (like 248045) → confirmed Issue F.

3. **If Issue F confirmed:** the fix is to D2H step k's logits and do argmax host-side, then push back to graph for step k+1's get_rows. That defeats some of the "fused = single compute" benefit but is correct.

   Alternative: do argmax PER-DEVICE first (each device argmax's its split slice), then a small reduction to pick the global winner. That keeps it on-device.

   Alternative: concat the split logits to ONE device before argmax. Adds a cross-device copy.

## Decision

Per the user's instruction "STOP BUILDING AND START TRACING" — I am done with code edits in this oneshot. The fused chain debug requires a fixture that exposes step-by-step argmax values to host for comparison with per-step. That's test infrastructure, not graph patching.

Working result of this session:
- Step 3 (kv-only hook) shipping at +12% MTP-vs-nomtp at d=1.
- Step 1 (fused) blocked on Issue F (argmax over split-mode lm_head output).
- Step 5 (KQ_mask bucketing) shipped (no measured benefit in this workload but premise preserved).
- Step 4 (D2D relay) intentionally fall-through to host bounce until row-selection contract is locked down.
- Step 2 (pipeline) blocked on Step 1.

The real ceiling at this quant on this 2-GPU setup, given Issue F: ~34.93 t/s (Step 3 d=1), measured. The +14% Step 2 lever is unreachable until Step 1 is unblocked.
