---
name: project-t5-3-scope-corrected-bundle-a-close
description: T5.3 scope corrected to PHASE-doc Bundle A close (partial seq_rm shadow + LLAMA_T5_TRACE only); supersedes prior memory that bundled the byte flip + kernel work into T5.3 — that is Bundle B (T5.5+) because PSKV singlewarp kernel has no block_table indirection and pre-emptive formula flip without kernel change breaks production byte-identity
metadata:
  node_type: memory
  type: project
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

production/2026-q2-next 2026-05-23. **Correction to
[[project-t5-2-shadow-landed-t5-3-next]].**

That earlier memory's "T5.3 = coherent Bundle A close" with steps:
(2) replace `inp_kv_idxs` formula, (3) reshape SET_ROWS view,
(4) reshape kernel K/V views, (5) remove CPY-fallback, was
architecturally infeasible AND scope-misaligned with the PHASE
doc.

## Why the prior scope was wrong

1. **Kernel has no `block_table` indirection.** Reading
   `ggml/src/ggml-cuda/fattn-per-slot-kv-singlewarp-sm75.cu:99-100,145-148`:
   the PSKV singlewarp does
   `K_base = K_direct + nb13*seq + nb12*head_kv`
   then `K_row = K_base + k*nb11` — a uniform-stride linear walk
   over positions. Any paged layout where seq's blocks scatter
   across the pool (i.e., non-identity mapping) breaks linear
   stepping at every block boundary. NO `nb[*]` strides setting
   can fix this — block_table indirection has to happen INSIDE
   the kernel. Step 4 of the prior plan was unsound.

2. **PHASE doc structures Bundle A vs Bundle B at this boundary.**
   `PHASE_NSTREAM_KV_PERF.md:2154-2160`:
   - **Bundle A (T5.1-T5.4):** allocator + WRITE-path bookkeeping.
     Does NOT touch READ/kernel.
   - **Bundle B (T5.5-T5.8):** READ path + kernel flip; kernel
     takes `block_table` as `src[6]`; formula change + view
     reshape + CPY-fallback removal all ship together with kernel
     learning the indirection.

3. **Premature byte flip breaks NPC.** Flipping `inp_kv_idxs`
   formula while the kernel still reads via uniform `k*nb11`
   means production writes go to byte addresses the kernel
   doesn't read from. verify-production-determinism breaks
   on first run. The two changes MUST ship together (Bundle B).

## Corrected T5.3 scope

T5.3 = **Bundle A bookkeeping closure** comprising:

1. Wire **partial `seq_rm` shadow** updates in
   `src/llama.cpp:llama_kv_cache_seq_rm` so the paged
   allocator's `written_tokens` and `block_table` stay
   consistent with `cells[]` after partial-range removal.
   This is the only remaining shadow-bookkeeping gap from T5.2.
2. Implement the **`LLAMA_T5_TRACE` NDJSON emitter** at
   `ik_llama.cpp/src/llama-paged-kv-trace.{h,cpp}` (header is
   currently a stub from T5.0). Emits BlockAllocEvent records
   on alloc_block / free_seq / write_tokens (defrag is no-op
   in Bundle A). Validated via
   `scripts/validate-paged-allocator-trace.py`.
3. Binding gate: `verify-production-determinism.sh @ 1455 MHz`
   at NP={1,2,4,8} ACCEPTANCE PASS (additive change only — no
   behavioural change).

Then T5.4 = Bundle A close perf gate: `llama-batched-bench`
NP=8 TG within ±2% of T4 C1-steady 26.49 t/s. Allocator
overhead-only — measures that shadow + trace add no measurable
cost.

## What Bundle B (T5.5-T5.8) covers

The work the prior memory mis-scheduled into T5.3:

- Kernel migration: PSKV singlewarp takes `block_table` as
  `src[6]`, dereferences per (seq, k/blk) to find physical
  pool slot, then computes byte address.
- `inp_kv_idxs` formula replacement.
- SET_ROWS target view reshape.
- CPY-with-view-offset fallback removal.
- All under coherent oneshot (per
  `[[feedback-oneshot-then-evaluate]]`); verify-production-
  determinism binds end-to-end.

PHASE doc estimates this at **60-100k tokens** for Bundle B
(line 2384). Largest single risk is T5.5 (kernel NPC at the
new signature).

## How to apply

- T5.3 = ~20-30k tokens. Do NOT touch kernel, formula, or
  views. Just bookkeeping + trace + verify.
- If you find yourself editing `src/llama-build-context.cpp`
  SET_ROWS path or any `ggml/src/ggml-cuda/fattn-*.cu` during
  T5.3, you've crossed into Bundle B — stop.

## Related

[[project-t5-2-shadow-landed-t5-3-next]] (superseded scope),
[[project-t5-1-paged-allocator-landed-dormant]],
[[project-t5-probe-falsified-path-c-override]],
[[feedback-oneshot-then-evaluate]],
[[feedback-bake-measurement-env-gates]].
