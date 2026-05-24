---
name: Source-read the reference stack before instrumenting
description: When reproducing a reference implementation (vLLM, upstream torch impl, paper code), READ the reference top-to-bottom as the first diagnostic step — not per-step instrumentation of your own code. Most "where is the bug?" answers are sitting in the reference for free.
type: feedback
originSessionId: 9b52ee2b-d969-442c-83ba-98396525f6fa
---
When the from-scratch implementation diverges from a reference (NaN cascade, wrong numerics, surprising magnitudes), the cheap first move is to source-read the reference end-to-end — not to instrument your own pipeline with per-step memcpys / NaN scans / printfs.

**Why:** A 15-minute source-read finds the kinds of bugs that are missing-step / wrong-order / wrong-dtype mistakes — bugs that are *invisible* from per-step diagnostics because the steps each run fine in isolation. Per-step instrumentation is only useful AFTER you've ruled out structural omissions.

**How to apply:**
- When the test against the reference fails in a way that suggests algorithmic divergence (NaN, magnitude wrong, output magnitudes that grow unbounded), open the reference source and read it line-by-line in the relevant code path.
- Diff your kernel's pipeline against the reference's pipeline at the OP level: list every op the reference applies, in order, and check yours has the same list in the same order.
- Pay special attention to:
  - Final norms / pre-norms (easy to miss — they're often the LAST line of a class's forward method)
  - Dtype casts at boundaries (input/output type vs accumulator type)
  - Attention mask direction (causal vs bidirectional within blocks)
  - Tensor layout assumptions (row-major vs column-major; [N, K] vs [K, N])
- Per-step instrumentation comes AFTER source-reading, not before. Only useful for tracking down reduction-order noise once the algorithm is verified to match.

**Concrete example (T4 DFlash closure):** All-NaN drafter logits. I almost added per-step memcpy + NaN scan throughout the 10-sub-kernel pipeline. User pushed back ("discuss more elegant angles"). 15-minute read of vLLM's `DFlashQwen3Model.forward` found 4 bugs at zero cost: F32-norm-as-`__half*` dtype mismatch (the root NaN), missing output_norm, wrong full-attention K-loop direction, missing drafter Q/K/V proj at query positions. Per-step diagnostics would have spent hours finding what was visible in the reference at first read.

**Aphorism:** Read the reference, *then* binary-search.
