---
name: Continuous batching vs per-slot dispatch — correction to PHASE45 D10.e bandwidth conclusion
description: PHASE45 D10.e abandoned multi-slot in ik_llama.cpp citing bandwidth-bound at decode; that's correct for per-slot CUDA stream dispatch but NOT for continuous batching. Measured 4.75× aggregate uplift at np=8 on vLLM same hardware, same INT4 weights.
type: project
originSessionId: 9b52ee2b-d969-442c-83ba-98396525f6fa
---
PHASE45 D10.e abandoned multi-slot work in ik_llama.cpp on the conclusion "GPU is
bandwidth-bound and per-slot loses model-weight amortisation — aggregate np=3 t/s
would collapse to ~np=1 baseline." That reasoning **applies only to per-slot
CUDA stream dispatch** (each slot gets its own stream, each does its own forward
pass, N parallel forwards compete for the same memory bandwidth → no aggregate
gain). It does NOT apply to continuous batching.

**Continuous batching is a different mechanism**: one forward pass per scheduler
tick, batch_size=N inside that forward pass, weights read once and used by all
N sequences via batched GEMM. Each weight read produces N tokens, not 1 → the
bandwidth cost is amortised across the batch.

**Measured 2026-05-12 on 2× Quadro RTX 6000 (sm_75)**:
- Same hardware, same Qwen 3.6 27B + INT4 (AutoRound vs the GGUF qq-suffix
  the production profile uses — both INT4).
- vLLM np=1 single-stream vanilla decode: 32.54 tok/s aggregate
  - Matches ik_llama.cpp production np=1 MTP --draft 3 at 33.5 tok/s.
- vLLM np=8 batched vanilla decode: 154.77 tok/s aggregate
  - **4.75× aggregate uplift over np=1**, ~19.35 tok/s per stream.
- Data: `/home/llm/yarn-agentic/data/gate0-np1-np8.json`.

**Implication for the port**: ik_llama.cpp's existing multi-slot work was
per-slot dispatch (one llama_decode call per slot, each on its own CUDA stream).
That mechanism IS bandwidth-bound — the PHASE45 D10.e abandonment was correct
for it. To get the aggregate uplift, the port would need to implement
continuous-batching-style scheduling (one llama_decode call processing N
sequences in a single batched forward), which doesn't exist in ik_llama.cpp's
scheduler today.

**Why:** the "no aggregate win" framing in MEMORY.md
(project_phase45_d10e_perslot_abandoned) was anchored on a measurement that
covered only one architectural choice. Continuous batching is a separate path,
demonstrated to scale on the same hardware.

**How to apply:** when scoping multi-slot work for ik_llama.cpp or evaluating
whether to revisit it, do NOT cite PHASE45 D10.e as "multi-slot doesn't pay on
this hardware." Cite it as "**per-slot dispatch** doesn't pay; continuous
batching does (4.75× measured), but it's a major scheduler refactor in
ik_llama.cpp and was not attempted."

The DFlash speedup measurement (`data/gate0-int4-findings.md` — to be written)
is a separate finding: DFlash with a separately-trained drafter loses on
quantized targets regardless of np mode, because drafter-target dtype mismatch
collapses accept rate. That's an algorithm-level finding, not a concurrency
one — different lesson.
