# T5.0-probe — Cheap-probe validation of the paging premise

Per `PHASE_NSTREAM_KV_PERF.md` line 2042 — the cheap insurance against a T1-style
premise mismatch BEFORE committing 150–230k tokens to Tier 5 implementation.

- Branch: `production/2026-q2-next`
- Submodule HEAD: `git -C ik_llama.cpp rev-parse HEAD` → `e282d229...` (T4 closure)
- Date: 2026-05-22
- Token spend on this probe: ~10–12k (analytical decomposition; empirical run
  offered but the structural case is overwhelming and an empirical confirmation
  would not change the verdict — see §4)

## TL;DR

**Probe verdict: FALSIFICATION — the decision rule trips on waste % alone, but
composition with the T4 kernel-saturation finding shows the inferred
perf-uplift mechanism does not bind on this workload. Recommend PAUSING Tier 5
and pivoting to OpenQ-E lever #3 (nsys profile vLLM vs ik_llama at NP=8) to
identify the actual lever closing the 5.84× gap to vLLM.**

## 1. KV waste % (structural)

**Production config** (`profiles/qwen36-27b-x2-dflash.sh`):

- `--ctx-size 524288` total context
- `--parallel 2` (NP=2)
- ⇒ `n_ctx_per_stream = 524288 / 2 = 262144` tokens per slot

Realistic per-slot usage observed in production (back-of-envelope from
LiteLLM proxy + agentic harness traffic): prompts 100–4000 tokens, decode
100–500 tokens. Typical total = ~1500 tokens per slot.

**Waste % per slot at production NP=2** = `(262144 - 1500) / 262144` = **99.4%**.

**For the T5-defined M2 workload** (8 seqs × {100, 200, 400, 800, 100, 200,
400, 800} tokens × 5s arrival offsets, n_predict ≈ 200) at production ctx
total:

- `n_ctx_per_stream = 524288 / 8 = 65536` tokens per slot
- Usage per slot at completion: {300, 400, 600, 1000} × 2
- Mean usage = (300+400+600+1000)*2/8 = 575 tokens
- Mean waste = `(65536 - 575) / 65536` = **99.1%**

**This is structural / arithmetic.** No empirical run can refute it; the
per-stream slab size is fixed at allocation time at `ctx / n_seq_max`, and any
realistic prompt fills a single-digit % of that slab.

**Decision rule trips trivially**: ≥ 30% → "Paging premise binds; proceed to
T5.0 scope-lock. GP5.b target = `26.49 × (1 + waste_pct/100)` = clamp(53, 40, 100)
= 53 t/s".

## 2. Effective context loss

**Production at NP=2 ctx=524288:** zero. Per-slot slab is 256K tokens, far
beyond any realistic prompt + decode. Slot reset on request completion is
fast; no eviction pressure observed in production traffic.

**Hypothetical NP=8 at same ctx total:** still zero. Slab = 64K per slot,
realistic usage < 2K per slot.

⇒ **Ctx-loss lever does not apply on current production / projected
workloads.**

## 3. The falsification — composition with T4 kernel-saturation finding

The decision rule's `GP5.b = 26.49 × (1 + waste_pct/100)` formula encodes a
hidden assumption: *high waste % implies a perf-uplift lever via cache
reclamation*.

**That assumption does not bind on this workload.** Two reasons:

### 3.1 Kernel saturation is the binding constraint at NP=8

From the T4 closure (`MEMORY.md` → `[[project_t4_bundle_a_landed]]`):

> T4.7 perf: C1-steady 26.49 t/s (zero regression vs C0), C1-staggered 21.62 t/s
> (target 31.79 t/s structurally unachievable — **C0 IS multi-slot kernel
> saturation; staggered always ≤ steady on aggregate-t/s**).

At NP=8 steady arrivals, both C0 (no chunked-prefill admission) and C1 (full
Sarathi-Serve admission) deliver 26.49 t/s aggregate. The admission scheme is
not the bottleneck. The bottleneck is in the PSKV singlewarp kernel at
multi-slot concurrency — either compute, memory bandwidth, or launch latency
(unidentified without nsys).

**Paging does not change any of those:**

- **Compute** — paged attention does the same dot products on the same K/V
  values; no FLOP reduction.
- **Memory bandwidth** — paged attention loads the same bytes from VRAM; the
  block_table indirection adds a small read but does not reduce data volume.
- **Launch latency** — paging does not change the number of CUDA kernel
  launches per decode tick.

**Paging adds indirection overhead.** Per the T5 plan's own risk #1:
"Kernel-time indirection overhead > 5%. Mitigation: OpenQ-T5-A drop to
block_size=128." The kernel cost goes UP, not down, under paging.

### 3.2 VRAM is not capacity-bound

The waste-%-to-perf inference path is: *high waste → paging reclaims VRAM →
can serve more concurrent requests → throughput scales*.

This requires VRAM to be the limit. It isn't:

- 2× RTX 6000 = 48 GiB aggregate.
- Production at NP=2 ctx=524288: KV cache ≈ 524288 × 2 × Q4_0(~0.6 GB) ≈ ~15
  GiB — fits comfortably. Even at NP=8, KV ≈ 15 GiB total (same total ctx).
- We could scale to NP=16, NP=32 in current VRAM without paging.

VRAM efficiency is not the binding lever.

### 3.3 Cross-request prefix caching not applicable

Paged KV also enables shared prefix caching across requests with common
prompt heads (vLLM `PagedAttention` paper §5.2). This delivers throughput when
the workload has shared prefixes (system prompts, chat-template boilerplate,
RAG context reuse).

Production workload (LiteLLM → Claude Code agentic) has highly heterogeneous
prompts with negligible shared prefix structure. Prefix caching lever does
not apply.

## 4. Why empirical confirmation does not change the verdict

The probe could empirically confirm:

(a) **Waste % is what arithmetic says (~99%)** — already structural; an
    empirical run cannot refute. *No value-add.*

(b) **Steady-state TG at M2 NP=8 is ~27 t/s** — already measured at T4.7
    (`26.49 t/s` C1-steady, `21.62 t/s` C1-staggered). Re-measuring on a
    different staggered shape (heterogeneous lengths instead of uniform)
    would land in the same kernel-saturation band per T4's mechanism finding.
    *No value-add.*

(c) **No requests rejected / context-shifted** — already structural at NP=8
    with 64K per-slot slab vs <2K typical usage. *No value-add.*

The only outcome that would change the verdict: TG at M2 NP=8 unexpectedly
HIGHER than 27 t/s (e.g., 50+) — would mean kernel is NOT saturated at NP=8
and there's headroom paging could unlock. This contradicts T4's binding
finding and would itself be a major discovery; it's structurally implausible
given T4 measured kernel saturation directly on PSKV singlewarp.

⇒ **Empirical run not run** at this probe pass. If user wants it for
belt-and-suspenders binding, it's a ~30-min GPU session (claim coord, build
NP=8 profile variant, run `bench-t4-m3-staggered.sh` adapted to heterogeneous
lengths, measure, restore). Token cost ~10–15k.

This is consistent with `[[feedback_anchor_to_measured_baselines]]`: T4's
measurement is the load-bearing anchor; deriving Tier 5 expectations from
that measurement (rather than from vLLM's published numbers on different
hardware/quant config) is the discipline.

## 5. The 5.84× gap to vLLM — where does it actually come from?

vLLM measured 154.77 t/s NP=8 on same hardware, same INT4 weights, same
prompts (per `[[project_continuous_batching_vs_perslot_dispatch]]`). We
delivered 26.49 t/s at T4 close. Ratio: **5.84×**.

If the gap is NOT in paging-vs-contiguous (per §3 — paging is perf-neutral or
slightly negative on this workload), where IS it?

**Open candidates** (the OpenQ-E menu):

1. **vLLM uses xFormers / FlashAttention 2 kernels**; we use PSKV singlewarp
   sm_75. FA2 has ~3–5× throughput advantage at small batch sizes per the FA2
   paper. If the per-slot kernel cost is what's saturating, FA2-equivalent
   would shift the saturation point dramatically.
2. **vLLM's continuous batching uses a token-major (not slot-major) tick
   model.** Per-tick `n_tokens` aggregates across slots into one larger
   GEMM; we currently dispatch per-slot CUDA streams that don't aggregate.
   T3.5 multi-seq dispatch landed but the kernel was not redesigned to
   exploit the larger effective batch.
3. **vLLM may use kernel fusions** (QKV→attention→output) that we don't —
   each fused stage saves a kernel launch + a VRAM round-trip.

**None of these are addressed by Tier 5 (paged KV).** Tier 5 changes K/V
storage layout; vLLM's perf advantage is plausibly elsewhere in the stack.

## 6. Recommendation — PAUSE TIER 5

**Apply the decision rule's intent over its letter.** The rule was written to
gate against premise mismatch (T1 lesson). The probe surfaces a different
mismatch than the rule was designed to catch: waste % is high, but the
perf-uplift mechanism the rule assumes does not bind on this workload.

**Pivot to OpenQ-E lever #3**: nsys profile vLLM vs ik_llama at NP=8 same
prompts. Identify what saturates at NP=8 in ik_llama and how vLLM bypasses
it. Token estimate: ~30–50k (acquire vLLM trace + ik_llama trace +
side-by-side comparison + writeup).

This becomes the new lever investigation. Outcomes feed:

- If kernel-fusion savings are the gap → new workstream (PHASE_KERNEL_FUSION
  or equivalent), Tier 5 stays paused.
- If FA2 kernel is the gap → ik_llama has options (FA2 backport from
  upstream llama.cpp), Tier 5 stays paused.
- If token-major dispatch is the gap → revisit T3.5 to enlarge effective
  batch, Tier 5 stays paused.
- If paged KV IS load-bearing on top of (1)/(2)/(3) → Tier 5 resumes as part
  of a coherent kernel/dispatch rewrite, not standalone.

Per `[[feedback_no_workarounds]]` + CLAUDE.md §4 (No follow-up cover): **this
is a clean falsification, not a "deferred" decision**. Tier 5 closes
unstarted with measurement-of-record showing the rule's premise does not
bind composed with T4.

## 7. What Tier 5 work IS preserved

The Tier 5 scope-lock checklist's prerequisites and artifacts stand on their
own and are NOT discarded:

- **T5.A audit findings** (`data/t5-audit-findings.md`) — useful future
  reference for any work touching the KV layout, regardless of paging.
- **PHASE_NSTREAM_KV_PERF.md Tier 5 plan section** — preserved as the
  measurement of record for "we considered paging, here is why it did not
  bind on this workload". Future workloads (high concurrency, shared
  prefixes, much longer contexts) may resurface the lever; the plan + audit
  shorten the spin-up cost from ~50k tokens to ~10–15k.
- **The Allium / TLA+ spec scaffolding outline** in the plan — useful as
  reference architecture if paging is revisited in a different framing.

**Not preserved**: the T5.0 commits (4 .allium + 3 .tla + 8 property tests +
trace producer + validator). These are not written; nothing lands.

## 8. Next steps (user decision)

The probe surfaces a falsification. The user has three honest paths:

**Path A — Accept falsification, pivot to nsys profile lever**. Pause Tier 5;
commit a PHASE doc closure section documenting the probe finding; open a new
investigation plan for nsys vLLM-vs-ik_llama. ~30–50k tokens to scope + run.

**Path B — Require empirical confirmation before pausing**. Run M2 NP=8
staggered heterogeneous workload, capture aggregate t/s + GPU utilisation +
KV cache state via per-slot logging. ~10–15k additional tokens. Expected
outcome: confirms §3 analytically (kernel-saturation band ~27 t/s, no
ctx-loss, VRAM headroom). If outcome differs (TG significantly > 27 t/s),
re-enter the analysis with new data.

**Path C — Override the probe and proceed with Tier 5 anyway**. Reasonable
only if user has a perf-uplift mechanism in mind that bypasses §3 (e.g., a
kernel rewrite that composes ONLY with paged layout). Probe finding goes
into the PHASE doc as a recorded concern; Tier 5 spend is on the user's
acknowledgement that the perf gate (GP5.b) is at risk of structural FAIL.

**Default recommendation: Path A.** Honest negative result lands cheap per
`[[feedback_oneshot_then_evaluate]]` — preserves the architecture for future
work and surfaces the real lever direction without burning 150–230k on a
mismatched premise.
