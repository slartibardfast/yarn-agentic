# T6.3 — DFlash deep-dive (axes 1+2+3+4)

Closes T6.3 per T6.1 follow-on. Four-axis characterisation of DFlash speculative decoding on Qwen 3.6 27B production (`--parallel 2`, ctx 262144 per slot, Q4_0+Hadamard KV, locked 1455 MHz).

- Matrix baseline: `t6.1-matrix-fixed-20260523T211929/cell-prod-baseline` (NP=8 DFlash on dm=4) and `cell-no-dflash` (NP=8 DFlash off).
- Sweeps: `t6.3-sweeps-20260523T230111/` (axes 2 + 3).
- nsys trace: `t6.3-nsys-dflash-20260523T225622/summary.md` (axis 4).

---

## Axis 1 — per-prompt acceptance histogram (offline)

Per-task acceptance for the matrix prod-baseline NP=8 DFlash-on cell, decomposed by prompt:

| prompt_idx | rate | prompt |
|---:|---:|---|
| 0 | 0.608 | Explain the difference between latent diffusion and... |
| 1 | 0.392 | Summarize the plot of King Lear in one... |
| 2 | 0.650 | Write Python code that fits a 2nd-degree polynomial to a... |
| 3 | 0.408 | What are the main causes of the Peloponnesian... |
| 4 | 0.488 | Translate to French: The early-morning fog lingered over... |
| 5 | 0.426 | List five practical steps for reducing memory allocations... |
| 6 | 0.454 | Describe the role of telomeres in cellular... |
| 7 | 0.808 | Write a haiku about a printing... |

Spread: **min 0.392, max 0.808, mean 0.529, range 0.416**.

**Finding:** acceptance is strongly content-dependent. Structured-form / code / haiku at top of range (predictable continuations); open-ended natural-language prose at bottom (King Lear plot, Peloponnesian War). The matrix-reported aggregate 0.42 acceptance is the cumulative-dflash-stat mean; the per-task mean here is 0.529. Same population, different counting (cum-stat includes warmup + intermediate decode states).

---

## Axis 2 — draft_max sweep at NP=8 (DFlash on)

Baseline: draft_max=4 at 11.03 t/s (matrix prod-baseline).

| draft_max | aggregate t/s | Δ vs dm=4 | acceptance mean | acceptance range |
|---:|---:|---:|---:|---:|
| **4** (baseline) | 11.03 | — | 0.529 | 0.392–0.808 |
| 2 | 11.58 | +5.0% | 0.732 | 0.614–0.933 |
| 3 | 11.21 | +1.6% | 0.610 | 0.419–0.831 |
| 5 | 11.27 | +2.2% | 0.533 | 0.392–0.676 |
| 6 | 10.59 | -4.0% | 0.502 | 0.330–0.716 |

**Finding:** acceptance falls monotonically as `draft_max` grows (0.732 at dm=2 → 0.502 at dm=6) — longer drafts mean more downstream tokens whose target log-probs fall below the drafter's distribution. Throughput peaks at **dm=2 (11.58 t/s, +5.0% over the dm=4 default)**, because at higher dm the drafter forward cost is paid for more tokens that will be rejected. **Critically, even at the optimum dm=2 the DFlash-on cell is still -43.4% vs no-DFlash (20.45 t/s)** — tuning draft_max does NOT recover DFlash to net-positive on this workload.

---

## Axis 3 — NP sensitivity (DFlash on vs off across NP)

Server `--parallel 2` throughout; NP varies client-side concurrent prompt count. draft_max=4 for DFlash-on cells.

| NP | DFlash on t/s | DFlash off t/s | Δ (on vs off) | accept_mean (on) |
|---:|---:|---:|---:|---:|
| 1 | 11.46 | 18.32 | -37.4% | 0.575 |
| 2 | 12.99 | 20.69 | -37.2% | 0.625 |
| 4 | 11.14 | 20.58 | -45.9% | 0.541 |
| 8 | 11.03 | 20.45 | -46.1% | 0.529 |

**Finding:** **DFlash is net-negative at every NP we measured (-37.4% at NP=1, -37.2% at NP=2, -45.9% at NP=4, -46.1% at NP=8).** The penalty narrows slightly at lower NP (less slot-contention overhead in the drafter forward path) but never crosses to net-positive. The no-DFlash side is near-flat across NP (~18–21 t/s, bench is kernel-saturated with 2 production slots regardless of how many client prompts queue). The DFlash side hovers ~11–13 t/s because the drafter+verify cycle is the per-token bottleneck. **At gate0's content mix, DFlash does not benefit from lower concurrency** — slot-contention is not what makes DFlash slow; the drafter's absolute cost relative to target Q4_0 matmul cost is.

---

## Axis 4 — nsys decode trace (DFlash on, NP=8)

Kernel attribution under DFlash:

- **`mul_mat_f16_pinned_kernel_wmma` 17.8%** — drafter forward (F16 weights, 5-layer Qwen 3.6 sidecar).
- `ncclDevKernel_AllReduce_Sum_f32_RING_LL` 26.5% — graph-split cross-GPU sync (NVLink installs 2026-05-24; expected to reduce dramatically).
- `mul_mat_q_split_k<Q4_0>` 13.3% — target Q4_0 matmul (down from 31.0% in T6.2 no-DFlash; DFlash absorbs many target steps).
- `cutlass_75_wmma_tensorop_h161616gemm` 12.2% — f16 cutlass (likely lm_head).
- `mul_mat_q_split_k<type 159>` 3.0% — Hadamard-rotated Q4_0_AR16 variant.
- `flash_attn_per_slot_kv_singlewarp_kernel` 2.1% — target attention (T3.5 ILP recovery worked; tiny share).
- `<unnamed>::attention_kernel` 1.3% — drafter's attention.
- `<unnamed>::q_norm_rope_kernel` 0.4% — drafter's QK norm.

Full table: [`t6.3-nsys-dflash-20260523T225622/summary.md`](/home/llm/yarn-agentic/data/t6.3-nsys-dflash-20260523T225622/summary.md).

**Cost attribution:** 
drafter forward = 17.8% of GPU time. At baseline 0.42 accept rate + dm=4, expected tokens-saved per drafter call = ~0.42 × 4 = 1.68 (best case if all drafts accepted up to first rejection). The drafter saves target matmul calls but adds its own forward, AllReduce contributions, and the inject_kv_fused / verify path overhead — at NP=8 with low-acceptance prompts, drafter cost exceeds savings, yielding the matrix's -46% verdict.

---

## Combined verdict

1. **DFlash is net-negative at every measured axis** (NP ∈ {1,2,4,8}, `draft_max` ∈ {2,3,4,5,6}). The matrix-headline -46.1% at NP=8 dm=4 is not an extreme case; it's representative of the operating region.

2. **Acceptance is content-dominated** (axis 1). Per-prompt acceptance ranges 0.39 (King Lear prose) → 0.81 (haiku), driven by prompt structure not by draft_max or NP. Tuning will not flip net-positive on gate0's content mix.

3. **dm=2 is the throughput sweet spot** for DFlash-on. Acceptance 0.73, +5% over dm=4 default. If DFlash stays on, dm=2 is a better default. But this is still ~43% below the no-DFlash side at every NP.

4. **Drafter forward cost (17.8% of GPU time, axis 4) is the proximate cost source.** The drafter saves target Q4_0 matmul time (T6.2 had it at 31%; T6.3 DFlash-on has it at 13.3%) — but at gate0 acceptance, the savings don't pay for the drafter's own forward + AllReduce + verify cost. DFlash works when target arithmetic intensity is high enough that drafter cost is amortised; gate0's varied prompts at Q4_0 KV are not that regime.

5. **`bench-t3.8-m3` still measures DFlash as a clear win.** That bench is identical-prompt × short × low concurrency — a near-best case for spec decoding. It does not generalise to mixed-prompt server workloads. T5/T3 closure docs that cited DFlash uplifts were measuring this case; gate0 is the production-shape case.

## Production implication (informational, not advocacy per T6 discipline)

Production profile (`qwen36-27b-x2-dflash.sh`) ships DFlash ON. At gate0-shape workloads it is materially slower than DFlash OFF. The decision to keep / drop / make-configurable DFlash on the production profile is a downstream call informed by this data; T6.3 itself does not advocate either way. Two factors worth weighing:

- The workload mix on this server: how often does production see bench-t3.8-m3-shape traffic (where DFlash wins) vs gate0-shape traffic (where it loses)? Production-mix observation belongs to a separate item, not T6.3.
- The NVLink install (2026-05-24) changes the AllReduce share dramatically. Re-measure these axes post-NVLink if the answer matters (T6.3 results will shift; magnitude TBD).

## T6 follow-ons opened by T6.3

- **T6.3.b** (NEW) — re-measure axes 2+3 after NVLink install. Drafter F16 traffic currently pays full PCIe-peer-access AllReduce cost; NVLink will reduce drafter share and may shift the dm=2 sweet spot.
- **T6.3.c** (NEW) — characterise DFlash on identical-prompt × short workload (bench-t3.8-m3 shape) to quantify the **upper-bound** acceptance ceiling on this drafter/target pair. Bounds the workload range where DFlash net-helps.
- **T6.7** — PSKV singlewarp deep-dive at the 2.1% residual under DFlash composition (already in T6 backlog).

## Discipline note (CLAUDE.md §4 / §5)

T6.3 closes with a net-negative finding across all measured axes. Per §4 (no follow-up cover): this is the result, named as such, not deferred to a hopeful future. Per §5: the data binds on the step's claim — workload-shape sensitivity was characterised on the four axes named in scope, and the answer is content-dominated acceptance × drafter-cost-exceeds-savings on gate0.
