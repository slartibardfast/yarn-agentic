---
name: TURBO / HARP_2B novel 2-bit research — abandoned, ship path is UD_IQ2_S Unsloth comparison
description: Consolidated history of the TURBO_KV_4B, TURBO_4B weight, HARP_2B, and trellis-family research. All abandoned 2026; surviving work is the Unsloth comparison benchmark.
type: project
originSessionId: 9b52ee2b-d969-442c-83ba-98396525f6fa
---
The novel 2-bit / 4-bit research thread on Qwen 3.5 (0.8B yardstick, 35B-A3B ship target) was abandoned. The pivot reframed the surviving work as a comparison against Unsloth's existing 2-bit kernels, not an attempt to ship novel quantization. This entry consolidates the chain so the individual step-snapshots can be retired.

## What was tried

### TURBO_KV_4B (RHT Vulkan KV cache)
Walsh-Hadamard Transform pre-quantization for the KV cache, on Vulkan. FA verified correct at the kernel level. Quality gap to Q4_0 measured: codebook NMSE 0.010 vs Q4_0's 0.005 — **a real codebook-quality gap, not a bug**. 9B-Paris baseline and 0.8B test showed the gap; the path was tested, not a kernel defect.

### TURBO_4B weight quantization
Extension of TURBO_KV_4B RHT + codebook from KV cache to weights. Target: QuIP#-class quality at Q4_K_M bitrate. No GGUF implementation existed; the work would have been from-scratch infrastructure. Never landed.

### HARP_2B family (novel 2-bit)
- **Formal target**: 0.8B was yardstick only; PPL ≤ 7.0 on 35B-A3B was the ship gate.
- **PPV finite-blocklength ceiling**: 6.25% MSE at R=2 N=128 Gaussian is Shannon's floor for the source distribution. Below it is impossible without changing the source — a hard information-theoretic wall.
- **AVX2 throughput ceiling**: 68× gap to IQ2_S on 0.8B. Further kernel work was gated on a 35B-A3B PPL win that never came. Ship path was reframed as HARP_2B_S.

### TURBO_2B (parked, then abandoned)
TURBO_2B was parked behind "don't propose deprecation or pick up Phases 37–41 until 35B-A3B eval is active." Workflow rule (kept in feedback memory): exhaust 0.8B before moving to 35B-A3B; parked items unpark as a batch. When 35B-A3B eval finally became active, the pivot abandoned this line entirely.

### Pivot: abandon novel 2-bit research → compare to Unsloth
HARP_2B_S was renamed to UD_IQ2_S_QWEN35; trellis family + TURBO_2B abandoned. Surviving work was a single Unsloth-comparison benchmark task. The rename-on-abandon was itself flagged in feedback memory (`feedback_dont_rename_on_abandon.md`): abandoning is a status change, not a rename — renaming implies ongoing maintenance.

## What durable lessons came out of it

- **The PPV ceiling at R=2 N=128 Gaussian is the source's Shannon floor.** Chasing below it requires changing the source distribution. Hard wall captured in physics; cite this before any future R=2 design attempt.
- **Codebook NMSE is the right correctness metric** for RHT-pre-quant schemes, not "kernel correctness." A kernel can be correct and the codebook can still under-represent the distribution; the latter is what produces the quality gap.
- **AVX2 throughput ceiling of 68× vs IQ2_S** on 0.8B was the gating constraint. Further AVX2 kernel work without a PPL win was rejected.
- **The yardstick rule**: 0.8B is signal only; MoE weight-mass distribution flips conclusions at 35B-A3B scale. Workflow: exhaust the small model first, then batch-unpark items at the big-model evaluation gate.
- **Don't rename a feature on the way out.** Abandoning is a status change.

## Surviving artifacts and references

- **`feedback_dont_rename_on_abandon.md`** — lesson from the HARP_2B_S → UD_IQ2_S_QWEN35 rename.
- **`feedback_0.8b_exhaust_then_35b.md`** — the workflow rule.
- **`feedback_08b_is_signal_for_35b.md`** — within-stderr 0.8B deltas can flip at MoE scale; don't write them off.
- **`feedback_vec_dot_type_Q8K.md`** — vec_dot_type=Q8_K declaration is mandatory for new low-bit types to use vpmaddubsw's 32 MAC/cycle path.

Any future low-bit work should start from the Unsloth UD_IQ2_S comparison baseline rather than re-inventing the trellis or RHT-codebook approaches.
