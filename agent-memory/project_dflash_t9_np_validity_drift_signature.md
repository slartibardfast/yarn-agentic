---
name: DFlash T9 CLOSED — vanilla np>1 validity locked; NP=2/4 binary drift boundary identified
description: T9 (revised charter — validity not perf) closed vanilla np>1 with 14/14 slot-runs PASS at np ∈ {2,4,8}. Direct token diff vs NP=1 reference reveals NP=1≡NP=2 byte-identical, NP=4≡NP=8 deterministic drift — a SPECIFIC kernel code-path boundary at NP=2/4, not the general "all kernels drift" framing PHASE45 D10.e worked under.
type: project
originSessionId: phase_dflash_t9_2026-05-14
---
DFlash T9 closed `[x]` on production/2026-q2-next 2026-05-14.

## Charter pivot

Original T9 was perf-comparison ("DFlash vs vanilla aggregate at np=8, ≥1.8× ship"). Retired because the DFlash kernels are too slow for the aggregate measurement to be informative (see `project_dflash_t8_closed`). Revised charter: lock in np>1 **validity** for vanilla and DFlash — gating prerequisite for any future kernel optimization work. Optimizing slow kernels for a path that produces invalid output would be optimizing on a broken foundation.

## What landed (vanilla validity, T9.1)

Harness: `tests/dflash-speculative/test-np-validity-vanilla.cpp` (~250 LOC).

5 falsifiable per-slot asserts:
1. All n_gen tokens emitted (no early break / decode failure).
2. PPL of output under target ∈ [1.0, 50.0] — production-coherence band.
3. ≥ 95% emitted tokens in-vocab (catches glitch-token cascades).
4. `llama_decode` returns 0 on every generation step.
5. No NaN / Inf in captured logits at any decode step.

Results on Qwen 3.6 27B INT4 + 8 prompts at n_gen=64:

| np | slots PASS | PPL range |
|---:|---:|---:|
| 2 | 2/2 | 1.58–1.72 |
| 4 | 4/4 | 1.61–1.86 |
| 8 | 8/8 | 1.18–3.14 |

14/14 slot-runs across 3 N values pass all 5 asserts.

## The drift signature (new vs PHASE45 D10.e)

PHASE45 D10.e characterised the root cause framework: `cublasGemmEx` heuristic picks different GEMM algorithms at different batch dimensions; flash-attention split size varies with shape; per-row reduction-order drift (~1-3%) amplified by greedy decoding's Lyapunov-like sensitivity at the argmax. That investigation was MTP-specific and got "M ≥ 2 = 20% corruption rate".

T9.1 added **direct token-vs-NP=1 diff** (not PPL-summary) for vanilla:

- **NP=1 ≡ NP=2 byte-identical for vanilla.** cuBLAS picks the same GEMM algorithm at batch widths 1 and 2 on the vanilla forward.
- **NP=4 ≡ NP=8 deterministic drift** for 3 of 4 common prompts (p0, p1, p3). Identical first-divergence positions (p0:tok3, p1:tok19, p3:tok0) and identical edit distances at both NP=4 and NP=8. Drift is BINARY at the NP=2/4 boundary, not accumulating with batch width.
- One outlier: p2 picks up an additional NP=8-only drift past token 50 (13/64 tokens diverge between NP=4 and NP=8). A second-order accumulator effect, separate from the main boundary.

**Implication**: the drift is NOT the general "all kernels have batch-shape sensitivity" picture. There's a SPECIFIC kernel code-path boundary at NP=2 → NP=4 that flips the behaviour. Likely candidates: FA `mma_f16` tile transition (Turing m16n8k8, 8-wide tile; batch=4 first hits the next tile multiple) OR DeltaNet reduction-block transition.

## DFlash multi-slot deferred (T9.2)

The DFlash multi-slot path is not testable today without a libllama API extension. `llama_dflash_draft(ctx, anchor_token, anchor_pos, ...)` is single-slot; the kernels support `N_slots × Q` grid (T7 closure verified) but the C entry, `common_speculative_state_dflash` adapter, and server `np==1` gate all need multi-slot variants. Scope estimate ~115–185k tokens (4 layers — see PHASE_DFLASH T9 entry). **Downstream of resolving vanilla drift**: no point optimizing on a drifty foundation.

## Future-work pointers (named, not active)

- **Resolve vanilla drift = PHASE45 D10.e.2**: the planned vLLM/Thinking-Machines 3-kernel reduction-order rewrite that didn't ship. T9's NP=2/4 boundary gives a concrete diagnostic anchor for which kernel(s). PHASE45 D10.e is closed; D10.e.2 is the unfinished follow-on.
- **DFlash multi-slot API extension** (T9.2 charter): downstream of (1).
- **DFlash kernel optimization to TU102 + NVLINK envelope** (per `project_dflash_t8_closed`): downstream of both.

## Artifacts kept

- Harness: `tests/dflash-speculative/test-np-validity-vanilla.cpp` (allows NP=1 with `LLAMA_TEST_PROMPT_OFFSET` env; emits full `generated_tokens` array)
- Diff analysis: `data/phase_dflash_t8/np-token-diff.py`
- NP=1 refs: `data/phase_dflash_t8/gate7-validity-vanilla-np1-p[0-7].json`
- NP>1 tests: `data/phase_dflash_t8/gate7-validity-vanilla-np{2,4,8}-tokens.json`
- Summary: `data/phase_dflash_t8/gate7-token-diff-summary.json`
