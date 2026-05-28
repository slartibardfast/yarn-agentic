# Op coverage — Qwen3.5-9B q4km, Vega 64

## Setup

`GGML_SCHED_DEBUG=2 GGML_VK_VISIBLE_DEVICES=1 llama-server` driven
with one completion request (the standard profiling workload). The
scheduler prints its assignment dump for every graph construction,
tagged by backend.

Parser: `parse_sched_debug.py` matches node lines of the form:

```
node #   N (  OP_NAME):   tensor_name  ( size ) [BackendTag    ] ...
```

## Results

- **Total nodes observed across all graph builds**: 17744
- **Total `## SPLIT #N: <backend>` lines**: 16
- **Distinct backends used**: exactly one (`Vulka` — truncated display
  of `Vulkan0`)
- **Ops landing on any CPU backend**: **ZERO**

Every single node in the Qwen3.5-9B q4km graph — across prompt eval,
MTP drafting, and MTP verification — is assigned to the Vulkan backend.
Zero `supports_op = false` rejections, zero CPU fallbacks, zero CPU↔GPU
copies during forward passes.

## Op histogram (all 17744 nodes)

| Op              | Count |  % of total |
|-----------------|------:|------------:|
| MUL_MAT         | 4144  | 23.4%       |
| RMS_NORM        | 1808  | 10.2%       |
| MUL             | 1808  | 10.2%       |
| GET_ROWS        | 1584  | 8.9%        |
| CPY             | 1552  | 8.7%        |
| ADD             | 1056  | 6.0%        |
| FUSED           |  912  | 5.1%        |
| SCALE           |  768  | 4.3%        |
| UNARY           |  768  | 4.3%        |
| L2_NORM         |  768  | 4.3%        |
| GLU             |  528  | 3.0%        |
| CONCAT          |  400  | 2.3%        |
| SSM_CONV        |  384  | 2.2%        |
| GATED_DELTA_NET |  384  | 2.2%        |
| ROPE            |  288  | 1.6%        |
| SET_ROWS        |  288  | 1.6%        |
| FLASH_ATTN      |  144  | 0.8%        |
| CONT            |  144  | 0.8%        |
| ARGMAX          |   16  | 0.1%        |

(Smaller ops omitted from the table.)

The `FUSED` entries are Phase 3/4's `GGML_OP_FUSED` scheduler op
(SILU_MUL, SIGMOID_MUL, GATE_PREP, ARGMAX_GET_ROWS-family). **912
fused dispatches per workload** — the fusion infrastructure is firing
on every forward pass.

## Findings

1. **Phase 4's mission is complete.** The scheduler has nothing to
   split off Vulkan. `graph splits = 1` (the empty CPU graph-entry
   sync) is the true floor.
2. **MUL_MAT dominates op count** (23%) but from step 1 we know its
   GPU time is dominated by `MUL_MAT_VEC` (single-token generation).
   The 4144 number is across all graph builds including prompt eval,
   where actual batched `MUL_MAT q4_K m=x n=218 k=y` runs very fast
   (6.3 TFLOPS on Vega, see step 1).
3. **FUSED op usage is significant** (912 ops = 5.1% of nodes).
   The scheduler is emitting fused dispatches wherever the pattern
   matches, matching the non-trivial fusion delta from step 2.
4. **GATED_DELTA_NET and SSM_CONV each run 384 times** — these are
   the SSM layers in Qwen3.5-9B's hybrid architecture. They're
   already Vulkan-covered; no follow-up Phase 6-style remediation
   needed.
5. **ARGMAX runs 16 times** — once per output token batch (MTP verify
   step), fused with MUL_MAT_VEC vocab head. Matches the per-op
   timing from step 1.

## Follow-ups not needed

The Phase 2 plan had a "what ops still land on CPU?" question for
step 8 as fallback when `test-backend-ops` isn't available. The
answer is: none. Phase 4 closed the coverage gap, and this workload
has no CPU-bound ops remaining.

**No follow-up Phase 6-style op remediation work is justified by
this data.** Future Phase 5+ effort should target per-op kernel
speed (the q5_k vocab head is still the top target per step 6),
not coverage gaps.

## Artifacts

- `vega-sched-debug-2026-04-11T233500Z.stderr` — raw scheduler debug output
- `summary-2026-04-11T233500Z.json` — parsed histogram JSON
- `parse_sched_debug.py` — the parser (from step 0 helper drop)
- `SUMMARY.md` — this file

## Caveats

- Single workload at n_ctx=4096 with the default fixed prompt. Other
  workloads (very long context, batched requests, multi-turn agent
  chains) may trigger different op mixes.
- The scheduler emits one split per graph build, and I observed 16
  splits across the single-prompt workload. The graph-build cadence
  is roughly 1 per token generated (prompt eval + per-token decoding),
  so 16 splits ≈ 16 graph builds for 218-prompt + drafting cycle.
