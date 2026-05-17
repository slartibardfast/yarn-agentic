# PHASE_NPC_HANDOVER — pick up at NPC.4 fix (ffn_up_gate MoE GEMM)

**Branch**: `production/2026-q2-next`
**Plan**: `PLAN_NP_CLOSURE.md`
**Status**: NPC.4 LOCALIZED. The diverging op is named. Next: fix it.

## TL;DR — layer 2 divergence is in the MoE expert GEMM, not DeltaNet

Walked every named intra-layer-2 tensor in fire order with the extended
`llama-state-capture` tool. Layer 2 decode-step 0 NP=1 vs NP=2:

| order | name | sz1 | szk | verdict | first | max\|Δ\| |
|---|---|---|---|---|---|---|
| 0–32 | DeltaNet attention path | … | … | **IDENTICAL** | — | — |
| 33 | `ffn_norm-2` | 20480 | 40960 | **IDENTICAL** | — | — |
| 34 | `ffn_up_gate-2` | 69632 | 139264 | **DIFFERS** | 1 | 1.490e-08 |
| 35 | `l_out-2` | 20480 | 40960 | **DIFFERS** | 0 | 6.855e-06 |

DeltaNet (incl. ssm_conv, conv_states, q/k/v_fused, delta_net_fused_raw,
new_state, attn_output) is byte-identical slot-0 for NP=1 vs NP=2 — fully
exonerated. The MoE input (`ffn_norm-2`) is identical. The first
diverging tensor is `ffn_up_gate-2` at element index 1 with a 1-ULP fp32
drift; that drift then amplifies to 6.855e-06 by `l_out-2` after gate/
down/expert-reduce.

## What `ffn_up_gate-2` actually is

In `src/llama-build-context.cpp` →  `llm_build_std_moe_ffn` →
`build_qwen35moe`. It is the fused up+gate projection on each selected
expert MLP:

```cpp
ggml_tensor * up_gate = ggml_mul_mat_id(ctx, expert_up_gate_weights,
                                         ffn_norm_out, expert_ids);
cb(up_gate, "ffn_up_gate", il);
```

Per-expert weight tensor: `model.layers[il].ffn_up_gate_exps`. Dispatch
goes through `ggml_mul_mat_id` (the "experts" batched matmul), which on
the CUDA backend may pick:
- MMQ (with stream_K, baked deterministic via CY.F.17), or
- cuBLAS GEMM (shape-dependent algo selection).

Hypothesis: the expert-MLP MMM path either (a) falls back to cuBLAS
for the batched-id case despite stream_K being baked, or (b) uses an MMQ
variant whose tile-shape choice still depends on n_tokens.

## What to do on resume — NPC.4 fix

1. Source-walk the CUDA dispatch for `ggml_mul_mat_id` to confirm which
   kernel actually runs for n_tokens=1 vs n_tokens=2 on a quantized
   weight tensor. Files of interest:
   - `ggml/src/ggml-cuda/mmq.cu`
   - `ggml/src/ggml-cuda/mmid.cu` (or equivalent for expert/batched-id)
   - `ggml/src/ggml-cuda/ggml-cuda.cu` dispatch entrypoint
2. If cuBLAS fires for the expert GEMM, find a way to either:
   - Force MMQ stream_K for `mul_mat_id` regardless of M.
   - Pin a cublasGemmEx algo that doesn't change with M.
3. Re-run the layer-2 intra-layer capture and confirm `ffn_up_gate-2` is
   IDENTICAL slot-0 NP=1 vs NP=2.
4. Re-run the full per-layer capture (l_out only, all 64 layers) and
   confirm layer 2..63 collapse to IDENTICAL.
5. Re-run the production NP-determinism harness; expect byte-identity
   close.

## Tools landed this session

- `ik_llama.cpp/examples/llama-state-capture/llama-state-capture.cpp`
  (submodule commit `eb93b39f`) — new flags:
  - `--all-in-layer` — capture every named non-quantized tensor at the
    listed layers (no prefix filter).
  - `--decode-only` — skip prefill ubatches.
  - Phase-tagged output paths: `{OUT_DIR}/{phase}/layer{LL}/{name}.ub{N}.bin`
    where phase is `prefill` or `decode-{step}`.
  - Manifest now carries `phase` and `order` (cb_eval fire order).
- `scripts/compare-intra-layer.py` — walks the NP=1 manifest in order,
  joins on (phase, name, ubatch_idx) with the NP=K manifest, compares
  slot-0 floats per tensor, prints first divergence.

## What NOT to redo

- Don't relocalize DeltaNet — empirically exonerated tensor-by-tensor.
- Don't relocalize ssm_conv — order 9 (`conv_output_raw-2`) is
  byte-identical.
- Don't retest `only_active_experts` — NPC.2 ruled it out; the bug is
  in the GEMM kernel inside the expert, not the gating that picks the
  expert.
- Don't trust prefill-only captures — decode-only.

## Reproduce

```bash
# NP=1 capture (intra-layer-2, decode-only)
LLAMA_CAPTURE_DECODE_STEPS=2 \
  ik_llama.cpp/build/bin/llama-state-capture \
  -m /opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf \
  --prompt-file data/audit-prompts/long/prompt-00.txt --prompt-id long-p00 \
  --all-in-layer --decode-only --layers 2 --np 1 \
  --out-dir /opt/models/yarn-audit-data/npc4-intra-np1 \
  --device CUDA0 -ngl 999 -fa on --ctx-size 8192 \
  --batch-size 2048 --ubatch-size 512 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --k-cache-hadamard --v-cache-hadamard --no-context-shift

# NP=2: swap --np 2, --ctx-size $((8192 * 2)),
# --out-dir /opt/models/yarn-audit-data/npc4-intra-np2

# Compare:
python3 scripts/compare-intra-layer.py \
  /opt/models/yarn-audit-data/npc4-intra-np1 \
  /opt/models/yarn-audit-data/npc4-intra-np2 \
  --phase decode-0 --layer 2
```

## In-flight commits (session 2026-05-17 part 3)

- `eb93b39f` (submodule) — llama-state-capture: --all-in-layer +
  --decode-only + phase-tagged outputs.
- new parent commit — NPC.4 localized; MEMORY + comparator + submodule
  bump.
- pending: this handover rewrite.

## Evidence preserved

- `/opt/models/yarn-audit-data/npc4-intra-np{1,2}/` — intra-layer-2
  captures with the new phase-tagged layout. 36 NP=1 records, 71 NP=2
  records. Comparator output included in MEMORY entry.
- Older `npc{1,2,3,4}-*` dirs from prior NPC steps kept for context.
