# PHASE_NPC_HANDOVER — pick up at NPC.4 (intra-layer-2 localization)

**Branch**: `production/2026-q2-next`
**Plan**: `PLAN_NP_CLOSURE.md`
**Predecessor handover**: `PHASE_NP_CLOSURE.md` (superseded)

## TL;DR — NPC.2 was wrong, NPC.4 starts over

The 2026-05-17 NPC.2 claim that `ssm_conv` was the divergence root
cause is **incorrect**. Re-verified empirically by removing the
n_kv==1 early-return at `ssm-conv.cu:639` so NP=1 and NP=2 dispatch
the SAME kernel — NP=1 captures came out byte-identical to pre-fix,
NP=2 captures changed bytes BUT slot-0 vs NP=1 divergence pattern
held with identical magnitude (max|Δ|=6.855e-06 at layer 2 ub2 idx=0).

The change has been reverted. See the 2026-05-17 MEMORY entry titled
"NPC.4 candidate (c) attempted: NOT the fix" for details.

## What the captures actually show

| Layer | Type | NP=1 vs NP=2 slot-0 @ ub2 |
|---|---|---|
| 0 | DeltaNet | IDENTICAL |
| 1 | DeltaNet | IDENTICAL |
| 2 | DeltaNet | **DIFFERS** — first diff, max\|Δ\|=6.855e-06 |
| 3 | Full-attn | DIFFERS, max\|Δ\|=2.892e-03 (1000× amplification by FA) |
| 4–63 | mixed | DIFFERS, max\|Δ\| grows |

Layers 0 and 1 are byte-identical. So layer-2 INPUT is the same in
NP=1 and NP=2. Yet layer-2 OUTPUT diverges. **The divergence is
created inside a single layer's compute path, not by cross-layer
state accumulation.**

## NPC.4 retake — localize the diverging op INSIDE layer 2

Layer 2 contains (in order):
1. RMSNorm on residual
2. DeltaNet projections: Q, K, V, G linear (cuBLAS GEMM)
3. ssm_conv (1D conv) — RULED OUT above
4. chunk_delta_rule kernel (delta-net.cu)
5. Output projection
6. Residual add
7. RMSNorm
8. MoE: gate selector → expert MLPs → reduce
9. Residual add

Suspect ranking (cheapest to instrument first):
1. **cuBLAS GEMM** for Q/K/V/G — cuBLAS picks algorithm by batch
   shape. n_tokens=1 (NP=1 decode) vs n_tokens=2 (NP=2 decode) can
   hit different algos with different reduction orders. Even with
   workspace pinned + TF32-off baked, the algo selector can vary.
2. **delta-net `chunk_delta_rule`** — already baked use_256
   default. Could still have intra-layer state-update sub-ULP drift
   that surfaces only at layer 2 (not layers 0/1) for unclear reasons.
3. **MoE gate / expert dispatch** — only_active_experts already
   ruled out at NPC.2 (`npc2-decode-no-ooae-np{1,2}/`). Could still
   differ in some other branch.
4. **RMSNorm or some other unbaked op** — least likely given the
   determinism work already invested.

## Tooling needed

The existing `llama-state-capture` captures per-layer `l_out` (the
layer-output tensor). To localize inside layer 2, we need to capture
INTERMEDIATE tensors: the output of each op within the layer.

Two options:
- (a) Add per-op cb_eval interception that writes named intermediate
  nodes from layer 2 (e.g. `Qcur-2`, `Kcur-2`, `attn_out-2`,
  `ffn_inp-2`, `ffn_out-2`). Probably 50–100 lines of new code in
  `llama-state-capture.cpp` plus an extension to the cb_eval filter.
- (b) Run with full graph capture (`GGML_DUMP_GRAPH` or similar) and
  diff every node by name. Heavier but no new code.

(a) is the smaller next step — extend the existing `--tensors`
argument to accept multiple comma-separated names beyond `l_out`.

## What to do on resume

1. Read the 2026-05-17 NPC.4 MEMORY entry for full evidence.
2. Decide between option (a) and (b) tooling — recommend (a).
3. Extend the capture tool with intra-layer tensor names.
4. Re-run NP=1 vs NP=2 captures with intra-layer names for layer 2.
5. Find the FIRST tensor inside layer 2 whose slot-0 row diverges.
6. That op is the actual NPC.4 root cause. Fix it. Verify with
   per-tensor capture + production harness.

## What NOT to redo

- Don't relocalize CY.F.17 (MMQ stream_K), A.1' (FA per-slot-kv),
  singlewarp, delta-net `use_256`, cuBLAS workspace+TF32. All sealed.
- Don't retest only_active_experts. Already ruled out.
- Don't trust prefill-only captures — bug is decode-only.
- Don't trust ssm_conv as the source — proven NOT the source above.

## Evidence dirs (preserved on /opt/models/)

NPC.2 baselines (pre-fix):
```
/opt/models/yarn-audit-data/npc2-decode-np{1,2}/           # default
/opt/models/yarn-audit-data/npc2-decode-no-ooae-np{1,2}/   # -no-ooae
```

NPC.4 candidate (c) attempt (post-revert): captures saved but the
binary that produced them is no longer current:
```
/opt/models/yarn-audit-data/npc4-fix-np{1,2}/
```

NPC.1 / NPC.3 (harness signatures):
```
/opt/models/yarn-audit-data/npc1-default/run-20260517T164308/
/opt/models/yarn-audit-data/npc3-ub2048/run-20260517T164721/
/opt/models/yarn-audit-data/npc3-ub1024/run-20260517T165010/
/opt/models/yarn-audit-data/npc3-ub200/run-20260517T165249/
```

## Commands quick reference

Reproduce harness failure (single-GPU):
```
DEVICE=CUDA0 RESULTS_DIR=/tmp/foo bash scripts/test-production-np-determinism.sh
```

Capture l_out per-layer decode-step state:
```
LLAMA_CAPTURE_DECODE_STEPS=2 ik_llama.cpp/build/bin/llama-state-capture \
  -m /opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf \
  --prompt-file data/audit-prompts/long/prompt-00.txt --prompt-id long-p00 \
  --tensors l_out --layers all --np {1|2} --out-dir <DIR> \
  --device CUDA0 -ngl 999 -fa on --ctx-size $((8192 * NP)) \
  --batch-size 2048 --ubatch-size 512 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --k-cache-hadamard --v-cache-hadamard --no-context-shift
```

Compare per-layer slot-0:
```
python3 scripts/compare-slot0.py NP1_DIR NP2_DIR --ub 2
```

## In-flight commits (session 2026-05-17 part 2)

- `f3100f3` PHASE_NP_CLOSURE handover (now superseded)
- `c5cb44a` handover doc (also superseded)
- new this session: NPC.4 retake MEMORY entry + this revised handover.

Pending untracked test mod in submodule (`tests/dflash-speculative/
test-trace-2-intra-layer-capture.cpp`) is pre-existing — not from this
session, leave for the owner.
