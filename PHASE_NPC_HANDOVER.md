# PHASE_NPC_HANDOVER — pick up at NPC.4

**Branch**: `production/2026-q2-next`
**Plan**: `PLAN_NP_CLOSURE.md`
**Predecessor handover**: `PHASE_NP_CLOSURE.md` (superseded by NPC.2 root cause)

## TL;DR

Root cause of production-stack NP-cross byte-identity failure is localized
to `ssm_conv` op (DeltaNet 1D conv) in
`ik_llama.cpp/ggml/src/ggml-cuda/ssm-conv.cu:639`. The branch picks
**different kernels** based on `n_kv`:

- NP=1 → `ssm_conv_single_seq_f32` fast path (early return)
- NP>1 → `ssm_conv_multi_seq_unique_f32_kernel` (different reduction order)

Both produce the same math; bits differ by sub-ULP fp32. Drift accumulates
through DeltaNet layers, FA at layer 3 amplifies 1000×, argmax flips at
decode step ~10. **NPC.1 + NPC.2 + NPC.3 are closed.** NPC.4 (the fix)
has not been attempted yet.

## Where things stand

| Phase | Status | Evidence |
|---|---|---|
| NPC.1 single-GPU baseline | ✓ closed | reproduces D-α and D-β on `--device CUDA0` |
| NPC.2 D-α localization | ✓ closed | first divergence at layer 2 decode step 0, max\|Δ\|=6.85e-6 |
| NPC.3 D-β localization | ✓ closed | UBATCH_SIZE=2048 → NP=4≡NP=8 byte-identical |
| **NPC.4 fix + bake** | **pending** | three candidates in MEMORY |
| NPC.5 multi-GPU closure | pending | gated on NPC.4 |
| NPC.6 ship | pending | gated on NPC.5 |

Token spend so far: ~120k. NPC.4 budget per plan: 30–80k.

## Critical file + line

```
ik_llama.cpp/ggml/src/ggml-cuda/ssm-conv.cu:639
  if (n_kv == 1 && src3->ne[0] == 1) {
      // single-seq fast path (lines 642–673), early return
      return;
  }
  // multi-seq path (lines 676–840): different kernels
```

The multi-seq path then attempts a *runtime* single-seq fast path inside
itself (lines 762–798), gated by `single_ok_d` set by
`ssm_conv_detect_runtime_single_seq`. At NP=2 decode step (2 tokens, 2
distinct slot ids), the detector likely rejects single_ok (because tokens
go to different slots), so the slow path fires.

## NPC.4 fix candidates (cheapest first)

**Candidate (c) — harmonize via runtime_single_seq fast path** (cheapest)
1. Read `ssm_conv_runtime_single_seq_f32` (lines 213+ of ssm-conv.cu) and
   compare its arithmetic against `ssm_conv_single_seq_f32`. Spec says
   they "mirror" each other. If they are bit-identical for the same
   slot, then the goal is to make NP=2 decode hit the same runtime
   fast path that NP=1 prefill+decode hits.
2. At NP=2 decode step, the detector check is "all tokens go to the
   same slot." That fails by construction. Modify the detector to be
   per-slot: a separate `single_ok[slot]` array, and dispatch a
   per-slot runtime fast path. Or change the multi_seq_unique kernel to
   per-slot-call the single_seq math.
3. Risk: if the runtime fast path arithmetic isn't actually bit-equal
   to the single_seq fast path, this fix doesn't close the binding.
   First-step test: capture conv_output at NP=1 single_seq vs NP=2 with
   the runtime fast path forced — do they match?

**Candidate (a) — route NP=1 through multi-seq path** (cheap, may regress)
1. Delete or env-gate the `n_kv == 1 && src3->ne[0] == 1` shortcut at
   line 639.
2. NP=1 now takes the multi-seq path, same as NP>1. Per-slot output
   should match across NPs by construction.
3. Risk: perf regression for the common NP=1 case. Measure with
   `llama-bench` before/after. If regression > 5%, fall back to (c) or
   (b).

**Candidate (b) — restructure multi_seq kernel reduction** (expensive)
1. Refactor `ssm_conv_multi_seq_unique_f32_kernel` so its per-slot
   reduction order matches `ssm_conv_single_seq_f32` exactly.
2. Risk: highest implementation cost. Save for last.

## Verification plan for NPC.4 fix

1. Rebuild with the fix.
2. Re-run NPC.2 captures (single-GPU NP=1 vs NP=2 with
   `LLAMA_CAPTURE_DECODE_STEPS=2`):
   ```
   /opt/models/yarn-audit-data/npc4-fix-np1/
   /opt/models/yarn-audit-data/npc4-fix-np2/
   ```
3. Use `scripts/compare-slot0.py` to verify all layers 0–63 IDENTICAL
   at decode step 0 (ub2).
4. If that passes, run the production harness:
   ```
   DEVICE=CUDA0 bash scripts/test-production-np-determinism.sh
   ```
   Expect PASS at NP=1/2/4/8 with default prompt + vlong.
5. Run F.3 corpus (`data/audit-prompts/`) for closure binding.
6. Then NPC.5: re-run at `--device CUDA0,CUDA1`. If PASS, done.

## What NOT to redo (per anti-pattern flags)

- Don't relocalize CY.F.17 (MMQ stream_K), A.1' (FA per-slot-kv route),
  singlewarp, delta-net `use_256`, cuBLAS workspace+TF32. All sealed.
- Don't re-test `only_active_experts` — already confirmed NOT the source
  (NPC.2 evidence dir `npc2-decode-no-ooae-np{1,2}/`).
- Don't trust prefill-only F.1 captures as closure binding. The bug is
  decode-only; the harness uses N_PREDICT=64 decode tokens.
- Don't trust the capture's *unit-test* arithmetic — verify bit-equality
  via the F.1 + binder loop at production state, single-GPU first.

## Evidence dirs (preserved on /opt/models/)

NPC.1 / NPC.3 (harness signatures):
```
/opt/models/yarn-audit-data/npc1-default/run-20260517T164308/
/opt/models/yarn-audit-data/npc3-ub2048/run-20260517T164721/
/opt/models/yarn-audit-data/npc3-ub1024/run-20260517T165010/
/opt/models/yarn-audit-data/npc3-ub200/run-20260517T165249/
```

NPC.2 (captures):
```
/opt/models/yarn-audit-data/npc2-np{1,2}/                  # prefill only, all layers IDENTICAL
/opt/models/yarn-audit-data/npc2-decode-np{1,2}/           # +2 decode steps, layer 2+ DIFFERS
/opt/models/yarn-audit-data/npc2-decode-no-ooae-np{1,2}/   # with -no-ooae, still DIFFERS
```

## Commands quick reference

Reproduce harness failure (single-GPU):
```
DEVICE=CUDA0 RESULTS_DIR=/tmp/foo bash scripts/test-production-np-determinism.sh
```

Capture decode-step state:
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

## In-flight commits (session 2026-05-17)

- `f3100f3` PHASE_NP_CLOSURE handover (now superseded)
- `c5cb44a` handover doc (superseded)
- new this session (HEAD): NPC.4 prep — capture decode-steps env,
  harness ubatch overrides, compare-slot0.py, NPC.2 root cause in
  MEMORY.

Pending untracked test mod in submodule (`tests/dflash-speculative/
test-trace-2-intra-layer-capture.cpp`) is pre-existing — not from this
session, leave for the owner.
