# PHASE45 D8.4 — multi-turn agentic bench evidence

Hardware: 2× Quadro RTX 6000 (TU102, sm_75, 24 GiB), CUDA 13, driver 595.58.03.
Model: Qwen 3.6 27B (V-F1.T1.qq-tool1lossless-vocab-fix.gguf, sha 85fb67a013a06216).
Profile flags: --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1
              -ngl 999 -fa on --ctx-size 262144 --threads 16 --parallel 1
              --batch-size 2048 --ubatch-size 512
              --cache-type-k q4_0 --cache-type-v q4_0
              --k-cache-hadamard --v-cache-hadamard --no-context-shift
Workload: 7-turn agentic conversation (`scripts/agentic-multiturn-corpus.json`)
          via /v1/chat/completions, 384 predicted tokens, temperature 0,
          2 runs per config.

Submodule HEAD: 0c4aefbf @ phase45-decompose (D8.3 landed:
common_speculative MTP impl forwards to libllama spec_loop).

## Numbers

| config                 | tg t/s (run 1) | tg t/s (run 2) | avg     | accept rate | ratio vs A |
|---|---|---|---|---|---|
| A_nomtp                | 29.73          | 29.65          | 29.69   | —           | 1.0000     |
| C_mtp_d3_ikv (hook=1)  | 35.74          | 35.80          | 35.77   | 0.6627      | **1.2049** |
| E_mtp_d3_no_ikv (hook=0)| 35.68         | 35.49          | 35.58   | 0.6559      | **1.1984** |

## Binding test: PASS

PHASE45 D8 binding (PHASE45.md row): "multi-turn agentic bench tg ≥
+19% vs nomtp baseline (the measured C config: -mtp --draft 3 +
INLINE_KV)".

C/A = 1.2049 (+20.5%). Floor of +19% cleared by 1.5 percentage
points. The libcommon shim refactor (D8.3) — common_speculative_state_mtp
forwarding through llama_spec_loop_gen_drafts → llama_spec_mtp_draft
— preserves the algorithmic behavior end-to-end. No regression vs
the pre-extraction path.

## Hook A/B (PHASE45_PHASE39_INTEGRATION §4 reopened lock — RESOLVED)

The integration check flagged PHASE45.md's "no INLINE_KV hook needed
because draft is single canonical writer of layer N-1" as a
provisional lock requiring D8.4 measurement, because PHASE36 had
introduced the hook as a measured perf win.

E (hook OFF) vs C (hook ON):
- E avg tg: 35.58 t/s
- C avg tg: 35.77 t/s
- Δ: 0.19 t/s = 0.53%
- Acceptance rate: hook OFF 0.6559 vs hook ON 0.6627 (±0.7pp)

Both within run-to-run variance (~0.7% in this configuration). Both
clear the +19% floor (E: +19.84%; C: +20.49%).

**Verdict: lock validated.** The hook is genuinely removable. The
PHASE45 single-canonical-writer architecture (DRAFT decoder owns
layer N-1 K/V writes) gives the same end-to-end perf as the
PHASE36 INLINE_KV hook approach. The hook can be deleted at D10
without measurable performance cost.

Why the hook was a win in PHASE36 but is a wash in PHASE45:
PHASE36's hook avoided a per-accept `MTP_OP_UPDATE_ACCEPTED`
decode by folding the layer N-1 K/V write into the verify forward.
With the spec_loop architecture, the draft decoder already writes
layer N-1 during its own forward — the writes commit on accept (no
rollback) or get purged via kv_seq_rm on reject. There's no
separate UPDATE_ACCEPTED step to amortize, so the hook becomes
strictly redundant.

## D8 status

**[x] CLOSED** based on the C/A measurement. The +19% binding gate
is bound on the actual production path (server through ported
common_speculative MTP impl through libllama spec_loop through
extracted spec_mtp_draft primitive). The hook A/B closes the
related architectural question.
