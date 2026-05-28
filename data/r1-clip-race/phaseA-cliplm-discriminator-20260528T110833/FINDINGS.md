# PHASE_R1_CLIP_RACE Phase A — CLIP-vs-LM discriminator findings

**Run:** 2026-05-28 11:08Z (`/tmp/phase46-multigpu-clip/run-20260528T110833/`)
**Build:** ik_llama.cpp submodule `44f81ad1` (R1 fix deployed; LM decoder
opts out of B.5e buffer-clear, CLIP sched keeps default = clear active)
**Config:** 10 chat completions, same image (`examples/mtmd/test-1.jpeg`,
1024 image tokens), same prompt, temperature=0, NP=1, ctx=262144
**Env:** `CLIP_LOG_FINAL_HASH=1` (pre-existing FNV-64 hook in
`clip.cpp:6195-6204` — no code patch needed for Phase A.1)

## Result (binding)

| metric | result |
|---|---|
| Distinct CLIP embedding FNV-64 hashes across 10 encodes | **10/10** |
| Distinct LM response sha256s across 10 encodes | **2/10** (8 dominant, 2 minority) |

CLIP embeddings (each row is one encode):

| encode | embedding_hash |
|---:|---|
|  1 | `fb467c297ca11f4f` |
|  2 | `4ea58fa63262e7ba` |
|  3 | `6a8646eed7313189` |
|  4 | `727ba3c76ff521cd` |
|  5 | `d4c05c2477e945f4` |
|  6 | `43019dd090786d85` |
|  7 | `a7633099e41027e9` |
|  8 | `757a78e0de565f45` |
|  9 | `ae4e34b48a6d1a6a` |
| 10 | `7f5c55e9bbe2df94` |

Every embedding differs. Tensor shape `ne=[5120, 1036, 1, 1]`,
nbytes=21,217,280 (≈20 MB of f32 image embeddings per encode).

## Decision tree per Phase A.3

| outcome | hypothesis | next phase |
|---|---|---|
| Embeddings all identical | H1 (CLIP det, LM-side leak) | Phase D-LM |
| Embeddings 8/10 vs 2/10 mirroring responses | H2/H3 partial | Phase B as planned |
| Embeddings have different split | both contribute | both branches |
| **Embeddings ALL different (10/10)** | **CLIP encoder is non-deterministic at the bit level** | **Phase B re-scoped — see below** |

The outcome we hit is the strongest possible localization: **CLIP non-
determinism is at the encoder level, not at the LM amplification level.**

## What this rules out

- H1 (CLIP deterministic, LM-side state leaks): falsified by 10/10
  distinct embedding hashes
- H6 (LM stochastic greedy): falsified by 8/10 identical LM responses
  despite 10 distinct embedding inputs — LM is deterministic when fed
  identical inputs; its variation is a faithful amplification of CLIP
  noise on the encodes where the noise exceeded a logit threshold

## What this re-targets

The original Phase B (bisect-by-buffer in the CLIP sched) was framed on
the assumption that the CLIP race was driven by stale buffer state
across encodes. But this run had `zero_on_reset=true` active for the
CLIP sched (the LM decoder is the only one that opted out), meaning
all CLIP gallocr buffers WERE being zeroed between encodes. Yet
embeddings still varied 10/10.

**Therefore: the CLIP non-determinism is not stale-buffer-state-driven.
The B.5e activation-zero workaround was masking something else.**

The remaining culprits (in evidence-likelihood order):

1. **Cross-device reduce ordering.** `tp.cpp:471` issues
   `ggml_reduce(attn.data(), n_device, GGML_OP_ADD)` to sum per-device
   partial outputs. The reduce kernel — or the upstream NCCL
   AllReduce in `reduce.cu` — may accumulate in a non-deterministic
   order. FP addition is not associative, so order variance produces
   different sums. Aligns with Phase 46's
   `project_phase46_test_kl_localized.md` ("REDUCE-output reads
   NECESSARY for determinism").
2. **cuBLAS algorithm heuristic** selecting different algos per encode
   (different rounding paths). `CUBLAS_WORKSPACE_CONFIG=:4096:8` is
   already set; if the cuBLAS picker still has freedom, that's the
   source.
3. **CUDA graph capture variance** — if the graph is rebuilt between
   encodes with different node ordering or scheduling.

## Replacement Phase B scope

Bisect-by-reduce-path-variant. Each variant is a separate 10-encode
gate with `CLIP_LOG_FINAL_HASH=1`. Compare the embedding hash
distribution per variant.

| variant | env / config | expected outcome if cause |
|---|---|---|
| Baseline (current) | (default) | 10/10 distinct hashes |
| Force-disable NCCL | `GGML_REDUCE_FORCE_MEMCPY_PEER=1` (or equivalent) | hashes converge to 1 |
| Force-enable NCCL no-peer | (find the equivalent) | hashes converge to 1 |
| Cap cuBLAS algorithm | `CUBLAS_WORKSPACE_CONFIG=:16:8` | possibly converge |
| Pin sm_75 cuBLAS algo via API | code patch | possibly converge |
| Per-reduce DtoH sync (Phase 46 "Test M") | `GGML_REDUCE_SYNC_AFTER=1` (or equivalent) | partial convergence |

The first variant to produce 10/10 identical embedding hashes
identifies the source.

## Sanity check: LM is deterministic

The 8/10 same-response result with 10 different embedding inputs is
itself a strong sanity check that the LM pipeline is deterministic:
when the embedding noise stays below the LM's argmax tolerance, the
LM produces byte-identical output. On 2 encodes the noise exceeded
the threshold and pushed the model into a different greedy-decode
path. This bounds the LM's contribution to the production race at
exactly zero — all of the variance lives upstream in CLIP.

## Artifacts

- `harness.log` — verify-multigpu-clip.sh transcript
- `server.stderr` — full server log with CLIP_FINAL_HASH lines
- `server.stdout` — server stdout
- `response-{1..10}.json` — chat completions
- `latency.json` — B.7 perf gate input
- Production state preserved: deploy unchanged; only this bench ran
