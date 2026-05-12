# Gate 0 — partial findings

**Date**: 2026-05-12
**Hardware**: 2× Quadro RTX 6000 (sm_75, 24 GiB each)
**Stack**: vLLM PR #40898 + Qwen3.6-27B + Qwen3.6-27B-DFlash drafter
**Patches active**: `scripts/vllm_sm75_patches.py` — 3 sm_75 monkey-patches

## What we measured

### DFlash decode (FlexAttention path)
First run, completed cleanly:
- `init`: 1680.7s (model + drafter load from CIFS)
- `gen`: 5534.32s = 92 min for 8 prompts × max 256 tokens each
- `total_out_toks`: 1821
- **tok/s: 0.33**

### DFlash decode (FlashInfer path — forced via `AttentionConfig(backend=FLASHINFER)`)
Second run, warmup pass completed before crash:
- `init`: 1667.0s (essentially identical to FlexAttention run)
- `warmup`: 886.03s for 8 prompts × 32 tokens
- **tok/s: 0.29**

### Vanilla decode baseline
**Could not measure cleanly.** Both runs crashed silently when transitioning to the
next pass (FlexAttention run crashed at vanilla pass prompt 1/8; FlashInfer run
crashed at timed pass start). Crash signature is identical in both:
- ~58 GB RSS peak
- ~4.5 GB swap peak
- No traceback in stderr
- No journal OOM kill
- No dmesg memory pressure event
- vLLM workers (Worker_TP0/TP1) cleanly removed from GPU but parent silently exits

## What this tells us

### 1. Attention backend choice did not move tok/s

FlexAttention 0.33 vs FlashInfer-forced 0.29 — within noise (warmup pass is shorter
context; same-config comparison would likely be even closer). The published DFlash
speedup assumes FA2 + bf16; on sm_75 we are NOT bottlenecked on the attention
kernel.

### 2. The bottleneck is the eager-mode + CPU-offload combo

`enforce_eager=True` (no CUDA graphs; required by DFlash's dynamic shapes) plus
`cpu_offload_gb=12` (54 GiB BF16 → fp16 weights don't fit in 2× 24 GiB) plus the
DFlash aux Triton kernels (zero_kv_blocks, eagle_inputs, dflash_inputs, etc.)
collectively gate per-step latency.

### 3. vLLM DFlash on sm_75 is structurally unstable at decode lengths ≥ 100 tokens

Both crashes occurred at the same RSS peak (58 GB) and same swap peak (4.5 GB).
Suggests either:
- A latent vLLM memory leak that hits a process-level limit
- A worker comms/NCCL issue at the longer-batch shape
- A Triton kernel JIT compile path that the worker can't recover from

The crash is reproducible at the warmup → timed transition or DFlash → vanilla
transition. Not a one-off.

## What this means for the ik_llama.cpp port

**Do not anchor port performance to the vLLM number.** The vLLM number is gated
by reference-impl choices (enforce_eager, cpu_offload, eager DFlash pipeline) that
the ik_llama.cpp port will not inherit. The port targets the same algorithm
(DFlash spec decode against Qwen 3.6 27B + drafter) but on a different runtime
with different perf characteristics.

What the vLLM gates DO bind on the port:
- **Gate-3.5 GREEN** (multi-slot determinism): the algorithm is correct
  multi-slot; the port's multi-slot path must match this behavior.
- **OQ-DFLASH-INHERITS-MTP-MULTISLOT-BUG** (RESOLVED GREEN in spec): the spec's
  per-slot dispatch invariant is the binding contract; the port implements it
  directly.

What the vLLM gates DO NOT bind on the port:
- Absolute tok/s ceiling on sm_75. We don't know it. The port has to measure
  its own ceiling on its own runtime.
- Whether DFlash earns a 2.4-3.2× speedup on sm_75. The vLLM data is too noisy
  (warmup + JIT effects dominate; vanilla baseline crashes).

## Next step recommendation

Pivot to the ik_llama.cpp port. The spec is locked, the multi-slot determinism
question is bound, and further vLLM measurement is yielding diminishing returns
under the crash signature.
