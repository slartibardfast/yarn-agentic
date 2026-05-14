# Peer handoff — PHASE31 MTP, ready for Quadro replication

State as of **2026-05-01** on the 3060 Ti host. PHASE29/30 (TurboQuant) are **abandoned** per user direction. **PHASE31 is MTP-only** on `ik_llama.cpp`. On 8 GB VRAM with `--cpu-moe` we proved correctness but tg is **negative** vs baseline — the cpu-moe bottleneck eats acceptance savings. **A Quadro with ≥ 24 GB VRAM should bind the throughput uplift under full GPU offload.**

## Inputs

- **Top-level repo:** https://github.com/slartibardfast/yarn-agentic — only the `ik_llama.cpp` submodule matters here.
- **Required submodule branch:** `fix/cuda-delta-net-emit-intermediates` on `slartibardfast/ik_llama.cpp` (commit `f9bb0efa`). **Not on main yet.** PR ready to open at https://github.com/slartibardfast/ik_llama.cpp/pull/new/fix/cuda-delta-net-emit-intermediates. Without it, MTP on CUDA crashes at the first prompt-eval batch through the linear-attention layer (`ggml-cuda/delta-net.cu:258` assertion fires when `emit_intermediates=true && n_tokens > 1`).
- **Model:** Qwen3.6-35B-A3B (qwen35moe arch, `nextn_predict_layers = 1`, 41 layers, 256/8 experts). BF16 source ~67 GB; Q4_K_M ~22 GB. We quantized with the `llama-quantize` from this branch.
- **Corpus:** wikitext-2 test set.

## Build

```sh
cd ik_llama.cpp
git fetch origin && git checkout fix/cuda-delta-net-emit-intermediates
cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
build/bin/llama-quantize <BF16-gguf> <Q4_K_M-gguf> Q4_K_M
```

CMake defaults to native arch on the build host. For broader binaries use `-DCMAKE_CUDA_ARCHITECTURES=75;80;86;89` (covers Turing through Ada Quadros).

## Throughput uplift run (the closing measurement we couldn't bind)

On 24+ GB Quadros, drop `--cpu-moe` so the MoE expert weights live on GPU. That's the configuration where MTP draft acceptance turns into real tg savings.

```sh
# Baseline
build/bin/llama-server -m Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --device CUDA0 -ngl 99 -fa on -no-mtp -c 4096 \
  --port 18080 --metrics

# MTP (separate run, after stopping baseline)
build/bin/llama-server -m Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --device CUDA0 -ngl 99 -fa on -mtp -c 4096 \
  --port 18080 --metrics
```

Hit each with the same prompt:

```sh
curl -sN http://127.0.0.1:18080/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"Write a 200-word essay about why birds are interesting:","n_predict":128,"temperature":0.0,"cache_prompt":false,"stream":false}' \
  | python3 -m json.tool
```

The MTP server logs print:
```
slot init: ... | speculative decoding context initialized
draft acceptance rate = <ratio> (<accepted> accepted / <generated> generated)
```

**Target:** MTP tg ≥ 1.5× baseline tg. The published claim (commit `fd77f898`) was 10.2 → 17.8 t/s = 1.74×.

## PPL parity check (we already passed this; re-run at full context for binding)

```sh
build/bin/llama-perplexity -m Qwen3.6-35B-A3B-Q4_K_M.gguf \
  -f wiki.test.raw --device CUDA0 -ngl 99 -fa on -c 4096 --chunks 64 -mtp
# Then again with -no-mtp instead of -mtp
```

**Expected:** byte-identical per-chunk numbers between the two runs. We verified at c=512/16 chunks (7.0974 ± 0.278 in both runs); should hold at c=4096/64 too.

## BF16 caveats by arch

- **sm_75 (Turing — Quadro RTX 6000 / RTX 8000 / T-series):** **no native BF16 tensor cores.** BF16 GEMM emulates to FP32 (~5–8× slower). Convert weights to F16 for sm_75 hosts; or accept the perf hit if you only care about correctness.
- **sm_86 (Ampere — RTX A6000 / A5000 / A4000):** native BF16. Same as our 3060 Ti.
- **sm_89 (Ada — RTX 6000 Ada / RTX 5000 Ada):** native BF16 + FP8.
- **sm_90 (Hopper — H100 PCIe variants if you have one):** native BF16, fp8, etc.
- **All archs:** `iqk_fa_*_*.cpp:19` rejects mixed `K=BF16, V=quant` flash attention. Always use `-ctk f16` if `-ctv` is anything quantized.

## What we already verified on the 3060 Ti

- Build clean at native sm_86, 475/475 targets.
- Q4_K_M quantization succeeded; MTP heads (`blk.40.nextn.{shared_head_norm, enorm, hnorm}`) preserved.
- MTP smoke: coherent text on `--cpu-moe`.
- Server: 85.3% draft acceptance (58/68).
- PPL parity: byte-identical chunk-by-chunk.
- Throughput on `--cpu-moe`: **−25%** in tg (server 20.22 → 15.17). Documented as hardware-bound, not a code bug.
- Delta-net CUDA bug isolated and patched on `fix/cuda-delta-net-emit-intermediates`.

## What's still open for the peer

1. **Throughput uplift number** under full GPU offload. The closing claim of PHASE31 needs Quadro hardware to bind.
2. **PR review/merge** for `fix/cuda-delta-net-emit-intermediates` — fix is correct but conservative (CPU fallback for n_tokens>1 emit). A future kernel extension would let CUDA own that path too.
3. **Larger-context PPL** at c=4096 or c=8192 with full offload — should still be parity but worth confirming.

## Out of scope (do not pursue)

- TurboQuant (PHASE29/30 abandoned). If revived, the user-named reference fork is `slartibardfast/llama-cpp-turboquant` — adopt, don't rebuild.
- HIP/AMD parity (no AMD hardware here).
- Vulkan turbo regression (PHASE30 closing condition d, no gpu-1 access).
- llama.cpp upstream maintenance (research vehicle only after PHASE29 close).

## Reference index

- `PHASE29.md`, `PHASE30.md` — abandoned phases with iter logs.
- `PHASE31.md` — current MTP phase, iter 1 logged.
- `MEMORY.md` — 2026-05-01 entries: pivot decision + PHASE31 findings.
- Defunct branches in `llama.cpp/`: `defunct/phase29`, `defunct/phase30`, `defunct/phase29-iter7-tq_v_4b-fa-v`.
