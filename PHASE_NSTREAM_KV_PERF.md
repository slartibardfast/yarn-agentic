# PHASE_NSTREAM_KV_PERF — recover the regression *and* unlock the dispatch ceiling

**Branch**: `production/2026-q2-next` (off submodule HEAD `16b608d1`).
**Predecessor**: `PHASE_NSTREAM_KV.md` — closed 2026-05-20. Bug C structurally closed; decode-side prefill gate removed; 6 correctness gates green; **-6.2 % TG NP=8 regression carried over**.
**Status**: Open. Direction tree below, starting hypothesis locked to Tier 2 (single graph + per-stream kernel-arg patch). Maximum-effort scope: Tier 2 → Tier 3 if Tier 2 lands clean.

## Why this phase exists — sharper framing post-research (2026-05-20)

After deep primary-source review, the regression we measured at G3.h is **not isolated**. Upstream [ggml-org/llama.cpp#14863](https://github.com/ggml-org/llama.cpp/issues/14863) reports a 33 % TG regression on dual RTX PRO 6000 Blackwell from the same PR #14363 we ported, multi-GPU only, tg128-only — exact pattern as ours. The fix is upstream-open ground.

The architectural fact that explains the regression: **we ported upstream's data structure but kept ik_llama.cpp's per-stream sequential dispatch**.

- Upstream: one `llama_decode` per tick with all streams in one ubatch; the `n_stream` axis at ne[3] handles fan-out inside the graph; **same graph reused tick over tick**.
- Ours: one `llama_decode` per active stream per tick; graph rebuilt per call because the baked `stream_id` view-offset changes. At NP=8 that's **8 graph builds per tick of pure overhead**.

The TU102 bandwidth math says the aggregate ceiling at NP=8 (MFU=30 %) for Qwen 35B-A3B INT4 is **300–600 tok/s**. We are almost certainly at 10–30 % of that today. The dispatch is the binding constraint, not the kernels.

## Hardware ground truth (probed 2026-05-20)

```
nvidia-smi nvlink -s   → all links inActive
nvidia-smi topo -m     → GPU0–GPU1 connection: PHB
```

**No NVLink bridge installed.** Cross-GPU is PCIe Gen3 through host bridge (~13 GB/s effective). Closest published fabric analogue: ik_llama.cpp [PR #1080](https://github.com/ikawrakow/ik_llama.cpp/pull/1080) on 4× RTX 3090 PCIe — Llama-3-70B Q4_0 at ~50 t/s gen on `--split-mode graph`. Graph-split (tensor-parallel) is still the right multi-GPU mode for our MoE workload because MoE all-reduce volume scales with active params (3 B) not total params (35 B).

## Direction tree (primary-sourced)

| Tier | What | Expected gain | Effort | Risk |
|------|------|---------------|--------|------|
| 1 | Per-stream `cudaGraphExec_t` cache | +6.2 % — recover regression | ~2 days | Low |
| **2** | **Single graph + `cudaGraphExecKernelNodeSetParams` per-stream patch** | **+2–3× at NP=8** | **1–2 weeks** | Medium |
| **3** | **Pivot dispatch to upstream's unified-stream model + per-sequence ragged FA** | **Approaches the vLLM-class ceiling on our stack** | **2–3 months** | High |
| 4 | Chunked-prefill admission control on top of Tier 3 | Small additional gain on prefill-heavy workloads | +2–4 weeks | Low |
| 5 | Full paged-KV port (V1 vLLM kernel, sm_70+ compatible) | Marginal beyond Tier 3 for our workload | 6+ months | Very high |

**SKIP**: vLLM pivot (loses Q4_0 + Hadamard + Q4_0 KV — uniquely SoTA on sm_75; nothing else in the ecosystem can consume the production weight stack); persistent-kernel megakernels (Luce sm_75 support is batch=1 only; Mirage is Hopper-first).

### Tier 1 — Per-stream graph cache

What the original PHASE_NSTREAM_KV_PERF hypothesis was. Keep as the **fallback** if Tier 2 hits an unexpected blocker. Mechanism: `std::array<unique_ptr<Prev>, MAX_N_STREAM>` or small map keyed by `stream_id`, look up in `can_reuse_graph`, store in `cache_prev`. Recovers the regression; no upside beyond.

### Tier 2 — Single graph + per-stream kernel-arg patch (LOCKED STARTING HYPOTHESIS)

NVIDIA's own pattern, documented at:
- [Optimizing llama.cpp AI Inference with CUDA Graphs](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/)
- [Employing CUDA Graphs in a Dynamic Environment](https://developer.nvidia.com/blog/employing-cuda-graphs-in-a-dynamic-environment/)
- [Constructing CUDA Graphs with Dynamic Parameters](https://developer.nvidia.com/blog/constructing-cuda-graphs-with-dynamic-parameters/)

NVIDIA applied this to per-*token* KV pointer updates ("identify the kernel nodes whose parameters need updating, call `cudaGraphExecKernelNodeSetParams` to patch them in-place"). We extend it to per-*stream* KV pointer + view-offset updates.

Mechanism sketch:
1. Capture graph using `stream=0`'s view offsets.
2. Identify the small set of CUDA kernel nodes that take K/V base pointers (the FA kernels, KV-write copy kernels, the cache-update path). Store node handles at capture time.
3. Before launch for stream `s`, call `cudaGraphExecKernelNodeSetParams` to patch the K/V base pointer to `K_base + s * head_dim * kv_size_per_stream * n_head_kv * sizeof(dtype)`.
4. Graph topology stays stream-invariant (only offsets differ), so `cudaGraphExecUpdate` semantics hold.

Known gotchas (from research):
- The "consecutive updates" disable counter in `ggml_backend_cuda_graph_compute` (introduced by [PR #7302](https://github.com/ggml-org/llama.cpp/pull/7302)) will fire if patched every tick; threshold needs bumping or the gate needs disabling for the per-stream case.
- ik_llama.cpp's Q4_0 / Q4_0_AR16 / Hadamard KV-write kernels take more parameter slots than upstream's — node enumeration needs care.
- `LLAMA_SET_ROWS=1` / `n_stream > 1` must be the new normal; falling back to unified mode reopens Bug C.

### Tier 3 — Unified-stream dispatch + per-sequence ragged attention

The structurally-correct end state. Stop dispatching `llama_decode` per stream. Instead:

a. **Server-side batch fusion.** Modify `process_batch_tokens` to collect all active slot tokens into one batch each tick (one llama_decode/tick). The 4D layout already supports this — each token's `seq_id` resolves to its stream's K/V slab via the existing per-stream view offsets.

b. **Per-sequence ragged FA kernel.** New CUDA FA variant that takes a `[n_stream]` length vector and iterates per-stream `kv_extent` inside the kernel. **Hard part on sm_75** — can't use FA2/FA3 (cp.async / sm_80+ gated), can't use FlashInfer's stock paged kernel (96 KB shared-mem requirement), can't use vLLM's V2 paged kernel (sm_80+). The only sm_75-viable reference is the **original vLLM `csrc/attention/attention_kernels.cu`** (pre-FA, sm_70+ compatible, ~1500 lines) — port as a ggml op.

c. **Drop the n_stream==1 guards on `build_k_shift` / `build_defrag` / `v_trans`** (currently `GGML_ASSERT(n_stream == 1)`) so multi-slot ctx_shift and defrag work natively.

d. **Optional: chunked-prefill admission** (Sarathi-Serve, [arXiv:2403.02310](https://arxiv.org/abs/2403.02310)) on top — only if measurement says it's worth it. Defer to Tier 4.

### Tier 4 — Chunked-prefill admission control

Splices new request prefill chunks into the same fused ubatch as running decodes. Mechanism is well-understood ([SGLang's `enable_mixed_chunk`](https://deepwiki.com/sgl-project/sglang/4.2-token-sampling-and-generation)). Defer to post-Tier-3 measurement.

### Tier 5 — Paged KV (skip unless multi-tenant)

V1 vLLM kernel is sm_70+ compatible and would port as a ggml op. But for a single-server NP=8 workload with broadly similar sequence lengths, the per-stream contiguous layout we already have is functionally equivalent and avoids the indirection cost. Defer indefinitely.

## What's NOT in scope (carried verbatim from prior PHASE_NSTREAM_KV_PERF)

- New correctness gates (PHASE_NSTREAM_KV's six gates remain the preservation checks).
- N-stream **layout** changes (the 4D port itself is closed).
- DFlash perf (separate workstream).
- PSKV singlewarp FA optimisation (separate ralph loop, currently cancelled).
- vLLM/SGLang/LMDeploy pivot — ruled out on evidence (see "Why we don't pivot to vLLM" below).

## Why we don't pivot to vLLM (despite the MEMORY note)

The MEMORY entry [project_continuous_batching_vs_perslot_dispatch.md](~/.claude/projects/-home-llm-yarn-agentic/memory/project_continuous_batching_vs_perslot_dispatch.md) attributes 4.75× aggregate uplift to vLLM at NP=8 on same hardware/weights. After primary-source review the number is **not reproducible on sm_75**:

1. **Weight format incompatibility.** vLLM cannot consume Q4_0 + Hadamard + Q4_0 KV natively. Marlin-AWQ INT4 is comparable not better; QuaRot Hadamard PR [vllm-project/vllm#15162](https://github.com/vllm-project/vllm/pull/15162) was *closed*; the later compressed-tensors path has no Turing track record.
2. **GGUF path is slow.** [vllm-project/vllm#8669](https://github.com/vllm-project/vllm/issues/8669) — Llama 3.1 70B Q4 GGUF on A100 80GB ran at 8.7 tok/s under vLLM (worse than our current production on much weaker hardware).
3. **FP8 KV impossible on TU102.** [Turing whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf) lists FP16/INT8/INT4/INT1 only — no FP8 math. vLLM's FP8 KV needs sm_86+ Triton.
4. **No working FA backend on Turing for modern models.** vLLM issue [#38918](https://github.com/vllm-project/vllm/issues/38918) — Gemma4 has zero working attention backends on sm_75 (FLASH_ATTN excluded, FLASHINFER head_size errors, TRITON_ATTN crashes, FLEX_ATTENTION CUDA-graph fails). Issue [#29743](https://github.com/vllm-project/vllm/issues/29743) closed as not planned for Qwen3-VL on Turing.
5. **The 4.75× number traces to Ampere/Blackwell with NVFP4 + FP8 KV** — neither available to us. Closest measured comparable ([LLMKube bakeoff](https://llmkube.com/blog/qwen3-6-27b-bakeoff)) was 3.7× on Blackwell consumer cards with NVFP4 weights. Strip the NVFP4 + FP8 KV advantage and the multiplier collapses.

**ik_llama.cpp's split-mode-graph already beats mainline 33 % TG / 6-9× PP** ([discussion #1247](https://github.com/ikawrakow/ik_llama.cpp/discussions/1247)) on our weight class. The kernels are SoTA for sm_75; only the dispatch is behind. The right move is to close the dispatch gap inside ik_llama.cpp, not to pivot to an engine that has dropped Turing maintenance.

## Binding gates (Tier 2 closure)

- **GP3.a** — `llama-batched-bench` TG NP=8 ≥ **+50 %** vs 26.00 t/s baseline (i.e. ≥ 39 t/s) on the production Qwen 35B-A3B + Q4_0 KV stack. Hard binding.
- **GP3.b** — `scripts/test-pp-serialization.sh` wall ≤ 11 s (recovers TG-overlap window — was 15.7 s post-port).
- **GP3.c** — `scripts/test-production-np-determinism.sh` byte-identity preserved, single + multi-GPU.
- **GP3.d** — `bin/test-dflash-np-multislot` GREEN unchanged.
- **GP3.e** — `scripts/r5-probe-c4.sh ITERS=20` = 0/20, single + multi-GPU. Bug C absence preserved.
- **GP3.f** — `scripts/validate-batch-composition-trace.py` against an NDJSON trace from `r5-probe-c4.sh` ITERS=20 — zero violations of `BatchComposition.tla` / `StreamIsolation.tla`. Spec layer preserved.

If Tier 2 misses GP3.a but lands the regression recovery cleanly (≥ 27.73 t/s = +6.2 %), evaluate: was the per-stream pointer patch landed correctly? Or did the disable-counter / unrelated invalidation fire? Surface the diagnostic per `feedback_negative_results_land_cheap_when_honest` — do not declare done.

## Binding gates (Tier 3 closure — provisional, locked at Tier 2 close)

- **GP3.g** — `llama-batched-bench` TG NP=8 ≥ **3× baseline** (≥ 78 t/s).
- **GP3.h** — Per-sequence ragged FA kernel passes `test-backend-ops` GREEN at all NP shapes.
- **GP3.i** — All Tier-2 gates remain GREEN.

## Tier 2 implementation cards (provisional, finalised at design lock)

1. **T2.a** — Probe `ggml_backend_cuda_graph_compute` (`ggml/src/ggml-cuda/ggml-cuda.cu`). Map every node that takes a K/V base pointer in the production graph at NP=8. Output: list of `cudaGraphNode_t` handles + their parameter offsets.
2. **T2.b** — Capture infrastructure: store handles + base-offsets in `Prev` struct; one-time capture at first decode for stream 0; subsequent dispatches re-use via `cudaGraphExecKernelNodeSetParams`.
3. **T2.c** — Bump or disable the "consecutive updates" counter for the per-stream case. Verify no regressions on single-stream production (NP=1) load.
4. **T2.d** — Wire into `can_reuse_graph`: drop the `n_stream > 1` short-circuit at `src/llama.cpp:616`; rely on per-input reuse checks (model upstream's decentralised `can_reuse_kq_mask`-style approach).
5. **T2.e** — Bench gate GP3.a. If positive, full correctness battery (GP3.b–GP3.f).

## Adjacent open follow-ups (carried; not in this phase)

- `build_k_shift` / `build_defrag` per-stream — needed for multi-slot ctx_shift / cache compaction. Lift in Tier 3 if reached.
- Non-FA `v_trans` per-stream — production runs FA-on; defer until non-FA path is needed.
- MLA (DeepSeek) — out of scope.

## Token estimate

Per CLAUDE.md §8:

- Tier 2 — node enumeration + capture infra + dispatch refactor + gates: **80–120 k tokens**.
- Tier 3 (if reached) — server-side batch fusion + ragged FA kernel port + gates: **150–250 k tokens** (kernel rewrite is the bulk).
- Diagnosis rounds if Tier 2 hits an unexpected invalidation: 30 k each, budget 2–3.

Total scope: ~120–400 k tokens depending on how far we run. The 4D port already invested ~140 k; this phase is where the return on that investment materialises.

## Primary research sources (from agents' deep review, 2026-05-20)

CUDA-graph mechanism:
- [NVIDIA — Optimizing llama.cpp with CUDA Graphs](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/)
- [NVIDIA — CUDA Graphs in a Dynamic Environment](https://developer.nvidia.com/blog/employing-cuda-graphs-in-a-dynamic-environment/)
- [NVIDIA — Constructing CUDA Graphs with Dynamic Parameters](https://developer.nvidia.com/blog/constructing-cuda-graphs-with-dynamic-parameters/)

Upstream context:
- [llama.cpp PR #14363 — high-throughput mode (the n_stream port)](https://github.com/ggml-org/llama.cpp/pull/14363)
- [llama.cpp issue #14863 — Blackwell multi-GPU TG regression](https://github.com/ggml-org/llama.cpp/issues/14863)
- [llama.cpp PR #7302 — avoid disabling CUDA graphs](https://github.com/ggml-org/llama.cpp/pull/7302)
- [llama.cpp discussion #4130 — parallelisation roadmap](https://github.com/ggml-org/llama.cpp/discussions/4130)
- [llama.cpp release notes April 2026 — Walsh-Hadamard KV](https://fazm.ai/blog/llama-cpp-release-april-2026)

ik_llama.cpp ground truth:
- [PR #1080 — graph parallel: next generation](https://github.com/ikawrakow/ik_llama.cpp/pull/1080)
- [Discussion #1247 — ik_llama.cpp vs mainline](https://github.com/ikawrakow/ik_llama.cpp/discussions/1247)

Continuous batching mechanism (Tier 3 reference):
- [Orca (OSDI '22)](https://www.usenix.org/system/files/osdi22-yu.pdf)
- [vLLM / PagedAttention (SOSP '23)](https://arxiv.org/abs/2309.06180)
- [Sarathi-Serve chunked prefill (OSDI '24)](https://arxiv.org/abs/2403.02310)
- [vLLM Anatomy 2025 (engine loop reference)](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)
- [SGLang scheduler internals (DeepWiki)](https://deepwiki.com/sgl-project/sglang/4.2-token-sampling-and-generation)

sm_75 / Turing constraints:
- [vLLM #38918 — no working attention backend on Turing](https://github.com/vllm-project/vllm/issues/38918)
- [vLLM #29743 — Qwen3-VL Turing closed-not-planned](https://github.com/vllm-project/vllm/issues/29743)
- [vLLM #8669 — GGUF Q4 at 8.7 t/s on A100](https://github.com/vllm-project/vllm/issues/8669)
- [Dao-AILab/flash-attention #542, #720 — FA2 Turing backlog stalled](https://github.com/Dao-AILab/flash-attention/issues/720)
- [Turing Tuning Guide (CUDA docs)](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html)

Hardware-bench analogues:
- [LLMKube Qwen3.6-27B bakeoff (Blackwell consumer, 3.7×)](https://llmkube.com/blog/qwen3-6-27b-bakeoff)
- [Himesh vLLM 4× RTX 3090 benchmarks (MoE PCIe behaviour)](http://himeshp.blogspot.com/2025/03/vllm-performance-benchmarks-4x-rtx-3090.html)
- [ThinkSmart Qwen3.5-35B-A3B on 4× RTX 3090 PCIe](https://thinksmart.life/research/posts/qwen35-35b-4x3090-vllm-pcie/)

Persistent-kernel / megakernel context (skip on sm_75):
- [Mirage MPK (arXiv 2512.22219)](https://arxiv.org/html/2512.22219v1)
- [Luce megakernel (sm_75 batch=1)](https://github.com/Luce-Org/lucebox-hub/blob/main/megakernel/README.md)
