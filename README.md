# yarn-agentic

Experiments in local hosting of large language models.

Experiments in progress:

- **Vulkan multi-GPU split-mode-graph support** for `llama.cpp` and `ik_llama.cpp` — running a single model across two non-NVLink GPUs without dropping ops to CPU fallback. See the [Plan](PLAN.md) or browse the Vulkan [phases](PHASE0.md).
- **Qwen3.5 MTP tool calling on Vega 64** — using native MTP weights for speculative decode on an 8 GiB Vega 64, with the mission of measuring tool-calling accuracy across candidate models. Start with the [peer host quickstart](phases/qwen35-mtp/PHASE1.md).
- **DFlash speculative decoding for Qwen 3.6 27B on sm_75** — porting vLLM PR #40898's diffusion-style sidecar drafter to ik_llama.cpp's CUDA backend on dual Quadro RTX 6000. Kernel layer argmax-equivalent to vLLM across 8 prompts × 4 mask positions; bench infrastructure for apples-to-apples spec comparison (none / mtp / dflash) with PPL-of-output landed; end-to-end measurement of record captured with TU102 + NVLINK optimization envelope named. See [PHASE_DFLASH.md](PHASE_DFLASH.md).
