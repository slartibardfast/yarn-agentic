# yarn-agentic

Experiments in local hosting of large language models.

Experiments in progress:

- **Vulkan multi-GPU split-mode-graph support** for `llama.cpp` and `ik_llama.cpp` — running a single model across two non-NVLink GPUs without dropping ops to CPU fallback. See the [Plan](PLAN.md) or browse the Vulkan [phases](PHASE0.md).
- **Qwen3.5 MTP tool calling on Vega 64** — using native MTP weights for speculative decode on an 8 GiB Vega 64, with the mission of measuring tool-calling accuracy across candidate models. Start with the [peer host quickstart](phases/qwen35-mtp/PHASE1.md).
