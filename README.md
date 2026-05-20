# yarn-agentic

Experiments in local hosting of large language models.

Workstreams:

- **Vulkan multi-GPU split-mode-graph support** for `llama.cpp` and `ik_llama.cpp` — running a single model across two non-NVLink GPUs without dropping ops to CPU fallback. See [docs/phases/00-vulkan-multigpu/](docs/phases/00-vulkan-multigpu/).
- **Qwen3.5 MTP tool calling on Vega 64** — using native MTP weights for speculative decode on an 8 GiB Vega 64, with the mission of measuring tool-calling accuracy across candidate models. Start with the [peer host quickstart](docs/phases/qwen35-mtp-tooling/PHASE1.md).
- **DFlash speculative decoding for Qwen 3.6 27B on sm_75** — porting vLLM PR #40898's diffusion-style sidecar drafter to ik_llama.cpp's CUDA backend on dual Quadro RTX 6000. Kernel layer argmax-equivalent to vLLM across 8 prompts × 4 mask positions; bench infrastructure for apples-to-apples spec comparison (none / mtp / dflash) with PPL-of-output landed; end-to-end measurement of record captured with TU102 + NVLINK optimization envelope named. See [docs/phases/70-dflash/PHASE_DFLASH.md](docs/phases/70-dflash/PHASE_DFLASH.md).

Active workstream: **`PHASE_NSTREAM_KV_PERF`** — per-stream graph cache to recover the -6.2 % TG-NP=8 regression carried over from `PHASE_NSTREAM_KV` (closed 2026-05-20, structurally closed Bug C via per-stream KV dispatch).

Documentation site is rendered from `docs/` via [mdBook](https://rust-lang.github.io/mdBook/). Per-workstream phase docs live under `docs/phases/<NN-name>/`; cross-cutting reference docs under `docs/reference/`. The single currently-active workstream lives at top-level `PLAN.md` and moves to `docs/phases/<NN-name>/` on closure.
