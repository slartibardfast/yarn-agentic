---
name: yarn-agentic backup branches
description: Where pre-consolidation WIP was preserved when the radv wrapper repos were retired on 2026-04-11.
type: reference
originSessionId: a5a02c13-3d1d-4998-badf-36c74b87c680
---
When `radv_llama.cpp` and `radv_ik_llama.cpp` were retired on 2026-04-11, one pocket of genuinely unique work was pushed to GitHub as a backup branch:

- **`slartibardfast/ik_llama.cpp` branch `backup/pre-consolidation-wip`** — 2174 lines of in-progress Vulkan shader work from the nested `radv_ik_llama.cpp/ik_llama.cpp` detached-HEAD checkout: new files `mul_mm_fused_up_gate.comp`, `mul_mm_moe_fused_up_gate.comp`, `mul_multi_add.comp`, plus modifications to `ggml/src/ggml-vulkan.cpp` and `ggml/src/vulkan-shaders/vulkan-shaders-gen.cpp`. Base commit `1a246b99` ("vulkan: dmabuf zero-copy cross-device transfer"). Commit sha: `7592719b`.

Everything else marked "unpushed" turned out to be already present on the forks (the local `radv_ik_llama.cpp` clone was severely stale, months behind its own remote, which shared history with `radv_llama.cpp`). The `radv_llama.cpp` and `radv_ik_llama.cpp` fork repos themselves remain on GitHub as historical archives — their content is subsumed by yarn-agentic.
