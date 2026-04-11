# Phase 21: nemotron_h_moe Architecture Support

## Goal

Add full support for the `nemotron_h_moe` architecture (Nemotron-3-Nano-30B-A3B) to our fork. This is a hybrid Mamba2 + Attention + MoE model with 52 layers, 128 experts, and 3 distinct layer types.

## Model

**Nemotron-3-Nano-30B-A3B** — available at `/opt/models/nemotron-3-nano/` in Q5_K_XL, Q6_K_XL, Q8_0, Q8_K_XL quantizations.

- 52 layers: Mamba2 SSM + standard attention + MoE FFN
- 128 experts, top-6 selection, 1 shared expert
- SSM: conv_kernel=4, state_size=128, n_group=8, inner_size=4096
- Upstream llama.cpp has full support (architecture, graph builder, all Vulkan shaders)

## Missing Vulkan Ops (5)

| Op | Layer Type | Complexity | Upstream Shader |
|---|---|---|---|
| `SSM_CONV` | Mamba2 — 1D convolution (kernel=4) | Medium | Yes |
| `SSM_SCAN` | Mamba2 — parallel associative scan | **High** | Yes |
| `SWIGLU` | Mamba2 — gated activation | Low | Yes |
| `ADD_ID` | MoE — indexed bias add | Low | Yes |
| `SET_ROWS` | MoE — scatter results | Medium | Yes (commented out in fork) |

Plus `ARGSORT` may need top_k extension for MoE routing.

## Architecture Support Gap

Beyond the 5 Vulkan ops:

1. **Architecture registration**: `LLM_ARCH_NEMOTRON_H_MOE` enum + tensor name mapping
2. **Graph builder**: `build_nemotron_h_moe()` — hybrid layer dispatch per layer type
3. **Recurrent state management**: Mamba2 conv state + SSM state caching (different from KV cache)
4. **MoE routing**: Expert selection, scatter/gather for parallel expert computation

## Approach

Port from upstream `llama.cpp` submodule (already checked out at `/home/llm/radv_llama.cpp/llama.cpp/`):

1. Port architecture enum and tensor mapping from upstream model loader
2. Port `llm_build_nemotron_h` and `llm_build_mamba_base` graph builders
3. Port the 5 missing Vulkan shaders from upstream `ggml-vulkan/vulkan-shaders/`
4. Add `supports_op` entries and pipeline creation for each new op
5. Test with Q8_0 first (simplest quant), then verify Q5_K_XL/Q6_K_XL

## Verify by

`llama-cli -m /opt/models/nemotron-3-nano/Q8_0.gguf -ngl 99` produces coherent text output on single GPU, then multi-GPU.

## References

- Upstream graph builder: `llama.cpp/src/models/nemotron-h.cpp`
- Upstream Mamba base: `llama.cpp/src/models/mamba-base.cpp`
- Upstream MoE FFN: `llama.cpp/src/llama-graph.cpp` (`build_moe_ffn`)
- Op trace: [PHASE17.md Part 2](PHASE17.md#part-2-nemotron_h_moe-nemotron-3-nano-30b-a3b)
