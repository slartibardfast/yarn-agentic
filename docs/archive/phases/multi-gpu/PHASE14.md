# Phase 14: Vulkan Support for Qwen3.5 Recurrent Layers

## Problem

Qwen3.5 is a hybrid attention+recurrent model. 75% of its layers (e.g. 24 out of 32 in the 4B model) use delta-net recurrent linear attention instead of standard multi-head attention. The recurrent layers use 4 operations that have no Vulkan support, producing 300+ graph splits and reducing inference to ~0.02 tok/s.

The standard attention layers (every 4th layer) use the same ops as llama architecture and work fine on Vulkan.

## Architecture

Layer pattern follows `full_attn_interval = 4` (configurable per model):
- Layers 0, 1, 2: recurrent (delta-net)
- Layer 3: standard attention
- Layers 4, 5, 6: recurrent
- Layer 7: standard attention
- ...and so on

Recurrent layers are built by `delta_net::build_layer_attn_linear_core` in `ik_llama.cpp/src/llama-delta-net.cpp:226-401`.

## Unsupported Operations

### 1. `ggml_l2_norm` (GGML_OP_L2_NORM) — 2 per recurrent layer

**Graph location**: `llama-delta-net.cpp:362-363`

```cpp
q_conv = ggml_l2_norm(ctx0, q_conv, eps_norm);
k_conv = ggml_l2_norm(ctx0, k_conv, eps_norm);
```

**Computes**: `x / sqrt(sum(x^2) + eps)` — L2 normalization per row. Structurally identical to RMS norm without a learned scale.

**Vulkan status**: Fully implemented but disabled.
- Shader: `vulkan-shaders/l2_norm.comp` (exists, 41 lines)
- Pipeline: `pipeline_l2_norm_f32` created at `ggml-vulkan.cpp:2811`
- Dispatch: `ggml_vk_l2_norm` at `ggml-vulkan.cpp:8028`
- All 6 call sites commented out: lines 6824, 7284, 9328, 9399, 9545, 10802

**Fix**: Uncomment 6 code blocks in `ggml-vulkan.cpp`. No new code.

**Impact**: Eliminates 48 splits (4B model).

### 2. `ggml_softplus` (GGML_UNARY_OP_SOFTPLUS) — 1 per recurrent layer

**Graph location**: `llama-delta-net.cpp:294`

```cpp
ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
```

**Computes**: `log(1 + exp(x))` element-wise. Used to compute the decay gate for the delta-net state update. Numerically stable form: `x > 20 ? x : log(1 + exp(x))`.

**Vulkan status**: Not in the UNARY op handler. The UNARY case in `ggml_backend_vk_supports_op` (`ggml-vulkan.cpp:10388-10403`) handles SILU, GELU, GELU_QUICK, RELU, TANH, SIGMOID — but not SOFTPLUS.

**Fix**: New `softplus.comp` shader (~23 lines, fork from `silu.comp` changing activation formula). Register pipeline, add SOFTPLUS to UNARY switch in `supports_op` and in `ggml_vk_get_pipeline`.

**Impact**: Eliminates 24 splits (4B model).

### 3. `ggml_ssm_conv` (GGML_OP_SSM_CONV) — 1 per recurrent layer

**Graph location**: `llama-delta-net.cpp:330`

```cpp
ggml_tensor * conv_output_raw = ggml_ssm_conv(ctx0, conv_states, qkv_mixed,
    model.layers[il].ssm_conv1d, inp_s_seq_qnext);
```

**Computes** (CPU reference: `ggml.c:22228-22331`): 1D convolution over SSM states with state management. For each token:
1. Shifts the conv_state left by 1 position
2. Inserts the new input token on the rightmost column
3. Dot product of shifted state with conv1d weights (kernel size `d_conv`, typically 4)
4. Outputs both the convolution result and updated conv_state

Takes 4 inputs: `conv_state` (previous state), `x` (new input), `conv1d_weight`, `state_seq` (sequence routing).

**Vulkan status**: No shader, no pipeline, no `supports_op` case.

**Fix**: New `ssm_conv.comp` shader (~100-150 lines). For token generation (n_tokens=1): single dispatch, trivial math. For prompt eval (n_tokens>1): sequential token dependency — each token's state depends on the previous token's output. Requires either host-side per-token dispatch loop or in-shader serial loop.

**Impact**: Eliminates 24 splits (4B model).

### 4. `ggml_delta_net` (GGML_OP_DELTA_NET) — 1 per recurrent layer

**Graph location**: `llama-delta-net.cpp:132` (via `build_fused_delta_net`)

```cpp
ggml_tensor * fused_result = ggml_delta_net(ctx0, q, k, v, g, beta, state_flat);
```

**Computes** (CPU reference: `ggml.c:22540-22665`): Core delta-net recurrent state update. Per head, per token:
1. L2-normalize q and k internally (redundant with external `ggml_l2_norm` — must replicate for correctness)
2. sigmoid(beta) for gating, exp(g) for decay (clamped to exp(50))
3. Attention score: `dot(k_norm, q_norm) * scale`
4. Per state row: vector-matrix product with state, gated update
5. State update: `state = decay * state + outer(v_new, k_norm)` with clamping to [-1e6, 1e6]

Takes 6 inputs: q, k, v, g (decay), beta, state_flat. Outputs concatenation of [output_tokens, new_state].

**Vulkan status**: No shader, no pipeline, no `supports_op` case.

**Challenges**:
- **Sequential token dependency**: State at token t depends on token t-1. Cannot parallelize across tokens.
- **Large state matrix**: `head_v_dim x head_v_dim` per head. For head_v_dim=128: 16K floats = 64 KB, exactly at shared memory limit on RDNA2/GCN5.
- **6 input buffers**: Current Vulkan shaders max out at 4 bindings. Descriptor set needs expansion.
- **Mixed operations**: sigmoid, exp, dot products, outer products, matrix-vector multiply, element clamping — all in one kernel.

**Fix**: New `delta_net.comp` shader (~200-300 lines). Recommended approach: per-token dispatch from host (simpler, correct for n_tokens=1 generation). Each dispatch processes all heads in parallel, reads/writes state in device memory.

**Impact**: Eliminates remaining 24+ recurrent splits (4B model).

## Operations Already Supported in Recurrent Layers

These ops used by `build_layer_attn_linear_core` work on Vulkan today:
- `mul_mat` (weight projections via `llm_build_lora_mm`)
- `silu` (activation after conv)
- `add`, `mul`, `scale` (residual, gating, scaling)
- `reshape`, `view`, `permute`, `cont` (zero-cost layout ops)
- `concat`, `cpy` (state assembly and copy-back)
- `get_rows` (output token selection)
- `fused_mul_unary` with SILU (gated output normalization)
- `rms_norm`, `fused_rms_norm` (layer normalization)

## Cumulative Impact (32-layer / 4B model, 24 recurrent layers)

| Step | Splits removed | Remaining | Difficulty |
|---|---|---|---|
| Baseline | — | 300+ | — |
| Enable L2 norm | 48 | ~252 | Trivial (uncomment) |
| + Softplus | 24 | ~228 | Easy (23-line shader) |
| + SSM conv | 24 | ~204 | Medium (new shader) |
| + Delta net | 24+ | ~2-3 | Hard (stateful kernel) |

After all four, only standard attention layer splits remain (2-3, same as llama architecture).

## Files

| File | Role |
|---|---|
| `ggml/src/ggml-vulkan.cpp` | All ops: pipeline, supports_op, dispatch, build_graph |
| `ggml/src/vulkan-shaders/l2_norm.comp` | Existing shader (uncomment usage) |
| `ggml/src/vulkan-shaders/silu.comp` | Template for softplus shader |
| `ggml/src/vulkan-shaders/vulkan-shaders-gen.cpp` | Register new shaders |
| `src/llama-delta-net.cpp:226-401` | Recurrent layer graph structure |
| `ggml/src/ggml.c:22228` | CPU reference: ssm_conv |
| `ggml/src/ggml.c:22540` | CPU reference: delta_net |

## Risks

1. **Delta net state size**: 128x128 = 64 KB per head hits the shared memory limit on RDNA2/GCN5. May require tiling for models with head_v_dim > ~90.

2. **Sequential token dependency**: Both ssm_conv and delta_net have causal token-by-token state dependencies. Token generation (n_tokens=1) is fine — one dispatch. Prompt eval needs serial dispatch loops from host.

3. **Double normalization**: The CPU delta_net implementation normalizes q/k internally despite the graph applying `ggml_l2_norm` externally beforehand. The Vulkan shader must replicate this redundant normalization to match CPU output numerically.

4. **6-buffer descriptor set**: delta_net takes 6 inputs. Current Vulkan shaders use at most 4 bindings. The descriptor set layout and push constant structure need extension.

5. **Testing models**: Need a Qwen3.5 GGUF (0.8B or 2B Q8_0) for verification on available hardware.
