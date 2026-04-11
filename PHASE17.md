# Phase 17: Vulkan Op Trace for Nemotron

## Part 1: LLM_ARCH_DECI (Nemotron-51B, Nemotron-Ultra-253B)

## Model Architecture

Nemotron models (Nemotron-3-Nano-4B, Llama-3.1-Nemotron-51B, Llama-3.1-Nemotron-Ultra-253B) use the `LLM_ARCH_DECI` architecture. The graph is built by `build_deci()` at `llama-build-context.cpp:2319`.

Key architectural features:
- **Variable per-layer structure**: each layer can have different `n_head`, `n_head_kv`, and `n_ff`
- **Attention-free layers** (`n_head == 0`): skip attention entirely (Nemotron-51B)
- **Linear attention layers** (`n_head > 0, n_head_kv == 0`): single `wo` projection, no KV cache (Nemotron-51B)
- **FFN-free layers** (`n_ff == 0`): skip FFN entirely (Nemotron-Ultra-253B)
- **Standard layers** (`n_head > 0, n_head_kv > 0, n_ff > 0`): full attention + gated FFN with SiLU

No MoE. No recurrent layers. Pure transformer with per-layer flexibility.

## Complete Op Trace

### Embedding

| Op | Source | Vulkan |
|---|---|---|
| `GET_ROWS` | `llm_build_inp_embd` (line 510) — token embedding lookup | **YES** |

### Per-Layer: Attention Norm

| Op | Source | Vulkan |
|---|---|---|
| `FUSED_RMS_NORM` | `llm_build_norm` (line 662) — RMS norm + scale fused | **YES** |

Only emitted when `n_head > 0` (layers with attention).

### Per-Layer: QKV Projection

| Op | Source | Vulkan |
|---|---|---|
| `MUL_MAT` ×3 | `llm_build_mul_mat_qkv` (line 1774-1782) — Wq, Wk, Wv projections | **YES** |
| `RESHAPE` ×2 | `ggml_reshape_3d` (line 2369, 2376) — Q [head_dim, n_head, n_tokens], K similar | **YES** (zero-cost) |
| `ROPE` ×2 | `ggml_rope_ext` (line 2368, 2375) — rotary position embedding on Q and K | **YES** |

Only emitted for standard attention layers (`n_head > 0, n_head_kv > 0`). Linear attention layers emit only one `MUL_MAT` (wo projection).

### Per-Layer: KV Cache Store

| Op | Source | Vulkan |
|---|---|---|
| `VIEW` ×2 | `llm_build_kv_store` (line 572, 584) — cache write targets | **YES** (zero-cost) |
| `CPY` ×2 | `llm_build_kv_store` (line 575, 598) — K, V → cache | **YES** |
| `TRANSPOSE` | `llm_build_kv_store` (line 594) — V transpose (only if `v_trans`) | **YES** (zero-cost) |

### Per-Layer: Attention Compute

#### Flash Attention Path (`cparams.flash_attn = true`)

| Op | Source | Vulkan |
|---|---|---|
| `PERMUTE` | `llm_build_kqv` (line 1514) — Q permute [head_dim, n_tokens, n_head] | **YES** (zero-cost) |
| `VIEW` ×2 | `llm_build_kqv` (line 1518, 1550) — K, V views from cache | **YES** (zero-cost) |
| `FLASH_ATTN_EXT` | `llm_build_kqv` (line 1556) — fused flash attention | **YES**\* |
| `RESHAPE` | `ggml_reshape_2d` (line 1572) — output reshape | **YES** (zero-cost) |

\*Flash attention requires:
- Head size 128 (Nemotron standard) — supported
- KV type F16 — supported on all devices; Q4_0/Q8_0 also supported
- `subgroup_shuffle` — required on non-coopmat2 devices (Vega, RDNA2 both support this)
- Q type F32 — always the case

#### Non-Flash Path (`cparams.flash_attn = false`)

| Op | Source | Vulkan |
|---|---|---|
| `PERMUTE` | `llm_build_kqv` (line 1514) — Q permute | **YES** (zero-cost) |
| `VIEW` ×2 | `llm_build_kqv` (line 1518, 1578/1583) — K, V views from cache | **YES** (zero-cost) |
| `TRANSPOSE` + `CONT` | `llm_build_kqv` (line 1588) — V make contiguous (if `!v_trans`) | **YES** |
| `MUL_MAT` | `llm_build_kqv` (line 1594) — K×Q attention scores | **YES** |
| `SOFT_MAX` | `ggml_soft_max_ext` (line 1626) — softmax with mask and scale | **YES** |
| `MUL_MAT` | `llm_build_kqv` (line 1633) — V×softmax(KQ) | **YES** |
| `PERMUTE` | `llm_build_kqv` (line 1636) — output permute | **YES** (zero-cost) |
| `CONT` | `ggml_cont_2d` (line 1639) — make contiguous | **YES** |

Nemotron does NOT use `attn_soft_cap` (no Grok-style softcap), so `ggml_softcap`/`ggml_softcap_max` are never emitted.

### Per-Layer: Output Projection

| Op | Source | Vulkan |
|---|---|---|
| `MUL_MAT` | `llm_build_kqv` (line 1691) — Wo projection | **YES** |

### Per-Layer: Residual + FFN Norm

| Op | Source | Vulkan |
|---|---|---|
| `SCALE` | `build_deci` (line 2401) — residual scale (if `f_residual_scale != 0`) | **YES** |
| `ADD` | `build_deci` (line 2407) — attention output + residual | **YES** |
| `FUSED_RMS_NORM` | `llm_build_ffn` → `llm_build_norm` (line 823) — FFN input norm | **YES** |

### Per-Layer: FFN (Single-GPU, fused path)

When `cparams.fused_up_gate = true` (default) and weights have no biases (standard Nemotron):

| Op | Source | Vulkan |
|---|---|---|
| `FUSED_UP_GATE` | `llm_build_ffn` (line 835) — up×b \* silu(gate×b) in single dispatch | **YES** |
| `MUL_MAT` | `llm_build_ffn` (line 841) — down projection | **YES** |

### Per-Layer: FFN (Single-GPU, decomposed fallback)

When `fused_up_gate = false` or weights have biases:

| Op | Source | Vulkan |
|---|---|---|
| `MUL_MAT` | `llm_build_ffn` (line 868) — up projection | **YES** |
| `MUL_MAT` | `llm_build_ffn` (line 889) — gate projection | **YES** |
| `FUSED_MUL_UNARY` | `llm_build_ffn` (line 910) — silu(gate) \* up fused | **YES** |
| `MUL_MAT` | `llm_build_ffn` (line 968) — down projection | **YES** |

### Per-Layer: FFN (Multi-GPU split path)

When weight tensors have `extra` set (split across devices):

| Op | Source | Vulkan |
|---|---|---|
| `FUSED_RMS_NORM` | `do_split_norm` — per-device norm | **YES** |
| `FUSED_UP_GATE` | `llm_build_ffn` (line 774) — per-device fused FFN | **YES** |
| `MUL_MAT` | `llm_build_ffn` (line 780) — per-device down projection | **YES** |
| `CPY` (cast) | `llm_build_ffn` (line 787) — cast to reduce_type (F16) | **YES** |
| `REDUCE` | `llm_build_ffn` (line 811) — cross-device sum | **YES** |
| `ADD` | `llm_build_ffn` (line 804) — residual add | **YES** |

### Per-Layer: Residual

| Op | Source | Vulkan |
|---|---|---|
| `SCALE` | `build_deci` (line 2424) — FFN residual scale (if `f_residual_scale != 0`) | **YES** |
| `ADD` | `build_deci` (line 2427) — FFN output + residual | **YES** |

### Last Layer: Output Selection

| Op | Source | Vulkan |
|---|---|---|
| `GET_ROWS` | `build_deci` (line 2391-2392) — select output tokens | **YES** |

### Final: Output Norm + LM Head

| Op | Source | Vulkan |
|---|---|---|
| `FUSED_RMS_NORM` | `build_deci` (line 2439) — final RMS norm | **YES** |
| `MUL_MAT` | `build_deci` (line 2443) — lm_head projection | **YES** |
| `SCALE` | `build_deci` (line 2446) — logit scale (if `f_logit_scale != 0`) | **YES** |

## KQ Mask (Input Tensor)

| Path | Op | Vulkan |
|---|---|---|
| Flash attn, causal | Mask created directly as F16 — no op | N/A |
| Flash attn, non-causal | `CPY` F32→F16 via `ggml_cast` | **YES** |
| Non-flash | Mask stays F32 — no op | N/A |

## Summary: All Ops by Category

### Compute Ops (dispatch GPU work)

| Op | Count per layer | Vulkan |
|---|---|---|
| `FUSED_RMS_NORM` | 2 (attn_norm + ffn_norm) + 1 (final) | **YES** |
| `MUL_MAT` | 4-5 (Wq, Wk, Wv, Wo, down) | **YES** |
| `FUSED_UP_GATE` | 1 (fused FFN path) | **YES** |
| `ROPE` | 2 (Q, K) | **YES** |
| `FLASH_ATTN_EXT` | 1 (flash path) | **YES** |
| `SOFT_MAX` | 1 (non-flash path) | **YES** |
| `FUSED_MUL_UNARY` | 1 (decomposed FFN fallback) | **YES** |
| `CPY` | 2 (KV cache writes) | **YES** |
| `CONT` | 1-2 (V transpose, output) | **YES** |
| `ADD` | 2-3 (residuals) | **YES** |
| `SCALE` | 0-2 (residual/logit scale) | **YES** |
| `GET_ROWS` | 1-2 (embedding, output select) | **YES** |

### Zero-Cost Layout Ops (no GPU dispatch)

| Op | Vulkan |
|---|---|
| `RESHAPE` | **YES** |
| `VIEW` | **YES** |
| `PERMUTE` | **YES** |
| `TRANSPOSE` | **YES** |

## Verdict

**All ops in the Nemotron graph are Vulkan-supported.** Expected graph splits: **2-3** (the minimum for any model — typically embedding split and output split from the multi-GPU scheduler).

No new shaders or `supports_op` additions are needed for Nemotron.

## Conditional Ops NOT in the Standard Path

These ops can appear in `build_deci()` under non-default settings but are NOT Vulkan-supported:

| Op | Trigger | Vulkan | Impact |
|---|---|---|---|
| `HADAMARD` | `cparams.k_cache_hadamard = true` | **NO** | +2 splits per layer. Not default. |

## Flash Attention Hardware Requirements

For flash attention on Nemotron (head_size=128, KV type F16):

| GPU | coopmat2 | subgroup_shuffle | FA supported |
|---|---|---|---|
| RX 6800 XT (RDNA2) | No | Yes | **Yes** (scalar path) |
| RX Vega (GCN5) | No | Yes | **Yes** (scalar path) |
| Polaris (GCN 7790+) | No | ? | Check `subgroup_shuffle` |

If flash attention is not supported on a device, the non-flash path is used — all ops in that path are also Vulkan-supported.

## Files Referenced

| File | Lines | Role |
|---|---|---|
| `src/llama-build-context.cpp` | 2319-2454 | `build_deci()` — main graph builder |
| `src/llama-build-context.cpp` | 652-691 | `llm_build_norm()` — RMS/layer norm |
| `src/llama-build-context.cpp` | 730-998 | `llm_build_ffn()` — FFN with fused/decomposed paths |
| `src/llama-build-context.cpp` | 1487-1707 | `llm_build_kqv()` — attention compute |
| `src/llama-build-context.cpp` | 543-600 | `llm_build_kv_store()` — KV cache writes |
| `src/llama-build-context.cpp` | 1769-1807 | `llm_build_mul_mat_qkv()` — Q/K/V projections |
| `ggml/src/ggml-vulkan.cpp` | 11059-11407 | `ggml_backend_vk_supports_op()` — Vulkan op support |

---

## Part 2: nemotron_h_moe (Nemotron-3-Nano-30B-A3B)

### Target Model

**Nemotron-3-Nano-30B-A3B** uses the `nemotron_h_moe` architecture — a hybrid Mamba2 + Attention + MoE model. Available at `/opt/models/nemotron-3-nano/` in Q5_K_XL, Q6_K_XL, Q8_0, and Q8_K_XL quantizations.

This is fundamentally different from DECI. DECI is a pure transformer with per-layer flexibility. `nemotron_h_moe` combines three distinct layer types: Mamba2 SSM, standard attention, and MoE FFN.

### Model Metadata (from GGUF)

- `block_count`: 52 layers
- `embedding_length`: 2688
- `attention.head_count`: 32, `key_length`: 128, `value_length`: 128
- `expert_count`: 128, `expert_used_count`: 6, `expert_shared_count`: 1
- `ssm.conv_kernel`: 4, `ssm.state_size`: 128, `ssm.group_count`: 8, `ssm.inner_size`: 4096
- `rope.dimension_count`: 84

### Fork Status

**Our fork (`ik_llama.cpp`) does not recognize `nemotron_h_moe`.** The architecture enum, model loading, tensor mapping, and graph builder are all absent. Only `LLM_ARCH_DECI` is present.

Upstream llama.cpp (added as `llama.cpp` submodule) fully supports all three Nemotron variants: `nemotron`, `nemotron_h`, `nemotron_h_moe`.

### Layer Types

Each of the 52 layers is one of three types, determined per-layer by `hparams.is_recurrent(il)` and `hparams.n_ff(il)`:

1. **Mamba2 SSM layers** — recurrent state-space model with 1D convolution
2. **Attention layers** — standard multi-head attention with RoPE
3. **MoE FFN layers** — 128 experts with sigmoid gating, top-6 selection, plus 1 shared expert

Graph builder: `llm_build_nemotron_h` (inherits `llm_build_mamba_base`) at upstream `src/models/nemotron-h.cpp`.

### Op Trace: Mamba2 SSM Layer

| Op | Purpose | Fork Vulkan | Upstream Vulkan |
|---|---|---|---|
| `RMS_NORM` | Input normalization | **YES** | **YES** |
| `MUL_MAT` | Input projection (in_proj) | **YES** | **YES** |
| `CONCAT` | Merge conv history + current token | **YES** | **YES** |
| **`SSM_CONV`** | 1D convolution (kernel=4) | **NO** | **YES** |
| `ADD` | Conv bias | **YES** | **YES** |
| `SILU` | Activation | **YES** | **YES** |
| **`SSM_SCAN`** | Structured state-space scan (d_state=128, n_group=8) | **NO** | **YES** |
| `CPY` | Store last SSM state | **YES** | **YES** |
| `MUL` + `ADD` | Skip connection (d matrix) | **YES** | **YES** |
| **`SWIGLU`** | Gate output with z projection | **NO** | **YES** |
| `GROUP_NORM` | Grouped RMS normalization (n_group=8) | **YES** | **YES** |
| `MUL_MAT` | Output projection (out_proj) | **YES** | **YES** |

### Op Trace: Attention Layer

| Op | Purpose | Fork Vulkan |
|---|---|---|
| `RMS_NORM` | Attention norm | **YES** |
| `MUL_MAT` ×3 | Q, K, V projections | **YES** |
| `RESHAPE` ×2 | Q, K reshape | **YES** (zero-cost) |
| `ROPE` ×2 | Rotary position embedding (dim=84) | **YES** |
| `CPY` ×2 | KV cache writes | **YES** |
| `FLASH_ATTN_EXT` | Flash attention (hs=128, F16 KV) | **YES** |
| `MUL_MAT` | Output projection (Wo) | **YES** |

All attention ops are fully supported — identical to the DECI attention path.

### Op Trace: MoE FFN Layer

| Op | Purpose | Fork Vulkan | Upstream Vulkan |
|---|---|---|---|
| `RMS_NORM` | FFN input norm | **YES** | **YES** |
| `MUL_MAT` | Gate logits (router) | **YES** | **YES** |
| `SIGMOID` | Expert probabilities | **YES** | **YES** |
| **`ARGSORT` (top_k)** | Select top-6 experts | Partial | **YES** |
| `GET_ROWS` | Extract routing weights | **YES** | **YES** |
| **`MUL_MAT_ID`** ×2+ | Expert up/down projections | **YES** | **YES** |
| **`ADD_ID`** | Expert-specific biases | **NO** | **YES** |
| `RELU` + `SQR` | Activation (ReLU²) | **YES** | **YES** |
| `MUL` | Apply routing weights | **YES** | **YES** |
| `MUL_MAT` ×2 | Shared expert up/down | **YES** | **YES** |
| **`SET_ROWS`** | MoE output aggregation | **NO** | **YES** |
| `ADD` | Combine MoE + shared expert | **YES** | **YES** |

### Missing Ops for nemotron_h_moe

**5 ops must be added** to the fork's Vulkan backend:

| Op | Used In | Complexity | Notes |
|---|---|---|---|
| `SSM_CONV` | Mamba2 — 1D convolution | Medium | Upstream has shader |
| `SSM_SCAN` | Mamba2 — parallel associative scan | **High** | Upstream has shader; most complex op |
| `SWIGLU` | Mamba2 — gated activation | Low | Upstream has shader |
| `ADD_ID` | MoE — indexed bias add | Low | Upstream has shader |
| `SET_ROWS` | MoE — scatter results | Medium | Upstream has shader (commented out in fork) |

**1 op may need extension:**

| Op | Issue |
|---|---|
| `ARGSORT` | Fork has standard argsort. Upstream added `argsort_top_k` for MoE routing. Need to verify compatibility. |

### Architecture Support Gap

Beyond the 5 missing Vulkan ops, the fork needs:

1. **Architecture registration**: Add `LLM_ARCH_NEMOTRON_H_MOE` (and `NEMOTRON_H`, `NEMOTRON`) to the architecture enum
2. **Model loading**: Tensor name mapping for all nemotron_h_moe tensors (401 tensors in the GGUF)
3. **Graph builder**: `build_nemotron_h_moe()` — hybrid layer dispatch (SSM vs attention vs MoE per layer)
4. **Recurrent state management**: Mamba2 requires conv state + SSM state caching (different from KV cache)
5. **MoE routing logic**: Expert selection, scatter/gather for parallel expert computation

The cleanest path is to port the upstream `llm_build_nemotron_h` and `llm_build_mamba_base` implementations, then ensure all ops have Vulkan shaders.

### Files Referenced (upstream llama.cpp)

| File | Role |
|---|---|
| `src/models/nemotron-h.cpp` | `llm_build_nemotron_h` graph builder |
| `src/models/mamba-base.cpp` | `llm_build_mamba_base` — SSM layer builder |
| `src/llama-graph.cpp` | `build_moe_ffn` — MoE FFN builder |
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | Upstream Vulkan backend — has all required shaders |

---

## Test Coverage Audit

Every compute op in the DECI hot path (and shared ops with nemotron_h_moe) was cross-referenced against `test-backend-ops.cpp`.

### Previously Well-Covered

| Op | Config | Tests | Notes |
|---|---|---|---|
| `MUL_MAT` | Q4_K/Q6_K × F32 | **Extensive** | Standard suite, all quant types |
| `ROPE` | head_dim=128, mode=0 | **Well covered** | 128-dim explicitly tested |
| `FLASH_ATTN_EXT` | hs=128, F16 KV, causal | **Well covered** | Full parameter matrix |
| `CPY` | F32→F16 KV writes | **Well covered** | All type combos |
| `GET_ROWS` | Embedding lookup | **Adequate** | Multiple types, batch sizes |
| `ADD` / `SCALE` | Residual connections | **Well covered** | Broadcast + non-broadcast |

### Gaps Found and Fixed

| Op | Before | After | Notes |
|---|---|---|---|
| FUSED_RMS_NORM | 0 tests | 7 | 4 epsilon values + 3 dimension variants |
| FUSED_MUL_UNARY | 0 tests | 8 | 3 activations × 2 dims + 2 edge cases |
| CONT | 1 test | 8 | F16 type, Nemotron-scale shapes. No quant implications — operates on F32/F16, never quantized weights. |
| SOFT_MAX | 52 tests | 63 | wg512 path (ncols>1024), F16 mask pipeline, Nemotron attention dims |

### Dual-GPU Verification (RADV VEGA10 + RADV NAVI21)

All tests verified on both GPUs:

| Suite | Vega (wave64) | 6800 XT (wave32) |
|---|---|---|
| FUSED_UP_GATE | 143/143 PASS | 143/143 PASS |
| MULTI_ADD | 12/12 PASS | 12/12 PASS |
| FUSED_RMS_NORM | 7/7 PASS | 7/7 PASS |
| FUSED_MUL_UNARY | 8/8 PASS | 8/8 PASS |
| CONT | 8/8 PASS | 8/8 PASS |
| SOFT_MAX | 63/63 PASS | 63/63 PASS |
| Standard ops | 1222/1222 PASS | 1222/1222 PASS |

### Final Test Counts

| Suite | Count | Status |
|---|---|---|
| Standard backend-ops | 1222/1222 | PASS |
| FUSED_UP_GATE (custom eval) | 143/143 | PASS |
| MULTI_ADD (custom eval) | 12/12 | PASS |
| **Total** | **1377** | **PASS** |

---

## REDUCE Op Implementation

### Problem

Multi-GPU split-mode graph (`-sm graph`) crashed with `Missing op: REDUCE`. The REDUCE op is emitted at 5 sites in `llama-build-context.cpp` (FFN, MoE, attention) for cross-device partial sum accumulation. It had no Vulkan shader and no `supports_op` entry.

### Architecture

`ggml_reduce()` creates a VIEW tensor with `op = GGML_OP_REDUCE` and `src[0..n-1]` pointing to partial sums from each GPU. The scheduler creates single-node REDUCE splits, provides barriers between devices, and expects the backend to handle the cross-device data movement.

The CUDA backend implements REDUCE using `cudaMemcpyPeerAsync` with ring-reduce and P2P memory access. Vulkan does not have P2P — each device has isolated memory.

### Vulkan Implementation

CPU-mediated host-staging approach:

1. Read local partial sum from the REDUCE node's view buffer to host memory
2. For each remote source, read its data to host memory via `ggml_vk_buffer_read` (submits transfer commands on the remote device)
3. Element-wise ADD on CPU (F32 or F16)
4. Write accumulated result back to local device buffer

The implementation lives in `ggml_vk_reduce()`, called early from `graph_compute` before any GPU command recording. REDUCE is added to `supports_op` and handled as a no-op return in `build_graph`.

### Changes

| File | Change |
|---|---|
| `ggml/src/ggml-vulkan.cpp` | `ggml_vk_reduce()` — CPU-mediated cross-device ADD |
| `ggml/src/ggml-vulkan.cpp` | `graph_compute` — early return for REDUCE graphs |
| `ggml/src/ggml-vulkan.cpp` | `build_graph` — `case GGML_OP_REDUCE: return false` |
| `ggml/src/ggml-vulkan.cpp` | `supports_op` — `case GGML_OP_REDUCE: return true` (F32/F16, contiguous) |

### Performance

Tested with Nemotron-Nano-4B-Q4_K_M on Vega + 6800 XT:

| Mode | Splits | Prompt (tok/s) | Gen (tok/s) | Notes |
|---|---|---|---|---|
| Single GPU (6800 XT) | 2 | 196 | 47 | Baseline |
| Multi-GPU, layer split | 3 | 98 | 18 | No REDUCE needed |
| Multi-GPU, graph split | 193 | 9 | 6.5 | 64 REDUCE ops per token |

The CPU-mediated REDUCE is a correctness implementation — not optimized. Each REDUCE does two synchronous GPU→host reads + one host→GPU write. For graph-split mode with 193 splits, this creates significant overhead. Layer-split mode (default) avoids REDUCE entirely.

### Future Optimization

- **dmabuf GPU-side REDUCE**: Use the existing dmabuf infrastructure to copy remote buffers as GPU→GPU, then dispatch an ADD shader. Eliminates CPU round-trip.
- **Async pipelining**: Overlap REDUCE copies with subsequent GPU compute.
- Currently only F32 and F16 types are supported (matches the model's `reduce_type` setting).

---

## Full Model Trace: Nemotron-Nano-4B (Graph Split Mode, 2 GPUs)

Complete computation graph for Llama-3.1-Nemotron-Nano-4B-v1.1 (Q4_K_M) with split-mode graph on Vega + 6800 XT.

### Graph Statistics

- **1606 graph nodes** across **193 splits**
- **64 REDUCE ops** (2 per layer × 32 layers)
- Weight split: ~25% Vega (8GB) / ~75% 6800 XT (16GB)

### Op Frequency

| Op | Count | Per Layer | Purpose |
|---|---|---|---|
| `MUL_MAT` | 321 | ~10 | Q/K/V projections, Wo, up, gate, down + lm_head |
| `CPY` | 256 | 8 | KV cache writes + F32→F16 casts for cross-device reduce |
| `FUSED_RMS_NORM` | 129 | 4 | attn_norm + ffn_norm (×2 devices) + final output_norm |
| `ROPE` | 128 | 4 | Q, K rotary position embedding (×2 devices) |
| `REDUCE` | 64 | 2 | Attention combine + FFN combine |
| `FUSED_UP_GATE` | 64 | 2 | Fused SiLU FFN (×2 devices) |
| `FLASH_ATTN_EXT` | 64 | 2 | Flash attention (×2 devices) |
| `ADD` | 64 | 2 | Residual connections (on Vulkan1, before REDUCE) |
| `GET_ROWS` | 4 | — | Embedding lookup + output token selection |

### Per-Layer Pattern (6 splits)

Each of the 32 layers creates 6 splits following an identical pattern:

```
Split N+0: Vulkan0 — Attention (device 0's head partition, ~8 heads)
  FUSED_RMS_NORM(attn_norm) → MUL_MAT(Wq) → MUL_MAT(Wk) → MUL_MAT(Wv)
  → ROPE(Q) → ROPE(K) → CPY(K→cache) → CPY(V→cache)
  → FLASH_ATTN_EXT → MUL_MAT(Wo) → CPY(F32→F16 cast)

Split N+1: Vulkan1 — Attention (device 1's head partition, ~24 heads)
  FUSED_RMS_NORM(attn_norm) → MUL_MAT(Wq) → MUL_MAT(Wk) → MUL_MAT(Wv)
  → ROPE(Q) → ROPE(K) → CPY(K→cache) → CPY(V→cache)
  → FLASH_ATTN_EXT → MUL_MAT(Wo) → CPY(F32→F16 cast) → ADD(residual)

Split N+2: Vulkan1 — REDUCE
  REDUCE(device0_attn_partial, device1_attn_partial) → attn_combined

Split N+3: Vulkan0 — FFN (device 0's column partition)
  FUSED_RMS_NORM(ffn_norm) → FUSED_UP_GATE(up, gate) → MUL_MAT(down) → CPY(F32→F16)

Split N+4: Vulkan1 — FFN (device 1's column partition)
  FUSED_RMS_NORM(ffn_norm) → FUSED_UP_GATE(up, gate) → MUL_MAT(down)
  → CPY(F32→F16) → ADD(residual)

Split N+5: Vulkan1 — REDUCE
  REDUCE(device0_ffn_partial, device1_ffn_partial) → l_out
```

### Complete Split Map

```
Split   0: CPU        GET_ROWS(token_embd)                 ← Embedding
Split 1-6: Vk0+Vk1   Layer 0 (attn → REDUCE → ffn → REDUCE)
Split 7-12:           Layer 1
  ...
Split 187-192:        Layer 31 (last layer)
Split 192:            Also includes FUSED_RMS_NORM(output_norm) + MUL_MAT(lm_head)
```

Total: 1 (embedding) + 6×32 (layers) = 193 splits. The final output norm and lm_head are merged into the last REDUCE split on Vulkan1.

### Key Observations

1. **Weight partitioning**: The ~25/75% VRAM split means Vega handles fewer heads (8 of 32 attention heads, ~25% of FFN columns). The scheduler assigns proportionally.

2. **REDUCE is always on Vulkan1**: Every REDUCE split runs on device 1 (6800 XT). Device 1 performs the residual ADD before REDUCE, so the result is a VIEW of device 1's partial sum + device 0's partial sum.

3. **Redundant work**: Both devices independently compute `FUSED_RMS_NORM` on the same input (the previous layer's output). This is necessary because each device only has its own weight partition, but the norm input must be the full vector. The scheduler copies the REDUCE result to both devices.

4. **CPY for type casting**: The `CPY` after `MUL_MAT(Wo)` and `MUL_MAT(down)` casts from F32 to F16 (`reduce_type = f16`), reducing cross-device transfer size by 2×.

5. **No scheduler inputs after Split #2**: Splits 3-192 all show `# 0 inputs`, meaning the scheduler doesn't need to copy data between splits — the REDUCE op handles cross-device data flow directly.
