# MTP Speculative Decoding for Qwen3.5 — Peer Instructions

## Goal

Enable MTP (Multi-Token Prediction) speculative decoding for Qwen3.5
models in llama.cpp (upstream). This is the llama.cpp submodule side of
the work. A parallel effort on ik_llama.cpp handles the Vulkan backend
and hybrid GPU scheduling.

## Background

Qwen3.5 models ship with built-in MTP layers — one extra transformer
block that predicts the next token given the current hidden state. This
enables speculative decoding without a separate draft model.

**Models:**
- Qwen3.5-27B (dense, `LLM_ARCH_QWEN35`) — simplest target
- Qwen3.5-35B-A3B (hybrid MoE + DeltaNet, `LLM_ARCH_QWEN35MOE`) — production target
- Qwen3.5-122B-A10B (hybrid MoE, same arch as 35B) — stretch target

**MTP tensors in HF safetensors** (35B-A3B example):
```
mtp.fc.weight                                          # lm_head projection
mtp.layers.0.self_attn.{q,k,v,o}_proj.weight           # full attention (NOT DeltaNet)
mtp.layers.0.self_attn.{q,k}_norm.weight                # QK norm
mtp.layers.0.mlp.experts.{0..255}.{gate,up,down}_proj.weight  # MoE FFN
mtp.layers.0.mlp.gate.weight                            # router
mtp.layers.0.mlp.shared_expert.{gate,up,down}_proj.weight
mtp.layers.0.mlp.shared_expert_gate.weight
mtp.layers.0.{input,post_attention}_layernorm.weight
mtp.norm.weight
mtp.pre_fc_norm_embedding.weight
mtp.pre_fc_norm_hidden.weight
```

Config keys: `mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings: false`

The dense 27B MTP layer is simpler (dense FFN instead of MoE).

**Current state of MTP in llama.cpp:**
- No MTP support exists. `speculative.cpp` has draft, eagle3, ngram — no MTP type.
- `convert_hf_to_gguf.py` explicitly strips MTP tensors.
- Upstream PR #20700 is WIP for dense Qwen3.5 only, not merged.

**Current state of MTP in ik_llama.cpp:**
- Substantial MTP infrastructure exists (KV cache ops, graph building, mtp_op_type enum)
- Currently gated to `LLM_ARCH_GLM4_MOE` only (line 5497 of llama.cpp)
- `nextn_predict_layers` hparam is the internal name for MTP layer count
- Architecture `LLM_ARCH_QWEN35MOE` and `LLM_ARCH_QWEN35` are registered
- Tensor loading for qwen35moe exists but skips MTP tensors

## What needs to happen (llama.cpp side)

### Step 1: GGUF conversion — preserve MTP tensors

**File:** `convert_hf_to_gguf.py`

The converter currently skips MTP tensors. We need a Qwen3.5-aware
conversion path that maps HF MTP tensor names to GGUF tensor names.

GGUF tensor naming convention (following the GLM4 NextN pattern):
```
# HF name                                    → GGUF name
mtp.layers.0.self_attn.q_proj.weight         → blk.{N}.attn_q.weight
mtp.layers.0.self_attn.k_proj.weight         → blk.{N}.attn_k.weight
mtp.layers.0.self_attn.v_proj.weight         → blk.{N}.attn_v.weight
mtp.layers.0.self_attn.o_proj.weight         → blk.{N}.attn_output.weight
mtp.layers.0.self_attn.q_norm.weight         → blk.{N}.attn_q_norm.weight
mtp.layers.0.self_attn.k_norm.weight         → blk.{N}.attn_k_norm.weight
mtp.layers.0.mlp.gate.weight                 → blk.{N}.ffn_gate_inp.weight
mtp.layers.0.mlp.experts.X.*                 → blk.{N}.ffn_*.X.weight
mtp.layers.0.mlp.shared_expert.*             → blk.{N}.ffn_*_shared.weight
mtp.layers.0.mlp.shared_expert_gate.weight   → blk.{N}.ffn_gate_inp_shared.weight
mtp.layers.0.input_layernorm.weight          → blk.{N}.attn_norm.weight
mtp.layers.0.post_attention_layernorm.weight → blk.{N}.post_attention_norm.weight
mtp.norm.weight                              → nextn_enorm.weight  (or similar)
mtp.pre_fc_norm_embedding.weight             → nextn_hnorm.weight
mtp.pre_fc_norm_hidden.weight                → nextn_shared_head_norm.weight
mtp.fc.weight                                → nextn_shared_head_head.weight
```

Where `N = n_layer` (i.e., MTP layers are appended after the main model
layers). This is the same convention ik_llama.cpp uses for GLM4 NextN.

The GGUF metadata must include:
- `qwen35moe.nextn_predict_layers = 1` (or equivalent KV key)

**Approach:** Add a `Qwen35MoeModel` class (or modify existing) in
`convert_hf_to_gguf.py` that handles MTP tensors. Study the existing
`Glm4MoeModel` class for the pattern.

**Verification:** After conversion, count tensors. The 35B-A3B should
have 733 + ~785 MTP tensors = ~1518 total. Run `gguf-dump` to verify
MTP tensor names and shapes.

### Step 2: Speculative decoding type — COMMON_SPECULATIVE_TYPE_MTP

**Files:** `common/speculative.h`, `common/speculative.cpp`

Add a new speculative type and state class:

```cpp
// In the type enum (common.h or wherever it's defined):
COMMON_SPECULATIVE_TYPE_MTP

// New state class:
struct common_speculative_state_mtp : public common_speculative_state {
    // MTP uses the target model's own context — no separate draft context
    // The MTP forward pass reuses the target model's KV cache for attention
    // and adds one transformer block (the MTP layer) after the main model
    
    llama_tokens draft(const llama_tokens & prompt, llama_token id_last) override;
    void accept(uint16_t n_accepted) override;
};
```

Key difference from draft model: MTP doesn't need a separate model or
context. It runs inside the target model's context using `mtp_op_type`
to control what the forward pass does.

### Step 3: Compatibility check fix

**File:** `common/speculative.cpp`, function `common_speculative_is_compat`

The current implementation uses a single `llama_decode` with 2 tokens.
For hybrid models (DeltaNet), checkpoint/restore only fires at batch
boundaries. This causes the compat check to fail incorrectly.

Fix: split into two single-token decode calls:
```cpp
// Instead of batch of 2:
llama_decode(ctx, llama_batch_get_one(tmp.data(), 1));
llama_decode(ctx, llama_batch_get_one(tmp.data() + 1, 1));
```

### Step 4: Recurrent memory OOB fix

**File:** `src/llama-memory-recurrent.cpp`

In the checkpoint creation logic, `cells[next_empty_cell].is_empty()`
is called without bounds checking. When all cells are in use (spec
decode uses parallel sequences), this reads past the array.

Fix: `if (next_empty_cell < size && cells[next_empty_cell].is_empty())`

## What you do NOT need to do

- **Vulkan backend work** — handled on the ik_llama.cpp side
- **GPU scheduling / split MTP** — handled on the ik_llama.cpp side
- **MoE expert optimization** — not needed for correctness
- **DeltaNet state rollback** — complex, defer to ik_llama.cpp

## Testing

Start with **Qwen3.5-27B** (dense). It's simpler:
- Dense FFN in MTP layer (no MoE routing)
- Pure attention (no DeltaNet layers)
- Already available at `/opt/models/qwen3.5-27b/`

1. Convert HF → GGUF with MTP tensors preserved
2. Verify MTP tensors present with `gguf-dump`
3. Load model, confirm `nextn_predict_layers = 1` in log output
4. Run basic text generation (no spec decode) to verify model still works
5. Enable MTP spec decode, measure acceptance rate and throughput

## Hardware constraints

This peer has fewer resources than the main workstation. Focus on:
- CPU inference is fine for testing
- Single GPU if available
- The 27B Q4_K_M (16G) is the right test model
- Don't worry about multi-GPU scheduling

## Reference code

- **ik_llama.cpp MTP infra:** `src/llama.cpp` lines 3615-3768 (MTP graph building),
  lines 1076-1165 (KV cache slot finding for MTP ops)
- **GLM4 NextN pattern:** `src/llama-load-tensors.cpp` lines 2619+
  (tensor loading for nextn_predict_layers)
- **Polaris branch:** `slartibardfast/llama.cpp:polaris-hybrid-cpu-opt` has a
  working MTP implementation with 28.1 t/s / 82% acceptance on RTX 5060 Ti.
  Study `common/speculative.cpp` on that branch for the `common_speculative_state_mtp` class.
- **Upstream PR #20700:** WIP, dense-only, but has convert_hf_to_gguf.py changes

## Deliverables

1. `convert_hf_to_gguf.py` patch that preserves Qwen3.5 MTP tensors
2. Qwen3.5-27B GGUF with MTP tensors (Q4_K_M or Q8_0)
3. `common_speculative_state_mtp` class wired into speculative.cpp
4. Compat-check and OOB fixes
5. Benchmark: acceptance rate and t/s with MTP on CPU for Qwen3.5-27B
