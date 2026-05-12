#!/usr/bin/env python3
"""
dflash_drafter_to_gguf.py
=========================

Convert a Qwen3.6-27B-DFlash drafter (z-lab/Qwen3.6-27B-DFlash, HF
architecture `DFlashDraftModel`) to a BF16 GGUF compatible with
ik_llama.cpp's forthcoming `LLM_ARCH_DFLASH_DRAFTER` graph.

Design rationale
----------------
The DFlash drafter is structurally a 5-layer Qwen3-style transformer with
two non-standard surfaces:

1. NO embed_tokens / NO lm_head.
   The drafter consumes already-embedded hidden states from the target
   (concatenated over `target_layer_ids`, fused via `fc`) and emits
   refined hidden states that the *target's* lm_head turns into logits.
   This is identical to the EAGLE/MTP pattern: the drafter is a "trunk
   adapter," not a standalone LM.

2. `fc.weight` (5120, 25600) — multi-layer hidden-state fusion.
   25600 = num_target_layers_used (5) × hidden_size (5120). The five
   target layers are configured per `dflash_config.target_layer_ids`;
   their hidden states are concatenated along the feature axis and
   projected back down to hidden_size before entering the drafter trunk.

3. `hidden_norm.weight` (5120,) — RMSNorm applied to the post-`fc`
   fused input, before the first transformer block.

Everything else (attention, MLP, norms) follows Qwen3 conventions:
GQA 32q/8kv, head_dim 128, SwiGLU MLP at intermediate 17408, per-head
q_norm/k_norm (RMSNorm over head_dim), RMSNorm trunk norms,
sliding-window attention on layers 0..3 + full attention on layer 4.

Tensor naming maps to GGUF block conventions so that ik_llama.cpp's
existing per-block dispatch can be reused once the arch is registered:
    blk.N.attn_{q,k,v,output,q_norm,k_norm,norm}
    blk.N.ffn_{gate,up,down,norm}
plus two architecture-specific tensors:
    dflash.fc        — the 5×hidden→hidden fusion
    dflash.hidden_norm — the post-fusion norm
and a trunk-level:
    output_norm      — final norm before handoff to target lm_head

KV metadata
-----------
Standard GGUF general/* and architecture/* keys, plus DFlash-specific:
    dflash_drafter.block_size               (16)
    dflash_drafter.mask_token_id            (248070)
    dflash_drafter.target_layer_ids         (uint32 array, 5 entries)
    dflash_drafter.num_target_layers        (64 — target's full depth)

No tokenizer is emitted. The drafter shares the target's tokenizer and
its `mask_token_id` is a real entry in the target's vocab.

LOADER CONTRACT (mandatory):
    The drafter GGUF is NOT a standalone model. It carries no
    tokenizer, no embed_tokens, and no lm_head — all three are
    delegated to a paired target GGUF that MUST be loaded alongside.

    Loaders encountering this GGUF without an associated target MUST
    fail with a clear error message of the form:

        ERROR: dflash_drafter GGUF requires a paired target model.
               The drafter has no tokenizer / embed_tokens / lm_head
               of its own — these are delegated to the target.
               Pass --target-model <path/to/qwen3.6-27b.gguf> alongside.

    Pairing is enforced loader-side, not at conversion time. The
    converter intentionally produces a tokenizer-less GGUF so that
    pairing is a structural requirement, not a runtime suggestion.

Output dtype
------------
BF16 throughout. The drafter is 3.3 GiB BF16 and fits comfortably in
VRAM alongside a Q4_0_AR16 target (~17 GiB) on 2× RTX 6000 (48 GiB).
sm_75 has no native BF16 hardware so runtime kernels will cast to F16
or F32 on dispatch, but the *stored* dtype stays BF16 to preserve the
drafter's calibration for future hardware migration.

Memory
------
Conversion reads the safetensors file as torch tensors and materializes
each tensor via `.contiguous().view(torch.uint16).numpy()` for the GGUF
writer. The `.contiguous()` may copy up to per-tensor size if the source
tensor is non-contiguous (safetensors typically returns contiguous, but
not contractually guaranteed). Peak transient RSS is approximately one
tensor size + one safetensors mmap — well under 2 GiB on this host.

Output path convention
----------------------
Recommended: /opt/models/recast-out/qwen3.6-27b-dflash-bf16.gguf
(matches the production `tool1lossless` GGUF colocation pattern).
Caller chooses via `--out`.

Usage
-----
    /opt/models/venv-vllm/bin/python \\
        scripts/dflash_drafter_to_gguf.py \\
        --src /opt/models/qwen36-27b-dflash \\
        --out /opt/models/recast-out/qwen3.6-27b-dflash-bf16.gguf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

import gguf

ARCH = "dflash_drafter"
BLOCK_PREFIX = "blk"


def hf_to_gguf_name(hf_name: str) -> str:
    """Map HF DFlashDraftModel tensor name → GGUF tensor name.

    All weights pass through unmodified shape. Only the *name* changes.
    """
    # Architecture-specific surfaces (no per-block prefix)
    if hf_name == "fc.weight":
        return "dflash.fc.weight"
    if hf_name == "hidden_norm.weight":
        return "dflash.hidden_norm.weight"
    if hf_name == "norm.weight":
        return "output_norm.weight"

    # Per-block tensors: layers.N.* → blk.N.*
    if not hf_name.startswith("layers."):
        raise ValueError(f"unexpected tensor name: {hf_name}")

    rest = hf_name[len("layers."):]
    layer_str, _, suffix = rest.partition(".")
    layer_i = int(layer_str)
    block_prefix = f"{BLOCK_PREFIX}.{layer_i}"

    mapping = {
        "input_layernorm.weight":           f"{block_prefix}.attn_norm.weight",
        "post_attention_layernorm.weight":  f"{block_prefix}.ffn_norm.weight",
        "self_attn.q_proj.weight":          f"{block_prefix}.attn_q.weight",
        "self_attn.k_proj.weight":          f"{block_prefix}.attn_k.weight",
        "self_attn.v_proj.weight":          f"{block_prefix}.attn_v.weight",
        "self_attn.o_proj.weight":          f"{block_prefix}.attn_output.weight",
        "self_attn.q_norm.weight":          f"{block_prefix}.attn_q_norm.weight",
        "self_attn.k_norm.weight":          f"{block_prefix}.attn_k_norm.weight",
        "mlp.gate_proj.weight":             f"{block_prefix}.ffn_gate.weight",
        "mlp.up_proj.weight":               f"{block_prefix}.ffn_up.weight",
        "mlp.down_proj.weight":             f"{block_prefix}.ffn_down.weight",
    }
    if suffix not in mapping:
        raise ValueError(f"unknown block tensor suffix: {suffix} (full: {hf_name})")
    return mapping[suffix]


def bf16_view_as_uint16(t) -> np.ndarray:
    """Convert a torch.bfloat16 tensor into a numpy uint16 array preserving the
    raw bit pattern. The gguf writer accepts uint16 buffers for BF16 tensors.
    """
    import torch
    assert t.dtype == torch.bfloat16, f"expected bf16, got {t.dtype}"
    # .view(torch.uint16) gives a uint16 alias with the same bit pattern.
    return t.contiguous().view(torch.uint16).numpy()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", required=True, type=Path,
                   help="Path to the DFlash drafter snapshot dir (containing config.json + model.safetensors)")
    p.add_argument("--out", required=True, type=Path,
                   help="Output GGUF path")
    args = p.parse_args()

    cfg_path = args.src / "config.json"
    st_path = args.src / "model.safetensors"
    for path, label in [(cfg_path, "config"), (st_path, "weights")]:
        if not path.exists():
            print(f"ERROR: missing {label} at {path}", file=sys.stderr)
            return 1

    with cfg_path.open() as f:
        cfg = json.load(f)

    # ---- Validate config matches the architecture we're converting ----
    if cfg.get("architectures") != ["DFlashDraftModel"]:
        print(f"ERROR: unexpected architectures: {cfg.get('architectures')}", file=sys.stderr)
        return 2
    if cfg.get("model_type") != "qwen3":
        print(f"ERROR: unexpected model_type: {cfg.get('model_type')}", file=sys.stderr)
        return 2

    n_layers = cfg["num_hidden_layers"]
    hidden = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    ff = cfg["intermediate_size"]
    eps = cfg["rms_norm_eps"]
    rope_theta = cfg["rope_theta"]
    ctx = cfg["max_position_embeddings"]
    sliding_window = cfg["sliding_window"]
    layer_types = cfg["layer_types"]
    block_size = cfg["block_size"]
    mask_token_id = cfg["dflash_config"]["mask_token_id"]
    target_layer_ids = cfg["dflash_config"]["target_layer_ids"]
    num_target_layers = cfg["num_target_layers"]
    vocab_size = cfg["vocab_size"]

    # Sanity: fc.weight is shaped (hidden, len(target_layer_ids) * hidden)
    expected_fc_in = len(target_layer_ids) * hidden
    if expected_fc_in != 25600 or hidden != 5120 or len(target_layer_ids) != 5:
        print(f"WARNING: shape assumptions don't match — fc_in={expected_fc_in}, "
              f"hidden={hidden}, num_target_pickoffs={len(target_layer_ids)}", file=sys.stderr)

    print(f"== DFlash drafter → GGUF ==")
    print(f"  src           : {args.src}")
    print(f"  out           : {args.out}")
    print(f"  layers        : {n_layers}  (types: {layer_types})")
    print(f"  hidden        : {hidden}")
    print(f"  heads q/kv    : {n_heads}/{n_kv}  head_dim={head_dim}")
    print(f"  ff            : {ff}")
    print(f"  rope_theta    : {rope_theta}")
    print(f"  ctx           : {ctx}  sliding_window={sliding_window}")
    print(f"  block_size    : {block_size}")
    print(f"  mask_token_id : {mask_token_id}")
    print(f"  target_layer_ids : {target_layer_ids}  (of {num_target_layers})")
    print()

    # ---- Open writer ----
    writer = gguf.GGUFWriter(args.out, ARCH)

    # ---- General metadata ----
    writer.add_name("Qwen3.6-27B-DFlash")
    writer.add_description(
        "DFlash diffusion-block drafter for Qwen3.6-27B target. "
        f"Pairs with target via target_layer_ids={target_layer_ids}, "
        f"block_size={block_size}, mask_token_id={mask_token_id}."
    )
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_BF16)

    # ---- Architecture metadata (idiomatic llama.cpp keys) ----
    writer.add_context_length(ctx)
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(ff)
    writer.add_block_count(n_layers)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv)
    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)
    writer.add_layer_norm_rms_eps(eps)
    writer.add_rope_freq_base(float(rope_theta))
    writer.add_rope_dimension_count(head_dim)
    writer.add_vocab_size(vocab_size)

    # Sliding window: scalar window_size + per-layer pattern
    # (llama.cpp convention from gemma2/gemma3/qwen3 hybrid encoders).
    # `add_sliding_window_pattern` with a Sequence[bool] emits one bool
    # per layer at key `<arch>.attention.sliding_window_pattern`,
    # where True == SWA / False == full_attention (per writer docstring).
    writer.add_sliding_window(sliding_window)
    layer_is_sliding: list[bool] = [t == "sliding_attention" for t in layer_types]
    writer.add_sliding_window_pattern(layer_is_sliding)

    # ---- DFlash-specific metadata ----
    writer.add_uint32(f"{ARCH}.block_size", block_size)
    writer.add_uint32(f"{ARCH}.mask_token_id", mask_token_id)
    writer.add_uint32(f"{ARCH}.num_target_layers", num_target_layers)
    writer.add_array(
        f"{ARCH}.target_layer_ids",
        [int(x) for x in target_layer_ids],
    )

    # ---- Tensors ----
    import torch  # imported here so the missing-torch case errors only when actually running
    seen_hf: set[str] = set()
    expected_hf: set[str] = set()

    # Build expected set from config
    expected_hf.update(["fc.weight", "hidden_norm.weight", "norm.weight"])
    for i in range(n_layers):
        for s in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ]:
            expected_hf.add(f"layers.{i}.{s}")

    n_tensors = 0
    total_bytes = 0
    with safe_open(str(st_path), framework="pt") as f:
        keys = list(f.keys())
        unknown = set(keys) - expected_hf
        missing = expected_hf - set(keys)
        if unknown:
            print(f"WARNING: unexpected tensors in checkpoint (skipping): {sorted(unknown)}",
                  file=sys.stderr)
        if missing:
            print(f"ERROR: missing expected tensors: {sorted(missing)}", file=sys.stderr)
            return 3

        for hf_name in sorted(expected_hf):
            t = f.get_tensor(hf_name)
            if t.dtype != torch.bfloat16:
                print(f"ERROR: tensor {hf_name} dtype {t.dtype}, expected bfloat16",
                      file=sys.stderr)
                return 4
            gguf_name = hf_to_gguf_name(hf_name)

            # 1D tensors (RMSNorm / per-head-dim norm weights) → F32 in GGUF.
            # ik_llama.cpp's CUDA fused_rms_norm asserts src1 (norm weight)
            # is GGML_TYPE_F32 (ggml-cuda/norm.cu:671). Casting BF16 norm
            # weights to F32 is exact (F32 strictly contains BF16's range
            # and precision), so this is lossless for the norm path.
            # 2D weight matrices stay BF16 to preserve drafter calibration.
            if t.dim() == 1:
                data_f32 = t.float().contiguous().numpy()
                writer.add_tensor(
                    gguf_name,
                    data_f32,
                    raw_dtype=gguf.GGMLQuantizationType.F32,
                )
                seen_hf.add(hf_name)
                n_tensors += 1
                total_bytes += data_f32.nbytes
                print(f"  {hf_name:55s} -> {gguf_name:45s} {tuple(t.shape)} f32 (norm)")
            else:
                data = bf16_view_as_uint16(t)
                writer.add_tensor(
                    gguf_name,
                    data,
                    raw_dtype=gguf.GGMLQuantizationType.BF16,
                )
                seen_hf.add(hf_name)
                n_tensors += 1
                total_bytes += data.nbytes
                print(f"  {hf_name:55s} -> {gguf_name:45s} {tuple(t.shape)} bf16")

    print()
    print(f"  total tensors : {n_tensors}")
    print(f"  total bytes   : {total_bytes:,}  ({total_bytes / (1024**3):.2f} GiB)")
    print(f"  writing {args.out} ...")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    sz = args.out.stat().st_size
    print(f"  wrote {sz:,} bytes  ({sz / (1024**3):.2f} GiB)")

    # ---- Round-trip check: reopen and verify structure ----
    print()
    print(f"  round-trip check ...")
    reader = gguf.GGUFReader(str(args.out), "r")
    rt_tensors = reader.tensors
    if len(rt_tensors) != n_tensors:
        print(f"ERROR: round-trip tensor count mismatch: wrote {n_tensors}, "
              f"read back {len(rt_tensors)}", file=sys.stderr)
        return 5
    rt_names = {t.name for t in rt_tensors}
    expected_names = {hf_to_gguf_name(h) for h in expected_hf}
    missing_rt = expected_names - rt_names
    extra_rt = rt_names - expected_names
    if missing_rt or extra_rt:
        print(f"ERROR: round-trip tensor name set mismatch.", file=sys.stderr)
        if missing_rt:
            print(f"  missing on read-back: {sorted(missing_rt)}", file=sys.stderr)
        if extra_rt:
            print(f"  unexpected on read-back: {sorted(extra_rt)}", file=sys.stderr)
        return 5
    # Verify weight tensor dtypes: 1D norm-shaped tensors → F32,
    # 2D weight matrices → BF16.
    bad_dtype = []
    for t in rt_tensors:
        expected = (gguf.GGMLQuantizationType.F32
                    if len(t.shape) == 1
                    else gguf.GGMLQuantizationType.BF16)
        if t.tensor_type != expected:
            bad_dtype.append((t.name, t.tensor_type, expected))
    if bad_dtype:
        print(f"ERROR: round-trip tensor dtype check failed:", file=sys.stderr)
        for name, got, want in bad_dtype:
            print(f"  {name}: got {got}, want {want}", file=sys.stderr)
        return 5

    # Verify required DFlash-specific KV + the sliding-window contract
    required_kv = [
        f"{ARCH}.block_size",
        f"{ARCH}.mask_token_id",
        f"{ARCH}.target_layer_ids",
        f"{ARCH}.num_target_layers",
        f"{ARCH}.attention.sliding_window",
        f"{ARCH}.attention.sliding_window_pattern",
    ]
    kv_names = {f.name for f in reader.fields.values()}
    missing_kv = [k for k in required_kv if k not in kv_names]
    if missing_kv:
        print(f"ERROR: round-trip missing required KV keys: {missing_kv}",
              file=sys.stderr)
        return 5

    print(f"  round-trip OK ({len(rt_tensors)} tensors, BF16, "
          f"{len(required_kv)} required KV keys present)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
