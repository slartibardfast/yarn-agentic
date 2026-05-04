#!/usr/bin/env python3
"""
autoround_to_q4_0_gguf.py
=========================

Direct repack of an Intel AutoRound INT4 (sym, group_size=128, AutoGPTQ packing)
HuggingFace checkpoint into a Q4_0 GGUF, with NO new quantization rounding for
the lossless-eligible portion of the weights.

Why this is mathematically lossless (for the lossless-eligible portion)
----------------------------------------------------------------------
- AutoRound sym INT4 codes are uint4 in [0, 15] representing values in [-8, 7]
  (implicit zero point 8). Dequant: w = (code - 8) * scale_fp16
- Q4_0 codes are uint4 in [0, 15] representing the same [-8, 7] range with the
  same -8 dequant offset. Block size = 32 input elements, one fp16 scale per block.
- A 128-group splits exactly into four contiguous 32-blocks that share the same scale.
- Codes copy 1:1, scales copy 1:1.

Lossless-vs-V-reorder split
---------------------------
Some Qwen3-Next linear_attn weights need a row/column permutation between HF
storage order and ggml broadcast order (in_proj_qkv, in_proj_z, out_proj,
in_proj_a, in_proj_b). That permutation crosses Q4_0 32-block boundaries, so
those tensors CANNOT be losslessly Q4_0'd. They fall through to upstream's
gptq dequant + ggml-side V-reorder + FP16 emit. This fall-through is DECLARED
in code comments and stderr output — never silent.

For AutoRound symmetric dumps that lack `.g_idx` (group index is implicit when
group_size is constant), we synthesize the dequant in-tool rather than relying
on upstream's gptq branch, which would KeyError on `.g_idx`.

Tensors AutoRound left at FP16 (1D / scalar params; non-quantized linears) flow
through as F16 via the standard converter path. Embeddings, layer norms, and
the LM head also flow through unchanged.

Prerequisites
-------------
- llama.cpp checkout with qwen3_5 architecture support
- pip install gguf safetensors torch numpy

Usage
-----
    python autoround_to_q4_0_gguf.py \\
        --model-dir /opt/models/hf-cache/.../Intel--Qwen3.6-27B-int4-AutoRound/snapshots/<sha> \\
        --outfile  /opt/models/Qwen3.6-27B-Q4_0-from-autoround.gguf \\
        --llama-cpp /home/llm/yarn-agentic/llama.cpp
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Core repacking primitives
# ---------------------------------------------------------------------------


def unpack_autogptq_int4(qweight: torch.Tensor) -> np.ndarray:
    """
    AutoGPTQ INT4 packing: each int32 in qweight[r, c] holds 8 codes for output
    channel c, spanning input rows 8r .. 8r+7. Code k is in bits [4k, 4k+4).

    Returns: uint8 ndarray, shape [in_features, out_features], values in [0, 15].
    """
    qw = qweight.detach().cpu().numpy().astype(np.uint32)   # uint32 view; signed shifts in numpy are UB-ish
    in_packed_rows, out_f = qw.shape                         # rows = in_features / 8 (8 codes per int32)
    in_f = in_packed_rows * 8
    codes = np.empty((in_f, out_f), dtype=np.uint8)
    for k in range(8):                                       # k=0 -> rows 0,8,16,...   k=1 -> rows 1,9,17,...
        codes[k::8, :] = ((qw >> (4 * k)) & 0xF).astype(np.uint8)
    return codes                                             # uint8 in [0,15]; sym INT4 reads (code-8) for signed value


def repack_w4g128_to_q4_0_bytes(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> tuple[bytes, tuple[int, int]]:
    """
    Convert a single AutoRound sym W4G128 tensor to a flat byte buffer of Q4_0
    blocks for a logical tensor of shape [out_features, in_features].

    Q4_0 block layout (18 bytes per 32 input elements):
        struct block_q4_0 {
            ggml_fp16_t d;       // fp16 scale
            uint8_t     qs[16];  // 32 nibbles: qs[j] low = code[j], high = code[j+16]
        };

    Tensor data layout: blocks laid out row-major over [out_features, in_features // 32].

    Returns: (q4_0_bytes, (out_features, in_features))
    """
    if group_size % 32 != 0 or group_size < 32:
        raise ValueError(f"group_size must be a multiple of 32, got {group_size}")
    blocks_per_grp = group_size // 32                          # 128/32 = 4 Q4_0 blocks per AutoRound group

    codes_in_out = unpack_autogptq_int4(qweight)               # [in, out] uint8 in [0,15]
    codes = codes_in_out.T                                     # [out, in]
    out_f, in_f = codes.shape
    if in_f % 32 != 0:
        raise ValueError(f"in_features={in_f} not divisible by 32")
    n_blocks = in_f // 32

    cb = codes.reshape(out_f, n_blocks, 32)
    lo = cb[:, :, :16]                                         # input positions 0..15  -> low nibble of qs[j]
    hi = cb[:, :, 16:]                                         # input positions 16..31 -> high nibble of qs[j]
    qs = (lo | (hi.astype(np.uint8) << 4)).astype(np.uint8)    # matches ggml_quants.c quantize_row_q4_0_ref

    # AutoRound scale layout in the safetensor: shape [n_groups, out_f] before .T
    sc = scales.detach().cpu().numpy().astype(np.float16).T    # -> [out, n_groups]
    n_groups = in_f // group_size
    if sc.shape != (out_f, n_groups):
        raise ValueError(f"scales shape (post-T) {sc.shape}, expected ({out_f}, {n_groups})")

    # Replicate each group's scale across its 4 child Q4_0 blocks (no rounding, just copy)
    sc_per_block = np.repeat(sc, blocks_per_grp, axis=1)       # [out, n_blocks]

    packed = np.empty((out_f, n_blocks, 18), dtype=np.uint8)
    packed[:, :, 0:2] = sc_per_block.view(np.uint8).reshape(out_f, n_blocks, 2)
    packed[:, :, 2:18] = qs
    return packed.tobytes(), (out_f, in_f)


def dequant_w4g128_sym(qweight: torch.Tensor, scales: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """
    Dequantize an AutoRound sym W4G128 packed tensor to a [out_features, in_features]
    FP32 torch.Tensor. Used for V-reorder remnants that cannot be losslessly Q4_0'd
    and need to flow through modify_tensors as a regular float tensor.
    """
    codes_in_out = unpack_autogptq_int4(qweight)               # [in, out] uint8 in [0,15]
    in_f, out_f = codes_in_out.shape
    if in_f % group_size != 0:
        raise ValueError(f"in_features={in_f} not divisible by group_size={group_size}")
    n_groups = in_f // group_size
    sc = scales.detach().cpu().numpy().astype(np.float32)      # [n_groups, out_f]
    if sc.shape != (n_groups, out_f):
        raise ValueError(f"scales shape {sc.shape}, expected ({n_groups}, {out_f})")
    sc_full = np.repeat(sc, group_size, axis=0)                # [in_f, out_f]
    w = (codes_in_out.astype(np.int32) - 8).astype(np.float32) * sc_full
    return torch.from_numpy(w.T.copy())                        # [out, in] HF convention


# ---------------------------------------------------------------------------
# Lossless-vs-V-reorder split
# ---------------------------------------------------------------------------

# Tensors that need V-channel row/col reorder in ggml broadcast layout. The
# permutation crosses Q4_0 32-block boundaries so we cannot losslessly repack;
# they go through dequant_w4g128_sym + modify_tensors V-reorder + FP16 emit.
V_REORDER_QWEIGHT_SUFFIXES = (
    ".linear_attn.in_proj_qkv.qweight",
    ".linear_attn.in_proj_z.qweight",
    ".linear_attn.out_proj.qweight",
    ".linear_attn.in_proj_a.qweight",
    ".linear_attn.in_proj_b.qweight",
)


def is_lossless_q4_0_candidate(qweight_name: str) -> bool:
    return not any(qweight_name.endswith(s) for s in V_REORDER_QWEIGHT_SUFFIXES)


# ---------------------------------------------------------------------------
# Self-check (Step 1.5): verify first emitted block round-trips bit-exactly
# ---------------------------------------------------------------------------


def _self_check_block0(qweight: torch.Tensor, scales: torch.Tensor, blob: bytes):
    """
    Block 0 (out=0, input positions 0..31) must contain:
      bytes 0..2:  fp16 d == scales[0, 0]
      bytes 2..18: 16 nibbles == codes from unpack_autogptq_int4(qweight)[:32, 0]
    """
    if len(blob) < 18:
        raise AssertionError(f"[Tool 1 self-check] blob shorter than one Q4_0 block ({len(blob)} bytes)")
    block0 = np.frombuffer(blob[:18], dtype=np.uint8)

    scale_fp16 = np.frombuffer(block0[:2].tobytes(), dtype=np.float16)[0]
    expected_scale = np.float16(scales[0, 0].detach().cpu().item())
    if scale_fp16 != expected_scale:
        raise AssertionError(
            f"[Tool 1 self-check] block-0 scale mismatch: got {scale_fp16}, expected {expected_scale}"
        )

    qs = block0[2:18]
    lo = qs & 0x0F
    hi = (qs >> 4) & 0x0F
    decoded = np.concatenate([lo, hi]).astype(np.uint8)        # 32 codes; positions 0..31 of out=0

    expected = unpack_autogptq_int4(qweight)[:32, 0].astype(np.uint8)
    if not np.array_equal(decoded, expected):
        raise AssertionError(
            f"[Tool 1 self-check] block-0 codes mismatch:\n"
            f"  got     ={decoded.tolist()}\n"
            f"  expected={expected.tolist()}"
        )


# ---------------------------------------------------------------------------
# convert_hf_to_gguf.py monkey-patch
# ---------------------------------------------------------------------------


def _strip_lm_prefix(name: str) -> str:
    """
    Strip the `language_model.` namespace inserted by VL multimodal HF dumps so
    the tensor name matches QWEN35 tensor_map entries.

    Upstream's Qwen3_5TextModel does not strip this; many other VL converters
    do (see InternVL / Glm4v / Kimi-VL / Nemotron paths in convert_hf_to_gguf).
    Pure name remap — bit-identical Q4_0 emit either way.
    """
    if name.startswith("model.language_model."):
        return "model." + name[len("model.language_model."):]
    if name.startswith("language_model."):
        return name[len("language_model."):]
    return name


def _is_vision_tensor(name: str) -> bool:
    """Vision-tower tensors that the text converter must skip."""
    return name.startswith(("model.visual.", "visual.", "model.vision_tower.", "vision_tower."))


def _remap_mtp_to_appended_layer(base: str, num_hidden_layers: int) -> str | None:
    """
    Map the MTP base names emitted by Qwen3_5TextModel.modify_tensors into
    `model.layers.{N}.<rest>` form so map_tensor_name resolves them.

    Returns the remapped base or None if `base` is not an MTP tensor.
    """
    if base.startswith("mtp.layers."):
        # mtp.layers.{k}.<rest> -> model.layers.{k + num_hidden_layers}.<rest>
        rest = base[len("mtp.layers."):]
        k_str, sep, tail = rest.partition(".")
        if not sep or not k_str.isdigit():
            return None
        return f"model.layers.{int(k_str) + num_hidden_layers}.{tail}"
    return None


def patch_convert(convert_module):
    """
    Patch ModelBase.dequant_model to:
      1) intercept quant_method='auto-round' (or 'auto_round')
      2) emit lossless-eligible .qweight triples directly as Q4_0 via
         gguf_writer.add_tensor(raw_dtype=Q4_0)
      3) replace V-reorder-remnant .qweight triples with a FP32 dequant lambda
         in self.model_tensors so they flow through the standard
         modify_tensors -> V-reorder -> FP16 emit path
      4) leave non-quantized tensors untouched

    Also patch Qwen3_5TextModel.modify_tensors to:
      - skip vision-tower tensors (text converter never emits them)
      - strip `model.language_model.` prefix used by the VL dump variant

    Both patches are pure name/skip remap with no quality impact on emitted
    bytes; they only make the converter run end-to-end on Intel's
    Qwen3.6-27B-int4-AutoRound dump (and any other VL-form auto-round dump).
    """
    import gguf
    from gguf.lazy import LazyBase

    Model = getattr(convert_module, "Model", None) or convert_module.ModelBase
    original_dequant = Model.dequant_model

    Qwen3_5TextModel = getattr(convert_module, "Qwen3_5TextModel", None)
    if Qwen3_5TextModel is not None:
        original_text_modify = Qwen3_5TextModel.modify_tensors

        def patched_text_modify(self, data_torch, name, bid):
            if _is_vision_tensor(name):
                return
            name = _strip_lm_prefix(name)
            yield from original_text_modify(self, data_torch, name, bid)

        Qwen3_5TextModel.modify_tensors = patched_text_modify

    def _materialize(t):
        """Force a possibly-lazy torch.Tensor to be eager (no-op if already eager)."""
        if isinstance(t, LazyBase):
            return LazyBase.to_eager(t)
        return t

    def patched_dequant_model(self):
        qc = self.hparams.get("quantization_config")
        if not isinstance(qc, dict):
            return original_dequant(self)
        method = qc.get("quant_method")
        if method not in ("auto-round", "auto_round"):
            return original_dequant(self)

        bits = qc.get("bits")
        sym = qc.get("sym", False)
        group_size = qc.get("group_size", 128)
        if bits != 4 or not sym:
            raise NotImplementedError(
                f"Tool 1 supports auto-round bits=4 sym=true only; got bits={bits} sym={sym}. "
                f"Surface to user before any workaround — do not silently substitute."
            )
        if group_size != 128:
            raise NotImplementedError(
                f"Tool 1 supports group_size=128 only; got {group_size}. "
                f"Surface to user before any workaround."
            )

        num_hidden_layers = self.hparams.get("num_hidden_layers")
        if num_hidden_layers is None and isinstance(self.hparams.get("text_config"), dict):
            num_hidden_layers = self.hparams["text_config"].get("num_hidden_layers")

        losslessly_handled: list[str] = []
        v_reorder_dequanted: list[str] = []
        skipped_unmapped: list[str] = []
        self_checked = False

        qweight_keys = [k for k in list(self.model_tensors.keys()) if k.endswith(".qweight")]

        for qw_name in qweight_keys:
            if _is_vision_tensor(qw_name):
                # Vision tower never carries .qweight in this dump anyway, but be defensive.
                continue
            base = qw_name[: -len(".qweight")]
            scales_key = base + ".scales"
            if scales_key not in self.model_tensors:
                skipped_unmapped.append(qw_name)
                continue

            if is_lossless_q4_0_candidate(qw_name):
                # ---- LOSSLESS PATH: direct Q4_0 emit ----
                # Compute the GGUF tensor name. For VL dumps strip the
                # `language_model.` namespace; for MTP body tensors apply the
                # `mtp.layers.{k}` -> `model.layers.{k + num_hidden_layers}` remap.
                map_input = _strip_lm_prefix(base)
                mtp_remapped = _remap_mtp_to_appended_layer(map_input, num_hidden_layers or 0)
                if mtp_remapped is not None:
                    map_input = mtp_remapped

                try:
                    gguf_name = self.map_tensor_name(map_input + ".weight")
                except (ValueError, KeyError):
                    skipped_unmapped.append(qw_name)
                    continue

                qweight_t = _materialize(self.model_tensors[qw_name]())
                scales_t = _materialize(self.model_tensors[scales_key]())

                blob, (out_f, in_f) = repack_w4g128_to_q4_0_bytes(
                    qweight_t, scales_t, group_size=group_size
                )

                if not self_checked:
                    _self_check_block0(qweight_t, scales_t, blob)
                    print(
                        f"[Tool 1] Self-check passed on first lossless emit ({gguf_name})",
                        file=sys.stderr,
                    )
                    self_checked = True

                # Reshape to (out_f, bytes_per_row) — the writer expects byte
                # shape in PyTorch order for uint8+raw_dtype tensors and will
                # convert it to logical shape via quant_shape_from_byte_shape;
                # the on-disk dim order is reversed at write_ti_data_to_file.
                bytes_per_row = (in_f // 32) * 18
                arr = np.frombuffer(blob, dtype=np.uint8).reshape(out_f, bytes_per_row)
                self.gguf_writer.add_tensor(
                    gguf_name,
                    arr,
                    raw_dtype=gguf.GGMLQuantizationType.Q4_0,
                )

                for suf in (".qweight", ".qzeros", ".scales", ".g_idx"):
                    self.model_tensors.pop(base + suf, None)
                losslessly_handled.append(gguf_name)
            else:
                # ---- V-REORDER REMNANT: replace with FP32 dequant lambda ----
                # AutoRound sym dumps lack .g_idx (group_size is constant), so we
                # cannot rely on upstream's gptq branch which KeyErrors on .g_idx.
                # Inline the dequant; modify_tensors then handles V-reorder + FP16.
                qw_gen = self.model_tensors[qw_name]
                sc_gen = self.model_tensors[scales_key]

                def make_dequant(qg=qw_gen, sg=sc_gen, gs=group_size):
                    def f():
                        qw = _materialize(qg())
                        sc = _materialize(sg())
                        return dequant_w4g128_sym(qw, sc, group_size=gs)
                    return f

                self.model_tensors[base + ".weight"] = make_dequant()
                for suf in (".qweight", ".qzeros", ".scales", ".g_idx"):
                    self.model_tensors.pop(base + suf, None)
                v_reorder_dequanted.append(base + ".weight")

        print(
            f"[Tool 1] Emitted {len(losslessly_handled)} lossless Q4_0 + "
            f"{len(v_reorder_dequanted)} V-reorder dequants. "
            f"Skipped {len(skipped_unmapped)} unmapped/incomplete.",
            file=sys.stderr,
        )
        if skipped_unmapped:
            print(f"[Tool 1] Skipped names: {skipped_unmapped[:5]}{'...' if len(skipped_unmapped) > 5 else ''}",
                  file=sys.stderr)

        # Don't call original_dequant: we've handled everything quantized.
        # Non-quant tensors (norms, embeds, lm_head, AutoRound's FP16 linear_attn
        # 1D params) remain in model_tensors and flow through the standard path.

    Model.dequant_model = patched_dequant_model


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-dir", type=Path, required=True,
                    help="Resolved snapshot dir of the AutoRound checkpoint")
    ap.add_argument("--outfile", type=Path, required=True)
    ap.add_argument("--llama-cpp", type=Path, default=Path("/home/llm/yarn-agentic/llama.cpp"),
                    help="Path to llama.cpp checkout (containing convert_hf_to_gguf.py)")
    ap.add_argument("--lazy", action="store_true",
                    help="Use lazy tensor loading (default eager for AutoRound dumps which fit in RAM)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Run convert_hf_to_gguf in dry-run mode")
    args = ap.parse_args()

    if not (args.model_dir / "config.json").exists():
        sys.exit(f"No config.json in {args.model_dir}")
    if not (args.llama_cpp / "convert_hf_to_gguf.py").exists():
        sys.exit(f"convert_hf_to_gguf.py not found at {args.llama_cpp}")

    sys.path.insert(0, str(args.llama_cpp))
    sys.path.insert(0, str(args.llama_cpp / "gguf-py"))

    spec = importlib.util.spec_from_file_location(
        "convert_hf_to_gguf",
        args.llama_cpp / "convert_hf_to_gguf.py",
    )
    convert = importlib.util.module_from_spec(spec)
    sys.modules["convert_hf_to_gguf"] = convert
    spec.loader.exec_module(convert)

    patch_convert(convert)

    saved_argv = sys.argv
    new_argv = [
        "convert_hf_to_gguf.py",
        str(args.model_dir),
        "--outfile", str(args.outfile),
        "--outtype", "f16",                                      # FP16 fallthrough; Q4_0 tensors bypass this
    ]
    if not args.lazy:
        new_argv.append("--no-lazy")
    if args.dry_run:
        new_argv.append("--dry-run")
    sys.argv = new_argv
    try:
        convert.main()
    finally:
        sys.argv = saved_argv


if __name__ == "__main__":
    main()
