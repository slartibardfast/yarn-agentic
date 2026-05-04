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

# V-row reorder candidates: V-channel row reorder permutes the OUT dim. Q4_0
# stores blocks along the IN dim, so a row permutation on the OUT dim is just
# rearranging top-level rows of the byte stream — Q4_0 codes copy 1:1 to the
# permuted out-positions, scales copy 1:1, AutoRound calibration is preserved.
# We handle these on the codes via _repack_with_v_row_perm rather than
# dequanting them, which preserves Intel's calibrated INT4 codes for these 96
# tensors (48 in_proj_qkv + 48 in_proj_z on Qwen3.6-27B).
V_ROW_PERM_QWEIGHT_SUFFIXES = (
    ".linear_attn.in_proj_qkv.qweight",
    ".linear_attn.in_proj_z.qweight",
)

# V-col reorder: permutes the IN dim in 16-element chunks. Q4_0's 32-block
# structure is broken by an in-dim permutation that crosses 16-element
# sub-block boundaries, so these still fall through to dequant_w4g128_sym +
# modify_tensors V-col-reorder + FP16 emit. (out_proj is the only tensor in
# this set that's actually quantized; in_proj_a/b are FP16 in the AutoRound
# dump so they don't go through .qweight.)
V_COL_PERM_QWEIGHT_SUFFIXES = (
    ".linear_attn.out_proj.qweight",
)

V_REORDER_QWEIGHT_SUFFIXES = V_ROW_PERM_QWEIGHT_SUFFIXES + V_COL_PERM_QWEIGHT_SUFFIXES


def is_lossless_q4_0_candidate(qweight_name: str) -> bool:
    """Trunk tensors that need no V-reorder at all — pure 1:1 lossless repack."""
    return not any(qweight_name.endswith(s) for s in V_REORDER_QWEIGHT_SUFFIXES)


def is_v_row_perm_candidate(qweight_name: str) -> bool:
    """V-row-reorder tensors — losslessly repacked to Q4_0 with permuted codes/scales."""
    return any(qweight_name.endswith(s) for s in V_ROW_PERM_QWEIGHT_SUFFIXES)


def is_v_col_perm_candidate(qweight_name: str) -> bool:
    """V-col-reorder tensors — losslessly repacked to Q4_0_AR16 with col-permuted codes/scales."""
    return any(qweight_name.endswith(s) for s in V_COL_PERM_QWEIGHT_SUFFIXES)


def _v_row_perm_indices(num_k_heads: int, num_v_heads: int, head_v_dim: int) -> np.ndarray:
    """
    Build the V-row reorder permutation as a flat 1-D index array for the V slice.
    Mirrors `_LinearAttentionVReorderBase._reorder_v_heads(v, dim=0, ...)`:
        original V layout: [G0_v0..v{r-1}, G1_v0..v{r-1}, ...] (grouped by K head)
        target V layout:   [G0_v0, G1_v0, ..., G0_v1, G1_v1, ...] (tiled, ggml broadcast)
    where r = num_v_per_k = num_v_heads // num_k_heads, each group is head_v_dim wide.

    Returns a (num_v_heads * head_v_dim,)-shape int64 ndarray giving the new
    out-row order in terms of original out-row indices.
    """
    num_v_per_k = num_v_heads // num_k_heads
    rows = num_v_heads * head_v_dim
    idx = np.arange(rows, dtype=np.int64).reshape(num_k_heads, num_v_per_k, head_v_dim)
    return idx.transpose(1, 0, 2).reshape(rows)


def repack_w4g128_with_v_row_perm(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    *,
    target_kind: str,             # "in_proj_qkv" or "in_proj_z"
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    group_size: int = 128,
) -> tuple[bytes, tuple[int, int]]:
    """
    Lossless V-row-reorder repack to Q4_0 bytes. Permutes codes + scales along
    the OUT dim; Q4_0 block structure (along IN dim) is untouched, so AutoRound
    calibration is preserved bit-for-bit.

    For `in_proj_qkv`: out is [Q | K | V] concatenation. Permute only the V slice.
    For `in_proj_z`:   out is purely V. Permute the entire out range.
    """
    codes_in_out = unpack_autogptq_int4(qweight)         # [in, out] uint8 in [0,15]
    in_f, out_f = codes_in_out.shape

    sc_groups_out = scales.detach().cpu().numpy().astype(np.float16)  # [n_groups, out]
    n_groups = in_f // group_size
    if sc_groups_out.shape != (n_groups, out_f):
        raise ValueError(
            f"scales shape {sc_groups_out.shape}, expected ({n_groups}, {out_f}) "
            f"for in_f={in_f} group_size={group_size}"
        )

    v_perm = _v_row_perm_indices(num_k_heads, num_v_heads, head_v_dim)   # int64 in [0, num_v_heads*head_v_dim)
    v_dim = num_v_heads * head_v_dim

    if target_kind == "in_proj_qkv":
        q_dim = head_k_dim * num_k_heads
        k_dim = head_k_dim * num_k_heads
        if out_f != q_dim + k_dim + v_dim:
            raise ValueError(
                f"in_proj_qkv out_f={out_f} != q_dim+k_dim+v_dim={q_dim+k_dim+v_dim} "
                f"(num_k_heads={num_k_heads} head_k_dim={head_k_dim} v_dim={v_dim})"
            )
        # Build full out-perm = [0..q_dim, q_dim..q_dim+k_dim, q_dim+k_dim+v_perm]
        full_perm = np.concatenate([
            np.arange(q_dim + k_dim, dtype=np.int64),
            (q_dim + k_dim) + v_perm,
        ])
    elif target_kind == "in_proj_z":
        if out_f != v_dim:
            raise ValueError(
                f"in_proj_z out_f={out_f} != v_dim={v_dim} "
                f"(num_v_heads={num_v_heads} head_v_dim={head_v_dim})"
            )
        full_perm = v_perm
    else:
        raise ValueError(f"unknown target_kind for V-row repack: {target_kind!r}")

    # Apply out-dim permutation. codes is [in, out] so axis=1; scales is [n_groups, out] so axis=1.
    codes_perm = codes_in_out[:, full_perm]
    sc_perm = sc_groups_out[:, full_perm]

    # Now repack to Q4_0 bytes — same structure as repack_w4g128_to_q4_0_bytes,
    # but using the already-permuted codes/scales. We can just re-use the
    # standard packer by reconstructing torch tensors with the right layout.
    # The packer expects qweight in AutoGPTQ's int32-packed form along in-dim.
    # Easier path: bypass the packer and emit Q4_0 bytes directly from codes+scales.
    blocks_per_grp = group_size // 32
    codes_out_in = codes_perm.T                              # [out, in]
    if in_f % 32 != 0:
        raise ValueError(f"in_features={in_f} not divisible by 32")
    n_blocks = in_f // 32

    cb = codes_out_in.reshape(out_f, n_blocks, 32)
    lo = cb[:, :, :16]
    hi = cb[:, :, 16:]
    qs = (lo | (hi.astype(np.uint8) << 4)).astype(np.uint8)

    sc_out_groups = sc_perm.T                                # [out, n_groups]
    sc_per_block = np.repeat(sc_out_groups, blocks_per_grp, axis=1)   # [out, n_blocks]

    packed = np.empty((out_f, n_blocks, 18), dtype=np.uint8)
    packed[:, :, 0:2] = sc_per_block.view(np.uint8).reshape(out_f, n_blocks, 2)
    packed[:, :, 2:18] = qs
    return packed.tobytes(), (out_f, in_f)


# ---------------------------------------------------------------------------
# V-col-perm repack: Q4_0_AR16 (16-element blocks) lossless on the V col
# permutation, which moves whole 16-element chunks of the IN dim. Mirrors
# upstream's _LinearAttentionVReorderBase._transform_nvfp4_weight col_perm
# computation (llama.cpp/convert_hf_to_gguf.py:5366-5371).
# ---------------------------------------------------------------------------


def _v_col_perm_indices(num_k_heads: int, num_v_heads: int, head_v_dim: int) -> np.ndarray:
    """
    Build the V-col reorder permutation as a flat 1-D index array along the IN dim.

    Mirrors `_LinearAttentionVReorderBase._reorder_v_heads(arange(...), dim=1, ...)`:
    starting from arange(num_v_heads * head_v_dim) viewed as
        [G0_v0 .. G0_v{r-1}, G1_v0 .. G1_v{r-1}, ...] (grouped, head_v_dim wide each)
    target order is
        [G0_v0, G1_v0, ..., G0_v1, G1_v1, ...] (tiled).

    Result is int64 of length `num_v_heads * head_v_dim`. The 16-aligned-chunk
    invariant is asserted: each consecutive 16-element slice of the result is
    of the form `[s, s+1, ..., s+15]` with `s % 16 == 0`. This is what makes
    Q4_0_AR16 (block_size = 16) lossless on the col permutation.
    """
    num_v_per_k = num_v_heads // num_k_heads
    n = num_v_heads * head_v_dim

    # _reorder_v_heads on a (1, n) tensor along dim=1: reshape last dim to
    # (num_k_heads, num_v_per_k, head_v_dim), swap the leading two axes.
    idx = np.arange(n, dtype=np.int64).reshape(num_k_heads, num_v_per_k, head_v_dim)
    col_perm = idx.transpose(1, 0, 2).reshape(n)

    # 16-aligned-chunk invariant. head_v_dim is typically 128 in Qwen3-Next,
    # so the group_starts are multiples of head_v_dim and within-chunk order
    # is +0..+15.
    if n % 16 != 0:
        raise ValueError(f"col_perm length {n} not a multiple of 16")
    chunks = col_perm.reshape(-1, 16)
    if not (chunks[:, 0] % 16 == 0).all():
        raise AssertionError("col_perm chunk starts not 16-aligned")
    expected = chunks[:, 0:1] + np.arange(16, dtype=np.int64)
    if not np.array_equal(chunks, expected):
        raise AssertionError("col_perm within-chunk order not contiguous +0..+15")

    return col_perm


def repack_w4g128_with_v_col_perm(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    *,
    num_k_heads: int,
    num_v_heads: int,
    head_v_dim: int,
    group_size: int = 128,
) -> tuple[bytes, tuple[int, int], np.ndarray, np.ndarray, np.ndarray]:
    """
    Lossless V-col-reorder repack to Q4_0_AR16 bytes. Permutes 16-element
    chunks of the IN dim of the AutoRound code tensor + replicates AutoRound's
    per-128 group scale across 8 child Q4_0_AR16 blocks (block_size = 16).

    Returns:
        (blob, (out_features, in_features), col_perm, codes_perm_in_out, scales_perm_per_block)
    where the auxiliary arrays are returned for the inline self-check and the
    bit-equivalence proof. blob layout: row-major over [out, n_blocks] of 10
    bytes each (2-byte fp16 d, 8-byte qs half-split: qs[j] low = code[j],
    high = code[j + 8]).
    """
    if group_size % 16 != 0 or group_size < 16:
        raise ValueError(f"group_size must be a multiple of 16, got {group_size}")
    n_blocks_per_group = group_size // 16  # 8 Q4_0_AR16 blocks per AutoRound group

    codes_in_out = unpack_autogptq_int4(qweight)               # [in, out] uint8 in [0,15]
    in_f, out_f = codes_in_out.shape
    if in_f % 16 != 0:
        raise ValueError(f"in_features={in_f} not divisible by 16 (Q4_0_AR16 block size)")
    if in_f % group_size != 0:
        raise ValueError(f"in_features={in_f} not divisible by group_size={group_size}")

    sc_groups_out = scales.detach().cpu().numpy().astype(np.float16)  # [n_groups, out]
    n_groups = in_f // group_size
    if sc_groups_out.shape != (n_groups, out_f):
        raise ValueError(
            f"scales shape {sc_groups_out.shape}, expected ({n_groups}, {out_f}) "
            f"for in_f={in_f} group_size={group_size}"
        )

    col_perm = _v_col_perm_indices(num_k_heads, num_v_heads, head_v_dim)
    if col_perm.shape[0] != in_f:
        raise ValueError(
            f"col_perm length {col_perm.shape[0]} != in_features {in_f} "
            f"(num_v_heads={num_v_heads} * head_v_dim={head_v_dim})"
        )

    # Apply col-perm along IN dim (axis 0 of the [in, out] codes tensor).
    codes_perm_in_out = codes_in_out[col_perm, :]                  # [in, out]

    # Per-Q4_0_AR16-block scale array: replicate each AutoRound group's scale
    # across its n_blocks_per_group child blocks (no rounding, byte-copy).
    n_blocks = in_f // 16
    scales_per_block_out = np.repeat(sc_groups_out, n_blocks_per_group, axis=0)  # [n_blocks, out]

    # Block-level permutation: each row of col_perm.reshape(-1, 16) is a
    # contiguous 16-chunk; the chunk's source block index is start // 16.
    block_perm = (col_perm.reshape(-1, 16)[:, 0] // 16).astype(np.int64)         # [n_blocks]
    scales_perm_per_block = scales_per_block_out[block_perm, :]                  # [n_blocks, out]

    # Byte-pack directly using AutoRound's calibrated codes/scales — do NOT
    # round-trip through Q4_0_AR16.quantize_blocks (which would re-quantize
    # from fp32 and lose calibration). INTERLEAVED nibble layout per Allium
    # spec, C kernel, CUDA kernel:
    #   qs[j] = code[2j] | (code[2j + 1] << 4),  j in [0, 8)
    # (NOT Q4_0's split-halves layout. Phase 2 caught a layout-disagreement
    # bug here that produced gibberish at runtime even though the
    # offline self-check passed self-consistently.)
    # Tensor data layout: blocks row-major over [out_features, n_blocks] of 10 bytes.
    codes_out_in = codes_perm_in_out.T                              # [out, in]
    cb = codes_out_in.reshape(out_f, n_blocks, 8, 2)                # last axis: (even, odd)
    lo = cb[:, :, :, 0]                                             # codes 0,2,4,...,14 -> low nibble
    hi = cb[:, :, :, 1]                                             # codes 1,3,5,...,15 -> high nibble
    qs = (lo | (hi.astype(np.uint8) << 4)).astype(np.uint8)         # [out, n_blocks, 8]

    # scales_perm_per_block is [n_blocks, out]; we want [out, n_blocks] for emit.
    # .T is a non-contiguous view; need a contiguous copy before the uint8 reinterpret.
    sc_out_blocks = np.ascontiguousarray(scales_perm_per_block.T)   # [out, n_blocks] fp16
    packed = np.empty((out_f, n_blocks, 10), dtype=np.uint8)
    packed[:, :, 0:2] = sc_out_blocks.view(np.uint8).reshape(out_f, n_blocks, 2)
    packed[:, :, 2:10] = qs

    return packed.tobytes(), (out_f, in_f), col_perm, codes_perm_in_out, scales_perm_per_block


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


def _self_check_block0_q4_0_ar16(
    blob: bytes,
    codes_perm_in_out: np.ndarray,
    scales_perm_per_block: np.ndarray,
):
    """
    First Q4_0_AR16 block of out=0 must satisfy:
      bytes 0..2:  fp16 d == scales_perm_per_block[0, 0]
      bytes 2..10: 8 nibble-pairs encoding codes_perm_in_out[:16, 0]
                   under INTERLEAVED layout (qs[j] = c[2j] | (c[2j+1] << 4)).
    """
    if len(blob) < 10:
        raise AssertionError(f"[Tool 1 self-check] blob shorter than one Q4_0_AR16 block ({len(blob)} bytes)")
    block0 = np.frombuffer(blob[:10], dtype=np.uint8)

    scale_fp16 = np.frombuffer(block0[:2].tobytes(), dtype=np.float16)[0]
    expected_scale = np.float16(scales_perm_per_block[0, 0])
    if scale_fp16 != expected_scale:
        raise AssertionError(
            f"[Tool 1 self-check Q4_0_AR16] block-0 scale mismatch: "
            f"got {scale_fp16}, expected {expected_scale}"
        )

    qs = block0[2:10]
    lo = qs & 0x0F
    hi = (qs >> 4) & 0x0F
    # Interleaved: byte j -> code[2j] (lo) + code[2j+1] (hi).
    decoded = np.empty(16, dtype=np.uint8)
    decoded[0::2] = lo
    decoded[1::2] = hi

    expected = codes_perm_in_out[:16, 0].astype(np.uint8)
    if not np.array_equal(decoded, expected):
        raise AssertionError(
            f"[Tool 1 self-check Q4_0_AR16] block-0 codes mismatch:\n"
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

        # Hparams may be at the top level (text-only Qwen3.5/3.6) or nested under
        # text_config (VL multimodal variant). Look in both.
        def _hparam(key, default=None):
            v = self.hparams.get(key)
            if v is not None:
                return v
            tc = self.hparams.get("text_config")
            if isinstance(tc, dict):
                return tc.get(key, default)
            return default

        num_hidden_layers = _hparam("num_hidden_layers")

        # Linear-attention V-row-perm needs head topology; only required if any
        # in_proj_qkv / in_proj_z .qweight is present in this dump.
        num_k_heads = _hparam("linear_num_key_heads")
        num_v_heads = _hparam("linear_num_value_heads")
        head_k_dim  = _hparam("linear_key_head_dim")
        head_v_dim  = _hparam("linear_value_head_dim")

        losslessly_handled: list[str] = []
        v_row_perm_handled: list[str] = []
        v_col_perm_handled: list[str] = []
        v_reorder_dequanted: list[str] = []
        skipped_unmapped: list[str] = []
        self_checked = False
        v_row_self_checked = False
        v_col_self_checked = False

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
            elif is_v_row_perm_candidate(qw_name):
                # ---- V-ROW-PERM PATH: lossless Q4_0 with permuted codes/scales ----
                # The V-row reorder permutes the OUT dim. Q4_0 stores blocks along
                # the IN dim, so we permute codes/scales along OUT and re-pack to
                # Q4_0 bytes — AutoRound's calibration is preserved bit-for-bit.
                if any(v is None for v in (num_k_heads, num_v_heads, head_k_dim, head_v_dim)):
                    raise RuntimeError(
                        f"V-row-perm tensor {qw_name!r} requires linear_num_key_heads / "
                        f"linear_num_value_heads / linear_key_head_dim / linear_value_head_dim "
                        f"in hparams (or text_config); got "
                        f"k={num_k_heads} v={num_v_heads} hk={head_k_dim} hv={head_v_dim}"
                    )

                map_input = _strip_lm_prefix(base)
                try:
                    gguf_name = self.map_tensor_name(map_input + ".weight")
                except (ValueError, KeyError):
                    skipped_unmapped.append(qw_name)
                    continue

                target_kind = (
                    "in_proj_qkv" if qw_name.endswith(".linear_attn.in_proj_qkv.qweight")
                    else "in_proj_z"
                )
                qweight_t = _materialize(self.model_tensors[qw_name]())
                scales_t = _materialize(self.model_tensors[scales_key]())

                blob, (out_f, in_f) = repack_w4g128_with_v_row_perm(
                    qweight_t, scales_t,
                    target_kind=target_kind,
                    num_k_heads=num_k_heads, num_v_heads=num_v_heads,
                    head_k_dim=head_k_dim, head_v_dim=head_v_dim,
                    group_size=group_size,
                )

                if not v_row_self_checked:
                    # Block-0 of the permuted output: row 0 (= permuted out-row[0]) at
                    # in-positions 0..31. For in_proj_qkv that's a Q-row (perm passthrough),
                    # so block-0 codes/scale match the original codes[0:32, 0] and
                    # scales[0, 0] — same as the trunk self_check. For in_proj_z row 0
                    # is V[v_perm[0]], which is V[0] (since v_perm starts at 0 in tiled
                    # layout because the first K-group's first V-head maps to original
                    # index 0). So block-0 still maps to original codes[0:32, 0].
                    _self_check_block0(qweight_t, scales_t, blob)
                    print(
                        f"[Tool 1] Self-check passed on first V-row-perm emit "
                        f"({gguf_name}, kind={target_kind})",
                        file=sys.stderr,
                    )
                    v_row_self_checked = True

                bytes_per_row = (in_f // 32) * 18
                arr = np.frombuffer(blob, dtype=np.uint8).reshape(out_f, bytes_per_row)
                self.gguf_writer.add_tensor(
                    gguf_name,
                    arr,
                    raw_dtype=gguf.GGMLQuantizationType.Q4_0,
                )

                for suf in (".qweight", ".qzeros", ".scales", ".g_idx"):
                    self.model_tensors.pop(base + suf, None)
                v_row_perm_handled.append(gguf_name)
            elif is_v_col_perm_candidate(qw_name):
                # ---- V-COL-PERM PATH: lossless Q4_0_AR16 with col-permuted codes/scales ----
                # The V-col reorder permutes the IN dim in 16-element-aligned chunks.
                # Q4_0_AR16's block_size = 16 makes this lossless: we permute whole
                # blocks (codes + replicated AutoRound scale) and emit raw_dtype Q4_0_AR16.
                if any(v is None for v in (num_k_heads, num_v_heads, head_v_dim)):
                    raise RuntimeError(
                        f"V-col-perm tensor {qw_name!r} requires linear_num_key_heads / "
                        f"linear_num_value_heads / linear_value_head_dim in hparams "
                        f"(or text_config); got k={num_k_heads} v={num_v_heads} hv={head_v_dim}"
                    )

                map_input = _strip_lm_prefix(base)
                try:
                    gguf_name = self.map_tensor_name(map_input + ".weight")
                except (ValueError, KeyError):
                    skipped_unmapped.append(qw_name)
                    continue

                qweight_t = _materialize(self.model_tensors[qw_name]())
                scales_t = _materialize(self.model_tensors[scales_key]())

                blob, (out_f, in_f), col_perm, codes_perm_in_out, scales_perm_per_block = (
                    repack_w4g128_with_v_col_perm(
                        qweight_t, scales_t,
                        num_k_heads=num_k_heads, num_v_heads=num_v_heads,
                        head_v_dim=head_v_dim,
                        group_size=group_size,
                    )
                )

                if not v_col_self_checked:
                    _self_check_block0_q4_0_ar16(blob, codes_perm_in_out, scales_perm_per_block)
                    print(
                        f"[Tool 1] Self-check passed on first V-col-perm emit "
                        f"({gguf_name}, kind=out_proj)",
                        file=sys.stderr,
                    )
                    v_col_self_checked = True

                n_blocks_per_row = in_f // 16
                bytes_per_row = n_blocks_per_row * 10
                arr = np.frombuffer(blob, dtype=np.uint8).reshape(out_f, bytes_per_row)
                self.gguf_writer.add_tensor(
                    gguf_name,
                    arr,
                    raw_dtype=gguf.GGMLQuantizationType.Q4_0_AR16,
                )

                for suf in (".qweight", ".qzeros", ".scales", ".g_idx"):
                    self.model_tensors.pop(base + suf, None)
                v_col_perm_handled.append(gguf_name)
            else:
                # ---- V-COL-REORDER REMNANT (defensive fallback): dequant→FP32→FP16 emit ----
                # Col-perm crosses Q4_0 32-block boundaries; cannot do losslessly
                # on codes. Replace with FP32 dequant lambda; modify_tensors handles
                # the V-col reorder + FP16 emit.
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

        summary_parts = [
            f"{len(losslessly_handled)} lossless Q4_0 trunk",
            f"{len(v_row_perm_handled)} V-row-perm Q4_0",
            f"{len(v_col_perm_handled)} V-col-perm Q4_0_AR16",
        ]
        if v_reorder_dequanted:
            summary_parts.append(f"{len(v_reorder_dequanted)} V-col-reorder dequants (fallback)")
        print(
            f"[Tool 1] Emitted "
            + " + ".join(summary_parts)
            + f". Skipped {len(skipped_unmapped)} unmapped/incomplete.",
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
    ap.add_argument("--ik-llama-cpp", type=Path,
                    default=Path("/home/llm/yarn-agentic/ik_llama.cpp"),
                    help="Path to ik_llama.cpp checkout. Its gguf-py provides the "
                         "Q4_0_AR16 quant type (id=159) needed for V-col-perm emit.")
    ap.add_argument("--lazy", action="store_true",
                    help="Use lazy tensor loading (default eager for AutoRound dumps which fit in RAM)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Run convert_hf_to_gguf in dry-run mode")
    ap.add_argument("--outtype", default="f16",
                    help="Output dtype for non-quantized passthrough tensors (f16/bf16/f32/auto). "
                         "Use bf16 to convert a non-AutoRound HF dump straight to BF16 GGUF — the "
                         "Qwen3_5TextModel VL prefix-strip + vision-skip patches still apply, the "
                         "quant_method='auto-round' check in the dequant_model patch falls through "
                         "to upstream when the dump has no AutoRound config.")
    args = ap.parse_args()

    if not (args.model_dir / "config.json").exists():
        sys.exit(f"No config.json in {args.model_dir}")
    if not (args.llama_cpp / "convert_hf_to_gguf.py").exists():
        sys.exit(f"convert_hf_to_gguf.py not found at {args.llama_cpp}")

    # Use upstream llama.cpp's gguf-py for the converter (it has MODEL_ARCH.MMPROJ
    # and other constants that ik_llama.cpp's gguf-py lacks). After the converter
    # imports gguf, we splice Q4_0_AR16 (id=159) and its (16, 10) size entry from
    # ik_llama.cpp's gguf-py into the in-memory enum + size table so the GGUF
    # writer accepts the new raw_dtype.
    ik_gguf_py = args.ik_llama_cpp / "gguf-py"
    if not (ik_gguf_py / "gguf" / "constants.py").exists():
        sys.exit(f"ik_llama.cpp gguf-py not found at {ik_gguf_py}")
    sys.path.insert(0, str(args.llama_cpp / "gguf-py"))
    sys.path.insert(0, str(args.llama_cpp))

    import gguf as _gguf_mod
    if not hasattr(_gguf_mod.GGMLQuantizationType, "Q4_0_AR16"):
        # IntEnum is sealed by EnumType; fall back to manual injection. Need to
        # populate (a) the internal lookup tables, (b) class attribute via
        # type.__setattr__ (EnumType.__setattr__ blocks adding members). On
        # Python 3.11+ both _member_map_/_value2member_map_ and the actual
        # class-level attribute must be set for hasattr / getattr to work.
        EnumCls = _gguf_mod.GGMLQuantizationType
        new_member = int.__new__(EnumCls, 159)
        new_member._name_ = "Q4_0_AR16"
        new_member._value_ = 159
        EnumCls._member_map_["Q4_0_AR16"] = new_member
        EnumCls._value2member_map_[159] = new_member
        EnumCls._member_names_.append("Q4_0_AR16")
        type.__setattr__(EnumCls, "Q4_0_AR16", new_member)
        # Size table: 16 elements per block, 10 bytes per block.
        _gguf_mod.constants.GGML_QUANT_SIZES[EnumCls.Q4_0_AR16] = (16, 10)
    if not hasattr(_gguf_mod.GGMLQuantizationType, "Q4_0_AR16"):
        sys.exit("Failed to register Q4_0_AR16 on upstream gguf module.")

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
        "--outtype", args.outtype,                               # default f16; FP16 fallthrough for AutoRound non-quant tensors. Q4_0 emits bypass this regardless.
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
