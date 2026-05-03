#!/usr/bin/env python3
"""Tool 3 — Per-tensor absmax-aware BF16 → FP16 selective recast.

Usage:
    recast_bf16_to_fp16.py \
        --input <src.gguf> \
        --output <dst.gguf> \
        --policy <policy.yaml> \
        --tier {dry-run,T1,T2,T3,T4,T5} \
        [--absmax-tsv <out.tsv>]

Tier semantics (Band C = absmax > FP16_MAX):
    dry-run : measure only; emit TSV; write no GGUF.
    T1      : Band-C tensors stay BF16. No kernel work. (default ship).
    T2-T5   : escalation tiers — see PHASE32-MTP-FP16-CANARY.md.

Bands:
    A : absmax <= 32768.0  → clean RNE cast, no Inf risk.
    B : 32768.0 < absmax <= 65504.0 → cast with edge-precision warning.
    C : absmax > 65504.0   → tier-dispatch.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass

# Use the in-tree gguf-py from llama.cpp.
sys.path.insert(0, "/home/llm/yarn-agentic/llama.cpp/gguf-py")

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter

FP16_MAX = 65504.0
FP16_HALF_RANGE = 32768.0


@dataclass
class Policy:
    """Per-tensor cast policy, regex-driven.

    A tensor name is checked against `preserve_bf16` first; on match it is
    forced to stay BF16 regardless of measured absmax. Otherwise it follows
    the tier-dispatch (cast to FP16 if Band A/B; tier-specific in Band C).
    """

    preserve_bf16: list[re.Pattern]
    cast_fp16: list[re.Pattern]

    @classmethod
    def from_yaml(cls, path: str) -> "Policy":
        try:
            import yaml
        except ImportError:
            data = json.loads(open(path).read())
        else:
            data = yaml.safe_load(open(path))
        return cls(
            preserve_bf16=[re.compile(p) for p in data.get("preserve_bf16", [])],
            cast_fp16=[re.compile(p) for p in data.get("cast_fp16", [".*"])],
        )

    def matches_preserve(self, name: str) -> bool:
        return any(p.search(name) for p in self.preserve_bf16)

    def matches_cast(self, name: str) -> bool:
        return any(p.search(name) for p in self.cast_fp16)


def bf16_to_fp32(arr: np.ndarray, logical_shape: list[int]) -> np.ndarray:
    """Lossless BF16 → FP32 via bit-shift (BF16 is FP32 high half).

    gguf-py exposes BF16 tensors as uint8 raw bytes with the last dim doubled
    (one byte per BF16 element pair). We re-view as uint16, shift left 16,
    and re-view as float32. Final shape matches `logical_shape` (GGUF stores
    shape in column-major order; numpy is row-major, so reshape to reversed).
    """
    if arr.dtype == np.float32:
        return arr
    if arr.dtype == np.float16:
        return arr.astype(np.float32)
    # BF16 raw layout from GGUFReader: uint8, last-dim doubled.
    if arr.dtype == np.uint8:
        u16 = arr.view(np.uint16)
        u32 = u16.astype(np.uint32) << 16
        fp32 = u32.view(np.float32)
        # Reshape to row-major version of logical shape (GGUF shape is reversed).
        target = list(reversed(logical_shape))
        return fp32.reshape(target)
    # BF16 as uint16 view (in case caller passed it that way).
    if arr.dtype == np.uint16:
        u = arr.astype(np.uint32) << 16
        return u.view(np.float32)
    # ml_dtypes.bfloat16 fallback
    return arr.astype(np.float32)


def classify_band(absmax: float) -> str:
    if absmax <= FP16_HALF_RANGE:
        return "A"
    if absmax <= FP16_MAX:
        return "B"
    return "C"


@dataclass
class TensorAction:
    out_dtype: GGMLQuantizationType  # "F16" / "BF16" target
    out_arr: np.ndarray  # data to emit
    band: str
    absmax: float
    note: str  # human-readable disposition
    per_tensor_scale: float = 1.0  # T2/T3: stored as scalar
    per_channel_scales: np.ndarray | None = None  # T4: shape (out_features,) FP32
    hadamard_d: int = 0  # T5: Hadamard size; 0 = no rotation


def _build_hadamard(d: int) -> np.ndarray:
    """Build a normalised Hadamard matrix of size d (must be power of 2).

    Walsh-Hadamard via Sylvester construction: H_2 = [[1,1],[1,-1]];
    H_{2k} = [[H_k, H_k], [H_k, -H_k]]. Normalised by 1/sqrt(d) so H @ H.T = I.
    """
    if d & (d - 1) != 0:
        raise ValueError(f"Hadamard size must be power of 2, got {d}")
    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < d:
        H = np.block([[H, H], [H, -H]])
    return (H / np.sqrt(d)).astype(np.float32)


def _walsh_hadamard_rows(x: np.ndarray) -> np.ndarray:
    """Apply normalised fast Walsh-Hadamard to each row of x in-place-equivalent.

    x: shape (rows, d) where d is power of 2.
    Returns x @ H_d (with H_d normalised by 1/sqrt(d)).
    Uses the iterative Sylvester butterfly — O(rows * d * log d).
    """
    rows, d = x.shape
    if d & (d - 1) != 0:
        raise ValueError(f"WHT requires d power-of-2, got {d}")
    out = x.astype(np.float32, copy=True)
    h = 1
    while h < d:
        # Butterfly: pairs separated by h
        for i in range(0, d, h * 2):
            a = out[:, i : i + h].copy()
            b = out[:, i + h : i + 2 * h]
            out[:, i : i + h] = a + b
            out[:, i + h : i + 2 * h] = a - b
        h *= 2
    return out / np.sqrt(d).astype(np.float32)


def _hadamard_size_for(in_dim: int) -> int:
    """Return largest power-of-2 ≤ in_dim. T5 rotates only the leading-pow2 prefix
    of the in_features axis if in_dim is not a clean power of 2 (we then pad
    by zeros — common for FFN sizes like 3584 = 2048 + 1024 + 512 doesn't pad
    cleanly; in practice we use d = next_pow2(in_dim) and pad)."""
    d = 1
    while d * 2 <= in_dim:
        d *= 2
    return d


def recast_tensor(
    name: str,
    src_arr: np.ndarray,
    src_type: GGMLQuantizationType,
    logical_shape: list[int],
    policy: Policy,
    tier: str,
    force_rescale: bool = False,
) -> TensorAction:
    """Decide what dtype this tensor lands at and produce the array.

    `force_rescale` (used for T2-T4 proof-out on models with no Band C):
    when True, *always* compute and apply a non-trivial scale (or Hadamard
    transform) to non-preserved BF16 tensors regardless of their absmax.
    The scale chosen is `absmax / 30000.0` (compresses the value range
    into [−30000, 30000], which is comfortably inside FP16 dynamic range).
    The runtime then multiplies by the recorded scale to recover original
    magnitude. End-to-end output is bit-identical at full precision; FP16
    rounding may introduce ULP-level deviation. This exercises the full
    loader/kernel path on small models.
    """
    # Pass-through for non-BF16 sources (already-quant trunk, F32 norms, etc.).
    if src_type != GGMLQuantizationType.BF16:
        return TensorAction(src_type, src_arr, band="-", absmax=0.0, note="passthrough")

    # BF16 source. Apply policy.
    if policy.matches_preserve(name):
        return TensorAction(
            GGMLQuantizationType.BF16, src_arr, band="-", absmax=0.0, note="policy-preserve"
        )

    fp32 = bf16_to_fp32(src_arr, logical_shape)
    absmax = float(np.abs(fp32).max()) if fp32.size else 0.0

    if absmax == 0.0:
        out = fp32.astype(np.float16)
        return TensorAction(GGMLQuantizationType.F16, out, "A", 0.0, "zero-tensor")

    band = classify_band(absmax)

    # --- Tier dispatch ---

    if tier == "dry-run":
        if band == "A" or band == "B":
            return TensorAction(GGMLQuantizationType.F16, fp32.astype(np.float16),
                                band, absmax, f"dry-run band {band}")
        return TensorAction(GGMLQuantizationType.BF16, src_arr, "C", absmax, "dry-run band C")

    if tier == "T1":
        if band in ("A", "B"):
            out = fp32.astype(np.float16)
            return TensorAction(GGMLQuantizationType.F16, out, band, absmax, f"T1 RNE-cast (band {band})")
        # Band C → keep BF16
        return TensorAction(GGMLQuantizationType.BF16, src_arr, "C", absmax, "T1 BF16 fallback")

    if tier in ("T2", "T3"):
        # Per-tensor scale. In Band C we MUST rescale; otherwise scale=1.0
        # unless force_rescale is on (proof-out path).
        if band == "C" or force_rescale:
            scale = absmax / 30000.0
        else:
            scale = 1.0
        rescaled = fp32 / scale if scale != 1.0 else fp32
        out = rescaled.astype(np.float16)
        if np.isinf(out).any():
            raise ValueError(f"{name}: tier {tier} produced Inf after rescale by {scale}")
        return TensorAction(
            GGMLQuantizationType.F16, out, band, absmax,
            f"{tier} per-tensor /{scale:.6g}" if scale != 1.0 else f"{tier} no-rescale",
            per_tensor_scale=scale,
        )

    if tier == "T4":
        # Per-channel scale. Operate per-row of W (axis 0 = output channels).
        # With force_rescale on, every row gets a scale; otherwise only rows
        # that overflow get a scale.
        if fp32.ndim != 2:
            # fall back to T2 for 1D tensors
            if band == "C" or force_rescale:
                scale = absmax / 30000.0
            else:
                scale = 1.0
            rescaled = fp32 / scale if scale != 1.0 else fp32
            out = rescaled.astype(np.float16)
            return TensorAction(
                GGMLQuantizationType.F16, out, band, absmax,
                f"T4 falls-back-T2 /{scale:.6g}",
                per_tensor_scale=scale,
            )
        row_absmax = np.abs(fp32).max(axis=1).astype(np.float32)
        if force_rescale:
            row_scales = np.maximum(row_absmax / 30000.0, np.float32(1e-30))
        else:
            row_scales = np.where(row_absmax > FP16_MAX, row_absmax / 30000.0, 1.0).astype(np.float32)
        rescaled = fp32 / row_scales[:, None]
        out = rescaled.astype(np.float16)
        if np.isinf(out).any():
            raise ValueError(f"{name}: T4 produced Inf after per-row rescale")
        n_scaled = int((row_scales != 1.0).sum())
        return TensorAction(
            GGMLQuantizationType.F16, out, band, absmax,
            f"T4 per-channel ({n_scaled}/{len(row_scales)} rows scaled)",
            per_channel_scales=row_scales,
        )

    if tier == "T5":
        # Hadamard rotation on the input-channel axis (axis 1 for a 2D weight
        # matrix shaped (out, in)). After rotation we re-measure absmax and
        # apply per-tensor scale to fit FP16 (proof-of-concept; Quarot/SpinQuant
        # use carefully-chosen H to bound post-rotation absmax).
        if fp32.ndim != 2:
            # Fall back to T2 for 1D tensors (no rotation possible).
            if band == "C" or force_rescale:
                scale = absmax / 30000.0
            else:
                scale = 1.0
            rescaled = fp32 / scale if scale != 1.0 else fp32
            return TensorAction(
                GGMLQuantizationType.F16, rescaled.astype(np.float16), band, absmax,
                f"T5 falls-back-T2 /{scale:.6g}",
                per_tensor_scale=scale,
            )
        in_dim = fp32.shape[1]
        d = _hadamard_size_for(in_dim)
        if d == in_dim:
            # Fast butterfly transform across the whole row
            rotated = _walsh_hadamard_rows(fp32)
        else:
            # Pad with zeros to next pow2; we'll need to track padding for runtime
            # un-rotation. For 0.8B all in_dims are pow2 (1024, 3584=non-pow2!).
            # Conservative path: rotate only the leading-d slice and leave the
            # tail unchanged. Runtime un-rotation must mirror this.
            rotated = fp32.copy()
            rotated[:, :d] = _walsh_hadamard_rows(fp32[:, :d])
        # Recompute absmax post-rotation (Hadamard normalised by 1/sqrt(d) so
        # absmax can rise by sqrt(d) in the worst case but typically much less).
        post_absmax = float(np.abs(rotated).max())
        if post_absmax > FP16_HALF_RANGE or force_rescale:
            scale = max(post_absmax / 30000.0, 1e-30)
        else:
            scale = 1.0
        rescaled = rotated / scale if scale != 1.0 else rotated
        out = rescaled.astype(np.float16)
        if np.isinf(out).any():
            raise ValueError(f"{name}: T5 produced Inf after rotation+scale")
        return TensorAction(
            GGMLQuantizationType.F16, out, band, absmax,
            f"T5 hadamard d={d}, post_absmax={post_absmax:.6g}, /{scale:.6g}",
            per_tensor_scale=scale,
            hadamard_d=d,
        )

    raise ValueError(f"unknown tier {tier!r}")


def copy_kvs(reader: GGUFReader, writer: GGUFWriter) -> None:
    """Copy all KV pairs from reader into writer except built-in / writer-added ones."""
    BUILTIN_KEYS = {
        "GGUF.version",
        "GGUF.tensor_count",
        "GGUF.kv_count",
        # Already added by GGUFWriter(arch=...).
        "general.architecture",
    }
    for f in reader.fields.values():
        if f.name in BUILTIN_KEYS:
            continue
        types = list(f.types)
        # A field has one or more "type tags" describing its layout. The
        # canonical decoding is in gguf-py/gguf/gguf_reader.py — we mirror
        # the relevant cases here.
        head = types[0]
        if head == GGUFValueType.STRING:
            val = bytes(f.parts[-1]).decode("utf-8", errors="replace")
            writer.add_string(f.name, val)
        elif head == GGUFValueType.ARRAY:
            if len(types) < 2:
                continue
            elem_t = types[1]
            data = f.contents()
            if elem_t == GGUFValueType.STRING:
                writer.add_array(f.name, list(data))
            else:
                # Numeric arrays — pass through as a Python list so writer
                # repacks them correctly.
                writer.add_array(f.name, list(data))
        elif head == GGUFValueType.BOOL:
            writer.add_bool(f.name, bool(f.parts[-1][0]))
        elif head == GGUFValueType.UINT8:
            writer.add_uint8(f.name, int(f.parts[-1][0]))
        elif head == GGUFValueType.INT8:
            writer.add_int8(f.name, int(f.parts[-1][0]))
        elif head == GGUFValueType.UINT16:
            writer.add_uint16(f.name, int(f.parts[-1][0]))
        elif head == GGUFValueType.INT16:
            writer.add_int16(f.name, int(f.parts[-1][0]))
        elif head == GGUFValueType.UINT32:
            writer.add_uint32(f.name, int(f.parts[-1][0]))
        elif head == GGUFValueType.INT32:
            writer.add_int32(f.name, int(f.parts[-1][0]))
        elif head == GGUFValueType.UINT64:
            writer.add_uint64(f.name, int(f.parts[-1][0]))
        elif head == GGUFValueType.INT64:
            writer.add_int64(f.name, int(f.parts[-1][0]))
        elif head == GGUFValueType.FLOAT32:
            writer.add_float32(f.name, float(f.parts[-1][0]))
        elif head == GGUFValueType.FLOAT64:
            writer.add_float64(f.name, float(f.parts[-1][0]))
        else:
            raise ValueError(f"unhandled KV type {head} for {f.name}")


def write_tsv(path: str, rows: list[tuple]) -> None:
    """Per-tensor TSV: name, src_type, out_type, band, absmax, n_elem, note."""
    with open(path, "w") as f:
        f.write("name\tsrc_type\tout_type\tband\tabsmax\tn_elem\tnote\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="source GGUF")
    ap.add_argument("--output", help="destination GGUF (omitted ⇒ dry-run)")
    ap.add_argument("--policy", required=True, help="policy YAML/JSON")
    ap.add_argument(
        "--tier",
        required=True,
        choices=["dry-run", "T1", "T2", "T3", "T4", "T5"],
        help="cast-method tier",
    )
    ap.add_argument("--absmax-tsv", help="dump per-tensor absmax/band/disposition TSV here")
    ap.add_argument(
        "--force-rescale",
        action="store_true",
        help="(T2-T5 proof-out) apply non-trivial scale/rotation to every cast tensor "
             "even if absmax is in Band A/B — exercises loader/kernel paths on "
             "models with no real Band C. Required to validate the runtime hooks.",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.tier != "dry-run" and not args.output:
        ap.error("--output is required unless --tier dry-run")

    policy = Policy.from_yaml(args.policy)
    reader = GGUFReader(args.input, "r")

    # Parse arch from KVs so the writer can be initialised.
    arch_field = reader.fields.get("general.architecture")
    if arch_field is None:
        sys.exit("source GGUF lacks general.architecture")
    arch = bytes(arch_field.parts[-1]).decode()

    writer = None
    if args.tier != "dry-run":
        writer = GGUFWriter(args.output, arch=arch)
        copy_kvs(reader, writer)

    # First pass: classify + collect actions; emit per-tensor log.
    tsv_rows: list[tuple] = []
    actions: list[tuple[str, TensorAction]] = []
    band_count = {"A": 0, "B": 0, "C": 0, "-": 0}

    for t in reader.tensors:
        try:
            src_type = GGMLQuantizationType(t.tensor_type)
        except ValueError:
            sys.exit(f"unknown tensor type {t.tensor_type} for {t.name}")
        logical_shape = [int(d) for d in t.shape]
        action = recast_tensor(t.name, t.data, src_type, logical_shape, policy, args.tier,
                               force_rescale=args.force_rescale)
        actions.append((t.name, action))
        band_count[action.band] += 1
        if args.verbose or action.band == "C":
            print(
                f"  [{action.band}] {t.name:<60} {src_type.name:>5} → {action.out_dtype.name:<5}"
                f"  absmax={action.absmax:.6g}  ({action.note})",
                file=sys.stderr,
            )
        tsv_rows.append(
            (
                t.name,
                src_type.name,
                action.out_dtype.name,
                action.band,
                f"{action.absmax:.6g}",
                int(np.prod(t.shape)),
                action.note,
            )
        )

    if args.absmax_tsv:
        write_tsv(args.absmax_tsv, tsv_rows)
        print(f"absmax tsv → {args.absmax_tsv}", file=sys.stderr)

    # Summary
    n_total = sum(band_count.values())
    print(
        f"summary: A={band_count['A']}  B={band_count['B']}  C={band_count['C']}  "
        f"passthrough/preserved={band_count['-']}  total={n_total}",
        file=sys.stderr,
    )
    n_C = band_count["C"]
    if n_C and args.tier == "T1":
        n_BF16_relevant = sum(1 for _, a in actions if a.note != "passthrough")
        if n_BF16_relevant > 0:
            pct_C = 100.0 * n_C / n_BF16_relevant
            print(
                f"T1 Band-C tensors: {n_C}/{n_BF16_relevant} BF16-eligible "
                f"({pct_C:.1f}% fall back to BF16). PHASE32 gate: ≤ 5% to ship T1.",
                file=sys.stderr,
            )

    if args.tier == "dry-run":
        return 0

    # --- Tier metadata: emit before tensor adds ---
    #
    # KV keys recognised by the ik_llama loader patch:
    #   recast.tier              : str — one of {T1,T2,T3,T4,T5}
    #   recast.force_rescale     : bool — set by --force-rescale; informational
    #   recast.scales.names      : str[]  — tensor names with per-tensor scale (T2/T3/T5)
    #   recast.scales.values     : f32[]  — corresponding scale values
    #   recast.hadamard.names    : str[]  — tensor names rotated (T5)
    #   recast.hadamard.values   : u32[]  — corresponding Hadamard sizes
    # Per-channel scales (T4) are emitted as sibling tensors "<name>.recast_scale".
    writer.add_string("recast.tier", args.tier)
    if args.force_rescale:
        writer.add_bool("recast.force_rescale", True)

    per_tensor_scales: dict[str, float] = {}
    hadamard_sizes: dict[str, int] = {}
    pc_names: list[str] = []
    pc_lengths: list[int] = []
    pc_values: list[float] = []
    for name, action in actions:
        if action.per_tensor_scale != 1.0:
            per_tensor_scales[name] = float(action.per_tensor_scale)
        if action.hadamard_d:
            hadamard_sizes[name] = int(action.hadamard_d)
        if action.per_channel_scales is not None:
            pc_names.append(name)
            pc_lengths.append(int(action.per_channel_scales.size))
            pc_values.extend(float(v) for v in action.per_channel_scales)
    if per_tensor_scales:
        writer.add_array("recast.scales.names", list(per_tensor_scales.keys()))
        writer.add_array("recast.scales.values", [float(v) for v in per_tensor_scales.values()])
        print(f"  recast.scales: {len(per_tensor_scales)} tensors with scale != 1.0", file=sys.stderr)
    if hadamard_sizes:
        writer.add_array("recast.hadamard.names", list(hadamard_sizes.keys()))
        writer.add_array("recast.hadamard.values", [int(v) for v in hadamard_sizes.values()])
        print(f"  recast.hadamard: {len(hadamard_sizes)} tensors rotated", file=sys.stderr)
    if pc_names:
        # Flat layout: names[i] has pc_values[offset[i]:offset[i]+lengths[i]]
        # offset[i] = sum(lengths[0..i-1])
        writer.add_array("recast.per_channel.names", pc_names)
        writer.add_array("recast.per_channel.lengths", [int(v) for v in pc_lengths])
        writer.add_array("recast.per_channel.values", [float(v) for v in pc_values])
        print(f"  recast.per_channel: {len(pc_names)} tensors, {len(pc_values)} total scale entries", file=sys.stderr)

    # Second pass: write tensors. Use add_tensor with raw_dtype.
    for name, action in actions:
        out_arr = action.out_arr
        if action.out_dtype == GGMLQuantizationType.BF16:
            # Pass-through: keep raw uint8 bytes from the reader.
            if out_arr.dtype != np.uint8:
                raise RuntimeError(
                    f"{name}: BF16 out has dtype {out_arr.dtype}, expected uint8 raw bytes"
                )
            writer.add_tensor(name, out_arr, raw_dtype=GGMLQuantizationType.BF16)
        elif action.out_dtype == GGMLQuantizationType.F16:
            if out_arr.dtype != np.float16:
                raise RuntimeError(
                    f"{name}: F16 out has dtype {out_arr.dtype}, expected float16"
                )
            writer.add_tensor(name, out_arr)
        elif action.out_dtype == GGMLQuantizationType.F32:
            writer.add_tensor(name, out_arr)
        else:
            writer.add_tensor(name, out_arr, raw_dtype=action.out_dtype)

        # T4 per-channel scales are emitted as flat KV arrays (recast.per_channel.*),
        # not as sibling tensors — siblings would be loaded as unmatched extras.

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
