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


def recast_tensor(
    name: str,
    src_arr: np.ndarray,
    src_type: GGMLQuantizationType,
    logical_shape: list[int],
    policy: Policy,
    tier: str,
) -> TensorAction:
    """Decide what dtype this tensor lands at and produce the array."""
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

    if band == "A":
        out = fp32.astype(np.float16)
        if np.isinf(out).any():
            raise ValueError(f"{name}: Band A produced Inf despite absmax {absmax} ≤ 32768")
        return TensorAction(GGMLQuantizationType.F16, out, "A", absmax, "RNE-cast")

    if band == "B":
        out = fp32.astype(np.float16)
        new_inf = int(np.isinf(out).sum())
        if new_inf > 0:
            raise ValueError(f"{name}: Band B produced {new_inf} Inf at boundary")
        return TensorAction(GGMLQuantizationType.F16, out, "B", absmax, "RNE-cast (edge)")

    # Band C dispatch.
    if tier == "dry-run":
        return TensorAction(
            GGMLQuantizationType.BF16, src_arr, "C", absmax, "dry-run (no cast)"
        )
    if tier == "T1":
        return TensorAction(GGMLQuantizationType.BF16, src_arr, "C", absmax, "T1 BF16 fallback")
    if tier in ("T2", "T3", "T4", "T5"):
        # Stub for escalation tiers — not implemented yet.
        raise NotImplementedError(
            f"Tier {tier} not yet implemented; run T1 first per PHASE32 escalation rule"
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
        action = recast_tensor(t.name, t.data, src_type, logical_shape, policy, args.tier)
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

    # Second pass: write tensors. Use add_tensor with raw_dtype.
    for name, action in actions:
        out_arr = action.out_arr
        if action.out_dtype == GGMLQuantizationType.BF16:
            # Pass-through: keep raw uint8 bytes from the reader.
            # GGUFWriter calls quant_shape_from_byte_shape internally.
            if out_arr.dtype != np.uint8:
                raise RuntimeError(
                    f"{name}: BF16 out has dtype {out_arr.dtype}, expected uint8 raw bytes"
                )
            writer.add_tensor(name, out_arr, raw_dtype=GGMLQuantizationType.BF16)
        elif action.out_dtype == GGMLQuantizationType.F16:
            # np.float16 → writer auto-detects.
            if out_arr.dtype != np.float16:
                raise RuntimeError(
                    f"{name}: F16 out has dtype {out_arr.dtype}, expected float16"
                )
            writer.add_tensor(name, out_arr)
        elif action.out_dtype == GGMLQuantizationType.F32:
            writer.add_tensor(name, out_arr)
        else:
            # Other passthrough cases (Q-types). Pass raw bytes + raw_dtype.
            writer.add_tensor(name, out_arr, raw_dtype=action.out_dtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
