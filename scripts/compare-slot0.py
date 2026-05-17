#!/usr/bin/env python3
"""Compare slot-0 of NP=K capture against NP=1 capture, per layer + per tensor.

Usage:
    compare-slot0.py NP1_DIR NP2_DIR [--tensor PREFIX] [--layers L1,L2,...] [--ub N]
"""
import argparse
import os
import struct
import sys


def fp32_bytes_per_token(n_embd=5120):
    return n_embd * 4


def cmp_floats(p1, p2, n_floats):
    """Return (identical, n_diff, first_diff_idx, max_abs_diff)."""
    n_bytes = n_floats * 4
    with open(p1, "rb") as f:
        a = f.read(n_bytes)
    with open(p2, "rb") as f:
        b = f.read(n_bytes)
    if len(a) != n_bytes or len(b) != n_bytes:
        return None
    af = struct.unpack(f"{n_floats}f", a)
    bf = struct.unpack(f"{n_floats}f", b)
    n_diff = 0
    first = -1
    max_diff = 0.0
    for i in range(n_floats):
        if af[i] != bf[i]:
            n_diff += 1
            if first < 0:
                first = i
            d = abs(af[i] - bf[i])
            if d > max_diff:
                max_diff = d
    return (n_diff == 0, n_diff, first, max_diff, af, bf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("np1_dir")
    ap.add_argument("np2_dir")
    ap.add_argument("--tensor", default="l_out", help="tensor prefix")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--ub", type=int, default=2, help="ubatch index")
    ap.add_argument("--n-embd", type=int, default=5120)
    args = ap.parse_args()

    layers = list(range(64)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]

    for L in layers:
        lpad = f"{L:02d}"
        f1 = f"{args.np1_dir}/layer{lpad}/{args.tensor}-{L}.ub{args.ub}.bin"
        f2 = f"{args.np2_dir}/layer{lpad}/{args.tensor}-{L}.ub{args.ub}.bin"
        if not os.path.exists(f1) or not os.path.exists(f2):
            print(f"layer {lpad}: SKIP (missing)")
            continue
        # NP=1 file is slot-0 (one token = n_embd floats); NP=2 is two tokens (2*n_embd floats)
        n_floats = args.n_embd  # compare slot-0 row only
        s1 = os.path.getsize(f1)
        s2 = os.path.getsize(f2)
        # at decode step ub2/ub3, NP=1 has 1 row, NP=2 has 2 rows. Always compare first n_embd floats.
        result = cmp_floats(f1, f2, n_floats)
        if result is None:
            print(f"layer {lpad}: SIZE-ERR f1={s1} f2={s2}")
            continue
        ident, n_diff, first, max_d, af, bf = result
        if ident:
            print(f"layer {lpad}: IDENTICAL")
        else:
            print(f"layer {lpad}: DIFFERS {n_diff}/{n_floats} first_idx={first} a={af[first]:.6e} b={bf[first]:.6e} max|Δ|={max_d:.3e}")


if __name__ == "__main__":
    main()
