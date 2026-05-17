#!/usr/bin/env python3
"""Walk an intra-layer capture pair (NP=1 vs NP=K) in fire order and report
the first diverging tensor.

Reads both manifests, joins on (phase, name), and for each shared tensor
compares the slot-0 row. For tensors whose size is identical across NP=1
and NP=K (state tensors, per-head buffers), the whole tensor is compared.
For token-sharded tensors (NP=K size == K * NP=1 size), only the first
NP=1-sized prefix (slot-0 row) is compared.

Usage:
    compare-intra-layer.py NP1_DIR NPK_DIR [--phase decode-0] [--np-k 2]
"""
import argparse
import json
import os
import struct
import sys


def load_manifest(d):
    with open(os.path.join(d, "manifest.json")) as f:
        return json.load(f)


def read_floats(p, n_floats):
    with open(p, "rb") as f:
        b = f.read(n_floats * 4)
    if len(b) != n_floats * 4:
        return None
    return struct.unpack(f"{n_floats}f", b)


def cmp_pair(a, b):
    n_diff = 0
    first = -1
    max_d = 0.0
    for i in range(len(a)):
        if a[i] != b[i]:
            n_diff += 1
            if first < 0:
                first = i
            d = abs(a[i] - b[i])
            if d > max_d:
                max_d = d
    return n_diff, first, max_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("np1_dir")
    ap.add_argument("npk_dir")
    ap.add_argument("--phase", default="decode-0")
    ap.add_argument("--np-k", type=int, default=2)
    ap.add_argument("--layer", type=int, default=None,
                    help="restrict to this layer (default: all)")
    args = ap.parse_args()

    m1 = load_manifest(args.np1_dir)
    mk = load_manifest(args.npk_dir)

    # Index NPK manifest by (phase, name) → record
    idx_k = {}
    for r in mk:
        if r["phase"] != args.phase:
            continue
        if args.layer is not None and r["layer"] != args.layer:
            continue
        idx_k[(r["phase"], r["name"], r["ubatch_idx"])] = r

    # Walk NP=1 manifest in execution order; for each record, look up the
    # NP=K counterpart and compare slot-0.
    ordered = sorted(
        [r for r in m1
         if r["phase"] == args.phase
         and (args.layer is None or r["layer"] == args.layer)],
        key=lambda r: r["order"],
    )

    print(f"phase={args.phase} layer={args.layer} np_k={args.np_k}")
    print(f"  NP=1 records: {len(ordered)}")
    print(f"  NP={args.np_k} records: {len(idx_k)}")
    print()
    print(f"{'ord':>4} {'name':36} {'sz1':>10} {'szk':>10} {'verdict':>12} "
          f"{'first':>8} {'max|Δ|':>12}")
    print("-" * 100)

    first_div = None
    for r in ordered:
        f1 = os.path.join(args.np1_dir, r["file"])
        key = (r["phase"], r["name"], r["ubatch_idx"])
        rk = idx_k.get(key)
        if rk is None:
            # Some NP=K runs fire the same name multiple times (per slot);
            # fall back to ubatch_idx=0 if exact match missing.
            for k2 in idx_k:
                if k2[0] == key[0] and k2[1] == key[1]:
                    rk = idx_k[k2]
                    break
        if rk is None:
            print(f"{r['order']:>4} {r['name']:36} {'-':>10} {'-':>10} "
                  f"{'NPK-MISSING':>12}")
            continue
        fk = os.path.join(args.npk_dir, rk["file"])
        sz1 = os.path.getsize(f1)
        szk = os.path.getsize(fk)

        # Compare the first min(sz1, szk) bytes — interpretable as slot-0
        # under either same-size (state/per-head) or 2× (token-sharded).
        n_cmp_floats = min(sz1, szk) // 4
        a = read_floats(f1, n_cmp_floats)
        b = read_floats(fk, n_cmp_floats)
        if a is None or b is None:
            print(f"{r['order']:>4} {r['name']:36} {sz1:>10} {szk:>10} "
                  f"{'READ-ERR':>12}")
            continue
        n_diff, first, max_d = cmp_pair(a, b)
        if n_diff == 0:
            print(f"{r['order']:>4} {r['name']:36} {sz1:>10} {szk:>10} "
                  f"{'IDENTICAL':>12}")
        else:
            print(f"{r['order']:>4} {r['name']:36} {sz1:>10} {szk:>10} "
                  f"{'DIFFERS':>12} {first:>8} {max_d:>12.3e}")
            if first_div is None:
                first_div = (r["order"], r["name"], first, max_d)

    print()
    if first_div is None:
        print("All shared tensors are slot-0 byte-identical.")
    else:
        order, name, first, max_d = first_div
        print(f"FIRST DIVERGENCE: order={order} name={name} "
              f"first_idx={first} max|Δ|={max_d:.3e}")


if __name__ == "__main__":
    main()
