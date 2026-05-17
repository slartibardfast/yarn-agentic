#!/usr/bin/env python3
"""Find the first autoregressive step at which two capture runs diverge.

Walks phases auto-0, auto-1, ... in order. At each step, compares every
l_out layer's slot-0 bytes between NP1_DIR and NPK_DIR. Reports the first
(step, layer) pair where slot-0 differs.

Usage:
    find-first-autoregress-divergence.py NP1_DIR NPK_DIR [--np-k 8]
"""
import argparse
import json
import os
import struct


def read_floats(p, n):
    with open(p, "rb") as f:
        b = f.read(n * 4)
    if len(b) != n * 4:
        return None
    return struct.unpack(f"{n}f", b)


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
    ap.add_argument("--np-k", type=int, default=8)
    ap.add_argument("--max-step", type=int, default=64)
    args = ap.parse_args()

    with open(os.path.join(args.np1_dir, "manifest.json")) as f:
        m1 = json.load(f)
    with open(os.path.join(args.npk_dir, "manifest.json")) as f:
        mk = json.load(f)

    # Index NPK by (phase, name)
    idx_k = {}
    for r in mk:
        idx_k[(r["phase"], r["name"])] = r

    # Group NP=1 records by phase, sort by order
    phases_n1 = {}
    for r in m1:
        phases_n1.setdefault(r["phase"], []).append(r)

    first_div = None
    print(f"phase                 layer  verdict     first   max|Δ|")
    print("-" * 60)
    for step in range(args.max_step):
        phase = f"auto-{step}"
        if phase not in phases_n1:
            continue
        recs = sorted(phases_n1[phase], key=lambda r: r["order"])
        step_status = "ALL-IDENTICAL"
        step_max = 0.0
        for r in recs:
            f1 = os.path.join(args.np1_dir, r["file"])
            rk = idx_k.get((phase, r["name"]))
            if rk is None:
                continue
            fk = os.path.join(args.npk_dir, rk["file"])
            sz1 = os.path.getsize(f1)
            szk = os.path.getsize(fk)
            n = min(sz1, szk) // 4
            a = read_floats(f1, n)
            b = read_floats(fk, n)
            if a is None or b is None:
                continue
            n_diff, first, max_d = cmp_pair(a, b)
            if n_diff > 0:
                step_status = f"DIVERGES@{r['name']}"
                step_max = max_d
                if first_div is None:
                    first_div = (phase, r["name"], first, max_d)
                print(f"{phase:18}  {r['layer']:>4}  DIFFERS    {first:>5}   {max_d:.3e}")
                break
        if step_status == "ALL-IDENTICAL":
            print(f"{phase:18}     -  ALL-IDENTICAL")
        else:
            # divergence detected this step; can keep walking to see
            # if it grows
            pass

    print()
    if first_div is None:
        print(f"No divergence across {args.max_step} autoregressive steps.")
    else:
        phase, name, first, max_d = first_div
        print(f"FIRST DIVERGENCE: {phase} {name} first_idx={first} max|Δ|={max_d:.3e}")


if __name__ == "__main__":
    main()
