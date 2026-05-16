#!/usr/bin/env python3
# TRACE-2 — slot 0 vs slot 1 diff for each captured intra-layer-3 intermediate at NP=2.
# All captures are F32. Per-slot slice = ne[0]*ne[1] (for tensors with ne[2]=2 tokens)
# or ne[0] (for tensors with ne[1]=2 tokens). The slot dim is whichever of ne[1] or ne[2]
# equals N (=2 here).

import os, json, struct

ROOT = "/home/llm/yarn-agentic/data/trace-2-2026-05-16"
N = 2

def load(name):
    bin_p = f"{ROOT}/t2-np{N}-{name}.bin"
    js_p  = bin_p + ".json"
    with open(js_p) as f:
        meta = json.load(f)
    ne = meta["ne"]
    with open(bin_p, "rb") as f:
        data = f.read()
    n = len(data) // 4
    floats = struct.unpack(f"<{n}f", data)
    # Determine slot dim. Look for an ne[i] in {1,2,3,4} that equals N and skip earlier dims that look like data dims.
    # Heuristic: slot dim is the first ne[i] == N for i in [1,2] (1 first since some tensors collapse).
    slot_dim = -1
    if ne[2] == N:
        slot_dim = 2
    elif ne[1] == N:
        slot_dim = 1
    else:
        raise RuntimeError(f"{name}: cannot find slot dim in ne={ne}")
    # Slice size = product(ne[0..slot_dim-1])
    slice_n = 1
    for i in range(slot_dim):
        slice_n *= ne[i]
    return ne, floats, slot_dim, slice_n

def diff_slots(name):
    ne, fs, slot_dim, slice_n = load(name)
    s0 = fs[0:slice_n]
    s1 = fs[slice_n:2*slice_n]
    n_diff = 0
    max_abs = 0.0
    for a, b in zip(s0, s1):
        ba = struct.pack("<f", a)
        bb = struct.pack("<f", b)
        if ba != bb:
            n_diff += 1
            d = abs(a - b)
            if d > max_abs:
                max_abs = d
    return ne, slot_dim, slice_n, n_diff, max_abs

names = [
    "l_out-2",
    "Qcur-1003", "Qcur-2003",
    "Kcur-1003", "Kcur-2003",
    "Qcur_hadamard-1003", "Qcur_hadamard-2003",
    "Kcur_hadamard-1003", "Kcur_hadamard-2003",
    "Vcur_hadamard-1003", "Vcur_hadamard-2003",
    "l_out-3",
]

print(f"{'name':30s}  {'ne':25s}  {'slot_dim':3s}  {'slice_n':>8s}  {'n_diff':>8s}  {'max|Δ|':>10s}")
print("-" * 100)
for name in names:
    try:
        ne, sd, sn, nd, mad = diff_slots(name)
    except FileNotFoundError as e:
        print(f"{name:30s}  MISSING ({e})")
        continue
    ne_str = f"[{ne[0]},{ne[1]},{ne[2]},{ne[3]}]"
    flag = "" if nd == 0 else "  ← DIVERGE"
    print(f"{name:30s}  {ne_str:25s}  {sd:3d}  {sn:8d}  {nd:8d}  {mad:10.3e}{flag}")
