#!/usr/bin/env python3
# TRACE-1 diff analysis.
# Per-slot per-layer .bin = n_embd float32 (5120 floats * 4 = 20480 bytes per file).
# Two analyses:
#   A) intra-NP slot uniformity: same-prompt → slots should produce byte-identical residuals.
#   B) cross-NP slot-0 invariance: slot 0's residual should match np=1's slot 0 across NP.

import os, struct, sys

ROOT = "/home/llm/yarn-agentic/data/trace-1-2026-05-16"
N_LAYER = 64  # writer wrote layers 0..63 (64 was skipped as MTP/aux)

def load_bin(path):
    with open(path, "rb") as f:
        data = f.read()
    n = len(data) // 4
    return struct.unpack(f"<{n}f", data)

def bit_diff(a, b):
    # Returns (n_diff_bits, max_abs_delta) — count of bit-pattern-mismatched floats.
    n = len(a)
    n_diff = 0
    max_abs = 0.0
    for x, y in zip(a, b):
        bx = struct.pack("<f", x)
        by = struct.pack("<f", y)
        if bx != by:
            n_diff += 1
            d = abs(x - y)
            if d > max_abs:
                max_abs = d
    return n_diff, max_abs

# ---- Analysis A: intra-NP slot uniformity ----
print("=" * 78)
print("A) intra-NP slot uniformity (slot s vs slot 0 at same NP)")
print("=" * 78)
for np in [2, 4, 8]:
    subdir = f"{ROOT}/np{np}"
    print(f"\n--- NP={np} ---")
    first_div_per_pair = {}
    for s in range(1, np):
        first_div = -1
        max_div_layer = -1
        max_div_amount = 0.0
        for L in range(N_LAYER):
            p0 = f"{subdir}/d1-np{np}-offset0-layer{L}-slot0.bin"
            ps = f"{subdir}/d1-np{np}-offset0-layer{L}-slot{s}.bin"
            if not (os.path.exists(p0) and os.path.exists(ps)):
                continue
            a = load_bin(p0)
            b = load_bin(ps)
            nd, mad = bit_diff(a, b)
            if nd > 0:
                if first_div == -1:
                    first_div = L
                if mad > max_div_amount:
                    max_div_amount = mad
                    max_div_layer = L
        if first_div == -1:
            print(f"  slot{s} vs slot0: ALL {N_LAYER} LAYERS BYTE-IDENTICAL")
        else:
            print(f"  slot{s} vs slot0: first divergent layer = {first_div}; "
                  f"max |Δ| = {max_div_amount:.3e} at layer {max_div_layer}")
        first_div_per_pair[s] = first_div

# ---- Analysis B: cross-NP slot-0 ----
print()
print("=" * 78)
print("B) cross-NP slot-0 invariance (slot 0 at NP=1 vs slot 0 at NP={2,4,8})")
print("=" * 78)
for np in [2, 4, 8]:
    np1_dir = f"{ROOT}/np1"
    npN_dir = f"{ROOT}/np{np}"
    first_div = -1
    max_div_layer = -1
    max_div_amount = 0.0
    for L in range(N_LAYER):
        p1 = f"{np1_dir}/d1-np1-offset0-layer{L}-slot0.bin"
        pN = f"{npN_dir}/d1-np{np}-offset0-layer{L}-slot0.bin"
        if not (os.path.exists(p1) and os.path.exists(pN)):
            continue
        a = load_bin(p1)
        b = load_bin(pN)
        nd, mad = bit_diff(a, b)
        if nd > 0:
            if first_div == -1:
                first_div = L
            if mad > max_div_amount:
                max_div_amount = mad
                max_div_layer = L
    if first_div == -1:
        print(f"  NP=1 slot0 vs NP={np} slot0: ALL {N_LAYER} LAYERS BYTE-IDENTICAL")
    else:
        print(f"  NP=1 slot0 vs NP={np} slot0: first divergent layer = {first_div}; "
              f"max |Δ| = {max_div_amount:.3e} at layer {max_div_layer}")
