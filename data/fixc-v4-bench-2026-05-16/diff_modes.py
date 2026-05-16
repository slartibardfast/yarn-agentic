#!/usr/bin/env python3
# FIX-C v4 bench: compare slot 0 vs slot 1 byte-identity at the FA
# output (flash_attn_per_slot_kv-1003 and -2003) and at l_out-3 across
# three dispatcher modes (wmma, vec_f32, vec_f16).
#
# Same-prompt NP=2 inputs → if FIX-C is correct, slot 0 ≡ slot 1 at
# the FA output for vec_f32 and vec_f16 modes (and l_out-3 differs
# only by downstream propagation, but should ALSO be ≡).

import os, json, struct

ROOT = "/home/llm/yarn-agentic/data/fixc-v4-bench-2026-05-16"

# slot_dim per tensor (per build-graph layout) — same as TRACE-6 v2 mapping.
SLOT_DIM = {
    "flash_attn_per_slot_kv-1003": 2,
    "flash_attn_per_slot_kv-2003": 2,
    "l_out-3":   1,
    "attn_combined-3": 1,
    "attn_out_with_input-3": 1,
    "kqv_wo-1003": 1, "kqv_wo-2003": 1,
}

def diff_one(mode, name):
    bin_p = f"{ROOT}/{mode}/t2-np2-{name}.bin"
    js_p  = bin_p + ".json"
    if not os.path.exists(bin_p):
        return None
    with open(js_p) as f:
        meta = json.load(f)
    with open(bin_p, "rb") as f:
        data = f.read()
    if meta["type"] != "f32":
        return None
    floats = struct.unpack(f"<{len(data)//4}f", data)
    ne = meta["ne"]
    sd = SLOT_DIM.get(name)
    if sd is None:
        return None
    slice_n = 1
    for i in range(sd):
        slice_n *= ne[i]
    s0 = floats[0:slice_n]
    s1 = floats[slice_n:2*slice_n]
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
    return (slice_n, n_diff, max_abs)

KEYS = [
    "flash_attn_per_slot_kv-1003",
    "flash_attn_per_slot_kv-2003",
    "kqv_wo-1003",
    "kqv_wo-2003",
    "attn_combined-3",
    "attn_out_with_input-3",
    "l_out-3",
]

print(f"{'mode':10s}  {'tensor':32s}  {'slice':>6s}  {'n_diff':>8s}  {'max|Δ|':>10s}")
print("-" * 80)
for mode in ["wmma", "vec_f32", "vec_f16"]:
    for k in KEYS:
        r = diff_one(mode, k)
        if r is None:
            print(f"{mode:10s}  {k:32s}  MISSING")
            continue
        slice_n, n_diff, max_abs = r
        flag = "" if n_diff == 0 else "  ← DIVERGE"
        print(f"{mode:10s}  {k:32s}  {slice_n:6d}  {n_diff:8d}  {max_abs:10.3e}{flag}")
    print()
