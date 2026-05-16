#!/usr/bin/env python3
# TRACE-6 v2 — corrected slot_dim selection.

import os, json, struct

ROOT = "/home/llm/yarn-agentic/data/trace-6-2026-05-16"
N = 2

# Per-tensor slot dim, derived from build graph layout.
SLOT_DIM = {
    "l_out-2": 1, "l_out-3": 1,
    "attn_combined-3": 1, "attn_out_with_input-3": 1,
    "kqv_wo-1003": 1, "kqv_wo-2003": 1,
    "kqv_wo_biased-1003": 1, "kqv_wo_biased-2003": 1,
    "flash_attn_reshaped-1003": 1, "flash_attn_reshaped-2003": 1,
    "Vcur_hadamard-1003": 1, "Vcur_hadamard-2003": 1,
    # q = permute(Qcur, 0, 2, 1, 3) → ne[1] is n_tokens
    "q-1003": 1, "q-2003": 1,
    # Qcur, Kcur, post-Hadamard versions: ne[2] is n_tokens
    "Qcur-1003": 2, "Qcur-2003": 2,
    "Qcur_hadamard-1003": 2, "Qcur_hadamard-2003": 2,
    "Kcur-1003": 2, "Kcur-2003": 2,
    "Kcur_hadamard-1003": 2, "Kcur_hadamard-2003": 2,
    # FA output: [head_dim, n_heads, n_tokens, n_seqs]
    "flash_attn_per_slot_kv-1003": 2, "flash_attn_per_slot_kv-2003": 2,
    "flash_attn_h-1003": 2, "flash_attn_h-2003": 2,
}

def load(name):
    with open(f"{ROOT}/t2-np{N}-{name}.bin.json") as f:
        meta = json.load(f)
    with open(f"{ROOT}/t2-np{N}-{name}.bin", "rb") as f:
        data = f.read()
    return meta, data

def diff_floats(name):
    meta, data = load(name)
    ne = meta["ne"]
    if meta["type"] != "f32":
        return (ne, None, None, None, None, "non-f32")
    floats = struct.unpack(f"<{len(data)//4}f", data)
    sd = SLOT_DIM.get(name)
    if sd is None:
        return (ne, None, None, None, None, "no slot_dim mapping")
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
    return (ne, sd, slice_n, n_diff, max_abs, None)

names_in_order = [
    "l_out-2",
    # FA inputs (pre-FA)
    "Qcur-1003", "Qcur-2003",
    "Kcur-1003", "Kcur-2003",
    "Qcur_hadamard-1003", "Qcur_hadamard-2003",
    "Kcur_hadamard-1003", "Kcur_hadamard-2003",
    "Vcur_hadamard-1003", "Vcur_hadamard-2003",
    "q-1003", "q-2003",
    # FA output
    "flash_attn_per_slot_kv-1003", "flash_attn_per_slot_kv-2003",
    "flash_attn_h-1003", "flash_attn_h-2003",
    "flash_attn_reshaped-1003", "flash_attn_reshaped-2003",
    # Output projection
    "kqv_wo-1003", "kqv_wo-2003",
    # Combine + residual
    "attn_combined-3",
    "attn_out_with_input-3",
    # Final layer-3 output
    "l_out-3",
]

print(f"{'name':36s}  {'ne':25s}  {'sd':2s}  {'slice':>6s}  {'n_diff':>8s}  {'max|Δ|':>10s}")
print("-" * 96)
for name in names_in_order:
    p = f"{ROOT}/t2-np{N}-{name}.bin"
    if not os.path.exists(p):
        print(f"{name:36s}  MISSING")
        continue
    ne, sd, sn, nd, mad, err = diff_floats(name)
    if err:
        ne_str = f"[{ne[0]},{ne[1]},{ne[2]},{ne[3]}]"
        print(f"{name:36s}  {ne_str:25s}  ({err})")
        continue
    ne_str = f"[{ne[0]},{ne[1]},{ne[2]},{ne[3]}]"
    flag = "" if nd == 0 else "  ← DIVERGE"
    print(f"{name:36s}  {ne_str:25s}  {sd:2d}  {sn:6d}  {nd:8d}  {mad:10.3e}{flag}")
