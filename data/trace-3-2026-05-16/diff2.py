#!/usr/bin/env python3
# TRACE-3 v2 — correct layout. K cache view ne=[256, 256, 2, 1] with
# strides nb0 = type-size, nb1 = 144 * 2 = 288 (per position), nb2 = 144 (per head_kv).
# Memory layout: pos0/head0 (144B), pos0/head1 (144B), pos1/head0, pos1/head1, ...
# So row r in flat 144-byte rows = pos (r//2), head (r%2).
#
# The KV cache uses global-packed layout (not per-slot-segmented). For our test:
#   slot 0 prefill writes positions [0..11]
#   slot 1 prefill writes positions [12..23]
#   slot 0 decode writes position 24
#   slot 1 decode writes position 25
# 52 nonzero rows = 26 positions × 2 heads.
#
# Compare slot 0's Q4_0 bytes vs slot 1's Q4_0 bytes for same prompt:
#   slot 0 prefill pos p, head h ← row 2*p + h  for p in [0..11]
#   slot 1 prefill pos p, head h ← row 2*(12 + p) + h  for p in [0..11]

import json, os

ROOT = "/home/llm/yarn-agentic/data/trace-3-2026-05-16"
N = 2
N_PROMPT = 12

def load(name):
    with open(f"{ROOT}/t2-np{N}-{name}.bin.json") as f:
        meta = json.load(f)
    with open(f"{ROOT}/t2-np{N}-{name}.bin", "rb") as f:
        data = f.read()
    return meta, data

def slice_row(data, r, bpr=144):
    return data[r*bpr:(r+1)*bpr]

def compare(name):
    meta, data = load(name)
    ne = meta["ne"]
    n_head_kv = ne[2]
    print(f"\n=== {name}  ne={ne} ===")
    # For each head_kv: compare slot 0 prefill positions to slot 1 prefill positions.
    diffs_total = 0
    for h in range(n_head_kv):
        head_diffs = 0
        for p in range(N_PROMPT):
            r_s0 = 2 * p + h
            r_s1 = 2 * (N_PROMPT + p) + h
            b0 = slice_row(data, r_s0)
            b1 = slice_row(data, r_s1)
            if b0 != b1:
                head_diffs += 1
        diffs_total += head_diffs
        status = "BYTE-IDENTICAL" if head_diffs == 0 else f"DIVERGE in {head_diffs}/{N_PROMPT} positions"
        print(f"  head_kv {h}: prefill positions slot0 vs slot1: {status}")
    # Also check the decode-step position 24 (slot 0 decode) vs 25 (slot 1 decode)
    for h in range(n_head_kv):
        r_s0 = 2 * 24 + h
        r_s1 = 2 * 25 + h
        b0 = slice_row(data, r_s0)
        b1 = slice_row(data, r_s1)
        eq = b0 == b1
        print(f"  head_kv {h}: decode-step pos 24 (slot0) vs pos 25 (slot1): {'BYTE-IDENTICAL' if eq else 'DIVERGE'}")
    return diffs_total

for name in ["k-1003", "v-1003", "k-2003", "v-2003"]:
    compare(name)
