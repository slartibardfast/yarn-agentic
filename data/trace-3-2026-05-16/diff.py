#!/usr/bin/env python3
# TRACE-3 — disambiguate Q4_0 KV cache content vs FA per-slot-kv kernel.
#
# `k-1003` is a view of the layer-3 K cache (Q4_0) for device 0.
# Shape [head_dim=256, n_kv=256, n_head_kv=2, n_seqs=1].
# Q4_0 block size 32 → 8 blocks per head-position → 18 bytes/block →
# 144 bytes per (head, position). Row-major ne[0]-fastest:
#   memory[head_dim_block(=8 blocks)][pos][head_kv][seq]
# But really ggml encodes Q4_0 along ne[0] only: 8 blocks × 18 bytes = 144 B per
# logical row. Each row at fixed (pos, head_kv, seq) is 144 B.
# Total view bytes (raw): n_kv * n_head_kv * 144 = 256 * 2 * 144 = 73728.
# (Reported 73872 may include a small Q4_0 alignment pad.)
#
# Per-slot layout: at decode time, slot s's prefilled K is at positions
# [s * ctx_per_slot, s * ctx_per_slot + n_prompt). For n_ctx=4096,
# n_seq_max=2 → ctx_per_slot=2048. But n_kv view is only 256 — so the view
# size is padded up from "max written position so far" to FATTN_KQ_STRIDE=256.
# This means slot 1's K is OUTSIDE the view (slot 1 starts at position 2048).
#
# That alone would explain a bug: if FA reads K via this view only and the
# view doesn't span slot 1's region, slot 1's queries get garbage / slot 0's K.
#
# This script reads the k/v captures byte-wise and inspects the distribution
# of non-zero blocks to localize slot data within the view.

import os, json, struct

ROOT = "/home/llm/yarn-agentic/data/trace-3-2026-05-16"
N = 2

def read_bin(name):
    p = f"{ROOT}/t2-np{N}-{name}.bin"
    with open(p, "rb") as f:
        return f.read()

def read_meta(name):
    p = f"{ROOT}/t2-np{N}-{name}.bin.json"
    with open(p) as f:
        return json.load(f)

def block_hashes(data, blocksz):
    # Return list of (offset, all-zero-bool, simple-hash) per block.
    out = []
    n = len(data) // blocksz
    for i in range(n):
        b = data[i*blocksz:(i+1)*blocksz]
        is_zero = all(x == 0 for x in b)
        h = sum(b) & 0xffffffff
        out.append((i, is_zero, h))
    return out

def analyze_q4_0_kv(name, view_label):
    meta = read_meta(name)
    data = read_bin(name)
    ne = meta["ne"]
    print(f"\n=== {name} ===")
    print(f"ne = {ne}, type = {meta['type']}, bytes = {len(data)}")

    # Q4_0 block size in bytes
    Q4_0_BS = 18  # 2 byte fp16 d + 16 byte quants
    # ne[0] = head_dim; n_blocks per head-position row = ne[0]/32
    n_blocks_per_row = ne[0] // 32
    bytes_per_row = n_blocks_per_row * Q4_0_BS
    n_rows = (len(data) // bytes_per_row)
    # n_rows should be ne[1] * ne[2] * ne[3]
    expected_rows = ne[1] * ne[2] * ne[3]
    print(f"bytes_per_row={bytes_per_row}, n_rows={n_rows}, expected_rows={expected_rows}")

    # Walk every row; classify into (zero, nonzero) by checking if all bytes are 0.
    nonzero_rows = []
    for r in range(n_rows):
        row = data[r*bytes_per_row:(r+1)*bytes_per_row]
        if any(b != 0 for b in row):
            nonzero_rows.append(r)
    print(f"nonzero rows: {len(nonzero_rows)} of {n_rows}")
    if len(nonzero_rows) < 30:
        print(f"  indices: {nonzero_rows}")
    else:
        print(f"  first 30 indices: {nonzero_rows[:30]}")
        print(f"  last 10 indices: {nonzero_rows[-10:]}")
    return data, ne, bytes_per_row

def compare_k_or_v(name, slot_region_size_rows, n_prompt):
    """For a Q4_0 K cache view, compare slot 0's region (positions [0, n_prompt))
    against slot 1's region (positions [slot_region_size_rows, slot_region_size_rows + n_prompt))."""
    data, ne, bpr = analyze_q4_0_kv(name, f"slot_region={slot_region_size_rows}")

    # ne[1] is n_kv. Row index of position p in head_kv h: r = h*ne[1] + p
    # (ggml row-major; ne[0] inside-row.)
    # Actually with ne[2]=n_head_kv and ne[1]=n_kv, the row index is h*ne[1] + p
    # if ne[2] is the slower dim. ggml index: i0 + i1*ne0 + i2*ne0*ne1 + ...
    # For Q4_0 the "row" is the ne[0] dimension. So total flat = i1*Q4_rows_per_pos + i2*ne[1]*Q4_rows_per_pos.
    # Per-row blocks = ne[0]/32. ne[1] positions, ne[2] heads.
    n_head_kv = ne[2]
    n_kv = ne[1]
    for h in range(n_head_kv):
        head_off = h * n_kv * bpr
        # Slot 0's positions [0, n_prompt) → rows [0, n_prompt) within this head
        # Slot 1's positions: if n_kv >= 2048+12, [2048, 2048+12) — likely outside view
        # We'll just compare positions [0..n_prompt) to positions [n_prompt..2*n_prompt)
        # to see if there's a pattern.
        s0 = data[head_off + 0*bpr : head_off + n_prompt*bpr]
        s1_at_n_prompt = data[head_off + n_prompt*bpr : head_off + 2*n_prompt*bpr]
        eq = (s0 == s1_at_n_prompt)
        print(f"  head_kv {h}: slot0 region [0,{n_prompt}) == positions [{n_prompt},{2*n_prompt})? {eq}")

# Per-device captures
for dev_suffix in ["1003", "2003"]:
    print(f"\n========== device suffix {dev_suffix} ==========")
    compare_k_or_v(f"k-{dev_suffix}", slot_region_size_rows=2048, n_prompt=13)
    compare_k_or_v(f"v-{dev_suffix}", slot_region_size_rows=2048, n_prompt=13)

# Also inspect q-1003 byte-vs-byte for slot 0 vs slot 1 — should be identical per TRACE-2
print()
print("=== q-1003 slot comparison ===")
m = read_meta("q-1003")
print(f"q-1003 ne = {m['ne']}")
data = read_bin("q-1003")
floats = struct.unpack(f"<{len(data)//4}f", data)
# ne = [256, 2, 12, 1]. Slot dim is ne[1]=2 (n_tokens=2 packed there).
# Slot 0 floats: indices where i1=0, all (i0, i2, i3). Per-slot slice = ne[0] = 256.
# But layout: i0 + i1*ne0 + i2*ne0*ne1 → slot 0 (i1=0) is interleaved with slot 1 (i1=1).
# For each (i0, i2, i3) triple, slot 0 is at i0 + 0*256 + i2*256*2 = i0 + i2*512.
# Slot 1 is at i0 + 1*256 + i2*256*2 = i0 + 256 + i2*512.
ne = m["ne"]
n_diff = 0
max_abs = 0.0
for i2 in range(ne[2]):
    for i0 in range(ne[0]):
        s0_idx = i0 + 0*ne[0] + i2*ne[0]*ne[1]
        s1_idx = i0 + 1*ne[0] + i2*ne[0]*ne[1]
        a = floats[s0_idx]; b = floats[s1_idx]
        ba = struct.pack("<f", a); bb = struct.pack("<f", b)
        if ba != bb:
            n_diff += 1
            d = abs(a - b)
            if d > max_abs:
                max_abs = d
print(f"q-1003 slot 0 vs slot 1: n_diff={n_diff}, max|Δ|={max_abs:.3e}")
