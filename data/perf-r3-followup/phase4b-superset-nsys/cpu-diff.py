#!/usr/bin/env python3
"""Diff per-function CPU samples across the 4 ctx points + 2 reps each.

For each ctx, sums leaf samples across both reps (more data, less noise).
Outputs:
  - Top 25 functions by ctx=256k sample count
  - For each: per-ctx counts + delta vs ctx=8k + slope per doubling
"""
import os, sqlite3, sys
from collections import defaultdict

OUT = "/tmp/perf-r3-followup-phase4b-20260528T092053"
CTXS = [("ctx8192", 8192), ("ctx32768", 32768), ("ctx131072", 131072), ("ctx262144", 262144)]

# Map: function_name -> per-ctx total sample count
counts = defaultdict(lambda: {c[1]: 0 for c in CTXS})

for label, ctx in CTXS:
    for rep in (1, 2):
        sqlite_path = f"{OUT}/{label}/trace.{rep}.sqlite"
        if not os.path.exists(sqlite_path):
            print(f"  MISSING: {sqlite_path}", file=sys.stderr)
            continue
        conn = sqlite3.connect(sqlite_path)
        cur = conn.cursor()
        cur.execute("""
            SELECT s.value AS fn, m.value AS mod, COUNT(*) AS samples
            FROM SAMPLING_CALLCHAINS scc
            JOIN StringIds s ON scc.symbol = s.id
            JOIN StringIds m ON scc.module = m.id
            WHERE scc.stackDepth = 0
            GROUP BY scc.symbol, scc.module
        """)
        for fn, mod, n in cur.fetchall():
            # Tag function with its module to disambiguate identical names
            key = f"{fn} [{os.path.basename(mod)}]"
            counts[key][ctx] += n
        conn.close()

# Rank by ctx=256k counts
ranked = sorted(counts.items(), key=lambda kv: -kv[1][262144])

print(f"{'function [module]':<78} {'8k':>6} {'32k':>6} {'128k':>6} {'256k':>6} {'Δvs8k':>7} {'×':>5}")
print("-" * 130)
for key, c in ranked[:30]:
    c8 = c[8192]; c32 = c[32768]; c128 = c[131072]; c256 = c[262144]
    if c8 + c32 + c128 + c256 < 30:
        continue  # skip noise
    delta_pct = ((c256 - c8) / c8 * 100) if c8 > 0 else float("inf")
    mult = (c256 / c8) if c8 > 0 else float("inf")
    delta_str = f"{delta_pct:+.0f}%" if delta_pct != float("inf") else " NEW"
    mult_str = f"{mult:.1f}x" if mult != float("inf") else " inf"
    print(f"{key[:78]:<78} {c8:>6} {c32:>6} {c128:>6} {c256:>6} {delta_str:>7} {mult_str:>5}")

# Now filter to ggml/llama functions and sort by absolute growth
print()
print("=== ggml/llama functions only, sorted by |Δsamples| (ctx256k − ctx8k) ===")
print(f"{'function [module]':<78} {'8k':>6} {'32k':>6} {'128k':>6} {'256k':>6} {'Δ':>7} {'×':>5}")
print("-" * 130)
ggml_llama = []
for key, c in counts.items():
    if ('libggml' in key or 'libllama' in key) and not key.startswith('0x'):
        c8 = c[8192]; c32 = c[32768]; c128 = c[131072]; c256 = c[262144]
        if c8 + c256 < 20:
            continue
        ggml_llama.append((key, c8, c32, c128, c256, c256 - c8))
ggml_llama.sort(key=lambda x: -abs(x[5]))
for key, c8, c32, c128, c256, delta in ggml_llama[:25]:
    mult = (c256 / c8) if c8 > 0 else float("inf")
    mult_str = f"{mult:.1f}x" if mult != float("inf") else " inf"
    print(f"{key[:78]:<78} {c8:>6} {c32:>6} {c128:>6} {c256:>6} {delta:>+7} {mult_str:>5}")

# Totals
print()
print("=== Totals (all functions) ===")
totals = defaultdict(int)
for key, c in counts.items():
    for k, v in c.items():
        totals[k] += v
print(f"  ctx=  8k: {totals[8192]:>7,} samples = ~{totals[8192]/1000:.1f}s thread-time")
print(f"  ctx= 32k: {totals[32768]:>7,} samples = ~{totals[32768]/1000:.1f}s thread-time")
print(f"  ctx=128k: {totals[131072]:>7,} samples = ~{totals[131072]/1000:.1f}s thread-time")
print(f"  ctx=256k: {totals[262144]:>7,} samples = ~{totals[262144]/1000:.1f}s thread-time")
print(f"  256k vs 8k: {totals[262144]/totals[8192]:.2f}× thread-time")
