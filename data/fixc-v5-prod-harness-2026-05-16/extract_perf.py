#!/usr/bin/env python3
# Extract per-slot decode rate from each harness JSON.
import json, os, glob, sys

def extract(rundir):
    out = {}
    for f in sorted(glob.glob(f"{rundir}/np*.json")):
        name = os.path.basename(f).replace(".json", "")
        with open(f) as fp:
            d = json.load(fp)
        tm = d.get("timings", {})
        out[name] = {
            "prompt_ms": tm.get("prompt_ms", 0),
            "prompt_per_second": tm.get("prompt_per_second", 0),
            "predicted_n": tm.get("predicted_n", 0),
            "predicted_ms": tm.get("predicted_ms", 0),
            "predicted_per_second": tm.get("predicted_per_second", 0),
        }
    return out

def summarize(label, rundir):
    data = extract(rundir)
    print(f"\n=== {label} ===")
    print(f"  Source: {rundir}")
    print(f"  {'slot':18s}  {'predicted_n':>10s}  {'predicted_ms':>10s}  {'per_slot_t/s':>12s}")
    np_aggregates = {}
    for k, v in data.items():
        np = k.split("-")[0].replace("np", "")
        ps = v["predicted_per_second"]
        np_aggregates.setdefault(np, []).append(ps)
        print(f"  {k:18s}  {v['predicted_n']:>10d}  {v['predicted_ms']:>10.1f}  {ps:>12.2f}")
    print(f"  --- aggregate per NP (sum of per-slot t/s) ---")
    for np, rates in sorted(np_aggregates.items(), key=lambda x: int(x[0])):
        agg = sum(rates)
        mean = agg / len(rates)
        print(f"  NP={np:2s}  count={len(rates)}  mean_per_slot={mean:.2f}t/s  aggregate={agg:.2f}t/s")

singlewarp = sorted(glob.glob("/home/llm/yarn-agentic/data/fixc-v5-prod-harness-2026-05-16/run-*"))[-1]
wmma = sorted(glob.glob("/home/llm/yarn-agentic/data/fixc-v5-prod-harness-wmma-2026-05-16/run-*"))[-1]

summarize("singlewarp (FIX-C v5)", singlewarp)
summarize("wmma (baseline)", wmma)
