#!/usr/bin/env python3
"""Parse the Vulkan perf logger stderr output into a structured JSON table.

The vk_perf_logger class (ggml-vulkan.cpp:1722) emits blocks like:

    ----------------
    Vulkan Timings:
    <op_name>: <count> x <avg_us> us = <total_us> us (<GFLOPS>/s)
    <op_name>: <count> x <avg_us> us = <total_us> us
    ...
    Total time: <total_us> us.

This script reads stderr output from stdin (or a file), finds every Timings
block, aggregates op counts/times across all blocks, and emits a JSON
summary with the per-op totals sorted by total time descending.

Usage:
    python3 parse_vk_perf.py path/to/server.stderr > summary.json
    cat server.stderr | python3 parse_vk_perf.py > summary.json
"""
import json
import re
import sys
from collections import defaultdict


# Example line:
#   MUL_MAT f16 m=2560 n=1 k=2560: 42 x 321.4 us = 13499 us (12.9 GFLOPS/s)
#   RMS_NORM_MUL: 18 x 6.3 us = 113 us
_LINE_RE = re.compile(
    r"^(?P<op>.*?): (?P<count>\d+) x (?P<avg_us>[\d.]+) us = (?P<total_us>[\d.]+) us(?: \((?P<gflops>[\d.]+) GFLOPS/s\))?\s*$"
)
_TOTAL_RE = re.compile(r"^Total time: (?P<total_us>[\d.]+) us\.\s*$")
_HEADER_RE = re.compile(r"^Vulkan Timings:\s*$")


def parse(stream):
    ops = defaultdict(lambda: {"count": 0, "total_us": 0.0, "gflops_samples": []})
    block_totals = []
    in_block = False
    for raw in stream:
        line = raw.rstrip()
        if _HEADER_RE.match(line):
            in_block = True
            continue
        if not in_block:
            continue
        m_total = _TOTAL_RE.match(line)
        if m_total:
            block_totals.append(float(m_total["total_us"]))
            in_block = False
            continue
        m = _LINE_RE.match(line)
        if not m:
            # Unexpected line inside a block — bail out of the block.
            in_block = False
            continue
        op_name = m["op"].strip()
        count = int(m["count"])
        total_us = float(m["total_us"])
        ops[op_name]["count"] += count
        ops[op_name]["total_us"] += total_us
        if m["gflops"] is not None:
            try:
                ops[op_name]["gflops_samples"].append(float(m["gflops"]))
            except ValueError:
                pass

    rows = []
    grand_total_us = 0.0
    for op_name, stats in ops.items():
        avg_us = stats["total_us"] / stats["count"] if stats["count"] else 0.0
        gflops = (
            sum(stats["gflops_samples"]) / len(stats["gflops_samples"])
            if stats["gflops_samples"] else None
        )
        rows.append({
            "op": op_name,
            "count": stats["count"],
            "total_us": round(stats["total_us"], 1),
            "avg_us": round(avg_us, 2),
            "avg_gflops_s": round(gflops, 1) if gflops is not None else None,
        })
        grand_total_us += stats["total_us"]

    rows.sort(key=lambda r: r["total_us"], reverse=True)
    for r in rows:
        r["share_pct"] = round(100.0 * r["total_us"] / grand_total_us, 2) if grand_total_us else 0.0

    summary = {
        "n_blocks": len(block_totals),
        "block_total_us_sum": round(sum(block_totals), 1),
        "ops_total_us_sum": round(grand_total_us, 1),
        "n_distinct_ops": len(rows),
        "ops": rows,
    }
    return summary


def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            summary = parse(f)
    else:
        summary = parse(sys.stdin)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
