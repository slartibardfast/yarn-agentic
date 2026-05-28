#!/usr/bin/env python3
"""Parse GGML_SCHED_DEBUG output to infer op coverage.

The scheduler prints per-op assignment lines that look like:

    split #0 (CPU), ...
    split #1 (Vulkan0), n_inputs = 1, n_outputs = 42, ...
    node #123 ( CUR = GGML_OP_MUL_MAT (q4_K)) ...

Counts how many ops landed on which backend and emits a summary with:
- total split count
- splits per backend
- op histogram per backend
- any GGML_OP_* names that appeared on CPU but not Vulkan

Usage:
    python3 parse_sched_debug.py <path/to/server.stderr>
"""
import json
import re
import sys
from collections import defaultdict


SPLIT_RE = re.compile(
    r"^split\s*#(?P<idx>\d+)\s*\((?P<backend>[^)]+)\).*?n_inputs\s*=\s*(?P<in>\d+).*?n_outputs\s*=\s*(?P<out>\d+)"
)
NODE_RE = re.compile(
    r"^node\s*#(?P<idx>\d+).*?CUR\s*=\s*(?P<op>GGML_OP_[A-Z_0-9]+)"
)
# Alternative split format (some versions):
SPLIT2_RE = re.compile(r"split\s*(?P<idx>\d+):\s*(?P<backend>\w+)")


def parse(text):
    splits = []
    nodes_by_backend = defaultdict(lambda: defaultdict(int))
    current_backend = None

    for line in text.splitlines():
        m = SPLIT_RE.match(line)
        if m:
            current_backend = m["backend"].strip()
            splits.append({
                "idx": int(m["idx"]),
                "backend": current_backend,
                "n_inputs": int(m["in"]),
                "n_outputs": int(m["out"]),
            })
            continue
        m = NODE_RE.search(line)
        if m and current_backend:
            nodes_by_backend[current_backend][m["op"]] += 1
            continue

    out = {
        "n_splits": len(splits),
        "splits": splits,
        "splits_by_backend": {},
        "ops_by_backend": {b: dict(ops) for b, ops in nodes_by_backend.items()},
        "op_counts_total": defaultdict(int),
    }
    for s in splits:
        out["splits_by_backend"].setdefault(s["backend"], 0)
        out["splits_by_backend"][s["backend"]] += 1
    for ops in nodes_by_backend.values():
        for op, cnt in ops.items():
            out["op_counts_total"][op] += cnt
    out["op_counts_total"] = dict(out["op_counts_total"])
    return out


def main():
    if len(sys.argv) < 2:
        print("usage: parse_sched_debug.py <stderr file>", file=sys.stderr)
        sys.exit(2)
    text = open(sys.argv[1]).read()
    summary = parse(text)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
