#!/usr/bin/env python3
"""Build the VRAM budget summary from vram-*-ctx*-<TS>.stderr files.

Parses the llama_memory_breakdown_print output and the model load
lines to extract: free VRAM at startup, model buffer size, K/V buffer
size, compute buffer size, OOM / success state, graph splits.

Usage:
    cd profile/vram-budget/
    python3 build_summary.py [TS]
"""
import json
import pathlib
import re
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).parent

V_LIST = ["f16", "q4_0", "tq_v_4b"]
CTX_LIST = [4096, 8192, 12288, 16384, 24576, 32768]

FREE_RE = re.compile(r"(\d+)\s*MiB free")
MODEL_BUF_RE = re.compile(r"Vulkan0 model buffer size\s*=\s*([\d.]+)\s*MiB")
KV_SIZE_RE = re.compile(
    r"size\s*=\s*([\d.]+) MiB.*?K \((\w+)\):\s*([\d.]+) MiB, V \((\w+)\):\s*([\d.]+) MiB"
)
COMPUTE_BUF_RE = re.compile(r"compute buffer size\s*=\s*([\d.]+)\s*MiB")
FAILED_RE = re.compile(r"failed to fit", re.IGNORECASE)
READY_RE = re.compile(r"all slots are idle")
GRAPH_SPLITS_RE = re.compile(r"graph splits = (\d+)")
MEM_BREAKDOWN_RE = re.compile(
    r"Vulkan0[^|]*\|\s*(\d+)\s*=\s*(\d+)\s*\+\s*\(\s*(\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\s*\)\s*\+\s*(\d+)"
)


def parse_stderr(path):
    text = pathlib.Path(path).read_text()
    d = {
        "free_mib": None,
        "model_buf_mib": None,
        "kv_mib": None,
        "k_name": None,
        "k_mib": None,
        "v_name": None,
        "v_mib": None,
        "compute_buf_mib": None,
        "graph_splits": None,
        "ready": False,
        "failed_to_fit": False,
        "mem_breakdown_total": None,
        "mem_breakdown_unaccounted": None,
    }
    m = FREE_RE.search(text)
    if m:
        d["free_mib"] = int(m.group(1))
    m = MODEL_BUF_RE.search(text)
    if m:
        d["model_buf_mib"] = float(m.group(1))
    m = KV_SIZE_RE.search(text)
    if m:
        d["kv_mib"] = float(m.group(1))
        d["k_name"] = m.group(2)
        d["k_mib"] = float(m.group(3))
        d["v_name"] = m.group(4)
        d["v_mib"] = float(m.group(5))
    m = COMPUTE_BUF_RE.search(text)
    if m:
        d["compute_buf_mib"] = float(m.group(1))
    m = GRAPH_SPLITS_RE.search(text)
    if m:
        d["graph_splits"] = int(m.group(1))
    d["ready"] = bool(READY_RE.search(text))
    d["failed_to_fit"] = bool(FAILED_RE.search(text))
    m = MEM_BREAKDOWN_RE.search(text)
    if m:
        d["mem_breakdown_total"] = int(m.group(1))
        d["mem_breakdown_unaccounted"] = int(m.group(7))
    return d


def main():
    ts = sys.argv[1] if len(sys.argv) > 1 else None
    rows = []
    for V in V_LIST:
        for CTX in CTX_LIST:
            pat = f"vram-{V}-ctx{CTX}-*.stderr" if ts is None else f"vram-{V}-ctx{CTX}-{ts}.stderr"
            cands = sorted(HERE.glob(pat))
            if not cands:
                rows.append({"v": V, "ctx": CTX, "status": "missing"})
                continue
            st = parse_stderr(cands[-1])
            status = (
                "ok" if st["ready"]
                else "oom" if st["failed_to_fit"]
                else "other-fail"
            )
            rows.append({"v": V, "ctx": CTX, "status": status, **st})
    out = {"ts": ts, "rows": rows}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
