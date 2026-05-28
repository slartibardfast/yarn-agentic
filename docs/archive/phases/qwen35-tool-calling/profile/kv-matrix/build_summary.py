#!/usr/bin/env python3
"""Build the KV-V quant matrix summary.

Reads every kv-<type>-<TS>.* triple in the current directory, extracts
K/V buffer sizes from the server stderr, median throughput from the
drive output, output fingerprint, and the perf-logger totals. Emits a
matrix JSON + markdown table.

Usage:
    cd profile/kv-matrix/
    python3 build_summary.py 2026-04-11T231000Z
"""
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).parent

V_TYPES = ["f16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "iq4_nl", "tq_v_4b"]

KV_SIZE_RE = re.compile(
    r"K \((?P<kname>\w+)\):\s*(?P<ksize>[\d.]+) MiB, V \((?P<vname>\w+)\):\s*(?P<vsize>[\d.]+) MiB"
)
GRAPH_SPLITS_RE = re.compile(r"graph splits = (\d+)")


def parse_startup(stderr_path):
    """Extract K/V sizes and graph splits from server stderr."""
    text = pathlib.Path(stderr_path).read_text()
    ksize = None
    vsize = None
    kname = None
    vname = None
    graph_splits = None
    m = KV_SIZE_RE.search(text)
    if m:
        ksize = float(m["ksize"])
        vsize = float(m["vsize"])
        kname = m["kname"]
        vname = m["vname"]
    m = GRAPH_SPLITS_RE.search(text)
    if m:
        graph_splits = int(m.group(1))
    return {
        "k_name": kname,
        "k_size_mib": ksize,
        "v_name": vname,
        "v_size_mib": vsize,
        "graph_splits": graph_splits,
    }


def parse_drive(drive_txt_path):
    text = pathlib.Path(drive_txt_path).read_text()
    start = text.find("{")
    if start < 0:
        return None
    return json.loads(text[start:])


def parse_perf(perf_json_path):
    return json.load(open(perf_json_path))


def main():
    ts = sys.argv[1] if len(sys.argv) > 1 else None
    rows = []
    reference_fingerprint = None
    for V in V_TYPES:
        pattern = f"kv-{V}-*.stderr" if ts is None else f"kv-{V}-{ts}.stderr"
        cands = sorted(HERE.glob(pattern))
        if not cands:
            rows.append({"v_type": V, "error": "no stderr found"})
            continue
        stem = cands[-1].with_suffix("")
        startup = parse_startup(str(stem) + ".stderr")
        drv = parse_drive(str(stem) + ".drive.txt")
        try:
            perf = parse_perf(str(stem) + ".perf.json")
        except Exception:
            perf = None
        if drv is None:
            rows.append({"v_type": V, **startup, "error": "no drive output"})
            continue

        fingerprint = None
        if drv["runs"]:
            fingerprint = drv["runs"][0]["content_head"][:120]
            if V == "f16":
                reference_fingerprint = fingerprint

        diff_from_ref_pct = None
        if reference_fingerprint and fingerprint:
            # Count chars that differ in the overlapping prefix.
            n = min(len(reference_fingerprint), len(fingerprint))
            if n > 0:
                diff = sum(
                    1 for a, b in zip(reference_fingerprint[:n], fingerprint[:n]) if a != b
                )
                diff_from_ref_pct = round(100.0 * diff / n, 2)

        top_ops = None
        if perf:
            top_ops = [
                {
                    "op": r["op"][:60],
                    "share_pct": r["share_pct"],
                    "total_ms": round(r["total_us"] / 1000, 2),
                }
                for r in perf["ops"][:10]
            ]

        agg = drv["aggregate"]
        rows.append({
            "v_type": V,
            **startup,
            "prompt_t_s_median": agg["prompt_per_second_median"],
            "predict_t_s_median": agg["predicted_per_second_median"],
            "mtp_acceptance_rate": agg["mtp_acceptance_rate"],
            "fingerprint_head": fingerprint,
            "fingerprint_diff_vs_f16_pct": diff_from_ref_pct,
            "perf_total_ms": round(perf["ops_total_us_sum"] / 1000, 1) if perf else None,
            "top_ops": top_ops,
        })

    print(json.dumps({"ts": ts, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
