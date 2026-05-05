#!/usr/bin/env python3
"""Decision-criteria evaluator for CUDA graph cache probe dumps.

Walks one or more probe dump directories (each containing
cuda<N>-<probe>.jsonl files emitted by ggml-cuda's cuda_graph_probe
infrastructure), computes per-probe summary statistics, and — when
invoked with --gate <name> — checks whether the corresponding decision
criteria pass.

Usage:
    parse-probe-dump.py /path/to/dump-dir [/path/to/dump-dir2 ...]
    parse-probe-dump.py --gate instrumentation /path/to/dump-dir

Gates:
    instrumentation  — evaluates the post-instrumentation decision
                        criteria: update success rate, distinct
                        topology-class count, update host-latency P95,
                        per-entry VRAM cost, destroy free-delta. Exits
                        0 on PASS, 1 on FAIL, 2 on ABORT (the
                        "instrumentation refutes the cache as the
                        primary attribution" branch).

Decision criteria thresholds match PHASE35-GRAPH-CACHE-REDESIGN.md §4.2.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


def load_jsonl(path: Path):
    """Yield parsed records from one .jsonl file. Skips malformed lines
    with a warning to stderr (rather than aborting) so a partial dump
    is still readable."""
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(
                    f"warning: malformed JSONL at {path}:{lineno}: {e}",
                    file=sys.stderr,
                )


def percentile(values, q):
    """Return the q-th percentile of values (q in 0..100). Linear interp
    between the two nearest ranks. Empty sequence returns None."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (q / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def collect(dirs):
    """Return aggregate stats across all dump dirs."""
    stats = {
        "n_records": 0,
        "by_probe": Counter(),
        "by_event": Counter(),
        "hit_counter": {
            "n_entries": 0,
            "topology_keys": set(),
            "shape_keys": set(),
            "hits_total_by_entry": [],
        },
        "timing": defaultdict(list),         # event -> [duration_us]
        "vram_delta": {
            "insert": [],                    # delta_bytes per insert
            "destroy": [],                   # delta_bytes per destroy
        },
        "update_failures": 0,
        "disable_too_many": 0,
    }
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            print(f"warning: not a directory: {d}", file=sys.stderr)
            continue
        for jsonl in d.glob("*.jsonl"):
            for rec in load_jsonl(jsonl):
                stats["n_records"] += 1
                probe = rec.get("probe", "")
                stats["by_probe"][probe] += 1
                if probe == "hit_counter":
                    stats["hit_counter"]["n_entries"] += 1
                    if "topology_key" in rec:
                        stats["hit_counter"]["topology_keys"].add(
                            rec["topology_key"]
                        )
                    if "shape_key" in rec:
                        stats["hit_counter"]["shape_keys"].add(rec["shape_key"])
                    if "hits_total" in rec:
                        stats["hit_counter"]["hits_total_by_entry"].append(
                            int(rec["hits_total"])
                        )
                elif probe == "timing":
                    ev = rec.get("event", "?")
                    stats["by_event"][ev] += 1
                    if "duration_us" in rec:
                        stats["timing"][ev].append(float(rec["duration_us"]))
                elif probe == "vram_delta":
                    ev = rec.get("event", "?")
                    if ev in stats["vram_delta"] and "delta_bytes" in rec:
                        stats["vram_delta"][ev].append(int(rec["delta_bytes"]))
                elif probe == "update_failures":
                    stats["update_failures"] += 1
                elif probe == "disable_too_many":
                    stats["disable_too_many"] += 1
    return stats


def render_summary(stats):
    print(f"records:          {stats['n_records']}")
    print("by_probe:")
    for k, v in stats["by_probe"].most_common():
        print(f"  {k:18s} {v}")
    print("by_event (timing):")
    for k, v in stats["by_event"].most_common():
        print(f"  {k:18s} {v}")

    hc = stats["hit_counter"]
    print("hit_counter:")
    print(f"  entries:           {hc['n_entries']}")
    print(f"  distinct topology: {len(hc['topology_keys'])}")
    print(f"  distinct shape:    {len(hc['shape_keys'])}")
    if hc["hits_total_by_entry"]:
        print(f"  hits/entry mean:   {mean(hc['hits_total_by_entry']):.2f}")
        print(f"  hits/entry max:    {max(hc['hits_total_by_entry'])}")

    print("timing (us):")
    for ev, vs in sorted(stats["timing"].items()):
        if vs:
            p95 = percentile(vs, 95)
            print(
                f"  {ev:18s} n={len(vs):<6d} "
                f"med={median(vs):.2f} mean={mean(vs):.2f} "
                f"p95={p95:.2f}"
            )

    for ev, vs in stats["vram_delta"].items():
        if vs:
            nonzero = sum(1 for v in vs if v != 0)
            print(
                f"vram_delta {ev:7s}: n={len(vs):<6d} "
                f"mean={mean(vs):.0f} bytes  nonzero={nonzero}/{len(vs)}"
            )

    print(f"update_failures:  {stats['update_failures']}")
    print(f"disable_too_many: {stats['disable_too_many']}")


def gate_instrumentation(stats):
    """Evaluate the post-instrumentation decision criteria.

    Thresholds per PHASE35-GRAPH-CACHE-REDESIGN.md §4.2:
      D1 update success rate >= 95%
      D2 distinct topology classes <= 10
      D3 update P95 latency < 100us  (informational only; affects later phase)
      D4 per-entry VRAM cost mean    (informational, no threshold)
      D5 destroy delta_bytes mean > 0
      D-ABORT  D1 < 80% AND D2 > 50  -> "not the graph cache" pivot

    Returns one of: "PASS", "FAIL", "ABORT".
    """
    hc = stats["hit_counter"]
    n_topo = len(hc["topology_keys"])

    n_updates = stats["by_event"].get("update", 0)
    n_update_fail = stats["update_failures"]
    update_success_rate = (
        (n_updates - n_update_fail) / n_updates if n_updates else None
    )

    update_p95 = percentile(stats["timing"].get("update", []), 95)
    insert_mean = (
        mean(stats["vram_delta"]["insert"])
        if stats["vram_delta"]["insert"]
        else None
    )
    destroy_mean = (
        mean(stats["vram_delta"]["destroy"])
        if stats["vram_delta"]["destroy"]
        else None
    )

    print()
    print("=== gate: instrumentation ===")

    def fmt(x, suffix=""):
        return f"{x}{suffix}" if x is not None else "n/a"

    print(
        f"D1 update success rate: "
        f"{fmt(update_success_rate * 100 if update_success_rate is not None else None, '%')} "
        f"(threshold >= 95%)"
    )
    print(f"D2 distinct topology classes: {n_topo} (threshold <= 10)")
    print(f"D3 update P95 latency:        {fmt(update_p95, ' us')} (informational; <100us advisory)")
    print(f"D4 per-entry VRAM cost mean:  {fmt(insert_mean, ' bytes')} (informational)")
    print(f"D5 destroy delta_bytes mean:  {fmt(destroy_mean, ' bytes')} (threshold > 0)")

    abort = (
        update_success_rate is not None
        and update_success_rate < 0.80
        and n_topo > 50
    )
    if abort:
        print(
            "ABORT: update success < 80% AND topology classes > 50 — "
            "graph-cache attribution refuted; pivot to cuda_pool / cuBLAS workspace probes."
        )
        return "ABORT"

    fails = []
    if update_success_rate is not None and update_success_rate < 0.95:
        fails.append("D1")
    if n_topo > 10:
        fails.append("D2")
    if destroy_mean is not None and destroy_mean <= 0:
        fails.append("D5")
    if fails:
        print(f"FAIL: criteria not met: {', '.join(fails)}")
        return "FAIL"

    print("PASS: all binding criteria met.")
    return "PASS"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("dirs", nargs="+", help="probe dump directories")
    p.add_argument(
        "--gate",
        choices=["instrumentation"],
        help="evaluate decision criteria for the named gate",
    )
    args = p.parse_args(argv)

    stats = collect(args.dirs)
    if stats["n_records"] == 0:
        print("error: no records found in supplied directories", file=sys.stderr)
        return 2

    render_summary(stats)
    if args.gate:
        verdict = gate_instrumentation(stats) if args.gate == "instrumentation" else None
        if verdict == "PASS":
            return 0
        if verdict == "FAIL":
            return 1
        if verdict == "ABORT":
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
