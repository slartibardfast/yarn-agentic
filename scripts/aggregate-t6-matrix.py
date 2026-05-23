#!/usr/bin/env python3
# T6.1 matrix aggregator — reads every data/<matrix-dir>/cell-*/cell.json,
# produces SUMMARY.md and SUMMARY.json with:
#   - per-cell line: id, config-axes (dflash/hadamard/defrag), wall, t/s, status
#   - per-feature delta line vs a chosen baseline cell
#   - honest "broken cell" treatment — cells where status_counts has non-200
#     entries are flagged separately and excluded from numeric delta tables
#
# Usage:
#   scripts/aggregate-t6-matrix.py data/t6.1-matrix-<timestamp>/
import json
import sys
from pathlib import Path

def load_cells(matrix_dir):
    cells = []
    for d in sorted(Path(matrix_dir).glob("cell-*")):
        cj = d / "cell.json"
        if not cj.is_file():
            continue
        with open(cj) as f:
            cells.append(json.load(f))
    return cells

def status_breakdown(cell):
    per = (cell.get("results") or {}).get("per_request") or []
    counts = {}
    for r in per:
        counts[r.get("status")] = counts.get(r.get("status"), 0) + 1
    return counts

def is_clean(cell):
    counts = status_breakdown(cell)
    n = sum(counts.values())
    n_200 = counts.get(200, 0)
    return n_200 == n and n > 0

def fmt_row(cell):
    cfg = cell.get("config") or {}
    res = cell.get("results") or {}
    counts = status_breakdown(cell)
    return {
        "cell_id":  cell.get("cell_id"),
        "dflash":   cfg.get("dflash"),
        "hadamard": cfg.get("k_cache_hadamard") and cfg.get("v_cache_hadamard"),
        "defrag":   cfg.get("defrag_thold"),
        "wall_s":   res.get("wall_secs"),
        "out_toks": res.get("total_output_toks"),
        "tps_agg":  res.get("tok_per_sec_aggregate"),
        "tps_slot": res.get("tok_per_sec_per_slot_mean"),
        "status":   counts,
        "clean":    is_clean(cell),
    }

def md_table(rows, headers):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)

def main():
    if len(sys.argv) < 2:
        print("usage: aggregate-t6-matrix.py <matrix-dir>")
        sys.exit(2)
    matrix_dir = Path(sys.argv[1])
    if not matrix_dir.is_dir():
        print(f"FAIL: not a dir: {matrix_dir}", file=sys.stderr)
        sys.exit(2)

    cells = load_cells(matrix_dir)
    summaries = [fmt_row(c) for c in cells]

    # JSON output (full)
    json_path = matrix_dir / "SUMMARY.json"
    with open(json_path, "w") as f:
        json.dump({"cells": summaries}, f, indent=2)

    # MD output
    lines = []
    lines.append(f"# T6.1 binary ablation matrix — summary")
    lines.append("")
    lines.append(f"Matrix dir: `{matrix_dir.name}`")
    lines.append(f"Cells: {len(cells)} ({sum(1 for s in summaries if s['clean'])} clean, {sum(1 for s in summaries if not s['clean'])} broken)")
    lines.append("")
    lines.append("Workload: gate0 reference, 8 prompts × 256 max_tokens, ignore_eos, fire_pattern=concurrent. Server: --parallel 2 (queue depth 6).")
    lines.append("")
    lines.append("## Per-cell results")
    lines.append("")
    rows = [
        (s["cell_id"], s["dflash"], s["hadamard"], s["defrag"],
         f"{s['wall_s']:.1f}" if s["wall_s"] else "-",
         s["out_toks"] or "-",
         f"{s['tps_agg']:.2f}" if s["tps_agg"] else "-",
         f"{s['tps_slot']:.2f}" if s["tps_slot"] else "-",
         str(s["status"]),
         "✓" if s["clean"] else "✗")
        for s in summaries
    ]
    lines.append(md_table(rows, ["cell_id", "dflash", "hadamard", "defrag", "wall_s", "out_toks", "t/s_agg", "t/s_slot", "status_counts", "clean"]))
    lines.append("")

    # Pick "no-defrag" as the safe baseline (DFlash + Had + defrag OFF) since
    # the defrag-on production default crashed under DFlash multi-slot.
    baseline = next((s for s in summaries if s["cell_id"] == "no-defrag" and s["clean"]), None)

    if baseline:
        lines.append("## Per-feature delta")
        lines.append("")
        lines.append(f"**Baseline:** `{baseline['cell_id']}` (DFlash ON, Hadamard ON, defrag OFF) = {baseline['tps_agg']:.2f} t/s aggregate. The defrag-OFF baseline is used because the production default (defrag 0.1) crashed in two of the four T6.1 cells when combined with DFlash multi-slot — see Findings.")
        lines.append("")
        b = baseline["tps_agg"]
        delta_rows = []
        for s in summaries:
            if not s["clean"] or s["cell_id"] == baseline["cell_id"]:
                continue
            cmp_tps = s["tps_agg"]
            abs_d = cmp_tps - b
            rel_d = (cmp_tps - b) / b * 100 if b else 0
            delta_rows.append((s["cell_id"], f"{cmp_tps:.2f}", f"{abs_d:+.2f}", f"{rel_d:+.1f}%"))
        lines.append(md_table(delta_rows, ["cell_id", "t/s_agg", "Δ vs baseline (t/s)", "Δ (%)"]))
        lines.append("")

    # Per-feature verdicts (compare matched pairs)
    pairs = [
        ("DFlash",  "no-defrag",        "no-dflash-nodefrag"),
        ("Hadamard","no-defrag",        "no-hadamard-nodefrag"),
        ("defrag",  "no-defrag",        "prod-baseline"),
    ]
    by_id = {s["cell_id"]: s for s in summaries}
    lines.append("## Per-feature verdict (binary on/off pairs)")
    lines.append("")
    for label, on_id, off_id in pairs:
        on, off = by_id.get(on_id), by_id.get(off_id)
        if not on or not off:
            lines.append(f"- **{label}:** comparison cells missing.")
            continue
        if not on["clean"] and not off["clean"]:
            lines.append(f"- **{label}:** both cells broken; no signal.")
            continue
        if not on["clean"]:
            lines.append(f"- **{label}:** ON cell `{on_id}` broken ({on['status']}); cannot attribute.")
            continue
        if not off["clean"]:
            # Treat the broken cell as "feature ON crashes" — itself a verdict.
            lines.append(f"- **{label} (ON vs OFF):** ON cell `{off_id}` CRASHED ({off['status']}). Feature is unsafe at this workload.")
            continue
        d = on["tps_agg"] - off["tps_agg"]
        rel = d / off["tps_agg"] * 100
        verdict = "net-positive" if d > 0.05 * off["tps_agg"] else ("net-negative" if d < -0.05 * off["tps_agg"] else "no-op (within noise)")
        lines.append(f"- **{label}:** ON ({on['cell_id']}) {on['tps_agg']:.2f} t/s vs OFF ({off['cell_id']}) {off['tps_agg']:.2f} t/s — Δ {d:+.2f} t/s ({rel:+.1f}%) — **{verdict}**.")
    lines.append("")

    out_md = matrix_dir / "SUMMARY.md"
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_md}")
    print(f"wrote {json_path}")

if __name__ == "__main__":
    main()
