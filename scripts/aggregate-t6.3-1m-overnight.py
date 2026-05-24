#!/usr/bin/env python3
"""
T6.3 1M overnight aggregator — synthesises the 7-phase test output into
SUMMARY.md. Reads phase{N}-*.json (per-request) + vram-*.csv (timeline).

Usage:
  scripts/aggregate-t6.3-1m-overnight.py <OUTDIR>
"""

import csv
import glob
import json
import os
import re
import sys


def load_phase(outdir, phase):
    rows = []
    for p in sorted(glob.glob(os.path.join(outdir, f"phase{phase}-*.json"))):
        try:
            with open(p, "r") as f:
                r = json.load(f)
            r["_label"] = os.path.basename(p).removesuffix(".json")
            rows.append(r)
        except Exception:
            pass
    return rows


def vram_timeline(outdir):
    files = sorted(glob.glob(os.path.join(outdir, "vram-*.csv")),
                   key=lambda p: int(re.search(r"vram-(\d+)\.csv", p).group(1)))
    timeline = []
    for p in files:
        ts = int(re.search(r"vram-(\d+)\.csv", p).group(1))
        try:
            with open(p, "r") as f:
                rdr = csv.reader(f)
                gpu_mibs = []
                for row in rdr:
                    if len(row) >= 2:
                        used = row[1].strip().split()[0]
                        gpu_mibs.append(int(used))
            timeline.append({"ts": ts, "gpu0_mib": gpu_mibs[0] if gpu_mibs else None,
                             "gpu1_mib": gpu_mibs[1] if len(gpu_mibs) > 1 else None})
        except Exception:
            pass
    return timeline


def fmt_secs(s):
    if s is None:
        return "?"
    if s < 60:
        return f"{s:.1f}s"
    return f"{s/60:.1f}m"


def main():
    if len(sys.argv) != 2:
        print("usage: aggregate-t6.3-1m-overnight.py <OUTDIR>", file=sys.stderr)
        sys.exit(2)
    outdir = sys.argv[1]

    phases = {p: load_phase(outdir, p) for p in (1, 2, 3, 4, 5, 6)}
    vram = vram_timeline(outdir)

    lines = []
    lines += [
        "# T6.3 1M Overnight Test — `qwen36-27b-x1-yarn-1m-mtp.sh`",
        "",
        f"OUTDIR: `{outdir}`",
        "",
        "## Phase 1 — Boot smoke",
        "",
    ]
    for r in phases[1]:
        if "error" in r:
            lines.append(f"- **FAIL**: {r['error']}")
        else:
            lines.append(f"- wall={fmt_secs(r['wall_s'])} ttft={fmt_secs(r['ttft_s'])} "
                         f"decode={fmt_secs(r['decode_s'])} n={r['n_predicted']}")

    lines += [
        "",
        "## Phase 2 — Cold 1M prefill",
        "",
        "| label | wall | ttft (= prefill) | decode | n_tokens |",
        "|---|---|---|---|---|",
    ]
    for r in phases[2]:
        if "error" in r:
            lines.append(f"| {r.get('_label')} | FAIL | — | — | — |")
            continue
        lines.append(f"| {r['_label']} | {fmt_secs(r['wall_s'])} | "
                     f"**{fmt_secs(r['ttft_s'])}** | {fmt_secs(r['decode_s'])} | "
                     f"{r['n_predicted']} |")

    # ---- Phase 3: cache validation ----
    lines += ["", "## Phase 3 — Cache validation (shared 853K prefix)", ""]
    lines += [
        "| label | ttft (prefill) | speedup vs cold | wall | n |",
        "|---|---|---|---|---|",
    ]
    cold_ttft = phases[2][0]["ttft_s"] if (phases[2] and "ttft_s" in phases[2][0]) else None
    for r in phases[3]:
        if "error" in r:
            lines.append(f"| {r.get('_label')} | FAIL | — | — | — |")
            continue
        speedup = f"{cold_ttft / r['ttft_s']:.1f}×" if (cold_ttft and r['ttft_s']) else "?"
        lines.append(f"| {r['_label']} | {fmt_secs(r['ttft_s'])} | "
                     f"{speedup} | {fmt_secs(r['wall_s'])} | {r['n_predicted']} |")

    if phases[3] and cold_ttft:
        max_speedup = max((cold_ttft / r['ttft_s'] for r in phases[3]
                           if r.get('ttft_s') and r['ttft_s'] > 0), default=None)
        verdict = "PASS (≥10×)" if max_speedup and max_speedup >= 10 else \
                  ("PARTIAL" if max_speedup and max_speedup > 1.5 else "FAIL")
        lines += ["", f"**Cache effectiveness**: best speedup {max_speedup:.1f}× — **{verdict}**"]

    # ---- Phase 4: NIAH ----
    lines += ["", "## Phase 4 — Needle-in-Haystack depth sweep", ""]
    lines += [
        "| depth | wall | ttft | first_tokens (needle expected: BRAVO-LIMA-7-EAGLE) |",
        "|---|---|---|---|",
    ]
    niah_hits = 0
    niah_total = 0
    for r in phases[4]:
        if "error" in r:
            lines.append(f"| {r.get('_label')} | FAIL | — | — |")
            continue
        ft = r.get('first_tokens', '')
        ft_clean = re.sub(r'\s+', ' ', ft)[:80]
        hit = "BRAVO" in ft and "LIMA" in ft and "EAGLE" in ft
        if hit:
            niah_hits += 1
        niah_total += 1
        marker = "✓" if hit else "✗"
        lines.append(f"| {r['_label']} | {fmt_secs(r['wall_s'])} | "
                     f"{fmt_secs(r['ttft_s'])} | {marker} {ft_clean} |")
    if niah_total:
        lines += ["", f"**NIAH accuracy**: {niah_hits}/{niah_total} = "
                  f"{100*niah_hits/niah_total:.0f}%"]

    # ---- Phase 5: W&P QA ----
    lines += ["", "## Phase 5 — War and Peace qualitative QA", ""]
    for r in phases[5]:
        if "error" in r:
            lines.append(f"- **FAIL**: {r['error']}")
            continue
        lines.append(f"- **{r['_label']}**: wall={fmt_secs(r['wall_s'])} "
                     f"ttft={fmt_secs(r['ttft_s'])} n={r['n_predicted']}")
        lines.append(f"  - first 200: `{r.get('first_tokens', '')[:200].strip()}`")

    # ---- Phase 6: soak ----
    lines += ["", "## Phase 6 — Stability soak", ""]
    soak_wall_total = 0.0
    soak_ttfts = []
    soak_errs = 0
    for r in phases[6]:
        if "error" in r:
            soak_errs += 1
            continue
        soak_wall_total += r.get('wall_s') or 0
        if r.get('ttft_s'):
            soak_ttfts.append(r['ttft_s'])
    lines += [
        f"- iterations: {len(phases[6])}",
        f"- failures:   {soak_errs}",
        f"- soak wall:  {soak_wall_total/3600:.2f} hr",
        f"- ttft median: {sorted(soak_ttfts)[len(soak_ttfts)//2]:.1f}s" if soak_ttfts else "- (no ttft data)",
    ]

    # ---- VRAM ----
    lines += ["", "## VRAM timeline", ""]
    if vram:
        gpu0_vals = [v['gpu0_mib'] for v in vram if v['gpu0_mib']]
        gpu1_vals = [v['gpu1_mib'] for v in vram if v['gpu1_mib']]
        lines += [
            f"- samples: {len(vram)}",
            f"- GPU0 min/max/last: {min(gpu0_vals)}/{max(gpu0_vals)}/{gpu0_vals[-1]} MiB" if gpu0_vals else "",
            f"- GPU1 min/max/last: {min(gpu1_vals)}/{max(gpu1_vals)}/{gpu1_vals[-1]} MiB" if gpu1_vals else "",
        ]
        if gpu0_vals and gpu1_vals:
            growth = (gpu0_vals[-1] + gpu1_vals[-1]) - (gpu0_vals[0] + gpu1_vals[0])
            pct = 100 * growth / (gpu0_vals[0] + gpu1_vals[0]) if (gpu0_vals[0] + gpu1_vals[0]) else 0
            lines.append(f"- net growth: {growth:+d} MiB ({pct:+.1f}%)")
            lines.append(f"- **VRAM stability**: {'PASS (<2%)' if abs(pct) < 2 else 'WATCH'}")

    # ---- Verdicts ----
    lines += ["", "## Combined verdicts", ""]
    n_failed_phases = sum(1 for p in (1,2,3,4,5,6) if any('error' in r for r in phases[p]))
    lines.append(f"- phases with at least one failure: {n_failed_phases}/6")
    lines.append("- See per-phase details above; final go/no-go for production swap is downstream.")

    out = os.path.join(outdir, "SUMMARY.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
