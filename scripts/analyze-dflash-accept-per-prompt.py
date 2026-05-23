#!/usr/bin/env python3
"""
T6.3 axis 1 — per-prompt DFlash acceptance analyzer (offline).

Parses an ik_llama server.log produced by the cross-engine bench cell and
derives per-task acceptance rates by diffing the slot's cumulative
'draft acceptance rate' counters across task completions.

Usage:
  scripts/analyze-dflash-accept-per-prompt.py <cell-dir>

Where <cell-dir> contains server.log + responses.json (the bench client
output). Emits:
  <cell-dir>/per-prompt-acceptance.json
  <cell-dir>/per-prompt-acceptance.md

Methodology:
- The server prints "draft acceptance rate = X (A accepted / G generated)"
  before each "slot released" event. (A, G) is PER-TASK, NOT cumulative;
  verified by inspecting the matrix prod-baseline log where slot 1 emits
  180 → 168 → 158 → 194 across four completions (non-monotone, hence
  per-task).
- Slot launches pair with task ids via "launch_slot_with_task ...
  id_slot=N id_task=T". We don't strictly need them for accept attribution
  since rate lines are per-task — they serve to label slot+task on the
  release event.
- Multiple acceptance lines can appear BEFORE the next release event
  (two slots completing within the same scheduling tick) — we queue
  pending accepts FIFO and pop them on each release.
- Prompt index ← completion order via latency sort against responses.json
  (smaller latency = earlier completion).
"""

import json
import os
import re
import sys
from collections import defaultdict

LAUNCH_RE = re.compile(r"launch_slot_with_task.*id_slot=(\d+) id_task=(\d+)")
RELEASE_RE = re.compile(r"release_slots.*id_slot=(\d+) id_task=(\d+) n_ctx=\d+ n_past=(\d+)")
ACCEPT_RE = re.compile(r"draft acceptance rate = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated\)")
STAT_RE = re.compile(
    r"statistics dflash: #calls\(b,g,a\) = (\d+) (\d+) (\d+), #gen drafts = (\d+), "
    r"#acc drafts = (\d+), #gen tokens = (\d+), #acc tokens = (\d+), "
    r"dur\(b,g,a\) = ([\d.]+), ([\d.]+), ([\d.]+) ms"
)


def parse_log(path):
    """Walk the log. Each task completion emits an ACCEPT line followed by
    an STAT line (optional) and then a RELEASE line. Multiple completions
    can interleave; queue ACCEPT/STAT in FIFO order and pop on RELEASE.

    Returns list of dicts in temporal release order, with PER-TASK accept
    counters (no diffing — rate lines are already per-task)."""
    out = []
    accept_q = []   # list of (accept, gen)
    stat_q = []     # list of dflash stat dicts

    with open(path, "r", errors="replace") as f:
        for line in f:
            m = ACCEPT_RE.search(line)
            if m:
                accept_q.append((int(m.group(2)), int(m.group(3))))
                continue
            m = STAT_RE.search(line)
            if m:
                stat_q.append({
                    "calls_bind": int(m.group(1)),
                    "calls_gen": int(m.group(2)),
                    "calls_accept": int(m.group(3)),
                    "gen_drafts": int(m.group(4)),
                    "acc_drafts": int(m.group(5)),
                    "gen_tokens": int(m.group(6)),
                    "acc_tokens": int(m.group(7)),
                    "dur_bind_ms": float(m.group(8)),
                    "dur_gen_ms": float(m.group(9)),
                    "dur_accept_ms": float(m.group(10)),
                })
                continue
            m = RELEASE_RE.search(line)
            if m:
                slot, task, n_past = int(m.group(1)), int(m.group(2)), int(m.group(3))
                acc, gen = accept_q.pop(0) if accept_q else (None, None)
                # dflash stat lines are cumulative across the server's
                # lifetime, not per-task. Capture but mark as such.
                cum_stat = stat_q.pop(0) if stat_q else None
                out.append({
                    "slot": slot,
                    "task": task,
                    "n_past_at_release": n_past,
                    "accept_count": acc,
                    "gen_count": gen,
                    "acceptance_rate": (acc / gen) if (gen and gen > 0) else None,
                    "dflash_cum_stat": cum_stat,
                })
    return out


# Gate0 reference prompts (in order) — mirror scripts/cross-engine-bench.sh.
PROMPTS = [
    "Explain the difference between latent diffusion and pixel-space diffusion in two sentences.",
    "Summarize the plot of King Lear in one paragraph.",
    "Write Python code that fits a 2nd-degree polynomial to a list of (x, y) pairs.",
    "What are the main causes of the Peloponnesian War?",
    "Translate to French: The early-morning fog lingered over the harbour until the trawlers cut through it.",
    "List five practical steps for reducing memory allocations in a hot inner loop in Rust.",
    "Describe the role of telomeres in cellular aging.",
    "Write a haiku about a printing press.",
]


def short_label(p):
    return p[:60].rsplit(" ", 1)[0] + "..."


def main():
    if len(sys.argv) != 2:
        print("usage: analyze-dflash-accept-per-prompt.py <cell-dir>", file=sys.stderr)
        sys.exit(2)
    cell_dir = sys.argv[1]
    log_path = os.path.join(cell_dir, "server.log")
    resp_path = os.path.join(cell_dir, "responses.json")
    if not os.path.isfile(log_path):
        print(f"FAIL: not a file: {log_path}", file=sys.stderr)
        sys.exit(2)

    per_task = parse_log(log_path)

    # Match completion (release) order ↔ client-side bench prompt index by
    # latency: ik_llama processes prompts in arrival order, but with 2
    # slots + 8 prompts the completion order is NOT prompt-index order.
    # The bench responses.json has per-request idx + latency_s. Sort by
    # latency and assume completion order matches sorted latency order
    # (smaller latency = earlier completion).
    resp = None
    if os.path.isfile(resp_path):
        with open(resp_path, "r") as f:
            resp = json.load(f)

    # Build prompt_idx ← completion order map using latency sort.
    prompt_map = {}
    if resp:
        per_req = sorted(resp["per_request"], key=lambda r: r["latency_s"])
        # per_task entries (excluding warmup) align with per_req in completion order.
        # The warmup is a single 32-token request; identify it as the first
        # entry with very low n_past_at_release.
        task_idx = 0
        warmup_seen = False
        for pt in per_task:
            if not warmup_seen and pt["n_past_at_release"] < 64:
                pt["prompt_idx"] = -1  # warmup
                pt["prompt"] = "<WARMUP>"
                warmup_seen = True
                continue
            if task_idx < len(per_req):
                pidx = per_req[task_idx]["idx"]
                pt["prompt_idx"] = pidx
                pt["prompt"] = PROMPTS[pidx] if 0 <= pidx < len(PROMPTS) else f"<idx {pidx}>"
                pt["latency_s"] = per_req[task_idx]["latency_s"]
                pt["bench_tokens"] = per_req[task_idx]["tokens"]
                task_idx += 1

    # Compute totals + the cross-prompt distribution.
    valid = [r for r in per_task if r.get("prompt_idx") not in (None, -1)]
    summary = {
        "cell_dir": os.path.abspath(cell_dir),
        "n_release_events": len(per_task),
        "n_real_tasks": len(valid),
        "n_warmup": sum(1 for r in per_task if r.get("prompt_idx") == -1),
        "per_task": per_task,
        "per_prompt_summary": [],
    }
    by_prompt = defaultdict(list)
    for r in valid:
        by_prompt[r["prompt_idx"]].append(r)
    for pidx in sorted(by_prompt):
        rs = by_prompt[pidx]
        a = sum(r["accept_count"] for r in rs)
        g = sum(r["gen_count"] for r in rs)
        summary["per_prompt_summary"].append({
            "prompt_idx": pidx,
            "prompt": PROMPTS[pidx] if 0 <= pidx < len(PROMPTS) else f"<idx {pidx}>",
            "n_completions": len(rs),
            "accept_total": a,
            "gen_total": g,
            "acceptance_rate": (a / g) if g > 0 else None,
        })

    out_json = os.path.join(cell_dir, "per-prompt-acceptance.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Human-readable markdown
    md_lines = [
        f"# Per-prompt DFlash acceptance — `{os.path.basename(os.path.abspath(cell_dir))}`",
        "",
        f"Release events: {summary['n_release_events']} ({summary['n_warmup']} warmup + {summary['n_real_tasks']} measured)",
        "",
        "Per-task accept (A) / gen (G) values are read directly from the "
        "server's per-task `draft acceptance rate = R (A / G)` log line.",
        "",
        "## Per-task acceptance (in completion order)",
        "",
        "| order | slot | task | prompt_idx | n_past | A | G | rate | bench_tokens | latency_s | prompt |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, r in enumerate(per_task):
        pidx = r.get("prompt_idx", "?")
        plabel = (
            "<WARMUP>" if pidx == -1
            else short_label(r.get("prompt", "?")) if r.get("prompt") else "?"
        )
        rate = r["acceptance_rate"]
        rate_s = f"{rate:.3f}" if rate is not None else "?"
        lat = r.get("latency_s")
        lat_s = f"{lat:.1f}" if lat is not None else "?"
        bt = r.get("bench_tokens", "?")
        md_lines.append(
            f"| {i} | {r['slot']} | {r['task']} | {pidx} | {r['n_past_at_release']} | "
            f"{r['accept_count']} | {r['gen_count']} | {rate_s} | {bt} | {lat_s} | {plabel} |"
        )
    md_lines += [
        "",
        "## Per-prompt summary (aggregated across all completions of that prompt)",
        "",
        "| prompt_idx | n_completions | accept | gen | rate | prompt |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for r in summary["per_prompt_summary"]:
        rate = r["acceptance_rate"]
        rate_s = f"{rate:.3f}" if rate is not None else "?"
        md_lines.append(
            f"| {r['prompt_idx']} | {r['n_completions']} | {r['accept_total']} | "
            f"{r['gen_total']} | {rate_s} | {short_label(r['prompt'])} |"
        )
    md_lines += [
        "",
        "## Spread",
        "",
    ]
    rates = [r["acceptance_rate"] for r in summary["per_prompt_summary"]
             if r["acceptance_rate"] is not None]
    if rates:
        md_lines += [
            f"- min: {min(rates):.3f}",
            f"- max: {max(rates):.3f}",
            f"- mean: {sum(rates)/len(rates):.3f}",
            f"- spread: {max(rates)-min(rates):.3f}",
        ]

    out_md = os.path.join(cell_dir, "per-prompt-acceptance.md")
    with open(out_md, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"[ok] wrote {out_json}")
    print(f"[ok] wrote {out_md}")


if __name__ == "__main__":
    main()
