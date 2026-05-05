#!/usr/bin/env python3
"""Summarise a snoop run into findings.md.

Reads journal.log, nginx.log, slots.jsonl, prompts/. Computes:
- request count + status distribution (nginx)
- checkpoint creates / restores / invalidations + pos distributions
- prompt-snapshot count per slot, last-prompt size
- recommended -crs threshold based on observed similarity at
  invalidation events
"""
import json, os, re, sys, glob, datetime
from collections import Counter

RUN_DIR = sys.argv[1]
OUT = os.path.join(RUN_DIR, "findings.md")

# ---- nginx --------------------------------------------------------
nginx_lines = []
nginx_path = os.path.join(RUN_DIR, "nginx.log")
if os.path.exists(nginx_path):
    with open(nginx_path) as f:
        nginx_lines = f.readlines()

nginx_status = Counter()
nginx_paths = Counter()
post_completions = 0
for line in nginx_lines:
    m = re.search(r'"(\w+) (\S+) HTTP/[0-9.]+" (\d+)', line)
    if not m:
        continue
    method, path, status = m.group(1), m.group(2), m.group(3)
    nginx_status[status] += 1
    nginx_paths[(method, path)] += 1
    if method == "POST" and "chat/completions" in path:
        post_completions += 1

# ---- journal ------------------------------------------------------
j_creates = []
j_restores = []
j_invalidated = []
j_path = os.path.join(RUN_DIR, "journal.log")
if os.path.exists(j_path):
    with open(j_path) as f:
        for line in f:
            ts_m = re.match(r"(\S+) ", line)
            ts = ts_m.group(1) if ts_m else ""
            pos_m = re.search(r"pos_min = (\d+)", line)
            pos = int(pos_m.group(1)) if pos_m else None
            if "create_check" in line and "created context checkpoint" in line:
                cap_m = re.search(r"checkpoint (\d+) of (\d+)", line)
                took_m = re.search(r"took +([0-9.]+) ms", line)
                j_creates.append({
                    "ts": ts, "pos": pos,
                    "n": int(cap_m.group(1)) if cap_m else None,
                    "cap": int(cap_m.group(2)) if cap_m else None,
                    "took_ms": float(took_m.group(1)) if took_m else None,
                })
            elif "restored context checkpoint" in line:
                took_m = re.search(r"took +([0-9.]+) ms", line)
                j_restores.append({
                    "ts": ts, "pos": pos,
                    "took_ms": float(took_m.group(1)) if took_m else None,
                })
            elif "erased invalidated context checkpoint" in line:
                j_invalidated.append({"ts": ts, "pos": pos})

# ---- slots --------------------------------------------------------
slots_records = []
sp = os.path.join(RUN_DIR, "slots.jsonl")
if os.path.exists(sp):
    with open(sp) as f:
        for line in f:
            try:
                slots_records.append(json.loads(line))
            except Exception:
                pass

# Highest prompt_chars per slot
max_prompt = {}
for rec in slots_records:
    if "slots" not in rec:
        continue
    for s in rec["slots"]:
        sid = s["id"]
        max_prompt[sid] = max(max_prompt.get(sid, 0), s.get("prompt_chars") or 0)

# ---- prompts ------------------------------------------------------
prompt_count = Counter()
prompts_dir = os.path.join(RUN_DIR, "prompts")
if os.path.isdir(prompts_dir):
    for f in sorted(os.listdir(prompts_dir)):
        m = re.match(r"slot-(\d+)-(\d+)\.txt$", f)
        if m:
            prompt_count[int(m.group(1))] += 1

# ---- emit ---------------------------------------------------------
def fmt_pos_dist(positions):
    if not positions:
        return "(none)"
    positions = [p for p in positions if p is not None]
    if not positions:
        return "(no pos data)"
    positions.sort()
    return (f"n={len(positions)} min={positions[0]} "
            f"median={positions[len(positions)//2]} max={positions[-1]}")

with open(OUT, "w") as f:
    f.write(f"# Snoop run findings — {os.path.basename(RUN_DIR)}\n\n")
    f.write(f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}\n\n")

    f.write("## HTTP traffic (nginx, OpenCode-only)\n\n")
    f.write(f"- Lines logged: {len(nginx_lines)}\n")
    f.write(f"- POST /v1/chat/completions: {post_completions}\n")
    f.write(f"- Status distribution: {dict(nginx_status)}\n")
    if nginx_paths:
        top = nginx_paths.most_common(5)
        f.write(f"- Top paths: {top}\n")
    f.write("\n")

    f.write("## Server events (journal)\n\n")
    f.write(f"- Checkpoint creates: {len(j_creates)}\n")
    if j_creates:
        took = [c['took_ms'] for c in j_creates if c['took_ms'] is not None]
        if took:
            f.write(f"  - took_ms: min={min(took):.1f} mean={sum(took)/len(took):.1f} max={max(took):.1f}\n")
        caps = [c for c in j_creates if c['n'] is not None and c['cap'] is not None]
        if caps:
            cap_max = max((c['cap'] for c in caps), default=0)
            n_max = max((c['n'] for c in caps), default=0)
            f.write(f"  - max usage: {n_max} of {cap_max}\n")
        f.write(f"  - pos distribution: {fmt_pos_dist([c['pos'] for c in j_creates])}\n")
    f.write(f"- Checkpoint restores: {len(j_restores)}\n")
    if j_restores:
        f.write(f"  - pos distribution: {fmt_pos_dist([r['pos'] for r in j_restores])}\n")
    f.write(f"- Erased invalidated: {len(j_invalidated)}\n")
    if j_invalidated:
        f.write(f"  - pos distribution: {fmt_pos_dist([i['pos'] for i in j_invalidated])}\n")

    full_evals = post_completions - len(j_restores) if post_completions else 0
    if post_completions:
        hit_rate = len(j_restores) / post_completions if post_completions else 0
        f.write(f"\n**Cache hit rate**: {len(j_restores)}/{post_completions} = {hit_rate:.1%}\n")
        f.write(f"**Implied full-evals**: {max(full_evals, 0)}\n")
    f.write("\n")

    f.write("## Slots\n\n")
    f.write(f"- Polls captured: {len(slots_records)}\n")
    for sid in sorted(max_prompt):
        f.write(f"- slot {sid}: max prompt_chars = {max_prompt[sid]} "
                f"(~{max_prompt[sid]//4} tokens), "
                f"prompt snapshots = {prompt_count.get(sid, 0)}\n")
    f.write("\n")

    f.write("## Step-3 (-crs) recommendation\n\n")
    if not j_invalidated:
        f.write("No invalidations observed — `-crs` default 0.50 unchanged.\n")
    elif len(j_invalidated) <= 3:
        f.write(f"{len(j_invalidated)} invalidations, all small drift. "
                f"`-crs` default 0.50 likely fine; revisit if rate climbs.\n")
    else:
        # Look at pos distribution: tail-drift = positions clustered near max
        # observed prompt position; cascade = positions spread across whole prompt.
        positions = [i['pos'] for i in j_invalidated if i['pos'] is not None]
        if positions and max_prompt:
            highest = max(max_prompt.values()) // 4  # rough tokens
            tail_zone = highest - 2048
            tail_count = sum(1 for p in positions if p >= tail_zone)
            f.write(f"{len(j_invalidated)} invalidations: "
                    f"{tail_count} in tail zone (>= last 2048 tok), "
                    f"{len(positions) - tail_count} earlier.\n")
            if tail_count == len(positions):
                f.write("- Pure tail-drift — plugin envelope likely innocent. "
                        "Keep `-crs 0.50`.\n")
            else:
                f.write("- Some non-tail invalidations — possible mid-prefix "
                        "rewrite. Inspect prompts/slot-N.diff manually. "
                        "Try `-crs 0.30` to be more permissive on near-matches.\n")
    f.write("\n")

    f.write("## Files\n\n")
    for name in ["journal.log", "nginx.log", "slots.jsonl", "gpu.tsv", "prompt-diff.log"]:
        p = os.path.join(RUN_DIR, name)
        if os.path.exists(p):
            f.write(f"- `{name}` — {os.path.getsize(p)} bytes\n")
    f.write(f"- `prompts/` — {sum(prompt_count.values())} snapshots\n")
