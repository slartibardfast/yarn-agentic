#!/usr/bin/env python3
"""Layer 2 prompt-diff capture.

Polls /slots every 2 s. For each slot whose `prompt` text changed
(by sha256), writes a snapshot file and appends a unified diff
versus the previous snapshot to a per-slot `.diff` log. The
timestamps in the diff headers let snoop-summarise.py join with
the journal `erased invalidated` events.
"""
import datetime, difflib, hashlib, json, os, sys, time, urllib.request

OUT_DIR = sys.argv[1]
PROMPTS = os.path.join(OUT_DIR, "prompts")
os.makedirs(PROMPTS, exist_ok=True)

URL = "http://127.0.0.1:8080/slots"
PERIOD = 2.0

last_hash = {}
seq = {}


def now_iso():
    return datetime.datetime.now().isoformat(timespec="milliseconds")


def write_snapshot(slot_id, prompt):
    n = seq.get(slot_id, 0)
    path = os.path.join(PROMPTS, f"slot-{slot_id}-{n:04d}.txt")
    with open(path, "w") as f:
        f.write(prompt)
    seq[slot_id] = n + 1
    return path, n


while True:
    try:
        with urllib.request.urlopen(URL, timeout=4) as r:
            slots = json.load(r)
        for s in slots:
            sid = s.get("id")
            prompt = s.get("prompt") or ""
            if not prompt:
                continue
            h = hashlib.sha256(prompt.encode("utf-8", "replace")).hexdigest()
            if last_hash.get(sid) == h:
                continue
            path, n = write_snapshot(sid, prompt)
            print(f"{now_iso()} slot {sid} snapshot {n} chars={len(prompt)} sha={h[:12]}")
            if n > 0:
                prev = os.path.join(PROMPTS, f"slot-{sid}-{n-1:04d}.txt")
                with open(prev) as f:
                    a = f.read().splitlines(keepends=True)
                b = prompt.splitlines(keepends=True)
                diff = list(difflib.unified_diff(
                    a, b,
                    fromfile=f"slot-{sid}-{n-1:04d}",
                    tofile=f"slot-{sid}-{n:04d}",
                    n=20,
                    lineterm="",
                ))
                if diff:
                    diff_path = os.path.join(PROMPTS, f"slot-{sid}.diff")
                    with open(diff_path, "a") as f:
                        f.write(f"\n=== {now_iso()} slot {sid} {n-1} -> {n} ===\n")
                        f.write("".join(diff))
                        f.write("\n")
            last_hash[sid] = h
    except Exception as e:
        print(f"{now_iso()} ERR {e}")
    time.sleep(PERIOD)
