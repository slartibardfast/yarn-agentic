#!/usr/bin/env python3
"""Drive a fixed workload against a running llama-server for overnight
profiling runs. Reads the fixed prompt from fixed_prompt.txt, hits
/completion N times, returns timings + a content fingerprint.

Usage:
    python3 drive.py [N_RUNS] [--base-seed 42] [--n-predict 256] \\
                     [--server-url http://127.0.0.1:9099] [--label unnamed]

Outputs JSON to stdout: {
  "label": str,
  "server_url": str,
  "prompt_path": str,
  "n_runs": int,
  "runs": [
    {"seed": int, "wall_s": float, "timings": {...}, "content_head": str},
    ...
  ],
  "aggregate": {
    "prompt_per_second_median": float,
    "predicted_per_second_median": float,
    "mtp_acceptance_median": float | null
  }
}
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request
import pathlib

HERE = pathlib.Path(__file__).parent
DEFAULT_PROMPT_PATH = HERE / "fixed_prompt.txt"


def run_one(server_url, prompt, seed, n_predict):
    url = server_url.rstrip("/") + "/completion"
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "seed": seed,
        "stream": False,
        "cache_prompt": False,  # force full prompt eval each run for stable timings
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        r = json.loads(resp.read())
    wall = time.time() - t0
    return r, wall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n_runs", type=int, nargs="?", default=3)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--n-predict", type=int, default=256)
    parser.add_argument("--server-url", default="http://127.0.0.1:9099")
    parser.add_argument("--prompt-path", default=str(DEFAULT_PROMPT_PATH))
    parser.add_argument("--label", default="unnamed")
    args = parser.parse_args()

    prompt = pathlib.Path(args.prompt_path).read_text()
    runs = []
    for i in range(args.n_runs):
        seed = args.base_seed + i
        try:
            r, wall = run_one(args.server_url, prompt, seed, args.n_predict)
        except Exception as e:  # noqa: BLE001
            print(
                f"[{i+1}/{args.n_runs}] ERROR seed={seed}: {type(e).__name__}: {e}",
                file=sys.stderr,
                flush=True,
            )
            continue
        timings = r.get("timings", {}) or {}
        runs.append({
            "seed": seed,
            "wall_s": round(wall, 3),
            "timings": timings,
            "content_head": (r.get("content") or "")[:200],
        })
        print(
            f"[{i+1}/{args.n_runs}] seed={seed} wall={wall:.1f}s "
            f"prompt={timings.get('prompt_n')}tok@{timings.get('prompt_per_second',0):.1f}tps "
            f"predict={timings.get('predicted_n')}tok@{timings.get('predicted_per_second',0):.1f}tps "
            f"draft_n={timings.get('draft_n',0)} accepted={timings.get('draft_n_accepted',0)}",
            file=sys.stderr,
            flush=True,
        )

    if runs:
        pps = [r["timings"].get("prompt_per_second", 0) for r in runs]
        tps = [r["timings"].get("predicted_per_second", 0) for r in runs]
        drafts = [r["timings"].get("draft_n", 0) for r in runs]
        accepts = [r["timings"].get("draft_n_accepted", 0) for r in runs]
        total_draft = sum(drafts)
        total_accept = sum(accepts)
        aggregate = {
            "prompt_per_second_median": round(statistics.median(pps), 3) if pps else None,
            "predicted_per_second_median": round(statistics.median(tps), 3) if tps else None,
            "mtp_draft_n_total": total_draft,
            "mtp_draft_n_accepted_total": total_accept,
            "mtp_acceptance_rate": (
                round(total_accept / total_draft, 4) if total_draft > 0 else None
            ),
        }
    else:
        aggregate = {
            "prompt_per_second_median": None,
            "predicted_per_second_median": None,
            "mtp_draft_n_total": 0,
            "mtp_draft_n_accepted_total": 0,
            "mtp_acceptance_rate": None,
        }

    out = {
        "label": args.label,
        "server_url": args.server_url,
        "prompt_path": args.prompt_path,
        "n_runs": args.n_runs,
        "n_predict": args.n_predict,
        "base_seed": args.base_seed,
        "runs": runs,
        "aggregate": aggregate,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
