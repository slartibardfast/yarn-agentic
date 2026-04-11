#!/usr/bin/env python3
"""MTP context sweep — drive /completion at varying prompt lengths and
record draft_n / draft_n_accepted.

Builds a synthetic prompt by repeating the fixed prompt until it's
approximately `target_tokens` long (uses a crude 0.28 tokens/char
heuristic for Qwen3.5; server normalises). Hits /completion with
n_predict=128, temperature=0, seed=42.

Usage:
    python3 mtp_ctx_sweep.py \\
        --server-url http://127.0.0.1:9099 \\
        --label q4km-vega \\
        --out profile/mtp-sweep/ctx-sweep-<UTC>.json \\
        --target 1024,2048,4096,6144
"""
import argparse
import json
import pathlib
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).parent
DEFAULT_PROMPT = HERE / "fixed_prompt.txt"


def build_prompt(base, target_tokens, mode="truncate"):
    """Approximate a prompt of target_tokens.

    mode="truncate": slice `base` down (or repeat once if base is too short).
    mode="repeat":   concatenate `base` repeatedly (historic behavior — biases
                     MTP acceptance because the model echoes the repeated text).
    """
    # ~0.28 tokens per char for English Qwen tokenization (rough)
    target_chars = int(target_tokens / 0.28)
    if mode == "repeat":
        reps = max(1, (target_chars + len(base) - 1) // len(base))
        return (base + "\n\n") * reps
    # truncate mode: take the first target_chars of `base`. If base is
    # shorter than target, repeat once and warn.
    if len(base) >= target_chars:
        return base[:target_chars]
    return (base + "\n\n") * (target_chars // len(base) + 1)


def run_one(server_url, prompt, seed=42, n_predict=128):
    url = server_url.rstrip("/") + "/completion"
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "seed": seed,
        "stream": False,
        "cache_prompt": False,
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
    parser.add_argument("--server-url", default="http://127.0.0.1:9099")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target", default="1024,2048,4096,6144")
    parser.add_argument("--n-predict", type=int, default=128)
    parser.add_argument("--prompt-path", default=str(DEFAULT_PROMPT))
    parser.add_argument("--mode", choices=["truncate", "repeat"], default="truncate")
    args = parser.parse_args()

    base = pathlib.Path(args.prompt_path).read_text()
    targets = [int(x) for x in args.target.split(",")]
    rows = []
    for t in targets:
        prompt = build_prompt(base, t, mode=args.mode)
        try:
            r, wall = run_one(args.server_url, prompt, n_predict=args.n_predict)
        except Exception as e:  # noqa: BLE001
            print(
                f"[target={t}] ERROR: {type(e).__name__}: {e}",
                file=sys.stderr,
                flush=True,
            )
            rows.append({
                "target_tokens": t,
                "error": f"{type(e).__name__}: {e}",
            })
            continue
        timings = r.get("timings", {}) or {}
        actual_prompt_tokens = timings.get("prompt_n", 0)
        row = {
            "target_tokens": t,
            "actual_prompt_tokens": actual_prompt_tokens,
            "wall_s": round(wall, 3),
            "prompt_ms": timings.get("prompt_ms"),
            "prompt_per_second": timings.get("prompt_per_second"),
            "predicted_n": timings.get("predicted_n"),
            "predicted_ms": timings.get("predicted_ms"),
            "predicted_per_second": timings.get("predicted_per_second"),
            "draft_n": timings.get("draft_n", 0),
            "draft_n_accepted": timings.get("draft_n_accepted", 0),
            "acceptance_rate": (
                round(timings.get("draft_n_accepted", 0) / timings.get("draft_n", 1), 4)
                if timings.get("draft_n", 0) > 0 else None
            ),
            "content_head": (r.get("content") or "")[:150],
        }
        rows.append(row)
        print(
            f"[prompt={actual_prompt_tokens}] predict={row['predicted_n']}t@{row['predicted_per_second'] or 0:.1f}tps "
            f"draft_n={row['draft_n']} accept={row['draft_n_accepted']} "
            f"rate={row['acceptance_rate']}",
            file=sys.stderr, flush=True,
        )

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "label": args.label,
        "server_url": args.server_url,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H%M%SZ", time.gmtime()),
        "rows": rows,
    }, indent=2))
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
