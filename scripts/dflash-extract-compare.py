"""dflash-extract-compare — load ik_llama-side and vLLM-side residual-stream
dumps and report:

  - self-consistency (SHA-256) across ik_llama runs (3 runs expected)
  - cosine similarity (per-token, mean across tokens) between
    ik_llama_mean and vllm for each layer
  - normalised MSE (NMSE = ||a - b||^2 / ||b||^2) between ik_llama_mean
    and vllm for each layer
  - basic shape parity check (n_tokens, n_embd)

Writes a JSON report and prints a per-layer table. Decision (PASS / NEUTRAL
/ FAIL) is the user's call — this tool reports the numbers, doesn't gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def per_token_cosine(a: np.ndarray, b: np.ndarray) -> dict:
    # a, b shape: [n_tokens, n_embd]
    assert a.shape == b.shape
    sims = []
    for t in range(a.shape[0]):
        sims.append(cosine(a[t], b[t]))
    sims = np.array(sims, dtype=np.float64)
    return {
        "mean": float(np.nanmean(sims)),
        "min": float(np.nanmin(sims)),
        "max": float(np.nanmax(sims)),
        "per_token": sims.tolist(),
    }


def nmse(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    denom = float(np.dot(b, b))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a - b, a - b) / denom)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iklama-dir", required=True)
    p.add_argument("--vllm-dir",   required=True)
    p.add_argument("--layers",     required=True, help="comma-separated indices")
    p.add_argument("--n-runs",     type=int, default=3)
    p.add_argument("--out",        required=True, help="output JSON path")
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip()]

    report = {
        "layers": layers,
        "n_runs_iklama": args.n_runs,
        "per_layer": {},
    }

    for il in layers:
        # ik_llama-side self-consistency.
        runs = []
        for r in range(1, args.n_runs + 1):
            path = os.path.join(args.iklama_dir, f"run{r}-layer{il}.npy")
            if not os.path.exists(path):
                print(f"missing {path}", file=sys.stderr)
                sys.exit(1)
            runs.append(np.load(path))
        shapes = [r.shape for r in runs]
        hashes = [hashlib.sha256(r.tobytes()).hexdigest() for r in runs]
        identical = len(set(hashes)) == 1
        mean_run = np.mean(np.stack(runs, axis=0), axis=0)

        # vLLM-side.
        vllm_path = os.path.join(args.vllm_dir, f"vllm-layer{il}.npy")
        if not os.path.exists(vllm_path):
            print(f"missing {vllm_path} (vllm not run yet?)", file=sys.stderr)
            report["per_layer"][str(il)] = {
                "iklama_shapes": [list(s) for s in shapes],
                "iklama_hashes": [h[:16] for h in hashes],
                "iklama_self_consistent": identical,
                "vllm_present": False,
            }
            continue
        v = np.load(vllm_path)

        n_min = min(mean_run.shape[0], v.shape[0])
        a = mean_run[:n_min].astype(np.float64)
        b = v[:n_min].astype(np.float64)

        report["per_layer"][str(il)] = {
            "iklama_shapes":          [list(s) for s in shapes],
            "iklama_hashes":          [h[:16] for h in hashes],
            "iklama_self_consistent": identical,
            "vllm_shape":             list(v.shape),
            "n_tokens_compared":      n_min,
            "cosine_per_token":       per_token_cosine(a, b),
            "cosine_full":            cosine(a, b),
            "nmse":                   nmse(a, b),
            "norms": {
                "iklama_l2": float(np.linalg.norm(a)),
                "vllm_l2":   float(np.linalg.norm(b)),
            },
        }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== ik_llama self-consistency ===")
    for il in layers:
        rec = report["per_layer"][str(il)]
        status = "OK" if rec.get("iklama_self_consistent") else "DIFFER"
        print(f"  layer {il:>2}: {status}  hashes={rec['iklama_hashes']}")

    print("\n=== ik_llama (mean) vs vLLM ===")
    print(f"{'layer':>6}  {'n_tok':>5}  {'cos_full':>10}  {'cos_mean':>10}  {'cos_min':>10}  {'nmse':>10}  iklama_L2 / vllm_L2")
    for il in layers:
        rec = report["per_layer"][str(il)]
        if not rec.get("vllm_present", True):
            print(f"  {il:>4}: vllm missing")
            continue
        print(f"  {il:>4}  {rec['n_tokens_compared']:>5}  "
              f"{rec['cosine_full']:>10.6f}  "
              f"{rec['cosine_per_token']['mean']:>10.6f}  "
              f"{rec['cosine_per_token']['min']:>10.6f}  "
              f"{rec['nmse']:>10.6f}  "
              f"{rec['norms']['iklama_l2']:.2f} / {rec['norms']['vllm_l2']:.2f}")

    print(f"\n=== wrote {args.out} ===")


if __name__ == "__main__":
    main()
