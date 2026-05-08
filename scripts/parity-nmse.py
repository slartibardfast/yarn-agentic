#!/usr/bin/env python3
"""Compare two llama-server completion JSON outputs for parity.

Reports byte-equality on the generated text. If --logits or --hidden are
present in either file (future extension), computes NMSE = mean((a-b)^2) /
mean(a^2). Threshold defaults to 1e-7, override via PARITY_NMSE_THRESHOLD.

Exit code: 0 = parity, 1 = divergent, 2 = harness error.
"""
import json
import os
import sys


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def first_diff(a: bytes, b: bytes) -> tuple[int, str, str]:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            ctx = max(0, i - 20)
            return i, a[ctx:i + 20].decode("utf-8", "replace"), b[ctx:i + 20].decode("utf-8", "replace")
    if len(a) != len(b):
        i = n
        return i, a[max(0, i - 20):i + 20].decode("utf-8", "replace"), b[max(0, i - 20):i + 20].decode("utf-8", "replace")
    return -1, "", ""


def nmse(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return float("inf")
    num = sum((x - y) ** 2 for x, y in zip(a, b))
    den = sum(x ** 2 for x in a)
    return num / den if den > 0 else float("inf")


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} A.json B.json", file=sys.stderr)
        return 2

    a_path, b_path = sys.argv[1], sys.argv[2]
    a = load(a_path)
    b = load(b_path)

    threshold = float(os.environ.get("PARITY_NMSE_THRESHOLD", "1e-7"))

    text_a = (a.get("content") or "").encode("utf-8")
    text_b = (b.get("content") or "").encode("utf-8")
    n_a = a.get("tokens_predicted", 0)
    n_b = b.get("tokens_predicted", 0)

    print(f"[parity] A: {n_a} tokens, {len(text_a)} bytes")
    print(f"[parity] B: {n_b} tokens, {len(text_b)} bytes")

    if text_a == text_b:
        print(f"[parity] RESULT: identical (parity = 0)")
        return 0

    pos, ctx_a, ctx_b = first_diff(text_a, text_b)
    print(f"[parity] RESULT: divergent at byte {pos}")
    print(f"  A[..{pos}+20]: {ctx_a!r}")
    print(f"  B[..{pos}+20]: {ctx_b!r}")

    # If either side dumped logits/hidden in a custom field, compute NMSE.
    for key in ("logits", "hidden_states"):
        if key in a and key in b:
            try:
                flat_a = [v for row in a[key] for v in row] if isinstance(a[key][0], list) else list(a[key])
                flat_b = [v for row in b[key] for v in row] if isinstance(b[key][0], list) else list(b[key])
                e = nmse(flat_a, flat_b)
                ok = "OK" if e < threshold else "FAIL"
                print(f"[parity] NMSE({key}) = {e:.3e} (threshold {threshold:.0e}) [{ok}]")
            except Exception as exc:
                print(f"[parity] NMSE({key}) error: {exc}")

    return 1


if __name__ == "__main__":
    sys.exit(main())
