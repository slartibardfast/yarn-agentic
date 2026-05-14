#!/usr/bin/env python3
"""Direct token-level diff of multi-slot outputs against NP=1 reference.

Loads:
  gate7-validity-vanilla-np1-p{0..7}.json — NP=1 single-slot refs per prompt
  gate7-validity-vanilla-np{2,4,8}-tokens.json — multi-slot generated tokens

For each prompt p_k present in both ref and NP=N test files (slot k),
computes:
  - first_divergence_pos: index of first token where NP=N slot != NP=1 ref
                          (-1 if they match all the way)
  - matching_prefix_len:  same as first_divergence_pos if divergence
                          occurred; n_gen otherwise
  - edit_distance:        token-level Levenshtein, capped at n_gen
  - hamming_dist:         count of positions where NP=N != NP=1 ref
                          (when both are same length)
  - rejoin_after_div:     longest run of matching tokens after the first
                          divergence; reveals whether the drift is
                          transient or permanent

Outputs a summary table + writes gate7-token-diff-summary.json.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load_json(path):
    with open(path) as f:
        return json.load(f)


def slot_tokens(record, slot_id):
    for sl in record["slots"]:
        if sl["slot"] == slot_id:
            return sl["generated_tokens"]
    return None


def levenshtein(a, b, cap=None):
    """Token-level edit distance, optionally capped for cheap early-exit."""
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        if cap is not None and min(curr) > cap:
            return cap + 1
        prev, curr = curr, prev
    return prev[m]


def diff_pair(ref, test):
    n_ref, n_test = len(ref), len(test)
    n_min = min(n_ref, n_test)
    first_div = -1
    for i in range(n_min):
        if ref[i] != test[i]:
            first_div = i
            break
    if first_div == -1 and n_ref == n_test:
        return {
            "n_ref": n_ref, "n_test": n_test,
            "first_divergence_pos": -1,
            "matching_prefix_len": n_min,
            "hamming_dist": 0,
            "edit_distance": 0,
            "rejoin_after_div": 0,
            "test_text_after_div": "",
        }
    mpl = first_div if first_div >= 0 else n_min
    hamming = sum(1 for i in range(n_min) if ref[i] != test[i]) + abs(n_ref - n_test)
    edit = levenshtein(ref, test)
    rejoin = 0
    if first_div >= 0:
        run = 0
        for i in range(first_div + 1, n_min):
            if ref[i] == test[i]:
                run += 1
                rejoin = max(rejoin, run)
            else:
                run = 0
    return {
        "n_ref": n_ref, "n_test": n_test,
        "first_divergence_pos": first_div,
        "matching_prefix_len": mpl,
        "hamming_dist": hamming,
        "edit_distance": edit,
        "rejoin_after_div": rejoin,
    }


def main():
    # Refs: 8 single-prompt NP=1 JSONs.
    refs = {}
    for k in range(8):
        p = os.path.join(HERE, f"gate7-validity-vanilla-np1-p{k}.json")
        if not os.path.exists(p):
            print(f"missing ref: {p}", file=sys.stderr); sys.exit(1)
        rec = load_json(p)
        # Each NP=1 ref has exactly one slot (slot_id=0) corresponding to
        # the offset-controlled prompt.
        toks = rec["slots"][0]["generated_tokens"]
        refs[k] = toks

    # Tests: NP=2, 4, 8 multi-slot JSONs.
    tests = {}
    for N in (2, 4, 8):
        p = os.path.join(HERE, f"gate7-validity-vanilla-np{N}-tokens.json")
        if not os.path.exists(p):
            print(f"missing test: {p}", file=sys.stderr); continue
        tests[N] = load_json(p)

    print(f"{'prompt':<7} {'np':>4} {'n_ref':>5} {'n_test':>6} {'first_div':>9} "
          f"{'match_prefix':>12} {'hamming':>7} {'edit_dist':>9} {'rejoin':>6}")
    print("-" * 80)

    summary = {"per_prompt": {}, "aggregates": {}}
    div_by_np = {2: [], 4: [], 8: []}
    edit_by_np = {2: [], 4: [], 8: []}

    for k in range(8):
        ref = refs[k]
        summary["per_prompt"][f"p{k}"] = {"n_ref": len(ref), "by_np": {}}
        for N in (2, 4, 8):
            if N not in tests: continue
            if k >= N: continue  # this prompt wasn't run at this NP
            t = slot_tokens(tests[N], k)
            if t is None: continue
            d = diff_pair(ref, t)
            print(f"p{k:<6} {N:>4} {d['n_ref']:>5} {d['n_test']:>6} "
                  f"{d['first_divergence_pos']:>9} {d['matching_prefix_len']:>12} "
                  f"{d['hamming_dist']:>7} {d['edit_distance']:>9} {d['rejoin_after_div']:>6}")
            summary["per_prompt"][f"p{k}"]["by_np"][str(N)] = d
            div_by_np[N].append(d["first_divergence_pos"] if d["first_divergence_pos"] >= 0 else len(ref))
            edit_by_np[N].append(d["edit_distance"])

    print("-" * 80)
    print(f"\nFirst-divergence-position summary (lower = drift earlier):")
    for N in (2, 4, 8):
        if not div_by_np[N]: continue
        xs = div_by_np[N]
        print(f"  NP={N}: per-prompt = {xs}  min={min(xs)}  max={max(xs)}  mean={sum(xs)/len(xs):.1f}")

    print(f"\nEdit-distance summary (lower = closer to ref):")
    for N in (2, 4, 8):
        if not edit_by_np[N]: continue
        xs = edit_by_np[N]
        print(f"  NP={N}: per-prompt = {xs}  min={min(xs)}  max={max(xs)}  mean={sum(xs)/len(xs):.1f}")

    summary["aggregates"] = {
        f"np{N}": {
            "first_div_per_prompt": div_by_np[N],
            "edit_dist_per_prompt": edit_by_np[N],
        } for N in (2, 4, 8) if div_by_np[N]
    }

    out_path = os.path.join(HERE, "gate7-token-diff-summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
