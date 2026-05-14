#!/usr/bin/env python3
"""Phase 3 / Gate 6 aggregation.

Reads gate6-spec-{none,mtp,dflash}.json from llama-bench (8 prompts × r=3
per spec), computes per-prompt geomean and the headline DFlash-vs-MTP
speedup / quality ratios per the locked Phase 3 plan.

Usage:
  python3 aggregate-gate6.py            # text summary
  python3 aggregate-gate6.py --json     # also emit gate6-summary.json
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load_spec(spec_label):
    # llama-bench mixes GPU/NCCL banner lines + "Device N: 110 MiB" log
    # lines into stdout, often inline with JSON tokens. Strip them with a
    # regex before parsing the rest as a single JSON array.
    import re
    path = os.path.join(HERE, f"gate6-spec-{spec_label}.json")
    with open(path) as f:
        text = f.read()
    # Remove "Device N: ... MiB" substrings (may appear mid-line).
    text = re.sub(r"Device \d+:\s+[\d.]+ MiB", "", text)
    # Remove NCCL banner lines.
    text = re.sub(r"^=+\s*NCCL.*$", "", text, flags=re.MULTILINE)
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < 0 or end < start:
        raise ValueError(f"{path}: no JSON array found")
    return json.loads(text[start:end + 1])


def tg_rows_by_prompt(rows):
    """{prompt_basename: row_dict} for tg128 rows only."""
    out = {}
    for r in rows:
        if r["test"] != "tg128":
            continue
        pf = os.path.basename(r.get("prompt_file", ""))
        out[pf] = r
    return out


def geomean(xs):
    xs = [x for x in xs if x > 0]
    if not xs:
        return 0.0
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    specs = {}
    for s in ("none", "mtp", "dflash"):
        specs[s] = tg_rows_by_prompt(load_spec(s))

    prompts = sorted(set().union(*(s.keys() for s in specs.values())))

    print(f"{'prompt':<10} {'none t/s':>10} {'mtp t/s':>10} {'dflash t/s':>11} "
          f"{'mtp/none':>10} {'dflash/none':>13} {'dflash/mtp':>11} "
          f"{'ppl_none':>10} {'ppl_mtp':>10} {'ppl_dflash':>11}")
    print("-" * 130)

    per_prompt = {}
    for p in prompts:
        if not all(p in specs[s] for s in ("none", "mtp", "dflash")):
            continue
        n = specs["none"][p]
        m = specs["mtp"][p]
        d = specs["dflash"][p]
        per_prompt[p] = {
            "none_t_s":     n["avg_ts"],
            "mtp_t_s":      m["avg_ts"],
            "dflash_t_s":   d["avg_ts"],
            "mtp_over_none":     m["avg_ts"] / n["avg_ts"],
            "dflash_over_none":  d["avg_ts"] / n["avg_ts"],
            "dflash_over_mtp":   d["avg_ts"] / m["avg_ts"],
            "ppl_none":     n["target_ppl_of_output"],
            "ppl_mtp":      m["target_ppl_of_output"],
            "ppl_dflash":   d["target_ppl_of_output"],
            "mtp_accept":   m["accept_rate"],
            "dflash_accept": d["accept_rate"],
        }
        pp = per_prompt[p]
        print(f"{p:<10} {pp['none_t_s']:>10.2f} {pp['mtp_t_s']:>10.2f} "
              f"{pp['dflash_t_s']:>11.3f} {pp['mtp_over_none']:>10.3f} "
              f"{pp['dflash_over_none']:>13.4f} {pp['dflash_over_mtp']:>11.4f} "
              f"{pp['ppl_none']:>10.4f} {pp['ppl_mtp']:>10.4f} {pp['ppl_dflash']:>11.4f}")

    print("-" * 130)

    summary = {
        "tg_t_s_geomean": {s: geomean([per_prompt[p][f"{s}_t_s"] for p in per_prompt])
                          for s in ("none", "mtp", "dflash")},
        "tg_speedup_geomean": {
            "mtp_over_none":    geomean([per_prompt[p]["mtp_over_none"] for p in per_prompt]),
            "dflash_over_none": geomean([per_prompt[p]["dflash_over_none"] for p in per_prompt]),
            "dflash_over_mtp":  geomean([per_prompt[p]["dflash_over_mtp"] for p in per_prompt]),
        },
        "tg_speedup_worst": {
            "dflash_over_mtp": min(per_prompt[p]["dflash_over_mtp"] for p in per_prompt),
        },
        "ppl_geomean": {s: geomean([per_prompt[p][f"ppl_{s}"] for p in per_prompt])
                       for s in ("none", "mtp", "dflash")},
        "ppl_ratio_worst": {
            "dflash_over_mtp": max(per_prompt[p]["ppl_dflash"] / per_prompt[p]["ppl_mtp"]
                                   for p in per_prompt if per_prompt[p]["ppl_mtp"] > 0),
        },
        "accept_geomean": {
            "mtp":    geomean([per_prompt[p]["mtp_accept"] for p in per_prompt]),
            "dflash": geomean([per_prompt[p]["dflash_accept"] for p in per_prompt]),
        },
    }

    g = summary["tg_t_s_geomean"]
    sp = summary["tg_speedup_geomean"]
    wp = summary["tg_speedup_worst"]
    pg = summary["ppl_geomean"]
    pw = summary["ppl_ratio_worst"]
    ac = summary["accept_geomean"]
    print(f"\nGeometric mean tg_t_s across {len(per_prompt)} prompts:")
    print(f"  none   = {g['none']:7.2f} t/s")
    print(f"  mtp    = {g['mtp']:7.2f} t/s")
    print(f"  dflash = {g['dflash']:7.3f} t/s")
    print(f"\nGeometric mean speedup vs MTP (the locked ship-gate baseline):")
    print(f"  dflash/mtp geomean = {sp['dflash_over_mtp']:.4f}")
    print(f"  dflash/mtp worst   = {wp['dflash_over_mtp']:.4f}")
    print(f"\nPPL of output (lower = closer to greedy target):")
    print(f"  none   geomean = {pg['none']:.4f}")
    print(f"  mtp    geomean = {pg['mtp']:.4f}")
    print(f"  dflash geomean = {pg['dflash']:.4f}")
    print(f"  worst dflash/mtp ratio = {pw['dflash_over_mtp']:.4f}")
    print(f"\nAcceptance rate geomean:")
    print(f"  mtp    = {ac['mtp']*100:.1f}%")
    print(f"  dflash = {ac['dflash']*100:.1f}%")

    # Locked thresholds
    print("\nLocked Phase 4 ship-gate (relative to MTP baseline):")
    print(f"  PASS    : geomean speedup ≥ 1.5×  AND worst ≥ 1.0× AND worst ppl ratio ≤ 1.05")
    print(f"  NEUTRAL : geomean speedup ∈ [1.0×, 1.5×)")
    print(f"  FAIL    : geomean speedup < 1.0×")
    if sp["dflash_over_mtp"] >= 1.5 and wp["dflash_over_mtp"] >= 1.0 and pw["dflash_over_mtp"] <= 1.05:
        verdict = "PASS"
    elif sp["dflash_over_mtp"] >= 1.0:
        verdict = "NEUTRAL"
    else:
        verdict = "FAIL"
    print(f"\n  ⇒ VERDICT: {verdict}")
    summary["verdict"] = verdict

    if "--json" in sys.argv:
        out_path = os.path.join(HERE, "gate6-summary.json")
        with open(out_path, "w") as f:
            json.dump({"per_prompt": per_prompt, "summary": summary}, f, indent=2)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
