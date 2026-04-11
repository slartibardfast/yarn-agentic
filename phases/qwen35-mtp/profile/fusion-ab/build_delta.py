#!/usr/bin/env python3
"""Build fusion A/B delta summary from the step 1 baseline and step 2 runs."""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent

BASELINE = ROOT / "per-op" / "q4km-vega-f16kv-2026-04-11T224817Z.json"
BASELINE_DRIVE = ROOT / "per-op" / "q4km-vega-f16kv-2026-04-11T224817Z.drive.json"
RUNS = {
    "no_fusion":    HERE / "no-fusion-q4km-vega-2026-04-11T225140Z",
    "no_multi_add": HERE / "no-multi-add-q4km-vega-2026-04-11T225330Z",
    "no_graph_opt": HERE / "no-graph-opt-q4km-vega-2026-04-11T225520Z",
}


def load_perf(p):
    return json.load(open(p))


def load_drive(p):
    # the drive.py output is JSON on stdout; if the tee file has other text, find the first '{'
    t = open(p).read()
    s = t.find('{')
    return json.loads(t[s:])


def main():
    base_perf = load_perf(BASELINE)
    base_drv = load_drive(BASELINE_DRIVE)
    base_ops = {r["op"]: r for r in base_perf["ops"]}

    rows = []
    for label, prefix in RUNS.items():
        perf = load_perf(str(prefix) + ".json")
        drv = load_drive(str(prefix) + ".drive.txt")
        ops = {r["op"]: r for r in perf["ops"]}

        total_us_delta = perf["ops_total_us_sum"] - base_perf["ops_total_us_sum"]
        prompt_delta = (
            drv["aggregate"]["prompt_per_second_median"]
            - base_drv["aggregate"]["prompt_per_second_median"]
        )
        predict_delta = (
            drv["aggregate"]["predicted_per_second_median"]
            - base_drv["aggregate"]["predicted_per_second_median"]
        )

        # Per-op delta: compare shared op names and note which ones appeared
        # only in one side.
        only_in_base = sorted(set(base_ops) - set(ops))
        only_in_run = sorted(set(ops) - set(base_ops))
        shared_with_delta = []
        for name in (set(base_ops) & set(ops)):
            b = base_ops[name]
            r = ops[name]
            shared_with_delta.append({
                "op": name,
                "base_us": b["total_us"],
                "run_us": r["total_us"],
                "delta_us": round(r["total_us"] - b["total_us"], 1),
                "base_count": b["count"],
                "run_count": r["count"],
            })
        shared_with_delta.sort(key=lambda x: abs(x["delta_us"]), reverse=True)

        rows.append({
            "label": label,
            "total_us_delta": round(total_us_delta, 1),
            "total_us_delta_pct": round(
                100.0 * total_us_delta / base_perf["ops_total_us_sum"], 2
            ),
            "prompt_t_s": drv["aggregate"]["prompt_per_second_median"],
            "prompt_delta_t_s": round(prompt_delta, 2),
            "predict_t_s": drv["aggregate"]["predicted_per_second_median"],
            "predict_delta_t_s": round(predict_delta, 2),
            "predict_delta_pct": round(
                100.0 * predict_delta / base_drv["aggregate"]["predicted_per_second_median"],
                2,
            ),
            "mtp_acceptance": drv["aggregate"]["mtp_acceptance_rate"],
            "only_in_baseline_top10": only_in_base[:10],
            "only_in_run_top10": only_in_run[:10],
            "biggest_op_deltas_top10": shared_with_delta[:10],
        })

    out = {
        "baseline_label": "q4km-vega-f16kv-fusion-on",
        "baseline_total_us": base_perf["ops_total_us_sum"],
        "baseline_prompt_t_s": base_drv["aggregate"]["prompt_per_second_median"],
        "baseline_predict_t_s": base_drv["aggregate"]["predicted_per_second_median"],
        "baseline_mtp_acceptance": base_drv["aggregate"]["mtp_acceptance_rate"],
        "runs": rows,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
