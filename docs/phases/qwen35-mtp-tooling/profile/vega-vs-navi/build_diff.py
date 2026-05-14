#!/usr/bin/env python3
"""Build Vega-vs-Navi per-op delta summary."""
import json
import pathlib

ROOT = pathlib.Path(__file__).parent
VEGA = ROOT.parent / "per-op" / "q4km-vega-f16kv-2026-04-11T224817Z.json"
VEGA_DRIVE = ROOT.parent / "per-op" / "q4km-vega-f16kv-2026-04-11T224817Z.drive.json"
NAVI = ROOT / "navi-per-op-2026-04-11T232800Z.json"
NAVI_DRIVE = ROOT / "navi-per-op-2026-04-11T232800Z.drive.txt"


def load_drive(p):
    t = pathlib.Path(p).read_text()
    s = t.find("{")
    return json.loads(t[s:])


def main():
    vega_perf = json.load(open(VEGA))
    navi_perf = json.load(open(NAVI))
    vega_drv = load_drive(VEGA_DRIVE)
    navi_drv = load_drive(NAVI_DRIVE)

    vega_ops = {r["op"]: r for r in vega_perf["ops"]}
    navi_ops = {r["op"]: r for r in navi_perf["ops"]}

    shared = []
    for name in set(vega_ops) & set(navi_ops):
        v = vega_ops[name]
        n = navi_ops[name]
        ratio_us = n["total_us"] / v["total_us"] if v["total_us"] > 0 else None
        vega_gf = v.get("avg_gflops_s")
        navi_gf = n.get("avg_gflops_s")
        gf_ratio = (navi_gf / vega_gf) if vega_gf and navi_gf else None
        shared.append({
            "op": name,
            "vega_us": v["total_us"],
            "navi_us": n["total_us"],
            "navi_vs_vega_ratio": round(ratio_us, 3) if ratio_us else None,
            "vega_gflops": vega_gf,
            "navi_gflops": navi_gf,
            "gflops_ratio": round(gf_ratio, 3) if gf_ratio else None,
            "share_pct_vega": v["share_pct"],
        })
    # Sort by vega share descending
    shared.sort(key=lambda x: x["share_pct_vega"], reverse=True)

    out = {
        "totals": {
            "vega_total_ms": round(vega_perf["ops_total_us_sum"] / 1000, 1),
            "navi_total_ms": round(navi_perf["ops_total_us_sum"] / 1000, 1),
            "navi_vs_vega_ratio_total": round(
                navi_perf["ops_total_us_sum"] / vega_perf["ops_total_us_sum"], 3
            ),
            "vega_prompt_t_s": vega_drv["aggregate"]["prompt_per_second_median"],
            "navi_prompt_t_s": navi_drv["aggregate"]["prompt_per_second_median"],
            "vega_predict_t_s": vega_drv["aggregate"]["predicted_per_second_median"],
            "navi_predict_t_s": navi_drv["aggregate"]["predicted_per_second_median"],
            "vega_mtp_accept": vega_drv["aggregate"]["mtp_acceptance_rate"],
            "navi_mtp_accept": navi_drv["aggregate"]["mtp_acceptance_rate"],
        },
        "n_shared_ops": len(shared),
        "per_op_top20": shared[:20],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
