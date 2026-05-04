#!/usr/bin/env python3
"""
Phase 0.D — aggregate multi-stream nsys traces into PP/TG breakdowns.

Usage:
    profile-aggregate.py <root-dir>

For each <root-dir>/np<N>-<mode>/trace.nsys-rep this script:
  1. Runs `nsys stats --report cuda_gpu_kern_sum,nvtx_pushpop_sum`
     to extract kernel and NVTX-range CSV summaries.
  2. Categorises kernels into buckets relevant to V-F1.T1.qq.
  3. Splits cuBLAS-dispatch NVTX ranges into PP (N>1) vs TG (N==1).
  4. Reads per-prompt-metrics.csv to compute throughput-per-slot and
     aggregate-system-throughput.
  5. Emits two tables (PP and TG) plus an MTP-crossover analysis.

Output goes to <root-dir>/PHASE32-CUDA-PROFILE-RESULTS.md.
"""

import argparse
import csv
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def categorize_kernel(name: str) -> str:
    n = name.lower()
    if "dequantize_block_q4_0_ar16" in n or "q4_0_ar16" in n:
        return "Q4_0_AR16 dequant"
    if "mul_mat_vec_q" in n or "mul_mat_q" in n or ("dequantize_mul_mat_vec" in n):
        return "Q4_0 native mul_mat"
    if "mul_mat_vec" in n:
        return "mul_mat_vec generic"
    if "dequantize_block_q4_0" in n:
        return "Q4_0 dequant (other)"
    if any(k in n for k in ("gemm", "sgemm", "hgemm", "gemv",
                            "cutlass", "volta_", "turing_")):
        return "cuBLAS GEMM"
    if "flash_attn" in n or "fattn" in n or "flashattention" in n:
        return "flash attention"
    if "rope" in n:
        return "RoPE"
    if "softmax" in n or "soft_max" in n:
        return "softmax"
    if any(k in n for k in ("rmsnorm", "norm_f32", "silu", "gelu", "add_f32",
                            "mul_f32", "scale_f32", "sigmoid", "gate_prep")):
        return "elementwise / norm / activation"
    if any(k in n for k in ("cpy", "memcpy", "memset")):
        return "memory copy / set"
    if any(k in n for k in ("mtp", "argmax", "fused_argmax", "topk")):
        return "MTP / argmax"
    if "delta_net" in n or "ssm_conv" in n or "softplus" in n:
        return "linear-attn (delta_net/ssm_conv)"
    if "reduce" in n:
        return "reduce"
    return f"OTHER: {name[:60]}"


def parse_nvtx_range(label: str):
    """
    Range labels emitted by ggml-cuda.cu look like:
        :op_mul_mat_cublas|N=1|beta-0
        :batched_cublas|N=128|attn_q_norm_proj
    Returns (dispatch, n, tensor) or None if format doesn't match.
    """
    if label.startswith(":"):
        label = label[1:]
    m = re.match(r"([a-zA-Z0-9_]+)\|N=(-?\d+)\|(.+)", label)
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3)


def run_nsys_stats(rep: Path, report: str) -> Path:
    """Run nsys stats, return the path to the produced CSV."""
    out_prefix = rep.with_suffix("")  # foo.nsys-rep -> foo
    out_csv = Path(f"{out_prefix}_{report}.csv")
    if out_csv.exists():
        return out_csv
    cmd = [
        "nsys", "stats",
        "--report", report,
        "--format", "csv",
        "--output", str(out_prefix),
        str(rep),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    return out_csv


def parse_kern_csv(path: Path):
    """Returns (cat_total_ns, total_ns) where cat_total_ns is dict[cat]=(time_ns, calls)."""
    cats = defaultdict(lambda: [0, 0])
    total = 0
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # header
        for r in reader:
            if not r or len(r) < 9:
                continue
            try:
                ttime = int(r[1])
                inst = int(r[2])
            except ValueError:
                continue
            name = r[8]
            cat = categorize_kernel(name)
            cats[cat][0] += ttime
            cats[cat][1] += inst
            total += ttime
    return cats, total


def parse_nvtx_csv(path: Path):
    """
    Returns dict[(dispatch, n_eq_1, tensor)] = (time_ns, calls)
    where n_eq_1 is True if N==1 (TG) else False (PP).
    """
    out = defaultdict(lambda: [0, 0])
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # header
        for r in reader:
            if not r or len(r) < 9:
                continue
            try:
                ttime = int(r[1])
                inst = int(r[2])
            except ValueError:
                continue
            label = r[8]
            parsed = parse_nvtx_range(label)
            if parsed is None:
                continue
            dispatch, n, tensor = parsed
            tg = (n == 1)
            out[(dispatch, tg, tensor)][0] += ttime
            out[(dispatch, tg, tensor)][1] += inst
    return out


def parse_metrics(path: Path):
    """Returns list of dicts with id, n_predict, prompt_per_second, predicted_per_second, predicted_n."""
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append({
                    "id": r["id"],
                    "n_predict": int(r["n_predict"]),
                    "pp_tps": float(r["prompt_per_second"]),
                    "tg_tps": float(r["predicted_per_second"]),
                    "pred_n": int(r["predicted_n"]),
                })
            except (KeyError, ValueError):
                continue
    return rows


def aggregate_throughput(metrics):
    """Mean PP/TG throughput across all corpus records (per-slot)."""
    if not metrics:
        return 0.0, 0.0
    pp = sum(m["pp_tps"] for m in metrics) / len(metrics)
    tg = sum(m["tg_tps"] for m in metrics) / len(metrics)
    return pp, tg


def build_pp_tg_buckets(nvtx, kern_cats, kern_total):
    """
    Given NVTX ranges (with PP/TG split) and unallocated kernel buckets,
    produce per-phase percentage tables.

    NVTX gives us cuBLAS dispatch time split by phase. Kernels not
    captured by NVTX (everything outside ggml_cuda_mul_mat) we assign
    to a single "non-mul_mat" bucket and don't try to phase-split.

    Strategy:
      - cuBLAS-dispatch-related kernel time: take from NVTX directly,
        split by phase.
      - Other kernel categories (flash attn, reduce, softmax, ...):
        report in a single "all phases" column for that bucket — we
        can't reliably split them without more instrumentation.

    Returns:
      pp_dispatch: dict[dispatch_label -> total ns] for PP
      tg_dispatch: dict[dispatch_label -> total ns] for TG
      non_dispatch: dict[cat -> (ns, calls)] not phase-split
    """
    pp_dispatch = defaultdict(int)
    tg_dispatch = defaultdict(int)
    pp_top_tensors = defaultdict(int)
    tg_top_tensors = defaultdict(int)
    for (dispatch, tg, tensor), (t, c) in nvtx.items():
        if tg:
            tg_dispatch[dispatch] += t
            tg_top_tensors[(dispatch, tensor)] += t
        else:
            pp_dispatch[dispatch] += t
            pp_top_tensors[(dispatch, tensor)] += t

    # Non-dispatch buckets (anything kernel that's not under our NVTX
    # ranges; in practice this is flash-attn, reduce, softmax, dequant,
    # elementwise — i.e. kernels that fire from elsewhere in ggml-cuda).
    # We can't precisely subtract NVTX-captured kernel time from kern_cats
    # without per-kernel correlation; report buckets verbatim and note
    # the cuBLAS bucket overlaps with NVTX dispatches.
    return pp_dispatch, tg_dispatch, pp_top_tensors, tg_top_tensors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path,
                    help="dir containing np<N>-<mode>/trace.nsys-rep")
    args = ap.parse_args()

    cells = []
    for sub in sorted(args.root.iterdir()):
        m = re.match(r"np(\d+)-(nomtp|mtp)", sub.name)
        if not m:
            continue
        np_v = int(m.group(1))
        mode = m.group(2)
        rep = sub / "trace.nsys-rep"
        metrics_csv = sub / "per-prompt-metrics.csv"
        if not rep.exists():
            print(f"SKIP {sub.name}: no trace.nsys-rep", file=sys.stderr)
            continue
        cells.append((np_v, mode, rep, metrics_csv))

    if not cells:
        print(f"No trace cells found under {args.root}", file=sys.stderr)
        sys.exit(1)

    out_path = args.root / "PHASE32-CUDA-PROFILE-RESULTS.md"
    out = []
    out.append(f"# PHASE32 CUDA Profile — Realistic Multi-Stream\n")
    out.append(f"Source traces: `{args.root}`\n\n")

    out.append("## Per-cell throughput (mean across agentic corpus)\n\n")
    out.append("| np | mode | per-slot PP t/s | per-slot TG t/s | aggregate PP t/s | aggregate TG t/s |\n")
    out.append("|---:|:-----|---:|---:|---:|---:|\n")

    cell_summary = []
    for np_v, mode, rep, metrics_csv in cells:
        metrics = parse_metrics(metrics_csv)
        pp_per_slot, tg_per_slot = aggregate_throughput(metrics)
        out.append(
            f"| {np_v} | {mode} | {pp_per_slot:.1f} | {tg_per_slot:.1f} "
            f"| {pp_per_slot * np_v:.1f} | {tg_per_slot * np_v:.1f} |\n"
        )
        cell_summary.append({
            "np": np_v, "mode": mode, "pp_per_slot": pp_per_slot,
            "tg_per_slot": tg_per_slot,
            "pp_aggregate": pp_per_slot * np_v,
            "tg_aggregate": tg_per_slot * np_v,
        })

    # MTP crossover analysis
    out.append("\n## MTP throughput crossover\n\n")
    out.append("Aggregate TG t/s. Crossover N is the smallest np where MTP ≥ nomtp.\n\n")
    out.append("| np | nomtp aggregate TG | mtp aggregate TG | MTP advantage |\n")
    out.append("|---:|---:|---:|---:|\n")
    by_np = defaultdict(dict)
    for c in cell_summary:
        by_np[c["np"]][c["mode"]] = c["tg_aggregate"]
    crossover = None
    for np_v in sorted(by_np):
        no = by_np[np_v].get("nomtp")
        mt = by_np[np_v].get("mtp")
        if no is None or mt is None:
            continue
        delta = mt - no
        out.append(f"| {np_v} | {no:.1f} | {mt:.1f} | {delta:+.1f} t/s ({100*delta/no:+.1f}%) |\n")
        if delta > 0 and crossover is None:
            crossover = np_v
    out.append(f"\n**Crossover:** np={crossover}\n" if crossover else
               "\n**Crossover:** not reached at tested np values.\n")

    # Per-cell kernel + NVTX breakdown
    out.append("\n## Kernel breakdown per cell\n\n")
    for np_v, mode, rep, _ in cells:
        kern_csv = run_nsys_stats(rep, "cuda_gpu_kern_sum")
        nvtx_csv = run_nsys_stats(rep, "nvtx_pushpop_sum")
        kern_cats, kern_total = parse_kern_csv(kern_csv)
        nvtx = parse_nvtx_csv(nvtx_csv)
        pp_d, tg_d, pp_top, tg_top = build_pp_tg_buckets(nvtx, kern_cats, kern_total)
        pp_total = sum(pp_d.values())
        tg_total = sum(tg_d.values())

        out.append(f"### np={np_v} mode={mode}\n\n")
        out.append(f"Total kernel GPU time: {kern_total/1e9:.3f} s\n\n")
        out.append("**Kernel categories (all phases, summed across both GPUs):**\n\n")
        out.append("| Bucket | Time (ms) | % | Calls | Avg µs |\n|---|---:|---:|---:|---:|\n")
        for cat, (t, inst) in sorted(kern_cats.items(), key=lambda x: -x[1][0])[:15]:
            avg_us = (t / inst / 1000.0) if inst else 0
            out.append(f"| {cat} | {t/1e6:.1f} | {100*t/kern_total:.1f}% | {inst:,} | {avg_us:.1f} |\n")
        out.append("\n")

        if nvtx:
            out.append(f"**cuBLAS-dispatch attribution (NVTX, PP=N>1, TG=N==1):**\n\n")
            out.append(f"PP dispatch total: {pp_total/1e6:.1f} ms\n\n")
            out.append("| Dispatch | PP ms | TG ms |\n|---|---:|---:|\n")
            for d in sorted(set(list(pp_d) + list(tg_d))):
                out.append(f"| {d} | {pp_d.get(d,0)/1e6:.1f} | {tg_d.get(d,0)/1e6:.1f} |\n")
            out.append("\n**Top-10 cuBLAS-attributable tensors (TG):**\n\n")
            out.append("| Dispatch | Tensor | TG ms |\n|---|---|---:|\n")
            for (d, tn), t in sorted(tg_top.items(), key=lambda x: -x[1])[:10]:
                out.append(f"| {d} | {tn} | {t/1e6:.1f} |\n")
            out.append("\n**Top-10 cuBLAS-attributable tensors (PP):**\n\n")
            out.append("| Dispatch | Tensor | PP ms |\n|---|---|---:|\n")
            for (d, tn), t in sorted(pp_top.items(), key=lambda x: -x[1])[:10]:
                out.append(f"| {d} | {tn} | {t/1e6:.1f} |\n")
            out.append("\n")

    out_path.write_text("".join(out))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
