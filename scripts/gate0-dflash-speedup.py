#!/usr/bin/env python3
"""
Gate 0 / M0 — DFlash speedup repro on 2x Quadro RTX 6000 (sm_75).

Reproduces z-lab's published DFlash speedup on our hardware. Compares
vLLM-with-DFlash to vLLM-no-speculative on the same target model + prompts.

Setup
-----
  - vLLM PR #40898 build (installed at /opt/models/venv-vllm)
  - Target: Qwen/Qwen3.6-27B BF16 from /mnt/archive/qwen3.6-27b-hf
  - Drafter: z-lab/Qwen3.6-27B-DFlash BF16 from /opt/models/qwen36-27b-dflash
  - tp=2, cpu_offload_gb to absorb 54 GiB BF16 over 48 GiB VRAM

Procedure
---------
  Two runs over the same prompt set, same SamplingParams, same seed:
    Run A: speculative_config={method=dflash, ...}
    Run B: no speculative_config (vanilla decode)

  Each run measures:
    - wall_time      : total inference time
    - total_out_toks : sum of generated tokens
    - tok_per_sec    : total_out_toks / wall_time
    - per_request_lat: list of latencies

  Reported: tok_per_sec(A) / tok_per_sec(B) = speedup ratio.
  Bonus: estimated accept rate from DFlash internal stats if exposed.

  Closes only on like-for-like — same model dtype, same max_tokens, same
  enforce_eager status, same gpu_memory_utilization and cpu_offload_gb.
  Per CLAUDE.md §3 ("anchor to measured baselines"), the published
  speedup is whatever it was on the paper's hardware. Our gate is:
  does enabling DFlash beat the vanilla decode on OUR hardware?

  Net positive → Gate 0 GREEN (DFlash earns its keep on sm_75).
  Neutral / negative → Gate 0 RED (paper's win does not transfer).
  Either is publishable.
"""

import sys
import os
import json
import time
from pathlib import Path

TARGET_DIR = os.environ.get(
    "GATE0_TARGET_DIR",
    "/opt/models/hf-cache/models--Intel--Qwen3.6-27B-int4-AutoRound/"
    "snapshots/a00e481620facd57da3a86eaa5c90e2e811d1aac",
)
DRAFTER_DIR = os.environ.get("GATE0_DRAFTER_DIR", "/opt/models/qwen36-27b-dflash")

# Spread of prompt lengths and types to approximate real serving load.
# Qwen 3.6 is a thinking model by design — measurements reflect production
# behavior (thinking ON). Acceptance rate (MAL ~3.3) is the operating point;
# tune around it via num_speculative_tokens.
PROMPTS = [
    "Explain the difference between latent diffusion and pixel-space diffusion in two sentences.",
    "Summarize the plot of King Lear in one paragraph.",
    "Write Python code that fits a 2nd-degree polynomial to a list of (x, y) pairs.",
    "What are the main causes of the Peloponnesian War?",
    "Translate to French: The early-morning fog lingered over the harbour until the trawlers cut through it.",
    "List five practical steps for reducing memory allocations in a hot inner loop in Rust.",
    "Describe the role of telomeres in cellular aging.",
    "Write a haiku about a printing press.",
]

MAX_TOKENS = 256
NUM_SPECULATIVE_TOKENS = int(os.environ.get("GATE0_NUM_SPEC_TOKENS", "15"))
SEED = 42

os.environ.setdefault("HF_HOME",                  "/mnt/archive/hf-cache")
os.environ.setdefault("TMPDIR",                   "/opt/models/tmp")
os.environ.setdefault("VLLM_CACHE_ROOT",          "/opt/models/cache/vllm")
os.environ.setdefault("TRITON_CACHE_DIR",         "/opt/models/cache/triton")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",  "/opt/models/cache/torch-inductor")
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE","/opt/models/cache/flashinfer")
os.environ.setdefault("VLLM_LOGGING_LEVEL",       "WARNING")

sys.path.insert(0, "/home/llm/yarn-agentic/scripts")
import vllm_sm75_patches
vllm_sm75_patches.apply_all()


def init_llm(spec_enabled: bool):
    """Construct the vLLM engine. Reused across np modes within a spec config."""
    from vllm import LLM

    print(f"\n=== init LLM — speculative={'DFlash' if spec_enabled else 'OFF'} ===")
    init_kwargs = dict(
        model=TARGET_DIR,
        tensor_parallel_size=2,
        # dtype="auto" lets vLLM pick activation dtype based on the quantization
        # config. INT4 AutoRound uses FP16 activations on sm_75.
        dtype="auto",
        # AutoRound writes GPTQ-format safetensors but the config.json carries
        # autoround_version instead of quant_method. vLLM's GPTQ loader reads
        # autoround_version (see vllm/.../quantization/gptq.py:152). We pass
        # quantization="gptq_marlin" explicitly since auto-detection looks for
        # quant_method which AutoRound omits. Marlin kernels are supported on
        # sm_75 (Turing).
        quantization="gptq_marlin",
        # INT4 model is ~7 GiB/GPU at TP=2. Cap vLLM total allocations at 80%
        # of VRAM (19.2 GiB), leaving ~4.8 GiB headroom for forward activations.
        # GMU=0.92 OOM'd: vLLM allocated 22 GiB for model + KV, forward needed
        # ~850 MiB more for DFlash's drafter + verify intermediate tensors.
        gpu_memory_utilization=0.80,
        # INT4 AutoRound: 27B × 0.5 bytes = ~13.5 GB total, ~7 GB/GPU at TP=2.
        # No offload needed. cpu_offload_gb=0 means weights live entirely on GPU,
        # eliminating the per-step CPU↔GPU traffic that dominated fp16 runs.
        cpu_offload_gb=0,
        # Re-enable CUDA graphs. vLLM warned that the previous max_num_scheduled_
        # tokens=4608 (default) was suboptimal for DFlash with 15 spec tokens;
        # bumping to 8192 gives the verify scheduler headroom and reduces graph
        # recapture for varying batch shapes.
        enforce_eager=False,
        max_num_batched_tokens=8192,
        # vLLM's custom_all_reduce.cuh fails on sm_75 with "invalid argument" at
        # determine_available_memory()'s forward profile. Force the standard NCCL
        # all-reduce path which is fully compatible with Turing.
        disable_custom_all_reduce=True,
        # Qwen 3.6 has DeltaNet recurrent layers → vLLM allocates Mamba cache
        # blocks (one per decode seq). Default max_num_seqs=256 exceeded the
        # available 241 blocks at GMU=0.80, blocking the vanilla pass at CUDA
        # graph capture. We only run 8 prompts → 16 is plenty with headroom.
        max_num_seqs=16,
        # Enable internal stats logging (spec-decode accept rate, etc.).
        # Default is True (suppressed). Needs VLLM_LOGGING_LEVEL=INFO to surface.
        disable_log_stats=False,
        seed=SEED,
        max_model_len=4096,
    )
    if spec_enabled:
        # Switch the spec method via env. method="mtp" uses the target's built-in
        # MTP layers (no separate drafter — analog to ik_llama.cpp's --draft N).
        # method="dflash" uses the separate Qwen3.6-27B-DFlash drafter.
        spec_method = os.environ.get("GATE0_SPEC_METHOD", "dflash")
        spec_cfg = {
            "method": spec_method,
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
        }
        if spec_method == "dflash":
            spec_cfg["model"] = DRAFTER_DIR
        init_kwargs["speculative_config"] = spec_cfg

    # Optional backend override. Set GATE0_ATTENTION_BACKEND=FLASHINFER to
    # force the FLASHINFER path (vLLM otherwise falls through to FLEX_ATTENTION
    # on sm_75, which is the slow path).
    backend_name = os.environ.get("GATE0_ATTENTION_BACKEND")
    if backend_name:
        from vllm.config.attention import AttentionConfig
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        backend_enum = AttentionBackendEnum[backend_name]
        init_kwargs["attention_config"] = AttentionConfig(backend=backend_enum)
        print(f"  forcing attention backend: {backend_name}")
    t_init = time.time()
    llm = LLM(**init_kwargs)
    init_secs = time.time() - t_init
    print(f"  init: {init_secs:.1f}s")
    return llm, init_secs


def _extract_per_req_timings(outs):
    """Extract per-request TTFT and TPOT from vLLM RequestOutput.metrics.

    Returns list of dicts (one per request). If metrics aren't populated
    (rare in v1 but defensive), TTFT/TPOT fall back to None.
    """
    rows = []
    for o in outs:
        n_out = len(o.outputs[0].token_ids)
        m = getattr(o, "metrics", None)
        ttft_s = None
        tpot_s = None
        decode_s = None
        if m is not None:
            try:
                # vLLM RequestMetrics fields: arrival_time, first_scheduled_time,
                # first_token_time, last_token_time, finished_time.
                if m.first_token_time is not None and m.arrival_time is not None:
                    ttft_s = m.first_token_time - m.arrival_time
                if (m.last_token_time is not None
                        and m.first_token_time is not None
                        and n_out > 1):
                    decode_s = m.last_token_time - m.first_token_time
                    # TPOT = decode duration / (output tokens - 1), since the
                    # first token's latency is TTFT (prefill-bound).
                    tpot_s = decode_s / (n_out - 1)
            except AttributeError:
                pass
        rows.append({
            "n_out": n_out,
            "ttft_s": ttft_s,
            "decode_s": decode_s,
            "tpot_s": tpot_s,
            "tg_tok_per_s": (1.0 / tpot_s) if tpot_s else None,
        })
    return rows


def measure_pass(llm, np_mode: int, label: str):
    """Run a single measurement pass at np=1 (sequential) or np=N (batched).

    Returns:
      dict with wall_clock_aggregate_tok_per_s (current metric, includes prefill),
      and decode_only_tg_tok_per_s (per-request TPOT averaged, decode-only).
    """
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, seed=SEED, max_tokens=MAX_TOKENS)
    warmup_sp = SamplingParams(temperature=0.0, seed=SEED, max_tokens=32)

    # Warmup: same shape as the upcoming pass so JIT cache is hot.
    if np_mode == 1:
        # Warm one prompt at a time (1 shape variant covers np=1).
        print(f"  [{label}] warmup np=1 (1 prompt × 32 tok)...")
        t_warm = time.time()
        _ = llm.generate([PROMPTS[0]], warmup_sp)
        warmup_secs = time.time() - t_warm
        print(f"  [{label}] warmup: {warmup_secs:.2f}s")
    else:
        print(f"  [{label}] warmup np={np_mode} ({np_mode} prompts × 32 tok)...")
        t_warm = time.time()
        _ = llm.generate(PROMPTS[:np_mode], warmup_sp)
        warmup_secs = time.time() - t_warm
        print(f"  [{label}] warmup: {warmup_secs:.2f}s")

    # Timed pass — selects `np_mode` prompts for batched runs. For np=1 we
    # iterate over ALL 8 prompts sequentially to stabilise the single-stream
    # measurement (more samples → less per-prompt variance).
    if np_mode == 1:
        # Submit prompts one at a time. Single-stream, no batching.
        print(f"  [{label}] timed np=1: 8 prompts sequentially...")
        all_outs = []
        t_gen = time.time()
        for p in PROMPTS:
            o = llm.generate([p], sp)
            all_outs.extend(o)
        gen_secs = time.time() - t_gen
        outs = all_outs
    else:
        # Submit np_mode prompts in one batched call. Use PROMPTS[:np_mode] so
        # np=4 actually measures np=4 (otherwise vLLM batches all 8 since
        # max_num_seqs=16 ≥ 8 and gives us the np=8 number).
        prompts_to_use = PROMPTS[:np_mode]
        print(f"  [{label}] timed np={np_mode}: {len(prompts_to_use)} prompts batched...")
        t_gen = time.time()
        outs = llm.generate(prompts_to_use, sp)
        gen_secs = time.time() - t_gen

    total_out_toks = sum(len(o.outputs[0].token_ids) for o in outs)
    aggregate_tok_per_s = total_out_toks / gen_secs

    per_req = _extract_per_req_timings(outs)
    valid_tpot = [r["tpot_s"] for r in per_req if r["tpot_s"] is not None]
    avg_tpot = sum(valid_tpot) / len(valid_tpot) if valid_tpot else None
    decode_only_tg_tok_per_s = (1.0 / avg_tpot) if avg_tpot else None

    valid_ttft = [r["ttft_s"] for r in per_req if r["ttft_s"] is not None]
    avg_ttft = sum(valid_ttft) / len(valid_ttft) if valid_ttft else None

    print(f"  [{label}] gen: {gen_secs:.2f}s  out_toks={total_out_toks}  "
          f"aggregate={aggregate_tok_per_s:.2f} tok/s  "
          f"decode_only_tg={decode_only_tg_tok_per_s or 'n/a'}  "
          f"avg_ttft={avg_ttft or 'n/a'}")

    return {
        "label": label,
        "np_mode": np_mode,
        "warmup_secs": warmup_secs,
        "gen_secs": gen_secs,
        "total_out_toks": total_out_toks,
        "tok_per_sec_aggregate_incl_prefill": aggregate_tok_per_s,
        "tok_per_sec_decode_only_tg": decode_only_tg_tok_per_s,
        "avg_ttft_s": avg_ttft,
        "per_req": per_req,
        "outputs": [o.outputs[0].text for o in outs],
    }


def main() -> int:
    if not Path(TARGET_DIR, "config.json").exists():
        print(f"ERROR: target not at {TARGET_DIR}; download not complete.")
        return 2
    if not Path(DRAFTER_DIR, "model.safetensors").exists():
        print(f"ERROR: drafter not at {DRAFTER_DIR}")
        return 2

    results = {}
    # Two LLM init cycles: DFlash on, then vanilla. Each LLM is reused for
    # np=1, np=4, np=8 measurements (3 cells per spec mode = 6 cells total).
    NP_MODES = [int(x) for x in os.environ.get("GATE0_NP_MODES", "1,4,8").split(",")]
    for spec_enabled, spec_label in [(True, "dflash"), (False, "vanilla")]:
        llm, init_secs = init_llm(spec_enabled)
        try:
            for np_mode in NP_MODES:
                label = f"{spec_label}_np{np_mode}"
                results[label] = measure_pass(llm, np_mode, label)
                results[label]["init_secs"] = init_secs  # shared across np modes
        finally:
            # Free GPU before constructing the next LLM
            del llm
            import gc; gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    print("\n=== Gate 0 / M0 summary (aggregate, includes prefill) ===")
    for label, r in results.items():
        a = r["tok_per_sec_aggregate_incl_prefill"]
        print(f"  {label:>14}: aggregate={a:6.2f} tok/s")

    print("\n=== Gate 0 / M0 summary (decode-only tg, ~ apples-to-apples vs llama-bench tg) ===")
    for label, r in results.items():
        t = r["tok_per_sec_decode_only_tg"]
        ttft = r["avg_ttft_s"]
        ts = f"{t:.2f}" if t is not None else "n/a"
        ttft_s = f"{ttft:.3f}s" if ttft is not None else "n/a"
        print(f"  {label:>14}: tg={ts:>7} tok/s  avg_ttft={ttft_s}")

    speedups = {}
    for np_mode in NP_MODES:
        d = results[f"dflash_np{np_mode}"]
        v = results[f"vanilla_np{np_mode}"]
        for metric_key, metric_label in [
            ("tok_per_sec_aggregate_incl_prefill", "aggregate"),
            ("tok_per_sec_decode_only_tg", "decode_only_tg"),
        ]:
            dv = d.get(metric_key)
            vv = v.get(metric_key)
            if dv and vv:
                speedups[f"np{np_mode}_{metric_label}"] = dv / vv

    print("\n=== Speedup ratios (DFlash / vanilla) ===")
    for k, s in speedups.items():
        verdict = "GREEN" if s > 1.05 else ("NEUTRAL" if s > 0.95 else "RED")
        print(f"  {k:>30}: {s:.2f}x   [{verdict}]")

    out_path = Path(os.environ.get(
        "GATE0_OUT_PATH",
        "/home/llm/yarn-agentic/data/gate0-dflash-speedup.json",
    ))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "results": results,        # all 4 cells (dflash/vanilla × np=1/np=8)
                "speedups": speedups,      # DFlash/vanilla ratios for each metric & np
                "prompts": PROMPTS,
                "max_tokens": MAX_TOKENS,
                "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
                "seed": SEED,
                "target": TARGET_DIR,
                "drafter": DRAFTER_DIR,
            },
            indent=2,
            default=str,  # for any non-JSON-native fields
        )
    )
    print(f"\nwrote: {out_path}")
    # Exit GREEN if np=1 shows decode-only speedup > 1.05x
    np1_speedup = speedups.get("np1_decode_only_tg") or speedups.get("np1_aggregate", 0)
    return 0 if (np1_speedup or 0) > 1.05 else 1


if __name__ == "__main__":
    sys.exit(main())
