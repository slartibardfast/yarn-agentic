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

TARGET_DIR = "/mnt/archive/qwen3.6-27b-hf"
DRAFTER_DIR = "/opt/models/qwen36-27b-dflash"

# Spread of prompt lengths and types to approximate real serving load.
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
NUM_SPECULATIVE_TOKENS = 15
SEED = 42

os.environ.setdefault("HF_HOME", "/mnt/archive/hf-cache")
os.environ.setdefault("VLLM_CACHE_ROOT", "/opt/vllm-runtime-cache")
os.environ.setdefault("TMPDIR", "/opt/tmp")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")


def run_pass(spec_enabled: bool, label: str) -> dict:
    from vllm import LLM, SamplingParams

    print(f"\n=== Pass {label} — speculative={'DFlash' if spec_enabled else 'OFF'} ===")
    init_kwargs = dict(
        model=TARGET_DIR,
        tensor_parallel_size=2,
        dtype="bfloat16",
        gpu_memory_utilization=0.92,
        cpu_offload_gb=12,
        enforce_eager=True,
        seed=SEED,
        max_model_len=4096,
    )
    if spec_enabled:
        init_kwargs["speculative_config"] = {
            "method": "dflash",
            "model": DRAFTER_DIR,
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
        }
    t_init = time.time()
    llm = LLM(**init_kwargs)
    init_secs = time.time() - t_init
    print(f"  init: {init_secs:.1f}s")

    sp = SamplingParams(temperature=0.0, seed=SEED, max_tokens=MAX_TOKENS)

    t_gen = time.time()
    outs = llm.generate(PROMPTS, sp)
    gen_secs = time.time() - t_gen

    total_out_toks = sum(len(o.outputs[0].token_ids) for o in outs)
    per_req_lat = [
        # If finish_time/start_time are exposed, use them; otherwise gen_secs / N is a
        # rough proxy. The aggregate tok/sec across the batch is the perf metric of
        # interest.
        gen_secs / len(outs)
        for _ in outs
    ]
    tok_per_sec = total_out_toks / gen_secs

    print(f"  gen:   {gen_secs:.2f}s  total_out_toks={total_out_toks}  tok/s={tok_per_sec:.2f}")
    return {
        "label": label,
        "speculative": "dflash" if spec_enabled else "off",
        "init_secs": init_secs,
        "gen_secs": gen_secs,
        "total_out_toks": total_out_toks,
        "tok_per_sec": tok_per_sec,
        "per_req_lat": per_req_lat,
        "outputs": [o.outputs[0].text for o in outs],
    }


def main() -> int:
    if not Path(TARGET_DIR, "config.json").exists():
        print(f"ERROR: target not at {TARGET_DIR}; download not complete.")
        return 2
    if not Path(DRAFTER_DIR, "model.safetensors").exists():
        print(f"ERROR: drafter not at {DRAFTER_DIR}")
        return 2

    # Pass A: DFlash on. Pass B: vanilla. Run in this order so init costs don't
    # bias the comparison (LLM is reconstructed for each pass to ensure clean
    # state).
    result_a = run_pass(spec_enabled=True, label="A_dflash_on")
    result_b = run_pass(spec_enabled=False, label="B_vanilla")

    speedup = result_a["tok_per_sec"] / result_b["tok_per_sec"]
    verdict = "GREEN" if speedup > 1.05 else "RED"

    print("\n=== Gate 0 / M0 summary ===")
    print(f"  vanilla     : {result_b['tok_per_sec']:.2f} tok/s")
    print(f"  with DFlash : {result_a['tok_per_sec']:.2f} tok/s")
    print(f"  speedup     : {speedup:.2f}x")
    print(f"  verdict     : {verdict}")

    out_path = Path("/home/llm/yarn-agentic/data/gate0-dflash-speedup.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "verdict": verdict,
                "speedup_ratio": speedup,
                "vanilla": result_b,
                "with_dflash": result_a,
                "prompts": PROMPTS,
                "max_tokens": MAX_TOKENS,
                "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
                "seed": SEED,
                "target": TARGET_DIR,
                "drafter": DRAFTER_DIR,
            },
            indent=2,
        )
    )
    print(f"\nwrote: {out_path}")
    return 0 if verdict == "GREEN" else 1


if __name__ == "__main__":
    sys.exit(main())
