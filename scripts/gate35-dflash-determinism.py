#!/usr/bin/env python3
"""
Gate-3.5 — DFlash multi-slot determinism test.

Resolves spec OQ-DFLASH-INHERITS-MTP-MULTISLOT-BUG.

Setup
-----
  - vLLM PR #40898 build (installed at /opt/models/venv-vllm)
  - Target: Qwen/Qwen3.6-27B BF16 from /mnt/archive/qwen3.6-27b-hf
  - Drafter: z-lab/Qwen3.6-27B-DFlash BF16 from /opt/models/qwen36-27b-dflash
  - 2 x Quadro RTX 6000 (sm_75, 24 GiB each, 48 GiB aggregate)
  - tp=2, cpu_offload_gb to absorb the 27B BF16 overflow over 48 GiB

Procedure
---------
  Same prompt, greedy (temperature=0, top_p=1, top_k=-1, seed pinned), DFlash
  speculative config enabled. Generate at three batch shapes:

    A. single request                            (n_par == 1)
    B. two identical requests in one batch       (n_par == 2)
    C. n_par == 2 again                          (rerun for self-stability)

  Compare token streams across A, B[0], B[1], C[0], C[1].

  GREEN if all five streams are byte-identical.
  RED if any pair differs. First differing token index reported.

A GREEN result means DFlash's single-batched-grid kernel dispatch on sm_75
does NOT exhibit the np>1 byte-drift that bit the PHASE45 D10.e MTP
multi-slot work. Per-slot dispatch (PerSlotVerifyDispatchAtMultiSlot
invariant) remains the spec ceiling but batched verify becomes safe at np>1.

A RED result confirms the bug is in the target's general multi-slot path
and DFlash inherits it. Per-slot dispatch is the only safe shipping shape.

Either outcome is publishable. Failures are not "blocked work" — the test
binds the spec invariant regardless of direction.
"""

import sys
import os
import json
import time
from pathlib import Path

TARGET_DIR = "/mnt/archive/qwen3.6-27b-hf"
DRAFTER_DIR = "/opt/models/qwen36-27b-dflash"
PROMPT = (
    "Write a single short paragraph about the history of the printing press."
)
MAX_TOKENS = 96
NUM_SPECULATIVE_TOKENS = 15
SEED = 42

# Route ALL caches off / — pinned via both env vars AND symlinks in
# /home/llm/.cache/*. The host's / partition runs at 95+% steady-state;
# any cache that defaults to $HOME/.cache/ will fill /.
os.environ.setdefault("HF_HOME",                  "/mnt/archive/hf-cache")
os.environ.setdefault("TMPDIR",                   "/opt/models/tmp")
os.environ.setdefault("VLLM_CACHE_ROOT",          "/opt/models/cache/vllm")
os.environ.setdefault("TRITON_CACHE_DIR",         "/opt/models/cache/triton")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",  "/opt/models/cache/torch-inductor")
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE","/opt/models/cache/flashinfer")
os.environ.setdefault("VLLM_LOGGING_LEVEL",       "WARNING")

# Apply sm_75-specific vLLM monkey-patches BEFORE importing vllm.
sys.path.insert(0, "/home/llm/yarn-agentic/scripts")
import vllm_sm75_patches
vllm_sm75_patches.apply_all()


def main() -> int:
    if not Path(TARGET_DIR, "config.json").exists():
        print(f"ERROR: target not at {TARGET_DIR}; download not complete.")
        return 2
    if not Path(DRAFTER_DIR, "model.safetensors").exists():
        print(f"ERROR: drafter not at {DRAFTER_DIR}")
        return 2

    from vllm import LLM, SamplingParams

    print("=== Gate-3.5: DFlash multi-slot determinism ===")
    print(f"target  : {TARGET_DIR}")
    print(f"drafter : {DRAFTER_DIR}")
    print(f"prompt  : {PROMPT!r}")
    print(f"max_tok : {MAX_TOKENS}, num_spec_tok : {NUM_SPECULATIVE_TOKENS}, seed : {SEED}")
    print()

    t0 = time.time()
    llm = LLM(
        model=TARGET_DIR,
        speculative_config={
            "method": "dflash",
            "model": DRAFTER_DIR,
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
        },
        tensor_parallel_size=2,
        dtype="float16",             # sm_75 has no BF16; vLLM casts BF16->fp16 at load
        gpu_memory_utilization=0.92,
        cpu_offload_gb=12,           # absorb 27B fp16 > 48 GiB VRAM
        enforce_eager=True,          # rule out CUDA-graph capture as noise source
        seed=SEED,
        max_model_len=8192,          # tight for determinism test, not a perf run
    )
    print(f"  LLM init: {time.time()-t0:.1f}s")

    # temperature=0 ⇒ greedy argmax; top_p/top_k irrelevant.
    sp = SamplingParams(
        temperature=0.0,
        seed=SEED,
        max_tokens=MAX_TOKENS,
    )

    def _capture(out_obj):
        """Pull token_ids + text + finish_reason from a RequestOutput."""
        comp = out_obj.outputs[0]
        return {
            "token_ids":     list(comp.token_ids),
            "text":          comp.text,
            "finish_reason": comp.finish_reason,
        }

    # A: single request (np=1 shape)
    print("\n[A] single request (np=1) ...")
    t_a = time.time()
    out_a = llm.generate([PROMPT], sp)
    secs_a = time.time() - t_a
    cap_a = _capture(out_a[0])
    tok_a = cap_a["token_ids"]
    print(f"  A took {secs_a:.2f}s, {len(tok_a)} toks")

    # B: two identical requests in one batch (np=2 shape)
    print("[B] two identical requests (np=2) ...")
    t_b = time.time()
    out_b = llm.generate([PROMPT, PROMPT], sp)
    secs_b = time.time() - t_b
    cap_b0 = _capture(out_b[0])
    cap_b1 = _capture(out_b[1])
    tok_b0 = cap_b0["token_ids"]
    tok_b1 = cap_b1["token_ids"]
    print(f"  B took {secs_b:.2f}s")

    # C: rerun np=2 (self-stability)
    print("[C] np=2 again (self-stability) ...")
    t_c = time.time()
    out_c = llm.generate([PROMPT, PROMPT], sp)
    secs_c = time.time() - t_c
    cap_c0 = _capture(out_c[0])
    cap_c1 = _capture(out_c[1])
    tok_c0 = cap_c0["token_ids"]
    tok_c1 = cap_c1["token_ids"]
    print(f"  C took {secs_c:.2f}s")

    captures = {
        "A   (np=1)":  cap_a,
        "B[0] (np=2)": cap_b0,
        "B[1] (np=2)": cap_b1,
        "C[0] (np=2)": cap_c0,
        "C[1] (np=2)": cap_c1,
    }
    timings = {"A": secs_a, "B": secs_b, "C": secs_c}

    streams = {
        "A   (np=1)": tok_a,
        "B[0] (np=2)": tok_b0,
        "B[1] (np=2)": tok_b1,
        "C[0] (np=2)": tok_c0,
        "C[1] (np=2)": tok_c1,
    }

    print("\n=== stream lengths ===")
    for k, v in streams.items():
        print(f"  {k}: {len(v)} toks")

    print("\n=== pairwise diff against A ===")
    ref = tok_a
    n_diff = 0
    for k, v in streams.items():
        if k.startswith("A "):
            continue
        if v == ref:
            print(f"  {k} == A : GREEN")
        else:
            first_diff = next(
                (i for i in range(min(len(ref), len(v))) if ref[i] != v[i]),
                min(len(ref), len(v)),
            )
            print(
                f"  {k} != A : RED at idx={first_diff} "
                f"(ref={ref[first_diff] if first_diff < len(ref) else None}, "
                f"got={v[first_diff] if first_diff < len(v) else None})"
            )
            n_diff += 1

    verdict = "GREEN" if n_diff == 0 else "RED"
    print(f"\n=== Gate-3.5 verdict: {verdict} ===")
    print(f"({n_diff} pairs diverged from the np=1 reference)")

    # Per-pair divergence index (where streams first differ vs A).
    divergence = {}
    for k, v in streams.items():
        if k.startswith("A "): continue
        if v == ref:
            divergence[k] = None
        else:
            idx = next(
                (i for i in range(min(len(ref), len(v))) if ref[i] != v[i]),
                min(len(ref), len(v)),
            )
            divergence[k] = {
                "first_diff_idx": idx,
                "ref_token":  ref[idx]  if idx < len(ref) else None,
                "got_token":  v[idx]    if idx < len(v) else None,
                "ref_prefix": ref[:idx][-20:],
                "got_prefix": v[:idx][-20:],
            }

    out_path = Path("/home/llm/yarn-agentic/data/gate35-dflash-determinism.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "verdict": verdict,
                "n_diff_pairs": n_diff,
                "streams": {k: v for k, v in streams.items()},
                "captures": captures,            # token_ids + text + finish_reason per output
                "timings_secs": timings,         # wall time per of the three runs
                "divergence": divergence,        # first-diff idx + surrounding context
                "prompt": PROMPT,
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
