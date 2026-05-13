"""gate3b-drafter-logits-vllm — dump vLLM PR #40898 DFlash drafter pre-argmax
logits for a fixed prompt at BLOCK_SIZE=4. The reference for the T4 closure
binding (drafter logits within 1e-5 NMSE vs vLLM).

Approach: hook DFlashQwen3ForCausalLM.compute_logits inside the vLLM worker
process via collective_rpc. Each invocation captures (n_query_positions,
draft_vocab_size) logits; we save the FIRST drafter forward only.

Output:
  data/dflash-extracts/drafter-logits-bs4-vllm.npy
    float32, shape [n_query_positions, vocab_size]
  data/dflash-extracts/drafter-prompt-tokens.npy
    int64, shape [n_prompt_tokens]
  data/dflash-extracts/drafter-meta.json
    block_size, n_query_positions, draft_vocab_size, target_vocab_size,
    drafter SHA, target SHA, prompt text, n_prompt_tokens
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default=os.environ.get(
        "GATE3B_TARGET",
        "/opt/models/hf-cache/models--Intel--Qwen3.6-27B-int4-AutoRound/"
        "snapshots/a00e481620facd57da3a86eaa5c90e2e811d1aac",
    ))
    p.add_argument("--drafter", default=os.environ.get(
        "GATE3B_DRAFTER", "/opt/models/qwen36-27b-dflash",
    ))
    p.add_argument("--prompt", default=
        "Explain the difference between latent diffusion and pixel-space "
        "diffusion in two sentences.")
    p.add_argument("--prompts-file", default=None,
                   help="Optional: file with one prompt per line. If set, "
                        "iterates all prompts in one vLLM session and dumps "
                        "per-prompt subdirs prompt-0/ prompt-1/ etc. into "
                        "out_dir. Overrides --prompt.")
    p.add_argument("--block-size", type=int, default=4,
                   help="DFlash block_size = number of speculative tokens")
    p.add_argument("--out-dir", default="/home/llm/yarn-agentic/data/dflash-extracts")
    args = p.parse_args()

    os.environ.setdefault("HF_HOME",                  "/mnt/archive/hf-cache")
    os.environ.setdefault("TMPDIR",                   "/opt/models/tmp")
    os.environ.setdefault("VLLM_CACHE_ROOT",          "/opt/models/cache/vllm")
    os.environ.setdefault("TRITON_CACHE_DIR",         "/opt/models/cache/triton")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",  "/opt/models/cache/torch-inductor")
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE","/opt/models/cache/flashinfer")
    os.environ.setdefault("VLLM_LOGGING_LEVEL",       "WARNING")
    # The runtime-installed cloudpickle patch (vllm_sm75_patches) only
    # applies in the LLM init process. At TP>1, the engine-core
    # subprocess decoder doesn't see the patch and raises
    # "Extension type code 2 is not supported". Allow cloudpickle
    # globally for the duration of this script — we trust the function
    # payload we ship (it's our own RPC function).
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import vllm_sm75_patches
    vllm_sm75_patches.apply_all()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Quick existence checks before paying the vLLM init cost (~20-30 min).
    if not (Path(args.target) / "config.json").exists():
        print(f"ERROR: target config not at {args.target}", file=sys.stderr)
        sys.exit(2)
    if not (Path(args.drafter) / "config.json").exists():
        print(f"ERROR: drafter config not at {args.drafter}", file=sys.stderr)
        sys.exit(2)

    from vllm import LLM, SamplingParams

    print(f"\n=== init vLLM (DFlash, BS={args.block_size}) ===", flush=True)
    t0 = time.time()
    # TP=1 — the vllm_sm75_patches.combine_hidden_states monkey-patch
    # replaces a class method on the LLM-init process's class object.
    # TP=2 workers run in subprocesses that get their own copy of the
    # class object and do NOT see the patch, triggering an fp16/fp32
    # dtype mismatch at the drafter's fc projection. T2's dflash-extract
    # script used TP=1 and worked. We mirror that here.
    #
    # Memory at TP=1 INT4 AutoRound: target ~14 GiB + drafter ~3.3 GiB
    # = ~17 GiB on one 24 GiB GPU. Plenty of headroom for activations
    # and KV cache; cpu_offload_gb=0 (no host RAM round-trips needed).
    llm = LLM(
        model=args.target,
        tensor_parallel_size=1,
        dtype="auto",
        quantization="gptq_marlin",
        gpu_memory_utilization=0.92,
        # Target's unquantized BF16 DeltaNet linear_attn weights (Lock #20:
        # AutoRound-preserved precision) plus the drafter push the total
        # past 24 GiB on one Quadro RTX 6000. cpu_offload_gb=4 frees the
        # last few GiB. Only matters for THIS one-shot reference run;
        # the offload round-trips slow individual forwards a bit, but
        # we only need one forward to capture the logits dump.
        cpu_offload_gb=4,
        enforce_eager=True,  # turn off CUDA graphs so the hook fires on every call
        max_num_batched_tokens=1024,
        disable_custom_all_reduce=True,
        max_num_seqs=1,
        disable_log_stats=False,
        seed=42,
        # Single short prompt + 1 generated token — keep KV cache tiny.
        # max_model_len=4096 OOMed the 64-layer target KV alloc on one
        # 24 GiB GPU at TP=1 INT4. 512 is plenty for our forward.
        max_model_len=512,
        speculative_config={
            "method": "dflash",
            "model": args.drafter,
            "num_speculative_tokens": args.block_size,
        },
    )
    print(f"  init: {time.time()-t0:.1f}s", flush=True)

    # Install hook on DFlashQwen3ForCausalLM.compute_logits via collective_rpc.
    # The hook stashes the FIRST call's output into the worker's state, then
    # detaches subsequent calls. We retrieve the stashed tensor via a second
    # collective_rpc after generate completes.
    target_layer_ids = [1, 16, 31, 46, 61]
    def worker_install_hook(self):
        import torch
        # Find the drafter on this worker. Walk the speculative_worker /
        # proposer / runner / model tree.
        runner = self.model_runner
        drafter = None
        # Common vLLM v1 path: drafter is the spec_decode proposer's model.
        for path_attr in ("speculative_worker", "spec_decode_worker", "drafter_runner"):
            sd = getattr(runner, path_attr, None)
            if sd is not None and hasattr(sd, "model"):
                drafter = sd.model
                break
        if drafter is None:
            # Fallback: scan named_modules of self for the DFlash class.
            for name, mod in (self.model_runner.model).named_modules():
                if mod.__class__.__name__ == "DFlashQwen3ForCausalLM":
                    drafter = mod
                    break
        if drafter is None:
            # Final fallback: walk worker-wide for the class.
            def _walk(obj, depth=0, seen=None):
                if seen is None:
                    seen = set()
                if id(obj) in seen or depth > 5:
                    return None
                seen.add(id(obj))
                if obj.__class__.__name__ == "DFlashQwen3ForCausalLM":
                    return obj
                for attr in dir(obj):
                    if attr.startswith("_"):
                        continue
                    try:
                        sub = getattr(obj, attr)
                    except Exception:
                        continue
                    if not isinstance(sub, (torch.nn.Module, object)):
                        continue
                    if isinstance(sub, (int, float, str, bytes, list, tuple, dict)):
                        continue
                    result = _walk(sub, depth + 1, seen)
                    if result is not None:
                        return result
                return None
            drafter = _walk(self)
        if drafter is None:
            return {"error": "could not find DFlashQwen3ForCausalLM"}

        # Stash a list to collect each compute_logits invocation. We keep
        # only the first call to minimise memory pressure.
        self._dflash_logit_captures = []
        original_compute_logits = drafter.compute_logits.__func__  # unbound

        def patched_compute_logits(self2, hidden_states):
            result = original_compute_logits(self2, hidden_states)
            if len(self._dflash_logit_captures) == 0:
                # Capture as fp32 cpu numpy.
                try:
                    arr = result.detach().to(torch.float32).cpu().numpy()
                    h_arr = hidden_states.detach().to(torch.float32).cpu().numpy()
                    self._dflash_logit_captures.append(
                        {"logits": arr, "hidden": h_arr}
                    )
                except Exception as e:
                    self._dflash_logit_captures.append({"error": str(e)})
            return result

        import types
        drafter.compute_logits = types.MethodType(patched_compute_logits, drafter)

        # Also install hooks on the TARGET model to capture per-source-
        # layer hidden states. These feed our kernel pipeline via
        # combine_features (T3) and are the input to which the drafter
        # logits comparison binds. Mirror the dflash-extract-vllm.py
        # hook strategy: walk decoder layers, register forward hooks
        # that sum (hidden_states, residual) and capture as fp32 cpu np.
        target_model = self.model_runner.model
        best = None
        best_path = None
        for nm, mod in target_model.named_modules():
            if not isinstance(mod, torch.nn.ModuleList):
                continue
            if len(mod) == 0:
                continue
            sample = mod[0]
            attr_set = set(name for name, _ in sample.named_children())
            looks_like_decoder = (
                ("mlp" in attr_set or "feed_forward" in attr_set) and
                ("self_attn" in attr_set or "attention" in attr_set or
                 "linear_attn" in attr_set or "attn" in attr_set)
            )
            if not looks_like_decoder:
                continue
            if best is None or len(mod) > len(best):
                best = mod
                best_path = nm
        self._dflash_target_captures = {}
        self._dflash_target_hook_handles = []
        if best is not None:
            print(f"[worker] target layers at {best_path!r}  n={len(best)}", flush=True)
            target_captures = self._dflash_target_captures
            def make_target_hook(layer_id):
                def hook(module, args_, output):
                    if isinstance(output, tuple) and len(output) == 2 and \
                       isinstance(output[0], torch.Tensor) and \
                       isinstance(output[1], torch.Tensor):
                        t = (output[0] + output[1])
                    else:
                        t = output[0] if isinstance(output, tuple) else output
                    if isinstance(t, torch.Tensor):
                        arr = t.detach().to(torch.float32).cpu().numpy()
                        target_captures[layer_id] = arr
                return hook
            for il in target_layer_ids:
                if 0 <= il < len(best):
                    self._dflash_target_hook_handles.append(
                        best[il].register_forward_hook(make_target_hook(il)))
        self._dflash_drafter_ref = drafter
        return {
            "installed": True,
            "drafter_class": type(drafter).__name__,
            "draft_vocab_size": getattr(drafter.config, "draft_vocab_size", None),
            "vocab_size": getattr(drafter.config, "vocab_size", None),
        }

    rpc_install = llm.collective_rpc(worker_install_hook)
    print(f"  hook install: {rpc_install}", flush=True)
    if rpc_install and rpc_install[0].get("error"):
        print(f"  ERROR installing hook: {rpc_install[0]['error']}", file=sys.stderr)
        return 1

    # rpc to RESET captures between prompts (clears the FIRST-call
    # gate on compute_logits and clears the per-layer target captures).
    def worker_reset(self):
        self._dflash_logit_captures = []
        self._dflash_target_captures.clear()
        return {"reset": True}

    # rpc to COLLECT current captures (does NOT remove hooks — keeps
    # them live for the next prompt).
    def worker_collect_iter(self):
        return {
            "captures": list(getattr(self, "_dflash_logit_captures", [])),
            "target_captures": dict(getattr(self, "_dflash_target_captures", {})),
        }

    # Build the prompt list. --prompts-file overrides --prompt.
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"  loaded {len(prompts)} prompts from {args.prompts_file}", flush=True)
    else:
        prompts = [args.prompt]

    sp = SamplingParams(temperature=0.0, max_tokens=1, seed=42)
    per_prompt_meta = []

    for prompt_idx, prompt in enumerate(prompts):
        # Reset captures so the per-prompt compute_logits + target hooks
        # fire fresh for this prompt.
        llm.collective_rpc(worker_reset)

        t1 = time.time()
        outs = llm.generate([prompt], sp)
        print(f"  [p{prompt_idx}] generate: {time.time()-t1:.2f}s; "
              f"n_prompt_tokens={len(outs[0].prompt_token_ids)}", flush=True)

        rpc_collect = llm.collective_rpc(worker_collect_iter)
        if not rpc_collect or not rpc_collect[0].get("captures"):
            print(f"  [p{prompt_idx}] ERROR: no captures collected", file=sys.stderr)
            return 1
        cap = rpc_collect[0]["captures"][0]
        if "error" in cap:
            print(f"  [p{prompt_idx}] ERROR in capture: {cap['error']}", file=sys.stderr)
            return 1

        logits = cap["logits"]
        hidden = cap["hidden"]
        target_caps = rpc_collect[0].get("target_captures", {})

        # Per-prompt subdir if multi-prompt mode, else flat (backward compat).
        prompt_dir = out_dir / f"prompt-{prompt_idx}" if len(prompts) > 1 else out_dir
        prompt_dir.mkdir(parents=True, exist_ok=True)

        np.save(prompt_dir / f"drafter-logits-bs{args.block_size}-vllm.npy", logits)
        np.save(prompt_dir / f"drafter-hidden-bs{args.block_size}-vllm.npy", hidden)
        np.save(prompt_dir / "drafter-prompt-tokens.npy",
                np.array(outs[0].prompt_token_ids, dtype=np.int64))

        gen_token_ids = list(outs[0].outputs[0].token_ids)
        if gen_token_ids:
            bonus_token_id = int(gen_token_ids[0])
            np.save(prompt_dir / "drafter-bonus-token.npy",
                    np.array([bonus_token_id], dtype=np.int64))
            print(f"  [p{prompt_idx}] bonus token = {bonus_token_id}", flush=True)
        else:
            bonus_token_id = None

        n_target_layers_captured = 0
        for il, arr in target_caps.items():
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            np.save(prompt_dir / f"target-layer{il}-bs{args.block_size}-vllm.npy",
                    arr.astype(np.float32))
            n_target_layers_captured += 1

        per_prompt_meta.append({
            "prompt_idx": prompt_idx,
            "prompt": prompt,
            "n_prompt_tokens": len(outs[0].prompt_token_ids),
            "bonus_token_id": bonus_token_id,
            "n_target_layers_captured": n_target_layers_captured,
            "logits_shape": list(logits.shape),
            "hidden_shape": list(hidden.shape),
        })
        print(f"  [p{prompt_idx}] wrote dump to {prompt_dir}", flush=True)

    # Per-prompt meta JSON at top level
    meta = {
        "block_size": args.block_size,
        "n_prompts": len(prompts),
        "drafter_path": args.drafter,
        "target_path": args.target,
        "vllm_drafter_class": rpc_install[0].get("drafter_class") if rpc_install else None,
        "per_prompt": per_prompt_meta,
    }
    # For backward compat when single-prompt: also save fields the
    # closure test reads at out_dir/* (already done above).
    if len(prompts) == 1:
        # Single-prompt mode — files were written directly to out_dir.
        pass
    meta_path = out_dir / f"drafter-meta-bs{args.block_size}.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  wrote {meta_path}", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
