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
    # script used TP=1 and worked. We mirror that here. At TP=1 the
    # 27B INT4 target fits in 1× 24 GiB GPU at ~14 GiB; drafter adds
    # ~3.3 GiB; cpu_offload_gb absorbs the rest.
    llm = LLM(
        model=args.target,
        tensor_parallel_size=1,
        dtype="auto",
        quantization="gptq_marlin",
        gpu_memory_utilization=0.92,
        cpu_offload_gb=16,
        enforce_eager=True,  # turn off CUDA graphs so the hook fires on every call
        max_num_batched_tokens=4096,
        disable_custom_all_reduce=True,
        max_num_seqs=4,
        disable_log_stats=False,
        seed=42,
        max_model_len=4096,
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

    # Drive a single forward — generate 1 token. The DFlash proposer will
    # run the drafter ONCE (block_size=4 → 5 query positions per slot).
    sp = SamplingParams(temperature=0.0, max_tokens=1, seed=42)
    t1 = time.time()
    outs = llm.generate([args.prompt], sp)
    print(f"  generate: {time.time()-t1:.2f}s; "
          f"n_prompt_tokens={len(outs[0].prompt_token_ids)}", flush=True)

    def worker_collect(self):
        caps = getattr(self, "_dflash_logit_captures", [])
        target_caps = getattr(self, "_dflash_target_captures", {})
        # Free hook handles to stop further captures.
        for h in getattr(self, "_dflash_target_hook_handles", []):
            try:
                h.remove()
            except Exception:
                pass
        return {"captures": caps,
                "target_captures": dict(target_caps),
                "drafter_class": type(getattr(self, "_dflash_drafter_ref", None)).__name__}

    rpc_collect = llm.collective_rpc(worker_collect)
    if not rpc_collect or not rpc_collect[0].get("captures"):
        print("  ERROR: no captures collected from worker", file=sys.stderr)
        return 1

    cap = rpc_collect[0]["captures"][0]
    if "error" in cap:
        print(f"  ERROR in capture: {cap['error']}", file=sys.stderr)
        return 1

    logits = cap["logits"]
    hidden = cap["hidden"]
    print(f"  captured drafter logits: shape={logits.shape}  dtype={logits.dtype}", flush=True)
    print(f"  captured drafter hidden: shape={hidden.shape}  dtype={hidden.dtype}", flush=True)
    print(f"  drafter class: {rpc_collect[0].get('drafter_class')}", flush=True)

    # Save the captured tensors to .npy + meta.json.
    logits_path = out_dir / f"drafter-logits-bs{args.block_size}-vllm.npy"
    hidden_path = out_dir / f"drafter-hidden-bs{args.block_size}-vllm.npy"
    np.save(logits_path, logits)
    np.save(hidden_path, hidden)
    print(f"  wrote {logits_path}", flush=True)
    print(f"  wrote {hidden_path}", flush=True)

    np.save(out_dir / "drafter-prompt-tokens.npy",
            np.array(outs[0].prompt_token_ids, dtype=np.int64))

    # Save target captures (per-source-layer hidden states) too.
    target_caps = rpc_collect[0].get("target_captures", {})
    n_target_layers_captured = 0
    for il, arr in target_caps.items():
        # Squeeze a leading batch dim if present.
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        tpath = out_dir / f"target-layer{il}-bs{args.block_size}-vllm.npy"
        np.save(tpath, arr.astype(np.float32))
        print(f"  wrote {tpath}  shape={arr.shape}", flush=True)
        n_target_layers_captured += 1

    meta = {
        "block_size": args.block_size,
        "n_query_positions": int(logits.shape[0]),
        "draft_vocab_size": int(logits.shape[1]) if logits.ndim >= 2 else None,
        "target_vocab_size": rpc_install[0].get("vocab_size") if isinstance(rpc_install, list) else None,
        "drafter_path": args.drafter,
        "target_path": args.target,
        "prompt": args.prompt,
        "n_prompt_tokens": len(outs[0].prompt_token_ids),
        "logits_dtype": str(logits.dtype),
        "hidden_dtype": str(hidden.dtype),
        "hidden_shape": list(hidden.shape),
        "vllm_drafter_class": rpc_collect[0].get("drafter_class"),
        "target_layers_captured": n_target_layers_captured,
    }
    meta_path = out_dir / f"drafter-meta-bs{args.block_size}.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  wrote {meta_path}", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
