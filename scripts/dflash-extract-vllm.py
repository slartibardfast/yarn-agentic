"""dflash-extract-vllm — dump vLLM-side residual-stream snapshots at the
DFlash drafter's source-layer indices, for a fixed prompt, on the production
Qwen 3.6 27B INT4 AutoRound model. Pairs with examples/dflash-extract from
ik_llama.cpp.

Outputs one .npy float32 file per layer to OUT_DIR with names
  vllm-layer<N>.npy   shape [n_tokens, n_embd]
matching the ik_llama side's naming.

The hook is a pre-forward hook on each requested decoder layer that grabs
the layer's input residual stream (the value that feeds into the next
layer). Why pre-hook on layer (il+1) rather than post-hook on layer il:
in vLLM's model loop the post-hook would still see pre-FFN-residual on
some layer compositions. The pre-hook on layer (il+1) gets the value
*after* residual + FFN of layer il, which is the canonical residual
stream at exit of layer il — matching ik_llama's `l_out-<il>` semantics.
"""

from __future__ import annotations

import os
import sys
import argparse
import time
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF-style snapshot dir for the target")
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--extract-layers", required=True, help="comma-separated layer indices")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-tokens-cap", type=int, default=-1)
    args = p.parse_args()

    os.environ.setdefault("HF_HOME",                  "/mnt/archive/hf-cache")
    os.environ.setdefault("TMPDIR",                   "/opt/models/tmp")
    os.environ.setdefault("VLLM_CACHE_ROOT",          "/opt/models/cache/vllm")
    os.environ.setdefault("TRITON_CACHE_DIR",         "/opt/models/cache/triton")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",  "/opt/models/cache/torch-inductor")
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE","/opt/models/cache/flashinfer")
    os.environ.setdefault("VLLM_LOGGING_LEVEL",       "WARNING")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import vllm_sm75_patches
    vllm_sm75_patches.apply_all()

    layers = [int(x) for x in args.extract_layers.split(",") if x.strip()]
    if not layers or len(layers) > 16:
        print("extract-layers must list 1..16 indices", file=sys.stderr)
        sys.exit(2)

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.prompt_file) as f:
        prompt = f.read()

    from vllm import LLM, SamplingParams

    # vLLM v1 routes the model to a worker process when TP>1 (or always in
    # 0.20+). We use collective_rpc to install forward hooks IN the worker
    # process and capture residuals via shared np arrays returned over RPC.
    # TP=1 single-process executor keeps the model in our process, where
    # standard forward hooks work directly.
    t0 = time.time()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        dtype="auto",
        quantization="gptq_marlin",
        gpu_memory_utilization=0.92,
        cpu_offload_gb=16,
        enforce_eager=True,
        max_num_batched_tokens=4096,
        disable_custom_all_reduce=True,
        max_num_seqs=4,
        seed=42,
        max_model_len=4096,
    )
    print(f"init: {time.time()-t0:.1f}s", flush=True)

    # Access the inner PyTorch model. With TP=1 v1 still uses a worker
    # subprocess via MP; reach into it via collective_rpc.
    engine = llm.llm_engine

    # We install hooks via collective_rpc on the worker and capture the
    # residuals as a list-of-arrays returned from RPC.
    layer_ids = layers
    n_tokens_cap_val = args.n_tokens_cap
    prompt_in = prompt

    def worker_extract(self):
        """Runs inside the vLLM worker process. self = WorkerWrapper."""
        import torch
        import numpy as np
        model = self.model_runner.model

        # Walk named_modules() for the longest ModuleList whose entries
        # carry decoder-layer-shaped attributes (mlp + attn or analogous).
        # Largest ModuleList in the model is the decoder layer stack.
        best = None
        best_path = None
        for nm, mod in model.named_modules():
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
        if best is None:
            # Fallback: just take the longest ModuleList.
            for nm, mod in model.named_modules():
                if isinstance(mod, torch.nn.ModuleList) and \
                   (best is None or len(mod) > len(best)):
                    best = mod
                    best_path = nm
        layers_module = best
        if layers_module is None:
            mlist_names = [(nm, len(mod), [c for c, _ in mod[0].named_children()][:6] if len(mod) else [])
                           for nm, mod in model.named_modules()
                           if isinstance(mod, torch.nn.ModuleList)]
            return {"error": "could not find decoder layers ModuleList",
                    "type": type(model).__name__,
                    "module_lists": mlist_names}
        print(f"[worker] found decoder layers at {best_path!r}  n={len(layers_module)}",
              flush=True)

        captures = {}

        def make_hook(layer_id):
            def hook(module, args_, output):
                # vLLM Qwen3_5DecoderLayer.forward returns (hidden_states,
                # residual) — a fused-residual representation where the next
                # layer's input_layernorm folds them via hidden_states +=
                # residual. The full residual stream at exit of layer il
                # (which is what ik_llama's l_out-<il> captures, post-FFN-
                # and-residual-add) is the SUM of the two outputs.
                if isinstance(output, tuple) and len(output) == 2 and \
                   isinstance(output[0], torch.Tensor) and \
                   isinstance(output[1], torch.Tensor):
                    t = (output[0] + output[1])
                else:
                    t = output[0] if isinstance(output, tuple) else output
                if not isinstance(t, torch.Tensor):
                    return
                arr = t.detach().to(torch.float32).cpu().numpy()
                captures[layer_id] = arr
            return hook

        handles = []
        for il in layer_ids:
            if il < 0 or il >= len(layers_module):
                continue
            handles.append(layers_module[il].register_forward_hook(make_hook(il)))

        # Drive a forward by running the worker's execute_model on a
        # synthetic batch isn't trivial — easier path is to return the
        # hook handles' captures after llm.generate runs. But generate runs
        # asynchronously across workers. Workaround: store hooks on the
        # worker and return the captures dict on a SECOND RPC call.
        self._dflash_captures = captures
        self._dflash_hook_handles = handles
        return {"installed": True, "n_layers": len(layers_module),
                "layer_ids_hooked": [il for il in layer_ids if il < len(layers_module)]}

    rpc_result = llm.collective_rpc(worker_extract)
    print(f"rpc install: {rpc_result}", flush=True)

    # Run a single-pass forward via generate.
    sp = SamplingParams(temperature=0.0, max_tokens=1)
    outs = llm.generate([prompt_in], sp)
    print(f"generated; prompt token count = {len(outs[0].prompt_token_ids)}", flush=True)

    def worker_collect(self):
        """Pull captured residuals back from the worker as a serialisable dict."""
        import numpy as np
        out = {}
        for il, arr in getattr(self, "_dflash_captures", {}).items():
            out[il] = arr  # numpy array, picklable
        # Remove hooks.
        for h in getattr(self, "_dflash_hook_handles", []):
            h.remove()
        return out

    collect_results = llm.collective_rpc(worker_collect)
    # collective_rpc returns a list (one entry per worker). For TP=1 it's
    # length 1; for TP>1 we'd take the first (residuals identical post-AR).
    captures = collect_results[0] if collect_results else {}

    if not captures:
        print("no captures — hooks did not fire", file=sys.stderr)
        sys.exit(1)

    n_prompt_tokens = len(outs[0].prompt_token_ids)
    if n_tokens_cap_val > 0:
        n_prompt_tokens = min(n_prompt_tokens, n_tokens_cap_val)

    for il in layer_ids:
        if il not in captures:
            print(f"layer {il}: hook fired no data", file=sys.stderr)
            continue
        arr = captures[il]
        if arr.ndim == 3:
            arr = arr[0]
        if arr.shape[0] > n_prompt_tokens:
            arr = arr[:n_prompt_tokens]
        out_path = os.path.join(args.out_dir, f"vllm-layer{il}.npy")
        np.save(out_path, arr.astype(np.float32))
        print(f"wrote {out_path}  shape={arr.shape}", flush=True)

    np.save(os.path.join(args.out_dir, "vllm-prompt-tokens.npy"),
            np.array(outs[0].prompt_token_ids, dtype=np.int64))
    print("done", flush=True)


if __name__ == "__main__":
    main()
