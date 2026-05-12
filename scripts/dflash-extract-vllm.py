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

    t0 = time.time()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=2,
        dtype="auto",
        quantization="gptq_marlin",
        gpu_memory_utilization=0.80,
        cpu_offload_gb=0,
        enforce_eager=True,
        max_num_batched_tokens=4096,
        disable_custom_all_reduce=True,
        max_num_seqs=4,
        seed=42,
        max_model_len=4096,
    )
    print(f"init: {time.time()-t0:.1f}s", flush=True)

    # Access the inner PyTorch model. vLLM v1 keeps it on the driver worker.
    engine = llm.llm_engine
    # vLLM v1 path uses model_executor.driver_worker.model_runner.model
    try:
        inner = engine.model_executor.driver_worker.model_runner.model
    except AttributeError:
        # v1 alt path
        inner = engine.engine_core.model_executor.driver_worker.model_runner.model

    # Qwen3.6-27B is Qwen3_5ForConditionalGeneration (multimodal-capable).
    # Decoder layers live at .model.language_model.layers per the HF arch.
    # Walk to find the layers list.
    layers_module = None
    for attr_path in (
        "model.language_model.layers",
        "language_model.layers",
        "model.layers",
        "layers",
    ):
        cur = inner
        ok = True
        for part in attr_path.split("."):
            if hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                ok = False
                break
        if ok and isinstance(cur, torch.nn.ModuleList):
            layers_module = cur
            print(f"found decoder layers at .{attr_path} (n_layers={len(cur)})", flush=True)
            break
    if layers_module is None:
        print("could not find decoder layers ModuleList", file=sys.stderr)
        sys.exit(1)

    # Capture buffers — one dict slot per configured layer.
    captures: dict[int, np.ndarray] = {}

    def make_hook(layer_id: int, is_post: bool):
        def hook(module, args_, output):
            # The "residual" tensor that flows out of a Qwen decoder layer in
            # vLLM is normally `output[0]` for tuple-returning forwards or
            # the tensor itself for tensor-returning forwards. vLLM's Qwen3
            # layer forward returns (hidden_states, residual) for fused
            # residual-norm; in that case hidden_states is post-attn+FFN.
            t = output[0] if isinstance(output, tuple) else output
            if t is None:
                return
            if not isinstance(t, torch.Tensor):
                return
            # Last token's residual? No — we want all positions.
            arr = t.detach().to(torch.float32).cpu().numpy()
            captures[layer_id] = arr
        return hook

    handles = []
    for il in layers:
        if il < 0 or il >= len(layers_module):
            print(f"layer {il} out of range [0,{len(layers_module)-1}]", file=sys.stderr)
            sys.exit(2)
        h = layers_module[il].register_forward_hook(make_hook(il, True))
        handles.append(h)

    # Run a single forward via vLLM.generate with max_tokens=1 so we only
    # do prefill (no autoregressive generation). The hook fires on the
    # prefill pass at all token positions.
    sp = SamplingParams(temperature=0.0, max_tokens=1)
    outs = llm.generate([prompt], sp)
    print(f"generated; prompt token count = {len(outs[0].prompt_token_ids)}", flush=True)

    for h in handles:
        h.remove()

    if not captures:
        print("no captures — hooks did not fire", file=sys.stderr)
        sys.exit(1)

    n_prompt_tokens = len(outs[0].prompt_token_ids)
    if args.n_tokens_cap > 0:
        n_prompt_tokens = min(n_prompt_tokens, args.n_tokens_cap)

    for il in layers:
        if il not in captures:
            print(f"layer {il}: hook fired no data", file=sys.stderr)
            continue
        arr = captures[il]
        # vLLM may flatten over batch in v1 — reshape to [n_tokens, n_embd]
        # if it came in 2-D as [n_tokens, n_embd] (it should). If 3-D
        # [bsz, n_tokens, n_embd], slice batch 0.
        if arr.ndim == 3:
            arr = arr[0]
        # cap to first n_prompt_tokens for parity with ik_llama side.
        if arr.shape[0] > n_prompt_tokens:
            arr = arr[:n_prompt_tokens]
        out_path = os.path.join(args.out_dir, f"vllm-layer{il}.npy")
        np.save(out_path, arr.astype(np.float32))
        print(f"wrote {out_path}  shape={arr.shape}", flush=True)

    # Also dump the prompt token IDs so the compare harness can verify
    # tokenisation parity with ik_llama.
    np.save(os.path.join(args.out_dir, "vllm-prompt-tokens.npy"),
            np.array(outs[0].prompt_token_ids, dtype=np.int64))
    print("done", flush=True)


if __name__ == "__main__":
    main()
