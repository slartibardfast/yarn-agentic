---
name: vLLM v1 EngineCore is a separate subprocess — runtime monkey-patches don't propagate
description: vLLM v1 (~0.20+) runs the model in a separate EngineCore subprocess that re-imports vLLM fresh. Runtime monkey-patches applied in the LLM-init process don't reach the worker. Three options for landing sm_75 fixes; inline venv source edits are the simplest.
type: reference
originSessionId: 9b52ee2b-d969-442c-83ba-98396525f6fa
---
vLLM v1 spawns the model worker in a separate subprocess (`EngineCore`). Subprocess uses `spawn` (not `fork`) on Linux to avoid CUDA-fork issues; spawn re-imports all of vLLM fresh in the child. Monkey-patches applied in the LLM-init process via `vllm_sm75_patches.apply_all()` modify the parent's class objects but the EngineCore subprocess has its OWN copy.

**Symptom**: `scripts/vllm_sm75_patches.py` reports "patch applied" at LLM-init time, but the bug it's meant to fix still fires inside the EngineCore subprocess at `generate()` time. Stack trace shows the error in `(EngineCore pid=NNNNN)` lines.

**Three options to land sm_75 fixes**:

1. **Inline source edits in the venv** (simplest, user-authorized in T4):
   - Edit `/opt/models/venv-vllm/lib/python3.13/site-packages/vllm/...` directly.
   - Reversible: keep `scripts/vllm_sm75_patches.py` as the documented source of truth.
   - Survives reinstalls only if the venv is rebuilt; not a problem in practice because the venv is project-locked.
   - **Each edit requires explicit user authorization** — Claude Code's auto-mode classifier denies site-packages edits by default. The user phrase "i authorize this direct edit" covers one specific edit; further edits to other files need separate authorization.

2. **Vendor a shadowing PYTHONPATH** (more work, fragile):
   - Copy the relevant module into the repo, patch it, prepend a path that shadows site-packages.
   - Requires recreating the partial `vllm/...` directory tree with `__init__.py` shims.
   - High maintenance cost on vLLM upgrades.

3. **vLLM plugin entry-point** (cleanest if available):
   - vLLM has a plugin system that loads at engine startup in BOTH processes.
   - Requires the patches to be packaged as importable modules with proper entry points.
   - Not investigated for the sm_75 patches yet; possible future migration target.

**The three sm_75 patches that need to land inline (or via a future plugin) in vLLM PR #40898**:

1. `vllm/model_executor/models/qwen3_dflash.py::DFlashQwen3ForCausalLM.combine_hidden_states` — cast `hidden_states` to `self.model.fc.weight.dtype` before matmul. Without it: `RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::Half`. Root: BF16 auto-promotion that doesn't apply at fp16-loaded models on sm_75 (no native BF16).

2. `vllm/v1/attention/backends/flex_attention.py:1116-1117` — `.view()` → `.reshape()` on KV cache from `kv_cache.unbind(0)`. The unbind output is not always contiguous; `.view()` requires contiguous. Same fix as `_patch_flex_attention_view_to_reshape`.

3. `vllm/v1/attention/backends/flex_attention.py::get_kernel_options` (use_direct_build branch) — round `BLOCK_M` / `BLOCK_N` to **largest-pow2-divisor** of the logical block size (`n & -n`). Triton's `tl.arange` requires POW2 AND the kernel grid must divide cleanly into the logical block size. Wrong direction (next-pow2-up) fails the divisibility constraint. Same fix as `_patch_flex_attention_block_n_pow2`.

**Other useful env vars in this stack**:
- `VLLM_ALLOW_INSECURE_SERIALIZATION=1` — required for `collective_rpc` to pass cloudpickled function objects between processes. The patch in `vllm_sm75_patches.py::_patch_serial_utils_allow_functions` is *also* needed because the env var only enables fallback paths; the patch is what actually wires `CUSTOM_TYPE_CLOUDPICKLE` through. With the env var set, vLLM accepts cloudpickled types in the EngineCore subprocess.

**vLLM run config at TP=1 INT4 AutoRound that fits 24 GiB Quadro RTX 6000**:
```
tensor_parallel_size=1
quantization="gptq_marlin"
gpu_memory_utilization=0.92
cpu_offload_gb=4              # need ~4 GiB headroom for activations + BF16 DeltaNet
enforce_eager=True            # turn off CUDA graphs so hooks fire on every call
max_num_batched_tokens=1024
max_num_seqs=1
max_model_len=512             # 4096 OOMs the target's 64-layer KV alloc
disable_custom_all_reduce=True
```

Generate wall-clock: ~3 min total (init + load + warmup + forward). Per-prompt forward after warm: ~5-7 sec.
