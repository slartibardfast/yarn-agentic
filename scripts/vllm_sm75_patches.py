"""
vllm_sm75_patches.py — runtime patches for vLLM DFlash on sm_75.

Import this BEFORE importing vllm. Each patch is documented with the
underlying bug, the affected file, and the fix.

These patches live in our repo (not in site-packages/) so they survive
vLLM reinstalls and stay version-controlled. When upstream vLLM ships
the fix, delete the corresponding patch here.
"""

from __future__ import annotations

import torch


def _patch_combine_hidden_states() -> None:
    """Fix dtype mismatch in DFlash drafter's `fc` projection.

    Bug
    ---
    `vllm/model_executor/models/qwen3_dflash.py::combine_hidden_states`
    calls `self.model.fc(hidden_states)`. On sm_75 (no BF16) we load
    the model with dtype="float16", which casts the drafter's fc.weight
    to fp16. But `target_hidden_states` arrives from EagleProposer's
    collection path as fp32 — the multi-layer concat upcasts. The
    matmul then fails:

      RuntimeError: expected mat1 and mat2 to have the same dtype,
                    but got: float != c10::Half

    The original code path assumed BF16, which auto-promotes more
    silently than fp16. Without this cast there is no DFlash on sm_75.

    Fix
    ---
    Cast `hidden_states` to `fc.weight.dtype` before calling.
    """
    from vllm.model_executor.models import qwen3_dflash

    # The class that owns `combine_hidden_states` varies across vLLM
    # versions. Locate it dynamically.
    target_cls = None
    for name in dir(qwen3_dflash):
        obj = getattr(qwen3_dflash, name)
        if isinstance(obj, type) and "combine_hidden_states" in obj.__dict__:
            target_cls = obj
            break
    if target_cls is None:
        raise RuntimeError(
            "vllm_sm75_patches: could not find class owning "
            "combine_hidden_states in vllm.model_executor.models.qwen3_dflash"
        )

    def patched(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.model.use_aux_hidden_state:
            return hidden_states
        needs_squeeze = hidden_states.dim() == 1
        if needs_squeeze:
            hidden_states = hidden_states.unsqueeze(0)
        fc_dtype = self.model.fc.weight.dtype
        if hidden_states.dtype != fc_dtype:
            hidden_states = hidden_states.to(dtype=fc_dtype)
        result = self.model.fc(hidden_states)
        if needs_squeeze:
            result = result.squeeze(0)
        return result

    target_cls.combine_hidden_states = patched
    print(
        f"[vllm-sm75-patch] {target_cls.__name__}.combine_hidden_states: "
        "fp16 cast applied",
        flush=True,
    )


def _patch_flex_attention_view_to_reshape() -> None:
    """Fix .view() on non-contiguous KV cache in FlexAttention backend.

    Bug
    ---
    `vllm/v1/attention/backends/flex_attention.py::FlexAttentionImpl.forward`
    at the DECODER attention-type branch calls:

      key_cache = key_cache.view(-1, self.num_kv_heads, self.head_size)
      value_cache = value_cache.view(-1, self.num_kv_heads, self.head_size)

    On sm_75 (no FA2 or fp16 flashinfer kernels for this shape) vLLM
    falls back to FlexAttention. The KV cache tensor returned by
    `kv_cache.unbind(0)` is not always contiguous when block_size and
    paged-cache strides interact, and torch's .view() requires
    contiguous source:

      RuntimeError: view size is not compatible with input tensor's
                    size and stride (at least one dimension spans
                    across two contiguous subspaces). Use .reshape(...)
                    instead.

    Fix
    ---
    Wrap FlexAttentionImpl.forward to monkey-patch the two .view()
    calls to .reshape(). Since the failing lines are inside a closure,
    we replace the entire forward via a method substitution that
    reaches the same shape via reshape.

    Simpler approach: subclass-aware patch — locate the offending
    function module and replace .view-on-cache with .reshape-on-cache
    in place via attribute substitution.
    """
    import torch

    # Easiest: replace tensor.view with a method that falls back to
    # reshape only when not contiguous. Done per-call via the
    # FlexAttentionImpl.forward wrapper.
    from vllm.v1.attention.backends import flex_attention as fa

    if not hasattr(fa, "FlexAttentionImpl"):
        print(
            "[vllm-sm75-patch] flex_attention: FlexAttentionImpl not "
            "found; skipping (vLLM may have refactored)",
            flush=True,
        )
        return

    cls = fa.FlexAttentionImpl
    original_forward = cls.forward

    def patched_forward(self, *args, **kwargs):
        # Monkey-patch torch.Tensor.view temporarily on a per-call
        # basis. .view() only differs from .reshape() in that .view()
        # rejects non-contiguous inputs; .reshape() copies if needed.
        # Result tensor shape is identical.
        orig_view = torch.Tensor.view

        def safe_view(tensor_self, *shape):
            try:
                return orig_view(tensor_self, *shape)
            except RuntimeError as e:
                if "contiguous subspaces" in str(e) or "size is not compatible" in str(e):
                    return tensor_self.reshape(*shape)
                raise

        torch.Tensor.view = safe_view
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            torch.Tensor.view = orig_view

    cls.forward = patched_forward
    print(
        "[vllm-sm75-patch] FlexAttentionImpl.forward: view->reshape "
        "fallback applied",
        flush=True,
    )


def _patch_flex_attention_block_n_pow2() -> None:
    """Force FlexAttention BLOCK_M/BLOCK_N to powers of 2.

    Bug
    ---
    `vllm/v1/attention/backends/flex_attention.py::get_kernel_options`
    computes `BLOCK_M` and `BLOCK_N` for the Triton kernel. When
    `attn_metadata.direct_build` is True (which it is for DFlash's
    KV-cache layout) the function passes `block_n` straight through
    from `attn_metadata.block_mask.BLOCK_SIZE`, which can be non-POW2.
    Triton then rejects:

      offs_n = kv_start + tl.arange(0, BLOCK_N)
      arange's range must be a power of 2

    Fix
    ---
    Wrap `get_kernel_options` and round `BLOCK_M` / `BLOCK_N` UP to
    the next power of 2 before returning. Larger-than-needed is fine
    (just rounds query coords up); the kernel internally masks oob
    positions.
    """
    import math

    from vllm.v1.attention.backends import flex_attention as fa

    if not hasattr(fa, "get_kernel_options"):
        print(
            "[vllm-sm75-patch] flex_attention: get_kernel_options "
            "not found; skipping",
            flush=True,
        )
        return

    original = fa.get_kernel_options

    def largest_pow2_divisor(n: int) -> int:
        """Largest power of 2 that divides n. For n=864 = 2^5 * 27 -> 32."""
        if n <= 0:
            return 1
        # bit trick: n & -n = lowest set bit = largest pow2 divisor
        return n & -n

    def patched(query, block_m, block_n, use_direct_build):
        # The kernel needs BLOCK_M/BLOCK_N to be (a) POW2 (Triton tl.arange
        # constraint) AND (b) divisor of the logical block_m/block_n (so
        # the kernel grid lines up with the block mask). The intersection
        # is "largest POW2 that divides the input". For typical block_n
        # like 864 (=32*27), that's 32.
        opts = original(query, block_m, block_n, use_direct_build)
        if "BLOCK_M" in opts and isinstance(opts["BLOCK_M"], int):
            pow2_m = largest_pow2_divisor(block_m if use_direct_build else opts["BLOCK_M"])
            opts["BLOCK_M"] = max(1, pow2_m)
        if "BLOCK_N" in opts and isinstance(opts["BLOCK_N"], int):
            pow2_n = largest_pow2_divisor(block_n if use_direct_build else opts["BLOCK_N"])
            opts["BLOCK_N"] = max(1, pow2_n)
        return opts

    fa.get_kernel_options = patched
    print(
        "[vllm-sm75-patch] FlexAttention.get_kernel_options: "
        "BLOCK_M/BLOCK_N -> largest POW2 divisor of input",
        flush=True,
    )


def _patch_precompute_context_kv_dtype() -> None:
    """Cast context_states to weight dtype in precompute_and_store_context_kv.

    Bug (predicted)
    ---------------
    Same shape as the combine_hidden_states bug — qwen3_dflash.py:409
    `precompute_and_store_context_kv` calls:

      ops.rms_norm(normed_context_states, context_states, hidden_norm_weight, eps)
      all_kv_flat = F.linear(normed_context_states, self._fused_kv_weight, ...)

    Both operations expect input dtype to match the weight dtype. On
    sm_75 the weights are fp16 (cast from BF16 at load), but
    context_states is `self._dflash_hidden_states` which was stashed
    from `target_hidden_states` in set_inputs_first_pass without a
    cast. EagleProposer's hidden-state collection path upcasts to
    fp32. dummy_run uses synthetic fp16 buffers so it survives;
    first real verify hits fp32 input → fp16 weight mismatch.

    Fix
    ---
    Cast `context_states` to `self._fused_kv_weight.dtype` at the
    function entry.

    Currently inactive — enable via apply_all() if a future iteration
    surfaces the matching crash.
    """
    from vllm.model_executor.models import qwen3_dflash

    target_cls = None
    for name in dir(qwen3_dflash):
        obj = getattr(qwen3_dflash, name)
        if isinstance(obj, type) and "precompute_and_store_context_kv" in obj.__dict__:
            target_cls = obj
            break
    if target_cls is None:
        print(
            "[vllm-sm75-patch] precompute_and_store_context_kv: class not "
            "found; skipping",
            flush=True,
        )
        return

    original = target_cls.precompute_and_store_context_kv

    def patched(self, context_states, context_positions, context_slot_mapping=None):
        # Ensure context_states is in the same dtype as the KV
        # projection weights. The original code computes _fused_kv_buffers
        # lazily; trigger that path if needed.
        if not hasattr(self, "_fused_kv_weight"):
            self._build_fused_kv_buffers()
        target_dtype = self._fused_kv_weight.dtype
        if context_states.dtype != target_dtype:
            context_states = context_states.to(dtype=target_dtype)
        return original(self, context_states, context_positions, context_slot_mapping)

    target_cls.precompute_and_store_context_kv = patched
    print(
        f"[vllm-sm75-patch] {target_cls.__name__}."
        "precompute_and_store_context_kv: dtype cast applied",
        flush=True,
    )


def apply_all() -> None:
    """Call once at startup, before constructing an LLM instance."""
    _patch_combine_hidden_states()
    _patch_flex_attention_view_to_reshape()
    _patch_flex_attention_block_n_pow2()
    _patch_precompute_context_kv_dtype()
