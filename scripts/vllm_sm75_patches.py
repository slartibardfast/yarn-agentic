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


def apply_all() -> None:
    """Call once at startup, before constructing an LLM instance."""
    _patch_combine_hidden_states()
    _patch_flex_attention_view_to_reshape()
