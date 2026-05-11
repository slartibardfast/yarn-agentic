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


def apply_all() -> None:
    """Call once at startup, before constructing an LLM instance."""
    _patch_combine_hidden_states()
