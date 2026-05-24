---
name: Split K cache PPL regression on Qwen3.5
description: Split K by RoPE boundary with q4_0 static dims causes 3x PPL regression on Qwen3.5 — not viable
type: project
originSessionId: a5a02c13-3d1d-4998-badf-36c74b87c680
---
Split K cache (polaris-hybrid-cpu-opt branch, commit 36669046a) partitions K at n_rot
boundary into rope and static tensors with independent quant types.

**Why:** Qwen3.5 has n_rot=64 of head_dim=256 (25% RoPE). Theory: pre-RoPE storage
preserves outlier structure for better quantization on the static 75%.

**Initial result (BUGGY, commit 36669046a):** Catastrophic PPL regression:
- f16:q4_0: PPL 15.92 (+188%) — caused by uninitialized RoPE positions + interleaved layout

**After bugfix (commit 4aecb3e40):** Split K works correctly:
- f16 baseline: PPL 5.536
- f16:q4_0: PPL 5.559 (+0.4%)
- q8_0:q4_0: PPL 5.559 (+0.4%) — 2.9x K compression at negligible quality cost
- q4_0 uniform: PPL 5.591 (+1.0%)

**How to apply:** Split K is viable after the bugfix. Merge polaris-hybrid-cpu-opt
(at 4aecb3e40+) into vulkan-phase4 for K cache compression benefits on Vega.
