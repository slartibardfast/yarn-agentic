---
name: master b768f0843 is the polaris MTP squash-merge
description: The "fix: recurrent state copy_cell" commit on llama.cpp master is actually a 94-file 12,364-line squash-merge of polaris MTP work onto master. Misleading subject line.
type: project
originSessionId: 9730b98b-a48a-46ed-a147-f48c8cb9810f
---
On llama.cpp master, commit `b768f0843` ("fix: recurrent state copy_cell treats 2D tensor as 1D, corrupting multi-slot state", David Connolly, 2026-04-14) **is not just the bug fix its subject describes**. It's a 94-file, 12,364-insertion squash-merge that bundles:

- Full MTP stack (`COMMON_SPECULATIVE_TYPE_MTP`, `common_speculative_state_mtp`, `--spec-type mtp` CLI, `build_mtp_head` for both `qwen35.cpp` and `qwen35moe.cpp`, `LLAMA_MTP_ROLLOUT` env, DeltaNet snapshot/restore, server wire-up, MTP test files)
- Split-K cache work
- TURBO_KV_4B Vulkan tests
- TQ_KV_1B tests
- The `copy_cell` bug fix the subject line names

**Why:** This rewrites the picture of "where MTP lives." Auto-memories from before 2026-04-14 say MTP is only on `polaris-hybrid-cpu-opt` or `ik_llama.cpp`. After 2026-04-14, master has it too, with the residual_window stack landed on top. The commit message hides this — `git log --oneline` makes b768f0843 look like a small bug fix.

**How to apply:** When asked about MTP on master, do not trust commit subject lines alone. Run `git log -S"COMMON_SPECULATIVE_TYPE_MTP" master` (or grep for `build_mtp_head`, `--spec-type mtp`) to find the actual provenance. Master ALSO has `attn_output_gate` support since the same commit (qwen35moe.cpp:138 `build_layer_attn` projects wq to 2x dim, splits into Q+gate, applies `ggml_fused_sigmoid_mul`). PHASE28 iteration 24 (2026-04-24) confirms master's MTP works end-to-end with `--spec-type mtp` on Vulkan.
