---
name: Vulkan MMVQ batch-shape divergence on RDNA2
description: Known RADV/ACO shader-compiler issue — Q8_0 integer-dot-product mul_mat produces batch-shape-dependent output per NUM_COLS pipeline variant on NAVI21
type: project
originSessionId: e06c78f3-43de-46c4-b7cf-81ea2dbd7d8f
---
On **AMD RDNA2** GPUs (NAVI21 / 6800 XT), the Vulkan Q8_0 `mul_mat_vecq` shader produces different f32 output for row 0 depending on which NUM_COLS pipeline variant is dispatched (NUM_COLS=1 vs NUM_COLS=2). Diff ~0.13-0.17 on typical projection shapes (1024→6144, 1024→1024).

**Isolated via**: `tests/test-vulkan-batch-invariance.cpp` (polaris fork). CPU PASS, VEGA10 (GCN5) PASS, NAVI21 FAIL.

**Root cause**: RADV/ACO compiler generates different machine code for separately-compiled shader variants with different NUM_COLS constants — likely FMA fusion or instruction scheduling differences on wave32 that aren't present on wave64 (VEGA). The `precise` GLSL qualifier doesn't suppress this. Sequential muls vs fma are equivalent in IEEE but different in RDNA2 codegen.

**Workaround**: `GGML_VK_DISABLE_MMVQ=1` falls back to the F32 dequantize path (same as VEGA uses), which is batch-shape invariant. Closes the gap from 0.131 to 0.0014 in the model-level test (100× improvement). Small perf cost on Q8_0 decode (MMVQ skipped).

**Impact**: affects MTP-IR intermediate rollback — the intermediate state after token 0 differs from a standalone 1-token decode on NAVI21. Reduces MTP acceptance on the 0.8B (25-40% vs 60% with snapshot+rerun).

**Why:** batch-shape invariance is important for any speculative decoding approach that relies on "state after N tokens in batch == state after N sequential decodes". MTP-IR intermediate rollback, tree-based speculative, and potential future batching optimizations all depend on this.

**How to apply:** use `GGML_VK_DISABLE_MMVQ=1` when benchmarking MTP-IR on NAVI21 until the shader fix lands. Test infrastructure (`test-vulkan-batch-invariance`, `test-intermediate-rollback`) verifies batch invariance on any GPU — add new batch-variant ops to the test as they're discovered.

**Future work**: proper fix would either:
- Force all NUM_COLS variants to produce bit-identical output per column (shader-level precision work)
- Or force a single NUM_COLS pipeline + host-side column looping (host-level dispatch change)
- Filed as MMVQ RDNA2 batch-invariance follow-up
