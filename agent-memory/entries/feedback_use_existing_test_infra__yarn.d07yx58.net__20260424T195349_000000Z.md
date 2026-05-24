---
name: Use and extend existing test infrastructure
description: Existing test frameworks first for anything they can express; standalone tests only for checks the framework can't cover
type: feedback
originSessionId: c1440a07-71f9-4a7a-a10b-10db7f6c5c11
---
General principle: when adding coverage, use the existing test framework first. Standalone tests are only appropriate for checks the framework cannot express.

Primary instance in this repo: for new ggml quantization types or ops, use `test-backend-ops` — add the type to `all_types[]` and run with `-b Vulkan` / `-b CPU`. Don't write standalone test files for things the framework already tests (MUL_MAT, GET_ROWS, dequant correctness).

The carveout is real but narrow — operations or behaviors the framework cannot express warrant a standalone file. Examples: flash-attention with custom KV cache layouts (`test-turbo-kv-residual-window-harness`, `test-flash-attn-lse-merge`), end-to-end model-load paths that need `llama_init_from_model`, composite graph math that tests a relationship between multiple ops. The filter: ask "could test-backend-ops express this check by adding a type/op entry?" — if yes, use it; if no, standalone is fine.

**Why:** The user caught me writing custom GPU test code (test-turbo-kv-gpu-roundtrip pattern) instead of using the existing framework that handles allocation, dispatch, CPU reference, and comparison automatically. The standalone tests duplicate work and miss edge cases the framework covers.

**How to apply:** For any new ggml type: (1) add to all_types[] in test-backend-ops.cpp, (2) verify CPU: `test-backend-ops -b CPU -o MUL_MAT`, (3) verify GPU: `test-backend-ops -b Vulkan -o MUL_MAT`. Only write standalone tests for operations that test-backend-ops cannot express.
