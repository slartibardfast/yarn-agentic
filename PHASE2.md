# PHASE 2: Implement Vulkan Split Buffer Type

**Goal**: Create `ggml_backend_vk_split_buffer_type()` that distributes tensor rows across multiple Vulkan devices.

## Status: COMPLETE

## Changes Made

**New file: `ggml/src/ggml-vulkan-multigpu.cpp`** (~250 lines):
- Uses only public ggml-backend APIs — no Vulkan headers needed
- `ggml_backend_vk_split_buffer_type_context` — placeholder for split ratios
- `ggml_backend_vk_split_buffer_context` — manages per-device allocations
- `init_tensor` — allocates per-device buffers via `ggml_backend_vk_buffer_type(i)` + `ggml_backend_buft_alloc_buffer()`
- `set_tensor` — copies host data to each device using per-device buffer's `set_tensor` (handles dim-0 row-interleaved quants, dim-1 splits, and replicated tensors — same logic as CUDA)
- `get_tensor` — not implemented (matches CUDA)
- `get_alloc_size` — sums padded sizes across devices
- `ggml_backend_buft_is_vk_split()` — exported for use by `ggml_backend_vk_supports_buft`

**`ggml/include/ggml-vulkan.h`** (1 line):
- Added `ggml_backend_vk_split_buffer_type()` declaration

**`ggml/src/ggml-vulkan.cpp`** (~5 lines):
- Updated `ggml_backend_vk_supports_buft` to accept split buffer type via `ggml_backend_buft_is_vk_split()`

**`ggml/src/CMakeLists.txt`** (1 line):
- Added `ggml-vulkan-multigpu.cpp` to `GGML_SOURCES_VULKAN`

**`src/llama.cpp`** (~7 lines):
- Added `#ifdef GGML_USE_VULKAN` block in `llama_default_buffer_type_split` returning `ggml_backend_vk_split_buffer_type()`
- Removed `LLAMA_SPLIT_MODE_GRAPH` from the Vulkan rejection check (only `LLAMA_SPLIT_MODE_ATTN` is now rejected)

## Key Design Choice
The split buffer implementation uses only public ggml-backend APIs (buffer type allocation, `set_tensor`/`get_tensor` via buffer interface). It does NOT require Vulkan headers or access to internal Vulkan types. This makes it self-contained, easy to review, and upstreamable independently.

## Verification
- `ggml-vulkan-multigpu.cpp` compiles with zero warnings
- `ggml-vulkan.cpp` compiles with zero warnings
- No changes to CUDA paths
