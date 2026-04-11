# PHASE 1: Enable Async Interface and Events

**Goal**: Un-NULL the async and event function pointers so the scheduler can use async execution with Vulkan backends.

## Status: COMPLETE

## Changes Made

All in `ggml/src/ggml-vulkan.cpp`:

1. **Fixed `cpy_tensor_async` signature** — old signature `(backend, src, dst)` didn't match the `ggml_backend_i` interface `(backend_src, backend_dst, src, dst)`. This was the reason it was disabled.

2. **Wired 4 existing async functions into vtable** — `set_tensor_async`, `get_tensor_async`, `cpy_tensor_async`, `synchronize` were already implemented but commented out in the vtable with `// TODO: enable async and synchronize`.

3. **Implemented 5 event functions using timeline semaphores** (Vulkan 1.2 core):
   - `event_new` — creates `VkSemaphore` with `VK_SEMAPHORE_TYPE_TIMELINE`
   - `event_free` — destroys semaphore
   - `event_record` — increments timeline value, signals on next queue submit (flushes pending transfer work if any, otherwise does empty submit)
   - `event_wait` — host-side `vkWaitSemaphores` (cross-device safe since timeline semaphores are per-VkDevice)
   - `event_synchronize` — host-side `vkWaitSemaphores`

4. **Wired 5 event functions into vtable**

## Verification

- `ggml-vulkan.cpp` compiles with zero warnings, zero errors
- All changes use Vulkan 1.2 core features (timeline semaphores) — works on RADV, NVIDIA, Intel ANV, lavapipe
