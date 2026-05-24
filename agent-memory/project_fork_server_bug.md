---
name: Fork server recurrent state bug (RESOLVED)
description: llama.cpp fork server garbled output for hybrid recurrent models — root cause was copy_cell treating 2D tensor as 1D, corrupting state on checkpoint copy
type: project
originSessionId: a5a02c13-3d1d-4998-badf-36c74b87c680
---
## Bug
Fork server garbled output: `"if of,ques about water..."` on Qwen3.5 at n_parallel >= 2.
CLI works perfectly at 85 t/s on same model/GPU.

## Root Cause (confirmed 2026-04-13)
TWO bugs in `copy_cell()` (our addition to upstream's `llama-memory-recurrent.cpp`):

### Bug 1: 2D tensor treated as 1D (garbled output)
State tensors are `ggml_new_tensor_2d(ctx, type, n_embd_s, mem_size)` → `ne[0] = n_embd_s`.
`copy_cell` computed `cell_elements = ne[0] / size = n_embd_s / mem_size`.
For mem_size=2: copies only HALF the state to an offset halfway through row 0,
corrupting the active state's second half with a copy of its first half.

**Fix:** Use `ne[0]` directly as cell_elements and `nb[1]` as cell_bytes (row stride).

### Bug 2: Checkpoint overflow (server crash on second request)
The `has_cell=true` path created checkpoint copies that filled ALL cells,
leaving none for other sequences. With np=4 and size=4, 3 decode steps
fill all cells. A different slot's request then hits
`GGML_ASSERT(empty_cell.is_empty())`.

**Fix:** Removed checkpoint creation entirely from `has_cell=true` path,
matching upstream behavior. Speculative decoding rollback handled by
`seq_rm` returning false.

## Verification
- 0.8B F16, np=4: 5 sequential requests, all correct (1+1=2 through 5+5=10)
- 9B Q4_K_M, np=4: 5 sequential requests, all correct
- Long generation (100 tokens): coherent output, no garbling

## File
`src/llama-memory-recurrent.cpp` — copy_cell() and find_slot()
