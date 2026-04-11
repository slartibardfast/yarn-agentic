# PHASE 11: Transfer Queue Utilization During Graph Execution

**Goal**: Route inter-device intermediate tensors through dedicated DMA transfer queues, overlapping data movement with compute shaders.

## Status: PLANNED

## Problem

The Vulkan backend correctly discovers and creates separate transfer queues (DMA engines) on GPUs that support them. However, these are only used for explicit `set_tensor`/`get_tensor` operations — never during graph execution.

During multi-GPU graph compute, intermediate tensors that need to move between devices are copied through the compute queue, stalling compute shaders while DMA occurs.

## Proposed Changes

**`ggml/src/ggml-vulkan.cpp`** (~80 lines):

1. Identify inter-device copy nodes in the compute graph:
   - During `graph_compute`, detect when a source tensor lives on a different device than the destination
   - Route these copies through `device->transfer_queue` instead of `device->compute_queue`

2. Insert timeline semaphore dependencies:
   - Compute queue signals semaphore after producing the source tensor
   - Transfer queue waits on that semaphore before copying
   - Compute queue waits on transfer completion semaphore before consuming the copied tensor

3. Separate command buffer for transfer work:
   - Create a transfer-specific command buffer per graph execution
   - Submit to transfer queue with semaphore dependencies
   - Allows DMA engine to work independently of compute units

## Design Notes

- Not all GPUs have separate transfer queues. AMD discrete GPUs (GCN, RDNA) typically have 1-2 dedicated DMA engines. Intel and lavapipe may not.
- When `device->single_queue == true`, fall back to compute queue (current behavior)
- Timeline semaphores enable fine-grained dependencies without pipeline barriers
- This is the Vulkan equivalent of CUDA streams for overlapping copy and compute

## Dependencies

- Phase 7 (async copies) provides the non-blocking copy infrastructure
- Phase 10 (async graph compute) enables the scheduler to submit overlapping work

## Expected Impact

- Hides DMA latency behind compute: while GPU computes layer N, DMA engine prefetches layer N+1's inputs from the other device
- Most impactful for bandwidth-bound operations (small matmuls, attention)
- Less impactful for compute-bound operations (large matmuls) where DMA time << compute time
- On AMD RDNA: ~10-15% improvement for models where inter-device transfer is on the critical path

## Verification

- Identical output to Phase 10 baseline
- GPU profiling (radv perf counters or `VK_EXT_calibrated_timestamps`) confirms overlap between compute and transfer queues
- No validation layer synchronization errors
