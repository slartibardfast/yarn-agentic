# Multi-GPU Per-Step Checkpoint for DeltaNet MTP

Per-step checkpoint enables speculative decoding rollback without
re-decoding. On single-GPU, it saves each draft step's DeltaNet
recurrent state (SSM + conv) and restores the accepted position
directly — a ~36 ms savings per speculative cycle. On multi-GPU
graph split, this path was disabled: all recurrent state tensors
are split across devices, and the checkpoint code skipped them.

This document describes the fix that enables per-step checkpoint
on graph-split configurations.

## The problem

Qwen3.6 27B has 64 layers. 48 are DeltaNet recurrent (pattern:
`(i+1) % 4 != 0`). Each recurrent layer carries two pieces of
state:

| State | Dimension (27B) | Size |
|-------|-----------------|------|
| SSM state | 786,432 | ~3 MiB |
| Conv state | 30,720 (= 10,240 × 3) | ~120 KiB |

On 2-GPU graph split (`--split-mode graph --tensor-split 1,1`),
every recurrent state tensor `s_l[il]` has `extra != nullptr`
pointing to a `ggml_split_tensor_t` with per-device sub-tensors.
The existing `per_step_alloc` checked:

```cpp
if (s_l[il]->extra != nullptr) continue;  // skip split tensors
```

All 48 layers skipped. Per-step disabled. The server fell back to
`GPU_FALLBACK`: save full state, then re-decode all accepted tokens
through the entire model. At draft depth 5 with 85% acceptance,
that is ~4 tokens × 9 ms/token = ~36 ms per cycle.

Without per-step, MTP draft depths ≥ 3 regressed below baseline:

| Config | Throughput | vs baseline |
|--------|-----------|-------------|
| No MTP | 33.5 t/s | — |
| d=1 | 35.0 t/s | +4.5% |
| d=3 | 23.1 t/s | −31% |
| d=5 | 23.4 t/s | −30% |

## The fix

Three changes in `src/llama.cpp`, one in `src/llama-delta-net.cpp`,
one in `src/llama-context.h`.

### 1. Per-device checkpoint allocation

`per_step_alloc` no longer skips split layers. For each split
layer, it iterates the per-device sub-tensors and allocates
per-device SSM and QKV checkpoint buffers.

The per-device dimensions are derived by proportional decomposition.
The state tensor packs `[conv_state | ssm_state]` contiguously, and
both scale linearly with the number of value heads assigned to each
device:

```
state_total   = ssm_state_dim + conv_state_dim
state_total_d = split->splits[d]->ne[0]

ssm_dim_d     = ssm_state_dim × (state_total_d / state_total)
conv_dim_d    = state_total_d − ssm_dim_d
qkv_dim_d     = conv_dim_d / (d_conv − 1)
```

The original code used `ssm_dim_d = state_total_d` (the full
per-device state size) as the SSM checkpoint dimension, and the
same for QKV. This overestimated QKV by ~80× (408K vs actual 5K
per device), causing OOM at allocation time — the QKV checkpoint
alone tried to reserve ~3 GiB per GPU.

With correct dimensions, total per-step buffer is ~292 MiB per GPU.

### 2. CUDA fast path for split-layer restore

The existing CUDA fast path called
`ggml_backend_cuda_per_step_restore_layers` with pointer arrays
covering all layers in a single kernel launch. It checked
`s_l[il]->extra != nullptr` and bailed to the host fallback for
any split layer.

The new code adds a second dispatch block that handles split layers:
it discovers which CUDA devices participate, computes per-device
dimensions using the same proportional decomposition, builds
per-device pointer arrays (dst from `split->splits[d]`, SSM/QKV
from `per_step_ssm_split[il][d]`, shadow from
`split_s_l_shadow[idx][d]`), and calls the restore kernel once per
device.

This eliminates the ~326 MiB/cycle of PCIe D2H+H2D transfers that
the host fallback was doing (read all per-step state to host, read
shadow conv state to host, reconstruct, write back to each device).

### 3. Host fallback for non-CUDA backends

The host fallback path was also extended with split-layer support.
It reads each device's per-step SSM state and QKV features, reads
the shadow conv state, reconstructs the conv state on the host, and
writes the result back to each device's sub-tensor. The conv state
reconstruction logic was factored into a shared lambda used by both
split and non-split paths.

### 4. Graph builder wiring (llama-delta-net.cpp)

The split-device `build_qkv` path now receives per-device
`per_step_ssm_split` and `per_step_qkv_split` tensors. During the
forward pass, each device's sub-computation saves its SSM state
and QKV features to the corresponding per-device checkpoint buffer,
enabling restore at any accepted step.

### 5. Init gate (llama_spec_ckpt_init)

Split tensors no longer disable per-step mode. Instead of returning
`false` when `sl->extra != nullptr`, the init function checks each
device's sub-tensor independently for GPU vs CPU residency.

## Results

| Config | Before | After | Change |
|--------|--------|-------|--------|
| No MTP | 33.5 t/s | 33.5 t/s | — |
| d=1 | 35.0 t/s | 35.3 t/s | +1% |
| d=3 | 23.1 t/s | 32.5 t/s | +41% |
| d=5 | 23.4 t/s | 32.4 t/s | +38% |

The d≥3 regression is eliminated. d=1 remains optimal on this
hardware because MTP draft acceptance drops sharply past the first
draft (86% → 59–63%), so the extra draft steps cost more compute
than they save in verify-step tokens.

## Remaining work

**Why d=1 wins over d=5 despite working per-step checkpoint.** The
per-draft scheduling overhead (graph build + alloc + compute per
step) still dominates. Two approaches identified in the plan:

1. **KQ_mask bucketing** — pad `n_kv` to bucket boundaries so
   consecutive draft steps reuse the same graph shape, enabling
   `can_reuse_graph()` to fire.
2. **Fused multi-draft graph** — build a single cgraph that chains
   N draft steps, amortizing scheduling overhead.

**Pipelining** — MTP tail (~5 ms) and DeltaNet state advancement
are sequential but use separate contexts. Could overlap on separate
GPUs for ~5 ms/cycle savings.

## Hardware

2× Quadro RTX 6000 (TU102, sm_75, 24 GiB each), CUDA 13.2,
`--split-mode graph --tensor-split 1,1`, 262K context.

Model: Qwen3.6 27B IQ4_XS with q4_0 Hadamard KV cache.
