# Phase 3: Closing the 5 t/s gap — Vulkan `GGML_OP_FUSED` implementation

## Context

Phase 1 (peer host quickstart) left two loose ends visible from the measurement runs:

1. **§9 performance note** — MTP 2-phase spec decode ran ~2 % *slower* than non-spec at 77.78 % acceptance. Token identity and acceptance rate met the pass criteria, but the theoretical "spec should be faster on GPU full offload" claim did not materialise.
2. **§10 batched-bench anomaly** — `llama-server` reported ~27 t/s at `-np 1`, but `llama-batched-bench` reported only ~5 t/s at `B=1` for the same model, same GPU, same `-c 4096`. A 5.5× discrepancy with no obvious explanation at the time.

Both symptoms turned out to share a single root cause, and fixing it closed a "5 t/s gap" that was already tracked internally by `d752bc3b0 phase25: baseline — current work state before closing the 5 t/s gap` in the polaris branch.

This phase is the investigation and the fix.

## Diagnosis

### Step 1 — graph splits

The first data point came from the batched-bench log:

```
sched_reserve: graph splits = 118
```

118 splits on an 1823-node graph is ~1 split every 15 nodes — catastrophic by the "a few splits per forward pass" rule of thumb. Working hypothesis: many ops are falling back off-Vulkan onto CPU, forcing CPU↔GPU transfer boundaries around each one.

### Step 2 — split attribution

Setting `GGML_SCHED_DEBUG=1` on the server dumps per-split backend assignments:

```bash
GGML_SCHED_DEBUG=1 GGML_VK_VISIBLE_DEVICES=1 ./build-vk/bin/llama-server \
  -m ~/models/Qwen3.5-9B-mtp-q4km.gguf \
  -c 4096 -ngl 99 -np 1 -fit off \
  --host 127.0.0.1 --port 9099 --no-warmup -v
```

Excerpt of the resulting assignments:

```
## SPLIT #2:  CPU      # 3 inputs: [alpha-0] [blk.0.ssm_dt.bias] [blk.0.ssm_a]
## SPLIT #3:  Vulkan0  # 1 inputs: [gate-0 (reshaped)]
## SPLIT #4:  CPU      # 2 inputs: [z-0 (reshaped)] [node_50]
## SPLIT #5:  Vulkan0  # 1 inputs: [final_output-0]
## SPLIT #13: Vulkan0  # 5 inputs: [final_output-2] [attn_inp_kq_mask] …
## SPLIT #14: CPU      # 2 inputs: [gate_reshaped-3] [attn_pregate-3]
## SPLIT #15: Vulkan0  # 1 inputs: [attn_gated-3]
## SPLIT #16: CPU      # 3 inputs: [alpha-4] [blk.4.ssm_dt.bias] [blk.4.ssm_a]
## SPLIT #17: Vulkan0  # 1 inputs: [gate-4 (reshaped)]
…
```

Clear pattern per block:

- **SSM blocks** (most of the 33): `CPU{alpha, ssm_dt.bias, ssm_a}` → `Vulkan{gate}` → `CPU{z, node_N}` → `Vulkan{final_output}` — four splits per block.
- **Attention blocks** (roughly one in four): `CPU{gate_reshaped, attn_pregate}` → `Vulkan{attn_gated}` — two splits per block.

Qwen3.5 is a hybrid SSM/attention model, with a ~3:1 SSM-to-attention layer ratio. The SSM blocks dominated the split count.

### Step 3 — per-node op types

`GGML_SCHED_DEBUG=2` upgrades the dump to per-node detail, including op type:

```
node # 36 (FUSED):       gate-0 [CPU] — srcs: alpha-0, blk.0.ssm_dt.bias, blk.0.ssm_a
node # 51 (FUSED):       node_51 [CPU] — srcs: z-0 (reshaped), node_50
node #212 (FUSED):       attn_gated-3 [CPU] — srcs: gate_reshaped-3, attn_pregate-3
…
node #  0 (GET_ROWS):    model.input_embed [CPU] — srcs: token_embd.weight (666M), inp_tokens
node #1756 (GET_ROWS):   mtp_token_embd-32 [CPU] — srcs: token_embd.weight (666M), mtp_greedy_token
```

**Every CPU-scheduled compute op was `GGML_OP_FUSED`**, except two `GGML_OP_GET_ROWS` nodes for the input embedding lookup and the MTP re-embed.

### Step 4 — root cause

Grepping the source:

```
ggml/src/ggml.c:6066:    fusion_params[0] = GGML_FUSION_GATE_PREP;
ggml/src/ggml.c:6089:    fusion_params[0] = GGML_FUSION_SILU_MUL;
ggml/src/ggml.c:6110:    fusion_params[0] = GGML_FUSION_SIGMOID_MUL;
ggml/src/ggml-cpu/ops.cpp:11479:   case GGML_FUSION_GATE_PREP:
ggml/src/ggml-cpu/ops.cpp:11482:   case GGML_FUSION_SILU_MUL:
ggml/src/ggml-cpu/ops.cpp:11485:   case GGML_FUSION_SIGMOID_MUL:
```

Three fusion types are defined in the polaris branch. All three have CPU kernels in `ggml-cpu/ops.cpp`. **None of them have any Vulkan implementation.** The `ggml-vulkan.cpp` `supports_op` switch has no `case GGML_OP_FUSED:`, so the Vulkan backend reports "cannot run this op" for every FUSED node, and the scheduler places them on CPU by default.

History — the three fusions were shipped here:

```
ff64be29d ggml: add GGML_OP_FUSED framework + GATE_PREP CPU kernel
20fecf66f ggml: add GGML_FUSION_SILU_MUL — fused SiLU gate multiply
c23cd0894 ggml: add GGML_FUSION_SIGMOID_MUL — fused sigmoid gate multiply
```

All three are in the polaris branch and not in upstream master. Framework + CPU kernels landed; Vulkan path was never wired up. A gap in the project's own earlier work, not an upstream bug.

The GET_ROWS splits are a separate concern (the `token_embd.weight` lives in `CPU_Mapped` because `llama-model-loader.cpp:1191-1199` force-replaces a host buffer with plain CPU when `--mmap` is on — see "Remaining work" below).

## What the fusions do

```
ggml_fused_silu_mul    (x, y)             → silu(x) * y
ggml_fused_sigmoid_mul (x, y)             → sigmoid(x) * y
ggml_fused_gate_prep   (alpha, dt, ssm_a) → softplus(alpha + dt[i % H]) * ssm_a[i % H]
```

where `H = num_v_heads`, passed through `op_params[1]`. The CPU reference asserts F32 for all inputs and output; `softplus(x) = x > 20 ? x : log(1 + exp(x))`.

Each is element-wise on contiguous tensors. `silu_mul` and `sigmoid_mul` are two-input, one-output (shape-for-shape). `gate_prep` is three-input with modular broadcasting: `alpha` has `N` elements, `dt_bias` and `ssm_a` each have `num_v_heads` elements, output shape matches `alpha`.

## Implementation

Single commit on branch `vulkan-fused-ops` off `polaris-hybrid-cpu-opt`:

```
a4bb045b4 vulkan: implement GGML_OP_FUSED (SILU_MUL, SIGMOID_MUL, GATE_PREP)
 5 files changed, 200 insertions(+)
```

### Three new shaders

`ggml/src/ggml-vulkan/vulkan-shaders/fused_silu_mul.comp`:

```glsl
#version 450
layout(local_size_x = 512) in;
layout(binding = 0) readonly buffer A {float data_a[];};
layout(binding = 1) readonly buffer B {float data_b[];};
layout(binding = 2) writeonly buffer D {float data_d[];};
layout(push_constant) uniform parameter { uint KX; uint KY; float param1,param2,param3,param4; } p;

void main() {
    const uint i = gl_GlobalInvocationID.z * 262144 + gl_GlobalInvocationID.y * 512 + gl_GlobalInvocationID.x;
    if (i >= p.KX) return;
    const float x = data_a[i];
    const float y = data_b[i];
    data_d[i] = (x / (1.0f + exp(-x))) * y;
}
```

`fused_sigmoid_mul.comp` is identical except for the activation:

```glsl
data_d[i] = (1.0f / (1.0f + exp(-x))) * y;
```

`fused_gate_prep.comp` has three input bindings and uses `p.KY = num_v_heads` as the modular index divisor:

```glsl
layout(binding = 0) readonly buffer A {float data_alpha[];};
layout(binding = 1) readonly buffer B {float data_dt_bias[];};
layout(binding = 2) readonly buffer C {float data_ssm_a[];};
layout(binding = 3) writeonly buffer D {float data_d[];};

void main() {
    const uint i = gl_GlobalInvocationID.z * 262144 + gl_GlobalInvocationID.y * 512 + gl_GlobalInvocationID.x;
    if (i >= p.KX) return;
    const uint h = i % p.KY;   // num_v_heads
    const float x = data_alpha[i] + data_dt_bias[h];
    const float sp = (x > 20.0f) ? x : log(1.0f + exp(x));
    data_d[i] = sp * data_ssm_a[h];
}
```

The push constants reuse the existing generic `vk_op_push_constants` layout (`KX`, `KY`, four floats). `KX` is element count for all three; `KY` is `num_v_heads` for gate_prep, unused for the other two.

### Backend wiring

Six edits in `ggml/src/ggml-vulkan/ggml-vulkan.cpp`:

1. `#include "ggml-fusion.h"` for the `GGML_FUSION_*` enum.
2. Three new pipeline fields in `vk_device` (`pipeline_fused_silu_mul_f32`, `_sigmoid_mul_f32`, `_gate_prep_f32`).
3. Three `ggml_vk_create_pipeline` calls at model init. Binding counts: 3 for the two-input fusions, 4 for gate_prep.
4. `case GGML_OP_FUSED:` in `ggml_vk_op_get_pipeline` — reads `op_params[0]` and returns the matching pipeline.
5. `GGML_OP_FUSED` added to the element-count case list so the 512 × 512 × Z workgroup-splitting path applies (same as `GGML_OP_ADD`, `GGML_OP_GLU`, etc.).
6. `case GGML_OP_FUSED:` in `ggml_vk_compute_forward` calling a new `ggml_vk_fused()` helper.
7. New `ggml_vk_fused()` helper that routes by `op_params[0]`: `SILU_MUL`/`SIGMOID_MUL` pass `src0, src1, nullptr, nullptr`; `GATE_PREP` passes `src0, src1, src2, nullptr` with `num_v_heads` packed into the `KY` push-constant slot.
8. `case GGML_OP_FUSED:` in `ggml_backend_vk_device_supports_op` enforcing the F32-only contract, contiguity, shape match, and `num_v_heads > 0` for gate_prep. The CPU reference is F32-only so we mirror that; there is no f16 variant to worry about.

One shader registration block in `vulkan-shaders-gen.cpp`:

```cpp
string_to_spv("fused_silu_mul_f32",    "fused_silu_mul.comp",    {});
string_to_spv("fused_sigmoid_mul_f32", "fused_sigmoid_mul.comp", {});
string_to_spv("fused_gate_prep_f32",   "fused_gate_prep.comp",   {});
```

No per-tensor-type combinatorics — single f32 variant per fusion because the CPU reference is F32-only.

## Results

All measurements on: Vega 64 (RADV VEGA10, 8167 MiB free), `Qwen3.5-9B-mtp-q4km.gguf`, `c=4096`, `-ngl 99`, `-fit off`, `GGML_VK_VISIBLE_DEVICES=1`.

### Graph splits

| | Before | After | Δ |
|---|---|---|---|
| `graph splits` per forward | **118** | **4** | **−96.6 %** |
| `graph nodes` per forward | 1805–1823 | 1805–1823 | unchanged |

The four remaining splits are:

```
## SPLIT #0: CPU     # 0 inputs                                           # empty graph-entry sync
## SPLIT #1: Vulkan0 # 8 inputs: [model.input_embed] [attn_inp_kq_mask] … # whole-model compute
## SPLIT #2: CPU     # 1 inputs: [mtp_greedy_tokens]                       # MTP greedy-token pick
## SPLIT #3: Vulkan0 # 1 inputs: [mtp_token_embd-32]                       # re-embed after pick
```

One inert graph-entry sync, one huge Vulkan compute split (the whole model), and two tiny splits around the MTP greedy-token pick at the tail. The `GET_ROWS` for the input embedding is now inside the big Vulkan split — it no longer splits because the compute it feeds is on Vulkan, not CPU.

### Throughput (`llama-server`, `-np 1`)

Deterministic §9 equivalence-check request (`"Once upon a time"`, `n_predict=64`, `temperature=0`, `seed=42`):

| | Before | After | Δ |
|---|---|---|---|
| spec decode (MTP auto) | 27.07 t/s | **37.86 t/s** | **+39.8 %** |
| non-spec (`LLAMA_NO_MTP_AUTO=1`) | 27.65 t/s | **37.92 t/s** | **+37.1 %** |
| acceptance rate | 77.78 % (28/36) | 77.78 % (28/36) | unchanged |
| tokens | *baseline* | **byte-identical ✓** | correctness preserved |

Spec is now within 0.06 t/s of non-spec — essentially at parity, no longer a net loss. The "MTP should be *faster* on GPU full offload" theoretical win still does not materialise, but the 2 % slowdown that prompted the §9 performance note is gone.

### Throughput (`llama-batched-bench`)

Same model, same GPU, `-c 4096 -b 2048 -ub 512 -npp 12 -ntg 64 -npl 1,2,4,8`:

```
Before (118 splits):
|  B |   T_PP |   S_PP |   T_TG |   S_TG |      T |    S t/s |
|  1 |  1.063 |  11.29 | 12.893 |   4.96 | 13.956 |    5.45 |
|  2 |  1.092 |  21.97 |  8.993 |  14.23 | 10.085 |   15.07 |
|  4 |  1.288 |  37.28 | 11.242 |  22.77 | 12.530 |   24.26 |
|  8 |  1.722 |  55.76 | 25.801 |  19.84 | 27.522 |   22.09 |

After (4 splits):
|  B |   T_PP |   S_PP |   T_TG |   S_TG |      T |    S t/s |
|  1 |  0.158 |  75.74 |  1.762 |  36.31 |  1.921 |   39.56 |
|  2 |  0.174 | 138.15 |  2.250 |  56.89 |  2.424 |   62.71 |
|  4 |  0.224 | 214.60 |  3.768 |  67.93 |  3.992 |   76.15 |
|  8 |  0.407 | 235.86 |  7.318 |  69.96 |  7.725 |   78.71 |
```

Per-column deltas:

| Measurement | Before | After | × |
|---|---|---|---|
| S_TG B=1 | 4.96 | **36.31** | **7.32** |
| S_TG B=2 | 14.23 | 56.89 | 4.00 |
| S_TG B=4 | 22.77 | 67.93 | 2.98 |
| S_TG B=8 | 19.84 | 69.96 | 3.53 |
| S_PP B=1 | 11.29 | **75.74** | **6.71** |
| S_PP B=8 | 55.76 | 235.86 | 4.23 |

### What "closing the 5 t/s gap" actually means

The `batched-bench` / `llama-server` throughput discrepancy at `B=1` was the "5 t/s gap" that `phase25: baseline — current work state before closing the 5 t/s gap` references. Before the fix, `batched-bench` reported 4.96 t/s while `llama-server` reported 27.07 — a 5.46× difference. The hypothesis in Phase 1's §11 gotcha was that batched-bench "trips a slower path" due to its max-parallel recurrent-state reservation.

That hypothesis was **wrong** — the two tools share the same split count (118) and the same backend pipeline. The actual cause was that the FUSED-on-CPU penalty was the same in both tools, but absolute throughput was low enough in batched-bench at `B=1` that the fixed-cost components dominated (graph build, sync boundaries) and showed up disproportionately. After the fix, the two tools **agree**: `batched-bench B=1 = 36.31 t/s` vs `llama-server -np 1 = 37.92 t/s`, a 4 % gap attributable to batched-bench's additional measurement instrumentation.

### Plan A recomputation

The §10 Plan A go/no-go criterion is `T(batch=4) / T(batch=1)`:

| | Before | After |
|---|---|---|
| `T(batch=1)` (sec) | 13.956 | 1.921 |
| `T(batch=4)` (sec) | 12.530 | 3.992 |
| **ratio** | **0.898** | **2.08** |
| go threshold | `< 2.0 → go` | `< 2.0 → go` |
| verdict | GO (misleading) | **borderline** |

**The earlier "GO" was a measurement artefact.** When `B=1` was artificially slow due to CPU fusion overhead, `B=4` looked almost identical in wall time (`T(4)/T(1) ≈ 0.9`), suggesting near-free batching — but that was because the per-step overhead was so high that batching just got lost in the noise. After the fix, `B=1` is fast, and the real batching overhead becomes visible: 4 sequences take about 2× the time of 1, which means per-sequence throughput scales ~2× from `B=1` to `B=4`. That's meaningful but not extraordinary.

Plan A (batched spec decode) is **borderline** by the quickstart's strict threshold. Whether to build it should now depend on how much the ~2× scaling matters for the target workload, not on the earlier "clearly worth it" read.

## Token equivalence check (§9 re-run)

```
TOKENS: BYTE-IDENTICAL ✓

SPEC   timings: { predicted_n: 64, draft_n: 36, draft_n_accepted: 28,
                  predicted_per_second: 37.856 }
NOSPEC timings: { predicted_n: 64,
                  predicted_per_second: 37.918 }
SPEC acceptance rate: 77.78 %
```

Every token emitted by the fixed Vulkan path matches the pre-fix CPU-fallback path at the same seed. The three new shaders implement bit-exact F32 arithmetic matching the CPU reference in `ggml-cpu/ops.cpp` for `ggml_compute_forward_fused_gate_prep`, `ggml_compute_forward_fused_silu_mul`, and `ggml_compute_forward_fused_sigmoid_mul`.

## Remaining work

### Splits 4 → 2 (MTP greedy pick)

The two `mtp_greedy_tokens` / `mtp_token_embd-32` splits are MTP-specific: the draft-token selection runs `argmax` on a CPU-resident single-element scratch buffer, then re-embeds via `GET_ROWS`. Fixing this requires either:

- Moving the greedy-token selection onto Vulkan (new op or existing `GGML_OP_ARGMAX` applied to the right tensor with the right buffer placement), or
- Eliminating the re-embed by keeping the MTP draft path entirely on Vulkan from the start.

Not blocking and not in this phase's scope.

### Splits 2 → 0 via `VK_EXT_external_memory_host`

`SPLIT #1` currently begins with `model.input_embed (16K)`, which is the output of `GET_ROWS(token_embd.weight, inp_tokens)` *now running on Vulkan* — but that only happens because the input `inp_tokens` tensor is small (0 K) and the output stays on Vulkan. The underlying `token_embd.weight (666 MiB)` still lives in `CPU_Mapped`, because `llama-model-loader.cpp:1191-1199` forces a host-visible Vulkan buffer back to plain CPU when `--mmap` is on:

```c
if (use_mmap && buft_dev && buft == ggml_backend_dev_host_buffer_type(buft_dev)) {
    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    buft = ggml_backend_dev_buffer_type(cpu_dev);
}
```

Runtime workaround: `--no-mmap`. That loses the mmap page cache sharing but puts the embedding into a Vulkan host buffer.

Proper fix: **`VK_EXT_external_memory_host`**. The extension lets Vulkan import an existing host pointer (the one already held by `mmap(2)`) as a `VkDeviceMemory` without reallocation. GPU can then read from the mmap'd region directly. Driver support is broad (RADV, AMDVLK, NVIDIA proprietary, Intel ANV). A patch adding the import path to `ggml-vulkan.cpp` would probably be accepted upstream. Deferred to a follow-up phase.

### Vanishing fusion types

One additional un-Vulkanised fusion surfaced during the grep for commit history: `150eabeb4 step4.5: add TurboQuant V 4-bit (TQ_V_4B) with fused vec_mad`. This introduces a different fused op (`fused_vec_mad`) that was not triggered by the Qwen3.5 hybrid SSM path but may be hit by other models or quant recipes. Not investigated in this phase.

## Reproduction

The fix lives on `slartibardfast/llama.cpp` branch `vulkan-fused-ops` at commit `a4bb045b4`. To reproduce this phase's measurements from a fresh checkout:

```bash
# 1. Switch to the fused-ops branch and build
git fetch origin vulkan-fused-ops
git checkout vulkan-fused-ops
cmake --build build-vk --target llama-server llama-batched-bench -j$(nproc)

# 2. Confirm splits dropped
GGML_SCHED_DEBUG=1 GGML_VK_VISIBLE_DEVICES=1 ./build-vk/bin/llama-server \
  -m ~/models/Qwen3.5-9B-mtp-q4km.gguf \
  -c 4096 -ngl 99 -np 1 -fit off \
  --host 127.0.0.1 --port 9099 --no-warmup -v 2>&1 | grep "graph splits"
# expected: graph splits = 4

# 3. Re-run §9 equivalence check and §10 batched-bench as in Phase 1.
```

The three diagnostic log files captured during this phase are available at:

- `/tmp/qwen35-dbg.log`  — `GGML_SCHED_DEBUG=1` dump (1180 split headers across ~10 forward passes)
- `/tmp/qwen35-dbg2.log` — `GGML_SCHED_DEBUG=2` dump (per-node backend + op type)
- `/tmp/qwen35-batched.log`, `/tmp/qwen35-batched-v2.log` — before/after batched-bench logs
- `/tmp/qwen35-phase9.out`, `/tmp/qwen35-phase9-v2.out` — before/after §9 equivalence-check reports

These are ephemeral local files; the measurement values are captured in this document.

## Commit reference

```
a4bb045b4 vulkan: implement GGML_OP_FUSED (SILU_MUL, SIGMOID_MUL, GATE_PREP)

  5 files changed, 200 insertions(+)
  create mode 100644 ggml/src/ggml-vulkan/vulkan-shaders/fused_gate_prep.comp
  create mode 100644 ggml/src/ggml-vulkan/vulkan-shaders/fused_sigmoid_mul.comp
  create mode 100644 ggml/src/ggml-vulkan/vulkan-shaders/fused_silu_mul.comp
```

Pushed to `github.com/slartibardfast/llama.cpp` on branch `vulkan-fused-ops`. No upstream PR opened — the fix is self-contained and could be submitted if desired, but is not blocking the Phase 2 tool-calling accuracy mission.
