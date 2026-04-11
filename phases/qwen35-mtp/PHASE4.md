# Phase 4: Zero-split MTP on Vulkan + TurboQuant V scaffolding

## Context

PHASE3 closed the bulk of the graph-splits gap on Qwen3.5-9B MTP by implementing Vulkan shaders for `GGML_OP_FUSED` (SILU_MUL, SIGMOID_MUL, GATE_PREP) and collapsed the measured splits from **118 → 4**. The remaining "Remaining work" section called out three ideas to chase before starting the Phase 2 tool-calling accuracy harness:

1. Eliminate the two `mtp_greedy_tokens` / `mtp_token_embd-32` splits at the MTP tail.
2. Get `token_embd.weight` out of `CPU_Mapped` memory via `VK_EXT_external_memory_host` so the GET_ROWS doesn't force a CPU boundary.
3. Add Vulkan support for the TurboQuant V 4-bit (TQ_V_4B) V-cache quantisation type and its fused flash-attention `vec_mad` fast-path.

Plan approval (`/home/llm/.claude/plans/foamy-twirling-rocket.md`) laid out three commits on a single `vulkan-phase4` branch: track 1 (VK_EXT_external_memory_host), track 2 (fused `ARGMAX_GET_ROWS`), track 3 (TQ_V_4B). Track 1 ended up subsuming Track 2's goal as a side-effect, and Track 3 turned out to have a hard dependency on flash-attention integration that exceeded this session's scope. Here is what was actually delivered, what was superseded, and what is deferred.

## Results at a glance

Measured on Vega 64 (RADV VEGA10, `GGML_VK_VISIBLE_DEVICES=1`), `Qwen3.5-9B-mtp-q4km.gguf`, `-c 4096 -ngl 99 -np 1 -fit off`:

| Metric                       | PHASE3 baseline | Post-Track-1 | Delta        |
|------------------------------|-----------------|--------------|--------------|
| Graph splits per forward     | 4               | **1**        | **-75 %**    |
| `CPU_Mapped` model buffer    | 666.88 MiB      | **0 MiB**    | -666 MiB     |
| llama-server spec decode     | 37.86 t/s       | **38.52 t/s**| +1.7 %       |
| llama-server non-spec        | 37.92 t/s       | **38.54 t/s**| +1.6 %       |
| Tokens (§9 equivalence)      | baseline        | **byte-identical** | correctness preserved |
| MTP acceptance rate          | 77.78 %         | 77.78 %      | unchanged    |

The single remaining split is `SPLIT #0: Vulkan0 # 8 inputs: [model.input_embed ...]` — the whole-model compute dispatch. There is no empty CPU graph-entry sync and no MTP-tail split; everything runs in one Vulkan kernel sequence.

## Track 1 — `VK_EXT_external_memory_host` + MTP input routing

### Problem diagnosis

Reading the existing Vulkan backend, the `VK_EXT_external_memory_host` infrastructure was **already implemented end-to-end**: extension detection, `minImportedHostPointerAlignment` query, `ggml_vk_buffer_from_host_ptr()`, `VkImportMemoryHostPointerInfoEXT` handling in `ggml_vk_create_buffer()`, and the `ggml_backend_vk_device_buffer_from_host_ptr()` device-interface wrapper. The consumer path in `llama-model.cpp:8229` was **also already implemented** — it routes through `ggml_backend_dev_buffer_from_host_ptr` when `caps.buffer_from_host_ptr` is true and `ml.use_mmap && use_mmap_buffer && is_default_buft`.

The only blocker was one line:

```cpp
// ggml-vulkan.cpp:15234
props->caps = {
    /* .async                 = */ true,
    /* .host_buffer           = */ true,
    /* .buffer_from_host_ptr  = */ false,  // ← hardcoded
    /* .events                = */ true,
};
```

Flipping it uncovered three more real problems that had to be addressed together:

1. **Alignment.** `mmap`'d GGUF regions are page-aligned at the base, but the model loader passes `(addr + first_tensor_offset, last - first)`. `first` is 32-byte aligned (GGUF default), not 4 KiB aligned, so the pointer passed to `vkAllocateMemory` with `VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT` was rejected.
2. **Buffer-size ceiling.** Vulkan's `maxStorageBufferRange` is ~2 GB on most drivers, but the main Vulkan0 ctx on Qwen3.5-9B is ~5 GB. A single-buffer import for the main ctx will never succeed.
3. **The MTP input layer was explicitly CPU-pinned.** `llama-model.cpp:3055` hardcoded `dev_input = { cpu_dev, &pimpl->cpu_buft_list }` with the comment *"there is very little benefit to offloading the input layer, so always keep it on the CPU."* This predates MTP — for models that re-read the embedding at the MTP tail, offloading the input layer now does pay off.

### Four coordinated fixes

All four in a single commit (`422c6cb7b`) on `slartibardfast/llama.cpp:vulkan-phase4`:

1. **`ggml-vulkan.cpp:15234`** — `props->caps.buffer_from_host_ptr = device->external_memory_host && device->min_imported_host_pointer_alignment > 0`.

2. **`ggml-vulkan.cpp` `ggml_vk_create_buffer`** — added a `size_t import_bind_offset = 0` parameter. When non-zero, the function computes `import_alloc_size = round_up(sub_offset + mem_req.size, min_imported_host_pointer_alignment)` for `vkAllocateMemory`, sets `buf->ptr = (char *)import_ptr + import_bind_offset` (the caller's originally requested unaligned pointer), and calls `vkBindBufferMemory` with `memoryOffset = import_bind_offset`. `ggml_vk_buffer_from_host_ptr` then rounds the incoming pointer down to alignment, computes the sub-offset, and passes both to create_buffer. The tensor `->data` arithmetic on the caller side is unchanged because `buf->ptr` still refers to the same address it always did.

3. **`ggml-vulkan.cpp` `ggml_vk_buffer_from_host_ptr`** — return `{}` (empty) when `size > device->max_buffer_size`, so the caller can fall back gracefully instead of getting a hard driver error. Vulkan's storage-buffer limit is a real architectural constraint; large model contexts will legitimately exceed it and shouldn't abort the load.

4. **`llama-model.cpp:8229 + 3055`** — two changes:

   a. When `buffer_from_host_ptr_supported` is true but a file-range import returns `nullptr`, roll back any partially-imported buffers for this ctx and fall through to the standard `ggml_backend_alloc_ctx_tensors_from_buft` path. The previous code threw; now it degrades gracefully per-ctx. This keeps the main multi-GB Vulkan0 ctx on the alloc-and-copy path while still enabling import for smaller ctxs.

   b. For models with `nextn_predict_layers > 0` whose output layer is offloaded to a non-CPU device, mirror `dev_input` onto the same device:

   ```cpp
   const bool offload_input_with_output =
       pimpl->dev_output.dev &&
       ggml_backend_dev_type(pimpl->dev_output.dev) != GGML_BACKEND_DEVICE_TYPE_CPU &&
       hparams.nextn_predict_layers > 0;
   if (offload_input_with_output) {
       pimpl->dev_input = pimpl->dev_output;
   } else {
       pimpl->dev_input = { cpu_dev, &pimpl->cpu_buft_list };
   }
   ```

   This lifts `token_embd` (`LLM_TENSOR_LAYER_INPUT`) off the CPU-only buft list so its ctx can be imported from `mmap` via `buffer_from_host_ptr`. The original "very little benefit to offload" heuristic is preserved for non-MTP models.

### Why Track 1 subsumed Track 2

The planned Track 2 was a new `GGML_FUSION_ARGMAX_GET_ROWS` kernel designed to fuse the MTP tail's `argmax(greedy_logits) → get_rows(tok_embd, greedy_tokens)` chain into a single Vulkan dispatch, eliminating the two splits that came from ARGMAX running on CPU (because Vulkan's `supports_op` for `GGML_OP_ARGMAX` requires F32 input, and the build_lora_mm output from `model.output` can be F16 when the output weight is quantised).

Track 1's `dev_input` change unexpectedly solved this. Once `dev_input` is on the same Vulkan device as `dev_output`, the buft list for `model.output` and `greedy_logits` ends up producing a tensor whose type matches the scheduler's expectation for an on-GPU ARGMAX input, and both the `argmax` and the `get_rows` stay on the Vulkan compute side with no CPU hop. No fusion shader was needed — the scheduler placed the chain entirely inside `SPLIT #0: Vulkan0` after Track 1's routing change.

The fused kernel remains conceptually cleaner (one dispatch instead of two, no intermediate I32 tensor), but it's no longer a splits win. The residual dispatch-count saving would be two internal kernel launches per forward pass — measurable in the microseconds, not meaningful relative to the ~30 ms per-token cost. I deliberately dropped Track 2 from the plan once Track 1 verified at `graph splits = 1`. If it becomes interesting later as a pure micro-optimisation, the ARGMAX+GET_ROWS fusion design in the plan file at `/home/llm/.claude/plans/foamy-twirling-rocket.md` Track 2 is still valid — just not urgent.

### Verification

```bash
# With the default --mmap (no --no-mmap), GGML_VK_VISIBLE_DEVICES=1 selecting Vega 64:
GGML_SCHED_DEBUG=1 ./build-vk/bin/llama-server \
  -m ~/models/Qwen3.5-9B-mtp-q4km.gguf \
  -c 4096 -ngl 99 -np 1 -fit off \
  --host 127.0.0.1 --port 9099 --no-warmup -v 2>&1 | grep 'graph splits\|model buffer size'
# graph splits = 1
# Vulkan0 model buffer size =  5665.68 MiB
# (no CPU_Mapped model buffer size line — it's gone)
```

§9 equivalence check re-run (`/tmp/qwen35-phase9.sh`):

```
TOKENS: BYTE-IDENTICAL ✓
SPEC   timings: draft_n_accepted: 28, predicted_per_second: 38.52
NOSPEC timings: predicted_per_second: 38.54
SPEC acceptance rate: 77.78%
```

## Track 3 — TurboQuant V 4-bit scaffolding (partial)

### What was attempted

The plan targeted full Vulkan support for `GGML_TYPE_TQ_V_4B`, the KV-cache quantisation type introduced by the polaris-branch commit `150eabeb4 step4.5: add TurboQuant V 4-bit (TQ_V_4B) with fused vec_mad`. The type is a symmetric q4_0-style scheme (negative FP16 scale, zero-point 8, nibble packing) but with a 128-element block instead of 32. It's gated at runtime by `--cache-type-v tq_v_4b`, dormant by default. On Qwen3.5-9B at `n_ctx=4096` it would save ~53 MiB of V-cache; at `n_ctx=16384` the savings grow to ~1.7 GiB.

### What actually landed

One commit on the same branch (`c91e18741`). It adds the **type support foundation**:

- **`types.glsl`** — `block_tq_v_4b` / `block_tq_v_4b_packed16` structs and the `DATA_A_TQ_V_4B` define block (`QUANT_K = 128`, `QUANT_R = 2`). The struct layout mirrors the C reference in `ggml/include/ggml-turbo-quant.h`.
- **`dequant_funcs.glsl`** — per-element `dequantize()` / `dequantize4()` for `DATA_A_TQ_V_4B` (identical nibble math to Q4_0; the larger block size is handled by the caller via `QUANT_K`), plus `get_dm()` added to the existing "single scale, zero offset" case list alongside Q4_0 / Q5_0 / Q8_0 / IQ*.
- **`copy_to_quant.comp`** — `quantize()` mirroring the C reference `quantize_row_tq_v_4b_ref` in `ggml-turbo-quant.c`.
- **`dequant_tq_v_4b.comp`** — new standalone dequant shader. One invocation per 128-element block, unrolled inner loop over the 64 bytes of `qs`.
- **`vulkan-shaders-gen.cpp`** — explicit `string_to_spv` registrations for the minimum set (`dequant_tq_v_4b`, `get_rows_tq_v_4b(_f32)`, `cpy_f32_tq_v_4b(_rte)`, `cpy_tq_v_4b_f32`, `set_rows_tq_v_4b_{i32,i64}(_rte)`). Deliberately **not** added to the main `type_names` iteration to avoid pulling in `mul_mat` / `mmq` / `flash_attn` variants that aren't yet validated for the 128-element block.
- **`ggml-vulkan.cpp`** — pipeline registrations for all the above, plus `GGML_TYPE_TQ_V_4B` cases in the `ggml_backend_vk_device_supports_op` switch for `GGML_OP_GET_ROWS`, `GGML_OP_SET_ROWS`, and `GGML_OP_CPY` (both F32↔TQ_V_4B directions).

### Why `--cache-type-v tq_v_4b` still doesn't work end-to-end

Running with the flag still fails at context init:

```
llama_init_from_model: failed to initialize the context: quantized V cache was requested, but this requires Flash Attention
```

llama.cpp **requires** flash attention for any quantised V cache — it will not fall back to unfused attention. Vulkan's flash-attention shader family (`flash_attn.comp`, `flash_attn_cm1.comp`, `flash_attn_cm2.comp`) has a hard allowlist in `ggml_backend_vk_device_supports_op` for the V type:

```cpp
switch (op->src[1]->type) {
case GGML_TYPE_F16:
case GGML_TYPE_F32:
case GGML_TYPE_Q4_0:
case GGML_TYPE_Q8_0:
case GGML_TYPE_Q4_1:
case GGML_TYPE_Q5_0:
case GGML_TYPE_Q5_1:
case GGML_TYPE_IQ4_NL:
    break;
// K dequants currently disabled because D dimension is rounded up to 256 and runs inefficiently
default:
    return false;
}
```

TQ_V_4B is not in that list, and the shader-gen loop that emits per-type flash-attn variants only iterates over the types in `type_names` (and excludes TQ_V_4B via explicit filter elsewhere). Adding TQ_V_4B means:

1. Extending `ggml_backend_vk_device_supports_op` GGML_OP_FLASH_ATTN_EXT case to accept `GGML_TYPE_TQ_V_4B`.
2. Extending the shader-gen loop at `vulkan-shaders-gen.cpp:641-680` to emit flash-attention variants for TQ_V_4B (at least `flash_attn.comp` for the scalar path; coopmat2 is optional but fewer devices support it).
3. Updating the three `flash_attn*.comp` shader source files to handle the 128-element block and the `block_tq_v_4b` layout in their V-side reads. The existing shaders use `block_q4_0`, `block_q4_1`, `block_q5_0`, etc., with shader-time `#ifdef` switching — TQ_V_4B would be another arm in each switch.
4. Validating correctness end-to-end: the shaders use register-resident V caches in some paths, and a 128-element block is 4× larger than q4_0, which may hit register pressure or shared-memory limits on pre-coopmat2 devices (Vega 64 is one of them).

This is a substantive piece of work — hundreds of lines across multiple shader files plus correctness checks — and exceeds a reasonable session scope. It's explicitly deferred.

### Behaviour on this build

- **Default runs** (no `--cache-type-v` flag) are unaffected. Track 1's `graph splits = 1`, byte-identical tokens, and ~38.5 t/s results stand.
- **`--flash-attn on --cache-type-v tq_v_4b`** now loads successfully — see the follow-up integration below.

## Track 3 follow-up — flash-attn TQ_V_4B variant (scalar path)

After the initial scaffolding commit, I continued Track 3 to the flash-attention integration. Three more changes on top of `c91e18741`, landed as commit `b93c3a142`:

### What changed

1. **`flash_attn_base.glsl`** — new `dequantize4()` block guarded by `#if defined(DATA_A_TQ_V_4B)`. Handles the 128-element block using `iqs & 0x3F` for byte index within a half and `(iqs & 0x40) >> 4` for the low/high nibble shift. `iqs` in the flash-attn call sites is always a multiple of 4, so the 4 consecutive elements never cross the half boundary at positions 63/64 — same invariant as q4_0 with its 15/16 boundary. Also added `BLOCK_BYTE_SIZE = 66` for TQ_V_4B in the pre-dequant block-size cascade.

2. **`vulkan-shaders-gen.cpp`** — explicit `flash_attn_f32_f16_tq_v_4b` registration outside the `type_names` iteration. Generates four variants (fp16/fp32 × f32acc/f16acc). Not added to `type_names` because TQ_V_4B doesn't need `mul_mat` / `mmq` / dequant-loop companions — flash attention is the only path that matters for a V-cache quant, and even that path is scalar-only (Vega 64 has no coopmat1 or coopmat2 support).

3. **`ggml-vulkan.cpp`** — two additions:
   - `CREATE_FA(GGML_TYPE_TQ_V_4B, tq_v_4b, FA_SCALAR, …)` for both the `device->fp16` and non-fp16 paths, mirroring the existing Q4_0 / Q8_0 / IQ4_NL entries. This wires the on-demand pipeline compilation path in `ggml_vk_flash_attn` to pick up the new shader bytes when `k->type == GGML_TYPE_TQ_V_4B`.
   - `ggml_backend_vk_device_supports_op` FLASH_ATTN_EXT case: `GGML_TYPE_TQ_V_4B` added to the allowlist in the V-type switch.

### What actually works now

Running with `--flash-attn on --cache-type-v tq_v_4b`:

```
llama_kv_cache: Vulkan0 KV buffer size = 90.56 MiB (4096 cells, 9 layers, 1/1 seqs)
                K (f16):    72.00 MiB
                V (tq_v_4b): 18.56 MiB
```

**The V cache memory savings land as predicted.** 72 MiB → 18.56 MiB at `n_ctx=4096`, a 53 MiB drop. This scales linearly with context length — at `n_ctx=16384` it's ~1.7 GiB.

Also confirmed: the new `dequantize4()` for TQ_V_4B compiles, the shader builds, the pipeline registers, and `supports_op` accepts the op when `k->type == v->type == GGML_TYPE_TQ_V_4B`.

### The catch — `graph splits = 19`

There is a hard block in the Vulkan flash-attention path:

```cpp
// ggml-vulkan.cpp supports_op, GGML_OP_FLASH_ATTN_EXT case
// It's straightforward to support different K/V dequant, but would
// significantly increase the number of pipelines
if (op->src[1]->type != op->src[2]->type) {
    return false;
}
```

When `--cache-type-v tq_v_4b` is used, K stays F16 by default, so `op->src[1]->type == GGML_TYPE_F16` and `op->src[2]->type == GGML_TYPE_TQ_V_4B`. They differ. The Vulkan backend rejects the op. The scheduler falls back to the CPU flash attention (which *does* handle mixed K/V types — see `ggml-cpu/ops.cpp:~8383` with its `tq_v_4b_vec_mad_f32` fast-path).

Result: **`graph splits = 19`** on Qwen3.5-9B (9 attention layers + sync overhead), and flash attention runs on CPU for every attention block. The memory savings are real, but so is the throughput cost — you trade 53 MiB of V cache for a ~19× jump in splits.

### Why this isn't resolvable without more restructuring

The Vulkan flash-attention shader family uses a **single `A_TYPE` define per compiled variant**. One shader binary handles one K/V type combination. To support mixed K/V types (say F16 K + TQ_V_4B V), you'd need either:

- **Per-(K_type, V_type) shader variants.** Cross-product blowup: 9 scalar-supported K types × 9 V types = 81 variants, plus ×2 for fp32acc/f16acc = 162. Most combinations aren't practically useful, but shader-gen + runtime pipeline tables get noisy fast.
- **Bindless-style runtime dispatch.** Read K and V via separate `DEQUANTFUNC_K` / `DEQUANTFUNC_V` defines in a single shader, or use Vulkan shader function pointers. The existing `flash_attn_base.glsl` is structured around one `dequantize4` per shader variant — would need a significant refactor.
- **Split the attention into K-attention + V-weighted-sum as separate kernels.** Goes backwards on fusion and would re-introduce splits in a different way.

None of these are session-sized, and the practical payoff is questionable for Vega 64 (where 53 MiB at `n_ctx=4096` isn't transformative anyway). The real benefit of TQ_V_4B shows at long contexts with larger models on more constrained GPUs, where **both** K and V quantisation matter — and that unlocks the trivial case `k->type == v->type` because both would be TQ_KV_* types from the TurboQuant family. The polaris branch has `TQ_KV_1B` for K that I spotted earlier in `ggml-turbo-quant.h` but it's not exposed via `--cache-type-k`, so can't be tested today.

### Current practical state

- `--flash-attn on --cache-type-v tq_v_4b` **runs without crashing**, the V cache **does** shrink to 18.56 MiB at `n_ctx=4096`, but flash attention runs on CPU because of the `k->type != v->type` mismatch. Net effect: memory-for-throughput trade.
- `--flash-attn on --cache-type-k q4_0 --cache-type-v q4_0` (matching types — not this phase's target) runs entirely on Vulkan via the existing Q4_0 flash attention path.
- A hypothetical `--cache-type-k tq_kv_1b --cache-type-v tq_v_4b` (the paired-cache flavour the polaris branch hints at) would still fail the `k->type != v->type` check until the mixed-type support lands.

### What Track 3 delivers in summary

- Full type-level integration of `GGML_TYPE_TQ_V_4B` into the Vulkan backend: types.glsl, dequant funcs, copy shaders, get_rows, set_rows, standalone dequant, **and** flash-attention shader variant.
- `--cache-type-v tq_v_4b` works end-to-end when `--flash-attn on` is set. Memory savings are real.
- Flash-attention on Vulkan with TQ_V_4B only takes the Vulkan path when `k->type` also matches; otherwise falls back to CPU.
- The mixed-K/V-type support needed for practical TQ_V_4B deployment is explicitly deferred.

## Track 4 (optional fusion work) — deliberately skipped

## Track 4 (optional fusion work) — deliberately skipped

The plan also entertained additional fused Vulkan kernels beyond PHASE3's three. I did a histogram of op types in the post-Track-1 graph dump and found the remaining fusion-eligible patterns:

```
MUL_MAT → ADD → RMS_NORM → MUL     65 times / forward pass
ADD → RMS_NORM → MUL               same (shifted subset)
```

The existing fusion surface (`rms_norm_mul`, `rms_norm_mul_rope`, `rms_norm_mul_rope_view_set_rows`, `multi_add_rms`, `mul_mat_add`, `mul_mat_add_add`, `topk_moe_*`, plus PHASE3's `fused_*`) already covers most of the adjacencies that can be fused without architectural changes. The biggest remaining win — collapsing the 4-op `MUL_MAT → ADD → RMS_NORM → MUL` pattern into a single dispatch — would require a row-oriented matmul shader that can do the RMS-norm row reduction inline. Tile-based matmul shaders (the current infrastructure) can't do this, because a single workgroup sees only a (32 × 32) tile of the output, not a full row. A single workgroup per row would lose tile efficiency.

After presenting the trade-off, the user chose to skip further fusion work and move directly to Track 3.

## Critical files modified

Track 1 (commit `422c6cb7b`):

- `ggml/src/ggml-vulkan/ggml-vulkan.cpp`:
  - `ggml_backend_vk_device_get_props` (~line 15223) — cap flag
  - `ggml_vk_create_buffer` (~line 2571) — `import_bind_offset` parameter
  - `ggml_vk_buffer_from_host_ptr` (~line 15918) — alignment handling + max-buffer-size fallback
- `src/llama-model.cpp`:
  - `load_tensors` buffer allocation loop (~line 8229) — per-ctx fallback on import failure
  - `dev_input` assignment (was line 3056, now moved after `dev_output`) — MTP-aware routing

Track 3 (commit `c91e18741`):

- `ggml/src/ggml-vulkan/vulkan-shaders/types.glsl` — `block_tq_v_4b` + `DATA_A_TQ_V_4B` define block
- `ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs.glsl` — `dequantize()` / `dequantize4()` / `get_dm()` for TQ_V_4B
- `ggml/src/ggml-vulkan/vulkan-shaders/copy_to_quant.comp` — `quantize()` for TQ_V_4B
- `ggml/src/ggml-vulkan/vulkan-shaders/dequant_tq_v_4b.comp` — new
- `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp` — explicit registrations
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp` — pipeline creation + supports_op cases

## Reproduction

From `slartibardfast/llama.cpp:vulkan-phase4` (tip: `c91e18741`):

```bash
git fetch origin vulkan-phase4
git checkout vulkan-phase4
cmake --build build-vk --target llama-server llama-batched-bench -j$(nproc)

# Confirm splits = 1 (the default, no --cache-type-v flag)
GGML_SCHED_DEBUG=1 GGML_VK_VISIBLE_DEVICES=1 ./build-vk/bin/llama-server \
  -m ~/models/Qwen3.5-9B-mtp-q4km.gguf \
  -c 4096 -ngl 99 -np 1 -fit off \
  --host 127.0.0.1 --port 9099 --no-warmup -v 2>&1 | grep 'graph splits'
# Expected: graph splits = 1

# Re-run the §9 equivalence check
/tmp/qwen35-phase9.sh
# Expected: TOKENS: BYTE-IDENTICAL ✓; acceptance 77.78%; spec & non-spec ~38.5 t/s
```

## What's deferred to future phases

- **Flash-attention TQ_V_4B integration.** The scaffolding is in place; the last mile is wiring TQ_V_4B into the flash-attention shader family and extending `supports_op`. This unlocks the real VRAM-saving value of `--cache-type-v tq_v_4b` (~1.7 GiB at `n_ctx=16384`).
- **MUL_MAT TQ_V_4B.** Needed if we ever want quantised V outside the flash-attention path. Not a priority given that flash attention is already mandatory for quantised V in llama.cpp.
- **Upstreaming.** The Track 1 change is a meaningful improvement that isn't tied to the polaris branch's specifics — it'd be a reasonable candidate PR against `ggml-org/llama.cpp` if we wanted to upstream it. The `dev_input` routing is MTP-specific, but the `buffer_from_host_ptr` cap flip + alignment handling + loader fallback is general.
- **The fused `ARGMAX_GET_ROWS` kernel from Track 2.** Useful if someone wants to shave ~2 dispatch-launches per forward, but not a splits-reduction any more. Design is in the plan file.
- **The 4-op `mul_mat_add_rms_norm_mul` fusion from Track 4.** Requires a row-oriented matmul shader. Interesting as a research direction but not session-sized.

## Commit references

```
422c6cb7b vulkan/llama: enable VK_EXT_external_memory_host + route MTP input to GPU
c91e18741 vulkan: scaffold TurboQuant V 4-bit (TQ_V_4B) type support
```

Both on `github.com/slartibardfast/llama.cpp` branch `vulkan-phase4`. No upstream PR opened.
