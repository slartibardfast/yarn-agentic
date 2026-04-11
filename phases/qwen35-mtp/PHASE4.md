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

### Resolution — mixed K/V flash-attn (commit `71ba1ed4a`)

Instead of deferring, the mixed-K/V limitation turned out to be addressable with a targeted refactor. The key insight: the cross-product blowup only matters if *every* combination must ship, but the runtime only actually needs `(F16, F16)` and `(F16, TQ_V_4B)` for the `--cache-type-v tq_v_4b` path. One extra shader variant, gated by an opt-in override, is enough.

**`flash_attn_base.glsl` refactor.** Previously the shared header defined a single `BLOCK_SIZE`, `BLOCK_BYTE_SIZE`, `A_TYPE_PACKED16`, and `dequantize4(ib, iqs, a_offset, binding_idx)` function that handled both K and V via the runtime `binding_idx` parameter. The refactor adds:

- `KV_SPLIT_PATH` detection — any `DATA_K_*` or `DATA_V_*` define activates the independent-type path.
- Under the split path: explicit K-side bindings (`K_BLOCK_SIZE`, `K_BLOCK_BYTE_SIZE`, `K_PACKED16_TQ` / etc., `dequantize4_k()`) and V-side bindings (`V_BLOCK_SIZE`, `V_BLOCK_BYTE_SIZE`, `V_PACKED16_TQ`, `dequantize4_v()`).
- Under the legacy path (when no `DATA_K_*` / `DATA_V_*` is set): `K_BLOCK_SIZE`, `V_BLOCK_SIZE`, `K_BLOCK_BYTE_SIZE`, `V_BLOCK_BYTE_SIZE` are aliased to the existing `BLOCK_SIZE` / `BLOCK_BYTE_SIZE`, and `dequantize4_k()` / `dequantize4_v()` are defined as thin wrappers that forward to the existing `dequantize4()` function.

This means `flash_attn_cm1.comp` and `flash_attn_cm2.comp` — which still use the single-type API — are completely unaffected. Only `flash_attn.comp` is refactored to use the new `K_*` / `V_*` macros at all five dequant sites (three K, two V) plus the K/V offset calculations.

**New shader variant.** `vulkan-shaders-gen.cpp` registers `flash_attn_f32_f16_k_f16_v_tq_v_4b` with `{DATA_K_F16=1, DATA_V_TQ_V_4B=1}`. This compiles four SPV variants (fp16/fp32 × f16acc/f32acc) exactly like the single-type path. Since `DATA_K_F16` only sets `K_BLOCK_SIZE=1` (no extra K binding — flash_attn.comp already declares raw f16 K bindings unconditionally), the K side uses the raw f16 load path while the V side uses the dedicated TQ_V_4B dequant.

**Runtime wiring.** `ggml-vulkan.cpp` adds:

- A dedicated pipeline map `pipeline_flash_attn_f32_f16_k_f16_v_tq_v_4b` alongside the existing `pipeline_flash_attn_f32_f16[GGML_TYPE_COUNT]` array (the existing map is keyed on `k->type` only, which can't represent the mixed case).
- A new `CREATE_FA_FOR_MAP` helper macro (the existing `CREATE_FA` macro was refactored to delegate through it, preserving its interface) with a single call that iterates the mixed map.
- A loosening of the `k->type != v->type` rejection in `supports_op`: the specific `(F16, TQ_V_4B)` pair is now accepted on the scalar path (`coopmat2` still rejects, since no coopmat2 variant was compiled).
- A nested-lookup in `ggml_vk_flash_attn`: when `k->type == F16 && v->type == TQ_V_4B`, route to the mixed pipeline map instead of the type-indexed array.

### Verification

Running `--flash-attn on --cache-type-v tq_v_4b` on Qwen3.5-9B + q4km:

```
llama_context: flash_attn    = enabled
llama_kv_cache: size =   90.56 MiB (  4096 cells,   9 layers,  1/1 seqs), K (f16):   72.00 MiB, V (tq_v_4b):   18.56 MiB
sched_reserve: graph splits = 1
```

`graph splits = 1` — down from **19** before this refactor, matching the default Track 1 baseline (the single split is the empty CPU graph-entry sync, present in all post-Track-1 runs). End-to-end inference verified via `POST /completion`:

```
prompt: "The capital of France is"
content: " Paris.\nThe capital of France is"
tokens_predicted: 8, predicted_per_second: 57.86
draft_n: 4, draft_n_accepted: 4
```

Correct output, MTP speculative decoding working, V cache savings realised, flash attention running entirely on Vulkan.

### Resolution expanded to all V quant types (commit `a5af4aa97`)

The reusable infrastructure got used immediately. With KV_SPLIT_PATH in place and the pair-keyed pipeline map proven out, extending mixed K=F16 + V=quant to cover **every** quantised V type already supported by the legacy single-type path is ~200 lines of additive shader code and a handful of wiring tweaks.

**Shader coverage.** `flash_attn_base.glsl` gained a full per-type branch cascade for both K and V sides, covering:

| Side  | Types                                                           |
|-------|-----------------------------------------------------------------|
| K     | F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, TQ_V_4B              |
| V     | F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, TQ_V_4B              |

Each K type has its own `dequantize4_k()` with the correct packed16 struct binding at binding 1; each V type has its own `dequantize4_v()` at binding 2. These are independent `#elif` branches under `KV_SPLIT_PATH`, so a given shader variant only compiles one K branch and one V branch.

**`kvalues_iq4nl` availability.** The IQ4_NL lookup table in `types.glsl` was gated on `DATA_A_IQ4_NL || DATA_A_IQ4_XS` — the mixed path needed it under `DATA_K_IQ4_NL || DATA_V_IQ4_NL` as well. One-line ifdef extension, covers both directions.

**Shader-gen loop.** `vulkan-shaders-gen.cpp` replaces the one-off `k_f16_v_tq_v_4b` registration with a loop over a `{name, define}` table for all seven V quants. Each entry emits four variants (fp16/fp32 × f16acc/f32acc) → 28 total mixed shaders. Only the K=F16 direction is populated today; K=quant + V=other is supported by the shader code but not yet registered in the gen, since there's no practical runtime configuration that would reach it.

**Pipeline map refactor.** The dedicated `pipeline_flash_attn_f32_f16_k_f16_v_tq_v_4b` slot is replaced with `std::map<std::pair<ggml_type, ggml_type>, std::map<vk_fa_pipeline_state, vk_pipeline>> pipeline_flash_attn_f32_f16_kv_mixed` — one map, pair-keyed on `(k_type, v_type)`. `ggml_vk_flash_attn` routes any `k->type != v->type` case through this map instead of the type-indexed array. The `CREATE_FA_FOR_MAP` calls in `ggml_vk_load_shaders` now bind a local reference to each pair's map slot (to sidestep the preprocessor comma issue in brace-init-list keys) and iterate once per supported pair.

**`supports_op` cascade.** Replaced the narrow `(F16, TQ_V_4B)` allowlist with a full switch over all seven supported V quant types when K is F16, rejecting any other K type or V type. Still scalar-only (coopmat2 falls back to single-type shaders).

### Verification — every combination

Each of the seven `--cache-type-v <quant>` values now loads on Qwen3.5-9B + q4km with `graph splits = 1`, V cache quantised as requested, zero errors:

| `--cache-type-v` | V cache size | graph splits |
|------------------|--------------|--------------|
| q4_0             | 20.25 MiB    | 1            |
| q4_1             | 22.50 MiB    | 1            |
| q5_0             | 24.75 MiB    | 1            |
| q5_1             | 27.00 MiB    | 1            |
| q8_0             | 38.25 MiB    | 1            |
| iq4_nl           | 20.25 MiB    | 1            |
| tq_v_4b          | 18.56 MiB    | 1            |

(K is F16 in every case, at 72.00 MiB.) All seven combinations run the attention entirely on Vulkan.

### What Track 3 delivers in summary

- Full type-level integration of `GGML_TYPE_TQ_V_4B` into the Vulkan backend: types.glsl, dequant funcs, copy shaders, get_rows, set_rows, standalone dequant, **and** flash-attention shader variants for both single-type and all mixed F16+quant combinations.
- `--cache-type-v <quant> --flash-attn on` runs entirely on Vulkan for all seven supported quant V types, with `graph splits = 1` and the expected per-type V-cache memory savings at `n_ctx=4096` (scaling linearly with context).
- The mixed-K/V-type infrastructure is fully reusable: adding a K quant combination (e.g. a future `TQ_KV_1B` K cache paired with a TurboQuant V) is now a matter of enabling one more shader variant in the gen loop and extending the supports_op allowlist — all the shader-side scaffolding is already in place.

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

Track 3 follow-up #1 (commit `b93c3a142`):

- `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_base.glsl` — `block_tq_v_4b_packed16` dequant path (initial single-type variant)
- `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp` — `flash_attn_f32_f16_tq_v_4b` variant registration
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp` — CREATE_FA entries + supports_op FLASH_ATTN_EXT allowlist

Track 3 follow-up #2 — mixed K/V (commit `71ba1ed4a`):

- `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_base.glsl` — KV_SPLIT_PATH overlay introducing independent `K_BLOCK_SIZE` / `V_BLOCK_SIZE` / `dequantize4_k` / `dequantize4_v`; legacy `DATA_A_*` path aliased to wrappers for backward compat with cm1/cm2
- `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp` — five dequant sites refactored to use the new K/V-specific macros (3 K, 2 V), plus K/V offset calculation split
- `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp` — `flash_attn_f32_f16_k_f16_v_tq_v_4b` variant with `DATA_K_F16 + DATA_V_TQ_V_4B` defines
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp` — `pipeline_flash_attn_f32_f16_k_f16_v_tq_v_4b` map, `CREATE_FA_FOR_MAP` macro refactor, loosened `k->type != v->type` check in `supports_op`, mixed-map routing in `ggml_vk_flash_attn`

## Reproduction

From `slartibardfast/llama.cpp:vulkan-phase4` (tip: `71ba1ed4a`):

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

# Confirm splits = 1 AND V cache quantised (--flash-attn required with --cache-type-v)
GGML_SCHED_DEBUG=1 GGML_VK_VISIBLE_DEVICES=1 ./build-vk/bin/llama-server \
  -m ~/models/Qwen3.5-9B-mtp-q4km.gguf \
  -c 4096 -ngl 99 -np 1 -fa on --cache-type-v tq_v_4b \
  --host 127.0.0.1 --port 9099 --no-warmup -v 2>&1 | grep -E 'graph splits|V \(tq_v_4b\)'
# Expected:
#   V (tq_v_4b):   18.56 MiB
#   graph splits = 1

# Re-run the §9 equivalence check (f16 V cache, default)
/tmp/qwen35-phase9.sh
# Expected: TOKENS: BYTE-IDENTICAL ✓; acceptance 77.78%; spec & non-spec ~38.5 t/s
```

## What's deferred to future phases

- **MUL_MAT TQ_V_4B.** Needed if we ever want quantised V outside the flash-attention path. Not a priority given that flash attention is already mandatory for quantised V in llama.cpp.
- **Additional K/V mixed-type combinations.** The `KV_SPLIT_PATH` infrastructure in `flash_attn_base.glsl` makes adding new combos (e.g. `TQ_KV_1B` K + `TQ_V_4B` V once the polaris branch exposes TQ_KV_1B via `--cache-type-k`) a matter of extending two `#if` blocks and registering one more shader variant.
- **Upstreaming.** The Track 1 change and the mixed-K/V refactor are both meaningful improvements that aren't tied to the polaris branch's specifics — they'd be reasonable candidate PRs against `ggml-org/llama.cpp` if we wanted to upstream them. The `dev_input` routing is MTP-specific, but the `buffer_from_host_ptr` cap flip + alignment handling + loader fallback and the `flash_attn_base.glsl` KV split are general.
- **The fused `ARGMAX_GET_ROWS` kernel from Track 2.** Useful if someone wants to shave ~2 dispatch-launches per forward, but not a splits-reduction any more. Design is in the plan file.
- **The 4-op `mul_mat_add_rms_norm_mul` fusion from Track 4.** Requires a row-oriented matmul shader. Interesting as a research direction but not session-sized.

## Commit references

```
422c6cb7b vulkan/llama: enable VK_EXT_external_memory_host + route MTP input to GPU
c91e18741 vulkan: scaffold TurboQuant V 4-bit (TQ_V_4B) type support
b93c3a142 vulkan: add flash_attn TQ_V_4B variant (scalar path only)
71ba1ed4a vulkan: support mixed K=F16 + V=TQ_V_4B flash attention
a5af4aa97 vulkan: support all K=F16 + V=quant flash-attention combinations
```

All on `github.com/slartibardfast/llama.cpp` branch `vulkan-phase4`. No upstream PR opened.
