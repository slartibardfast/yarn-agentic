---
name: production-kv-cache-config-q4-0-hadamard-rotation
description: The ik_llama.cpp production server (profiles/qwen36.sh) uses Q4_0 quantized KV cache with Hadamard rotation on both K and V. llama-bench does NOT support the Hadamard flags and silently runs with rotation OFF.
metadata: 
  node_type: memory
  type: reference
  originSessionId: 9b52ee2b-d969-442c-83ba-98396525f6fa
---

The production llama-server on this host (Qwen 3.6 27B target) runs with:

```
--cache-type-k q4_0 --cache-type-v q4_0
--k-cache-hadamard --v-cache-hadamard
-fa on
```

`-k-cache-hadamard / -v-cache-hadamard` apply a Hadamard transform to K and V values before they're stored in the cache, which improves quantization fidelity at the 4-bit width. This is a production-on setting; any FA kernel work targeting production must validate against the post-Hadamard Q4_0 K/V distribution, NOT raw F16 random tensors.

**Sources:**
- Flag parsing: `common/common.cpp:1666-1672` (`-khad / --k-cache-hadamard` and `-vhad / --v-cache-hadamard`)
- Env vars: `common/common.cpp:513-514` (`LLAMA_ARG_K_CACHE_HADAMARD`, `LLAMA_ARG_V_CACHE_HADAMARD`)
- cparams field: `common/common.h:384-385` (`bool k_cache_hadamard`, `bool v_cache_hadamard`)
- Activated server-side at: `common/common.cpp:3658-3659`
- Production profile: `profiles/qwen36.sh` (search for `cache-hadamard`)

**llama-bench gap (as of 2026-05-14, commit 31e12e5):**

llama-bench does NOT respect Hadamard. `examples/llama-bench/llama-bench.cpp` references neither `k_cache_hadamard` nor the env vars. Its `to_llama_cparams()` (line 1161) leaves those cparams at their default (false). This means **any baseline taken with llama-bench at production KV settings is silently NOT representative** of production (the rotation is off).

To match production with llama-bench, patch needed:
1. Add `-khad / -vhad` flags to `parse_cmd_params` (mirror common.cpp:1666-1672)
2. Add env-var read for `LLAMA_ARG_K_CACHE_HADAMARD` / `LLAMA_ARG_V_CACHE_HADAMARD`
3. Add `k_cache_hadamard` / `v_cache_hadamard` to `cmd_params_instance` struct (~line 1059)
4. Set cparams fields in `to_llama_cparams()` (~line 1167-1180)

Patch is ~30 LOC. Filed as PLAN.md G1 follow-up gate.

**Workaround until patched**: use `nsys profile` on the production llama-server, or on `tests/dflash-speculative/test-np-validity-vanilla.cpp` which constructs cparams directly with `cparams.k_cache_hadamard = true` (or could be extended to). nsys captures per-kernel wall-clock including FA + combine kernels, giving the production-realistic timing breakdown without needing llama-bench.

**Why this matters for kernel work:**

The `fattn_per_slot_kv_sm75` replacement kernel must:
- Accept Q4_0 K and V (inline-dequant to fp16 before mma; mirror `fattn-mma-f16.cu:102-105`'s `ggml_get_to_fp16_cuda` call)
- Process the post-rotation values — Hadamard is applied at build-graph time (cparams), not in the FA kernel itself
- Be perf-bound against production wall-clock at Q4_0 + Hadamard, not F16 raw

The pre-merge baseline in `data/deltanet/perf/baseline/llama-bench-shapes.json` is the F16-KV baseline only (with the SUMMARY.md warning); the production-config baseline is a pre-merge gate (G2 in PLAN.md).
