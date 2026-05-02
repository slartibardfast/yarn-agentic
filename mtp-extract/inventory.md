# MTP Extraction Inventory

Total commits: **121** between merge-base `847e1919` and fork tip `a0d0e06e`.


## Classification counts

- **merge**: 2
- **mixed**: 13
- **mtp_core**: 25
- **mtp_doc**: 1
- **mtp_test**: 1
- **out_scope**: 79


## Commit table (chronological, oldest first)

| Class | SHA | Subject | Author | Files (count) |
|---|---|---|---|---|
| out_scope | `ad87b110b1` | iqk: graceful scalar fallbacks for non-AVX2 x86 | David Connolly | 5 |
| mixed | `c23137660b` | vulkan: multi-GPU split mode graph support | David Connolly | 5 |
| out_scope | `6fc1bb3ebd` | vulkan: async cross-device copy pipeline | David Connolly | 1 |
| out_scope | `cb9d3f8a2e` | vulkan: per-copy staging pool for cross-device transfers | David Connolly | 1 |
| out_scope | `1fe1433916` | vulkan: parallel split buffer uploads across devices | David Connolly | 1 |
| out_scope | `ff04f7afe8` | vulkan: defer graph compute fence wait to synchronize() | David Connolly | 1 |
| out_scope | `e451b2681b` | vulkan: dmabuf zero-copy cross-device transfer | David Connolly | 3 |
| out_scope | `5f13c1fc6b` | vulkan: add FUSED_UP_GATE shader and dispatch | David Connolly | 3 |
| out_scope | `7d8a13984e` | Fix FUSED_UP_GATE shader bug and add exhaustive backend-ops tests | David Connolly | 4 |
| out_scope | `5e9e73728b` | vulkan: fix empty-graph fence hang and MULTI_ADD descriptor range | David Connolly | 2 |
| out_scope | `c549c07992` | vulkan: add FUSED_RMS_NORM and FUSED_MUL_UNARY backend-ops tests | David Connolly | 1 |
| out_scope | `eebc5967c0` | vulkan: expand CONT and SOFT_MAX test coverage | David Connolly | 1 |
| mixed | `a6c5c1e88d` | vulkan: fix rope_freqs tensor shape for explicit head_dim models | David Connolly | 1 |
| out_scope | `e741259613` | vulkan: implement REDUCE op for multi-GPU split-mode graph | David Connolly | 1 |
| out_scope | `ff3dd6e0be` | vulkan: GPU-accelerated REDUCE via dmabuf + ADD shader | David Connolly | 1 |
| out_scope | `6c7418d74d` | vulkan: fix SUM_ROWS for 4D tensors, fix CPU SUM_ROWS index bug | David Connolly | 3 |
| out_scope | `0b974fd3d5` | vulkan: fix K-quant dequant bounds check for batched tensors | David Connolly | 5 |
| out_scope | `1af683b5e5` | vulkan: fix get_tensor_async race condition on rBAR devices | David Connolly | 2 |
| out_scope | `d036534fa2` | vulkan: fix backend-ops test failures (Phase 0 Rounds 2-5) | David Connolly | 19 |
| out_scope | `05c0734225` | iqk: add scalar iq3_xxs fallback; fix CPY iq4_nl test threshold | David Connolly | 1 |
| out_scope | `4095d4b8ab` | vulkan: add subgroup reduction shader variants (disabled pending debug) | David Connolly | 3 |
| out_scope | `600e0ed446` | vulkan: enable subgroup reduction with require_full_subgroups | David Connolly | 1 |
| out_scope | `75f123d2a3` | Phase 20: MMVQ shader infrastructure and debugging (disabled) | David Connolly | 14 |
| merge | `5e03600b29` | Merge upstream/main: resolve bf16 and CMakeLists conflicts | David Connolly | 12 |
| out_scope | `a17fd20164` | Phase 20: fix MMVQ NaN — x4 quantize format + subbuffer port | David Connolly | 2 |
| out_scope | `dff50ac597` | Phase 20: MMVQ working — 5.3x speedup on 7B Q8_0 (6800 XT) | David Connolly | 3 |
| out_scope | `b014824ca9` | vulkan: split AMD_GCN architecture into GCN3 and GCN5 | David Connolly | 1 |
| mtp_core | `8300312a82` | llama: skip up_gate fusion for token gen (N=1) — 1.5-2× tg speedup | David Connolly | 1 |
| mtp_core | `c3620963dd` | Phase 20f: fix ggml_reduce nhave > 1 assertion in multi-GPU graph split | David Connolly | 1 |
| out_scope | `e9c368095a` | Phase 20c: Vega RPM via explicit f16vec2 packing in mul_mat_vec shaders | David Connolly | 16 |
| out_scope | `2454916c71` | Phase 20d: Vega RPM via explicit f16vec2 packing in mul_mm matmul-mat shader | David Connolly | 2 |
| mixed | `e2182ac2ef` | vulkan: remove phase references from in-tree code comments | David Connolly | 3 |
| out_scope | `2f27f41021` | vulkan: add MOE_FUSED_UP_GATE shader and dispatch | David Connolly | 4 |
| out_scope | `59595854c5` | vulkan: add GROUPED_TOPK shader and dispatch | David Connolly | 4 |
| out_scope | `8f5fd58e29` | vulkan: enable L2_NORM with non-contiguous input support | David Connolly | 3 |
| out_scope | `c79ef94b27` | vulkan: add SOFTPLUS unary op | David Connolly | 3 |
| out_scope | `36484fff24` | vulkan: add MUL_MULTI_ADD op (MoE expert combine) | David Connolly | 3 |
| out_scope | `3d8468866e` | vulkan: add SIGMOID and scalar-broadcast variants for FUSED_MUL_UNARY | David Connolly | 7 |
| out_scope | `ca9bb90fa9` | vulkan: add SSM_CONV (single-seq fast path) | David Connolly | 5 |
| out_scope | `302e4b7299` | vulkan: SSM_CONV multi-sequence slow path | David Connolly | 5 |
| out_scope | `92f99e68ef` | vulkan: SSM_CONV multi-seq unique-fast path with atomic gating | David Connolly | 5 |
| out_scope | `10c83a1cc9` | vulkan: add DELTA_NET shader and dispatch | David Connolly | 4 |
| out_scope | `f6dc7cda23` | vulkan: disable f16acc mul_mat_vec on Vega by default | David Connolly | 1 |
| out_scope | `e67f3740a8` | vulkan: fix f16acc overflow in K-quant scale multiply + stress tests | David Connolly | 6 |
| merge | `99842dc8b0` | Merge upstream ik_llama.cpp (6 commits: Gemma4 MoE graph parallel, mixed KV cache, Hadamar | David Connolly | 4 |
| mixed | `6c54173642` | vulkan: DELTA_NET inplace state writeback — eliminate CONCAT from recurrent layers | David Connolly | 5 |
| mixed | `5cf4620291` | vulkan: GGML_OP_FUSED framework + GATE_PREP fusion + SILU_MUL stub | David Connolly | 7 |
| mixed | `c4e93e3f24` | vulkan: port TURBO_KV_4B KV cache compression with wave32+wave64 FWHT | David Connolly | 13 |
| out_scope | `0510e55b7f` | tests: add TURBO_KV_4B to backend-ops test suite | David Connolly | 1 |
| mtp_core | `da426345fd` | models/qwen35: enable MTP dual-path graph building | David Connolly | 2 |
| mtp_core | `b885b1d523` | models/qwen35: full MTP support for dense and MOE architectures | David Connolly | 5 |
| mtp_core | `75ad66735e` | models/qwen35: fix MTP runtime — multi-rope sections + KV cache allocation | David Connolly | 2 |
| mtp_core | `459c7fa942` | models/qwen35: fix MTP acceptance — use build_std_attention + pre-norm hidden state | David Connolly | 1 |
| mtp_core | `6970506c3a` | models/qwen35: single-pass MTP graph + logit extraction API | David Connolly | 4 |
| mtp_core | `f4760094dd` | server: wire inline MTP draft from single-pass graph | David Connolly | 1 |
| mtp_core | `ad488f83ff` | models/qwen35: fix single-pass MTP — full-position greedy logits + deferred filtering | David Connolly | 1 |
| mtp_core | `07c73a5af4` | models/qwen35: clear inline MTP state per decode for correct hybrid warmup | David Connolly | 1 |
| mtp_core | `fcc5ab1b56` | models/qwen35: fix MTP hidden state + eliminate two-pass fallback cascade | David Connolly | 2 |
| mtp_core | `c55643e6df` | models/qwen35: always-on MTP head + shifted-token prompt eval | David Connolly | 3 |
| mtp_core | `973f83ad80` | fix: search for result_output by name instead of assuming last graph node | David Connolly | 2 |
| mtp_core | `ce1a71474d` | models/qwen35: switch to full-position greedy logits for MTP (matching fork) | David Connolly | 1 |
| mtp_core | `bac58db4d6` | models/qwen35: complete single-pass MTP — full head + working draft extraction | David Connolly | 3 |
| mtp_core | `ab6cf6845c` | models/qwen35: FastMTP vocabulary trimming (248K → 32K) | David Connolly | 1 |
| mtp_core | `e0d7bf2b61` | models/qwen35: move input embedding to GPU for MTP models | David Connolly | 1 |
| mtp_core | `fd77f89850` | perf: eliminate MTP graph splits — 10.2 → 17.8 t/s (12% overhead) | David Connolly | 1 |
| mtp_core | `aab04197c3` | perf: offload input embedding to GPU when fully offloaded | David Connolly | 1 |
| mtp_core | `24f64b1e20` | perf: fix cross-device MTP — co-locate MTP + output on last-main-layer GPU | David Connolly | 1 |
| mixed | `f18cbc5d0e` | vulkan: add K-quant get_rows shaders + preserve MTP layers at F16 | David Connolly | 6 |
| mixed | `c87ce9ebba` | vulkan: complete scalar flash attention for all element-independent KV cache types | David Connolly | 4 |
| out_scope | `716a6539ae` | vulkan: K-quant CPY/SET_ROWS shaders + pipeline wiring | David Connolly | 5 |
| out_scope | `da0774f4b6` | fix: K-quant CPY pipeline in RTE branch + test fixtures | David Connolly | 3 |
| out_scope | `0c755aa50f` | vulkan: TURBO_KV_4B flash attention via shared memory RHT pre-dequant | David Connolly | 4 |
| out_scope | `33ab336e48` | fix: add TURBO_KV_4B to CPY supports_op — eliminates 13 graph splits | David Connolly | 1 |
| out_scope | `2856f40b9f` | wip: TURBO_KV_4B FA debugging — shmem fix, null fallback, binding layout | David Connolly | 3 |
| out_scope | `d983c940bf` | fix: TURBO_KV_4B pipeline creation — pass actual SPIR-V data | David Connolly | 2 |
| out_scope | `92508e47c4` | fix: TURBO_KV_4B FA — packed16 byte indexing + split_k identification | David Connolly | 2 |
| out_scope | `872821d4c8` | fix: TURBO_KV_4B FA — zero-init kv_sh to prevent NaN from unfilled KV positions | David Connolly | 2 |
| out_scope | `56e02b4fed` | fix: TURBO_KV_4B dequant CPY — use proper push constants + shader | David Connolly | 4 |
| out_scope | `5bf167afa2` | wip: TURBO_KV_4B round-trip debugging — RHT correctness issue | David Connolly | 2 |
| out_scope | `8e1691b9d5` | wip: TURBO_KV_4B RHT verified correct on CPU (NMSE=0.0096), GPU wrong | David Connolly | 2 |
| out_scope | `b54654455f` | fix: TURBO_KV_4B FWHT butterfly sign inversion — correct round-trip | David Connolly | 1 |
| out_scope | `ad7047aa84` | perf: fp16-consistent quantization in TURBO_KV_4B CPY | David Connolly | 1 |
| out_scope | `2d824f6374` | wip: TURBO_KV_4B FA — non-packed struct access, same incorrect output | David Connolly | 1 |
| out_scope | `f22ee0a2e9` | fix: TURBO_KV_4B FA — correct codebook, sign function, and dequant approach | David Connolly | 3 |
| out_scope | `ff74c13edf` | fix: TURBO_KV_4B FA crash — large-rows-only pipelines + V inverse RHT + end-to-end test | David Connolly | 3 |
| out_scope | `bd9f82d8ea` | fix: TURBO_KV_4B FA V buffer binding — was reading K data for V | David Connolly | 3 |
| out_scope | `4902a1f70f` | wip: TURBO_KV_4B FA — Q rotation + V shmem inverse RHT, outputs London (close) | David Connolly | 1 |
| out_scope | `656aca3980` | wip: TURBO_KV_4B FA — iterating on Q rotation, K norm, V shmem FWHT | David Connolly | 1 |
| out_scope | `5c055da445` | fix: TURBO_KV_4B FA — re-enable Q rotation, 9B model produces Paris | David Connolly | 1 |
| out_scope | `0b595bc2a7` | wip: TURBO_KV_4B FA — full inverse RHT on K+V, memoryBarrierShared | David Connolly | 1 |
| out_scope | `873ca1767a` | wip: TURBO_KV_4B FA — shmem FWHT K+V, strongest barriers, still London | David Connolly | 1 |
| out_scope | `348c827efe` | wip: TURBO_KV_4B FA — proven FWHT correct, GPU execution context is the bug | David Connolly | 1 |
| out_scope | `a3d1abddc1` | fix: TURBO_KV_4B FA — shmem FWHT at WG=128, 9B produces Paris | David Connolly | 2 |
| out_scope | `da93cdb2a1` | wip: TURBO_KV_4B FA — enforce subgroup size experiment, back to shmem | David Connolly | 1 |
| out_scope | `1bbff75c01` | fix: CPU TURBO_KV_4B FA — graceful fallback + F32 from_float, outputs Paris | David Connolly | 2 |
| out_scope | `3dd954e72a` | wip: TURBO_KV_4B FA — WG=64 breaks FA structure, back to shmem WG=128 | David Connolly | 2 |
| out_scope | `4d95084ea0` | wip: TURBO_KV_4B FA investigation — out-of-place FWHT, vec4 write, codebook-only | David Connolly | 1 |
| out_scope | `bf17b5e5c8` | wip: ISA trace — ACO generates correct butterfly (mul sign + add) | David Connolly | 1 |
| out_scope | `6a5734bd49` | wip: ISA confirms ACO eliminates v_sub_f32 from FWHT butterfly | David Connolly | 1 |
| out_scope | `dd189a0b21` | root cause confirmed: Mesa NIR eliminates V butterfly subtraction | David Connolly | 1 |
| out_scope | `70d2cf6c78` | fix: TURBO_KV_4B FA — disable V FWHT (Mesa NIR miscompilation workaround) | David Connolly | 1 |
| out_scope | `324015f1c8` | fix: TURBO_KV_4B FA — re-enable V FWHT, fix confirmed correct | David Connolly | 1 |
| out_scope | `6c06d4923f` | fix: TURBO_KV_4B FA — clamp scores for positions beyond KV to -inf | David Connolly | 1 |
| out_scope | `1ff0d86ede` | wip: TURBO_KV_4B FA — V pre-scale and subgroup FWHT both fail | David Connolly | 1 |
| out_scope | `92d2044145` | fix: TURBO_KV_4B FA — use K/V stride for multi-head block indexing | David Connolly | 1 |
| out_scope | `f291df5659` | docs: TURBO_KV_4B debugging notes — multi-head stride bug class | David Connolly | 1 |
| out_scope | `2c0b3028f2` | docs: move debugging writeup to yarn-agentic mdBook | David Connolly | 1 |
| mixed | `7904c6c4b9` | feat: MTP chained rollout — port, CPU fix, Vulkan perf fixes | David Connolly | 16 |
| out_scope | `25dff615bb` | test: MTP-IR standalone harnesses for delta_net + rollout | David Connolly | 7 |
| mtp_doc | `e0e3c35504` | test: MTP chained rollout matrix + regression anchors + bench | David Connolly | 97 |
| mtp_core | `826b71e3de` | fix: MTP-IR rollback — ggml_cpy→ggml_scale for dn_result_keep | David Connolly | 1 |
| mtp_core | `0278ccc26e` | cleanup: stale comments + FINDINGS addendum after rollback fix | David Connolly | 2 |
| mixed | `055210c6b6` | vulkan: mul_mat_vec batch-invariance — collapse pipelines, disable MMVQ | David Connolly | 19 |
| mixed | `f29ef6cd63` | vulkan: batch-invariance for MUL_MAT_ID and FLASH_ATTN, remove force_bi flag | David Connolly | 2 |
| mixed | `716718e7ce` | llama: disable fused MoE up-gate and dense fused up-gate for batch-invariance | David Connolly | 2 |
| mixed | `8e2e1281b9` | vulkan: Phase 3 findings — MMVQ and fused-up-gate net-negative to re-enable | David Connolly | 2 |
| mtp_core | `3e3bb0f962` | llama: enable fused_moe for all n — 5× MoE tg win, shader proven BI | David Connolly | 1 |
| mtp_core | `6cf65bc109` | llama: gate dense fused up-gate on cparams.fused_up_gate (P3 of cleanup) | David Connolly | 1 |
| mtp_core | `b7098b7d09` | WIP: ik_llama MTP-on-35B-A3B port — modified core files | David Connolly | 7 |
| mtp_test | `737d607a7c` | WIP: ik_llama MTP-on-35B-A3B port — correctness probes | David Connolly | 16 |
| out_scope | `a0d0e06e9e` | tools/mesa-repro: minimal Vulkan compute repro for Mesa GPU driver bug | David Connolly | 4 |


## mtp_core: full file list per commit


### `8300312a82` — llama: skip up_gate fusion for token gen (N=1) — 1.5-2× tg speedup

- `src/llama-build-context.cpp`

### `c3620963dd` — Phase 20f: fix ggml_reduce nhave > 1 assertion in multi-GPU graph split

- `src/llama-build-context.cpp`

### `da426345fd` — models/qwen35: enable MTP dual-path graph building

- `src/llama-build-context.cpp`
- `src/llama.cpp`

### `b885b1d523` — models/qwen35: full MTP support for dense and MOE architectures

- `src/llama-build-context.cpp`
- `src/llama-hparams.cpp`
- `src/llama-hparams.h`
- `src/llama-load-tensors.cpp`
- `src/llama-model.cpp`

### `75ad66735e` — models/qwen35: fix MTP runtime — multi-rope sections + KV cache allocation

- `src/llama-build-context.cpp`
- `src/llama-hparams.cpp`

### `459c7fa942` — models/qwen35: fix MTP acceptance — use build_std_attention + pre-norm hidden state

- `src/llama-build-context.cpp`

### `6970506c3a` — models/qwen35: single-pass MTP graph + logit extraction API

- `include/llama.h`
- `src/llama-build-context.cpp`
- `src/llama-context.h`
- `src/llama.cpp`

### `f4760094dd` — server: wire inline MTP draft from single-pass graph

- `examples/server/server-context.cpp`

### `ad488f83ff` — models/qwen35: fix single-pass MTP — full-position greedy logits + deferred filtering

- `src/llama-build-context.cpp`

### `07c73a5af4` — models/qwen35: clear inline MTP state per decode for correct hybrid warmup

- `src/llama.cpp`

### `fcc5ab1b56` — models/qwen35: fix MTP hidden state + eliminate two-pass fallback cascade

- `examples/server/server-context.cpp`
- `src/llama-build-context.cpp`

### `c55643e6df` — models/qwen35: always-on MTP head + shifted-token prompt eval

- `src/llama-build-context.cpp`
- `src/llama-context.h`
- `src/llama.cpp`

### `973f83ad80` — fix: search for result_output by name instead of assuming last graph node

- `src/llama-build-context.cpp`
- `src/llama.cpp`

### `ce1a71474d` — models/qwen35: switch to full-position greedy logits for MTP (matching fork)

- `src/llama-build-context.cpp`

### `bac58db4d6` — models/qwen35: complete single-pass MTP — full head + working draft extraction

- `examples/server/server-context.cpp`
- `src/llama-build-context.cpp`
- `src/llama.cpp`

### `ab6cf6845c` — models/qwen35: FastMTP vocabulary trimming (248K → 32K)

- `src/llama-build-context.cpp`

### `e0d7bf2b61` — models/qwen35: move input embedding to GPU for MTP models

- `src/llama.cpp`

### `fd77f89850` — perf: eliminate MTP graph splits — 10.2 → 17.8 t/s (12% overhead)

- `src/llama-build-context.cpp`

### `aab04197c3` — perf: offload input embedding to GPU when fully offloaded

- `src/llama.cpp`

### `24f64b1e20` — perf: fix cross-device MTP — co-locate MTP + output on last-main-layer GPU

- `src/llama.cpp`

### `826b71e3de` — fix: MTP-IR rollback — ggml_cpy→ggml_scale for dn_result_keep

- `src/llama-delta-net.cpp`

### `0278ccc26e` — cleanup: stale comments + FINDINGS addendum after rollback fix

- `src/llama-delta-net.cpp`
- `tests/mtp-matrix/FINDINGS.md`

### `3e3bb0f962` — llama: enable fused_moe for all n — 5× MoE tg win, shader proven BI

- `src/llama-build-context.cpp`

### `6cf65bc109` — llama: gate dense fused up-gate on cparams.fused_up_gate (P3 of cleanup)

- `src/llama-build-context.cpp`

### `b7098b7d09` — WIP: ik_llama MTP-on-35B-A3B port — modified core files

- `examples/server/server-context.cpp`
- `examples/server/server-context.h`
- `include/llama.h`
- `src/llama-delta-net.cpp`
- `src/llama.cpp`
- `tests/CMakeLists.txt`
- `tests/test-backend-ops.cpp`

## mtp_test: full file list per commit


### `737d607a7c` — WIP: ik_llama MTP-on-35B-A3B port — correctness probes

- `tests/mtp-matrix/35b/README.md`
- `tests/mtp-matrix/semantics/test-35b-mtp-flag-invariance.sh`
- `tests/mtp-matrix/semantics/test-35b-mtp-logits-sanity.sh`
- `tests/mtp-matrix/semantics/test-35b-prompt-sweep.sh`
- `tests/test-35b-batch-invariance-sweep.cpp`
- `tests/test-35b-batch-invariance.cpp`
- `tests/test-35b-fmoe-logits-diff.cpp`
- `tests/test-35b-full-accept-drift.cpp`
- `tests/test-35b-layer-drift.cpp`
- `tests/test-35b-mtp-draft-token-at.cpp`
- `tests/test-35b-pos-i-sequential-equivalence.cpp`
- `tests/test-35b-rollback-sweep-correctness.cpp`
- `tests/test-35b-server-flow-drift.cpp`
- `tests/test-35b-trajectory-drift.cpp`
- `tests/test-35b-verify-cycle-drift.cpp`
- `tests/test-delta-net-emit-path-invariance.cpp`

## mtp_doc: full file list per commit


### `e0e3c35504` — test: MTP chained rollout matrix + regression anchors + bench

- `tests/.gitignore`
- `tests/mtp-matrix/AUDIT.md`
- `tests/mtp-matrix/BENCH.md`
- `tests/mtp-matrix/FINDINGS.md`
- `tests/mtp-matrix/README.md`
- `tests/mtp-matrix/bench/bench-rollout-throughput.sh`
- `tests/mtp-matrix/coherence/test-coherence-deterministic.sh`
- `tests/mtp-matrix/coherence/test-coherence-long-r1.sh`
- `tests/mtp-matrix/coherence/test-coherence-r1.sh`
- `tests/mtp-matrix/coherence/test-coherence-r2.sh`
- `tests/mtp-matrix/coherence/test-coherence-r3.sh`
- `tests/mtp-matrix/coherence/test-coherence-r4.sh`
- `tests/mtp-matrix/coherence/test-coherence-r5.sh`
- `tests/mtp-matrix/coherence/test-coherence-r6.sh`
- `tests/mtp-matrix/coherence/test-coherence-r7.sh`
- `tests/mtp-matrix/coherence/test-coherence-r8.sh`
- `tests/mtp-matrix/coherence/test-multi-prompt-quality.sh`
- `tests/mtp-matrix/lib/_build_matrix.sh`
- `tests/mtp-matrix/lib/_common.sh`
- `tests/mtp-matrix/lib/_logits.sh`
- `tests/mtp-matrix/lib/_quality.sh`
- `tests/mtp-matrix/regressions/test-accept-floor.sh`
- `tests/mtp-matrix/regressions/test-double-request-same-answer.sh`
- `tests/mtp-matrix/regressions/test-exact-paris.sh`
- `tests/mtp-matrix/regressions/test-golden-snapshots.sh`
- `tests/mtp-matrix/regressions/test-graph-metrics.sh`
- `tests/mtp-matrix/regressions/test-tps-floor.sh`
- `tests/mtp-matrix/run-all.sh`
- `tests/mtp-matrix/run-parallel.sh`
- `tests/mtp-matrix/scheduler/test-growing-shape-sweep.sh`
- `tests/mtp-matrix/scheduler/test-r3-prompt-sweep.sh`
- `tests/mtp-matrix/scheduler/test-reserve-compute-reserve.sh`
- `tests/mtp-matrix/scheduler/test-reserve-reserve-no-compute.sh`
- `tests/mtp-matrix/scheduler/test-same-shape-repeat.sh`
- `tests/mtp-matrix/scheduler/test-shape-flip-flop.sh`
- `tests/mtp-matrix/semantics/test-backend-equivalence-r1.sh`
- `tests/mtp-matrix/semantics/test-iter0-matches-single-pass.sh`
- `tests/mtp-matrix/semantics/test-logit-sanity.sh`
- `tests/mtp-matrix/semantics/test-per-iteration-preservation.sh`
- `tests/mtp-matrix/server/test-llama-cli-r3-repro.sh`
- `tests/mtp-matrix/server/test-no-mtp-flag.sh`
- `tests/mtp-matrix/server/test-startup-no-warmup-r3.sh`
- `tests/mtp-matrix/server/test-startup-parallel-2-r1.sh`
- `tests/mtp-matrix/server/test-startup-r1.sh`
- `tests/mtp-matrix/server/test-startup-r3.sh`
- `tests/mtp-matrix/shape/_shape_template.sh`
- `tests/mtp-matrix/shape/test-shape-t1-r1.sh`
- `tests/mtp-matrix/shape/test-shape-t1-r2.sh`
- `tests/mtp-matrix/shape/test-shape-t1-r3.sh`
- `tests/mtp-matrix/shape/test-shape-t15-r1.sh`
- `tests/mtp-matrix/shape/test-shape-t15-r2.sh`
- `tests/mtp-matrix/shape/test-shape-t15-r3.sh`
- `tests/mtp-matrix/shape/test-shape-t16-r1.sh`
- `tests/mtp-matrix/shape/test-shape-t16-r2.sh`
- `tests/mtp-matrix/shape/test-shape-t16-r3.sh`
- `tests/mtp-matrix/shape/test-shape-t17-r1.sh`
- `tests/mtp-matrix/shape/test-shape-t17-r2.sh`
- `tests/mtp-matrix/shape/test-shape-t17-r3.sh`
- `tests/mtp-matrix/shape/test-shape-t2-r1.sh`
- `tests/mtp-matrix/shape/test-shape-t2-r2.sh`
- `tests/mtp-matrix/shape/test-shape-t2-r3.sh`
- `tests/mtp-matrix/shape/test-shape-t3-r1.sh`
- `tests/mtp-matrix/shape/test-shape-t3-r2.sh`
- `tests/mtp-matrix/shape/test-shape-t3-r3.sh`
- `tests/mtp-matrix/shape/test-shape-t31-r1.sh`
- `tests/mtp-matrix/shape/test-shape-t31-r2.sh`
- `tests/mtp-matrix/shape/test-shape-t31-r3.sh`
- `tests/mtp-matrix/shape/test-shape-t32-r1.sh`
- `tests/mtp-matrix/shape/test-shape-t32-r2.sh`
- `tests/mtp-matrix/shape/test-shape-t32-r3.sh`
- `tests/mtp-matrix/shape/test-shape-t33-r1.sh`
- `tests/mtp-matrix/shape/test-shape-t33-r2.sh`
- `tests/mtp-matrix/shape/test-shape-t33-r3.sh`
- `tests/mtp-matrix/shape/test-shape-t4-r1.sh`
- `tests/mtp-matrix/shape/test-shape-t4-r2.sh`
- `tests/mtp-matrix/shape/test-shape-t4-r3.sh`
- `tests/mtp-matrix/shape/test-shape-t5-r1.sh`
- `tests/mtp-matrix/shape/test-shape-t5-r2.sh`
- `tests/mtp-matrix/shape/test-shape-t5-r3.sh`
- `tests/mtp-matrix/shape/test-shape-t6-r1.sh`
- `tests/mtp-matrix/shape/test-shape-t6-r2.sh`
- `tests/mtp-matrix/shape/test-shape-t6-r3.sh`
- `tests/mtp-matrix/shape/test-shape-t7-r1.sh`
- `tests/mtp-matrix/shape/test-shape-t7-r2.sh`
- `tests/mtp-matrix/shape/test-shape-t7-r3.sh`
- `tests/mtp-matrix/shape/test-shape-t8-r1.sh`
- `tests/mtp-matrix/shape/test-shape-t8-r2.sh`
- `tests/mtp-matrix/shape/test-shape-t8-r3.sh`
- `tests/mtp-matrix/snapshots/cpu-r1-capital.txt`
- `tests/mtp-matrix/snapshots/cpu-r1-lang.txt`
- `tests/mtp-matrix/snapshots/cpu-r1-moon.txt`
- `tests/mtp-matrix/snapshots/cpu-r1-sky.txt`
- `tests/mtp-matrix/snapshots/vulkan-r1-capital.txt`
- `tests/mtp-matrix/snapshots/vulkan-r1-lang.txt`
- `tests/mtp-matrix/snapshots/vulkan-r1-moon.txt`
- `tests/mtp-matrix/snapshots/vulkan-r1-sky.txt`
- `tests/mtp-matrix/summarize.sh`

## mixed: full file list per commit


### `c23137660b` — vulkan: multi-GPU split mode graph support

- `ggml/include/ggml-vulkan.h`
- `ggml/src/CMakeLists.txt`
- `ggml/src/ggml-vulkan-multigpu.cpp`
- `ggml/src/ggml-vulkan.cpp`
- `src/llama.cpp`

### `a6c5c1e88d` — vulkan: fix rope_freqs tensor shape for explicit head_dim models

- `src/llama-load-tensors.cpp`

### `e2182ac2ef` — vulkan: remove phase references from in-tree code comments

- `ggml/src/ggml-vulkan.cpp`
- `ggml/src/vulkan-shaders/vulkan-shaders-gen.cpp`
- `src/llama-build-context.cpp`

### `6c54173642` — vulkan: DELTA_NET inplace state writeback — eliminate CONCAT from recurrent layers

- `ggml/src/ggml-vulkan.cpp`
- `ggml/src/ggml.c`
- `ggml/src/vulkan-shaders/delta_net.comp`
- `ggml/src/vulkan-shaders/vulkan-shaders-gen.cpp`
- `src/llama-delta-net.cpp`

### `5cf4620291` — vulkan: GGML_OP_FUSED framework + GATE_PREP fusion + SILU_MUL stub

- `ggml/include/ggml-fusion.h`
- `ggml/include/ggml.h`
- `ggml/src/ggml-vulkan.cpp`
- `ggml/src/ggml.c`
- `ggml/src/vulkan-shaders/fused_gate_prep.comp`
- `ggml/src/vulkan-shaders/vulkan-shaders-gen.cpp`
- `src/llama-delta-net.cpp`

### `c4e93e3f24` — vulkan: port TURBO_KV_4B KV cache compression with wave32+wave64 FWHT

- `common/common.cpp`
- `ggml/include/ggml-turbo-kv.h`
- `ggml/include/ggml.h`
- `ggml/src/CMakeLists.txt`
- `ggml/src/ggml-turbo-kv.c`
- `ggml/src/ggml-vulkan.cpp`
- `ggml/src/ggml.c`
- `ggml/src/vulkan-shaders/cpy_f32_turbo_kv_4b.comp`
- `ggml/src/vulkan-shaders/dequant_turbo_kv_4b.comp`
- `ggml/src/vulkan-shaders/get_rows_turbo_kv_4b.comp`
- `ggml/src/vulkan-shaders/turbo_kv_4b_rht.comp`
- `ggml/src/vulkan-shaders/types.comp`
- `ggml/src/vulkan-shaders/vulkan-shaders-gen.cpp`

### `f18cbc5d0e` — vulkan: add K-quant get_rows shaders + preserve MTP layers at F16

- `ggml/src/ggml-vulkan.cpp`
- `ggml/src/vulkan-shaders/get_rows_q4_k.comp`
- `ggml/src/vulkan-shaders/get_rows_q5_k.comp`
- `ggml/src/vulkan-shaders/get_rows_q6_k.comp`
- `ggml/src/vulkan-shaders/vulkan-shaders-gen.cpp`
- `src/llama-quantize.cpp`

### `c87ce9ebba` — vulkan: complete scalar flash attention for all element-independent KV cache types

- `common/common.cpp`
- `ggml/src/ggml-vulkan.cpp`
- `ggml/src/vulkan-shaders/flash_attn_base.comp`
- `ggml/src/vulkan-shaders/vulkan-shaders-gen.cpp`

### `7904c6c4b9` — feat: MTP chained rollout — port, CPU fix, Vulkan perf fixes

- `common/speculative.cpp`
- `common/speculative.h`
- `examples/server/server-context.cpp`
- `ggml/include/ggml.h`
- `ggml/src/ggml-alloc.c`
- `ggml/src/ggml-vulkan.cpp`
- `ggml/src/ggml.c`
- `ggml/src/vulkan-shaders/delta_net.comp`
- `include/llama.h`
- `src/llama-build-context.cpp`
- `src/llama-context.h`
- `src/llama-delta-net.cpp`
- `src/llama-delta-net.h`
- `src/llama.cpp`
- `tests/CMakeLists.txt`
- `tests/test-backend-ops.cpp`

### `055210c6b6` — vulkan: mul_mat_vec batch-invariance — collapse pipelines, disable MMVQ

- `ggml/src/ggml-vulkan.cpp`
- `ggml/src/vulkan-shaders/mul_mat_vec.comp`
- `ggml/src/vulkan-shaders/mul_mat_vec_base.comp`
- `ggml/src/vulkan-shaders/mul_mat_vec_iq1_m.comp`
- `ggml/src/vulkan-shaders/mul_mat_vec_iq1_s.comp`
- `ggml/src/vulkan-shaders/mul_mat_vec_iq2_s.comp`
- `ggml/src/vulkan-shaders/mul_mat_vec_iq2_xs.comp`
- `ggml/src/vulkan-shaders/mul_mat_vec_iq2_xxs.comp`
- `ggml/src/vulkan-shaders/mul_mat_vec_iq3_s.comp`
- `ggml/src/vulkan-shaders/mul_mat_vec_iq3_xxs.comp`
- `ggml/src/vulkan-shaders/mul_mat_vec_q2_k.comp`
- `ggml/src/vulkan-shaders/mul_mat_vec_q3_k.comp`
- `ggml/src/vulkan-shaders/mul_mat_vec_q4_k.comp`
- `ggml/src/vulkan-shaders/mul_mat_vec_q5_k.comp`
- `ggml/src/vulkan-shaders/mul_mat_vec_q6_k.comp`
- `ggml/src/vulkan-shaders/mul_mat_vecq.comp`
- `ggml/src/vulkan-shaders/scripts/inject_no_contraction.py`
- `ggml/src/vulkan-shaders/vulkan-shaders-gen.cpp`
- `tests/mtp-matrix/35b/BI-matrix.md`

### `f29ef6cd63` — vulkan: batch-invariance for MUL_MAT_ID and FLASH_ATTN, remove force_bi flag

- `ggml/src/ggml-vulkan.cpp`
- `tests/mtp-matrix/35b/BI-matrix.md`

### `716718e7ce` — llama: disable fused MoE up-gate and dense fused up-gate for batch-invariance

- `src/llama-build-context.cpp`
- `tests/mtp-matrix/35b/BI-matrix.md`

### `8e2e1281b9` — vulkan: Phase 3 findings — MMVQ and fused-up-gate net-negative to re-enable

- `ggml/src/ggml-vulkan.cpp`
- `src/llama-build-context.cpp`

## sneaky_dep: full file list per commit
