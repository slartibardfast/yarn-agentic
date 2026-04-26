# MEMORY.md

## Submodule Branch Layout
- `dc/vulkan-split-mode-graph` — 6 commits: multi-GPU split mode graph for Vulkan (Phases 1-12)
- `dc/iqk-scalar-fallbacks` — 1 commit: graceful scalar fallbacks + AVX2 compile flags for non-AVX2 x86
- Branch prefix `dc/` follows upstream convention (`ik/`, `fcp/`, `s6/`)
- Commit messages use `vulkan:` or `iqk:` prefix (no phase references)
- Code comments must not reference phase numbers

## All Phases Complete
- Phases 1-10, 12 implemented. Phase 11 skipped (no opportunity with 3 graph splits).
- dmabuf zero-copy (Phase 12) dropped 7B overhead from ~15 ms/tok to ~1 ms/tok
- 13B multi-GPU faster than single-GPU for token generation (doubled memory bandwidth)

## CLAUDE.md Review (2026-03-08)
- Removed 28 "Phase N:" references from code comments across ggml-vulkan.cpp and ggml-vulkan-multigpu.cpp
- Moved AVX2 CMake compile flags from vulkan branch to IQK branch
- dmabuf detection mismatch in print_gpu_info accepted as cosmetic (device init does full probe)

## Hardware
- **Local (zen2)**: Ryzen 9 3950X, RX 6800 XT + Vega 64, AVX2
- **Alternate (retro)**: Xeon X5650, Polaris 12 + lavapipe, no AVX2 — available if needed

## Phase 13: FUSED_UP_GATE
- Shader bug found and fixed: gate accumulation loop in mul_mm_fused_up_gate.comp was missing cache_b reload. With WNITER>1 and N<BN, gate pass reused stale zeros from UP pass, producing all-zero output.
- CPU backend ABORTs on FUSED_UP_GATE (no implementation). Tests use decomposed reference: separate mul_mat + fused_mul_unary on CPU, then compare via NMSE.
- K-quant types (Q4_K, Q6_K) have block size 256; test K dimensions must be multiples of block size, not just 32.
- GGML_VK_VISIBLE_DEVICES=0 limits to single GPU (VEGA10) — avoids dual-GPU init overhead in tests.
- 50/50 FUSED_UP_GATE tests pass, 1187/1187 standard backend-ops tests pass on RADV VEGA10.

## Bug Fixes Found During Testing (2026-03-09)
- **Empty-graph fence hang**: `graph_compute` set `compute_pending=true` unconditionally. Test framework sentinel nodes (GGML_OP_NONE) produce graphs with zero GPU submissions. Next `synchronize()` spins forever on unsignaled fence. Fix: guard with `submit_count > 0`.
- **MULTI_ADD descriptor range**: `ggml_vk_op_f32` has an incontiguous-op block that overwrites `x_sz` with `ggml_nbytes(src0)`. For MULTI_ADD's strided view_2d, this gives only the view's logical size, not the full expert data span the shader reads. With `robustBufferAccess`, out-of-range reads silently return 0 — shader appeared to sum only expert 0. Fix: override x_sz *after* the incontiguous block.
- **FUSED_UP_GATE M=1 NMSE instability**: Single-element output near zero produces huge NMSE from tiny absolute errors. Fixed by increasing K from 32 to 64 so the output has more signal.
- Final test counts: 1190 standard + 143 FUSED_UP_GATE + 12 MULTI_ADD all pass on RADV VEGA10.

## Nemotron Architecture Confusion (2026-03-09)
- Target model is **Nemotron-3-Nano-30B-A3B** which uses `nemotron_h_moe` (hybrid Mamba2+Attention+MoE), NOT `LLM_ARCH_DECI`.
- DECI is a pure transformer variant (Nemotron-51B, Ultra-253B). Different architecture entirely.
- `nemotron_h_moe` is not recognized by our ik_llama.cpp fork — needs architecture registration, graph builder, and 5 missing Vulkan ops (SSM_CONV, SSM_SCAN, SWIGLU, ADD_ID, SET_ROWS).
- Upstream llama.cpp (added as `llama.cpp` submodule) has full support including Vulkan shaders for all required ops.
- MUL_MAT_ID (expert matmul) is already in our fork's Vulkan backend.

## Submodule Layout Update
- `ik_llama.cpp` — our fork with multi-GPU split mode
- `llama.cpp` — upstream reference for nemotron_h_moe and other missing architectures

## Vulkan REDUCE Op (2026-03-09)
- REDUCE is a cross-device collective ADD used by split-mode graph (`-sm graph`). CUDA uses P2P `cudaMemcpyPeerAsync`; Vulkan has no P2P.
- Implemented as CPU-mediated host staging: `ggml_vk_buffer_read` from each device → CPU ADD → `ggml_vk_buffer_write` back.
- The scheduler's special REDUCE handling (identity tensor_id_copy, no n_inputs increment) was NOT changed — CUDA depends on it. Instead, REDUCE is handled entirely in the Vulkan backend.
- `ggml_vk_reduce()` is called from `graph_compute` before the dryrun/build loop, since REDUCE splits are always single-node graphs.
- Performance: CPU round-trip is slow for graph-split (193 splits, 6.5 tok/s). Layer-split (default, 3 splits, 18 tok/s) doesn't use REDUCE. Future: dmabuf GPU→GPU + ADD shader.

## Phase 18: dmabuf REDUCE (2026-03-10)
- Replaced CPU-mediated REDUCE with dmabuf GPU-to-GPU copy + ADD shader dispatch on destination device.
- Graph-split prompt eval: 9→47 tok/s (5.3×). Token gen: 6.5→7.8 tok/s (+20%).
- Token gen improvement modest because per-REDUCE data is small (6KB F16) — fence latency dominates, not bandwidth.
- Can't bind dmabuf import buffer directly as storage buffer (only eTransferSrc|eTransferDst). Must copy to temp device-local buffer first.
- Separate descriptor pool (1 set) for REDUCE's ADD dispatch avoids entangling with graph pipeline's descriptor management.

## SUM_ROWS Bug (2026-03-10)
- Three bugs: GPU descriptor range only covered ne00×ne01 (not all rows), shader had no bounds check for extra dispatch workgroups, CPU had `ne0` (=1) instead of `ne02` in row index calculation.
- All fixed in one commit. SUM_ROWS now passes on both Vega and 6800 XT.

## Vega 10 (GCN5) Optimization Research
- No native DP4A — integer dot product emulation via float math (dequant+FMA at 4-8 cycles) outperforms "correct" integer emulation. The real gap vs DP4A hardware is 2-4× not 8×.
- Unexploited: Vega's Rapid Packed Math (f16vec2 packed FP16 arithmetic). llama.cpp Vulkan shaders don't use RPM in fallback paths.
- Optimization strategy: f16vec2 packed arithmetic + careful VGPR budgeting (≤32 regs for max occupancy) + full 64-lane wavefront subgroup reductions.
- AMDVLK is extinct; RADV is the sole Vulkan driver for GCN. RADV continues improving but dedicated GCN shader optimization in inference frameworks is an unfilled niche.

## Phase 0: K-Quant Dequant Bounds Check Bug (2026-03-11)
- 5 K-quant dequant shaders (q2_k through q6_k) used `p.M * p.K / QUANT_K` for bounds checking. For multi-batch tensors, `p.M×p.K` only covers one batch — remaining batches left as uninitialized garbage in prealloc buffer.
- Fix: change to `p.nel / QUANT_K` (nel = total elements across all batches).
- This single fix resolved 7 of 10 test-backend-ops failures: 4 direct (q4_K×f16 batched) + 3 indirect (prealloc contamination for iq3_xxs).
- Key insight: "flaky" test failures in GPU backends can be prealloc buffer contamination from a completely different operation's bug.

## Phase 0 Round 2: Push Constant & get_offsets Alignment (2026-03-12)
- Aligned fork's mul_mat_vec push constant struct with upstream: added fusion_flags, base_work_group_y (non-ID) / expert_i1, nbi1 (ID).
- Moved `batch_stride_a / QUANT_K` division into get_offsets() (matching upstream), removed inline `a_offset / QUANT_K` from mul_mat_vec.comp and 12 specialized shaders.
- Fixed iq3_xxs MUL_MAT and MUL_MAT_ID (both now pass).
- 3 failures remain: iq4_xs MUL_MAT (NMSE=0.040), bf16 k=1 MUL_MAT (NMSE=4.2), iq4_xs MUL_MAT_ID (NMSE=0.032). All reproduce in isolation — NOT contamination.
- bf16 is the deepest puzzle: GLSL is truly identical to upstream, push constants now match, yet massive errors. Only remaining structural difference is 2 extra descriptor bindings (Fuse0/Fuse1) in upstream SPIR-V.

## Vega Inference Bug: get_tensor_async Race Condition (2026-03-11)
- **Root cause**: `get_tensor_async` calls `buffer_read_2d_async` which does a synchronous memcpy from host-visible (rBAR) GPU buffer. But `get_tensor_async` is called BEFORE `synchronize()`, so GPU compute may still be in flight. On Vega (slower), the CPU reads stale data; on 6800 XT (faster), the GPU usually finishes in time — pure timing-dependent race.
- **Symptoms**: garbage output when ngl > n_layers (output matmul on GPU). ngl <= n_layers works (output stays on CPU). All backend-ops tests pass (they use synchronous reads).
- **Misleading clue**: inserting submit+fence_wait mid-graph "fixed" the issue — this forced the compute to complete before the next `get_tensor_async` read.
- **Fix**: for host-visible non-UMA buffers, `get_tensor_async` records a deferred memcpy in `ctx->pending_host_memcpys` (new field on backend context) instead of copying immediately. `synchronize()` processes these after `sync_compute` waits for the compute fence.
- **Why transfer_ctx->out_memcpys didn't work**: `graph_cleanup` in `sync_compute` clears `gc.contexts`, destroying the transfer context. The weak_ptr `ctx->transfer_ctx` expires, so `synchronize()` exits early without processing `out_memcpys`.
- **Key insight**: `buffer_read` (synchronous) checks `eHostVisible && uma` before direct memcpy, but `buffer_read_2d_async` only checked `eHostVisible`. The UMA check prevented the issue on UMA devices where compute and host share coherent memory. For discrete GPUs with rBAR, the async path was broken.

## Phase 0 Round 3: Structural Alignment with Upstream (2026-03-13)
- Added Fuse0/Fuse1 buffer bindings (binding 3, 4) to mul_mat_vec_base.comp; moved IDS from binding 3 to 5. Pipeline creation 3→5 (non-ID), 4→6 (ID). Dispatch passes dummy Fuse subbuffers.
- Disabled spirv-opt for bf16 (issue #15344) and rope (issue #16860) shaders in vulkan-shaders-gen.cpp. BUT: bf16 was already effectively unoptimized (17044 bytes matches no-opt output). Not the root cause.
- **Neither change fixed any test failure.** Kept for structural correctness (SPIR-V and pipeline layout now match upstream).
- Remaining 5 failures unchanged: bf16 k=1 (NMSE~3-6), iq4_xs MUL_MAT (NMSE~0.02), iq4_xs MUL_MAT_ID (NMSE~0.03), 2x CPY f32→iq4_nl.
- Key remaining upstream differences: (1) iq4_xs dequantize4 reads individual bytes vs upstream's packed32+unpack8, (2) no data_a_packed32/v4/packed16 buffer aliases, (3) bf16 k=1 passes alone but fails in full suite despite using own buffer (not prealloc).
- Build system caveat: changing vulkan-shaders-gen.cpp does NOT auto-trigger shader regen. Must manually rm -rf build/ggml/src/vulkan-shaders-gen-prefix, rebuild tool, then re-run it.

## Phase 0 Round 4: bf16 k=1 Was a CPU Bug, Not GPU (2026-03-13)
- **Critical lesson**: bf16 MUL_MAT k=1 (NMSE=2.88) was NOT a GPU bug. The GPU produced correct output. The CPU reference (IQK) produced garbage because `iqk_set_kernels_float` requires `ne00 % 32 == 0` for bf16 — k=1 fails this check, IQK returns false, fallback also broken.
- **Fix**: Added scalar bf16 mul_mat fallback in IQK for `ne00 < k_step`. Three-tier dispatch: AVX512BF16 (ne00%32), generic SIMD (ne00%k_step), scalar (any ne00).
- **Methodology that found it**: Wrote minimal standalone test printing GPU vs CPU values side-by-side. GPU: correct. CPU: all zeros. Took 10 minutes.
- **Methodology that failed**: 6+ hours exhaustively comparing GPU dispatch code, push constants, SPIR-V bytecode, spec constants vs upstream. Everything matched. The bug wasn't there.
- **Rule**: ALWAYS verify CPU reference output before debugging GPU. Large NMSE (>1.0) is a red flag for CPU reference bugs, not GPU bugs. Edge-case dimensions (k=1) break CPU alignment assumptions.
- Remaining 4 failures: iq4_xs MUL_MAT, iq3_xxs MUL_MAT (marginal), iq4_xs MUL_MAT_ID, 2x CPY f32→iq4_nl.

## Phase 0 Round 5: iq4_xs Was Also a CPU Bug (2026-03-13)
- **Same pattern as bf16**: iq4_xs MUL_MAT (NMSE=0.027) was NOT a GPU bug. GPU matched float64 expected at machine epsilon (1.8e-14). IQK's AVX2 `DequantizerIQ4XS` kernel has a systematic computation error in its unsigned-value bias compensation.
- **Fix**: Added scalar iq4_xs×Q8_K dot product (`mul_mat_iq4_xs_q8_K_scalar`) using signed `kvalues_iq4nl` directly. Used on non-AVX512 systems. CPU NMSE: 0.027 → 2.97e-05.
- **Collateral fix**: iq3_xxs MUL_MAT (marginal, NMSE=0.00065) now passes — likely IQK precision improvement from the same scalar fallback approach.
- **iq4_xs dequantize4 byte-by-byte vs packed32**: Verified mathematically equivalent (same bits after shift+mask). NOT a bug despite the code difference.
- **ggml-cpu fallback path is broken**: When IQK returns false from `iqk_set_kernels_kquants`, ggml-cpu's standard `ggml_vec_dot` fallback produces zeros. Must fix IQK internally, not disable it.
- Remaining 3 failures: MUL_MAT_ID(iq3_xxs) marginal, 2x CPY f32→iq4_nl.
- Also discovered: `ggml_internal_get_type_traits(GGML_TYPE_IQ4_XS).to_float` requires `ggml_init()` first — the `ggml_table_f32_f16` lookup table is populated lazily during init. Standalone tests that skip init get zeros from dequantize functions.

## Phase 20g: Multi-GPU MoE root cause = GROUPED_TOPK CPU fallback (2026-04-08)

Target: Qwen3.5-35B-A3B-UD-IQ3_XXS (12.17 GiB / 34.66 B params, MoE 256 experts × 8 active, 40 layers, n_kv_head=2, head_dim=256). Dual-GPU rig: RDNA2 6800 XT 16 GiB + Vega 64 8 GiB.

Measured baselines (single-GPU 6800 XT, post-Phase-20f reduce fix):
- pp256 / tg64 = **0.71 / 0.31 t/s** with default flags
- pp256 / tg64 = **3.42 / 0.31 t/s** with `-fmoe 0` (5× pp gain, tg unchanged because Phase 20b already skips fused-moe for N=1)
- Compare dense Llama-2-13B Q8_0 same GPU, similar VRAM (12.88 GiB): **511 / 27.7 t/s** — MoE is **150-720× slower per token**.

Multi-GPU layer-split (`-sm layer -ts 0.67/0.33`) gives **0.63 / 0.27 t/s** — same as single-GPU. **Multi-GPU is not the bottleneck.**

Root cause found via `llama-cli ... 2>&1`: prints **`graph splits = 322`** for Qwen3.5 MoE on Vulkan (compare ~3 splits for dense models). 322 splits = ~161 CPU↔GPU round-trip boundaries × ~10 ms each = ~3.2 sec/token of pure sync overhead, matching measured wall time. GPU-side compute is healthy (~15 ms/token via `GGML_VK_PERF_LOGGER=1` per-op timings).

The split source: **`GGML_OP_GROUPED_TOPK` is not in the Vulkan supports_op list** at `ggml/src/ggml-vulkan.cpp:9905-9923`. The op is defined in `ggml.h:658` and used by `llm_build_moe_ffn` at `llama-build-context.cpp:1131` for expert routing (`ggml_grouped_topk`). Every layer's expert selection therefore falls back to CPU, splitting the graph into hundreds of fragments.

**Phase 20h scope** (next plan): implement `GROUPED_TOPK` on the Vulkan backend (one new compute shader + dispatch wiring). Expected outcome: graph_splits drops from 322 → ~3, tg perf goes from 0.31 → ~25-40 t/s for Qwen3.5-35B-A3B IQ3_XXS, matching dense-model expectations.

Bug A (separate): `-fmoe=1` (default) makes pp eval 5× slower than `-fmoe=0` for MoE on Vulkan. Phase 20b's N=1 skip works for tg but the prompt-eval path still goes through fused MoE. Either fix the fused-moe pp path or change the default to `-fmoe=0` for MoE models on Vulkan.

## Phase 20h: Vulkan MOE_FUSED_UP_GATE shader (2026-04-08)

Bug A fixed by adding the missing GGML_OP_MOE_FUSED_UP_GATE Vulkan implementation. The op was defined in ggml.h and implemented for CPU+CUDA, but not Vulkan, so multi-token MoE prompt-eval went through unfused fallback.

Implementation: extended the existing dense `mul_mm_fused_up_gate.comp` with a `MUL_MAT_ID` build flag mirroring the upstream `mul_mm.comp` MUL_MAT_ID branches. Push constants are layout-shared between dense and MoE — macro aliases (`p_nei0`/`p_nei1`/`p_nbi1`/`p_ne11`) re-interpret the dense `ne0X/broadcast` slots as the MoE indirection fields, so a single C struct serves both. Backported the full IQ family (IQ1_S..IQ4_XS) into the dense fused_up_gate gen as well, since IQ3_XXS is the target model.

The non-obvious bug: a buffer subbuffer scoping mistake. The dispatch passed `b_sz = sizeof(float) * ne10 * ne11` to the b binding, which is one token's worth of data. The shader strides across tokens via `row_idx.y * batch_stride_b`, so it needs the FULL tensor visible to the binding. Fix: `b_sz = sizeof(float) * ne10 * ne11 * b->ne[2]`. Single-token cases passed (only token 0 was needed), but multi-token cases produced zero output for tokens > 0 because the GPU's bounds-protected reads outside the bound subbuffer returned zero. Diagnosed by reading `buf_b` shared memory back into `dst` from the write loop and observing col 1 = 0 even though the load formula was correct.

Headline result: Qwen3.5-35B-A3B-UD-IQ3_XXS pp256 on 6800 XT with default `-fmoe=1`: 3.47 t/s (vs the broken 0.71 t/s pre-fix and the unfused 3.42 t/s baseline). The fused path is now at parity with unfused on this model — the inference graph no longer falls back to CPU for the fused MoE FFN op. Test coverage: 40 new test_moe_fused_up_gate cases pass on both Vega 64 (RDNA-GCN5) and 6800 XT (RDNA2) on top of the existing 1237/1237 baseline and 143/143 dense FUSED_UP_GATE cases.

Note for Phase 20i: the headline `tg32 = 0.31 t/s` is unchanged because token gen is bottlenecked by `GROUPED_TOPK` falling back to CPU (322 graph splits per token). 20h was a prerequisite (so the fused MoE path is GPU-resident), but the tg headline win has to wait for Phase 20i.

## Phase 20i: Vulkan GROUPED_TOPK shader (2026-04-08)

Added the GGML_OP_GROUPED_TOPK Vulkan implementation as a single fused compute shader (`grouped_topk_f32.comp`), one workgroup per token row. All stages (per-group sort + sum, group selection, masking, global sort, write top-k indices) run in shared memory with no temp buffers. 7/7 test cases pass on both Vega 64 and 6800 XT covering Qwen3.5 and DeepSeek-style shapes.

**However**: this op is NOT what was bottlenecking Qwen3.5-35B-A3B-UD-IQ3_XXS. The Phase 20g diagnosis was wrong on the root cause. With `GGML_SCHED_DEBUG=2` it's clear that Qwen3.5 uses ordinary `ARGSORT` (which is already on Vulkan) for expert routing, not `GROUPED_TOPK`.

**The real picture: Qwen3.5-35B-A3B is a hybrid Mamba-Transformer model.** The GGUF metadata shows `qwen35moe.ssm.conv_kernel = 4`, `ssm.state_size = 128`, `ssm.inner_size = 4096`, `full_attention_interval = 4`. So 30 of 40 layers are SSM (state-space-model) layers, each producing several CPU ops on Vulkan. Full breakdown of CPU ops per inference graph (`GGML_SCHED_DEBUG=2`):

| Count | Op | Status on Vulkan |
|---|---|---|
| 120 | `L2_NORM` | Missing — 4 per SSM layer (q_fused, k_fused × 30 SSM layers) |
| 80 | `MUL_MULTI_ADD` | Missing — MoE expert gather-and-add (`routed_out` per layer × 80 MoE blocks) |
| 60 | `UNARY` (softplus) | Missing — `a_softplus` (DELTA-net alpha activation) per SSM layer × 2 |
| 60 | `SSM_CONV` | Missing — state-space 1D conv per SSM layer × 2 |
| 60 | `DELTA_NET` | Missing — the heavy delta-net update per SSM layer × 2 |
| 40 | `FUSED_MUL_UNARY` | Has shape mismatch — `shared_expert_gate × ffn_shexp_out`, supports_op `ggml_are_same_shape` rejects it |
| 2 | `GET_ROWS` | Token embedding init |

**Total: 422 CPU ops → 322 init-time graph splits → 724 sched-debug SPLIT lines per call.**

For each SSM layer: roughly 5 CPU splits + 5 Vulkan return splits = 10 splits/layer. 30 SSM layers × 10 = 300 splits, plus a few extras = 322. Phase 20i shipped GROUPED_TOPK because it's correct and the right op for DeepSeek-V3, BailingMoE, etc. — but it does NOT move the Qwen3.5 needle at all. Closing the Qwen3.5 gap requires implementing the 6 missing/broken ops above. The highest leverage is probably **L2_NORM** (largest count, simplest shader), followed by **SOFTPLUS unary** (trivial), then **MUL_MULTI_ADD** (moderate). The SSM/DELTA-NET ops are the most complex but also the heaviest computationally — they should be the last priority because they hold the biggest perf gain when ported.

The decision for Phase 20j: rather than trying to chase Qwen3.5, it might be better to pick a target model that doesn't depend on SSM. If the goal stays Qwen3.5, the next phase should be a single multi-op port that includes at least L2_NORM + SOFTPLUS so the layer-internal splits collapse, then iterate.

## Upstream prior work — almost everything needed exists in upstream llama.cpp (2026-04-08)

Verified via `gh pr list --repo ggml-org/llama.cpp --search "vulkan <op>"`. Of the 6 ops blocking Qwen3.5-A3B, **5 of them have merged Vulkan implementations in upstream llama.cpp** that we can port directly:

| Op | Upstream Vulkan PR | Status | Notes |
|---|---|---|---|
| L2_NORM | #19604 + #20350 | MERGED | Already 90% wired locally — shader, dispatch fn exist; case statements commented out in supports_op |
| SOFTPLUS (UNARY) | #17319 | MERGED | giuseppe — bulk unary ops including SOFTPLUS, STEP, ROUND, CEIL, FLOOR, TRUNC |
| SSM_CONV | #16463 + #20379 | MERGED | giuseppe (base) + ProgenyAlpha (PP scaling fix) |
| (GATED_)DELTA_NET | #19504 (op) + **#20334 (Vulkan)** | MERGED | am17an added the op def + ProgenyAlpha added the Vulkan impl. Upstream calls it `GATED_DELTA_NET`; ik fork calls it `DELTA_NET` — need to verify op_params layout matches |
| MUL_MULTI_ADD | — | N/A | ik-specific. No upstream equivalent. Greenfield, but logic is straightforward (gather + sum across experts) |
| FUSED_MUL_UNARY broadcast | — (#17319 covers scalar) | N/A | Local — extend our existing supports_op shape check + maybe add broadcast variant to the existing shader |

Reference: ik_llama.cpp PR #1251 (closed, "Qwen 3 Next experiment" by YurkoHoshko) — Codex-generated CUDA-only port of `ssm_conv.cu`, `gated_delta_net.cu`, etc. Provides the op_params layout reference for translating between upstream `GATED_DELTA_NET` and ik fork `DELTA_NET`.

**This converts Phase 20j-20o from "write 4-6 new shaders from scratch" to "port 4 existing shaders + 1 small greenfield + 1 supports_op fix."** The shipping order — L2_NORM (trivial uncomment) → SOFTPLUS (port) → SSM_CONV (port) → FUSED_MUL_UNARY broadcast (local fix) → MUL_MULTI_ADD (greenfield) → DELTA_NET (port + op_params translation).

## Phase 20n: Vulkan SSM_CONV full coverage (2026-04-08)

Ported all 5 SSM_CONV CUDA kernels from ik PR #1251 (`ggml-cuda/ssm-conv.cu`, 608 lines) to Vulkan in three layers:

**Single-sequence fast path** (Qwen3.5 single-stream inference, the headline target):
- 3 SPVs: `ssm_conv_x.comp` (NC4 + general) + `ssm_conv_final_state.comp`
- Parallel over (row, token); two dispatches per call (conv output + final-state writeback)
- Qwen3.5-35B-A3B-UD-IQ3_XXS on 6800 XT: graph splits 122 → **62**, **pp256 3.47 → 12.69 t/s (+266%)**, tg32 0.32 → 0.36 (modest because DELTA_NET still dominates the per-token critical path)

**Multi-sequence slow path** (correctness for any n_kv, any sq layout):
- 6 SPVs: `ssm_conv_init_states.comp` (NC4 + general) + `ssm_conv_slow.comp` (NC4 × HAS_MULTI_SEQ × {ungated})
- Init kernel pre-fills dst_state from src0 for all n_kv sequences so untouched seqs survive the batch
- Slow kernel walks tokens serially per row, handling self-recurrence (state shift), invalid seq ids, and multi-target fanout

**Multi-sequence unique-fast path** (parallel-over-tokens optimization for serving):
- 7 SPVs: `ssm_conv_validate.comp` + `ssm_conv_unique.comp` (NC4 + general) + `ssm_conv_slow.comp` GATED variants (NC4 × HAS_MULTI_SEQ)
- GPU-side `fast_path_ok` atomic flag — validate kernel atomically clears it if seq map has out-of-range / fanout / recurrence
- Both unique-fast and slow_gated dispatched together; one early-exits based on the flag (matches CUDA reference exactly)
- Persistent SSBO `ssm_conv_atomic_buf` per `vk_device`, lazily allocated, grown on demand (rounded to 4096 entries)
- Layout: `[fast_path_ok, seq_seen[n_kv], seq_ids[n_t]]`

Total: 6 shader files, 16 SPV variants, ~700 LOC across shaders + C++ wiring.

Verification (Vega 64 + 6800 XT both):
- 1258/1258 baseline tests pass (1252 + 6 new SSM_CONV cases for multi-seq)
- 13 SSM_CONV cases: 7 single-seq + 6 multi-seq (unique × 2 batch sizes × 2 nc, recurrent × 2 nc, fanout × 1)
- No regression on dense MUL_MAT, FUSED_UP_GATE, MOE_FUSED_UP_GATE, L2_NORM, GROUPED_TOPK, MUL_MULTI_ADD, FUSED_MUL_UNARY

The non-obvious bit: ggml itself enforces `sq->ne[0] == n_kv` (line 10337 of `ggml/src/ggml.c`), so the "n_kv=1, sq->ne[0]>1" corner case I considered planning for is unreachable. There are exactly 4 dispatch paths: single-seq fast, multi-seq init+slow, multi-seq init+validate+unique+slow_gated (the validate decides at runtime which of unique/slow_gated does the work).

Remaining CPU bottlenecks for Qwen3.5: 60 DELTA_NET (the heaviest op, next phase) + 2 GET_ROWS. After DELTA_NET ships, splits should drop to ~3 and tg should jump significantly.

## Phase 20j-20m: Qwen3.5-A3B Vulkan op ports (2026-04-08)

Shipped 4 of the 6 missing ops over a single working session:

| Phase | Op | Source | CPU instances cleared | Splits change |
|---|---|---|---|---|
| 20j | `L2_NORM` (non-contig) | Ported upstream `l2_norm.comp` (PR #19604), our local fork already had the shader and dispatch fn but commented out — replaced contiguous-only variant with the upstream non-contig one and uncommented the case statements | 120 | 322 → 262 |
| 20k | `UNARY` SOFTPLUS | Ported upstream `softplus.comp` (PR #17319) + new pipeline arrays + wired into existing CREATE_UNARY macro | 60 | 262 → 202 |
| 20l | `MUL_MULTI_ADD` | Greenfield (ik-only op). Wrote `mul_multi_add.comp` from scratch — single-shader, one workgroup per (k_block, token) pair, accumulates the per-expert weighted sum | 80 | 202 → 122 |
| 20m | `FUSED_MUL_UNARY` (SIGMOID + scalar broadcast) | Local fix. The op was already supported for SILU/GELU/RELU same-shape, but Qwen3.5 uses `ggml_fused_mul_unary(scalar_gate, [n_ff], SIGMOID)` for shared-expert gating — needed (1) a new `fused_mul_sigmoid.comp` shader, (2) `BCAST` define added to the existing 3 shaders + new sigmoid one for the scalar-broadcast case, (3) `supports_op` extended to accept `ggml_nelements(src0) == 1`, and (4) the dispatch picks the bcast variant when src0 is a single element. The 8 new pipelines (silu/gelu/relu/sigmoid × bcast × f32+f16) cover both the existing fusions and the new Qwen3.5 path. | 40 (compute moved) | 122 (unchanged — these ops were already adjacent to SSM CPU stretches so didn't add new splits, but the compute is now on GPU) |

**Total: 322 → 122 splits, -62%.** Remaining 122 splits are SSM_CONV (60) + DELTA_NET (60) + 2 GET_ROWS, all part of the SSM/Mamba layer stretches.

Headline tg32 on 6800 XT: 0.31 → 0.32 t/s — barely moved despite the split count drop. The SSM/DELTA-NET layers still execute on CPU and consume the bulk of the per-layer time, so eliminating the cheap ops (L2_NORM, SOFTPLUS, MUL_MULTI_ADD, FUSED_MUL_UNARY) just frees them from CPU but doesn't reduce the critical path. The big wins are still SSM_CONV and DELTA_NET.

**Phase 20n (deferred for next iteration): SSM_CONV.** The ik fork uses a 4-arg `ggml_ssm_conv(s, x, c, sq)` (state, input, conv-weights, sequence-indices) which is the older "stateful" form, NOT the upstream 2-arg form (`ggml_ssm_conv(a, b)`) that the upstream Vulkan shader handles. I attempted a port using the upstream `ssm_conv.comp` but had to back it out — the shader doesn't match our op semantics. Reference: ik PR #1251 (closed, "Qwen 3 Next experiment") added a CUDA port (`ggml-cuda/ssm-conv.cu`, 608 lines, ProgenyAlpha-style). Need to either port that CUDA logic to Vulkan, or refactor the ik op to match upstream's 2-arg form (riskier).

**Phase 20o (deferred): DELTA_NET.** Same problem — ik fork has a different signature than upstream. The upstream Vulkan port (PR #20334, ProgenyAlpha) handles `GATED_DELTA_NET`. ik's `DELTA_NET` op shares some semantics but the CPU implementation is in `iqk/iqk_cpu_ops.cpp`. Will need study against ik PR #1251 + upstream #20334 to map between the two forms. This is the heaviest op (most expensive in compute) so it's the biggest perf win.

f16acc dispatch counter (Phase 20c instrumentation) on this MoE model showed `hits=0 fallbacks=43704`. Hits=0 is correct on RDNA2 (f16acc is Vega-only). On Vega the same 43704 dispatches would hit the f16acc path — but the bottleneck is graph_splits, not compute, so the f16acc work cannot move the needle until Phase 20h lands.

## Phase 20o: Vulkan DELTA_NET — final ops-coverage milestone (2026-04-08)

The recurrent linear-attention core. ik fork's `ggml_delta_net` is a 6-arg
op (q, k, v, g, beta, state) with a TRANSPOSED state storage layout
(`state[col*head_dim + row]`) compared to upstream's `ggml_gated_delta_net`
(`state[v_idx*S + k_idx]`). Wrote a custom shader matching ik's
algorithm; the upstream PR #20334 served only as a parallelization template.

**Architecture**: one workgroup per (head, seq); each thread holds ONE row
of state in registers (HEAD_DIM floats). State load/store happens once per
dispatch — amortized across n_tokens. HEAD_DIM ∈ {64, 128} via #define
(matches iqk fast path). Two reduction strategies: shmem (universal) and
subgroup-add (only when HEAD_DIM == subgroup_size — Vega + h64).

**Two bugs found and fixed during implementation:**
  1. Spec-constant aliasing — using `local_size_x_id = 0` while also
     declaring `constant_id = 0 const uint HEAD_DIM` produced two distinct
     SPIR-V spec constants both with SpecId=0; only one got bound. The
     workgroup ran with size 1. Fix: switched to `#define HEAD_DIM` baked
     in at SPV-gen time, two SPVs per reduction strategy.
  2. Dispatch-grid scaling — `ggml_vk_dispatch_pipeline` divides
     `elements` by `wg_denoms`. With `wg_denoms = {64,1,1}` and elements
     `{H_v=2, n_seqs=1, 1}`, ceil(2/64) = 1 → only ONE workgroup
     dispatched. Fix: pass `{H_v * HEAD_DIM, n_seqs, 1}` so the grid
     resolves to (H_v, n_seqs, 1) workgroups after the divide.

**Test coverage**: 19 cases × 2 GPUs (RDNA2 + Vega) all pass.

**Headline measurement** (Qwen3.5-35B-A3B-UD-IQ3_XXS, 6800 XT, sm none):

| Metric          | Pre-Phase-20 | Phase 20m | Phase 20n | Phase 20o |
|---|---:|---:|---:|---:|
| pp256 t/s       | 0.71 (-fmoe) |     —     |    —      | **146.41** |
| tg64 t/s        | 0.31         |   0.32    |   0.46    |  **18.18** |
| graph splits/tok| 322          |   122     |    62     |    **2**   |

**~206× pp256 and ~58× tg64 vs the original baseline.** Every recurrent
op in Qwen3-Next now runs on Vulkan; only 2 backend boundaries remain per
token. After this phase the model is production-ready on RDNA2 Vulkan.

**Vega NaN issue**: was a pre-existing f16acc bug in Q4_K/Q5_K/Q6_K shaders,
NOT in DELTA_NET. Fixed in Phase 20p below.

## Phase 20p: f16acc fix — correct mixed-precision implementation (2026-04-09)

Phase 20c (f16acc mul_mat_vec for Vega RPM) set `FLOAT_TYPE=float16_t`,
making the ENTIRE accumulation chain f16. This caused NaN on Qwen3.5-35B-A3B
(which uses Q6_K for 252 of 733 tensors). Test-first approach isolated it:

| Type | b=10 | b=50 | Root cause |
|---|---|---|---|
| Q8_0, Q2_K, Q3_K, IQ* | OK | OK | No f16 scale multiply or small scales |
| **Q4_K** | OK | **NaN** | 6-bit scale × dot overflows f16 at B=50 |
| **Q5_K** | **NaN** | **NaN** | 6-bit scale × dot overflows f16 at B=10 |
| **Q6_K** | **NaN** | **NaN** | 8-bit scale (±128) × dot overflows at B=10 |

The overflow is in `f16vec2(dot, dot) * f16vec2(scale, scale)` — the sub-block
scale × dot-product multiply in the Q4_K/Q5_K/Q6_K USE_F16ACC blocks.

**Fix (two parts)**:
1. Promoted the 3 scale-multiply sites to explicit `float` arithmetic.
2. Changed `FLOAT_TYPE=float` in vulkan-shaders-gen.cpp for all f16acc
   variants. temp[], sccache[], tmpsh[], reduce_result all f32 now.

The 8 inner f16vec2 fma's (the v_pk_fma_f16 RPM benefit) are UNCHANGED.
ISA confirmed: 1175 v_pk_fma_f16 instructions still emitted on Vega.

**32 new stress tests** (`test_mul_mat_stress` with B∈[-10,10] and B∈[-50,50])
for 8 quant types × 2 n-values. These failed before the fix (Q4_K/Q5_K/Q6_K
NaN) and pass after. All 1309 tests pass on Vega.

Qwen3.5-35B-A3B generates coherent text on Vega with f16acc enabled.

## Phase 21: Dispatch reduction for hybrid Mamba tg (2026-04-09)

Profiling (GGML_VK_PERF_LOGGER=1) revealed the hybrid Mamba model's tg
bottleneck was NOT DELTA_NET compute — it was dispatch overhead:
- RDNA2: 41 ms CPU dispatch overhead across 810 dispatches (~50 μs each)
- Vega: GPU pipeline stalls of 100-250× between dependent dispatches
  (CONCAT 2540 μs vs 17 μs on RDNA2 for the same 2 MB copy)

**Tier 1: Inplace state writeback** — DELTA_NET writes SSM state directly
to KV cache (src[5]) via STATE_INPLACE shader variants, eliminating
CONT+CONCAT+CPY chain. GPU stall total: 198.8 → 69.5 ms (-65%).

**Tier 2: GGML_OP_FUSED framework** — ported from phase25-decode-perf
branch. Single op enum + fusion_id dispatch. First fusion: GATE_PREP
(add+softplus+mul → 1 dispatch, saves 48 dispatches/token). SILU_MUL
stub also landed (CPU kernel + Vulkan pipeline reusing fused_mul_silu).

Combined: RDNA2 pp256 +7%, tg +1.8%. The 2.5× tg gap to dense is
architectural (2× more ops per token for Mamba), not an optimization bug.

Polaris-jit branch's megakernel/JIT was confirmed dead (12% slower than
standard dispatch). Op-level fusions are the proven pattern.

## Model survey and deployment findings (2026-04-09)

Downloaded and benchmarked Qwen3.5-122B-A10B UD-Q4_K_XL (72G, 3 split
files) and Qwen3.5-35B-A3B Q4_K_M (20.5G single file) for multi-GPU
inference on the RDNA2 + Vega rig.

**Qwen3.5-122B-A10B UD-Q4_K_XL** (122B params, 10B active MoE):
- 72G model, -ngl 14, -sm layer -ts 0.67/0.33, 65K context viable
- Vulkan0: 14.9G, Vulkan1: 6.3G — both within VRAM
- tg: 1.6-2.1 t/s — CPU-bandwidth-bound (56 of 80 layers on DDR4)
- A 122B model running at interactive speed on consumer AMD hardware

**Qwen3.5-35B-A3B Q4_K_M** (35B params, 3B active MoE):
- 20.5G model, -ngl 999 -sm layer, 65K context, ALL layers on GPU
- Vulkan0: 14.8G (layers 0-27), Vulkan1: 7.1G (layers 28-40 + lm_head)
- pp256: 117 t/s, tg64: 11.3 t/s, graph splits: 3
- Multi-GPU +18% over single-GPU (11.3 vs 9.6 t/s)
- 2× faster than existing 5 t/s server on the same model
- Profiling: 82% of wall time is CPU dispatch overhead (1482 dispatches
  × ~51 μs each). GPU compute is only 16 ms. Our Phase 20 Vulkan ops
  (DELTA_NET 19 μs, SSM_CONV 8 μs, FUSED 4 μs) are negligible.

**IQ3_XXS vs Q4_K_M for tool calling**: the 11 vs 18 t/s gap is small.
Q4_K_M has PPL 6.6 vs ~7.0 and KLD 0.55 vs 1.53 — substantially
better structured output reliability. For tool calling use cases, Q4_K_M
on dual-GPU is the correct deployment choice.

**Hybrid Mamba context efficiency**: Qwen3.5-35B-A3B at 65K context uses
only 1.3G of KV cache (10 attention layers out of 40). A pure transformer
of similar size would need ~10G+. This makes long-context deployable on
consumer GPU VRAM where pure transformers cannot fit.

**Disk cleanup**: removed 164G of unused models (Nemotron-3-Nano 3 of 4
quants, GLM-4.7-Flash, embedding models, HF safetensors duplicates) to
make room for the 122B download. 158G free on /opt after cleanup.

## Build Notes
- Use clang (GCC 15 has -Wtemplate-body errors)
- `-DGGML_IQK_FLASH_ATTENTION=OFF` on non-AVX2 hosts
- Remote: use absolute `-S`/`-B` paths with cmake (SSH starts in /home/llm)

## Phase 23: Parametric GPU codebook for TURBO_*B (2026-04-16)

Replaced hardcoded `const float[]` codebook arrays in `turbo_rht.glsl`
with a storage buffer binding (TURBO_CB_BINDING). Each TURBO_*B shader
now reads its codebook from a per-bitrate `vk_buffer` owned by
`vk_device_struct`, populated at device init with the published
Lloyd-Max Gaussian centroids. A custom codebook (imatrix-weighted or
per-model Lloyd-Max) can overwrite the buffer contents without shader
recompilation — no CPU fallback needed.

Binding layout:
- dequant_turbo: 3 bindings (A, D, codebook) — was 2
- mul_mat_vec_turbo: 6 bindings (A, B, D, Fuse0, Fuse1, codebook) — was 5

Added a `std::vector<vk_subbuffer>` overload of `ggml_vk_dispatch_pipeline`
because the existing `std::initializer_list` signature can't handle a
binding count that depends on runtime type. Dispatches for TURBO types
append the codebook buffer; all other types use the old fixed binding
list.

Verified: 40/40 backend-ops MUL_MAT pass on TURBO_3B/4B/5B across wave32
(RX 6800 XT) and wave64 (Vega). TURBO_4B end-to-end PPL 20.84 @ 20 chunks
matches pre-change baseline. q4_0/q4_k unaffected.

## Phase 23: Codebook pipeline + D2 outlier protection complete (2026-04-17)

End-to-end custom codebook pipeline shipped:
- `llama-quantize --codebook PATH` reads centroids, applies via
  `turbo_set_quantize_codebook`, embeds tensors into output GGUF.
- `llama_model_loader::apply_turbo_codebooks()` reads embedded tensors
  and forwards to Vulkan via `ggml_backend_reg_get_proc_address`.
- `turbo_set_quantize_codebook` now derives effective `cent_max` from
  `max(|centroid|)` so tool-normalized ([-1,1]) codebooks scale
  correctly through the quantizer. Without this fix PPL was 4439
  (vs 20.40 with fix).

Unsloth Dynamic 2.0-style outlier promotion implemented:
- Bumps attn_v/k/q/qkv/output and edge ffn_down within TURBO family
  where UD targets match, falling back to Q6_K/Q8_0 for the
  highest-precision tensors. Code in `llama_tensor_get_type_impl`.

PPL findings: on SSM-hybrid qwen35-0.8b, TURBO+D2 is **dominated by
k-quants** at the same bpw (TURBO_4B-D2 20.54 @ 5.85 vs Q4_K_M 19.67
@ 5.50). On dense transformer Qwen2.5-Coder-1.5B (via Q8_0→quant
path), gap narrows dramatically: TURBO_4B-D2 16.21 @ 5.04 bpw vs
Q4_K_M 15.80 @ 5.08 bpw — only +0.41 PPL at same bitrate.

**TURBO_2B-D2 on dense is 33.63 PPL @ 3.29 bpw** — a genuine usable
quality point (vs 354 PPL on SSM-hybrid at higher 4.37 bpw, vs 1638
PPL for pure TURBO_2B with E8P). The 10x improvement on the same
bitrate is entirely from the dense model's weight distribution
matching RHT's incoherence design target.

Key insight: RHT+codebook works on attention/FFN weights
(incoherence target) but underperforms on SSM state projections.
The TURBO thesis is viable for dense transformers at all bitrates
(2B through 5B); marginal for SSM-heavy models where k-quants
dominate. Vulkan TURBO_2B GLSL port is now worth doing (previously
deferred because 2B was broken; now it's a useful quality point
where GPU support matters for performance).

## 2026-04-18 — per-layer sensitivity sweep on qwen35-0.8b + pause

Ran a fresh singleton-layer sweep: for each layer L ∈ [0, 25),
quantize HARP_2B_S with ONLY blk.L attn+ffn promoted to Q5_K,
measure wikitext-2 (20 chunks) PPL vs baseline 33.78. Five parallel
Bash batches covered layers 0-4, 5-9, 10-14, 15-19, 20-24.

Key findings (full table in DATA.md § Ablation 11):
- **L23 is the dominant sensitivity spike** (−3.77 PPL alone,
  2× any other layer). Drives most of the "last 3" Abl7 effect.
- **L24 is neutral** (+0.00). Terminal-layer attn+ffn don't move
  the needle; the already-Q6_K output projection is the bottleneck.
- **Top-5 spikes: L23, L15, L3, L14, L19.** Total singleton ΔPPL
  = −10.0 (linear-sum prediction).
- **Sparse-top5 combined recipe VALIDATED**: HARP_2B_S + Q5_K on
  {3,14,15,19,23} hits **PPL 25.21 @ ~4.64 bpw** — beats dense
  same-bpw recipes (mid-7 at 27.68, edge-3 at 27.74) and is within
  ~1.8 PPL of edge-7 (23.43) at 0.43 bpw less. 85% layer-independence
  (actual 25.21 vs predicted 23.8 → 1.4 PPL cross-layer redundancy).
- **Implication**: sparse-targeted promotion of sensitivity spikes
  dominates dense edge-promotion recipes at the same bpw budget.

Lesson re: CPU parallelism: llama-quantize ignores `-t` and pins all
cores. A running quantize is a host-wide exclusive claim. Parallel
sweeps alongside a live quantize run ~2× slow for nothing. Feedback
memory saved to .claude/…/feedback_no_cpu_parallel_sweeps.md with a
TODO (task #59) to patch llama-quantize to honor `-t`.

Pause state (to pick up):
- Per-layer sensitivity sweep: complete, committed to DATA.md.
- Sparse-top5 recipe: validated, results in DATA.md.
- t/s Pareto bench: partial. IQ2_XS, Q3_K_M, Q4_K_M, Q5_K_M done
  (results in coord/results/ts-pareto.txt, not committed because
  coord/results/ is gitignored as volatile state). TURBO_2B and
  HARP_2B_S bench interrupted mid-PPL; kill was clean.
- Coord: gpu-0 IDLE, gpu-1 IDLE, no pending queue. Kill logged to
  coord/gpu.log.
- Committed: DATA.md, COORD.md, scripts/{sens_sweep_batch.sh,
  ts_pareto_bench.sh, harp_2b_s_35b_a3b.sh}, .gitignore update.
- HARP_2B_S quant at /tmp/sens/bench/qwen35-0.8b-harp-2b-s.gguf
  (416 MB) preserved — next session can re-bench it without
  re-quantizing.

## 2026-04-18 — Stale-gguf masquerading as a Vega shader bug

While resuming the t/s Pareto bench, TURBO_2B came back with PPL ≈ 5M
on both CPU and Vulkan1 (Vega). First hypothesis: Vega wave64 shader
bug in `dequant_turbo.comp`. Spent time scoping test-backend-ops,
`supports_op` paths, and planning a multi-device dequant test.

Actual root cause: `qwen35-0.8b-turbo-2b-imat.gguf` (Apr 16 22:46 UTC)
predated submodule commit `63f7cea1d` "TURBO: fix codebook scale
convention" (Apr 17 09:13 UTC) by ~11 hours. The old gguf stores
centroids in `[-1, 1]` (max-abs-normalized); current runtime expects
`[-cent_max, cent_max]` (published convention, 2-bit cent_max=1.5104).
Result: decoded values ~1.5× too small, catastrophic reconstruction
error, PPL blows up.

The "Vega" leg was a red herring: `ggml-vulkan.cpp:15697-15700`
explicitly returns false for `MUL_MAT(GGML_TYPE_TURBO_2B)` with the
comment "TURBO_2B uses E8P lattice VQ — not yet ported to GLSL. Fall
back to CPU for now." So Vulkan never touched the dequant; both
"devices" were CPU running the same stale gguf.

Confirmed by running `test-turbo-4b-roundtrip` Test 10 — fresh
quantize→dequant with current code gives clean RMSE=0.28 rel=0.29.
Code is correct. Data was stale.

Lesson: when a quantized gguf produces garbage, first check
`git log --since=<gguf-mtime> ggml/src/ggml-turbo-kv.c` for
convention-breaking changes in the submodule. GGUF has no version
metadata for turbo centroids, so the mismatch is silent — this is a
trap future-me will fall into again unless we add a header key.

Cleanup: 11 stale TURBO/codebook ggufs deleted from `/home/llm/models/`.
K-quants and HARP_2B_S unaffected (HARP_2B_S is IQ2_S+D2, no turbo
centroids). Re-quantize pending with current tools.

## 2026-04-19 — Correction: stale-gguf was also wrong; real bug is in weighted quant

The stale-gguf lesson above is **also wrong** as a root-cause claim.
Keeping it in place because the symptom chain (wrong diagnosis chained
twice) is itself the lesson.

After deleting the 11 stale ggufs and re-quantizing with current tools,
TURBO_2B with `--imatrix` still produced PPL 3.9M. The same f16 source
quantized **without** `--imatrix` gave PPL 252 (5 chunks) / ~352
(20 chunks) — matching the PHASE23 baseline. So the bug lives in the
imatrix code path, not the gguf format, not the Vulkan shaders.

Real root cause in `llama.cpp/ggml/src/ggml-turbo-kv.c`:
`quantize_block_turbo` (bulk, no-imatrix) has a dedicated E8P lattice
branch for `bits==2` at lines 619-636 — packs two bytes per 8-element
group via `e8p_encode_16bit`. `quantize_block_turbo_weighted` (imatrix
per-block loop) had **no such branch** and fell through to scalar
Lloyd-Max index packing. Dequant always decodes bits==2 as E8P, so the
imatrix path's scalar-packed bytes came back as garbage E8P codes.
TURBO_3B/4B/5B were unaffected because scalar Lloyd-Max matched both
sides for bits>=3.

Fix: delegate `quantize_block_turbo_weighted` → `quantize_block_turbo`.
Weights were ignored anyway per the existing `(void)weights;` comment.
Submodule commit `5e5bda3cf`. Test `weighted_bulk_identity` added to
`test-turbo-4b-roundtrip` — asserts bit-identity across all four
turbo bitrates; catches this class of divergence by construction.

**Meta-lesson**: I chased two wrong hypotheses in sequence (Vega shader
bug → stale gguf) before the real cause surfaced. Each was plausible
in isolation but ruled out by a minimal repro that I should have run
earlier. Rule of thumb: after proposing a fix, run the single cheapest
repro that would distinguish "fix worked" from "symptom persists". If
symptom persists, the hypothesis is wrong — don't elaborate it, start
over. This is especially easy to miss when the "fix" step is expensive
(delete+requantize) and produces a clean-looking command-line
completion.

**Reusable diagnostic**: for TURBO_*B PPL blowups, run
`test-turbo-4b-roundtrip` Test 10 (CPU quant→dequant roundtrip) first.
If it passes, the code is correct; the bug is in data format or in a
different code path (imatrix, custom codebook). If it fails, the bug
is in the core quantize/dequant. Now that Test 11 exists, add it to
the bisection: Test 11 failure isolates to the weighted-vs-bulk
divergence.

## 2026-04-19 — t/s Pareto bench closed on qwen35-0.8b

Final 20-chunk CPU PPL on post-fix TURBO_2B: **353.2750 ± 20.65**,
matches prior 352.39 yardstick within stderr. Pareto row updated at
`coord/results/ts-pareto.txt` (volatile, gitignored). The full table
is preserved in PHASE23.md under "t/s Pareto bench — closed".

Finding worth keeping: on qwen35-0.8b at 2-bit budgets, HARP_2B_S
(IQ2_S substrate + D2 routing) sits at PPL 33.78 / 396.8 MB / 878 pp128,
vs IQ2_XS's 48.93 / 352.8 MB / 936 pp128. HARP_2B_S trades ~44 MB and
~7% pp throughput for 15 PPL of headroom. TURBO_2B is outside the
pareto frontier at 0.8B size. Whether that flips at 35B-A3B is the
open question (MoE expert-FFN weight mass + lattice VQ vs scalar
codebooks) — gated until HARP_2B_S is T4-benched there.

## 2026-04-19 — HARP_2B throughput ceiling + `vec_dot_type` discipline

Track B (AVX2 decoder) landed pp128=13 on qwen35-0.8b. IQ2_S at matched
2 bpw hits 879 on the same host. Root cause is not tunable SIMD:

1. `vec_dot_type = F32` forgoes `vpmaddubsw` (32 MAC/cycle int8 dotprod)
   for fp32 FMA (8 MAC/cycle). 4× datatype-level penalty before any
   decoder work, independent of bpw.
2. Trellis decoder is 5-10 instructions per weight with a 128-step
   serial state chain (`state[i] = f(state[i-1])`); OoO can't reorder
   past the dependency. ~50% of block-kernel time goes into the chain.

Realistic AVX2 ceiling with the existing `vec_dot_t` contract is
~200 pp128, not the 1200 originally gated. The gate was miscalibrated
against Q4_K_M (4.5 bpw, Q8_K int8 path), not IQ2_S at matched bit
budget — IQ2_S is the relevant rival.

**Reusable lesson**: any new low-bit ggml quant type should declare
`vec_dot_type = Q8_K` and emit 8-bit integer output from the decoder,
not fp32. That single contract choice is a 4× throughput multiplier
on AVX2 and compounds on AVX-512 VNNI.

**35B-A3B note**: the 68× gap on 0.8B is expected to compress 10-20×
at 35B because the model is RAM-resident (memory-bandwidth-bound, not
decoder-bound). Throughput decisions on HARP_2B are gated on 35B-A3B
quality showing a win worth the kernel investment; otherwise
consolidate on HARP_2B_S (IQ2_S substrate + D2 routing, already
879 pp128 / 33.78 PPL on 0.8B).

Path I (HARP_2B_E8) confirmed this framing from the other direction:
a 40 B block at 2 bpw cannot carry TCQ state bits between 8-D E8
emissions, so the trellis degenerates to per-group E8P at the E8P
Gaussian floor (~10% NMSE). Type retained in tree (`GGML_TYPE_HARP_2B_E8 = 53`)
but not a ship candidate at this block layout.

## 2026-04-19 — Per-layer L needs per-L LUT (Track A finding)

Track A implemented per-layer L policy for HARP_2B V=1 end-to-end
(dispatcher, codebook side tensor, loader registration, quantize
propagation). Measured PPL 143.43 at P3 sensitivity-driven policy vs
127.85 at uniform L=14 — a 15-PPL regression.

Root cause: `turbo-codebook` trains the LUT at L=14 only. Upgrading
selected layers to L=16 applies the same LUT against a wider trellis
whose state-emission distribution doesn't match the codebook's
training distribution. The calibration mismatch penalty (~+1 PPL per
upgraded layer) outweighs the trellis-state gain (~+0.2 PPL per
upgraded layer).

**Corrected mental model**: per-layer L is not a zero-cost lever. It
only pays if the LUT is co-trained per L.

**Order of operations for any future V=1 work:**
1. Per-L LUT co-training (add L=16 training pass to turbo-codebook,
   emit two LUT tensors `harp.lut.L14` and `harp.lut.L16`).
2. Per-layer L policy becomes a real lever only after (1).
3. Per-role LUTs (attn/ffn/expert) are a second-order refinement.

None of this closes the 4× PPL gap to HARP_2B_S at matched 2 bpw on
0.8B. HARP_2B V=1's structural ceiling on small dense models is real:
V=1 single-scalar emission + 128-element single-scale block cannot
match IQ2_S's 8-D lattice codebook. 35B-A3B may be different (MoE
weight-mass + memory-bound regime); that's where the question unparks.

**All Track A plumbing is correct and stays in the tree** — if V=1
ever gets reopened, the dispatcher side is done, only the LUT-training
side needs extending.

## 2026-04-19 — Strategic pivot: abandon trellis 2-bit, compare to Unsloth

The research cycle on novel 2-bit codecs for Qwen3.5 concluded with
**no novel codec that beats IQ2_S at matched bpw on the 0.8B yardstick.**
HARP_2B V=1 (127-143 PPL), HARP_2B_V3 (similar), HARP_2B_E8 (STOP at
T1 NMSE 10.15%), and TURBO_2B (353 PPL) all underperform the plain
IQ2_S encoder (33.78 PPL with our D2 routing wrapper).

### Renames
- `HARP_2B_S` → `UD_IQ2_S_QWEN35`. Ftype integer 46 preserved for
  gguf back-compat. CLI keeps `HARP_2B_S` as a deprecated alias for
  scripts.
- The new name is honest: it's *Unsloth Dynamic 2.0 applied to IQ2_S,
  with Qwen3.5 SSM carveouts*. No novel codec.

### Abandoned (code stays in-tree as artifacts; no active work)
- HARP_2B (V=1 trellis, quantlut_sym LUT, RHT encoder)
- HARP_2B_V2 (V=2 K=2)
- HARP_2B_V3 (V=2 K=4, task #51 completed as research artifact)
- HARP_2B_E8 (Path I TCQ+E8 hybrid, stopped at T1 gate)
- TURBO_2B (RHT + 4-level Lloyd-Max, parked separately prior to pivot)
- Path J MoE per-expert lattice
- HARP_2B Vulkan GLSL port

### Tasks deleted: #37, #38, #39, #40, #41, #45, #50, #55
New task #61: Benchmark UD_IQ2_S_QWEN35 vs Unsloth's Qwen3.5 GGUFs.

### Decisive next step
Download `unsloth/Qwen3.5-0.8B-GGUF` and `unsloth/Qwen3.5-35B-A3B-GGUF`,
measure their PPL + t/s at matched bpw against our UD_IQ2_S_QWEN35.
- If Unsloth wins: adopt their GGUFs, retire UD_IQ2_S_QWEN35 from the
  build.
- If we win: probably the delta is the SSM carveouts (+ possibly
  sparse-top-5 layer promotion). Document the delta; propose SSM-aware
  patch upstream to Unsloth if it's clean.

### Only surviving methodological contribution
Sparse-top-5 sensitivity-driven layer promotion (task #57 finding) —
if it helps beyond what Unsloth does, it's a separable contribution
worth publishing independently of the Qwen3.5 recipe.

## 2026-04-19 — 2-bit area abandoned after Unsloth dominance benchmark

Final benchmark on qwen35-0.8b, same wikitext-2 / 20 chunks / -t 16:

| Recipe | Size MB | PPL |
|---|---:|---:|
| Ours UD_IQ2_S_QWEN35 | 396.8 | 33.78 |
| Unsloth UD-IQ2_M | 371.9 | 28.72 |
| Unsloth UD-IQ3_XXS | 398.2 | 24.20 |
| Unsloth UD-Q2_K_XL | 417.7 | 26.26 |

Ours is strictly dominated. Unsloth already handles the gated-DeltaNet
SSM tensors (different — and better — choices than ours: Q8_0 ssm_beta,
Q5_K ssm_out, F32 conv1d) and already does per-layer sensitivity-driven
promotion as part of Dynamic 2.0. No piece of our Qwen3.5-specific
work survives the comparison as a contribution.

**Abandoned**: all 2-bit efforts (TURBO_2B, HARP_2B V=1/V=2/V=3, Path I
E8 hybrid, UD_IQ2_S_QWEN35 active use, Path J MoE lattice). Code stays
in-tree as implementation artifacts. Tasks #37-#41, #45, #50, #55
deleted; task #61 (benchmark) completed. For any Qwen3.5 2-bit ship
need, use `unsloth/Qwen3.5-*-GGUF` directly.

**Literature review completed** (PHASE23 retrospective):
- QAT is the path we didn't explore. EfficientQAT (arXiv 2407.11062,
  ACL 2025): 2-bit Llama-2-70B in 41 h, <3pp degradation. Bit-by-Bit
  (2604.07888), UPQ (2506.09104), BitDistiller, OneBit. Our project
  was PTQ-only; published 2-bit SOTA is QAT.
- VPTQ: vector PTQ at 2-bit with LUT codebooks. Similar family to
  AQLM/QuIP#. Reports 0.01-0.68 PPL improvements over prior 2-bit
  SOTA on LLaMA-2 and Mistral-7B.
- SpQR: outlier isolation + 2-bit base; we used imatrix but not
  explicit outlier isolation.
- Gated-DeltaNet 2-bit quantization specifically remains unexplored
  in public literature. Our HARP_2B S0–S6 delta-rule ablation was the
  right instinct but never ran cleanly to completion on 0.8B because
  the quality baseline HARP_2B itself was inadequate.

**Preconditions to reopen 2-bit work** (documented in PHASE23):
1. QAT pipeline (PTQ-only is saturated at the PPV ceiling).
2. Fused decode-GEMM kernel or Q8_K-emit decoder (else throughput
   dead on arrival).
3. A specific question on gated-DeltaNet (only genuinely open research
   direction we could have owned).
4. A model family Unsloth doesn't cover (currently none — their
   Dynamic 2.0 covers Qwen3.5 0.8B and 35B-A3B).

None are true today.

## 2026-04-19 — Don't rename on the way out

When abandoning a line of work, don't rename it on the way out. I
renamed `HARP_2B_S` → `UD_IQ2_S_QWEN35` earlier today after deciding
the recipe was worth naming honestly. Then we ran the Unsloth
comparison, saw strict dominance, and abandoned the whole thing.

The rename implied "still maintaining this under a different name"
when the correct signal was "dead." Extra commit noise, extra confusion
for anyone reading the history, and zero product value. The rename
was only committed to a local submodule branch (never pushed) and has
been reverted in the same branch. Net code impact: zero.

**Rule**: abandoning a ftype / feature / module is a one-line status
change, not a rename. If the old name is misleading, explain the
misleading in the retrospective, don't fix it with a rename.

All 6 HARP / TURBO 2-bit ftypes (TURBO_2B, HARP_2B, V2, V3, E8, HARP_2B_S)
are Abandoned, period. Use `unsloth/Qwen3.5-*-GGUF` for any Qwen3.5
2-bit need.

## 2026-04-23 — turbo-kv-4b.allium: 7-file distillation rolled back

Commit `b103f90` (authored by a heavily-quantized model agent) split
seven algorithm-level rules — `L2_norm`, `normalize`, `rescale`,
`RHT_forward`, `RHT_inverse`, `nearest_centroid`, `reconstruct_codebook`
— out of `turbo-kv-4b.allium` into six individual `.allium` files. The
stated intent was separation of concerns.

The experiment did not pay off. No cross-file reuse ever materialised:
the distilled rules were used by exactly one spec (`turbo-kv-4b`), and
the split introduced three concrete defects:

1. **Composite / distilled arity drift.** `QuantizeBlock`,
   `DequantizeBlock` and `VectorDot` called `normalize`,
   `reconstruct_codebook`, `RHT_forward`, `RHT_inverse` and
   `nearest_centroid` with the wrong number of arguments relative to
   the distilled rule definitions. `allium check` did not catch this
   because let-binding call sites aren't validated against rule
   signatures.
2. **Inconsistent result access.** Some composite rules accessed
   `.data` on rule results (treating them as envelopes), others used
   the result directly as a vector. Inconsistent both within and
   across files.
3. **Dead declarations.** `value BlockId { index: Integer }` was left
   behind unreferenced.

Rolled back in `93b6ec2` — all seven rules inlined back into
`turbo-kv-4b.allium`, the six standalone files deleted, the arity
drift and `.data` inconsistency fixed, four latent semantic bugs
fixed at the same time (`ReconstructionPreservesNorm` pointing at a
nonexistent `dequantize` rule and undefined `quantization_error_bound`;
`max(...)` used without `abs(...)` contrary to the scope comment;
undeclared `Vectors` / `Seeds` collections in invariants; missing
`norm >= 0` guard on `rescale`).

**Lesson:** for this repository, don't distil algorithm-level rules
into separate `.allium` files unless there is concrete cross-file
reuse or clear abstraction value. Single-file specs are the default.
Critical review is warranted on spec output generated by heavily
quantized model agents before accepting it — arity, access-pattern
consistency, and dead declarations are the usual failure modes.


## PHASE25: Optimisation cascade lessons — TURBO_KV_4B AVX2

Recording three non-obvious lessons from the PHASE25 vectorisation
session (2026-04-24). These are surprising results — not things you
can derive from reading the code or the Agner tables alone.

1. **fp32-scales repack surfaces bit-exact CPU/GPU parity for free.**
   The block layout was originally `(fp16 norm, fp16 inv_std, ...)`
   with 4 wasted bytes. Repacking to fp32 kept the same 72 B/128 elem
   footprint but made CPU and GPU produce IDENTICAL norm / inv_std
   on the same input — they had been last-bit different via
   independent fp16 rounding paths. This made test-turbo-kv-gpu-quantize
   a byte-exact gate instead of an RMSE-tolerance one, and saved
   ~84 ns/call on AVX2 quantize. When a hot path reads fp16 scales
   more than once per call, check whether the fp16 constraint is
   load-bearing before preserving it.

2. **Compiler already auto-vectorised the trivial Step-2 normalise
   loop.** Widening `for (i) rotated[i] = src[i] * inv_norm;` to
   explicit `_mm256_mul_ps` saved only ~1 ns on bench-turbo-kv-quantize.
   The real savings from this work (~24 ns) came from FUSING it
   with the RHT sign-flip in one pass over the 128-float buffer —
   eliminating a second load/store sweep. **If a loop is "obviously
   vectorisable" and the compiler has -O3 + -march, vectorising by
   hand may be a no-op; look for structural wins (pass fusion, cache
   locality) instead.**

3. **Per-call init branch bit harder than expected on a 128-byte LUT.**
   The signmask lookup table was initially protected by
   `if (init_flag) return;` at the top of every hot-path call. The
   branch predicted perfectly but still ate ~20 ns/call — measurable
   on a 330 ns baseline. Moving init to `__attribute__((constructor))`
   so the table is populated at shared-lib load time recovered the
   full expected saving. **For hot-path LUTs on functions called
   hundreds of times per token, pay the module-constructor cost
   rather than gating every call on a predicted branch.**

The ambient bench variance on this Zen 2 host is ~20–40 ns/call
(330–357 ns range when nothing changed). Minimum-of-3 or
median-of-5 is the right reading, not a single run.

**Reference:** PHASE25.md final state + `reference/agner/` for the
per-uarch projection. Landed as `2ee704b` + earlier commits; see
llama.cpp commits `4ad8efd1e` (RHT + fusion + signmask LUT) and
`6e0fb3e2f` (fp32 repack + fp64 L2_norm).

## 2026-04-26 — Host hard-hung; replaced zram with disk swap + systemd-oomd

Host (yarn.d07yx58.net, RX 6800 XT) hard-hung sometime after
2026-04-26 10:20 UTC after a ~12 h uptime. Root cause from boot-1
journal:

1. Apr 25 22:48 → Apr 26 00:10 — `llama-gguf` crashed 6× and
   systemd-coredump couldn't store any of them ("No space left on
   device" — root partition full at the time).
2. Apr 26 01:29 → 01:33 — kernel WARN-style stack traces in amdgpu
   display code (`dc_state_create_copy`,
   `update_planes_and_stream_state`, `amdgpu_dm_atomic_commit_tail`)
   coincident with **"Write-error on swap-device 253:0"**. Block
   device 253:0 is `/dev/zram0` — the compressed-RAM swap. zram had
   nothing left to compress into → the page allocator failed inside
   the display path.
3. Apr 26 01:50 → 04:32 — repeated `amdgpu: Freeing queue vital
   buffer ... queue evicted` (compute queues being torn down).
4. Apr 26 04:32 → 10:20:08 — kernel ring buffer silent. Only
   NetworkManager DHCP renewals in userspace.
5. Apr 26 10:20:08 — last journal entry. **No shutdown sequence.**
   12-minute gap before next boot. Hard hang, not a panic (a panic
   would have flushed something).

The fix landed in this session:

- **Removed zram entirely.** Compressed RAM-backed swap is a foot-gun
  on a 64 GiB box that runs llama-server / quantize / convert
  workloads — when it can't compress further it doesn't gracefully
  fall through to disk swap, it returns ENOMEM mid-allocation, which
  crashes the kernel WARN handler for amdgpu display code among
  others. Removed `/etc/systemd/zram-generator.conf` and
  `modprobe -r zram`. Sole swap is now `/dev/nvme0n1p5` (64 GiB,
  prio -2, GPT-auto-mounted).
- **Enabled `systemd-oomd`** with `SwapUsedLimit=80%`,
  `DefaultMemoryPressureLimit=60%`, `DefaultMemoryPressureDurationSec=20s`.
  Drop-ins on `user.slice` and `user-1001.slice` set
  `ManagedOOMSwap=kill` + `ManagedOOMMemoryPressure=kill` — so when
  the inference cgroup blows up, the cgroup gets killed cleanly
  before the kernel goes into a death spiral.
- **Sysctl tuning** in `/etc/sysctl.d/99-llm-host.conf`:
  `vm.min_free_kbytes=524288` (was 67584 — 66 MiB → 512 MiB),
  `vm.watermark_scale_factor=50` (was 10 — 1% → 5%). These give the
  page allocator real headroom for atomic allocations, which is
  exactly what failed in the WARN trace.
- **`MemoryMax=60G` cap on `llama-server.service`** via user-level
  drop-in (`~/.config/systemd/user/llama-server.service.d/memcap.conf`).
  Hard cgroup ceiling — the unit gets SIGKILLed at 60 GiB long before
  global pressure builds.
- **`amdgpu.ppfeaturemask=0xffff5fff`** in both `arch-lts.conf` and
  `arch-zen.conf` loader entries (was `0xffffffff`). Drops only
  GFXOFF (bit 15, 0x8000) and DCEFCLK DPM (bit 13, 0x2000); keeps
  OverDrive and GFX DCS. GFXOFF is the most-cited RDNA2 hang root
  cause and DCEFCLK DPM directly touches the display path that
  WARNed. Takes effect on next reboot. **Reboot pending.**

**Lesson:** zram is fine on memory-constrained desktops. On a server
with on-disk swap available, it's a worse failure mode — failed
compression returns mid-allocation rather than spilling to disk, and
the kernel handlers above zram aren't audited for ENOMEM-during-WARN.
Use plain on-disk swap + systemd-oomd PSI policies + cgroup
`MemoryMax` instead.

Loader entries are backed up at `*.bak.<timestamp>`. Recovery if
ppfeaturemask wedges: boot the archinstall entry
`2026-03-03_20-22-53_linux-lts.conf` (no overrides) or restore the
backup.

## 2026-04-26 — Vulkan FA-LSE: coopmat2-capable Ampere refuses LSE at supports_op

Built llama.cpp on RTX 3060 Ti host (Arch, CUDA 13.2, Vulkan 1.4.341).
`test-backend-ops -b Vulkan0 -o FLASH_ATTN_EXT` reports 4624/4624 supported
cases pass, but every `lse=1` case prints `not supported [Vulkan0]`. The
gate at `ggml/src/ggml-vulkan/ggml-vulkan.cpp:15898-15902` refuses LSE
when `device->coopmat2` is true; the 3060 Ti reports `matrix cores:
NV_coopmat2`, so the dispatcher never reaches the cm1 LSE shader on
Ampere. **TRANSFER.md (2026-04-26 snapshot) was wrong** to predict that
the 3060 Ti would exercise cm1 — coopmat2 capability is checked at the
device level and short-circuits before per-shape shader selection. To
close substep 6.5's runtime claim, port cm2 first (task #40), or add a
runtime knob to disable cm2 selection. PHASE28.md iter 32 records the
correction; substep 6.5 box stays `[ ]`.

## 2026-04-26 — cm2 LSE port aborted; dispatcher fallback to cm1 chosen instead

Attempted a per-line port of `flash_attn_cm2.comp` to add an LSE writeback
branch (per iter 28 audit: scalar epilogue gated on
`gl_LocalInvocationIndex == 0`, vec4-aligned HSV+4 dst). Build was clean
but `test-backend-ops` flagged NMSE ~46 across all 38 lse=1 cases on the
3060 Ti — large enough to be a layout/index bug, not numerical drift.
Without shader-side `printf` instrumentation the bug was not isolable in
reasonable time. **Pivoted**: reverted the cm2 shader edits and added a
dispatcher rule in `get_fa_tuning_params` that downgrades FA_COOPMAT2 to
FA_COOPMAT1 (or scalar) when `lse_mode` is set. The cm1 LSE branch is
the verified one and is now exercised on Ampere via this fallback. Lesson
for the future cm2 LSE port (if perf demands it): the cooperative tensor
store at `coopMatStoreTensorNV` is the structurally invasive piece; a
per-element scalar epilogue around `coopMatPerElementNV` mirroring the
GQA path is the right shape, but `gqa_iq1`-driven address math in the
non-GQA branch needs careful binding against `flash_attn.comp:811` and
the dst layout from `ggml_flash_attn_ext_lse` (shape
`[HSV+4, n_heads, n_queries, n_seqs]`). Add shader debug-printf
instrumentation BEFORE attempting again.

## 2026-04-26 — Ampere Vulkan turbo_kv_4b dequant regression vs Vega

While running substep 6.6's PPL gate on the RTX 3060 Ti, every config that
uses `--cache-type-k turbo_kv_4b` regresses badly: rw=0 PPL=30.59 and
rw=128 PPL=35.18 against a baseline of 17.28 (bf16/f16 KV). f16+rw=128
matches rw=0 within Δ=0.001, so the FA residual-window two-pass path
itself is correct on Ampere. The CPU-side `TURBO_KV_4B_DEBUG` validator
also flags MISMATCHes (small absolute errors, e.g. 0.08–0.21 against
ref values ~135) — same dequant disagreement that Vega doesn't surface.
Vega reference logs (`reference/ppl/results/Qwen3.5-0.8B-BF16/turbo_kv_4b.log`)
recorded turbo_kv_4b PPL ~17.66, so this is a NVIDIA-Vulkan-specific
regression, not a code change in this session. Likely a per-shape quant
kernel issue or a packing-layout mismatch surfaced by Ampere's Vulkan
driver. **Out of scope for step 6**; track separately. Investigation
hooks: try with `--device CPU` to see whether dequant agrees on CPU, and
run `test-backend-ops -b Vulkan0 -o MUL_MAT -t TURBO_KV_4B` (or whatever
the registered type test is) to bisect at the op level.

## 2026-04-26 — Qwen3.6-35B-A3B tokenizer hash registered as qwen35 (UNVERIFIED)

`convert_hf_to_gguf.py` rejected the 35B-A3B safetensors with
`NotImplementedError: BPE pre-tokenizer was not recognized` (chkhsh
`1444df51289cfa8063b96f0e62b1125440111bc79a52003ea14b6eac7016fd5f`). I
added a one-line entry mapping that hash to `res = "qwen35"` (the same
pre-tokenizer family used by Qwen3.5-9B-Instruct, which IS registered).
Conversion then completed successfully — `/opt/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-BF16.gguf`
(67 GB BF16, 753 tensors). **Tokenization parity vs the HF
`transformers` tokenizer is not yet verified.** If outputs look garbled
in inference tests, this is the first place to look — Qwen 3.6 may use
slightly different pre-tokenizer rules from 3.5. Cross-check by encoding
a few strings through both `tokenizers.AutoTokenizer.from_pretrained(
"Qwen/Qwen3.6-35B-A3B")` and llama.cpp's runtime, comparing token IDs.

## 2026-04-26 — Qwen3.6-35B-A3B BF16 GGUF smoke-verified on CPU

CPU-only `llama-cli` smoke against `/opt/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-BF16.gguf`
with prompt "The capital of France is" produced coherent English
(thinking-style reasoning prelude) — confirming the tokenizer hash
registered as `qwen35` is functionally correct at the round-trip level.
Throughput on Xeon (no GPU offload): prompt 23.8 t/s, generation 9.0 t/s.
8 GB VRAM is too small for full BF16 offload; partial `-ngl` or a
quantized GGUF (Q4_K_M ≈ 17 GB) is needed for GPU acceleration. NOTE:
`-no-cnv` is rejected by `llama-cli` ("--no-conversation is not supported
by llama-cli; please use llama-completion instead"); the run wedged in
interactive mode and produced a 1.1 GB log of empty `> ` prompts before
being killed. Future smokes: use `llama-completion` for one-shot
generation, or pipe a `/exit` command into `llama-cli` stdin.
