---
name: Vulkan mul_mat_vec BI root cause was MMVQ routing, not NUM_COLS pipelines
description: Batch-invariance failure at K≥2048 on RDNA2 turned out to be n=1 routing to the integer-dot MMVQ shader while n=N fell back to dequant-mmv. The NUM_COLS spec-const divergence theory was a red herring.
type: project
originSessionId: e06c78f3-43de-46c4-b7cf-81ea2dbd7d8f
---
Phase 1 landed 2026-04-20. The long-held theory that ACO produced divergent ISA across NUM_COLS spec-const pipelines was wrong for this specific failure. The actual cause:

`ggml_vk_should_use_mmvq` in ggml-vulkan.cpp gates integer-dot MMVQ on `ne11==1 && ne10>=2048 && AMD`. The BI test runs n=1 (ne11=1) and n=N (ne11=N>1) back-to-back; n=1 routed to MMVQ, n=N fell back to the dequant mul_mat_vec. Two different algorithms → ~5 % magnitude delta (one whole q8_0 block worth of accumulation off) = the K=2048 BI cliff.

**Why:** MMVQ and dequant-mmv apply the same scale/sum math but in different ISA / different reduction order. Also: MMVQ requires a q8_1 y-quantization pre-pass which introduces its own quantization rounding independent of the fp32 dequant-mmv path.

**How to apply:** When investigating "same shader can't possibly give different output" on Vulkan mul_mat_vec, first check if the two batch sizes route to different shaders entirely (MMVQ vs dequant vs p021 vs nc) at the `ggml_vk_mul_mat` routing gate. Only once both paths hit the same shader does the NUM_COLS / spec-const theory matter.

Phase 1 fix shipped both: (a) NUM_COLS pipeline collapse (defensive — all 8 pipelines now compile identically) and (b) MMVQ disabled in source pending Phase 3 (extend MMVQ to ne11>1). With both, BI_MUL_MAT 42/42 byte-identical, no env flags.

**2026-04-20 follow-up commit f29ef6cd**: same routing-split pattern held for MUL_MAT_ID (mat-vec-id at n=1 vs mat-mat-id at n>1) and FLASH_ATTN (split_k + small_rows + GQA gqa_ratio remap all fire only at n=1). Fix: route everything through the n>1 path (mat-mat-id for MUL_MAT_ID; split_k=1, small_rows=false, gqa_ratio remap disabled for FA). Full BI suite now 101/101 byte-identical, `GGML_VK_FORCE_BATCH_INVARIANT` env flag deleted entirely. Perf cost of forcing n=1 onto the n>1 path is not yet measured.

**Model-level validation**: `test-35b-pos-i-sequential-equivalence` on qwen35-0.8b-q4-k-m.gguf Vulkan0 shows 0/64 mismatches at pairs=8 batch=8. The argmax of logits_ith(i) matches sequential-batch-1 decode through the same prefix, at every position, batch=1..8. No env flags set. End-to-end batch-invariance confirmed.

**35B-A3B validation (2026-04-20, commit 716718e7)**: qwen3.5-35B-A3B-MTP-Dynamic.gguf CPU-spilled on 16 GB 6800 XT via `--fit`. 0/64 mismatches pairs=8 batch=8. All four historic-regression tests pass byte-identical over 20 steps (`test-35b-trajectory-drift`, `test-35b-full-accept-drift`, `test-35b-server-flow-drift`, `test-35b-batch-invariance-sweep`). No env flags, no `-no-fmoe` / `-no-fug`. Sixth and seventh routing splits were in `llama-build-context.cpp` — fused MoE up-gate (`can_use_fmoe && n_tokens > 1`) and dense fused up-gate (`cur->ne[1] > 1`) both only fire above a batch-size threshold and produce byte-different output from the unfused path. Gated off via `if (false)` until the fused kernels can be rewired to produce byte-identical output to the unfused path at every n_tokens. Perf: 0.8B tg128 unchanged; 35B-A3B tg64 1.75 t/s (CPU-spilled); 35B pp loss from disabled fused kernels not yet measured.

**Phase 3 findings (2026-04-20, commits 8e2e1281 → 3e3bb0f9)**:

Initial Phase 3 pass (8e2e1281) concluded "no perf to recover" based on 0.8B hybrid measurement. That conclusion was wrong — 0.8B is delta-net-dominated and doesn't exercise the MoE fused path at all. Re-measured on the right targets:

- **Fused MoE up-gate: +416% tg on 35B-A3B** (tg64 1.75 → 9.03 t/s, pp512 33 t/s). Unfused MoE dispatches tiny kernels per expert per token and is catastrophic at n=1. Extended the BI ops test to 35B-like shape (K=5120, M=2560, n_exp=128, n_used=8) — shader is byte-identical at N=2,4. BI suite now 105/105. Re-enabled for all n in commit 3e3bb0f9.
- **Dense fused up-gate on 4B Q8_0: -12% tg, +8% pp** with fused on. Shader is BI (0/64 on 0.8B) but the BM=64/BN=64 tile wastes its B-tile at n=1. Kept off for all n — typical interactive workloads are tg-dominated.
- **MMVQ re-enable on 4B Q8_0: -1.1% tg, 0% pp.** The q8_1 quant pre-pass + VGPR overhead of NUM_COLS=8 specialization eats the integer-dot speedup on RDNA2.
- **FA split_k / small_rows / gqa_ratio remap: 0% recovery** on 0.8B and 4B tg shapes. KV reads dominate; tile shape is marginal.

**MoE CPU-spill caveat**: on 35B under `--fit` where experts land on CPU (16 GB GPU too small), ggml-backend scheduler picks different CPU/GPU splits per batch size. fused_moe-on regresses pos-i 0→8/16 and test-35b-full-accept-drift fails step 4 in that configuration. The Vulkan shader itself IS BI (proven by 105/105 BI ops at 35B scale); the drift is scheduler-level routing. GPU-resident deployments (≥26 GiB single GPU per `project_mtp_ir_status`) get correct output AND the 5× perf win. 16 GB dev boxes doing CPU-spilled 35B can opt out with `-no-fmoe` if they need BI for speculative decoding on this hardware specifically.

Final state: mul_mat_vec collapse + MMVQ off + MUL_MAT_ID uniform + FA uniform + fused MoE on + dense fused off. Commits on origin/main: 055210c6, f29ef6cd, 716718e7, 3e3bb0f9, 6cf65bc1 (8e2e1281 superseded).

**P1 diagnostic attempt (2026-04-21)**: wrote `tests/test-35b-layer-drift.cpp` that uses `ggml_backend_sched_set_eval_callback` to capture pos-0 slices of every named intermediate tensor at N=1 and N=2 and diff them. Ran with `fused_moe on` and with `-no-fmoe` for comparison. Both configs show the SAME set of divergent tensors (delta-net state, conv state, DELTA_NET output, SSM_CONV output) at the SAME indices with NEAR-IDENTICAL magnitudes — including in the configuration that passes full-model BI. Conclusion: the drift source that makes `test-35b-full-accept-drift` fail step 4 with fmoe on but pass with fmoe off is NOT visible at the tensor-value layer. It must be one of: (a) in a tensor whose "pos-0" slice my heuristic (first n/2 elements) misidentifies, (b) a scheduler-level decision (op placement, allocator reuse) that doesn't produce tensor-value differences in captured intermediates but alters final logits through a more subtle path, (c) accumulation across decode steps where step-N state is fed back to step-N+1 through ggml-backend temporaries that my callback doesn't capture. Further investigation would need per-node logit argmax comparison at each decode step, not per-tensor capture. Tool stays in tree as a regression anchor for future work.

Separate discovery: the NoContraction injection in `vulkan-shaders-gen.cpp` had NEVER been running — its relative path resolved against the CMake build cwd (`build-vk/ggml/src`) not the source tree. Every prior "ACO ignores NoContraction" conclusion was a test of no-op code. With the path fixed and 20 decorations landing per SPV, the BI result is the same as without — so ACO genuinely does ignore NoContraction for this class of divergence, but now we actually know that (not just assumed it).
