# DFlash batched-pinned dispatch — canonical writeup

**Branch**: `production/2026-q2-next`
**Closed**: 2026-05-19
**Predecessors**:
- `PHASE_DFLASH_MULTISLOT.md` — multi-slot orchestrator (closed 2026-05-18).
- MEMORY.md entries `2026-05-19 — DFlash Path A foundation: F16-recast target is the new default`, `2026-05-19 — DFlash Path A CLOSED — pinned-HMMA dispatch landed`, `2026-05-19 — DFlash combine_features + inject_kv_fused — batched-pinned collapse`.
- `project_dflash_t8_closed.md` — pre-perf measurement of record (1.14 t/s, ~1% of TU102 peak).

This file is the single doc to read for the final state; predecessors are kept for historical depth.

## Result

All matmul work in the DFlash drafter pipeline — 47 GEMMs per cycle (35 drafter forward projections + 1 lm_head + 10 inject_kv K/V projections × 5 layers + 1 combine_features FC) — now dispatches through a single canonical entry point `dflash_gemm_npc` (forwards to `ggml_cuda_mul_mat_f16_pinned`, HMMA m16n8k16, NPC-by-construction across batch composition). The five non-matmul kernels that consume those GEMM outputs (`q_norm_rope`, `k_norm_rope`, `cache_write_kv`, `silu_mul`, `residual_add`, plus new post-process kernels for combine and inject) carry fp32 internally and fuse the F32→F16 cast at their natural store boundaries.

The pre-recast BF16 lm_head was converted F16 via `scripts/recast_bf16_to_fp16.py` (T1, Band-A absmax = 0.36; FP16 has 10 mantissa bits vs BF16's 7 → mantissa-improving cast). The recast tool's F16 `raw_passthrough` writer bug (halving the last dim) was caught and fixed in the same pass.

**End-to-end perf** (test-dflash-np-multislot, F16 target, locked clocks 1455 MHz, dual Quadro RTX 6000):

| State | NP=1 ms/cycle | NP=1 tok/s | NP=8 ms/cycle | NP=8 aggregate tok/s |
|---|---:|---:|---:|---:|
| Pre-Path-A (scalar fp32) | 1830 | 2.2 | 3138 | 10.2 |
| + Path A drafter forward + lm_head | 211 | 19.0 | 1499 | 21.3 |
| + combine_features batched-pinned | 105 | 38.1 | 654 | 48.9 |
| + inject_kv batched-pinned (final) | **35.7** | **112.2** | **97.5** | **328.3** |

Cumulative from pre-Path-A baseline:
- NP=1: **51.3× per-cycle**, 51× throughput.
- NP=8: **32.2× per-cycle**, 32× aggregate throughput.

**NPC byte-identity** preserved at every commit, every gate.

## How to verify

```bash
cd /home/llm/yarn-agentic
sudo bash scripts/gpu-clocks.sh lock              # 1455 MHz both GPUs
bash scripts/verify-production-determinism.sh     # NPC NP={1,2,4,8} multi-GPU
T=/opt/models/recast-out/qwen3.6-27b-V-F1.T1.lm_head-f16.gguf
D=/opt/models/qwen36-27b-dflash/qwen36-27b-dflash-f16.gguf
cd ik_llama.cpp/build
./bin/test-dflash-drafter-lm-head                                    # 5/5 NMSE/cos
./bin/test-dflash-combine-features                                   # 8/8 NMSE/cos
./bin/test-dflash-inject-fused                                       # 8/8 NMSE/cos
DFLASH_TARGET_GGUF=$T ./bin/test-dflash-closure                      # 8/8 vs vLLM
./bin/test-dflash-np-invariance                                      # 4 seeds × NP{1,2,4,8}
LLAMA_TEST_TARGET=$T LLAMA_TEST_DRAFTER=$D ./bin/test-dflash-np-multislot   # slot-0 NP{1,2,4,8}
```

The F16-recast target `qwen3.6-27b-V-F1.T1.lm_head-f16.gguf` is the new canonical DFlash target. Pointing at the pre-recast BF16 target fires the loader's defensive `output.weight->type != GGML_TYPE_F16` guard at `src/llama-dflash.cpp` with a hint at `scripts/recast_bf16_to_fp16.py`.

## What ships

| Artifact | Path | Role |
|---|---|---|
| Canonical GEMM launcher | `ik_llama.cpp/ggml/src/ggml-cuda/dflash/dflash-gemm.{cu,cuh}` | `dflash_gemm_npc(weight, act, dst_f32, K, N_cols, n_rows, stream)` thin forwarder to `ggml_cuda_mul_mat_f16_pinned` |
| Drafter forward (rewritten) | `ik_llama.cpp/ggml/src/ggml-cuda/dflash/dflash-drafter-forward.cu` | 35 GEMM calls + 5 F32-input downstream kernel overloads |
| lm_head (rewritten) | `ik_llama.cpp/ggml/src/ggml-cuda/dflash/dflash-drafter-lm-head.cu` | One pinned call, ~30 LOC; old scalar fp32 GEMV retired |
| combine_features (rewritten) | `ik_llama.cpp/ggml/src/ggml-cuda/dflash/dflash-combine-features.cu` | One pinned call + `combine_features_norm_kernel`; scalar fp32 FC retired |
| inject_kv (rewritten) | `ik_llama.cpp/ggml/src/ggml-cuda/dflash/dflash-inject-kv.cu` | 2 pinned calls/layer + `inject_kv_postprocess_kernel`; scalar fp32 K/V proj retired |
| Recast tool fix | `scripts/recast_bf16_to_fp16.py` | F16 `raw_passthrough` writer dtype fix (halved last-dim bug) |
| Default DFlash target | `/opt/models/recast-out/qwen3.6-27b-V-F1.T1.lm_head-f16.gguf` | F16 lm_head; new canonical target |
| Spec amendments | `specs/dflash/kernel-design.md` §6.1.A, §6.2.A, §6.6.A | Pinned canonical, cuBLAS forbidden in `ggml-cuda/dflash/*.cu` |
| Test gate tightening | `tests/dflash-speculative/test-dflash-{drafter-lm-head,combine-features,inject-fused}.cpp` | Strict ULP gates → NMSE ≤ 1e-5 AND cos ≥ 0.99999 (HMMA fragment tree differs from serial-fp32 reference) |

## The four commit pairs (all default-on)

All on submodule `production/2026-q2-next`. Each commit on the submodule is paired with a parent-repo commit bumping the submodule pointer and/or amending the spec/MEMORY.md.

| # | What | Submodule commit | Parent commit | Pinning |
|---|---|---|---|---|
| 1 | Recast BF16 lm_head → F16 + tool fix + DFlash loader F16 plumbing | `37f28896` | `88b92e7`, `ad78c0b` | closure 8/8 + NPC NP{1,2,4,8} unchanged at the recast (perf-neutral foundation) |
| 2 | Path A: drafter forward 35 GEMMs + lm_head → `dflash_gemm_npc` | `03606b66`, `581e0734` | `b9032d4`, `2427841` | drafter-forward GEMM portion 3132 → 17 ms (184×); end-to-end NP=1 8.7×, NP=8 2.1× |
| 3 | combine_features → batched pinned + norm sub-kernel | `8b2a9843` | (bundled in #4) | combine kernel 108 → ~3 ms (36×); end-to-end NP=1 2.0×, NP=8 2.3× |
| 4 | inject_kv_fused → batched pinned + postprocess sub-kernel | `020eba3d` | `6aa98f7` (spec), `7697392` (bump + MEMORY) | inject kernel 70 → ~5 ms (14×); end-to-end NP=1 2.9×, NP=8 6.7× |

## Final nsys decomposition (closure test, 8 prompts × 1 cycle each)

| Kernel | Total | % GPU | Instances/cycle |
|---|---:|---:|---:|
| `mul_mat_f16_pinned_kernel_wmma` | 249 ms | **92.9%** | 47 (35 drafter + 1 lm_head + 10 inject + 1 combine) |
| `q_norm_rope_kernel` | 7 ms | 2.6% | 5 |
| `attention_kernel` (drafter, fp32 SWA/full) | 6.6 ms | 2.5% | 5 |
| `k_norm_rope_kernel` | 1.9 ms | 0.7% | 5 |
| `silu_mul_kernel` | 1.1 ms | 0.4% | 5 |
| `rmsnorm_kernel` | 1.1 ms | 0.4% | 11 |
| `residual_add_kernel` | 0.7 ms | 0.3% | 10 |
| `inject_kv_postprocess_kernel` | 0.5 ms | 0.2% | 5 |
| `combine_features_norm_kernel` | 0.1 ms | ~0% | 1 |
| `cache_write_kv_kernel` | 0.1 ms | ~0% | 5 |
| `select_output_kernel` | 0.04 ms | ~0% | 1 |

Pipeline is **matmul-dominated** — the optimal terminal structure on TU102 without rewriting pinned itself.

## Why pinned, not cuBLAS

The original Path A plan (and task #59) was titled "cuBLAS dispatch". Exploration surfaced `ggml_cuda_mul_mat_f16_pinned` — an already-baked, NPC-by-construction HMMA m16n8k16 kernel — making cuBLAS unnecessary:

- **Shape regime is tall-skinny across the whole pipeline** (M ∈ [4, 88], N up to 248320, K up to 25600) → memory-bound, not compute-bound. cuBLAS HGEMM's tall-skinny algo selection switches with shape and is NPC-hostile without an ALGO0 pin and a per-shape determinism micro-test (task #54 was that micro-test).
- **Pinned is byte-identity-by-construction** (single CTA per output cell, fixed compile-time K-loop, fp32 accumulator inside HMMA fragments, no Split-K, no atomics, no shape-dependent algo selection) → no NPC re-validation needed at the matmul layer.
- **Zero new code at the matmul kernel level** — only a thin forwarder TU `dflash-gemm.{cu,cuh}`.

`cublasGemmEx`, `cublasHgemm`, and any handle from `ctx.cublas_handle(...)` are **forbidden** in `ggml/src/ggml-cuda/dflash/*.cu` (spec §6.1.A, §6.2.A, §6.6.A). Task #54 stays parked-exploratory in case pinned underperforms its roofline on Gate/Up/Down (M=40 N=17408) and we want cuBLAS as a backup — none observed.

## Latent issues caught and fixed

1. **Recast tool F16 raw_passthrough halved last-dim** (`scripts/recast_bf16_to_fp16.py`). gguf-py's writer divides `tensor_shape[-1]` by 2 when the passed dtype is `np.uint8`. The original code hardcoded `np.uint8` because BF16/quantized tensors arrive as uint8 byte-buffers — but F16 tensors arrive as `numpy.float16` with element-shape already correct. Fix: pass `t.data.dtype` rather than hardcoded `np.uint8`. Invisible on the 0.8B canary (zero F16 source tensors); exposed by the 27B target's 98 F16 passthroughs that would have silently lost half their data.

2. **Stale `dflash-target-shared-loader.h` BF16 signature** caught at Path A build. The lm_head migration touched the kernel + the production loader but missed the test loader header. Fixed in the same submodule commit (`37f28896`).

3. **Stale `§6.2 dflash_inject_kv_fused` header** in `specs/dflash/kernel-design.md`. An earlier `§6.1.A` append at parent commit `b36f5fe` inadvertently clobbered the `### 6.2` header line. Restored in `6aa98f7` alongside the §6.2.A / §6.6.A amendments.

4. **Stale `test-dflash-drafter-lm-head.cpp` BF16 references** caught at Path A build (same kind of miss as #2). Migrated to F16 in the same submodule commit (`03606b66`).

## Test-gate philosophy revision (recorded)

The pre-Path-A scalar fp32 kernels matched a serial fp32 scalar reference byte-perfect; their unit tests gated on byte-identity or 1-ULP fp16 distance. **Pinned HMMA fragment reduction trees are a different reduction order from a serial K-loop**, so byte-identity is unachievable in principle; ULP-distance is mis-leading near zero magnitude (a +0 / -0 pair shows as 32768 ULP in fp16 bit-pattern subtraction).

Revised gate for all three rewritten kernels: **byte-identical OR NMSE ≤ 1e-5 AND cos ≥ 0.99999**. Same precedent as the post-S59 lm_head test. Observed: NMSE in the 1e-9 to 1e-12 range across all sweeps, cos = 1.0 to 6 decimals.

The production correctness gate is `test-dflash-closure` (argmax-equivalent vs vLLM on 8 prompts × 4 mask positions). That gate is unchanged and passed at every commit.

## Diagnostic methodology recorded

- **nsys decomposes; closure gates the answer.** Path A's drafter-forward + lm_head swap exposed two next-tier bottlenecks (combine_features 51.6%, inject_kv 33.6%) only after the nsys re-profile post-Path-A. Without that profile, the natural next target would have been "drafter forward needs more optimization" — wrong; drafter forward had already collapsed to 13.7% of GPU. The diagnostic rigor of measure-then-decide caught the actual top-of-list.
- **Layout coincidence is gold.** Both combine_features and inject_kv had their inputs already laid out as `[M, K]` row-major where M=N_slots*MAL_anchors and K=L_src*D_d (combine) or K=D_d (inject) — zero pack kernels required. Reading the existing kernel's pointer arithmetic before designing the rewrite saved an entire transpose-kernel iteration on each.
- **Test-gate revision must accompany kernel-dispatch revision.** When the math switches from serial-fp32 to HMMA-fragment-reduction, byte-identity-strict unit tests become noise generators that mask real wins. The production binding (closure vs vLLM) is the meaningful gate; unit tests downgrade to NMSE/cos with the closure as final arbiter.
- **The bottleneck cleanly moves.** Each rewrite shifted the dominant kernel:
  - pre-Path-A: drafter forward (~98% scalar GEMM)
  - post-Path-A: lm_head (89%)
  - post-lm_head: combine_features (51.6%)
  - post-combine: inject_kv (33.6%)
  - post-inject: pinned WMMA matmul itself (92.9%)
  Each move was predictable from the profile decomposition and accomplished via the same launcher `dflash_gemm_npc`.

## What's still open (non-blocking)

- **DFlash drafter `attention_kernel`** — currently 2.5% of GPU, fp32 SWA/full attention. Optimization possible (HMMA + softmax in fragments) but at 2.5% the win is bounded and the determinism story for fp32-softmax-via-tensor-cores is non-trivial. Out of scope until pinned itself plateaus.
- **Pinned WMMA tuning** — at 92.9% of GPU, further perf is in pinned itself: block geometry, K-tile prefetching, dual-issue ILP. The existing `mul-mat-f16-pinned.cu` was tuned for a different shape regime; tall-skinny DFlash shapes may have headroom to ~2-3× via a DFlash-specialized variant. Optional follow-on workstream.
- **Multi-slot DFlash server profile** — no `profiles/qwen36-27b-x*-dflash.sh` exists yet. Adding one is a simple config copy; intentionally left out because production live-serving stays on `qwen36-27b-x1-mtp.sh` until the new perf wins are exercised in a soak test.
- **`scripts/verify-production-determinism.sh` hardcodes the pre-recast target name.** The full multi-GPU verify still passes (graph isn't DFlash-specific), but the script will need a follow-up to update its `$MODEL` to the F16-recast target if DFlash is added to the canonical determinism harness.

## Companion memory entries

- MEMORY.md `2026-05-19 — DFlash Path A foundation: F16-recast target is the new default` — recast landing + tool fix.
- MEMORY.md `2026-05-19 — DFlash Path A CLOSED — pinned-HMMA dispatch landed` — drafter forward + lm_head.
- MEMORY.md `2026-05-19 — DFlash combine_features + inject_kv_fused — batched-pinned collapse` — this milestone.
- `feedback_bake_measurement_env_gates` — confirmed: no `LLAMA_*_ENABLE` flags added at any step; pinned dispatch is baked or reverted.
- `feedback_determinism_must_co_optimize_perf` — confirmed: NPC byte-identity preserved at every commit alongside the perf collapse.
- `feedback_no_host_concerns_in_code` — confirmed: no Phase/Step nomenclature in kernel names, file names, or branch names (planning docs use it; code does not).
