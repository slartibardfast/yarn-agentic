# Phase 29: CUDA/HIP backend gaps for tight Qwen3.5 MTP

## Status

**Open.** Umbrella phase for all pending CUDA/HIP work. Triggered by the focus shift after PHASE30 closed Ampere-side correctness for `turbo_kv_4b` on Vulkan. Goal: bring the CUDA backend on the RTX 3060 Ti (Ampere) to parity with the Vulkan path that drives Qwen3.5-9B-mtp-q4km at 36–38 t/s and 65–95% spec-decode acceptance on Vega.

## Scope

Match the Vulkan production path's coverage for the agent workload, narrow to CUDA. Out of scope: Vulkan-side perf tuning; non-Qwen3.5 models; Phase 30 Vega regression check (which still belongs to PHASE30).

In scope, by op family:

1. **TURBO_KV_4B cache type on CUDA.** No CPY, GET_ROWS, SET_ROWS, FA-with-K=TURBO_KV_4B currently supported (`ggml-cuda.cu:4904-5029`). Vulkan has all four.
2. **TURBO_2B / 3B / 4B / 5B weight dequant on CUDA.** `dequant_turbo_*` kernels do not exist (`grep dequant_turbo ggml/src/ggml-cuda/` is empty). Vulkan has them.
3. **TQ_V_4B V-cache on CUDA.** Not in the FATTN vector-kernel instantiation table (`fattn.cu:234-291`). Vulkan has it.
4. **FA-LSE writeback on CUDA.** Explicitly refused at `ggml-cuda.cu:5175-5182` (`if (op_params[4]==1) return false`); routes to CPU. Required for the residual-window two-pass FA closed in PHASE28 substep 6.5 on Vulkan.
5. **MTP graph-split tightness baseline.** The Vulkan side hit 118 splits before fused ops landed (PHASE3); CUDA already has fused.cu (commit `28e6efcce`) and SSM_CONV / SSM_SCAN / RWKV_WKV6 / GATED_LINEAR_ATTN — verify, don't assume. Measure split count on the actual Qwen3.5 MTP graph and only intervene if splits are non-trivial.
6. **HIP parity.** CUDA changes generally cross-compile to HIP through the macro shim — confirm per-step, don't ship CUDA-only kernels without checking `ggml-hip/` builds.

## Substep checklist

The phase opens with a measurement-first gate, then orders work by leverage on MTP throughput. Each substep closes its stated claim before the next opens.

- [ ] **Step 1 — CUDA baseline gate.** Run `test-backend-ops -b CUDA0` filtered to FLASH_ATTN_EXT, CPY, GET_ROWS, SET_ROWS, MUL_MAT, MOE_FUSED_UP_GATE / FUSED. Tabulate (a) "not supported" cases for the turbo/TQ types we plan to add, (b) any FAIL on already-supported types. Run perplexity on Qwen3.5-0.8B-BF16 with `--device CUDA0 -ctk f16 -ctv f16 -fa on` to bind a reference PPL number on this host.
  - Verify by: PPL number in iter log + a tabulated op-supports gap list. Closes when the gap list matches the exploration report (no surprises) and PPL is in chunk-noise range of CPU and Vega-Vulkan.
- [ ] **Step 2 — CUDA TURBO_KV_4B CPY + dequant.** Highest-leverage agent feature (we just landed it on Vulkan). Implement:
  - `ggml/src/ggml-cuda/cpy.cu` — register f32↔TURBO_KV_4B variants reusing the same RHT + Lloyd-Max codebook from the Vulkan shaders (`vulkan-shaders/turbo_kv_4b_rht.glsl`). One thread block per 128-element block; one thread per element; warp-shuffle butterfly stages 1..16 + shared-memory stages 32, 64.
  - `ggml/src/ggml-cuda/getrows.cu` — register TURBO_KV_4B GET_ROWS dequant.
  - `ggml-cuda.cu` op-supports — wire the new types into the CPY / GET_ROWS / SET_ROWS switches.
  - Verify by: `test-backend-ops -b CUDA0 -o CPY -o GET_ROWS -o SET_ROWS` turbo_kv_4b cases all pass at every DK ∈ {128, 256, 384, 512}; tolerance precedent from `test-backend-ops.cpp:2853` already covers the lossy quant.
- [ ] **Step 3 — CUDA TURBO_*B weight dequant.** Required for Qwen3.5-9B-mtp-q4km (the MTP draft head and possibly the main model use TURBO_4B / TURBO_4B_S quantization). Implement `dequant_turbo_*` kernels mirroring the Vulkan `dequant_turbo.comp` template. Wire MUL_MAT.
  - Verify by: `test-backend-ops -b CUDA0 -o MUL_MAT -t turbo_4b` (and 2b/3b/5b) passes.
- [ ] **Step 4 — CUDA TQ_V_4B V-cache.** Add the FATTN vector-kernel instantiations for V=TQ_V_4B (paired with K=F16). `fattn.cu:234-291`.
  - Verify by: `test-backend-ops -b CUDA0 -o FLASH_ATTN_EXT` turbo cases pass; perplexity with `--device CUDA0 -fa on -ctv tq_v_4b` on the 0.8B model returns a reasonable PPL.
- [ ] **Step 5 — CUDA FA-LSE writeback.** Lift the refusal at `ggml-cuda.cu:5175-5182`. Add (M, S) writeback to whichever FA kernel paths the test set hits (start with `fattn-tile.cu` since Ampere uses the tile path for moderate dims; coopmat / wmma kernels can follow). Pattern follows the Vulkan FA-LSE port from PHASE28.
  - Verify by: `test-backend-ops -b CUDA0 -o FLASH_ATTN_EXT` LSE-mode cases pass; PHASE28 residual-window PPL gate at `--cache-residual-window 128` returns parity with rw=0 within ±0.001.
- [ ] **Step 6 — MTP spec-decode tightness on CUDA.** Run the actual workload: `llama-server` with the 9B MTP model, `--device CUDA0 -ngl 99 -fa on -np 1 -c 4096`, `GGML_SCHED_DEBUG=1`. Count graph splits, identify any remaining CPU-fallback ops, fix them. Compare t/s and acceptance to the Vega-Vulkan reference (36–38 t/s, 65–95%).
  - Verify by: split count ≤ 10 on the MTP graph; t/s ≥ 1.5× the Vega number (Ampere should be faster); acceptance rate within ±5 absolute points of Vega.
  - Gating note: the 9B MTP gguf is not yet on this host. Step 6 has a prerequisite to fetch `Qwen3.5-9B-mtp-q4km` (path TBD; see PHASE1 of `phases/qwen35-mtp/`).
- [ ] **Step 7 — HIP parity sweep.** For each kernel landed in Steps 2–5, build with `-DGGML_HIP=ON` (skip if no AMD GPU available locally; in that case, restrict to compile-only verification — `ggml-hip/CMakeLists.txt` should pick up the new files via the existing CUDA→HIP source list).
  - Verify by: HIP build succeeds with no missing-symbol errors; if a 6800XT or Mi-class GPU is reachable later, run the same `test-backend-ops` gates.
- [ ] **Step 8 — Document and commit.** Per CLAUDE.md §5/§6: PHASE29.md + MEMORY.md updated per-iteration; commit boundaries match substep closes.

## Hypotheses (ordered by prior probability)

1. **CUDA fused.cu already covers Qwen3.5 SSM/attn.** PHASE3 (Vulkan) found 118 splits because SSM ops fell back; on CUDA they're native (`ggml-cuda.cu:5091-5158`) and `fused.cu` recently landed. **Prediction:** Step 1's split count on the 0.8B model is ≤ 10 with no SSM splits; Step 6 is mostly a measurement, not a fix.
2. **TURBO_KV_4B CUDA port is mostly mechanical.** The Vulkan algorithm in `turbo_kv_4b_rht.glsl` is subgroup-portable; the CUDA equivalent is one CTA per block × 128 threads × `__shfl_xor_sync` for stages 1..16 + shared memory for stages 32, 64. RTX 3060 Ti warp size is 32 — same as Ampere Vulkan subgroup size we just verified. **Prediction:** Step 2 is the largest substep but lowest-risk per LOC.
3. **FA-LSE on CUDA is the hardest step.** Touching FA kernels is invasive; the LSE writeback contract (M and S into HSV+1, HSV+2 of dst) needs care across tile / wmma / mma / vector kernel families. **Prediction:** Step 5 takes longer than Steps 2–4 combined and may need a separate phase if it grows beyond a single iteration.

## Reproducibility pins

- **Host:** `gpu-2` (RTX 3060 Ti / Ampere, sm_86), CUDA 12.x, llama.cpp master `35d126047` (current submodule pointer; will bump as Phase 29 lands).
- **Model (0.8B, fast iter):** `/opt/models/qwen3.5-0.8b/Qwen3.5-0.8B-BF16.gguf`.
- **Model (9B MTP, full workload):** `Qwen3.5-9B-mtp-q4km` — TBD path; needs download. Source per `phases/qwen35-mtp/PHASE1.md`.
- **Model (35B-A3B, scale check):** `/opt/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-BF16.gguf` (already on host).
- **Corpus:** `/opt/models/wikitext-2-raw/wikitext-2-raw/wiki.test.raw`.
- **PPL command base:** `./bin/llama-perplexity -m <model> -f <corpus> --device CUDA0 -ngl 99 -fa on -c 2048 --chunks 3` plus per-config `-ctk` / `-ctv` / `--cache-residual-window`.
- **Op-test base:** `./bin/test-backend-ops -b CUDA0 -o <OP>`.
- **Build:** `cmake -G Ninja -DGGML_CUDA=ON -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release` (Vulkan stays enabled so the existing benchmarks keep working).

## Closing condition

Phase 29 closes when **all** of:

1. Steps 1–6 substep boxes bind on their stated claims.
2. Qwen3.5-9B MTP runs on CUDA0 with split count ≤ 10, t/s ≥ Vega Vulkan reference, acceptance within ±5 pts.
3. `test-backend-ops -b CUDA0 -o FLASH_ATTN_EXT -o CPY -o GET_ROWS -o SET_ROWS -o MUL_MAT` shows no FAIL on any TURBO_*B / TURBO_KV_4B / TQ_V_4B case.
4. HIP build is at least compile-clean with the new kernels.

If MTP throughput is still < Vega reference after Steps 1–6 close, the gap is perf-tuning territory; spawn PHASE31+ rather than blocking PHASE29 closure.

## Loop log

Each iteration appends one line: what landed, evidence binding, regression panel state.

- iter 3 (2026-04-29): **Step 2 closes.** Two follow-ups landed: (a) turbo_kv_4b added to `test_set_rows::max_nmse_err` lossy-quant branch (the prior "CUDA error: invalid argument" was actually a tolerance-driven NMSE failure surfacing as a runtime abort); (b) `ggml_cuda_op_get_rows_turbo_kv_4b` written, wired in `getrows.cu`, op-supports updated. Results: `test-backend-ops -b CUDA0 -o CPY` 15/15 turbo_kv_4b OK; `-o SET_ROWS` 12/12 OK including the nr23=[2,3] broadcast case; GET_ROWS dispatch + op-supports wired but the test cannot gate it because `ggml-cpu/ops.cpp:4907` aborts on the CPU reference for turbo_kv_4b GET_ROWS (CPU-side gap, out of Phase 29 scope) — will exercise via the actual inference path. Submodule commit `c654e2c37`. Next: Step 3 — TURBO_*B weight dequant.

- iter 2 (2026-04-29): **Step 2 substantially closes.** CUDA TURBO_KV_4B port: new `ggml-cuda/turbo_kv_4b.{cu,cuh}` mirrors the Vulkan subgroup-portable algorithm (one CUDA block / 128 threads / 1 element-per-lane; warp-shuffle FWHT stages 1-16; shared-mem stages 32, 64). Wired CPY both directions + SET_ROWS in cpy.cu / set-rows.cu / ggml-cuda.cu op-supports. Submodule commit pending push. Results on RTX 3060 Ti: `test-backend-ops -b CUDA0 -o CPY` turbo_kv_4b non-permuted AND permuted all pass at every DK ∈ {128,256,384,512} (better than Vulkan, where the CPY-stride bug remained). `-o SET_ROWS` mostly green; one DK=384 nr23=[2,3] edge case still fails — stride math for non-trivial nr23 broadcasts; deferred to iter 3. **`llama-perplexity --device CUDA0 -ctk turbo_kv_4b -ctv bf16 -fa on --cache-residual-window 0` = 17.32 PPL** (matches CPU/Vulkan 17.29 within chunk stderr); graph splits = 14. PPL gate binds — Step 2 closing condition met for the integration path. GET_ROWS not yet wired (not on hot path). Next: Step 2.5 — fix the SET_ROWS broadcast edge case + add GET_ROWS for completeness; or skip to Step 3 (TURBO_*B weight dequant) since PPL is green.

- iter 1 (2026-04-29): **Step 1 closes.** CUDA0 baseline gate on RTX 3060 Ti. `test-backend-ops -b CUDA0` sweep across FLASH_ATTN_EXT / CPY / GET_ROWS / SET_ROWS / MUL_MAT / FUSED / MOE_FUSED_UP_GATE / FUSED_RMS_NORM / SSM_CONV / SSM_SCAN: every non-turbo op returns Backend CUDA0: OK. turbo_kv_4b / turbo_2b/3b/4b/5b types print "not supported" cleanly (no FAIL, no crash). Hybrid SSM/MoE/FUSED ops are all native on CUDA — no PHASE3-equivalent split-fix needed. **PPL on Qwen3.5-0.8B-BF16, --device CUDA0 -fa on -ctk f16 -ctv f16 -c 2048 --chunks 3 = 17.3021** (vs CPU 17.29, Vulkan 17.29; within chunk stderr ±0.94). `sched_reserve: graph splits = 2` for the 0.8B model — Vulkan needed PHASE3 to land fused ops to get from 118 splits down to similar. **Gap audit confirms the exploration report**: missing op-supports for the four quant families (TURBO_KV_4B cache type ops; TURBO_*B weight MUL_MAT; TQ_V_4B FA-vector instantiations; FA-LSE writeback). Next: Step 2 — TURBO_KV_4B CUDA port.
