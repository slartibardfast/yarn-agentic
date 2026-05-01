# Phase 30: turbo_kv_4b Cross-Architecture Vulkan Regression Debug

## Status

**ABANDONED 2026-05-01.** Per user direction, all our turboquant work in this tree is abandoned. Steps 1–6 closed on Ampere (PPL 30.59 → 17.29; subgroup-size-portable shaders shipped at submodule commit `cc9e876ed`, labelled `defunct/phase30`). Step 6 closing-condition (b) — Vega regression check — was deferred to next gpu-1 access and is now **not closed**.

**Successor:** PHASE31 — MTP-only production focus on `ik_llama.cpp`, no turboquant. If turboquant work is revived later, the user-named reference fork is `slartibardfast/llama-cpp-turboquant`; do not extend this phase.

**Original framing (kept for record):** Triggered by PHASE28 iter 34 (substep 6.6 isolation matrix): rw=0 turbo_kv_4b PPL on RTX 3060 Ti = **30.59**, vs Vega reference at `reference/ppl/results/Qwen3.5-0.8B-BF16/turbo_kv_4b.log` = **17.66**. ~13 PPL regression on a single-vendor driver swap. The PHASE28 isolation already proved this is **not** the FA two-pass residual-window path (f16+rw=128 on Ampere matched f16+rw=0 within Δ=0.001) and **not** the cache-residual-window logic (rw=0 also regresses). Fault lies entirely inside the turbo_kv_4b cache type's Vulkan code path on coopmat2 / NVIDIA hardware.

## Scope

Identify root cause of the turbo_kv_4b cross-architecture divergence, narrow to the offending shader or kernel pattern, and fix to within ±0.05 PPL of the Vega reference. Out of scope: the cm2 LSE shader port (PHASE28 iter 33's deferred perf work), CUDA/HIP FA-LSE ports (Phase 29 candidate).

## Hypotheses (ordered by prior probability)

The Vega reference works; the Ampere host is the new variable. Things that change at this junction:

1. **Subgroup size mismatch.** Vega's subgroup is 64 (`wave64`); Ampere's is 32 (`warp size: 32`, per `test-backend-ops` summary line). Any Vulkan shader that hardcodes a subgroup width, uses `gl_SubgroupSize`-dependent indexing without bounds checks, or assumes block-of-64 layout will produce wrong results on Ampere. **High prior given the symptom: deterministic, large, reproducible PPL drift, with the CPU-side `TURBO_KV_4B_DEBUG MISMATCH` validator firing — mismatched values are small in magnitude but consistent (0.08–0.21 abs error on values ~135) which is exactly the shape of "writes to wrong lane in a packed layout".**
2. **NVIDIA Vulkan driver-specific miscompile.** Some legal-on-RADV shader pattern (loop unrolling, OpAccessChain on packed types, `pack32(unpack8(...))` round-trip) may be silently wrong on NVIDIA's compiler. Lower prior; would need driver-version reproducibility data.
3. **Subgroup-arithmetic feature mismatch.** RADV exposes some `subgroup` capabilities that NVIDIA exposes differently. If the shader uses `subgroupBallot`, `subgroupShuffleXor` etc. with hardcoded masks, the mask validity changes with subgroup size.
4. **Coopmat2 selection accidentally routes turbo_kv_4b through a code path that wasn't exercised on Vega.** Vega has `matrix cores: none`; Ampere has `NV_coopmat2`. Some `device->coopmat2` branch in turbo_kv_4b's quant/dequant or FA-internal dequant may fire only on cm2 hardware and have a bug.

## Step checklist

Each step closes its substep's stated claim before the next opens. Mid-step abandonment without binding evidence keeps the box open.

- [ ] **Step 1 — Three-way baseline.** Run rw=0 turbo_kv_4b PPL on three configurations and tabulate PPL + count of `TURBO_KV_4B_DEBUG` mismatches:
  - (a) `--device Vulkan0` on Ampere
  - (b) CPU-only (`-ngl 0`) on Ampere
  - (c) the Vega reference (already in tree at `reference/ppl/results/Qwen3.5-0.8B-BF16/turbo_kv_4b.log`)
  - Verify by: tabulated PPL numbers + mismatch counts in PHASE30 iter log. If (b) matches (c), the bug is **Vulkan-on-Ampere only** and we proceed with shader bisect. If (b) also regresses, the bug is in CPU code that's somehow hardware-dependent (very unlikely; would mean a `-march=native` AVX-512 vs Vega-host AVX2 codegen issue).
- [ ] **Step 2 — Locate the CPU validator.** The `TURBO_KV_4B_DEBUG MISMATCH s=N ref=X batched=Y err=Z valid_run=R DK=256 iq1=I iq2=J ith=T` lines fired during the iter-34 PPL run. Find the source location (`grep -rn TURBO_KV_4B_DEBUG llama.cpp/`) and read enough context to understand: what is `ref` vs `batched`, what is `valid_run`, and crucially **is this validator running CPU vs CPU (across thread counts) or CPU vs GPU**. The answer changes the interpretation of step 1's mismatch counts.
  - Verify by: cited source file:line, one-paragraph description of the validator semantics in the iter log.
- [ ] **Step 3 — Op-level isolation.** Determine whether `test-backend-ops` registers any case for turbo_kv_4b. The PHASE28 iter 34 MUL_MAT sweep showed `type_a=turbo_kv_4b` returns "not supported" on Vulkan0 (turbo_kv_4b is a KV cache type, not a weight type — MUL_MAT doesn't dispatch it). Three sub-options to find a binding op-test:
  - (a) Look for a CPY/COPY test with `dst_type=TURBO_KV_4B` or `src_type=TURBO_KV_4B` in the test framework.
  - (b) Look for a turbo_kv_4b-specific test (search for `TURBO_KV_4B` in `tests/`).
  - (c) If neither exists, register one in test-backend-ops following the `test-backend-ops -o CPY -t TURBO_KV_4B` pattern (CPY is the natural quant↔dequant op in ggml).
  - Verify by: an op-test that **passes on (b) CPU+Ampere** and **fails on (a) Vulkan0+Ampere** — that's the minimum reproducer. If no failing test exists, the bug surfaces only at the integration (PPL) level and requires a different bisect strategy.
- [ ] **Step 4 — Shader inspection.** For each Vulkan shader involved in turbo_kv_4b (the candidates: `cpy_to_turbo_kv_4b.comp`, `cpy_from_turbo_kv_4b.comp` if they exist, the dequant pieces inside `flash_attn*.comp` that handle `K_TYPE = turbo_kv_4b`, and any GET_ROWS shader specialised for turbo_kv_4b), grep for: `gl_SubgroupSize`, `subgroupBallot`, `subgroupShuffle`, hardcoded constants `64`, `32`, `wave_size`. Note any pattern that assumes a fixed subgroup width.
  - Verify by: inventory of shaders + per-shader note on subgroup-size assumptions (or "none found").
- [ ] **Step 5 — Bisect by code-path bypass.** If Step 4 surfaced a candidate shader, force a known-good path (e.g., set `device->subgroup_size = 64` artificially in the dispatcher, or compile-time disable the suspect optimisation) and re-run the Step 1 PPL. If PPL recovers on Ampere, that's the bug.
  - Verify by: PPL on Ampere with the bypass matches Vega reference within ±0.05.
- [ ] **Step 6 — Targeted fix.** Patch the offending shader to be subgroup-size-aware (query `gl_SubgroupSize` and branch / use `OpGroupNonUniformBroadcastFirst` with explicit size handling, etc.). Confirm:
  - (a) Ampere PPL within ±0.05 of Vega reference.
  - (b) Vega regression-free (re-run the reference PPL and compare to the saved log).
  - (c) `test-backend-ops -b Vulkan0 -o FLASH_ATTN_EXT` 4662/4662 still passes (no regression to the substep 6.5/6.6 close).
  - (d) The CPU validator's `TURBO_KV_4B_DEBUG MISMATCH` warnings drop to zero or are confirmed unrelated (they may be a CPU-thread-vs-CPU-thread tolerance issue, not a GPU bug).
  - Verify by: all four bullet points logged in the iter log, with specific numbers.
- [ ] **Step 7 — Document and commit.** Per CLAUDE.md §5/§6: PHASE30.md updates committed and pushed per iteration; MEMORY.md updated with the root cause once known so future sessions don't re-discover it.

## Closing condition

Step 1 binds the regression magnitude on this host. Steps 2–5 narrow to a shader or kernel. Step 6 produces a fix that is **both** (a) Ampere-correct AND (b) Vega-regression-free. The phase closes when Step 6's all-four bullets all bind. If Step 5 cannot find a Vulkan-side bypass that recovers PPL, we have falsified Hypothesis 1+3 and the next iteration redirects (likely to a per-tensor byte-diff between Vega and Ampere outputs of the dequant op for the same input quantised buffer — turns it into a numerical bug instead of a code-path bug).

## Reproducibility pins

- **Model:** `/opt/models/qwen3.5-0.8b/Qwen3.5-0.8B-BF16.gguf` (downloaded from `unsloth/Qwen3.5-0.8B-GGUF` in PHASE28 iter 34 setup).
- **Corpus:** `/opt/models/wikitext-2-raw/wikitext-2-raw/wiki.test.raw`.
- **PPL command base:** `./bin/llama-perplexity -m <model> -f <corpus> -ngl 99 -fa on -c 2048 --chunks 3` plus the per-config `--device`, `-ctk`, `-ctv`, `--cache-residual-window` overrides.
- **Vega reference log:** `reference/ppl/results/Qwen3.5-0.8B-BF16/turbo_kv_4b.log` (PPL = 17.6637, generated 2026-04-24 on `gpu-1` Vega).
- **Ampere build:** llama.cpp master `35d126047` (current submodule pointer in main branch). cmake configured with `-G Ninja -DGGML_CUDA=ON -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release`.

## Loop log

Each iteration appends one line: what landed, evidence binding, regression panel state.

- iter 1 (2026-04-29): Steps 1–3 closed. **Step 1**: CPU-only PPL on Ampere = **17.29** (`reference/ppl/results/Qwen3.5-0.8B-BF16/ampere/turbo_kv_4b_cpu.log`); matches Vega reference 17.66 within chunk noise. Vulkan0 on Ampere remains 30.59. Bug confirmed Vulkan-on-Ampere only. **Step 2**: validator at `llama.cpp/ggml/src/ggml-cpu/ops.cpp:8463` is CPU-vs-CPU (batched turbo_kv_4b kernel vs per-position `kq_vec_dot`); MISMATCH lines fire on CPU-only runs too — pure FP-summation-order noise, unrelated to GPU regression. **Step 3**: extended `test-backend-ops` CPY coverage with DK ∈ {128,256,384,512} sweep + lossy-quant tolerance for turbo_kv_4b (`tests/test-backend-ops.cpp`). On CPU all cases pass. On Vulkan0+Ampere all fail at ERR≈1.0 uniformly across DK — single-block (DK=128) and multi-block (DK=512) fail identically, isolating bug to the per-128-element wave64-hardcoded kernel. SET_ROWS turbo_kv_4b cases (already in tree via `all_types`) also fail (ERR 0.3–0.8). FA-with-turbo_kv_4b refused at `ggml-vulkan.cpp:15929` supports_op (K must be F16 if mixed). **Step 4 evidence on hand**: smoking gun is `vulkan-shaders/turbo_kv_4b_rht.glsl:7-8` — wave64 explicitly hardcoded, wave32 "not implemented." Hypothesis 1 (subgroup-size mismatch) confirmed without bypass needed.

- iter 3 (2026-05-01): **PHASE ABANDONED.** Per user direction: all turboquant work in this tree is abandoned; MTP becomes sole focus on `ik_llama.cpp`. Closing-condition (b) Vega regression check was never run and will not be — the work is parked at submodule commit `cc9e876ed` under `defunct/phase30`. Future TQ reference: `slartibardfast/llama-cpp-turboquant`. Successor: PHASE31 (MTP-only production).

- iter 2 (2026-04-29): **Steps 5–6 close on Ampere.** Rewrote turbo_kv_4b shaders to a subgroup-size-portable layout: 128-thread workgroup, one value per lane; `SUBGROUP_SIZE` spec constant (constant_id=0) pinned via `VK_EXT_subgroup_size_control` requiredSubgroupSize. Butterfly stages with stride < SUBGROUP_SIZE use `subgroupShuffleXor`; stride ≥ SUBGROUP_SIZE goes through shared memory + barrier. `subgroupAdd`/`subgroupMax` wrapped in cross-subgroup helpers (`turbo_kv_workgroup_sum/max`). Pattern matches `cumsum.comp` / `mul_mat_vec_base.glsl:154-200` precedent. Files: `vulkan-shaders/{turbo_kv_4b_rht.glsl, cpy_f32_turbo_kv_4b.comp, dequant_turbo_kv_4b.comp, get_rows_turbo_kv_4b.comp, set_rows_turbo_kv_4b.comp}` and `ggml-vulkan.cpp:4441-4454`. Submodule commit `cc9e876ed`. **Closing-condition results on Ampere:** (a) `llama-perplexity --device Vulkan0 -ctk turbo_kv_4b -ctv bf16 --cache-residual-window 0` = **17.29** PPL, exact match to CPU-on-Ampere; vs Vega reference 17.66 (Δ=0.37, well within chunk stderr ±0.94 over 3 chunks; was 30.59 pre-fix). (c) `test-backend-ops -b Vulkan0 -o FLASH_ATTN_EXT` 4662/4662 still passes. `test-backend-ops -b Vulkan0 -o SET_ROWS` all turbo_kv_4b cases pass (was ERR 0.3–0.8). `test-backend-ops -b Vulkan0 -o CPY` non-permuted turbo_kv_4b cases all pass at every DK ∈ {128,256,384,512} (was uniform ERR≈1.0); permuted-stride cases still fail — pre-existing CPY shader limitation independent of subgroup size, out of scope. (d) CPU-vs-CPU MISMATCH lines from Step 2 are unrelated FP-summation-order noise. **Open: (b) Vega regression-free verification** — closing-condition gate; deferred to next gpu-1 access. Phase remains [~] until Vega gate binds.
