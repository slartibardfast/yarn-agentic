# Phase 28: TURBO_KV_4B — Residual Window Implementation

## Status

**In progress.** Autonomous loop self-paces against a ~1h budget per iteration cycle. This phase executes the design locked in PHASE27 (decisions Q1–Q4 resolved) and closes the 18 SKIPs in `test-turbo-kv-residual-window-pbt.cpp`.

## Scope (inherited from PHASE27)

K-side residual window on `TURBO_KV_4B`. CPU + Vulkan FA dispatch. `--cache-residual-window N` CLI flag, default 128 (Q1). Clamp to `n_ctx` with a warning (Q2). Full buffer at init (Q3). Bytes-for-bytes GGUF persistence (Q4).

## Step checklist

Each step mirrors PHASE27's numbered implementation plan plus a Step 0 harness folded in at user request. Loop iterations update this list.

- [x] **Step 0.** Build `test-turbo-kv-residual-window-harness` — minimal C++ integration test that loads a model, constructs a context with varied `residual_window` / `type_k` combinations, emits structured output on stderr/stdout, exits cleanly. Used to retro-verify Steps 1–2 and to forward-verify every subsequent step without relying on noisy llama-cli stderr.
- [x] **Step 1.** Add `residual_window` to cparams + CLI flag + clamp-to-n_ctx warning.
- [x] **Step 2.** Allocate fp16 side-buffer in `llama-kv-cache.cpp` at context init.
- [x] **Step 3.** Rolling-write path on KV-cache append. Quantise-on-eviction.
- [x] **Step 4.** Overlay write correctness — ring-buffer slot indexing verified by per-slot peek.
- [x] **Step 5.** Two-pass CPU FA dispatch. Online-softmax merge across passes. _All three subtasks closed: (a) 5a — multi-thread cpy_k_window write race fixed by slicing source rows to the surviving last `rw` per stream before set_rows (iter 16); (b) 5b — turbo_kv_4b switched to post-RoPE storage to eliminate the rope-on-the-fly composition issue with FA_LSE merge (iter 21); (c) 5c — turbo_kv_4b rw=0 (17.6640) vs rw=128 (17.6637) match within noise on Qwen3.5 0.8B 3-chunk Wiki PPL, thread-invariant across -t 1 / 4 / 8._
- [ ] **Step 6.** Two-pass Vulkan FA dispatch. Re-run `test-turbo-kv-attention-pbt` at `rw=128`.
- [ ] **Step 7.** GGUF state save/load with fp16 window (Q4 decision).
- [ ] **Step 8.** 9B IQ3_XXS PPL gate: ≤ +0.05 above F16 baseline.
- [x] **Step 9.** Convert 18 SKIPs in `test-turbo-kv-residual-window-pbt.cpp` to `rc::check`. _Closed via iter 10 + 15 + 22 + 23: PBT regenerated from overlay-model spec, all attention-read SKIPs flipped to PASS, 5 reconcile obligations added; final state 28 PASS / 0 SKIP / 0 FAIL._

## Out-of-plan items landed

Small self-contained work adjacent to residual window, done during idle time in the loop:

- [x] T3.2 — stale `@guidance` prose fix in `turbo_kv_4b_attention.allium` (PHASE26 Tier 3.2).

## Step 6 detailed plan — Two-pass Vulkan FA dispatch

This section is written so a downstream agent unfamiliar with llama.cpp's Vulkan backend can execute the plan end-to-end without further architectural decisions. Read this section in full before touching code. Where it says "verify by", do that verification — every substep has an explicit binding gate.

### Why this work, in one paragraph

Step 5 (CPU) closed the two-pass FA + LSE-merge read path that lets the residual-window overlay coexist with `turbo_kv_4b`-quantised main-cache attention. The closing claim was 3-chunk Wiki PPL parity on Qwen3.5 0.8B: rw=0 (17.6640) vs rw=128 turbo_kv_4b (17.6637), thread-invariant across `-t 1/4/8`. Step 6 ports that read path to Vulkan. The narrow blocker is that the Vulkan FA op asserts `ne0 == HSV` and so refuses to emit the LSE output shape `ne0 = HSV+2` (VKQ unscaled in rows `[0..HSV)`, online-softmax max `M` at row `HSV`, denominator `S` at row `HSV+1`).

### Critical reframing — why we do NOT disable Vulkan split-K

Vulkan FA's existing **split-K** path is *not* a competing "splitting" that Step 6 should disable. Split-K is **intra-op GPU parallelism over the K dimension**. The two-pass FA is **op-graph-level partitioning of visible positions via masks**. They are orthogonal axes. Disabling split-K under LSE would make each LSE op run in a single workgroup grid; Pass A sees the long-tail K and would lose most of the GPU. Better: the split-K reduce shader (`flash_attn_split_k_reduce.comp`) already does **online-softmax merge of per-workgroup `(M, S, VKQ_unscaled)` partials** — exactly the LSE math. The right port is to add an `lse_mode` flag that **skips the final divide-by-S step** in two write-out sites:
- the reduce shader (when `split_k > 1`), and
- the primary shader's direct-to-dst path (when `split_k == 1`).

That is the single conceptual change Step 6 is built around.

### Closing condition

Step 6 → `[x]` only when ALL of the following bind:
1. The CPU regression panel is unchanged: f16+rw=128 PPL on CPU still 17.6352, turbo_kv_4b+rw=128 on CPU still 17.6637, `test-flash-attn-lse-merge` 9/9 on CPU.
2. `ctest -R 'flash-attn-lse'` passes on **both** CPU and Vulkan.
3. `ctest -R 'turbo-kv-residual-window-pbt'` is 28 PASS / 0 SKIP / 0 FAIL on **both** CPU and Vulkan.
4. Harness end-to-end runs on Vulkan with rw=128 + turbo_kv_4b + `--check-window` exits 0 with per-layer touched-slot counts matching `min(append_n, rw)`.
5. **PPL parity gate on Vulkan**: Qwen3.5 0.8B BF16, 3-chunk Wiki, `-fa on -ngl 99 --device Vulkan1`. f16+rw=0 baseline ≈ f16+rw=128 ≈ turbo_kv_4b+rw=128, all three within ±0.05 PPL of the baseline, deterministic across two consecutive runs.
6. Iteration log captures evidence for each substep landing.

If gate (5) fails on the binding claim (turbo_kv_4b + rw=128), Step 6 stays open with a 6.11 subtask isolating cause. Same discipline as iter 15b's reopen.

### Vulkan tropes a downstream agent must know

These conventions are how llama.cpp's Vulkan backend works in practice. Internalize them before editing:

- **Push constants vs spec constants vs descriptor sets.** Push constants are a small (≤128 bytes by spec, sometimes 256 on some HW) per-dispatch payload set on the C++ side via `vkCmdPushConstants`; in shaders read via `layout(push_constant) uniform parameter { ... } p;`. Spec (specialization) constants are compile-time values baked into SPIR-V at pipeline-create time; in GLSL `layout(constant_id = N) const TYPE name = default;`. Descriptor sets bind buffers/images. **Convention in llama.cpp Vulkan**: structural variants (head_dim, dtype, mask shape, gqa_ratio) are **spec constants** and produce separate pipeline objects; runtime feature gates (sink presence, mask presence, `lse_mode`) are **push constants** in a single pipeline. `lse_mode` is therefore a push-constant field — adding it does **not** double the FA pipeline count.

- **The FLASH_ATTN_EXT op_params slot map.**
  - Slot 0..2 (floats): `scale`, `max_bias`, `logit_softcap` — set via `ggml_set_op_params(tensor, &floats, 12)`.
  - Slot 3 (i32): precision (`GGML_PREC_F32` or default) — set via `ggml_flash_attn_ext_set_prec()`.
  - Slot 4 (i32): `lse` flag (1 = LSE mode, 0 = standard) — set via `ggml_set_op_params_i32(result, 4, 1)` inside `ggml_flash_attn_ext_lse()` in `ggml.c:5495`.
  - Read on the kernel side via `ggml_get_op_params_i32(dst, 4)`.

- **LSE output layout** (canonical reference; CPU implements this; Vulkan must match):
  - Tensor shape: `{ HSV+2, n_heads, n_queries, batch }` (slot 0 of `ne[]` is `HSV+2`).
  - Per (head, query): row `[0..HSV)` holds **unscaled** VKQ (i.e. the online-softmax numerator before division by S); row `HSV` holds `M` (running max); row `HSV+1` holds `S` (running denominator). Quote from CPU implementation: `ggml-cpu/ops.cpp:8217-8220` and `:8582`.
  - The "unscaled" qualifier is critical. The standard FA op outputs `O = numerator / S`. LSE outputs `numerator` (unscaled) and `S` separately, so the merge graph can re-do the online-softmax across two FA results.

- **Vulkan FA shader file map** (`ggml/src/ggml-vulkan/vulkan-shaders/`):
  - `flash_attn_base.glsl` — shared header (constants, push-constant struct, helper functions, layout decls).
  - `flash_attn.comp` — scalar fallback path, also the entry point that selects coopmat at compile time via `#ifdef`.
  - `flash_attn_cm1.comp` — coopmat1 path (older cooperative-matrix extension).
  - `flash_attn_cm2.comp` — coopmat2 path (newer extension).
  - `flash_attn_mask_opt.comp` — variant for specific mask shapes.
  - `flash_attn_split_k_reduce.comp` — the per-row reduce that combines split-K partials. **Small (~120 lines), clean — read it in full before editing.**

- **Per-workgroup write-out in primary shader (split_k > 1 path).** Already LSE-format-compatible — quotes from `flash_attn.comp`:
  - Line 541 (or 519 for GQA): writes `Of[r][d]` (= unscaled VKQ partial).
  - Line 547 (or 528): writes `Lf[r]` (= S partial).
  - Line 548 (or 529): writes `Mf[r]` (= M partial).
  - The split-K buffer layout is documented at `ggml-vulkan.cpp:9306-9310`: `[HSV, ne1, k, ne2, ne3]` for VKQ, then `[ne1, k, ne2, ne3]` for L and M (concatenated), all backed by `ctx->prealloc_split_k`. **No change is needed in this path** — partial writes already produce LSE-format data; the reduce shader's job changes.

- **Direct-to-dst write in primary shader (split_k == 1 path).** This is the path that must learn LSE. From `flash_attn.comp:554` onward:
  1. Sink-handling block (lines 555–574) — must execute regardless of `lse_mode`. Sinks shift M and rescale L; LSE outputs must reflect this.
  2. `Lfrcp` computation (lines 576–579) — when `lse_mode`, **do not invert** L. Either skip this block or compute and never apply.
  3. The post-`Lfrcp` divide of VKQ by L — when `lse_mode`, **do not apply**. Write VKQ untouched.
  4. Write rows `[0..HSV)` as today (with the divide skipped), then add two writes: M to row `HSV`, L to row `HSV+1`, per (head, query).

- **Reduce shader.** ~120 lines at `flash_attn_split_k_reduce.comp`. The 4 lines that need attention:
  - Line 95: `L = (L == 0.0) ? 0.0 : 1.0 / L;` — when `lse_mode`, skip the inversion (keep `L` as the un-inverted denominator = S).
  - Line 114: `O *= L;` — when `lse_mode`, skip.
  - Line 119: writes `O` to `data_d[(i3 * p.ne2 + i2) * p.ne1 * D + D * n + d]` — the stride `p.ne1 * D` assumes `ne0 == D`. When `lse_mode`, dst's `ne0` is `D+2`, so stride must be `p.ne1 * (D + 2)`.
  - Add: when `lse_mode`, also write `m_max` to `data_d[(i3*p.ne2+i2)*p.ne1*(D+2) + (D+0)*p.ne1 + n]` and `L` (un-inverted) to the next row. Guard by a `if (gl_WorkGroupID.y == 0 && tid == 0)` (one workgroup writes M/S per row, no race).

- **Subgroup uniformity and conditional writes.** From the persisted memory `feedback_barrier_design.md`: zero-init shared memory before conditional writes; ensure subgroup uniformity for branches. `lse_mode` is a push constant, uniform across the entire dispatch — every thread takes the same branch, no divergence concern. The new conditional writes for M/S in the reduce shader add no shared-memory access, so no barrier discipline change.

- **Build pipeline (GLSL → SPIR-V).** llama.cpp builds shaders during cmake configure/build via `glslc`. Shader sources live in `vulkan-shaders/`. On a shader edit, run `cmake --build build -j` from `llama.cpp/`; cmake re-detects the changed source. If the build cache becomes stale (rare), `rm -rf llama.cpp/build/ggml/src/ggml-vulkan/vulkan-shaders-gen/` and rebuild. SPIR-V compile errors print full file:line. **Pitfall**: forgetting to rebuild after a shader edit — the pipeline cache will still serve the old SPIR-V.

- **Common dispatcher pitfalls.**
  - **Push-constant struct must match between C++ and GLSL.** If the C++ `vk_flash_attn_push_constants` adds a `uint32_t lse_mode;` field, the GLSL `layout(push_constant) uniform parameter { ... }` block must add it at the **same byte offset**. Append at the end of both, in the same order.
  - **Push-constant size budget**: the existing struct is ≤128 bytes. Adding one `uint32_t` (4 bytes) is safe.
  - **Alignment**: scalar fields (uint, float) align to 4 bytes; vec2/vec3/vec4 align to 8/16/16. Append `uint32_t lse_mode` at the end — safe.
  - **Forgetting to set the new field at dispatch time** = silent zero. Do a one-shot debug print in the dispatcher for the iteration that introduces the field.

- **Test framework conventions (from llama.cpp/CLAUDE.md).** Prefer `test-backend-ops` (`tests/test-backend-ops.cpp`) when the framework can express the test. For LSE specifically, the existing `test-flash-attn-lse{,-merge}.cpp` tests are CPU-only standalone — extend them to multi-backend rather than duplicating into `test-backend-ops`, because the merge-graph composition is awkward to express through the framework's per-op test surface.

- **Existing diagnostic infrastructure to be aware of.**
  - `LLAMA_DIAG_5B_*` env-var bisection switches were removed in iter 21 — do not look for them.
  - `GGML_VK_PERF_LOGGER=1` — enables Vulkan dispatch-level profiling. Use during 6.10 if PPL gate fails to localize.
  - `--device Vulkan1` selects Vega; `Vulkan0` selects 6800 XT (output of `llama-perplexity --list-devices` is authoritative).

### Pre-work — read-only investigations, must answer before substeps

PW1 — PW3 are read-only. Each gets a one-paragraph entry in the iteration log when answered.

- **PW1 — Cast on Vulkan.** The merge graph in `llama.cpp/src/llama-graph.cpp` (~line 2149–2152) inserts a `ggml_cast` to recast Pass B's K from F32 (the type `ggml_get_rows` always emits on CPU) back to the overlay's native float type. The recon agent reported `GGML_OP_CAST` is **not** dispatched on Vulkan (no `case GGML_OP_CAST:` in `ggml-vulkan.cpp`). **Verify by**: read `llama-graph.cpp:2149` and the surrounding `build_input_window_reorder` function; trace the graph for the target config (`--device Vulkan1 --cache-type-k turbo_kv_4b --cache-residual-window 128`) by running `tests/test-turbo-kv-residual-window-harness --device Vulkan1 --rw 128 --type-k turbo_kv_4b --append 50` and watching for unsupported-op errors or scheduler fallback messages. **If the cast fires and Vulkan does not support it**: STOP. Surface this as a blocker; the resolution is either a refactor of `build_input_window_reorder` to ensure dtype already matches (skipping the cast for our combo) or adding `GGML_OP_CAST` Vulkan support (separate workstream). Do not work around silently.

- **PW2 — `ggml_get_rows` dtype on Vulkan.** CPU's `ggml_get_rows` emits F32 unconditionally — that's why the cast in PW1 exists. On Vulkan, does `ggml_get_rows` preserve source dtype, or also emit F32? **Verify by**: read the Vulkan dispatcher for `GGML_OP_GET_ROWS` at `ggml-vulkan.cpp:10486` and inspect output type construction. If preserved, the recast becomes a no-op the scheduler should optimise out. If F32, PW1's question is acute.

- **PW3 — Which shader variant dispatches for our shape.** Qwen3.5 0.8B has GQA 14:2 with head_dim 128. Which of `flash_attn.comp` / `flash_attn_cm1.comp` / `flash_attn_cm2.comp` does the dispatcher select on Vulkan0 (RDNA2 6800 XT) and Vulkan1 (Vega 10)? **Verify by**: read the variant-selection block in `ggml_vk_flash_attn` (`ggml-vulkan.cpp:~9126`); if opaque from a code read, plan to add a one-shot `fprintf(stderr, "[FA-VARIANT] ...")` in the dispatcher in 6.2 and run a single FA op on each device. Whichever variants dispatch are the ones substep 6.5 must extend; the others can be deferred.

### Substeps — execute in order

Each substep is independently committable. Iteration log gets one line per substep landing.

#### 6.1 — Lift `ne0 == HSV` assertion in Vulkan FA dispatcher

**File**: `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp`, function `ggml_vk_flash_attn` (~line 9126); assertion at line 9155.

**Change**:
```cpp
const bool lse_mode = (ggml_get_op_params_i32(dst, 4) == 1);
const int64_t ne0_expected = lse_mode ? (HSV + 2) : HSV;
GGML_ASSERT(dst->ne[0] == ne0_expected);
```

**Verify by**: build (`cmake --build llama.cpp/build -j`); construct a minimal LSE op via `ggml_flash_attn_ext_lse` against a Vulkan backend; dispatch enters `ggml_vk_flash_attn` without asserting. Output is incorrect at this point — that's expected; this substep only lifts the gate.

#### 6.2 — Plumb `lse_mode` through push constants

**Files and changes**:
1. `ggml-vulkan.cpp:1114-1117` (block `vk_flash_attn_push_constants`) — append `uint32_t lse_mode;`.
2. `ggml-vulkan.cpp:1628-1635` (block `vk_op_flash_attn_split_k_reduce_push_constants`) — append `uint32_t lse_mode; uint32_t ne0_dst;` (the dst stride `ne0_dst` is needed by the reduce shader because, under LSE, output stride changes from `D` to `D+2`; see "Reduce shader" trope above).
3. `ggml-vulkan.cpp` dispatcher pc construction sites (~line 9379–9415, both `split_k > 1` and `split_k == 1` branches) — populate the new fields from `lse_mode` (read at top of dispatcher per 6.1) and dst stride.
4. Shader push-constant blocks — append matching fields in:
   - `flash_attn_base.glsl` (the shared header included by `flash_attn{,_cm1,_cm2,_mask_opt}.comp`).
   - `flash_attn_split_k_reduce.comp` (lines 13–20, the `parameter` block).

**Behaviour change**: none yet. The flag is plumbed but unused.

**Verify by**: rebuild; existing FA ops still pass (`ctest -R 'flash-attn'` on CPU still 9/9, on Vulkan still passes); add a one-shot `fprintf(stderr, "[FA-VK] lse=%u var=%s split_k=%u\n", ...)` in the dispatcher; run `tests/test-flash-attn-lse --backend Vulkan` (extending it minimally first if needed) — confirm the print reports `lse=1` when LSE op dispatched. Remove the print when 6.7's multi-backend test lands.

**Bookkeeping**: PW3's variant-print lives here for one iteration, removed at Step close.

#### 6.3 — Reduce shader: LSE branch

**File**: `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_split_k_reduce.comp` (~120 lines; read it whole).

**Change**:
1. Replace line 95 (`L = (L == 0.0) ? 0.0 : 1.0 / L;`) with:
   ```glsl
   if (p.lse_mode == 0) {
       L = (L == 0.0) ? 0.0 : 1.0 / L;
   }
   // when lse_mode == 1, L stays as the un-inverted denominator (= S in CPU naming)
   ```
2. Replace line 114 (`O *= L;`) with:
   ```glsl
   if (p.lse_mode == 0) {
       O *= L;
   }
   ```
3. Replace the dst stride at line 119 to use `p.ne0_dst`:
   ```glsl
   data_d[(i3 * p.ne2 + i2) * p.ne1 * p.ne0_dst + p.ne0_dst * n + d] = O;
   ```
4. Add an LSE-only writeback after the existing write, gated on `p.lse_mode == 1` and `gl_WorkGroupID.y == 0 && tid == 0` (so exactly one thread per (n, batch) writes M and S):
   ```glsl
   if (p.lse_mode == 1 && gl_WorkGroupID.y == 0 && tid == 0) {
       uint base = (i3 * p.ne2 + i2) * p.ne1 * p.ne0_dst + p.ne0_dst * n;
       data_d[base + D + 0] = m_max;  // M
       data_d[base + D + 1] = L;       // S (un-inverted)
   }
   ```

**Subtlety on indexing**: the existing reduce flattens `(d, n, i2, i3)` with stride `D`. For LSE the stride is `D+2`. The fix above replaces `D` with `p.ne0_dst` everywhere; `p.ne0_dst = D + 2` when `lse_mode`, else `D`. Audit every `D` in this shader after editing — there should be only the index arithmetic uses; inner-loop semantics use `D` as a count (unchanged).

**Verify by**: rebuild shaders; dispatch a Vulkan FA_LSE op with `split_k > 1` (e.g. small KV-per-WG forces split-K via the dispatcher's `split_k = shader_core_count * 2 / total_wgs` formula); compare against CPU FA_LSE on identical input element-wise. Tolerance ~5e-4 abs (matches CPU FA_LSE vs FA gap). Per-element diff > tolerance = bug; report which element diverged (head, query, channel) for triage.

#### 6.4 — Primary shader split_k==1 LSE branch (`flash_attn.comp`)

**File**: `flash_attn.comp` from line 554 (post-split-K-write `return`) through the final dst write at ~line 600+.

**Change**: when `p.lse_mode == 1` (push constant; available because of 6.2):
1. Sink-handling block (lines 555–574) — execute unchanged. Sinks shift M and rescale L; LSE outputs must reflect this.
2. `Lfrcp` computation (lines 576–579) — wrap in `if (p.lse_mode == 0) { ... }` (compute only when scaling will be applied).
3. The VKQ × `Lfrcp` multiply that follows — wrap in `if (p.lse_mode == 0) { ... }` (skip the divide when LSE).
4. The dst write of VKQ rows — extend the row count: when `p.lse_mode`, after writing rows `[0..HSV)` per (query, head), also write `Mf[r]` at row `HSV` and `Lf[r]` (post-sink) at row `HSV+1`. Guard the M/S writes by `d_tid == 0 && col_tid == 0` (one thread per (query, head) writes the scalars; matches the existing pattern at line 545).

**Subtlety**: the dst tensor stride is no longer `p.ne1 * HSV`; it's `p.ne1 * HSV` when `lse_mode == 0` and `p.ne1 * (HSV + 2)` when `lse_mode == 1`. This shader currently computes destination indices directly from `HSV`. When LSE, replace with the actual `ne0` stride — pass it as a push constant or compute from `HSV + 2 * lse_mode`.

**Verify by**: rebuild; dispatch a Vulkan FA_LSE op shape that bypasses split-K (small KV count, single workgroup); compare against CPU element-wise at ~5e-4 abs tolerance.

#### 6.5 — Coopmat variants (gated on PW3)

**Files**: `flash_attn_cm1.comp`, `flash_attn_cm2.comp`. **Skip if PW3 confirms neither variant dispatches on Vulkan0 or Vulkan1 for our shape** — most likely both lack coopmat support for this head_dim/dtype combo.

**Change**: mirror 6.3 + 6.4 patterns in each variant. The coopmat output is held in a cooperative matrix; the unscaled VKQ is the value before final scaling by L — same data already computed before the divide.

**Verify by**: per-variant numerical comparison vs CPU at the same tolerance. Force the variant via the dispatcher's selection knob (typically a CMake feature flag; check `vulkan-shaders/CMakeLists.txt`).

#### 6.6 — Force fp32 accumulation when LSE

**File**: `ggml-vulkan.cpp`, FA dispatcher.

**Change**: when `lse_mode == 1`, override the precision selector to `GGML_PREC_F32`. M is exponent-scale-sensitive; coopmat fp16 accumulation can drift the merge result.

**Verify by**: dispatch LSE op at long context (e.g. KV=2048); compare vs CPU at fp32; agreement within 5e-4 abs. If 6.5 was skipped (no coopmat dispatch), this substep is a no-op for our hardware but still lands as a defensive guard.

#### 6.7 — Multi-backend `test-flash-attn-lse{,-merge}`

**Files**:
- `llama.cpp/tests/test-flash-attn-lse.cpp`
- `llama.cpp/tests/test-flash-attn-lse-merge.cpp`

**Change**: wrap each test case in a backend loop following the pattern in `test-turbo-kv-attention-pbt.cpp:22`. Pseudocode:
```cpp
for (auto * backend : { ggml_backend_cpu_init(), ggml_backend_vk_init(0) }) {
    if (!backend) continue;
    for (auto & tc : test_cases) {
        run(backend, tc);
    }
    ggml_backend_free(backend);
}
```
Use `ggml_backend_vk_init(0)` for the first available Vulkan device; for tests that need a specific device, use `ggml_backend_vk_init(ggml_backend_vk_dev_to_index("Vulkan1"))` or equivalent — check existing tests for the canonical pattern. Tolerance: keep CPU at ~1e-7 short-context; relax Vulkan to 5e-4 abs (matches FA_LSE-vs-FA gap).

**Verify by**: `cd llama.cpp/build && ctest -R 'flash-attn-lse'` passes on both CPU and Vulkan; both binaries report the backend they ran against in their output for log-grep clarity.

#### 6.8 — Vulkan numerical sub-check in `test-turbo-kv-residual-window-pbt`

**File**: `llama.cpp/tests/test-turbo-kv-residual-window-pbt.cpp`.

**Change**: the existing five `ReadKVRecentFromOverlay` PBT obligations are structural — they re-derive the visibility predicate from `set_input_kq_mask_pass_a`/`_pass_b` over a grid of (rw, max_pos). Add **one numerical sub-check** under that obligation: build a small `build_attn_mha_two_pass`-equivalent graph (FA_LSE × 2 + merge) on Vulkan; compare to single-pass FA on the merged K/V. Total stays 28 obligations; this is a numerical leg of an existing one. Tolerance: ~5e-4 abs.

**Verify by**: `ctest -R 'turbo-kv-residual-window-pbt'` reports 28/0/0 on both CPU and Vulkan.

#### 6.9 — Harness end-to-end on Vulkan

**File**: `llama.cpp/tests/test-turbo-kv-residual-window-harness.cpp` — no source change; usage check.

**Verify by**:
```
cd llama.cpp/build
./bin/test-turbo-kv-residual-window-harness \
    -m /opt/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0-MTP.gguf \
    --device Vulkan1 -ngl 99 \
    --rw 128 --type-k turbo_kv_4b \
    --append 200 --check-window
```
Exits 0; per-layer touched-slot counts == 128 (= `min(append_n=200, rw=128)`).

#### 6.10 — PPL parity gate on Vulkan (binding gate)

**Setup**: claim `gpu-1` (Vega) per `coord/` flock protocol before running. `gpu-0` (6800 XT) is reserved for `llama-server.service` — needs explicit user authorisation; **do not claim without asking**.

**Configs** (Qwen3.5 0.8B BF16; 3-chunk Wiki; `-fa on -ngl 99 --device Vulkan1`):

| Config | `--cache-type-k` | rw |
|---|---|---|
| A | f16 | 0 |
| B | f16 | 128 |
| C | turbo_kv_4b | 128 |

Wikitext source: `/home/llm/models/wikitext-2-raw-test.txt` (the *real* plain-UTF-8 file; **NOT** `/opt/models/wikitext-2-raw-v1-test.txt` which is a Parquet file and crashes the tokenizer with "invalid codepoint"). Use `--chunks 3` (or whatever count gives ~3 chunk boundaries). Run each config twice consecutively for determinism check.

**Closing gate**: A ≈ B ≈ C within ±0.05 PPL of A. B-vs-A drift > 0.05 = overlay correctness regression. C-vs-A drift > 0.05 = post-RoPE codebook + overlay regression. Two-run determinism: same PPL to ≥4 decimal places.

**Verify by**: outputs and exact PPL numbers go in iteration log entry. If gate fails, do NOT close 6.11. Open a 6.11.x subtask isolating cause (mask divergence / shader tile-order / coopmat fp16 residue / cast fallback to CPU silently).

**Release** `gpu-1` to IDLE in `coord/gpu-1.state` after the runs.

#### 6.11 — Close

PHASE28.md Step 6 → `[x]` only when 6.1–6.10 all green and the closing condition list above binds. Iteration log captures each substep landing as a single line. Remove the one-shot debug prints from 6.2 / PW3.

### Files touched (canonical list)

Edits during Step 6 — kept inventory so a downstream agent can cross-check git diff:

- `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` — push-constant blocks (~1114, ~1628), dispatcher (~9126, ~9155, ~9379–9415).
- `llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_base.glsl` — push-constant struct.
- `llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp` — line 554+ (split_k==1 branch).
- `llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_split_k_reduce.comp` — lines 13–20 (push-constant block), 95, 100–119 (writeback).
- `llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm1.comp` — gated on PW3.
- `llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm2.comp` — gated on PW3.
- `llama.cpp/tests/test-flash-attn-lse.cpp` — backend loop wrapper.
- `llama.cpp/tests/test-flash-attn-lse-merge.cpp` — backend loop wrapper.
- `llama.cpp/tests/test-turbo-kv-residual-window-pbt.cpp` — numerical Vulkan case under existing obligation.
- `PHASE28.md` (this file) — iteration log entries; checkbox flip on Step 6.

Read-only references (do not edit during Step 6):
- `llama.cpp/ggml/src/ggml.c:5460-5504` (`ggml_flash_attn_ext_lse` API).
- `llama.cpp/ggml/src/ggml-cpu/ops.cpp:8217-8226, 8580-8596` (CPU LSE write layout).
- `llama.cpp/src/llama-graph.cpp:~2149-2152` (PW1 read site).

### Build and test commands (paste-friendly)

```
# 1. Rebuild after code or shader edit
cd /home/llm/yarn-agentic/llama.cpp/build
cmake --build . -j

# 2. CPU regression (fast, run after every edit)
ctest -R 'flash-attn-lse|turbo-kv' --output-on-failure

# 3. Vulkan tests (after substeps 6.1-6.7)
ctest -R 'flash-attn-lse' --output-on-failure
ctest -R 'turbo-kv-residual-window-pbt' --output-on-failure

# 4. Harness on Vulkan (substep 6.9)
./bin/test-turbo-kv-residual-window-harness \
    -m /opt/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0-MTP.gguf \
    --device Vulkan1 -ngl 99 \
    --rw 128 --type-k turbo_kv_4b \
    --append 200 --check-window

# 5. PPL gate (substep 6.10) — claim gpu-1 first via coord/
./bin/llama-perplexity \
    -m /opt/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0-MTP.gguf \
    --device Vulkan1 -ngl 99 -fa on \
    --cache-type-k f16 --cache-type-v f16 --cache-residual-window 0 \
    -f /home/llm/models/wikitext-2-raw-test.txt --chunks 3
# (repeat with --cache-type-k f16 ... --cache-residual-window 128 for B,
#  and --cache-type-k turbo_kv_4b ... --cache-residual-window 128 for C)
```

### Risk register

- **R1 — `ggml_cast` not on Vulkan.** PW1 is the answer. If it fires for our config, Step 6 stops and we discuss a refactor or cast op support. Severity: hard blocker.
- **R2 — Reduce shader stride math under HSV+2.** Lines 103, 119 use `p.D` as the dst stride. `p.ne0_dst` push constant addresses this. Don't forget `o_offset` reads from the **split-K buffer** still use `p.D` (those stride into the input partial, which is HSV-sized) — only the dst writes change.
- **R3 — Coopmat variant divergence.** Each cooperative-matrix shader has its own write-out unpacking. Each must independently learn LSE. Mitigation: do scalar (6.3 + 6.4) first; gate 6.5 on PW3.
- **R4 — Stride assumptions downstream of FA.** Views / permutes / reshapes consuming the FA output assume `ne0 == HSV`. With `HSV+2` they break silently. Mitigation: search `ggml_view`/`ggml_permute`/`ggml_reshape` adjacent to FA output consumption; `build_attn_mha_two_pass` and `build_fa_lse_merge` are dtype/dim-agnostic at the op level (sub/exp/mul/get_rows on the LSE outputs).
- **R5 — Coopmat fp16 accumulation.** M is exponent-scale-sensitive. 6.6 forces fp32 when LSE.
- **R6 — Disabling split-K accidentally.** Substeps 6.1 / 6.2 must NOT change the `lse_mode == 0` path. Regression panel (CPU PPL, CPU FA test) catches it; Vulkan single-pass FA on a non-LSE op with `split_k > 1` should still take the existing path unchanged.
- **R7 — Tolerance brittleness Vega vs RDNA2.** Tile shapes differ; numerical results may need per-device tolerance. Mitigation: 5e-4 abs is the existing FA_LSE-vs-FA tolerance; if Vega needs looser, document why explicitly in iteration log.
- **R8 — Pipeline cache stale after shader edit.** If a numerical test fails mysteriously after a shader edit, suspect cache. Mitigation: `rm -rf llama.cpp/build/ggml/src/ggml-vulkan/vulkan-shaders-gen/*-spv` and rebuild.
- **R9 — Push constant byte-offset mismatch.** If C++ struct and GLSL `parameter` block disagree on field order, every dispatch reads garbage with no error. Mitigation: when adding fields, append at the end of both, in the same order. Match the names too — divergent names compile but signal intent drift.

### Open questions

- **OQ1 — Cast dependency** (PW1).
- **OQ2 — PPL tolerance on Vulkan.** ±0.05 was Step 5's CPU bound; coopmat fp16-accumulation paths may need ±0.1. Decide before 6.10, document in log.
- **OQ3 — Should `qwen_pp512_rw128` test case extend to Vulkan inline as a sentinel?** Or is the harness-driven PPL the only inline gate? Decide as 6.7 lands.
- **OQ4 — `test-backend-ops` integration.** llama.cpp/CLAUDE.md prefers it. The LSE op_param sweep and the merge-graph composition fit awkwardly. Revisit if 6.7 ends up duplicating significant infra.

### Iteration log entry template

For each substep landing, append one line to "## Loop log" of the form:
```
- Iteration N: substep 6.X landed (llama.cpp master `<sha>`). <what changed in one sentence>. <verification evidence in one sentence>. <regression panel state>.
```
Match the prose style of iterations 22 / 23 for consistency.

### Estimated complexity

Wall-clock per substep (focused work, not including review/discussion):

- 6.1, 6.2, 6.6: ~1h each (dispatcher + push-constant edits).
- 6.3: ~3h (reduce shader edit + roundtrip debug).
- 6.4: ~3h (primary scalar shader edit).
- 6.5: ~3–6h **per** coopmat variant if PW3 confirms — gate on PW3.
- 6.7: ~2h per test.
- 6.8: ~2h.
- 6.9: ~30min active + harness wallclock.
- 6.10: ~30min active + ~15min PPL run wallclock per config × 6 runs (3 configs × 2) ≈ 90 min wallclock.
- 6.11: trivial.

Total: ~1.5 days minimum, ~3 days realistic, dominated by 6.3/6.4/6.5 shader debug and 6.10 PPL wallclock.

## Loop log

Each iteration appends a single line noting what landed.

- Iteration 1: loop started; PHASE28.md stub created; T3.2 prose fix in `turbo_kv_4b_attention.allium` (`allium check`: 0 errors).
- Iteration 2: Step 1 landed on llama.cpp master (`b746a733a`) — `residual_window` wired through `llama_cparams`, `llama_context_params`, `common_params`, `--cache-residual-window` flag, clamp-to-n_ctx warning. `--help` smoke confirms flag + default 128. Runtime clamp smoke deferred (llama-cli stderr is noisy during prompt eval; will verify once Step 2 gives a cleaner integration point). Also: cleanup commit (`afa8baf9c`) strips host-planning references from the four turbo_kv PBT skip-stubs per `feedback_no_host_concerns_in_code`.
- Iteration 3: Step 2 landed on llama.cpp master (`1bcd5b179`) — fp16 rolling-tail tensor per layer allocated via `ggml_new_tensor_3d(GGML_TYPE_F16, n_embd_k_gqa, residual_window, n_stream)` when `residual_window > 0`, nullptr otherwise. Constructor signature on `llama_kv_cache` gains `residual_window` before the filter/reuse callbacks; callers in `llama-model.cpp`, `llama-memory-hybrid.cpp`, and `llama-kv-cache-iswa.cpp` updated. iSWA and hybrid paths pass 0 (follow-on work flagged in inline comments — those paths don't overlap with current TURBO_KV target configs). Smoke: `llama-cli --cache-residual-window 0` and `... --cache-residual-window 128 --cache-type-k turbo_kv_4b` both load and begin decoding without crash. Loop stopped at 36/60 min to avoid landing a partial Step 3.
- Iteration 4: Step 0 harness landed on llama.cpp master (`57c4c2254`) — `test-turbo-kv-residual-window-harness` provides a decode-free llama_context init test with grep-friendly output. Running the harness against Qwen3.5 0.8B immediately exposed a Step-2 gap: the 0.8B is a hybrid DeltaNet+attention model that routes through `llama_memory_hybrid`, where the prior Step 2 hardcoded `residual_window=0`. Same commit extends the param wire through all 4 cache classes (`llama_memory_hybrid`, `llama_memory_hybrid_iswa`, `llama_kv_cache_iswa`, plus the already-wired `llama_kv_cache`). Harness confirms allocation now scales linearly on the hybrid path: `rw=0` → 4.48 MiB, `rw=128` → 5.36 MiB (+0.88 MiB for fp16 tail), `rw=9999` → 7.98 MiB clamped with warning. Steps 1 + 2 now fully verified against a real model init path.
- Iteration 5: Step 3 landed on llama.cpp master (`c57cff006`). Graph-build wiring for fp16 overlay writes on K cache append: new `cpy_k_window` method on `llama_kv_cache` (+ context wrapper) emits `ggml_set_rows` from the current K projection into a flat view of `layers[ikv].k_window_fp16`, using I64 slot indices `s*residual_window + (pos % residual_window)`. `llm_graph_input_attn_kv` gains a `self_k_window_idxs` field populated every ubatch via the existing set_input callback. The main K cache is unchanged — the fp16 buffer is a pure overlay. Harness extended with `--append N` to drive real decodes; `rw=128 ctx=512 type_k=turbo_kv_4b --append 200` on Qwen3.5 0.8B Q8_0 decodes all 200 tokens with KV buffer growing from 3.84 MiB reported to 4.59 MiB allocated (0.75 MiB fp16 window delta). No read path yet — that arrives with Step 5's two-pass FA dispatch.
- Iteration 6: mid-step design fix for native-weight awareness. The Step-2 overlay allocation was hardcoded `GGML_TYPE_F16`, which silently risks overflow on BF16-trained models (Qwen3.5 et al.) whose K activations can exceed fp16's ±65504 on wide-range heads. Fix: new `residual_window_type_k` cparam (public `llama_context_params` + internal `llama_cparams`) accepting F16, BF16, or COUNT=auto. Auto-resolution at context init reads `model.layers[i].wk->type`: any BF16 → BF16 overlay; all F16/F32 → F16 overlay; any quantised + no F16 → BF16 overlay (safe default, covers IQ3_XXS/Q8_0/... of BF16-trained sources). CLI: `--cache-residual-window-type {auto|f16|bf16}`. Harness: `--rw-type-k` added. Four-way verification on Qwen3.5 0.8B (`7abbf6215`): BF16 GGUF auto → bf16, Q8_0 GGUF auto → bf16 (quantised-no-f16 branch), BF16 + explicit f16 → f16, BF16 + explicit bf16 → bf16. All 200-decode runs succeed. Memory cost identical (both types are 2 bytes). Step 3 tick retained; the design-fix is scoped to Step 2's allocation choice rather than Step 3's wiring.
- Iteration 7: Step 4 landed on llama.cpp master (`6b2331f94`). New public peek API `llama_memory_residual_window_peek(mem, il, stream, slot, dst, dst_size)` via virtual methods on `llama_memory_i` with default-0 returns; concrete impl on `llama_kv_cache` uses `ggml_backend_tensor_get` so it works across backends. iSWA and the two hybrid memory types forward to inner attn cache. Harness gains `--check-window`: iterates every (layer, slot) after `--append` and asserts exactly `min(append_n, rw)` non-zero slots per layer. **Latent bug caught by the new test and fixed in the same commit**: `llm_graph_input_mem_hybrid::set_input` called individual set_input_* methods directly instead of delegating to `llm_graph_input_attn_kv::set_input`, so `k_window_idxs` was never populated and every hybrid-model decode wrote to slot 0 regardless of position — a silent no-op that would have corrupted Step 5's FA read path on Qwen3.5 (all sizes are hybrid). Three-way Qwen3.5 0.8B BF16 verification: rw=128/append=200 → 128/128 slots touched, rw=128/append=50 → 50/128, rw=32/append=200 → 32/32 (all 7 attention layers pass).
- Iteration 8: native-float auto-resolution extended to type_k and type_v (llama.cpp master `0249a91ea`). V cache previously defaulted to F16 unconditionally; on Qwen3.5 (BF16-native), V = Wv @ x inherits the same wide-range-head overflow risk that the overlay dtype auto-resolution already guards against. Fix: same wk-tensor-dtype rule now gates type_k, type_v, and residual_window_type_k in `llama_init_from_model`. Defaults flip from F16 to GGML_TYPE_COUNT (= auto); CLI gains `--cache-type-k auto` / `--cache-type-v auto`. Explicit user choices respected verbatim (including quantised like turbo_kv_4b). Harness verifies all three resolve to bf16 on BF16 GGUF; Q8_0 GGUF hits the quantised-no-f16 branch and also resolves to bf16. No behavioural change for users passing explicit types. On-path for the attention-read path: ensures the two-pass FA can't be corrupted by V-cache overflow independent of whether the overlay logic is correct.
- Iteration 9: spec realignment via `/allium:weed` then `/allium:tend`. Weed pass surfaced a root-level divergence — `turbo_kv_residual_window.allium` was written for an **eviction** model (fp16 window, quantise-on-eviction); implementation built an **overlay** model (always-quantise + parallel fp16/bf16 ring buffer). CoverageCompleteNoOverlap invariant was structurally false, AppendRowWithEviction / QuantizedHead / Tensor / FloatRow / min_seq_for_quantisation were dead legacy declarations, and the entire overlay-dtype auto-resolution feature plus the peek-API surface were unspec'd. Tend pass rewrote the spec (main commit `f13f500`, 255 insertions / 148 deletions) to the overlay model: new entities KVector + Context, enum OverlayDtype, rules ComputeOverlaySlot + AppendOverlayRow + ResolveOverlayDtypeAuto, invariants ResidualWindowWithinContext + OverlayDtypeIsFloat, actor OverlayInspector + surface OverlayPeek. Attention-read rules kept as explicitly-flagged-aspirational (ReadKVRecentFromOverlay / ReadKVTailFromMainCache). Six new open questions documented: auto-resolution placement, RoPE-boundary representation, read-path structure (two-pass FA+LSE vs assembled-K), rollback semantics, persistence, cross-backend equivalence. `allium check`: 0 errors, 3 warnings (external-entity source hints — same pattern as sister specs). The 18 SKIP placeholders in test-turbo-kv-residual-window-pbt.cpp are now obsolete post-rewrite; require a follow-up `/allium:propagate` pass to regenerate from the new spec before Step 5 implementation.
- Iteration 10: `/allium:propagate` regenerated PBTs from the overlay-model spec (llama.cpp master `3e9f9b5bf`). 19 obsolete eviction-model SKIPs replaced with 17 real assertions + 6 aspirational SKIPs. Coverage of all 23 spec obligations: entity fields (KVector, Context), enum comparability (OverlayDtype), config defaults (residual_window=128, overlay_dtype), rule success/failure for ComputeOverlaySlot / AppendOverlayRow / ResolveOverlayDtypeAuto, invariants (ResidualWindowWithinContext, OverlayDtypeIsFloat), peek surface (actor/exposure/provides). The 6 SKIPs all target ReadKVRecentFromOverlay / ReadKVTailFromMainCache — the attention read path is the last major work item. CMake gains `llama` link dep for the public C API calls. Exit 0; harness still passes. Two incidental spec-vs-code notes surfaced for a future weed+tend cycle: (1) config.overlay_dtype spec default `f16` vs code default `GGML_TYPE_COUNT=auto`; (2) Context.n_layer is a model-level attribute, not a cparam. Both non-blocking. Spec now has a concrete rule to implement (ReadKVRecentFromOverlay) and PBTs that will flip SKIP→PASS when the implementation lands.
- Iteration 11: read-path approach recorded as `@guidance` on ReadKVRecentFromOverlay (main commit `da6b080`) — two-pass FA with online-softmax merge; rationale (avoids full-cache dequant that would defeat turbo_kv_4b compression), merge formula, and FA-kernel leverage points spelled out in the spec. Then foundation piece landed on llama.cpp master (`8cafefe2b`): new public API `ggml_flash_attn_ext_lse` emits the unscaled (VKQ, M, S) per (head, query) required by the merge. Implementation reuses the existing FLASH_ATTN_EXT op type with a flag in op_params slot 4; kernel exposes the (M, S) state that the existing threaded-reduction path already computes internally. CPU dispatcher forces single-chunk non-tiled execution when the flag is set (parallelism deferred). New test `test-flash-attn-lse` verifies FA_LSE and FA agree numerically after manual (VKQ / S) scaling across 5 cases (masked/unmasked, 64/128 head dim, 1/4 batch, 8..512 kv). Short-ctx agreement to ~1e-7 relative; at K=512 the agreement degrades to ~1e-4 absolute (FA's chunked split-KV path vs LSE's single-chunk — accumulation order, not numerics). Non-CPU backends not yet plumbed; that's part of the Vulkan port.
- Iteration 12: merge-graph foundation landed on llama.cpp master (`33f140f5a`) — `test-flash-attn-lse-merge` constructs the online-softmax merge of two FA_LSE outputs from existing ggml ops only (add/sub/abs/exp/mul/div/scale/clamp; no new kernel). Verified against single-pass FA over 8 cases: symmetric split, unmasked, early/late split offsets, Qwen-dims decode + PP batch 4, long-context 512-key split, and the empty-pass-A handover corner (Pass A entirely masked → M=-inf, S=0, kernel emits zeros; merge must still produce a finite result equal to Pass B). All pass with max abs diff ~5e-4, which is the existing FA_LSE vs FA gap (chunked split-KV vs single-chunk accumulation order), not a merge defect. Robustness trick: clamp M to >= -1e30 before the elementwise max to avoid NaN from (-inf + +inf) in the |diff| branch. This is the merge the two-pass read path in `build_attn_mha` will call — it's been tested in isolation so the integration step can focus on shapes and mask construction, not numerics.
- Iteration 13: two-pass FA dispatch landed on llama.cpp master (`cc3688df6`) — identity pass. `build_attn_mha_two_pass` in `llm_graph_context` runs `ggml_flash_attn_ext_lse` twice (pass_a mask: main-cache positions older than residual_window; pass_b mask: positions inside the window) and merges via `build_fa_lse_merge`. Both passes still read from the main K/V cache at this stage — the two masks partition the base kq_mask so the merged output is numerically equal to single-pass FA (verification step). Plumbing: new `set_input_kq_mask_pass_a`/`_pass_b` on `llama_kv_cache` derive the per-pass masks from the base kq_mask then apply the p0-vs-p1-rw tail filter; new `self_kq_mask_pass_a`/`_pass_b` fields on `llm_graph_input_attn_kv` populated by the existing hybrid and kv set_input callbacks. Null-buffer guard added to `set_input_kq_mask` for the case where the base mask tensor is built but unreferenced by the graph (scheduler then skips buffer alloc). LSE ops named via `LLAMA_TENSOR_NAME_FATTN` to satisfy the sched_reserve FA device-mismatch probe. Verification on Qwen3.5 0.8B: harness with rw=128 + append=50 + check-window PASS on both f16 and turbo_kv_4b caches; CPU-only Wiki-3-chunk perplexity: rw=0 → 16.6672, rw=128 → 16.6655 (within fp32 accumulation noise as designed). Follow-up commit will replace pass_b's K with the overlay via a ggml_get_rows reorder; Vulkan FA_LSE port is queued alongside the residual-window Vulkan work (current Vulkan FA shader asserts on `ne0 == HSV` because it doesn't know about the LSE flag).
- Iteration 14: overlay-K read path landed on llama.cpp master (`a4ad9194c`). `build_attn_mha_two_pass` now consumes distinct K/V per pass — Pass A from the main cache (tail masked past the shared stream-wide max_pos boundary), Pass B from the fp16/bf16 overlay reordered into position order via `ggml_get_rows(overlay, window_reorder)` + re-cast back to the overlay's native float type (ggml_get_rows always emits F32; without the recast Pass A's F16 K and Pass B's F32 K trigger a multi-thread divergence in the CPU FA kernel's per-K-type vec_dot dispatch). V comes from the same main cache sliced by ggml_get_rows with a per-ubatch `v_window_idxs` tensor populated from `v_cells`+`sinfo`. New input tensors `self_window_reorder` + `self_v_window_idxs` on `llm_graph_input_attn_kv`; new `get_k_window`/`get_v_window`/`build_input_window_reorder`/`build_input_v_window_idxs` + setters on `llama_kv_cache`. Pass B mask shape shrinks from `[n_kv, n_tps, 1, n_stream]` to `[rw, n_tps, 1, n_stream]`: slot s corresponds to position `max_pos_stream - rw + 1 + s` and visibility reduces to `0 <= p_slot <= p_i`. Mask-partition fix: Pass A uses the shared stream-wide max_pos boundary (not per-query `p_i - rw`) — the earlier per-query scheme produced NaN for tokens whose small `p_i` pushed their recent-window below the boundary, so they got masked in both passes and the merge divided by S_a+S_b=0. Verification on Qwen3.5 0.8B f16 + f16 overlay, -t 1, CPU, 3-chunk Wiki PPL: rw=0 → 16.6672, rw=128 overlay → 16.6633 (design-verified match). Known-open follow-ups: (1) multi-thread FA_LSE regression at small K_len — single-thread always correct but PPL climbs to 11.77 at -t 8 on this config, root-cause pending (the LSE dispatcher pins execution to thread 0 and barriers the rest, so the divergence is downstream in the merge graph or the wdata scratch layout); (2) pre-RoPE K overlays (turbo_kv_4b, split-K) are gated off — the overlay stores what `cpy_k` gets, which for those cache types is pre-RoPE, so Pass B would need RoPE-on-the-fly before the FA_LSE call; (3) `n_stream > 1` (multi-sequence eval) falls back to single-pass.
- Iteration 15: the 6 read-path PBT SKIPs in `test-turbo-kv-residual-window-pbt.cpp` now assert (llama.cpp master `f45bbe65c`). Each SKIP flipped to a structural check re-derives the Pass A / Pass B visibility predicate that `llama_kv_cache::set_input_kq_mask_pass_a` / `_pass_b` enforces, then verifies over a grid of (rw, max_pos) that every (attended_pos, query_pos) cell lands in exactly one pass and the pass matches the spec's declared source (overlay for recent, main_cache for tail). End-to-end numerical coverage remains the PPL check from iteration 14. Result: 23 PASS / 0 SKIP / 0 FAIL (was 17/6/0).
- Iteration 15b: Step 5 checklist UN-closed. The earlier mark was wrong — two-pass infrastructure landed but the feature isn't working at default thread count and doesn't fire for turbo_kv_4b (the target cache type of PHASE27 scope). Reopening the box to block Step 6 (Vulkan) and Step 8 (PPL gate) on a correct CPU baseline. Subtasks added in the checklist: 5a multi-thread FA_LSE fix, 5b pre-RoPE overlay for turbo_kv_4b, 5c turbo_kv_4b PPL A/B.
- Iteration 16: subtask 5a fixed (llama.cpp master `8f24f47e8`). Root cause was NOT in FA_LSE — the kernel pins to thread 0 and barriers correctly. The race was in `cpy_k_window` (set_rows into the overlay ring buffer): when `n_tps > residual_window`, multiple source tokens map to the same overlay slot via `pos % rw`. Single-thread set_rows wrote rows in order so the last write (newest position) won; multi-thread set_rows partitions source rows across threads and the final value at each colliding slot became order-dependent. Fix: slice `k_cur` and `k_window_idxs` to the last `rw` rows per stream before set_rows — those are the only rows whose data survives the collisions anyway, and each maps to a unique slot. Verification on Qwen3.5 0.8B BF16, f16 cache + f16 overlay, 3-chunk Wiki PPL: -t 1, -t 4, -t 8 all give 17.6352 deterministically (was 17.64 / 17.95 / 18.13 racy pre-fix). `test-flash-attn-lse-merge` gained a `qwen_pp512_rw128` case mirroring the PPL workload (passes nth=4). Subtasks 5b (pre-RoPE overlay for turbo_kv_4b) and 5c (turbo_kv_4b PPL A/B) remain open.
- Iteration 17: spec resolved the RoPE-boundary open question (parent commit `b0d1231`, allium check 0 errors). `ReadKVRecentFromOverlay`'s `@guidance` now spells out: post-RoPE-store caches read the overlay slice directly; pre-RoPE-store caches (turbo_kv_4b, split-K) apply the same `ggml_rope_multi` recipe used on the main-cache read path, keyed on a position tensor for the recent-window slice. Cross-instance numerical equivalence is captured as a behavioural property in @guidance rather than a top-level invariant — single-point-in-time invariants don't express it (per language reference §"Recognising expressible invariants"). The MTP-rollback half of the original open question is already covered by the separate `Rollback semantics` open question.
- Iteration 18: subtask 5b infrastructure landed but **NOT WORKING** — gated off (llama.cpp master `7c3556d95`). New plumbing: `build_input_window_k_pos` / `set_input_window_k_pos` (`[rw * 4]` I32 tensor in position order), `self_window_k_pos` field on `llm_graph_input_attn_kv`, RoPE-on-the-fly block in `build_attn` that mirrors the main-cache read recipe. Dispatch gate left at `self_k_pos == nullptr` so pre-RoPE caches keep falling back to single-pass FA — Qwen3.5 0.8B turbo_kv_4b at -t 8 stays at 17.6968 (matches rw=0 baseline). When the gate is lifted (turbo_kv_4b + rw=128 with two-pass active), PPL regresses to 18.5815 (+0.84, race-free, scales linearly with rw, dtype-insensitive within ±0.05 across F16/BF16/F32 at the FA boundary). Investigation has ruled out: multi-thread race, position-tensor stride/section indexing, slot↔position alignment, K dtype at the FA boundary. Open hypothesis: per-position K@RoPE divergence between main-cache path (dequant + rope) and overlay path (cast + rope) for the same position p — needs a numerical comparison test. Step 5 stays `[ ]` with subtask 5b reopened (RoPE alignment infra landed, read-path numerical correctness still incomplete).
- Iteration 19: bisection deepens (llama.cpp master `ac5f397e5`). Two env-gated diagnostic switches added — `LLAMA_DIAG_5B_OPEN_GATE` lifts the gate so turbo_kv_4b actually exercises two-pass + rope-on-the-fly; `LLAMA_DIAG_5B_PASS_B_FROM_MAIN` substitutes Pass B's K source with `ggml_get_rows(dequant(main_K), v_window_idxs)` — Pass B then reads the same data Pass A reads, just at different cell ranges. Results on Qwen3.5 0.8B turbo_kv_4b -t 8 3-chunk Wiki: rw=0 single-pass 17.6968 → OPEN_GATE only (overlay→cast→rope) 18.5815 → OPEN_GATE+PASS_B_FROM_MAIN (main→cast→rope) 18.5999. Both two-pass configs agree within 0.02 — **the overlay is NOT the bug source**. Same code with rw=512 (Pass B covers entire batch, Pass A masked): 35.1224 (+17.4 vs single-pass). PPL degradation scales linearly with rw (Pass B's visible-position count). Diff between f16-cache (works) and turbo_kv_4b (broken): the rope-on-the-fly step applied at attention time. Hypothesis for next iteration: investigate whether the rope-on-the-fly path produces the expected post-rope values at the FA boundary, via a `ggml_backend_sched_eval_callback` that dumps per-position K@RoPE for layer 0.
- Iteration 20: turbo_kv_4b quant pipeline verified clean. `test-turbo-kv-rht`: 9/9 pass (RHT roundtrip bit-exact, orthogonality preserved, multi-block consistency exact). `test-turbo-kv-4b-attn`: 8/8 pass at max_err=0.000000 across head_dim=128/256, valid_count=1/10/500, varied stride. `dequant(quant(K_pre))` is faithful per-block. Single-pass FA off+on agree at 17.69 vs 17.70 for turbo_kv_4b rw=0 (FA itself correct). The +0.84 gap must arise only in the *composition* of (turbo_kv_4b store) × (rope-on-the-fly) × (FA_LSE two-pass merge). Existing test-flash-attn-lse-merge `qwen_pp512_rw128` case validates the merge math against a full-FA reference at the EXACT shape and split point used in PPL — passes at 9.287e-04 max abs diff. So the merge math is provably right for matched random inputs. Either (a) the production graph feeds different K to Pass A vs Pass B (graph aliasing or scheduler issue), or (b) some interaction between the rope-on-the-fly node and its consumers that I haven't isolated. Direct cb_eval instrumentation deferred to a fresh-context iteration; current session is leaving 5b in the gated-off state.
- Iteration 21: subtask 5b FIXED via design flip (llama.cpp master `d0137193a`). After bisection ruled out overlay storage, K dtype at FA boundary, multi-thread race, and per-block quant fidelity, the residual hypothesis was that the rope-on-the-fly *composition* with FA_LSE merge was structurally incorrect — Pass A's full-cache rope output and Pass B's window-slice rope output produced merge results that diverged from single-pass FA on the same K cache. Fix: **flip turbo_kv_4b to post-RoPE storage**. RoPE is now applied to Kcur in `qwen35.cpp` before `cpy_k` (the existing `if (!inp->self_k_pos)` guard auto-fires once `build_input_k_pos` returns null for turbo_kv_4b). Main cache holds `quant(K_pre @ RoPE(p))`; read path collapses to `get_k → cast → FA`, identical to f16/bf16. Two-pass + LSE merge then reduces to the (proven-correct) f16-cache configuration. Single-pass turbo_kv_4b PPL: 17.6968 (pre-RoPE) → 17.6640 (post-RoPE), 0.033 BETTER within noise — codebook calibration unaffected because per-block normalize+RHT homogenises the RoPE-rotated K_pre distribution before quantisation. Two-pass turbo_kv_4b rw=128: 17.6637, thread-invariant across -t 1/4/8. The dead pre-RoPE rope-on-the-fly block in `build_attn` and the diagnostic env-var bisection switches from iter 19 are removed; `build_input_window_k_pos` / `set_input_window_k_pos` infrastructure remains for future split-K + residual_window combinations but is dormant for all currently-tested cache types. f16 path unaffected (still 17.6352, 5a preserved). 5c trivially closed by the same data: rw=0 (17.6640) vs rw=128 (17.6637) match within noise. Step 5 done. **MTP compatibility note** (per user direction): post-RoPE storage relies on llama.cpp's standard `k_shift` mechanism for any future MTP position-shift requirements, the same as f16/bf16 caches. Position-agnostic storage of pre-RoPE K is no longer free; that tradeoff is deferred to a redesign phase if MTP rewind semantics demand it.
- Iteration 22: MTP-gap closure plan A→B→C→D landed (llama.cpp master `971235dbc`). Phase A — spec narrowed: strategy (A) restore-from-main selected for ReconcileOverlayOnSequenceRemoval; Shift/Divide/Copy/Keep deferred behind hard runtime guards. Phase B — `llama_kv_cache::reconcile_overlay_after_removal(stream_idx)` implemented with the restore-from-main strategy: per ring slot, find highest-position surviving cell whose `pos % rw == s`, dequant from main K (handles F32/F16/BF16 plus quantised types via `traits->to_float`), convert to overlay dtype (F16/BF16), `ggml_backend_tensor_set` to slot. Hooked into `seq_rm` after the cell-removal loop. No-op when `residual_window == 0`. Phase C — harness gains `--seq-rm-then-peek <p_min> <p_max>` mode that snapshots overlay slot bytes before/after `llama_memory_seq_rm` and counts changed slots vs expected `|{ p % rw : p in [p_min, p_max] }|`. **Known limitation documented inline**: on hybrid models like Qwen3.5, public `llama_memory_seq_rm` fails at the recurrent layer (SSM state cannot roll back without checkpoints) before reaching the attention cache. Direct exercise of reconcile via the harness requires either a non-hybrid model or a test-only API bypassing `llama_memory_hybrid::seq_rm`; behavioural verification is deferred to the MTP integration smoke test (Phase F). Phase D — PBT propagation: 5 new `ReconcileOverlayOnSequence*` obligations added; total 28 (was 23). Removal asserts (code-existence level); Shift/Divide/Copy/Keep stay SKIP pending guards/strategy. Regression panel clean: f16+rw=128 PPL 17.6352, turbo_kv_4b+rw=128 PPL 17.6637, test-flash-attn-lse-merge 9/9, PBT 24 PASS / 4 SKIP / 0 FAIL.
- Iteration 23: MTP-gap closure plan E→F→G landed (llama.cpp master `5f2df7b4c`). Phase E — hard runtime asserts at the top of `seq_add`, `seq_div`, `seq_cp`, `seq_keep` fail loudly when `residual_window > 0`, each naming the corresponding spec rule (`ReconcileOverlayOnSequence{Shift,Divide,Copy,Keep}`) so callers grep straight to the deferral. PBT obligations for the four guards flip from SKIP to PASS — the "intentional gap, guarded" condition is now testable. Total 28 PASS / 0 SKIP / 0 FAIL. Phase F — MTP integration smoke via `test-mtp-snapshot` on Qwen3.5 0.8B BF16 + turbo_kv_4b cache: FULL snapshot/restore PASS at both rw=0 and rw=128 (parity); PARTIAL snapshot/restore FAIL at both rw=0 and rw=128 with identical divergence (got 6749 expected 557). The PARTIAL failure is a pre-existing recurrent-state partial-restore issue unrelated to overlay reconciliation (out of scope for this batch); the smoke gate "MTP-style ops don't regress with reconcile" is satisfied because behaviour is identical at both rw values. Phase G — final weed: 5/5 reconcile rules align with code (1 implemented, 4 guarded as intentional gaps with hard asserts). All cross-references between spec rule names and code hooks correct in both directions. Open questions still open (per design): reconciliation strategy selection for the four deferred ops, overlay persistence, cross-backend equivalence. Regression panel clean: f16+rw=128 PPL 17.6352, turbo_kv_4b+rw=128 PPL 17.6637, test-flash-attn-lse-merge 9/9.
- Iteration 24: live-server MTP smoke gate (Vega only, sequential, single GPU). `llama-server` with `Qwen3.5-0.8B-Q8_0-MTP.gguf` + `--spec-type mtp -fa on --device Vulkan1 -ngl 99`, prompt `"The capital of France is"`, n_predict=64, temp=0, seed=42. Run A — `--cache-residual-window 0 --cache-type-k f16 --cache-type-v f16` (port 9001): `draft_n=42 draft_n_accepted=22` → 52.4% acceptance, content `" Paris.\nThe capital of France is Paris.\n..."` (5x repeat), `predicted_per_second=123.14`. Run B — `--cache-residual-window 128 --cache-type-k turbo_kv_4b --cache-type-v f16` (port 9002): `draft_n=42 draft_n_accepted=22` → 52.4% acceptance (bit-identical to Run A), content identical character-for-character, `predicted_per_second=65.82`. Bit-identical draft acceptance at temp=0 demonstrates MTP draft path produces the same accept/reject decisions whether or not the overlay is active — the overlay neither corrupts attention reads under speculative decoding nor invalidates after each verify-step's seq_rm. Throughput delta (123 → 66 pps) is the expected overlay-write + overlay-read + reconcile cost on Vega (Vulkan FA, no Vulkan FA_LSE port yet so two-pass falls back to a single-pass shape that still pays the overlay maintenance cost). Step 5 fully closed across all gates (CPU PPL parity, multi-thread invariance, MTP smoke parity).
- Iteration 25: Step 6 pre-work PW1 + PW2 answered; harness gains `--ngl`. **PW1 (cast on Vulkan)**: `ggml_cast` does *not* produce a `GGML_OP_CAST`; it produces a `GGML_OP_CPY` whose dst tensor has a different dtype (`ggml.c:3569`). The recon agent's "GGML_OP_CAST not dispatched on Vulkan" was technically true but moot — there is no separate `GGML_OP_CAST` enum value. Vulkan's `supports_op` for `GGML_OP_CPY` (`ggml-vulkan.cpp:16024`) explicitly lists every dtype pair the merge graph actually uses: F32→{F16,BF16,F32,Q1_0,Q4_0,Q4_1,Q5_0,Q5_1,Q8_0,IQ4_NL,TQ_V_4B,TURBO_KV_4B}, the reverse to F32, F16↔F16, F32↔I32. **Conclusion: no blocker.** **PW2 (`ggml_get_rows` dtype on Vulkan)**: the Vulkan dispatcher (`ggml-vulkan.cpp:9576-9590`) selects pipeline by `dst->type` — supports F16 dst (pipeline_get_rows[T]) and F32 dst (pipeline_get_rows_f32[T]). Not forced to F32 like CPU. So Pass B's overlay-K reorder can request its native dtype directly when Vulkan-bound; if the graph builder emits F32 (matching CPU emit), the follow-on cast is the supported F32→F16 CPY. Either path is dispatchable. **Conclusion: no blocker.** Harness change (llama.cpp `<pending>`): `--ngl N` flag added to `tests/test-turbo-kv-residual-window-harness.cpp` (default 0 = CPU; >0 offloads to active GPU backend). Required for substep 6.10 PPL gate and for these PW probes. Runtime evidence on Vulkan: probe 1 — TinyLlama-Q2_K, `--ngl 99 --rw 0 --append 10` on `GGML_VK_VISIBLE_DEVICES=1` (Vega): clean, `Vulkan0 compute buffer 66.5 MiB`, `HARNESS_OK rw=0 ctx=512 type_k=f16 rw_type_k=auto decoded=10`. Probe 2 — same command with `--rw 128`: hits `ggml_new_object: not enough space in the context's memory pool (needed 667680, available 667312)` inside `build_attn_mha_two_pass` at `ggml_mul` call (the merge graph's `ggml_mul(ctx, VKQ_a, sa)` etc.). 368 bytes short — graph object pool budget is just under what the merge graph requires. **Identical crash on CPU at the same call site** (probe 3, same harness invocation without `--ngl`) — *not Vulkan-specific*. This is a pre-existing graph-budget gap in the two-pass FA path that was never exercised on llama-arch models (Step 5's PPL evidence is on the qwen35 builder; TinyLlama uses `llm_build_llama`, different node-construction pattern). Two findings recorded as open subtasks: (a) graph object pool overflow in `build_attn_mha_two_pass` for llama-arch + n_ctx=512 — must be closed before substep 6.10 PPL gate. (b) The recon agent's PW1 framing was misleading; updated mental model: cast-as-CPY is the correct lens. PW3 (shader variant selection on each device) deferred to substep 6.2 where the dispatcher print lives. Substep 6.1 (lift the FA `ne0 == HSV` assertion) is independently actionable — proceeding next.
- Iteration 26: substeps 6.1 + 6.2 landed in llama.cpp (local — push to master is permission-gated, will batch with 6.3+ as a PR). **6.1**: `ggml_vk_flash_attn` assertion `ne0 == HSV` widened to allow `HSV + 2` when the LSE flag is read from `op_params[4]`. The supports_op refusal at `GGML_OP_FLASH_ATTN_EXT` stays in place — scheduler still routes LSE-flagged FA to CPU until shader math lands. test-backend-ops -o FLASH_ATTN_EXT 2/2 backends pass on Vega: non-LSE dispatch unaffected. **6.2**: `lse_mode` plumbed end-to-end through the FA dispatch path without changing kernel behaviour. Subtle constraint discovered: `vk_flash_attn_push_constants` is **already at the 128-byte Vulkan-spec ceiling** — the plan's "adding one uint32_t (4 bytes) is safe" prediction was wrong. Worked around by packing the LSE bit into `mask_n_head_log2` bit 25 (next free after `SINK_ENABLE_BIT` at bit 24). Macros: `VK_FA_LSE_ENABLE_BIT` (C++), `LSE_ENABLE_BIT` (GLSL). The reduce-shader push-constant struct (`vk_op_flash_attn_split_k_reduce_push_constants`, 24 bytes) had room and gained two explicit fields `lse_mode` + `ne0_dst` (output row stride: `D` standard, `D+2` LSE). GLSL mirrors in `flash_attn_base.glsl` and `flash_attn_split_k_reduce.comp`. Dispatcher `mask_n_head_log2` ORs the bit; `pc2` populates the two new reduce-shader fields. Verified: test-backend-ops -o FLASH_ATTN_EXT 3280/3280 pass on Vega. No behavioural delta — the flag is plumbed but unused until 6.3 (reduce shader LSE branch) and 6.4 (primary shader split_k==1 LSE branch). Plan correction: any future docs that claim the FA push-constants budget has slack should be updated; the budget is exact.
- Iteration 27: substep 6.3 landed in llama.cpp (`0141c040a`, local). `flash_attn_split_k_reduce.comp` gained the LSE branch: when `p.lse_mode != 0` the inverse-L step is skipped, the `O *= L` scaling is skipped, the dst row stride switches from `D` to `p.ne0_dst` (= `D+2`), and a single guarded invocation (`gl_WorkGroupID.y == 0 && tid == 0`) writes `effective_M = max(m_max, sink)` to row `D` and the un-inverted denominator `L_pre_invert` to row `D+1`. Non-LSE behaviour is preserved exactly: the new conditionals collapse to the original code when `p.lse_mode == 0` and `p.ne0_dst == D`. **Self-audit miss**: the same commit had to scrub host-planning references (`PHASE28 substep …`, `PHASE28 iter 24`, `post-PHASE28-iter21`) from llama.cpp source and shaders that I had introduced in the 6.1 + 6.2 commits; user flagged the violation and the `feedback_no_host_concerns_in_code` memory was sharpened to require a per-commit grep self-check. Verified: test-backend-ops `-b Vulkan0 -o FLASH_ATTN_EXT` 3280/3280 pass on Vega after the LSE branch + comment scrub (regression-free). The supports_op refusal still routes any LSE-flagged FA to CPU, so this commit is shape-correct and dormant on Vulkan until 6.4 lifts the refusal alongside the primary shader's split_k==1 LSE branch. Step 6 box stays `[ ]`.
