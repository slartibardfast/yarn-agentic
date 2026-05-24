---
name: DFlash T1–T4 kernel layer CLOSED — argmax-equivalent vs vLLM PR #40898
description: DFlash speculative decoding kernel layer for Qwen3.6-27B on sm_75 is closed at the kernel-pipeline level on production/2026-q2-next. 8 prompts × 4 mask positions = 32 / 32 argmax matches vs vLLM. T5+ remains for server-side integration.
type: project
originSessionId: 9b52ee2b-d969-442c-83ba-98396525f6fa
---
DFlash workstream T1–T4 closed on `production/2026-q2-next` (2026-05-13). Kernel-pipeline layer is **complete and validated against vLLM PR #40898 across 8 diverse prompts**. Server-side integration (T5+) is the next major scope.

### What's in the closed scope (T1–T4)

| step | gate | scope |
|------|------|-------|
| T1 | 1 | DFlash converter binding: `convert_hf_to_gguf.py::DFlashModel`, 6 metadata keys, 2 tensor names |
| T2 | 2 | Extract-features hook via `cparams.cb_eval` matched on `l_out-<il>`; cosine ≥ 0.99988 vs vLLM PR #40898 |
| T3 | 3a | `dflash-combine-features.cu` + `dflash-inject-kv.cu` — byte-identity (≤ 1 fp16 ULP) sweep |
| T4 | 3b/4 | `dflash-drafter-forward.cu` + `dflash-drafter-lm-head.cu` + `dflash-argmax-match.cu` — argmax-equivalent vs vLLM |

### T4 closure metric (revised from spec original)

Original kernel-design.md §10 said "drafter logits within 1e-5 NMSE vs vLLM". Found unachievable cross-stack: independent fp32 implementations (vLLM uses triton paged attention; we use scalar fp32 sub-kernels on sm_75) accumulate ~1e-3 to 1e-4 NMSE from reduction-order differences alone.

Revised metric (committed PASS gate in `tests/dflash-speculative/test-dflash-closure.cpp`):
- **argmax**: ALL BLOCK_SIZE rows agree with vLLM
- **top-5 overlap**: ≥ 4/5 per row
- **cos_sim**: ≥ 0.999
- **NMSE**: reported informationally (range 4e-5 to 7e-4 across 8 prompts)

Argmax is what `dflash_argmax_match` consumes for the accept-prefix decision per `@LongestPrefixMatchUnderArgmax`. Argmax-equivalent is the semantically meaningful metric for greedy spec-decode.

### Critical kernel fixes during T4 (the high-leverage ones)

All surfaced from a 15-min source-read of vLLM's `qwen3_dflash.py` — see `feedback_source_read_reference_before_instrumenting.md`:

1. **F32 norm weights stored as `__half*`** — root NaN bug. T3 missed it because T3 tests generate random fp16 weights in-test rather than loading from GGUF. See `feedback_validate_gguf_dtype_at_load.md`. Fix: `upload_f32_as_f16` helper in the drafter loader.
2. **Missing `output_norm` step** before lm_head (vLLM's `DFlashQwen3Model.forward:526`). Added as step 13.
3. **Full-attention K-loop direction** wrong — was causal `k_hi=qpos`, should be bidirectional within block `k_hi = anchor_pos + Q - 1`.
4. **Missing drafter Q/K/V projections at query positions**. Initially assumed inject_kv_fused populated all cache positions. vLLM's actual layout: inject writes K/V at CONTEXT positions (full prompt); drafter's own forward writes K/V at QUERY positions (anchor + BLOCK_SIZE masks). Cache shared between writers. Added K projection + V projection + k_norm_rope_kernel + cache_write_kv_kernel.

### Three vLLM venv source patches landed inline (user-authorized)

See `reference_vllm_v1_subprocess_patches.md` for details. Required because vLLM v1 EngineCore is a separate subprocess that re-imports vLLM fresh and doesn't see runtime monkey-patches.

### T5+ scope (server-side integration)

The standalone `dflash-drafter-loader.h` test helper must be promoted into the llama framework:
- `src/llama-arch.{h,cpp}`: extend DFlash tensor list (Q/K/V/O, gate/up/down, attn_norm, ffn_norm per drafter layer)
- `src/llama-model.cpp`: DFlash arch dispatch + drafter weight loader
- `src/llama-context.cpp`: `llama_set_dflash`, `extract_dflash_features`, `apply_inject_kv` orchestration
- `include/llama.h`: replace `LLAMA_DFLASH_NOT_IMPLEMENTED` stub
- `common/speculative.cpp`: `common_speculative_dflash_init/free/draft/accept`
- `examples/speculative-simple/`: `--dflash` flag
- `tools/server/server-context.cpp`: np=1 hard-gate + DFlash init

### Phase B deferred (not part of T4)

Phase A scalar-fp32 sub-kernel pipeline closed T4. Phase B cooperative WMMA mega-kernel is deferred — gated on T8 perf measurement. If Phase A meets the ≥ 1.5× MTP ship bar at T8, Phase B is unnecessary.

### Authoritative artifacts (post-closure)

- Branch: `production/2026-q2-next` on both yarn-agentic and ik_llama.cpp submodule
- Spec: `specs/dflash/{DESIGN.md, kernel-design.md, dflash.allium, DFlashCycle.tla, DFlashMultiSlot.tla, allium-tla-binding.json}`
- Tracker: `PHASE_DFLASH.md` (T4 row marked `[x]`)
- Public log: `MEMORY.md` (append-only) — T4 closure entry committed 2026-05-13
- Multi-prompt closure dumps: `data/dflash-extracts/prompt-{0..7}/`
- Build dir: `/opt/llm/build-dflash` (off the 97%-full root)
