# CUDA-engine work — peer pickup handoff

**Purpose:** complete state-of-affairs for the CUDA-backend work on Qwen 3.6 27B production, written so a peer engineer can pick this up cold. Read this end-to-end before touching anything; then follow pointers to the PHASE docs + MEMORY.md for detail.

**Status as of 2026-05-24:** Production stable on DFlash via `profiles/qwen36-27b-x2-dflash.sh`. About to switch to **Vulkan backend** — this doc preserves what we learned with CUDA so the Vulkan workstream knows what it's recreating, what it might inherit free, and what it must re-derive.

---

## 1. Hardware + engine

- **Host**: 2× Quadro RTX 6000 (TU102, sm_75, 24 GiB each), 48 GiB aggregate VRAM
- **CUDA**: 13.2, driver 595.58.03, peer-access PCIe (PHB topology; NVLink hardware install pending per `nvidia-smi nvlink --status` showing "inActive" as of session close)
- **Engine**: `ik_llama.cpp` submodule at `production/2026-q2-next`, currently `711212a6`
- **Build**: `cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF && cmake --build build -j 32` from `ik_llama.cpp/`
- **Production binary**: `ik_llama.cpp/build/bin/llama-server`

## 2. Production profile (what runs now)

`profiles/qwen36-27b-x2-dflash.sh` (host-side, host symlinked at `profiles/active.sh`):

- 2-slot (`--parallel 2`) × 256K context-per-slot (`--ctx-size 524288`)
- DFlash speculative decoding on (`--spec-type dflash`), drafter at `qwen36-27b-dflash-f16.gguf`, `--draft-max 4`
- Q4_0 KV cache + Hadamard rotation (`--cache-type-k q4_0 --cache-type-v q4_0 --k-cache-hadamard --v-cache-hadamard`)
- Prompt cache 40 GiB RAM, 64 checkpoints (`--cache-ram 40960 --ctx-checkpoints 64`)
- Graph-split across both GPUs (`--device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1`)
- F16 lm_head, FA on, port 8080

Run via systemd: `systemctl --user start llama-server.service` reads `profiles/active.sh`.

## 3. Tier history (one-line each)

Detail in `PHASE_*.md` files at repo root. Listed in landing order:

| tier | status | what it closed | doc |
|---|---|---|---|
| T1-T2 | CLOSED | n-stream KV foundation + Bug C identification | `PHASE_NSTREAM_KV_PERF.md` |
| T3 | CLOSED | unified-stream dispatch + PSKV singlewarp + framing B | `PHASE_NSTREAM_KV_PERF.md` |
| T3.5 | CLOSED | PSKV ILP recovery: TG +2.95%, PP +9.17%, ncu per-CTA −32.7% | `PHASE_NSTREAM_KV_PERF.md`; mem `project_pskv_ilp_recovery_landed` |
| T4 | CLOSED | chunked-prefill admission landed; T4.7 perf gate FAIL (C0=multi-slot kernel saturation; staggered ≤ steady always) | `PHASE_NSTREAM_KV_PERF.md`; mem `project_t4_bundle_a_landed` |
| T5.1-T5.8 | CLOSED | paged ADDRESSING (allocator + write/read/K-shift/defrag/trace + bake-out) | `PHASE_NSTREAM_KV_PERF.md`; mem `project_t5_*` family |
| T5.9 | CLOSED | paged BACKING — block-major K/V tensor; user `--kv-pool-blocks N` override delivers ctx ≥ 1M feasibility | `PHASE_NSTREAM_KV_PERF.md`; mem `project_t5_9_paged_backing_closed` |
| T6.0 | CLOSED | re-verified vLLM gap = 6.37× (was wrongly cited 5.84×); locked T6.0.b cell schema | `PHASE_T6_CHARACTERISATION.md` |
| T6.1 | CLOSED | binary ablation matrix at gate0 NP=8 + T6.6 SEGV root-caused + fixed | `PHASE_T6_CHARACTERISATION.md`; mem `project_t6_1_matrix_closed_with_segv` + `project_t6_6_segv_root_caused_and_fixed` |
| T6.2 | CLOSED | nsys + ncu kernel attribution at production decode | `PHASE_T6_CHARACTERISATION.md` |
| T6.3 | CLOSED | DFlash deep-dive (net-negative across all 4 axes at gate0) | `PHASE_T6_CHARACTERISATION.md`; mem `project_t6_3_dflash_deep_dive_closed` |
| T6.3.j | CLOSED | 1M-ctx perf ceiling analysis kills the "1M Yarn + MTP" production candidate | `PHASE_T6_CHARACTERISATION.md`; mem `project_t6_3_j_1m_ctx_ceiling` |

## 4. Engineering bugs fixed (commits to know)

These submodule commits are load-bearing for any future work:

| commit | what it fixes | when to invoke |
|---|---|---|
| `3ee7816f` | T6.6 — `kv_cache_init`'s `n_layer` truncation vs `llm_build_context::n_layer` mismatch; K-shift + defrag loops read past k_l vector end; under DFlash heap layout, byte past end was `0x1ea30` (non-null), survived nullptr skip, SEGV'd on `->extra`. Fix: bound by `std::min(n_layer, k_l.size())`. | Always-on. NPC binding. |
| `a69f19de` | T6.3 1M-overnight prep — `delta_net::delta_net()` was setting `save_per_step_states = save_per_step_ssm && batch.n_tokens > 1`; per-step buffers sized at spec-ckpt-init for `max_tokens = drafted.size() + 1 = 2`; prefill ubatches overflow `ggml_view_2d` at `build_layer_attn_linear_core` line 631. Fix: gate on `batch.n_tokens <= per_step_max_allocated`. | Required for any MTP build. Complements PHASE45 D10 multi-slot guard. |
| `711212a6` | T6.3 second-overnight prep — server `process_batch_tokens` was using single-row `llama_set_draft_input_hidden_state(...)` but dst tensor `inp_mtp_states` shape is `(n_embd, n_tokens)`; `prepare_mtp_graph_inputs` then memcpy'd `n_embd × n_tokens × sizeof(float)` bytes from a `n_embd`-only buffer → SIGSEGV in libc AVX2 `vmovdqu`. Fix: use `_multi` variant. | Required for any MTP-with-`batch.n_tokens > 1` path. Backend-agnostic (server-side code). |

All three are in the submodule's `production/2026-q2-next` branch; parent repo bumps committed and pushed.

## 5. Performance characteristics (CUDA-measured, may not carry to Vulkan)

Measured this session under CUDA + sm_75 + PCIe peer-access (no NVLink):

| measurement | value | doc / data |
|---|---:|---|
| Production decode (DFlash, gate0 NP=8 mixed prompts) | **11.03 t/s** | T6.1 matrix prod-baseline cell |
| Vanilla decode (no DFlash, gate0 NP=8) | **20.45 t/s** | T6.1 matrix no-dflash cell |
| Vanilla decode (single-slot 1M+YaRN) | **17.04 t/s @ empty ctx** | data/t6.3-1m-smoke-vanilla-* |
| MTP decode (single-slot 1M+YaRN, post-fix) | **13.71 t/s @ empty ctx**, 77% acceptance | data/t6.3-1m-yarn-mtp-postfix-* |
| **Prefill at 1M+YaRN+MTP, 350K-position chunks** | **9.7 t/s** (pathological — checkpoint thrash + DRAM-bound forward) | T6.3.j evidence |
| nsys kernel attribution at production NP=2 no-spec | `mul_mat_q_split_k<Q4_0>` 31%, NCCL AllReduce 26.5%, PSKV 3.2% | T6.2; data/t6.2-nsys-prod-* |
| nsys kernel attribution under DFlash NP=8 dm=4 | `mul_mat_f16_pinned_kernel_wmma` (drafter) **17.8%**, target Q4_0 13.3%, AllReduce 26.5%, PSKV 2.1% | T6.3 axis 4; data/t6.3-nsys-dflash-* |
| ncu kernel deep-dive on dominant `mul_mat_q_split_k<Q4_0>` | DRAM 44.3%, compute 17.5%, occupancy-bound at 25% (shared-mem-limited, 40-45 KiB/block) | T6.2.b; data/t6.2-ncu-* |
| DFlash per-prompt acceptance (axis 1) | range **0.392** (King Lear prose) → **0.808** (haiku), mean 0.529 | T6.3 axis 1 |
| `draft_max` sweep optimum | dm=2 at 11.58 t/s (+5% over dm=4 default) but still −43% vs no-DFlash 20.45 | T6.3 axis 2 |
| NP sensitivity sweep | DFlash net-negative at ALL NP ∈ {1,2,4,8}; −37% at NP=1, −46% at NP=8 | T6.3 axis 3 |
| Bandwidth ceiling for Qwen3.6-27B hybrid (16/64 full-attn × Q4_0) | 322 t/s @ 262K, 156 @ 524K, 80 @ 1M (PEAK; real ~40%) | T6.3.j math |

### What carries to Vulkan, what doesn't

| category | carries? | reason |
|---|---|---|
| **Bandwidth ceilings** | YES | DRAM bandwidth is hardware-level, backend-agnostic. Vulkan compute shaders also read KV from DRAM. |
| **Per-prompt acceptance (DFlash 27B drafter)** | YES | Model-pair quality is model-level, backend-independent. Community-reported poor at this 27B pair; not a CUDA artefact. |
| **MTP DeltaNet bugs we fixed** | YES (likely) | Both fixes are in `src/llama-delta-net.cpp` + `examples/server/server-context.cpp` — backend-agnostic graph builder + server-side code. Vulkan backend uses the same paths. |
| **NCCL AllReduce 26.5%** | NO | NCCL is CUDA-only. Vulkan multi-GPU uses GPUDirect/different sync primitives or none. |
| **nsys + ncu attribution** | NO | CUDA profiling tools. Vulkan needs RenderDoc / Nsight Graphics / `radv` perfetto / similar. |
| **`mul_mat_q_split_k` shared-mem analysis** | NO | CUDA kernel. Vulkan has its own `mul_mat_vec_q` / similar shaders with different occupancy / shmem layout. |
| **Q4_0 dispatch path** | partial | Q4_0 weight format unchanged; how the GPU reads it differs (CUDA tile-load vs Vulkan compute shader). |
| **PSKV singlewarp ILP recovery** | NO | CUDA-specific kernel; Vulkan has no PSKV equivalent (its FA path is different). |
| **PHASE45 D10 spec-ckpt PER_STEP/GPU_FALLBACK dispatch** | YES | Logic in the kv-cache layer above any backend kernel. |

## 6. T6.3 deep-dive summary

DFlash is the only feature with a material effect at gate0 NP=8 production-shape workload, and it's **net-negative**:

- 4-axis characterisation: per-prompt acceptance (range 0.4-0.8, content-dominated); draft_max sweep (dm=2 sweet spot but still −43% vs no-DFlash); NP sensitivity (net-negative at every measured NP); kernel attribution (drafter forward = 17.8% of GPU time, not amortised by 0.42-0.53 acceptance).
- Independent confirmation: Paterson @ RTX 3090 Ti (sm_86 Ampere, single GPU) measured Qwen3.6-27B + DFlash collapse from 46.9 t/s @100 output → 30.1 t/s @ 2000 output; production switched to MTP. Community reports 12-30% acceptance at n_spec=8-15.
- vLLM measured the same pair: vanilla 154 t/s, DFlash 35 t/s — **4.4× penalty** vs vLLM vanilla (worse than our 1.85× penalty). DFlash hurts vLLM MORE, partly because vLLM disables CUDA graphs for DFlash dynamic shapes.

**The parking-from-DFlash architecture** (single-slot + 1M ctx + YaRN + MTP + cache + transparent queue) was scoped and attempted; failed:

- Two MTP bugs found and fixed (a69f19de + 711212a6) — these are real and ship in the engine going forward.
- But the 1M ctx + MTP + YaRN combo runs at 9.7 t/s prefill on this hardware. Per the bandwidth math, **100+ t/s at 1M ctx is not on the table for this hardware** even with NVLink. The user-stated 100+ t/s constraint means 1M ctx is infeasible until tensor-core matmul + NVLink (T7 territory).
- Recalibrated parking ladder: 262K native (comfortable, 130 t/s pre-NVLink) or 524K with YaRN factor=2.0 (borderline, ~75 t/s post-NVLink). Production stays on DFlash until measured.

## 7. Open subtasks (named per CLAUDE.md §4)

Each is a concrete next step a peer can pick up. References to detail in PHASE docs + memory entries.

### From T6.3 DFlash deep-dive
- **T6.3.b** — re-measure DFlash axes 2+3 after NVLink install. AllReduce currently 26.5%; NVLink reduces small-message latency ~10×. Drafter share + dm=2 sweet spot may shift.
- **T6.3.c** — characterise DFlash on bench-t3.8-m3-shape (identical short prompts) to bound the upper acceptance ceiling. Quantifies the workload range where DFlash net-helps.

### From T6.3 MTP-swap validation
- **T6.3.g** — extract any unresolved coredumps if intermittent crashes resurface (the two we caught are fixed; future ones likely point to different bugs)
- **T6.3.h** — overnight retry once a viable architecture is chosen; the 1M-ctx version is NOT the right next step per T6.3.j

### From T6.3.j perf-ceiling analysis (LOAD-BEARING for parking decision)
- **T6.3.k** — measure prefill t/s at 262K + 524K with the (a69f19de + 711212a6)-fixed build. ubatch sweep at the chosen target. Decides whether 524K passes the 100+ t/s gate or whether we stay at 262K native.
- **T6.3.l** — post-NVLink re-measure T6.2 + T6.3.k. AllReduce drops from 26.5% to ~5% with NVLink; re-run to update the cost surface.
- **T6.3.m** — long-ctx prefill nsys characterisation. Identify dominant kernels at 350K-position ubatch. Informs T7 matmul kernel rewrite priority.
- **T6.3.n** — `--ctx-checkpoints-interval` investigation. Default 512 is wrong for long-prompt prefill (creates a 152 MiB checkpoint per 512 tokens; thrash at long ctx). Default likely needs raising to a value that exempts intra-prefill checkpointing.

### Pre-existing (not addressed this session)
- T6.4 T4 chunked-prefill admission characterisation
- T6.5 T5.9 paged BACKING characterisation
- T6.7 per-slot-kv FA dispatch characterisation
- T6.8 Hadamard accuracy characterisation
- T6.9 T3 unified-stream dispatch characterisation
- T6.10 closure synthesis

## 8. How to verify production health (cold-start)

```bash
cd /home/llm/yarn-agentic

# 1. Confirm git + submodule state
git log --oneline -3
cd ik_llama.cpp && git rev-parse --short HEAD  # expect 711212a6 or newer
cd ..

# 2. Confirm clocks locked (binding for any bench)
bash scripts/gpu-clocks.sh status  # expect 1455 MHz Enabled both GPUs

# 3. Production smoke
systemctl --user is-active llama-server.service
curl -fsS http://127.0.0.1:8080/health  # expect {"status":"ok"}

# 4. Determinism check (~10 min)
bash scripts/verify-production-determinism.sh
# expected: PASS at NP={1,2,4,8}

# 5. Production bench (T6.2 reference shape)
bash scripts/bench-t3.8-m3.sh
# reference numbers in data/t3.8-m3-*/

# 6. T6 audit
grep -E "^- \*\*T6\.3\." PHASE_T6_CHARACTERISATION.md  # should list 8 named subtasks
ls /home/llm/.claude/projects/-home-llm-yarn-agentic/memory/MEMORY.md  # 128+ entries
```

## 9. Build commands (CUDA reference)

```bash
# ik_llama.cpp (production engine)
cd /home/llm/yarn-agentic/ik_llama.cpp
cmake -B build -G Ninja \
      -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF
cmake --build build -j 32

# llama.cpp (reference / not in production)
cd /home/llm/yarn-agentic/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build -j 32

# Debug rebuild (with symbols for gdb)
cd /home/llm/yarn-agentic/ik_llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -G Ninja
cmake --build build -j 32 --target llama-server
```

## 10. Where to find detail beyond this doc

- **PHASE_T6_CHARACTERISATION.md** — Tier 6 measurement work (T6.0 → T6.3.j). Most active doc for current state.
- **PHASE_NSTREAM_KV_PERF.md** — Tier 2-5 history; paged KV; spec-ckpt PER_STEP design; PHASE45 D10 multi-slot dispatch guard.
- **PHASE_DFLASH_MULTISLOT.md** — DFlash multi-slot evolution (Phases 1-6); per-slot scratch sizing; cb_eval demux.
- **PHASE_ASYNC_REDUCE.md** — async reduce work.
- **PHASE_MMQ_Q4_0_AR16.md** — Q4_0_AR16 Hadamard-rotated weight format.
- **PHASE_NP_DETERMINISM_CLOSED.md** — cross-NP byte-identity contract + ALGO0 cuBLAS algo pin.
- **MEMORY.md** (repo root, 489 KB, append-only) — project-scope decisions, incidents, scoping changes.
- **`/home/llm/.claude/projects/-home-llm-yarn-agentic/memory/`** (host-local, not in repo) — agent's per-incident memory entries with detail beyond what's in PHASE docs. 129 files; ~720 KB. `MEMORY.md` there is the index. Contains all `project_t6_3_*`, `feedback_*`, `reference_*` continuity. If transferring hosts, rsync this dir (see TRANSFER.md §2).
- **`data/`** — bench outputs + nsys traces + per-cell JSONs. T6.x cells follow the schema in `specs/t6-characterisation-cell.allium`.
- **`specs/`** — Allium specifications. `allium check` runs in CI; specs are commit-gated.
- **`scripts/`** — bench drivers + analysis tools. Notable: `verify-production-determinism.sh` (NPC contract), `bench-t3.8-m3.sh` (perf gate), `gpu-clocks.sh` (clock lock).

## 11. Vulkan-switch implications (what changes)

**What stays the same:**
- Same model files, same Q4_0 KV cache format, same Hadamard rotation
- All host-side / server / kvcache / spec-ckpt code (the 3 commits above ALL apply)
- All specs (`specs/*.allium`)
- All bench harnesses
- NPC contract (cross-NP byte-identity at the model's output level)

**What needs re-doing under Vulkan:**
- Build: `cmake -DGGML_VULKAN=ON` instead of `-DGGML_CUDA=ON`. Note: Vulkan target needs SPIRV compiler chain available (`glslc` or `glslangValidator`).
- Production profile: same flags except `--device CUDA0,CUDA1` becomes whatever Vulkan device selection syntax is (`--device VULKAN0,VULKAN1`?). Verify against `llama-server --help`.
- Per-kernel attribution (T6.2 equivalent): can't use nsys. Use RenderDoc + Vulkan timestamps, or `radv` perfetto trace, or `vk_radv_thread_trace_dir` env for Mesa/RADV. Re-run at production decode shape to get the Vulkan-side cost surface.
- Production decode baseline: re-measure at gate0 NP=8 mixed prompts (T6.1 matrix) to get the new t/s numbers. T6.1 verdicts (DFlash net-negative, Hadamard no-op, defrag no-op) are at the cell-level workload, not kernel-level, so they likely hold direction-wise but the absolute numbers will change.
- 1M-ctx bandwidth ceiling math: still holds (DRAM-bound) but **the Vulkan compute-shader-saturation regime might be different** (e.g., different occupancy / register pressure / wave size). The ceilings above are upper bounds; Vulkan-real efficiency could be higher OR lower.
- AllReduce: NCCL is CUDA-only. Vulkan multi-GPU needs different sync (or single-GPU operation per layer with explicit `vkCmdCopyBuffer` cross-GPU). Re-measure cross-GPU sync cost as a first-class question.

**What's likely BROKEN under Vulkan and needs investigation:**
- Q4_0 matmul kernels — `mul_mat_q_split_k<Q4_0>` is CUDA-specific. Vulkan ggml has its own `mul_mat_vec_q` kernels. Coverage of Q4_0 + Hadamard rotation under Vulkan needs verification.
- PSKV singlewarp — CUDA-only. Falls back to `ggml_flash_attn_ext` (Vulkan's standard FA) under Vulkan. Performance impact unknown.
- DeltaNet kernel — Qwen3.6 hybrid layers. Verify Vulkan support; potential CPU fallback at long ctx.
- MTP / nextn_predict_layers graph composition — backend-agnostic for the graph builder, but kernel-level matmuls inside MTP layers go through Vulkan kernels.
- Memory pool / `--kv-pool-blocks` override — T5.9 paged BACKING uses CUDA `ggml_backend_cuda_buffer_type_split` for graph-split storage. Vulkan equivalent may differ.

**Suggested Vulkan workstream entry:**

1. Build first, confirm `llama-server -m <small-model.gguf>` boots under Vulkan (no graph-split) and produces sensible tokens. Don't touch production until smoke passes.
2. Run `bash scripts/verify-production-determinism.sh` with the Vulkan-built binary at single GPU first. If NPC fails, that's the first investigation.
3. Multi-GPU: get a baseline cross-GPU sync working. The CUDA path used graph-split + tensor-split 1,1 + NCCL AllReduce. The Vulkan equivalent needs explicit design.
4. Re-derive a T6.1-equivalent matrix under Vulkan to confirm DFlash net-negative direction holds.
5. If the absolute numbers under Vulkan look promising (e.g., Q4_0 matmul kernel has better occupancy than CUDA's shared-mem-bound 25%), that ALONE could change the parking decision. Don't forecast — measure.

---

## Quick navigation summary

| if you need | go to |
|---|---|
| "what does production run" | §2 above + `profiles/active.sh` |
| "what's the latest measured number for X" | §5 above + `data/t6.*` cells |
| "why did we park DFlash from production" | T6.3 closure in `PHASE_T6_CHARACTERISATION.md`; memory `project_t6_3_dflash_deep_dive_closed` |
| "why isn't 1M ctx the production target" | T6.3.j in `PHASE_T6_CHARACTERISATION.md`; memory `project_t6_3_j_1m_ctx_ceiling` |
| "what's the next concrete task" | §7 above (T6.3.k is the highest-priority unconditional) |
| "how do I verify nothing's broken" | §8 above |
| "what's CUDA-specific and won't carry to Vulkan" | §5 sub-table + §11 |
| "what's the engine commit history" | `cd ik_llama.cpp && git log --oneline -30` |
| "what's the project-level decision log" | `MEMORY.md` (repo root) |
| "what's the per-incident agent memory" | `/home/llm/.claude/projects/-home-llm-yarn-agentic/memory/` |

If something here is wrong or stale, the canonical sources of truth are: PHASE docs for tier-level state, MEMORY.md for project decisions, the auto-memory entries for per-incident detail, and the bench `data/` for measured numbers. This doc is the synthesis; trust the sources when they disagree.
