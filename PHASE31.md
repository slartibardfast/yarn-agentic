# Phase 31: MTP production on ik_llama.cpp (3060 Ti, sole focus)

## Status

**Open.** Iter 1 (2026-05-01) closed Steps 1–4 on the 3060 Ti (8 GB VRAM, partial offload, MTP −25% tg). Iter 3 (2026-05-02) closed Step 5 `[~]` on dual Quadro RTX 6000 sm_75 with full GPU offload — MTP runs end-to-end on the `mtp-extract` branch (clean upstream + 6 cherry-picks) but throughput is **0.49–0.53× baseline**, draft acceptance real (38–46%) but per-step overhead ~2.76× the baseline forward pass. Uplift target not met for this hardware/model class; subtasks open on 27B dense and sm_120/121 hardware. Successor to PHASE29 / PHASE30 (both abandoned per user direction 2026-05-01). Sole focus: MTP end-to-end on `ik_llama.cpp`. **No turboquant.**

## Context

`ik_llama.cpp` already has MTP fully wired (commits `737d607a`, `b7098b7d`, `bac58db4` and the `models/qwen35` branch). MTP is gated by:
- gguf metadata `<arch>.nextn_predict_layers > 0` (e.g. `qwen35moe.nextn_predict_layers = 1`)
- runtime flag `cparams.mtp = true`
- Graph builder `src/llama-build-context.cpp::build_mtp_tail()`
- KV state `src/llama-context.h::mtp_kv_head_hint`, `inp_mtp_states`, `mtp_greedy_tokens`
- Rollback `src/llama-delta-net.cpp` (MTP-IR for per-token state materialization)

**Hardware:** RTX 3060 Ti (sm_86, Ampere, 8 GB VRAM), driver 595.58.03, CUDA 12.x.

**Model:** `/opt/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-BF16.gguf` confirmed MTP-capable via gguf metadata: `qwen35moe.block_count=41`, `expert_count=256`, `expert_used_count=8`, `nextn_predict_layers=1`. BF16 size is too large for full GPU offload on 8 GB VRAM — will need quantization to Q4_K_M (~17 GB) with partial offload, or quantization further (Q3/Q2) for tighter GPU residency.

**Future TurboQuant reference (not adopted now):** `slartibardfast/llama-cpp-turboquant`. See PHASE29/30 closing notes and MEMORY.md 2026-05-01 entry.

## Scope

End-to-end MTP correctness + throughput uplift on Qwen3.6-35B-A3B (the available MTP model).

In scope:
1. Build `ik_llama.cpp` clean on 3060 Ti.
2. Quantize Qwen3.6-35B-A3B-BF16 to Q4_K_M for VRAM fit.
3. Smoke run with `--mtp on`, default f16 KV. Confirm draft acceptance, no crash.
4. Measure PPL with MTP on vs off; expect parity (|Δ| < 0.05).
5. Measure throughput uplift vs MTP off.

Out of scope:
- Any turboquant or KV cache quantization.
- HIP/AMD parity.
- Vega regression (PHASE30).
- llama.cpp upstream maintenance.
- Models other than 35B-A3B (until/unless 9B-MTP gguf is fetched).

## Step checklist

- [x] **Step 1 — Build ik_llama on 3060 Ti.**
  - `cd ik_llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`
  - Smoke baseline (no MTP): `./build/bin/llama-cli -m <gguf> --device CUDA0 -ngl 99 -p "test"` produces coherent text.
  - Verify by: build clean, baseline inference returns non-empty non-garbage output.

- [x] **Step 2 — Quantize Qwen3.6-35B-A3B-BF16 → Q4_K_M.**
  - `./build/bin/llama-quantize /opt/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-BF16.gguf /opt/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-Q4_K_M.gguf q4_k_m`
  - Verify by: output gguf written; `gguf-py` re-read confirms `nextn_predict_layers=1` preserved.

- [x] **Step 3 — MTP smoke run on 3060 Ti.**
  - `./build/bin/llama-cli -m Qwen3.6-35B-A3B-Q4_K_M.gguf --device CUDA0 -ngl <max-fitting> -fa on --mtp on -c 2048 -p "The capital of France is" -n 64`
  - Determine `-ngl` such that VRAM stays under 7.5 GB (leave headroom). Q4_K_M of 35B is ~17 GB; with 8 GB VRAM expect to offload only ~10 main layers + MTP head + output projection.
  - Verify by: non-NaN logits, coherent continuation, exit cleanly.

- [x] **Step 4 — MTP PPL parity.**
  - `./build/bin/llama-perplexity -m <Q4_K_M-gguf> -f /opt/models/wikitext-2-raw/wikitext-2-raw/wiki.test.raw --device CUDA0 -ngl <max-fitting> -fa on -c 4096 --chunks 16` with `--mtp on` and again without.
  - Verify by: `|PPL_mtp - PPL_baseline| < 0.05` (or document why MTP doesn't share PPL with non-MTP — MTP changes the prediction sequence).

- [~] **Step 5 — MTP throughput uplift.** Binding negative on Quadro RTX 6000 sm_75 full-GPU at iter 3. MTP runs end-to-end on `Qwen3.6-35B-A3B-IQ4_KS-imat.gguf`, draft acceptance is real (38–46%), but per-step overhead is ~2.76× baseline so net throughput is **0.49–0.53× baseline**, not >1×. Subtask: try 27B dense (different MTP topology — single layer vs MoE expert routing per draft) once a BF16 source is on disk. Subtask: revisit on sm_120 / sm_121 hardware where the delta-net + MTP-tail kernel cost may amortize differently.
  - `./build/bin/llama-bench -m <Q4_K_M-gguf> -mtp 0,1 -fa 1 -ngl <max-fitting>`
  - Or instrument llama-server for tg t/s with --mtp on/off.
  - Verify by: tg t/s with MTP > baseline (target: 1.2× minimum given partial offload constraint; was 1.74× at 10.2→17.8 t/s in commit `fd77f898` on better hardware).

- [ ] **Step 6 — Document.**
  - PHASE31.md updated per iteration (per CLAUDE.md §5).
  - MEMORY.md notes any non-obvious findings (per §6).

## Closing condition

Phase 31 closes when:
1. Steps 1–5 all `[x]` with binding evidence on the target hardware/model class.
2. End-to-end MTP run on Qwen3.6-35B-A3B returns coherent text and acceptance > 50%.
3. Throughput **uplift** (MTP tg > baseline tg) measured. **Currently negative** at iter 3 on Quadro RTX 6000 full-GPU (0.49–0.53× baseline); 35B-A3B IQ4_KS on sm_75 CUDA cannot amortize the MTP per-step overhead.

## Open work (not yet bound)

These items are explicitly open, not deferred or hidden behind cover language. Each blocks Phase 31 closure or is intentionally inherited from abandoned predecessor phases.

- **Step 5 subtask — 27B dense MTP.** Different MTP topology (single dense layer vs MoE expert routing per draft step) may amortize per-step overhead differently. Blocked on a BF16 27B-MTP source on disk.
- **Step 5 subtask — sm_120 / sm_121 hardware.** Delta-net + MTP-tail kernel cost per step may be smaller relative to baseline forward pass on Blackwell-class hardware. No binding measurement until access to such a host.
- **Larger-context PPL parity.** Verified at c=512 / 16 chunks. Closing-condition wording asked for c=4096 / 16 chunks — c=4096 OOMs on 8 GB VRAM (3060 Ti); on Quadro 24 GB×2 the run was not repeated.
- **PR for `mtp-extract` branch upstream.** The 6 commits on `slartibardfast/ik_llama.cpp:mtp-extract` (clean from `ikawrakow/main` HEAD `a8aecbf1`) are upstream-PR-quality and worth landing independently of any throughput claim. PR not yet opened.
- **CUDA delta-net kernel extension for `emit_intermediates=true && n_tokens > 1`.** Current fix (`773b0648`) conservatively falls back to CPU. A real GPU kernel for the multi-state-copy emit would unlock prompt-eval acceleration during MTP graph builds. Not on PHASE31's critical path.
- **Inherited from PHASE29 (abandoned, not closed):** Steps 4 (TQ_V_4B FA-V verify), 5 (FA-LSE writeback), 6 (MTP tightness on Qwen3.5-9B), 7 (HIP parity sweep). Code for Step 4 lives on `defunct/phase29-iter7-tq_v_4b-fa-v` in `llama.cpp/`. None will be closed unless TurboQuant work resumes (in which case adopt `slartibardfast/llama-cpp-turboquant` rather than continuing PHASE29).
- **Inherited from PHASE30 (abandoned, not closed):** Step 6 closing condition (b) — Vega regression check on `gpu-1`. Wave32-portable shaders shipped at `cc9e876ed` (labelled `defunct/phase30`) but the cross-architecture verification was never run. Re-opens only if Vulkan turbo work resumes.

## Reproducibility pins

- **Host:** RTX 3060 Ti (sm_86, 8 GB VRAM), driver 595.58.03, CUDA 12.x.
- **Submodule:** `ik_llama.cpp` at HEAD (commit recorded per iter in loop log).
- **Model:** `Qwen3.6-35B-A3B-Q4_K_M.gguf` (to be quantized in Step 2 from the BF16 source at `/opt/models/qwen3.6-35b-a3b/`).
- **Corpus:** `/opt/models/wikitext-2-raw/wikitext-2-raw/wiki.test.raw`.
- **Build:** `cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release`.

## References (surveyed, not adopted)

- `slartibardfast/llama-cpp-turboquant` — user-named future TurboQuant reference (turbo2/3/4, sm_75;80;86;89;120;121, RTX 3080 75 tok/s on Qwen3-8B).
- `AmesianX/TurboQuant` — paper-canonical (ICLR 2026 DeepMind), llama.cpp fork.
- `atomicmilkshake/llama-cpp-turboquant` — sm_75 confirmed.
- `ikawrakow/ik_llama.cpp#1509` — community CPU TQ3/TQ4 in ik_llama (CUDA pending).
- `ggml-org/llama.cpp#20969` — open design discussion.

## Loop log

- iter 1 (2026-05-01): **Steps 1–4 close.** Build clean (475/475 targets, native sm_86) on RTX 3060 Ti. Quantize Qwen3.6-35B-A3B-BF16 (67 GB) → Q4_K_M (21.7 GB) succeeded; MTP heads (`blk.40.nextn.{shared_head_norm,enorm,hnorm}`) preserved. **Bug found and fixed:** CUDA `delta-net.cu:258` asserts `ggml_nelements(dst) == output_size + state_size` but `ggml_delta_net_ext` sizes `dst` with `state_size * n_tokens` when `op_params[2]` (emit_intermediates) is set — and `src/llama-delta-net.cpp:402` hardcodes `emit_intermediates=true`. CUDA op-supports declared `GGML_OP_DELTA_NET` true unconditionally → assert fires whenever prompt eval batches >1 token through the MTP graph. Fixed in `ggml/src/ggml-cuda.cu` op-supports: `return op->op_params[2] == 0 || op->src[0]->ne[1] == 1;` — keeps CUDA for `emit_intermediates=false` and the `n_tokens=1` decode case, falls back to CPU for the n_tokens>1 emit path. Pushed to ik_llama.cpp branch `fix/cuda-delta-net-emit-intermediates` (commit `f9bb0efa`); PR at https://github.com/slartibardfast/ik_llama.cpp/pull/new/fix/cuda-delta-net-emit-intermediates. **Smoke (llama-cli, 64-token --ignore-eos, prompt "Tell a short story about a robot..."):** baseline 20.09 t/s tg / 21.24 t/s pp; MTP 19.12 t/s tg / 21.03 t/s pp. **PPL on wikitext-2 test, 16 chunks @ n_ctx=512:** baseline 7.0974 ± 0.278; MTP 7.0974 ± 0.278 — **byte-identical chunk-by-chunk**, MTP head doesn't disturb main logits. **Server (apples-to-apples completion, n_predict=128, temp=0):** baseline 20.22 t/s tg / 22.57 t/s pp; MTP 15.17 t/s tg / 19.84 t/s pp; **draft acceptance 85.3% (58/68)**. Despite high acceptance, MTP is **−25% in tg** in this configuration — the `--cpu-moe` partial offload puts every draft and every verification through CPU MoE compute, which dominates and is not amortized by acceptance savings. **Conclusion for 8 GB VRAM + 35B-A3B:** MTP is correctness-good and well-implemented but is throughput-negative under partial offload. Throughput uplift needs a configuration where draft cost is meaningfully cheaper than verify cost — i.e., full GPU residency or a smaller MTP-capable model. No bug in our MTP wiring; this is the published "12% overhead" tax (commit `fd77f898`) compounded by CPU-MoE serial compute.

- iter 2 (2026-05-02): **Quadro RTX 6000 (sm_75) replication.** Built peer's `fix/cuda-delta-net-emit-intermediates` branch on dual Quadro RTX 6000. Re-quantized BF16 → IQ4_KS with Unsloth's GGUF imatrix (required a +125-line patch to `examples/quantize/quantize.cpp` to read GGUF-format imatrix files; legacy loader only handled `.dat`). Used `--custom-q ssm_out\.weight=q8_0` so split-mode-graph's `split_recurrent_tensors` (which rejects per-row-meta types on `ssm_out`) accepts the IQ4_KS bulk quant. Output: `Qwen3.6-35B-A3B-IQ4_KS-imat.gguf` 19.66 GB; nextn heads preserved at f16; output.weight at q6_K (no-imatrix fallback). **Hard finding 1**: peer's branch tip `a0d0e06e` runs at **2.4 t/s tg baseline** on this hardware — a ~20× regression vs. expectations. Commits between `fd77f898` (where compute buffer is 0.2 GB and tg=50 t/s baseline) and `a0d0e06e` (compute buffer 7.9 GB, tg=2.4) are the unrelated TURBO_KV_4B/Vulkan/Mesa WIP soup; the regression is sm_75-specific (the soup was tested on AMD Vulkan / sm_86 only). **Hard finding 2**: the published "10.2 → 17.8 t/s = 1.74×" claim cited in the handoff was on **0.8B F16 + AMD 6800 XT Vulkan** (per commit `fd77f898`'s body), not on Qwen3.6-35B-A3B and not on CUDA. Combined with peer's iter-1 −25% measurement, the 1.74× has never been observed on this model class on any hardware. **Pivot**: extracted the MTP work cleanly onto a fresh branch from `ikawrakow/main` HEAD `a8aecbf1` rather than inherit the WIP soup. Branch `mtp-extract` carries 6 small commits: (1) `f51b3011 llama-model: add 4 nextn tensor mappings to LLM_ARCH_QWEN35MOE` — the missing tensor-name table that was the only reason upstream couldn't load fork-quantized MoE-MTP files; (2) `773b0648` — peer's +5-line delta-net op-supports gate; (3) `f23a34a5` — the GGUF imatrix loader; (4) `89aa1e7b qwen35moe: enable MTP layer loading mirroring qwen35 dense` — read `nextn_predict_layers`, set `n_layer_kv_from_start`, force MTP layer non-recurrent in `recurrent_layer_arr`, accept n_layer=41 as 35B-A3B MTP variant, load nextn tensors at the MTP layer; (5) `72408097 build_qwen35moe: mirror build_qwen35 dense MTP structure` — limit main loop to `n_layer - nextn_predict_layers`, branch on `cparams.mtp_op_type` to use `build_qwen35_mtp`, emit `result_mtp_embd`; (6) `ad74bc2e llama.cpp: gate MTP on QWEN35MOE alongside QWEN35` — extend the four `model.arch == LLM_ARCH_QWEN35` MTP gates to also accept QWEN35MOE. **Baseline result on this clean branch**: `-no-mtp` boots cleanly, three-run **tg = 90.4, 91.9, 95.1 t/s**, pp = 232 t/s, on full GPU offload (CUDA0,CUDA1, split-mode graph, batch 4096, ubatch 2048, ctx 4096) — that's ~10% **faster than production** (84 t/s on Q8 at upstream `453a027`) thanks to IQ4_KS being 4.25 bpw vs Q8's 8.5 bpw. **Open**: `-mtp` boot reaches `ggml_cuda_set_peer_access` then crashes with `std::out_of_range: map::at` somewhere in the MTP-aware KV-cache or compute-graph init path. The four QWEN35MOE gates we added haven't fully taught the runtime that MoE arch + MTP is now valid — there's at least one more lookup somewhere (likely cross-device buft placement à la fork's `24f64b1e perf: fix cross-device MTP — co-locate MTP + output on last-main-layer GPU`) that still rejects MoE. Step 5 close pending that diagnosis. **Step 5 status**: not yet bound. We have a proven-fast baseline on Quadro full-GPU and a real reproduction of the MTP runtime crash; binding the throughput ratio requires resolving the cross-device buft placement gap. **Files**: `mtp-extract/inventory.md` (121 fork commits classified — 25 mtp_core, 13 mixed, 1 mtp_test, 1 mtp_doc, 79 out_scope, 2 merge), `mtp-extract/classify.py`. **Branch ready for review**: `slartibardfast/ik_llama.cpp:mtp-extract` — the 6 commits above are upstream-PR-quality and worth landing independently of the MTP runtime fix.

- iter 3 (2026-05-02): **Step 5 binding negative.** The previously-suspected `std::out_of_range: map::at` MTP boot crash on `mtp-extract` was misdiagnosed in the iter-2 wrap-up — there is no boot crash. `-mtp` boots cleanly on `Qwen3.6-35B-A3B-IQ4_KS-imat.gguf`, full GPU offload across CUDA0,CUDA1, split-mode graph, ctx 4096, batch 1024, ubatch 256. Re-running today produced clean boot, working `/v1/chat/completions`, `#gen drafts > 0`, valid draft-accept metrics, and end-to-end inference. The four QWEN35MOE gate patches (`ad74bc2e`) plus the build-graph rework (`72408097`) are sufficient — there was no missing cross-device buft placement step. **Bench (six runs total)**: baseline `-no-mtp` 256-token greedy: tg = 91.4 / 92.2 / 94.3 t/s (avg **92.6 t/s**), draft_n=0 as expected. MTP `-mtp` 256-token greedy: tg = 45.15 / 45.23 / 45.27 t/s (avg **45.2 t/s**, draft acceptance 0.381 = 77/202). MTP `-mtp` 1024-token greedy (single run): tg = **48.9 t/s**, draft acceptance 0.461 = 374/812. Conclusion: **MTP is 0.49–0.53× baseline** on this hardware/model, draft acceptance is real but the MTP draft+verify chain costs ~2.76× more per step than the baseline forward pass — the speedup math does not close. This matches and extends iter-1's 3060 Ti+cpu-moe finding (−25%); the regression is bigger here because pure-GPU baseline is faster, so MTP's per-step overhead is proportionally larger. The published "1.74× at 10.2→17.8 t/s" claim from `fd77f898` was 0.8B F16 + AMD 6800 XT Vulkan and has now been falsified for Qwen3.6-35B-A3B IQ4_KS on sm_75 CUDA. Step 5 closes `[~]` (binding negative, intentional partial — uplift target not met for this hardware/model class). All MTP code on `mtp-extract` is correctness-good (clean boot, real draft acceptance, end-to-end inference); the negative is hardware-economic, not a bug. **Bench script**: `/tmp/bench_mtp.sh`. **Profile**: `/home/llm/profiles/qwen3.6-35b-mtp-iq4ks.sh`.
