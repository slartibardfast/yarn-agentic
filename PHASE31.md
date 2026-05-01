# Phase 31: MTP production on ik_llama.cpp (3060 Ti, sole focus)

## Status

**Open.** Iter 1 (2026-05-01) closed Steps 1–4 on the 3060 Ti. **Step 5 (throughput uplift) does not bind on 8 GB VRAM with `--cpu-moe`** — measurement showed −25% tg, the opposite of the claim. Phase remains open; binding the uplift requires hardware that fits the model fully on GPU. See `HANDOFF.md` for the peer (Quadro ≥ 24 GB) instructions. Successor to PHASE29 / PHASE30 (both abandoned per user direction 2026-05-01). Sole focus: get MTP working end-to-end on `ik_llama.cpp`. **No turboquant.**

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

- [ ] **Step 5 — MTP throughput uplift.**
  - `./build/bin/llama-bench -m <Q4_K_M-gguf> -mtp 0,1 -fa 1 -ngl <max-fitting>`
  - Or instrument llama-server for tg t/s with --mtp on/off.
  - Verify by: tg t/s with MTP > baseline (target: 1.2× minimum given partial offload constraint; was 1.74× at 10.2→17.8 t/s in commit `fd77f898` on better hardware).

- [ ] **Step 6 — Document.**
  - PHASE31.md updated per iteration (per CLAUDE.md §5).
  - MEMORY.md notes any non-obvious findings (per §6).

## Closing condition

Phase 31 closes when:
1. Steps 1–5 all `[x]` with binding evidence (on the 3060 Ti or a peer host with sufficient VRAM).
2. End-to-end MTP run on Q4_K_M-quantized Qwen3.6-35B-A3B returns coherent text and acceptance > 50%.
3. Throughput **uplift** (MTP tg > baseline tg) measured and documented. The 3060 Ti's 8 GB VRAM forces `--cpu-moe`, which makes MTP throughput-negative — the bind needs ≥ 24 GB VRAM (full GPU offload). See `HANDOFF.md`.

## Open work (not yet bound)

These items are explicitly open, not deferred or hidden behind cover language. Each blocks Phase 31 closure.

- **Step 5 — Throughput uplift number.** Never bound: the 3060 Ti measurement showed MTP **−25%** tg under `--cpu-moe` partial offload. The fix is a host with ≥ 24 GB VRAM (RTX A6000 / RTX 6000 Ada / Quadro RTX 8000 etc.) running with full GPU offload. `HANDOFF.md` is the peer playbook.
- **Larger-context PPL parity.** Verified at c=512 / 16 chunks. Closing-condition wording asked for c=4096 / 16 chunks — not run because compute-buffer OOMs at c=4096 on 8 GB. Defers to peer host.
- **PR for `fix/cuda-delta-net-emit-intermediates`.** Branch pushed to `slartibardfast/ik_llama.cpp` at commit `f9bb0efa`; PR not opened. URL in HANDOFF.md. Without merge, downstream consumers of `slartibardfast/ik_llama.cpp main` still hit the assert in MTP graphs with `n_tokens > 1`.
- **Submodule pointer for ik_llama.cpp** in this top-level repo stays at `a0d0e06e` (pre-fix). Do not advance it until the fix is merged on `slartibardfast/ik_llama.cpp main`; otherwise production users of this top-level repo would silently take an unmerged branch.
- **CUDA delta-net kernel extension to handle `emit_intermediates=true && n_tokens > 1` natively.** Current fix conservatively falls back to CPU for that path. A real GPU kernel for the multi-state-copy emit would unlock prompt-eval acceleration during MTP graph builds. Not on PHASE31's critical path; spawn-as-needed phase.
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

- iter 2 placeholder — Steps 5–6 close-out follow.
