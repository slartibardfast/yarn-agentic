# Phase 2 profiling sweep — results

This document is the morning-readable synthesis of the overnight
profiling run described in [`OVERNIGHT.md`](OVERNIGHT.md). All raw
data lives under [`profile/`](profile/), each step has its own
`SUMMARY.md` with the detailed findings, and the append-only run
log is [`profile/overnight.log`](profile/overnight.log).

Target workload everywhere: **Qwen3.5-9B-mtp-q4km** on Vega 64 via
the Vulkan backend, driven with `profile/fixed_prompt.txt` (a
~218-token incident-report instruction), `-fa on`, seed 42, temperature
0. Exceptions noted per section.

Binary: `llama-server b8783-71ba1ed4a` (fork
`slartibardfast/llama.cpp@a5af4aa97` branch `vulkan-phase4`).

---

## TL;DR — what the night changed

1. **Fusion and graph optimization are worth ~6.4% of total GPU time
   at minimum.** `GGML_VK_DISABLE_FUSION` costs 2.76%, `DISABLE_GRAPH_OPTIMIZE`
   costs 3.68%. The biggest single hit is to the vocab-head ARGMAX:
   disabling graph optimize turns a 5.9 ms fused dispatch into a
   1508 ms unfused dispatch (255× slowdown). Phase 3/4 matters.
2. **`iq4_nl` should be the default `--cache-type-v`** for agent work.
   20.2 MiB V cache with zero observed output divergence from f16.
   **`q4_0` is a trap** — same memory, but changes the generation
   trajectory (97.5% fingerprint diff from f16).
3. **MTP survives context length.** Generation t/s is flat from 184
   to 4440 tokens (36.9–38.0 t/s) and acceptance rate stays in the
   66–95% range depending on prompt character (65–70% for agent/
   directive prompts, 90%+ for natural continuation).
4. **K-cache quantisation is the next big lever** for long-context
   VRAM on Vega 64. V quantisation alone only buys one context tier
   (8K → 12K) because K stays F16. At n_ctx=16384 f16 K is 256 MiB
   vs ~113 MiB for quantised V — K is the bigger consumer once V is
   quant.
5. **Navi 21 is 1.56× faster on generation and 2.73× faster on prompt
   eval** for the exact same workload on the exact same binary.
   The Vega-targeting Phase 4 work benefits Navi 21 too. The machine
   already has Navi 21 (16 GB, device 0). If sustained agent
   throughput is the goal, Navi 21 should probably be the default.
6. **Zero ops land on CPU.** Phase 4's mission is complete: the
   Vulkan backend handles every op in Qwen3.5-9B's graph. No
   coverage-gap work is justified; future effort should target
   per-op kernel speed instead.
7. **`mul_mat_vec_q5_k_f32_f32` compiles at 256 VGPR**, the Vega SIMD
   register ceiling. This is the vocab-head kernel and the clearest
   concrete optimization target on Vega. On Navi 21 the same kernel
   gets 1.72× GFLOPS thanks to wave32's more-favourable register
   allocation, so a Vega-specific fix is less critical if Navi is
   the primary target.

---

## Step 1 — Per-op profile (Vega 64, q4km, default F16 KV)

758 `vk_perf_logger` blocks captured over 21.7 s of GPU work, 100
distinct op combinations. 3 runs × 218-prompt × 256-generate.

Top 10 by total GPU time:

| Op (shortened)                                             | Share | Total ms | Calls   | GFLOPS/s |
|------------------------------------------------------------|------:|---------:|--------:|---------:|
| MUL_MAT_VEC q4_K 12288×4096 ×2 (double gate)               | 19.9% | 4307.6   | 24816   | 1160     |
| MUL_MAT_VEC q5_K 248320×4096 + ARGMAX (vocab head)         | 15.1% | 3262.7   |   755   | 471      |
| RMS_NORM_MUL ×2 (4096)                                     |  8.3% | 1795.1   |   752   | 852      |
| MUL_MAT_VEC q5_K 8192×4096 + SCALE + GET_ROWS + GET_ROWS   |  8.2% | 1767.8   | 17273   | 656      |
| MUL_MAT_ADD MUL_MAT_VEC q6_K 4096×12288                    |  7.5% | 1630.0   | 12784   | 790      |
| MUL_MAT_ADD MUL_MAT_VEC q4_K 4096×12288                    |  5.1% | 1107.9   | 12032   | 1094     |
| MUL_MAT_VEC q6_K 32×4096 + SILU + MUL_MAT_VEC q4_K 4096×4096 | 3.7% | 795.7  | 18048   | 767      |
| MUL_MAT_VEC q5_K 8192×4096 + MUL_MAT_VEC q5_K 1024×4096    |  3.3% |  722.3   |  6768   | 786      |
| **MUL_MAT q4_K 12288×218×4096 (prompt eval, batched)**     |  3.2% |  685.0   |    99   | **6342** |
| MUL_MAT_VEC q6_K 4096×4096 + SCALE                         |  2.7% |  578.3   | 12032   | 698      |

Top 3 hotspots together account for 43% of GPU time. Quant MUL_MAT_VEC
kernels run at 655–1160 GFLOPS/s (~10% of Vega peak). Batched prompt-eval
MUL_MAT runs much higher (6342 GFLOPS/s) because it's hitting the shader
array with a 218-wide batch.

[Detail: `profile/per-op/q4km-vega-f16kv-2026-04-11T224817Z.*`]

---

## Step 2 — Fusion A/B

Three runtime disable knobs tested against the step 1 baseline.

| Run                          | Total GPU Δ         | Predict t/s Δ   | MTP accept |
|------------------------------|--------------------:|----------------:|-----------:|
| baseline (fusion on)         | 21666 ms            | 35.88 t/s       | 69.9%      |
| `GGML_VK_DISABLE_FUSION=1`   | +598 ms (+2.76%)    | -2.80%          | 69.9%      |
| `DISABLE_MULTI_ADD=1`        | +3 ms (≈0%)         | -0.08%          | 69.9%      |
| `DISABLE_GRAPH_OPTIMIZE=1`   | +796 ms (+3.68%)    | -1.64%          | 73.5%      |

`DISABLE_FUSION` only disables scheduler-level pattern fusion
(RMS_NORM_MUL, MUL_MAT_ADD) — the graph-builder fusions from Phase 3
(MUL_MAT_VEC + SILU + MUL_MAT_VEC, MUL_MAT_VEC + ARGMAX, etc.) are
always on and can't be toggled at runtime.

`DISABLE_MULTI_ADD` has no effect because the multi-add pattern
doesn't appear in Qwen3.5-9B's graph.

`DISABLE_GRAPH_OPTIMIZE` is the biggest hit; the dominant cause is
the vocab-head ARGMAX being unfused from its MUL_MAT_VEC predecessor.
The unfused ARGMAX is a 255× slowdown on that op alone.

Interesting side effect: MTP acceptance goes **up** to 73.5% when
graph-optimize is disabled. That's a real numerical-precision
difference from the op reordering — not noise.

[Detail: `profile/fusion-ab/SUMMARY.md`]

---

## Step 3 — MTP draft acceptance vs context length

Two prompt modes tested (synthetic repeat and natural-prose truncate)
at n_ctx = 184 to 4440 tokens.

### Natural prose truncate mode (the realistic one)

| prompt tok | predict t/s | draft_n | accepted | rate  |
|-----------:|------------:|--------:|---------:|------:|
| 184        | 38.0        | 67      | 60       | 89.6% |
| 365        | 37.6        | 79      | 49       | 62.0% |
| 746        | 37.7        | 66      | 61       | 92.4% |
| 2844       | 36.9        | 66      | 62       | 93.9% |
| 2844       | 36.9        | 67      | 61       | 91.0% |

Generation t/s is **essentially flat** across a 24× context sweep
(36.9 to 38.0). Flash attention keeps attention cost bounded enough
that MTP's efficiency dominates.

### Key observation

MTP acceptance depends on **prompt character**, not context length.
Instructive / agent-style prompts ("write an incident report", "call
this tool") sit around 65–70% acceptance. Natural-prose continuation
prompts reach 90%+. The step 1 baseline of 69.9% is representative of
agent workloads, not a pessimistic outlier.

[Detail: `profile/mtp-sweep/SUMMARY.md`]

---

## Step 4 — KV-V quant matrix

All 7 supported V quant types + f16 reference, same workload, same seed.

| V type    | V size  | prompt t/s | predict t/s | MTP%  | Fingerprint diff vs f16 |
|-----------|--------:|-----------:|------------:|------:|------------------------:|
| **f16**   | 72.0 Mi | 292.6      | **35.96**   | 69.9% | —                       |
| q4_1      | 22.5 Mi | 291.8      | 35.69       | 65.1% | 0.0%                    |
| q5_0      | 24.8 Mi | 291.8      | 35.57       | 70.1% | 0.0%                    |
| q5_1      | 27.0 Mi | 292.1      | 35.59       | 65.4% | 0.0%                    |
| q8_0      | 38.2 Mi | 291.4      | 35.60       | 68.2% | 0.0%                    |
| **iq4_nl**| **20.2 Mi** | 292.6  | 35.42       | 65.4% | **0.0%**                |
| tq_v_4b   | 18.6 Mi | 293.0      | 35.57       | 67.5% | 44.2%                   |
| q4_0      | 20.2 Mi | 291.6      | 35.63       | 66.7% | **97.5%** ⚠             |

- Every type gives `graph splits = 1` (Phase 4 regression PASS).
- Throughput is flat across types (prompt 291–293, predict 35.4–36.0).
- **`q4_0` is a trap**: same 20.2 MiB footprint as `iq4_nl` but changes
  the generation trajectory completely.
- **`iq4_nl` is the recommended default** — 20.2 MiB with zero observed
  output divergence.
- `tq_v_4b` wins on raw memory (18.6 MiB) but has 44.2% fingerprint
  diff, so it's a choice you make for long-context VRAM pressure, not
  for day-to-day agent work.

[Detail: `profile/kv-matrix/SUMMARY.md`]

---

## Step 5 — VRAM budget curve

OOM cliffs on Vega 64 (8 GiB) for Qwen3.5-9B q4km + `-fa on`, K stays F16:

| V type    | Max usable n_ctx | OOM cliff |
|-----------|-----------------:|----------:|
| **f16**   | 8192             | 12288     |
| q4_0      | 12288 (+50%)     | 16384     |
| tq_v_4b   | 12288 (+50%)     | 16384     |

V-cache quantisation buys **one more context tier** (8K → 12K). The
cliff is not proportional to V-cache savings, because:

- K stays F16 throughout (192 MiB at n_ctx=12288 → 256 MiB at 16384,
  which is larger than the ~113 MiB quantised V at the same ctx).
- Compute buffer is ~1014 MiB fixed cost (12% of total VRAM).
- llama.cpp's fit-check reserves roughly 1 GiB headroom below the
  raw VRAM ceiling.

**Phase 5 implication: K-cache quantisation is the next big lever**
for long-context on this hardware. Quantising K alone to q4_0 would
save ~190 MiB at n_ctx=16384 and plausibly push the cliff to 24576+.

[Detail: `profile/vram-budget/SUMMARY.md`]

---

## Step 6 — Pipeline stats bonus

`GGML_VK_PIPELINE_STATS` substring filter captures register/shared-mem
counts at pipeline compile time.

Captured entries (one driven run + the startup flash_attn):

| Pipeline                              | SGPR | VGPR    | Spill S/V |
|---------------------------------------|-----:|--------:|----------:|
| flash_attn_f32_f16_aligned_f32accf16  | 96   | 128     | 0 / 0     |
| **mul_mat_vec_q5_k_f32_f32 (large)**  | 48   | **256** | **0 / 0** |
| mul_mat_vec_q6_k_f32_f32 (large)      | 48   | 128     | 0 / 0     |
| mul_mat_vec_q4_k_f32_f32 (large)      | 48   | 128     | 0 / 0     |
| mul_mat_vec_q5_k_f32_f32 (small)      | 48   | 64      | 0 / 0     |
| mul_mat_vec_q6_k_f32_f32 (small)      | 48   | 64      | 0 / 0     |
| mul_mat_vec_q4_k_f32_f32 (small)      | 48   | 64      | 0 / 0     |
| mul_mat_vec_q8_0_f32_f32 (large)      | 48   | 64      | 0 / 0     |
| mul_mat_vec_q8_0_f32_f32 (small)      | 48   | 36      | 0 / 0     |

**Headline**: `mul_mat_vec_q5_k_f32_f32` at VGPR=256 is at Vega's SIMD
register ceiling. 1-wave occupancy on the vocab head. This is a concrete
optimization target — any kernel change that drops VGPR < 128 would
double occupancy and likely significantly improve the top-3 hotspot.

Zero register spills anywhere. Flash attention kernels are comfortable
at VGPR=128.

[Detail: `profile/pipeline-stats/SUMMARY.md`]

---

## Step 7 — Vega 64 vs Navi 21

Same binary, same workload, `GGML_VK_VISIBLE_DEVICES=0` (Navi) vs `=1`
(Vega).

| Metric            | Vega 64 | Navi 21 | Navi / Vega |
|-------------------|--------:|--------:|------------:|
| Total GPU time    | 21666 ms| 13745 ms| **0.63×**   |
| Prompt t/s        | 291.7   | 796.9   | **2.73×**   |
| Predict t/s       | 35.88   | 55.82   | **1.56×**   |
| MTP acceptance    | 69.9%   | 68.1%   | 0.97×       |
| q4_K prompt-eval GFLOPS | 6342 | 23312 | **3.68×**   |
| q5_K vocab head GFLOPS  |  471 |   810 | **1.72×**   |
| RMS_NORM_MUL GFLOPS     |  852 |  1442 | **1.69×**   |

Output divergence: first 130 chars byte-identical, then one vowel
differs ("summarization" vs "summarisation") — numerical precision
drift between GCN and RDNA2 in the quant-dequant path. Both outputs
are semantically identical.

**Implication**: the machine already has Navi 21 (16 GiB VRAM, device
0). 1.56× generation speedup with zero code changes, plus 2× the VRAM
means n_ctx=32K fits comfortably where Vega OOMs at 16K.

[Detail: `profile/vega-vs-navi/SUMMARY.md`]

---

## Step 8 — Op coverage

`GGML_SCHED_DEBUG=2` parse of one driven workload:

- Total nodes observed: **17744**
- Distinct backends: **1** (Vulkan0)
- Ops landing on CPU: **0**
- FUSED op dispatches: **912** (5.1% of all nodes)

Phase 4's coverage work is complete. Future effort should target
per-op kernel speed, not coverage gaps.

[Detail: `profile/op-coverage/SUMMARY.md`]

---

## Phase 5 priorities, ranked by data

Ranked by data-grounded value, not by speculation:

### P0 — `mul_mat_vec_q5_k` register pressure

**Evidence**:
- Step 1: vocab head is 15.1% of GPU time (second hotspot).
- Step 6: compiles at VGPR=256, 1-wave occupancy on Vega.
- Step 7: same kernel gets 1.72× GFLOPS on Navi 21 thanks to wave32.

**Action**: drop VGPR below 128 (target 64) via tile reorg, subgroup
reductions, or smaller per-thread intermediate buffers. Double
occupancy on Vega → expected 1.3–1.5× speedup on this kernel → 5–7%
global generation throughput win.

### P1 — Consider Navi 21 as primary target

**Evidence**:
- Step 7: 1.56× generation, 2.73× prompt, 2× VRAM.
- Step 5: Vega OOMs at n_ctx=12288 with f16 KV; Navi 21 fits
  n_ctx=32768 with room to spare.

**Action**: a config flip, not a code change. If the user is willing
to dedicate Navi 21 to inference, the entire stack gets a free
throughput and headroom upgrade.

### P2 — K-cache quantisation

**Evidence**:
- Step 5: V quantisation alone buys only one context tier on Vega.
- K is 256 MiB at n_ctx=16384 (larger than quantised V at the same ctx).
- Step 4: the mixed K/V flash-attn infrastructure from Phase 4 already
  supports independent K/V types.

**Action**: extend the mixed flash-attn variant set to cover K=quant
combinations. The infrastructure is already in place; it's a registration
+ shader-gen loop edit. Plausibly pushes Vega's long-context cliff from
12K to 24K+.

### P3 — ARGMAX standalone kernel speed

**Evidence**:
- Step 2: disabling graph optimize turns ARGMAX from 5.9 ms to 1508 ms
  (255× slower). The standalone ARGMAX kernel is catastrophically slow;
  only the MUL_MAT_VEC+ARGMAX fused version is fast.
- Step 1: the fused path is already 15.1% of GPU time.

**Action**: if graph-optimize ever gets conditionally disabled (for
debugging, backend expansion, or a future model variant that doesn't
match the fusion pattern), the standalone ARGMAX becomes a cliff.
Worth investigating why the standalone kernel is so slow — probably
an occupancy or memory-coalescing issue on a 248320-wide reduction.
Not urgent, because the fused path is always taken today.

### Not priorities (evidence-backed)

- **More Phase 2 harness comparison models.** Step 7 shows the hardware
  story dominates. Two-model ranking is enough for the Phase 2 writeup.
- **Op coverage expansion on Vulkan.** Step 8 shows zero CPU fallbacks.
  Nothing to cover.
- **Multi-add fusion optimization.** Step 2 shows the pattern doesn't
  fire in this workload.

---

## Wake-up summary

- All 9 steps of the overnight plan completed. Zero WAKE_USER markers.
- Total artifacts under `profile/`: per-op, fusion-ab, mtp-sweep,
  kv-matrix, vram-budget, pipeline-stats, vega-vs-navi, op-coverage
  (each with raw stderr + parsed JSON + SUMMARY.md).
- Regression check passes: `Qwen3.5-9B-mtp-q4km` with `--cache-type-v
  tq_v_4b --flash-attn on` loads with `graph splits = 1`.
- `git status` clean on both repos.
- Suggested next action: read this document top-to-bottom, then
  decide whether to pursue the P0 q5_k VGPR reduction or the P1
  Navi-21-as-primary config switch. Both are actionable; the Navi
  switch is a one-line change and the VGPR fix needs real shader
  work.

## Artifacts index

```
phases/qwen35-mtp/OVERNIGHT.md        # the plan
phases/qwen35-mtp/PROFILING.md        # this document
phases/qwen35-mtp/profile/
├── overnight.log                     # append-only run log
├── fixed_prompt.txt                  # the driver prompt
├── long_prompt.txt                   # the step-3 natural passage
├── drive.py                          # /completion driver
├── parse_vk_perf.py                  # vk_perf_logger parser
├── mtp_ctx_sweep.py                  # step-3 ctx sweep driver
├── per-op/                           # step 1 raw + summary
├── fusion-ab/                        # step 2 raw + delta + summary
├── mtp-sweep/                        # step 3 two JSON runs + summary
├── kv-matrix/                        # step 4 8 runs + matrix summary
├── vram-budget/                      # step 5 sweep outputs + summary
├── pipeline-stats/                   # step 6 two filter runs + summary
├── vega-vs-navi/                     # step 7 Navi run + diff + summary
└── op-coverage/                      # step 8 sched debug + summary
```
