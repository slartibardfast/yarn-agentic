# Vega 64 vs Navi 21 — Qwen3.5-9B q4km, identical workload

## Setup

- Same binary (`b8783-71ba1ed4a`), same model (`Qwen3.5-9B-mtp-q4km`),
  same prompt (`fixed_prompt.txt`), same seeds (42/43/44), n_predict=256
- `-fa on`, default F16 KV cache, `GGML_VK_PERF_LOGGER=1`
- Vega 64 via `GGML_VK_VISIBLE_DEVICES=1` (8 GiB RADV VEGA10, GCN)
- Navi 21 via `GGML_VK_VISIBLE_DEVICES=0` (16 GiB RADV NAVI21, RDNA2,
  RX 6800 XT)

## Aggregate

| Metric           | Vega 64    | Navi 21    | Navi / Vega |
|------------------|-----------:|-----------:|------------:|
| Total GPU time   | 21666 ms   | 13745 ms   | **0.63×**   |
| Prompt t/s       | 291.7      | 796.9      | **2.73×**   |
| Predict t/s      | 35.88      | 55.82      | **1.56×**   |
| MTP acceptance   | 69.9%      | 68.1%      | 0.97×       |

**Navi 21 is 2.73× faster on prompt eval and 1.56× faster on token
generation** for the same workload on the same binary.

The prompt-eval / generation gap is explained by hardware bottlenecks:
prompt eval is compute-bound (large batched matmuls hitting the shader
array), generation is memory-bandwidth-bound (single-token matmuls
streaming weights from VRAM). Navi 21 has ~2.5× Vega's theoretical
FLOPS but only ~1.06× Vega's memory bandwidth, so prompt eval wins by
more than generation.

## Output correctness

First 130 characters of generation are byte-identical between Vega and
Navi. At char 130 the two diverge by **one vowel**:

```
vega: ...A (summarization job)...
navi: ...A (summarisation job)...
```

Same model, same seed, same temperature 0 — but RDNA2 and GCN evaluate
the quant dequant path with slightly different floating-point rounding,
which propagates into the token sampler and flips a low-probability
token occasionally. Both outputs are grammatically valid and semantically
identical. Not a correctness bug, just a real-world expression of
numerical non-determinism across hardware generations.

## Top-10 hot ops, side by side

| Op (shortened)                                       | Vega ms | Navi ms | ratio | Vega GFLOPS | Navi GFLOPS | GF ratio |
|------------------------------------------------------|--------:|--------:|------:|------------:|------------:|---------:|
| MUL_MAT_VEC q4_K 12288×4096 ×2 (double gate)         | 4307.6  | 3327.3  | 0.77  | 1160        | 1531        | 1.32     |
| MUL_MAT_VEC q5_K 248320×4096 + ARGMAX (vocab head)   | 3262.7  | 1934.1  | 0.59  | 471         | 810         | **1.72** |
| RMS_NORM_MUL ×2 (4096)                               | 1795.1  | 1081.6  | 0.60  | 852         | 1442        | 1.69     |
| MUL_MAT_VEC q5_K 8192×4096 + SCALE + GET_ROWS ×2     | 1767.8  | 1097.9  | 0.62  | 656         | 1077        | 1.64     |
| MUL_MAT_ADD MUL_MAT_VEC q6_K 4096×12288              | 1630.0  | 1235.5  | 0.76  | 790         | 1065        | 1.35     |
| MUL_MAT_ADD MUL_MAT_VEC q4_K 4096×12288              | 1107.9  | 874.7   | 0.79  | 1094        | 1412        | 1.29     |
| MUL_MAT_VEC q6_K 32×4096 + SILU + MUL_MAT_VEC q4_K   | 795.7   | 504.5   | 0.63  | 767         | 1234        | 1.61     |
| MUL_MAT_VEC q5_K 8192×4096 + MUL_MAT_VEC q5_K 1024×  | 722.3   | 515.7   | 0.71  | 786         | 1123        | 1.43     |
| **MUL_MAT q4_K 12288×218 (prompt-eval batched)**     | **685.0** | **186.4** | **0.27** | **6342** | **23312** | **3.68** |
| MUL_MAT_VEC q6_K 4096×4096 + SCALE                   | 578.3   | 381.6   | 0.66  | 698         | 1079        | 1.54     |

## Findings

1. **The batched MUL_MAT for prompt eval is 3.68× faster on Navi 21.**
   At n=218 tokens, Navi hits **23.3 TFLOPS** on the q4_K matmul vs
   Vega's **6.3 TFLOPS**. That's the prompt-eval speedup in one number.

2. **The vocab head (q5_k 248320×4096 + ARGMAX) is 1.72× faster on Navi
   in GFLOPS terms**, the biggest gain among the generation-path
   matmul_vec ops. This is the same kernel that hit 256 VGPR on Vega
   and was occupancy-limited. RDNA2's wave32 / 256 VGPR per wave
   architecture means the same per-work-item register footprint
   gets ~2× the wavefronts per CU compared to GCN's wave64 on Vega.

3. **RMS_NORM_MUL is 1.69× faster on Navi** (852 → 1442 GFLOPS).
   Normalisation is memory-traffic-dominated (read input, compute
   reciprocal-sqrt, write back) and Navi 21's improved L2 cache
   helps more than bandwidth alone would suggest.

4. **MTP acceptance is nearly identical** (69.9% Vega vs 68.1% Navi),
   confirming that the drafting logic works correctly on both GPUs.
   The 1.8% gap is the kind of measurement noise we saw earlier
   from numerical-precision shifts between runs.

5. **Phase 4's tq_v_4b mixed K/V shader work applies cleanly to both
   GPUs.** The step 4 KV matrix was Vega-only; Navi 21 should see
   similar savings at better absolute throughput.

## Implications for future work

- **Navi 21 is already on the machine.** If the user's goal is
  sustained agent throughput, running Qwen3.5-9B on Navi 21 via
  `GGML_VK_VISIBLE_DEVICES=0` is a **1.56× generation speedup with
  no code changes** and **2× the VRAM** (16 GiB vs 8 GiB — means
  n_ctx = 32768 fits comfortably with the q8_0 V cache where Vega
  OOMs at 16384 with q4_0).
- The Phase 3/4 fusion and mixed K/V work was done for Vega targeting;
  all of it benefits Navi 21 too, at no additional cost.
- **Priority for Phase 5 shifts**: register-pressure reduction on
  the q5_k vocab head kernel is still the right next target, but
  the ROI on a Vega-specific optimization is lower than on a kernel
  that helps both GPUs. Any kernel tweaks should be benchmarked on
  both before committing.
- The user may want to reconsider Vega 64 as the "primary" target.
  Unless there's a reason to keep Navi 21 free for other workloads
  (display, other users, etc.), the math says to run on Navi.

## Artifacts

- `navi-per-op-2026-04-11T232800Z.stderr` — Navi raw perf-logger output
- `navi-per-op-2026-04-11T232800Z.json` — parsed per-op summary
- `navi-per-op-2026-04-11T232800Z.drive.txt` — driver aggregate
- `vega-vs-navi-diff-2026-04-11T232900Z.json` — shared-ops delta table
- `build_diff.py` — the script that built the delta
- `SUMMARY.md` — this file
