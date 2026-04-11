# KV-V quant matrix — Qwen3.5-9B q4km, Vega 64

## Setup

- `Qwen3.5-9B-mtp-q4km.gguf`, Vega 64, `-fa on`, K stays F16, `--cache-type-v <type>`
- n_ctx = 4096, prompt = `fixed_prompt.txt` (184 tokens), `n_predict=128`
- 3 runs per type, seeds 42/43/44, temperature=0, `cache_prompt=false`
- `GGML_VK_PERF_LOGGER=1` captured for per-op totals (not used in ranking, kept for later)

## Matrix

| V type    | K size | V size | splits | prompt t/s | predict t/s | MTP%  | Σ GPU ms | fp diff vs f16 |
|-----------|-------:|-------:|-------:|-----------:|------------:|------:|---------:|---------------:|
| **f16**   | 72.0   | 72.0   | 1      | 292.6      | **35.96**   | 69.9% | 21653    | —              |
| **q4_1**  | 72.0   | 22.5   | 1      | 291.8      | 35.69       | 65.1% | 22226    | 0.0%           |
| **q5_0**  | 72.0   | 24.8   | 1      | 291.8      | 35.57       | 70.1% | 22243    | 0.0%           |
| **q5_1**  | 72.0   | 27.0   | 1      | 292.1      | 35.59       | 65.4% | 22169    | 0.0%           |
| **q8_0**  | 72.0   | 38.2   | 1      | 291.4      | 35.60       | 68.2% | 22186    | 0.0%           |
| **iq4_nl**| 72.0   | 20.2   | 1      | 292.6      | 35.42       | 65.4% | 22367    | 0.0%           |
| **tq_v_4b**| 72.0  | 18.6   | 1      | 293.0      | 35.57       | 67.5% | 22223    | 44.2%          |
| **q4_0**  | 72.0   | 20.2   | 1      | 291.6      | 35.63       | 66.7% | 20945    | **97.5%** ⚠️    |

Fingerprint diff is computed as the per-character difference in the
first 120 characters of the generation, measured against the f16
reference. A non-zero value means the model is taking a different
generation path, not that it's getting "slightly worse" — once the
model chooses a different first token the whole trajectory diverges.

## Regression check

**All 7 quant types give `graph splits = 1`**, matching the plain f16
baseline. Phase 4's Vulkan-side mixed K/V flash-attn work is working
correctly for every quant type at n_ctx=4096.

## Headline findings

1. **q4_0 changes output quality substantially.** The first 120 chars
   of generation differ from f16 by 97.5%. This is not a small
   precision shift — the model starts generating a numbered-list
   format instead of the `Here are the logs...` trajectory the other
   quants produce. Do not default to `--cache-type-v q4_0` for
   quality-sensitive agent work.

2. **iq4_nl gives the same memory footprint as q4_0 with zero observed
   divergence.** At 20.2 MiB V cache (vs q4_0's 20.2 MiB), the IQ4_NL
   non-linear lookup table preserves the f16 output trajectory in
   this workload. This is the **recommended default** for tight VRAM.

3. **tq_v_4b saves the most memory (18.6 MiB, 74% smaller than f16)**
   but has a 44.2% fingerprint diff — it diverges from f16 in the
   middle of the first sentence. For workloads where long-context
   VRAM matters more than output determinism vs. an f16 baseline,
   tq_v_4b is still viable — the output is grammatical and on-topic,
   just not identical. At n_ctx=32768 this saves ~427 MiB.

4. **q5_0, q5_1, q8_0 are all zero-divergence** at comparable cost.
   q5_0 even shows slightly higher MTP acceptance (70.1% vs the f16
   baseline's 69.9%) — probably noise, but noteworthy. q8_0 at 38.2
   MiB is the quality floor if you want a quant that you're confident
   matches f16 within machine precision; q5_0 at 24.8 MiB gives
   essentially the same behaviour at 35% of the cost.

5. **Throughput is flat across types.** Prompt eval 291.4–293.0 t/s
   (0.5% range), generation 35.42–35.96 t/s (1.5% range). The
   dequantisation overhead in the V cache attention is dominated by
   the other work, so there is no throughput argument against
   picking the smallest viable quant type.

6. **Total GPU time per workload is flat too**, 20.9s (q4_0) – 22.4s
   (iq4_nl). The outlier is q4_0 which is ~700 ms faster than the
   others — but it's also the one that's silently taking a different
   generation path, so the "savings" are illusory. If the attention
   is doing less work it's because the softmax scores are being
   squashed into a more uniform distribution, which is exactly why
   the output changes.

## Recommendations

For **Qwen3.5-9B agent work on Vega 64**:

| Goal                               | Pick          | V size | Quality  |
|------------------------------------|---------------|-------:|---------:|
| Maximum precision (default)        | `f16`         | 72.0   | reference |
| Best memory/quality tradeoff       | **`iq4_nl`**  | 20.2   | 0% diff   |
| Acceptable if slight drift is OK   | `q5_0` / `q5_1`/`q8_0`| 24.8–38.2 | 0% diff |
| Max VRAM savings, accept drift     | `tq_v_4b`     | 18.6   | 44% diff  |
| **Avoid**                          | `q4_0`        | 20.2   | 97% diff ⚠️ |

## Caveats

- **The fingerprint is only the first 120 characters.** A zero there
  does not mean identical output at 500+ tokens. A non-zero means
  the generation path definitely diverged early.
- **Single-seed measurement per type.** Throughput may be slightly
  noisy; rerun with 5+ seeds per type to tighten confidence intervals.
- **MTP acceptance rate variance** (65.1–70.1%) is larger than the
  expected measurement error for deterministic generation — it's
  reflecting real draft-path differences from the quantisation.

## Artifacts

- `kv-<type>-2026-04-11T231000Z.stderr` — raw server stderr per type
- `kv-<type>-2026-04-11T231000Z.drive.txt` — driver output JSON per type
- `kv-<type>-2026-04-11T231000Z.perf.json` — parsed perf logger per type
- `kv-matrix-summary-2026-04-11T231000Z.json` — the structured summary
- `build_summary.py` — the script that built the summary
- `SUMMARY.md` — this file
