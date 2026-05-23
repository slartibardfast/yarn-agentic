# T6.1 binary ablation matrix — summary

Matrix dir: `t6.1-matrix-fixed-20260523T211929`
Cells: 4 (4 clean, 0 broken)

Workload: gate0 reference, 8 prompts × 256 max_tokens, ignore_eos, fire_pattern=concurrent. Server: --parallel 2 (queue depth 6).

## Per-cell results

| cell_id | dflash | hadamard | defrag | wall_s | out_toks | t/s_agg | t/s_slot | status_counts | clean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no-defrag | True | True | -1.0 | 162.1 | 1822 | 11.24 | 2.97 | {200: 8} | ✓ |
| no-dflash | False | True | 0.1 | 89.1 | 1822 | 20.45 | 5.16 | {200: 8} | ✓ |
| no-hadamard | True | False | 0.1 | 185.5 | 2048 | 11.04 | 3.43 | {200: 8} | ✓ |
| prod-baseline | True | True | 0.1 | 180.5 | 1991 | 11.03 | 3.28 | {200: 8} | ✓ |

## Per-feature delta

**Baseline:** `prod-baseline` = 11.03 t/s aggregate.

| cell_id | t/s_agg | Δ vs baseline (t/s) | Δ (%) |
| --- | --- | --- | --- |
| no-defrag | 11.24 | +0.21 | +1.9% |
| no-dflash | 20.45 | +9.42 | +85.4% |
| no-hadamard | 11.04 | +0.01 | +0.1% |

## Per-feature verdict (binary on/off pairs)

- **DFlash:** ON (prod-baseline) 11.03 t/s vs OFF (no-dflash) 20.45 t/s — Δ -9.42 t/s (-46.1%) — **net-negative**.
- **Hadamard:** ON (prod-baseline) 11.03 t/s vs OFF (no-hadamard) 11.04 t/s — Δ -0.01 t/s (-0.1%) — **no-op (within noise)**.
- **defrag:** ON (prod-baseline) 11.03 t/s vs OFF (no-defrag) 11.24 t/s — Δ -0.21 t/s (-1.9%) — **no-op (within noise)**.
