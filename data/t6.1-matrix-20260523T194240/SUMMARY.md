# T6.1 binary ablation matrix — summary

Matrix dir: `t6.1-matrix-20260523T194240`
Cells: 6 (4 clean, 2 broken)

Workload: gate0 reference, 8 prompts × 256 max_tokens, ignore_eos, fire_pattern=concurrent. Server: --parallel 2 (queue depth 6).

## Per-cell results

| cell_id | dflash | hadamard | defrag | wall_s | out_toks | t/s_agg | t/s_slot | status_counts | clean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no-defrag | True | True | -1.0 | 169.3 | 1765 | 10.42 | 3.10 | {200: 8} | ✓ |
| no-dflash | False | True | 0.1 | 89.0 | 1822 | 20.47 | 5.16 | {200: 8} | ✓ |
| no-dflash-nodefrag | False | True | -1.0 | 86.9 | 1822 | 20.96 | 5.30 | {200: 8} | ✓ |
| no-hadamard | True | False | 0.1 | 116.5 | 512 | 4.40 | 8.70 | {200: 2, 0: 6} | ✗ |
| no-hadamard-nodefrag | True | False | -1.0 | 161.4 | 2048 | 12.69 | 3.70 | {200: 8} | ✓ |
| prod-baseline | True | True | 0.1 | 148.7 | 512 | 3.44 | 6.30 | {200: 2, 0: 6} | ✗ |

## Per-feature delta

**Baseline:** `no-defrag` (DFlash ON, Hadamard ON, defrag OFF) = 10.42 t/s aggregate. The defrag-OFF baseline is used because the production default (defrag 0.1) crashed in two of the four T6.1 cells when combined with DFlash multi-slot — see Findings.

| cell_id | t/s_agg | Δ vs baseline (t/s) | Δ (%) |
| --- | --- | --- | --- |
| no-dflash | 20.47 | +10.05 | +96.4% |
| no-dflash-nodefrag | 20.96 | +10.54 | +101.1% |
| no-hadamard-nodefrag | 12.69 | +2.27 | +21.7% |

## Per-feature verdict (binary on/off pairs)

- **DFlash:** ON (no-defrag) 10.42 t/s vs OFF (no-dflash-nodefrag) 20.96 t/s — Δ -10.54 t/s (-50.3%) — **net-negative**.
- **Hadamard:** ON (no-defrag) 10.42 t/s vs OFF (no-hadamard-nodefrag) 12.69 t/s — Δ -2.27 t/s (-17.9%) — **net-negative**.
- **defrag (ON vs OFF):** ON cell `prod-baseline` CRASHED ({200: 2, 0: 6}). Feature is unsafe at this workload.
