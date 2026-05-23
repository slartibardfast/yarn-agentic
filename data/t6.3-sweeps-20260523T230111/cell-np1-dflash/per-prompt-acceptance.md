# Per-prompt DFlash acceptance — `cell-np1-dflash`

Release events: 2 (1 warmup + 1 measured)

Per-task accept (A) / gen (G) values are read directly from the server's per-task `draft acceptance rate = R (A / G)` log line.

## Per-task acceptance (in completion order)

| order | slot | task | prompt_idx | n_past | A | G | rate | bench_tokens | latency_s | prompt |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 3 | -1 | 46 | 23 | 28 | 0.821 | ? | ? | <WARMUP> |
| 1 | 1 | 13 | 0 | 270 | 177 | 308 | 0.575 | 256 | 22.3 | Explain the difference between latent diffusion and... |

## Per-prompt summary (aggregated across all completions of that prompt)

| prompt_idx | n_completions | accept | gen | rate | prompt |
|---:|---:|---:|---:|---:|---|
| 0 | 1 | 177 | 308 | 0.575 | Explain the difference between latent diffusion and... |

## Spread

- min: 0.575
- max: 0.575
- mean: 0.575
- spread: 0.000
