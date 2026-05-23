# Per-prompt DFlash acceptance — `cell-np2-dflash`

Release events: 3 (1 warmup + 2 measured)

Per-task accept (A) / gen (G) values are read directly from the server's per-task `draft acceptance rate = R (A / G)` log line.

## Per-task acceptance (in completion order)

| order | slot | task | prompt_idx | n_past | A | G | rate | bench_tokens | latency_s | prompt |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 3 | -1 | 46 | 23 | 28 | 0.821 | ? | ? | <WARMUP> |
| 1 | 1 | 13 | 1 | 267 | 185 | 278 | 0.665 | 256 | 36.9 | Summarize the plot of King Lear in one... |
| 2 | 0 | 14 | 0 | 270 | 178 | 304 | 0.586 | 256 | 39.4 | Explain the difference between latent diffusion and... |

## Per-prompt summary (aggregated across all completions of that prompt)

| prompt_idx | n_completions | accept | gen | rate | prompt |
|---:|---:|---:|---:|---:|---|
| 0 | 1 | 178 | 304 | 0.586 | Explain the difference between latent diffusion and... |
| 1 | 1 | 185 | 278 | 0.665 | Summarize the plot of King Lear in one... |

## Spread

- min: 0.586
- max: 0.665
- mean: 0.625
- spread: 0.080
