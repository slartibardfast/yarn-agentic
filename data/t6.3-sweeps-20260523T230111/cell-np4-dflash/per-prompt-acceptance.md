# Per-prompt DFlash acceptance — `cell-np4-dflash`

Release events: 5 (1 warmup + 4 measured)

Per-task accept (A) / gen (G) values are read directly from the server's per-task `draft acceptance rate = R (A / G)` log line.

## Per-task acceptance (in completion order)

| order | slot | task | prompt_idx | n_past | A | G | rate | bench_tokens | latency_s | prompt |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 3 | -1 | 46 | 23 | 28 | 0.821 | ? | ? | <WARMUP> |
| 1 | 1 | 13 | 0 | 270 | 189 | 264 | 0.716 | 256 | 33.8 | Explain the difference between latent diffusion and... |
| 2 | 0 | 14 | 1 | 267 | 159 | 377 | 0.422 | 256 | 50.8 | Summarize the plot of King Lear in one... |
| 3 | 0 | 16 | 2 | 215 | 139 | 216 | 0.644 | 194 | 81.0 | Write Python code that fits a 2nd-degree polynomial to a... |
| 4 | 1 | 15 | 3 | 269 | 154 | 404 | 0.381 | 256 | 86.3 | What are the main causes of the Peloponnesian... |

## Per-prompt summary (aggregated across all completions of that prompt)

| prompt_idx | n_completions | accept | gen | rate | prompt |
|---:|---:|---:|---:|---:|---|
| 0 | 1 | 189 | 264 | 0.716 | Explain the difference between latent diffusion and... |
| 1 | 1 | 159 | 377 | 0.422 | Summarize the plot of King Lear in one... |
| 2 | 1 | 139 | 216 | 0.644 | Write Python code that fits a 2nd-degree polynomial to a... |
| 3 | 1 | 154 | 404 | 0.381 | What are the main causes of the Peloponnesian... |

## Spread

- min: 0.381
- max: 0.716
- mean: 0.541
- spread: 0.335
