# Per-prompt DFlash acceptance — `cell-prod-baseline`

Release events: 9 (1 warmup + 8 measured)

Per-task accept (A) / gen (G) values are read directly from the server's per-task `draft acceptance rate = R (A / G)` log line.

## Per-task acceptance (in completion order)

| order | slot | task | prompt_idx | n_past | A | G | rate | bench_tokens | latency_s | prompt |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 3 | -1 | 46 | 23 | 28 | 0.821 | ? | ? | <WARMUP> |
| 1 | 0 | 14 | 2 | 220 | 143 | 220 | 0.650 | 199 | 28.7 | Write Python code that fits a 2nd-degree polynomial to a... |
| 2 | 1 | 13 | 0 | 270 | 180 | 296 | 0.608 | 256 | 38.5 | Explain the difference between latent diffusion and... |
| 3 | 0 | 15 | 1 | 267 | 155 | 395 | 0.392 | 256 | 83.1 | Summarize the plot of King Lear in one... |
| 4 | 1 | 16 | 4 | 278 | 168 | 344 | 0.488 | 256 | 86.6 | Translate to French: The early-morning fog lingered over... |
| 5 | 0 | 17 | 3 | 269 | 158 | 387 | 0.408 | 256 | 138.7 | What are the main causes of the Peloponnesian... |
| 6 | 1 | 18 | 6 | 266 | 164 | 361 | 0.454 | 256 | 139.0 | Describe the role of telomeres in cellular... |
| 7 | 1 | 20 | 7 | 264 | 194 | 240 | 0.808 | 256 | 170.1 | Write a haiku about a printing... |
| 8 | 0 | 19 | 5 | 271 | 160 | 376 | 0.426 | 256 | 180.5 | List five practical steps for reducing memory allocations... |

## Per-prompt summary (aggregated across all completions of that prompt)

| prompt_idx | n_completions | accept | gen | rate | prompt |
|---:|---:|---:|---:|---:|---|
| 0 | 1 | 180 | 296 | 0.608 | Explain the difference between latent diffusion and... |
| 1 | 1 | 155 | 395 | 0.392 | Summarize the plot of King Lear in one... |
| 2 | 1 | 143 | 220 | 0.650 | Write Python code that fits a 2nd-degree polynomial to a... |
| 3 | 1 | 158 | 387 | 0.408 | What are the main causes of the Peloponnesian... |
| 4 | 1 | 168 | 344 | 0.488 | Translate to French: The early-morning fog lingered over... |
| 5 | 1 | 160 | 376 | 0.426 | List five practical steps for reducing memory allocations... |
| 6 | 1 | 164 | 361 | 0.454 | Describe the role of telomeres in cellular... |
| 7 | 1 | 194 | 240 | 0.808 | Write a haiku about a printing... |

## Spread

- min: 0.392
- max: 0.808
- mean: 0.529
- spread: 0.416
