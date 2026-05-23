# Per-prompt DFlash acceptance — `cell-draftmax-5`

Release events: 9 (1 warmup + 8 measured)

Per-task accept (A) / gen (G) values are read directly from the server's per-task `draft acceptance rate = R (A / G)` log line.

## Per-task acceptance (in completion order)

| order | slot | task | prompt_idx | n_past | A | G | rate | bench_tokens | latency_s | prompt |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 3 | -1 | 46 | 23 | 28 | 0.821 | ? | ? | <WARMUP> |
| 1 | 0 | 14 | 2 | 220 | 143 | 220 | 0.650 | 199 | 28.7 | Write Python code that fits a 2nd-degree polynomial to a... |
| 2 | 1 | 13 | 0 | 270 | 180 | 296 | 0.608 | 256 | 38.4 | Explain the difference between latent diffusion and... |
| 3 | 0 | 15 | 1 | 267 | 155 | 395 | 0.392 | 256 | 83.0 | Summarize the plot of King Lear in one... |
| 4 | 1 | 16 | 4 | 278 | 168 | 344 | 0.488 | 256 | 86.4 | Translate to French: The early-morning fog lingered over... |
| 5 | 1 | 18 | 5 | 271 | 186 | 275 | 0.676 | 256 | 123.0 | List five practical steps for reducing memory allocations... |
| 6 | 0 | 17 | 3 | 269 | 158 | 387 | 0.408 | 256 | 135.3 | What are the main causes of the Peloponnesian... |
| 7 | 1 | 19 | 6 | 266 | 163 | 365 | 0.447 | 256 | 175.1 | Describe the role of telomeres in cellular... |
| 8 | 0 | 20 | 7 | 264 | 179 | 301 | 0.595 | 256 | 176.6 | Write a haiku about a printing... |

## Per-prompt summary (aggregated across all completions of that prompt)

| prompt_idx | n_completions | accept | gen | rate | prompt |
|---:|---:|---:|---:|---:|---|
| 0 | 1 | 180 | 296 | 0.608 | Explain the difference between latent diffusion and... |
| 1 | 1 | 155 | 395 | 0.392 | Summarize the plot of King Lear in one... |
| 2 | 1 | 143 | 220 | 0.650 | Write Python code that fits a 2nd-degree polynomial to a... |
| 3 | 1 | 158 | 387 | 0.408 | What are the main causes of the Peloponnesian... |
| 4 | 1 | 168 | 344 | 0.488 | Translate to French: The early-morning fog lingered over... |
| 5 | 1 | 186 | 275 | 0.676 | List five practical steps for reducing memory allocations... |
| 6 | 1 | 163 | 365 | 0.447 | Describe the role of telomeres in cellular... |
| 7 | 1 | 179 | 301 | 0.595 | Write a haiku about a printing... |

## Spread

- min: 0.392
- max: 0.676
- mean: 0.533
- spread: 0.284
