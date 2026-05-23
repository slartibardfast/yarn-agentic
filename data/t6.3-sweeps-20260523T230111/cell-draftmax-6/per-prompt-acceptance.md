# Per-prompt DFlash acceptance — `cell-draftmax-6`

Release events: 9 (1 warmup + 8 measured)

Per-task accept (A) / gen (G) values are read directly from the server's per-task `draft acceptance rate = R (A / G)` log line.

## Per-task acceptance (in completion order)

| order | slot | task | prompt_idx | n_past | A | G | rate | bench_tokens | latency_s | prompt |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 3 | -1 | 46 | 23 | 28 | 0.821 | ? | ? | <WARMUP> |
| 1 | 1 | 13 | 0 | 270 | 189 | 264 | 0.716 | 256 | 33.7 | Explain the difference between latent diffusion and... |
| 2 | 0 | 14 | 1 | 267 | 159 | 377 | 0.422 | 256 | 50.5 | Summarize the plot of King Lear in one... |
| 3 | 1 | 15 | 2 | 277 | 166 | 346 | 0.480 | 256 | 83.3 | Write Python code that fits a 2nd-degree polynomial to a... |
| 4 | 0 | 16 | 3 | 269 | 158 | 387 | 0.408 | 256 | 105.1 | What are the main causes of the Peloponnesian... |
| 5 | 1 | 17 | 4 | 278 | 145 | 440 | 0.330 | 256 | 146.3 | Translate to French: The early-morning fog lingered over... |
| 6 | 0 | 18 | 6 | 266 | 157 | 387 | 0.406 | 256 | 161.4 | Describe the role of telomeres in cellular... |
| 7 | 1 | 19 | 5 | 271 | 184 | 282 | 0.652 | 256 | 185.2 | List five practical steps for reducing memory allocations... |
| 8 | 0 | 20 | 7 | 264 | 180 | 297 | 0.606 | 256 | 193.5 | Write a haiku about a printing... |

## Per-prompt summary (aggregated across all completions of that prompt)

| prompt_idx | n_completions | accept | gen | rate | prompt |
|---:|---:|---:|---:|---:|---|
| 0 | 1 | 189 | 264 | 0.716 | Explain the difference between latent diffusion and... |
| 1 | 1 | 159 | 377 | 0.422 | Summarize the plot of King Lear in one... |
| 2 | 1 | 166 | 346 | 0.480 | Write Python code that fits a 2nd-degree polynomial to a... |
| 3 | 1 | 158 | 387 | 0.408 | What are the main causes of the Peloponnesian... |
| 4 | 1 | 145 | 440 | 0.330 | Translate to French: The early-morning fog lingered over... |
| 5 | 1 | 184 | 282 | 0.652 | List five practical steps for reducing memory allocations... |
| 6 | 1 | 157 | 387 | 0.406 | Describe the role of telomeres in cellular... |
| 7 | 1 | 180 | 297 | 0.606 | Write a haiku about a printing... |

## Spread

- min: 0.330
- max: 0.716
- mean: 0.502
- spread: 0.386
