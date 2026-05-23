# Per-prompt DFlash acceptance — `cell-draftmax-3`

Release events: 9 (1 warmup + 8 measured)

Per-task accept (A) / gen (G) values are read directly from the server's per-task `draft acceptance rate = R (A / G)` log line.

## Per-task acceptance (in completion order)

| order | slot | task | prompt_idx | n_past | A | G | rate | bench_tokens | latency_s | prompt |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 3 | -1 | 46 | 22 | 25 | 0.880 | ? | ? | <WARMUP> |
| 1 | 0 | 15 | 2 | 215 | 132 | 183 | 0.721 | 194 | 27.6 | Write Python code that fits a 2nd-degree polynomial to a... |
| 2 | 1 | 14 | 0 | 270 | 178 | 226 | 0.788 | 256 | 34.5 | Explain the difference between latent diffusion and... |
| 3 | 0 | 16 | 1 | 267 | 142 | 339 | 0.419 | 256 | 86.6 | Summarize the plot of King Lear in one... |
| 4 | 1 | 17 | 4 | 278 | 144 | 327 | 0.440 | 256 | 92.9 | Translate to French: The early-morning fog lingered over... |
| 5 | 1 | 19 | 3 | 269 | 182 | 219 | 0.831 | 256 | 127.4 | What are the main causes of the Peloponnesian... |
| 6 | 0 | 18 | 6 | 266 | 150 | 309 | 0.485 | 256 | 136.8 | Describe the role of telomeres in cellular... |
| 7 | 1 | 20 | 5 | 271 | 158 | 288 | 0.549 | 256 | 174.7 | List five practical steps for reducing memory allocations... |
| 8 | 0 | 21 | 7 | 264 | 168 | 259 | 0.649 | 256 | 177.2 | Write a haiku about a printing... |

## Per-prompt summary (aggregated across all completions of that prompt)

| prompt_idx | n_completions | accept | gen | rate | prompt |
|---:|---:|---:|---:|---:|---|
| 0 | 1 | 178 | 226 | 0.788 | Explain the difference between latent diffusion and... |
| 1 | 1 | 142 | 339 | 0.419 | Summarize the plot of King Lear in one... |
| 2 | 1 | 132 | 183 | 0.721 | Write Python code that fits a 2nd-degree polynomial to a... |
| 3 | 1 | 182 | 219 | 0.831 | What are the main causes of the Peloponnesian... |
| 4 | 1 | 144 | 327 | 0.440 | Translate to French: The early-morning fog lingered over... |
| 5 | 1 | 158 | 288 | 0.549 | List five practical steps for reducing memory allocations... |
| 6 | 1 | 150 | 309 | 0.485 | Describe the role of telomeres in cellular... |
| 7 | 1 | 168 | 259 | 0.649 | Write a haiku about a printing... |

## Spread

- min: 0.419
- max: 0.831
- mean: 0.610
- spread: 0.412
