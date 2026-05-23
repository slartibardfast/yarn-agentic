# Per-prompt DFlash acceptance — `cell-draftmax-2`

Release events: 9 (1 warmup + 8 measured)

Per-task accept (A) / gen (G) values are read directly from the server's per-task `draft acceptance rate = R (A / G)` log line.

## Per-task acceptance (in completion order)

| order | slot | task | prompt_idx | n_past | A | G | rate | bench_tokens | latency_s | prompt |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 3 | -1 | 46 | 19 | 22 | 0.864 | ? | ? | <WARMUP> |
| 1 | 1 | 17 | 0 | 270 | 161 | 188 | 0.856 | 256 | 38.3 | Explain the difference between latent diffusion and... |
| 2 | 0 | 18 | 1 | 267 | 144 | 221 | 0.652 | 256 | 46.2 | Summarize the plot of King Lear in one... |
| 3 | 1 | 19 | 2 | 277 | 143 | 224 | 0.638 | 256 | 87.0 | Write Python code that fits a 2nd-degree polynomial to a... |
| 4 | 0 | 20 | 3 | 269 | 143 | 222 | 0.644 | 256 | 95.2 | What are the main causes of the Peloponnesian... |
| 5 | 1 | 21 | 4 | 278 | 164 | 182 | 0.901 | 256 | 124.5 | Translate to French: The early-morning fog lingered over... |
| 6 | 0 | 22 | 5 | 271 | 141 | 228 | 0.618 | 256 | 141.5 | List five practical steps for reducing memory allocations... |
| 7 | 1 | 23 | 7 | 264 | 166 | 178 | 0.933 | 256 | 161.1 | Write a haiku about a printing... |
| 8 | 0 | 24 | 6 | 266 | 140 | 228 | 0.614 | 256 | 176.8 | Describe the role of telomeres in cellular... |

## Per-prompt summary (aggregated across all completions of that prompt)

| prompt_idx | n_completions | accept | gen | rate | prompt |
|---:|---:|---:|---:|---:|---|
| 0 | 1 | 161 | 188 | 0.856 | Explain the difference between latent diffusion and... |
| 1 | 1 | 144 | 221 | 0.652 | Summarize the plot of King Lear in one... |
| 2 | 1 | 143 | 224 | 0.638 | Write Python code that fits a 2nd-degree polynomial to a... |
| 3 | 1 | 143 | 222 | 0.644 | What are the main causes of the Peloponnesian... |
| 4 | 1 | 164 | 182 | 0.901 | Translate to French: The early-morning fog lingered over... |
| 5 | 1 | 141 | 228 | 0.618 | List five practical steps for reducing memory allocations... |
| 6 | 1 | 140 | 228 | 0.614 | Describe the role of telomeres in cellular... |
| 7 | 1 | 166 | 178 | 0.933 | Write a haiku about a printing... |

## Spread

- min: 0.614
- max: 0.933
- mean: 0.732
- spread: 0.319
