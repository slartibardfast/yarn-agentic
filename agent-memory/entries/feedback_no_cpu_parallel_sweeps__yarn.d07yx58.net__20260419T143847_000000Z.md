---
name: llama-quantize -t flag works — you can overlap if you cap both sides
description: llama-quantize honors -t N (caps at ~N+4 threads including overhead); supersedes earlier belief it pinned all cores. Overlap with other CPU work is safe as long as the total threads across siblings fits the host.
type: feedback
originSessionId: c1440a07-71f9-4a7a-a10b-10db7f6c5c11
---

`llama-quantize`'s positional `nthreads` argument (`llama-quantize <in> <out> <ftype> <nthreads>`) **does** cap the thread pool. Observed on this host (Ryzen 9 3950X, 32 HW threads):

| `-t N` / positional arg | peak threads (ps -o nlwp) |
|---|---|
| 1  | 5  (1 worker + ~4 overhead) |
| 4  | 9  |
| 16 | 20 |

Verified 2026-04-19 against the current build-vk at `/home/llm/yarn-agentic/llama.cpp/build-vk/bin/llama-quantize`. The code path is `ggml-quant.cpp:1127-1130, 1441, 1608, 1617` — `nthread_use = max(1, min(nthread, nchunk))` per tensor.

**Why this memory changed:** Earlier entry (same file, now overwritten) claimed `llama-quantize` pins all cores regardless of `-t`. That was written when a parallel sweep looked slow — but the real cause was something else (5 parallel quantize siblings + a bench: the slowdown was real workload contention across several nthreads=default processes, not one hidden-all-core pinning). Re-verified today with scaling test `-t 1, 4, 16` → peak threads scale linearly; earlier conclusion was wrong.

**How to apply:**
- Overlap two `llama-quantize` runs by passing small `-t` to each, e.g. `-t 8` per process for two siblings on a 32-thread host.
- `llama-perplexity` and `llama-bench` also respect `-t`. Budget the total.
- The previous "sequential only" rule was overcautious. Sequential is still fine if you're not in a hurry, but parallel with explicit thread caps works.
- When scheduling, watch actual `ps -o nlwp` rather than trusting the `-t` value to be exact — quantize adds ~4 threads of overhead (main + a few ggml-base threads), so `-t 28` on a 32-thread host leaves almost no slack.

**Caveat**: the positional-argument syntax (`llama-quantize in out ftype nthreads`) is what's tested; if someone passes `-t` as a flag (GNU-style), verify it's parsed the same way before building the sweep around it.
