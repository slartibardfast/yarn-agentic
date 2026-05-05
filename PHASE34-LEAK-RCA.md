# PHASE 34 — Production OOM RCA

## Symptoms

Production `llama-server` (qwen36-27b-x1, ik_llama @ `32da7ca1`) crashed with
`CUDA error: out of memory` → SIGABRT after agentic prefill bursts under
real OpenCode traffic.

Timeline observed in `/tmp/snap-llama-server/run-20260505T154037/` before
the reboot:

| snapshot | host RSS | GPU0 used | GPU1 used | GPU0 free | GPU1 free |
|----------|----------|-----------|-----------|-----------|-----------|
| baseline | 7.1 GiB  | ~22 GiB   | ~22 GiB   | ~2 GiB    | ~2 GiB    |
| `leaking-1` (T+12 min) | 7.1 GiB | trending up | trending up | shrinking | shrinking |
| `near-oom` (T+25 min) | 23.2 GiB | +3.5 GiB | +3.5 GiB | <0.5 GiB | <0.5 GiB |
| `pre-oom`  (T+36 min) | 25.9 GiB | crashed shortly after | | | |

Crash signature: `CUDA error: out of memory` →
`status=6/ABRT, code=dumped`.

## Root cause — two independent unbounded growth paths, both bounded
in code but bounded too high for this VRAM-constrained config.

### Primary (GPU OOM): `cuda_graphs` cache cap is 3.5 GiB on Turing

`ggml/src/ggml-cuda.cu:4258-4297` — the cache stores
`cudaGraphExec_t` instances keyed by graph topology. Capped at 128 entries
by default (`GGML_CUDA_GRAPH_MAX` env override).

A single 27B inference graph instance on Turing is ~25-30 MiB of device
memory. **128 × ~30 MiB = ~3.8 GiB per device** — and we have two devices.

The active profile uses `--ctx-size 1048576 --parallel 1 --batch-size 2048
--ubatch-size 512` with a 27B-MoE model that exercises wildly varying
shapes during prefill (chunked into 512-token ubatches, residual-window
attention with shifting window, MTP-off but checkpoint-restore micro-steps).
Each distinct shape is a fresh cache key.

Server starts with **only ~2 GiB free per GPU**. As the graph cache fills
toward the cap, free shrinks linearly until `cudaMalloc` fails. Eviction
*does* work — but the steady-state at full cap (~3.5 GiB per device) is
already past the OOM threshold.

This is the crash cause.

### Secondary (host RSS): two host caches sized too generously

1. **`--ctx-checkpoints 64`** — `examples/server/server-context.cpp:3349`.
   Each checkpoint is a `std::vector<uint8_t>` of `llama_state_seq_get_size`
   bytes. At `--ctx-size 1048576` with q4_0 KV + RHT, observed size ≈ 150 MiB.
   **64 × 150 MiB = 9.6 GiB host per slot.**

2. **`--cache-ram 40960`** — `examples/server/server-task.cpp:1137-1204`.
   Server-prompt cache holds full `llama_state_seq` blobs for similar-
   prefix retrieval. Capped at 40 GiB.

Both caps are enforced (we audited the eviction paths). The host OOM
is bounded by 40 + 9.6 ≈ 50 GiB. Crash hit at ~26 GiB host RSS, so the
host was *not* the actual trigger — but it was on its way.

The MEMORY.md auto-memory `feedback_no_tmp_for_large_artifacts.md`
applies indirectly: under host-RSS pressure /tmp tmpfs would be the next
to fail.

## Why this only just started biting

PHASE 33 enabled the per-block `qnext_dispatch` analyzer + ssm_conv
runtime fast-path. That changes the set of graph-shape variants the
backend produces (more conditional paths, more distinct cache keys).
The cap was acceptable when only ~30 distinct shapes were generated in a
session; under PHASE 33 traffic patterns the cache can reach 128 in
minutes.

`master b768f0843` (auto-memory) bundled in 12k lines of unrelated
changes — confirming the env where this regression slipped in is dense.

## Mitigations (ordered by ship cost)

### M1 — env-var lower-bound the graph cache (no rebuild)

Edit `profiles/qwen36-27b-x1.sh`:

```diff
 exec /home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-server \
+    # Cap CUDA graph cache to keep ~2 GiB GPU headroom for cublas / pool growth
+    # under shape-varying agentic traffic. Default 128 → 3.8 GiB on Turing.
+    GGML_CUDA_GRAPH_MAX=24 \
     -m /opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless.gguf \
```

(The env must be exported before the exec, so use `env GGML_CUDA_GRAPH_MAX=24
exec ...` or move it to a `Environment=` line in the systemd unit if the
profile is sourced.)

24 entries × ~30 MiB = ~720 MiB per device. Conservative; raise to 32-48
once we know steady-state hit-rate.

### M2 — reduce host caches to match observed working set

```diff
-    --cache-ram 40960 \
+    --cache-ram 8192 \
-    --ctx-checkpoints 64 \
+    --ctx-checkpoints 16 \
```

Bounds host pressure at 8 + (16×150 MiB) ≈ 10.4 GiB instead of ~50 GiB.

### M3 — proper fix (next phase)

The cuda_graphs cap of 128 is a heuristic, not derived from available
VRAM. Make `ggml_cuda_get_graph_cache_max` query
`cudaMemGetInfo`-derived headroom at backend init and pick a cap that
leaves a safety margin. Same for `--cache-ram` if its working set
exceeds host ratio of total RAM.

## Verification plan

1. Apply M1 + M2.
2. Restart the service.
3. Run a soak with the same agentic traffic pattern that crashed it
   (snoop replay or live OpenCode session).
4. Monitor with the existing heartbeat script — assert GPU0/1 free stays
   >= 1 GiB and host RSS stays < 16 GiB after 60 min.
5. If stable, raise `GGML_CUDA_GRAPH_MAX` to 48 and repeat.

## Action items

- [ ] Apply M1 + M2 to `profiles/qwen36-27b-x1.sh`.
- [ ] Add the env var via systemd `Environment=` in `~/.config/systemd/user/llama-server.service` so it doesn't depend on shell-style env on the exec line.
- [ ] Soak.
- [ ] (Followup) Implement M3 in ik_llama and upstream a PR.
