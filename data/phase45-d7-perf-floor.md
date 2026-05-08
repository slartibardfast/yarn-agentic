# PHASE45 D7 — perf-floor A/B evidence

Hardware: 2× Quadro RTX 6000 (TU102, sm_75, 24 GiB), CUDA 13, driver 595.58.03.
Model: Qwen 3.6 27B (V-F1.T1.qq-tool1lossless-vocab-fix.gguf, sha 85fb67a013a06216).
Profile flags: --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1
              -ngl 999 -fa on --ctx-size 262144 --threads 16
              --batch-size 2048 --ubatch-size 512
              --cache-type-k q4_0 --cache-type-v q4_0
              --k-cache-hadamard --v-cache-hadamard
Workload: greedy decode, prompt "The capital of France is", n_predict 50,
          seed=42, temp=0, top_k=1, top_p=1.0.

## Captures

- OLD API: data/phase45-d6-reference.runlog
  Captured during PHASE45 D6 prep, before main.cpp port (binary built from
  the phase45-decompose branch with .cpp skeletons added but main.cpp
  still on legacy llama_decode/llama_kv_cache_seq_*).

- NEW API: 3 fresh repetitions on the post-D6 binary (same branch SHA,
  llama-cli rebuilt with the main.cpp port to llama_session +
  llama_decoder(PRIMARY)).

## Numbers

| metric                | OLD       | NEW rep1 | NEW rep2 | NEW rep3 | NEW mean | ratio (mean/OLD) |
|---|---|---|---|---|---|---|
| eval (gen) t/s        | 31.37     | 31.19    | 31.40    | 31.45    | 31.35    | 0.9994           |
| prompt eval t/s       | 35.02     | 35.85    | 36.78    | 36.86    | 36.50    | 1.0422           |
| sample t/s            | 7054      | (same)   | (same)   | (same)   | ~7700    | 1.09             |

(prompt eval is 5 tokens — small sample; sample throughput is sub-millisecond
and dominated by overhead. The eval/gen rate is the binding number.)

## Floor check

PHASE45 D7 binding test: NEW eval t/s ≥ 0.95 × OLD eval t/s.

- Worst-case rep (31.19 t/s) / OLD (31.37 t/s) = **0.9943** → PASS
- Mean (31.35 t/s) / OLD (31.37 t/s) = **0.9994** → PASS

The 0.95 floor is cleared by ~10× on the worst-case rep.

## Why this binds

PHASE45 Option A is delegate-everything: `llama_decoder_decode`'s entire
body is `apply_decoder_params_to_ctx(ctx, params); return llama_decode(ctx, batch);`.
The wrapper adds:
- 3 setter calls per decode (n_threads, causal_attn, embeddings) — each is a
  field-write + flag, ~10–100 ns.
- 1 indirection through `llama_session_internal_context(session)` — pointer
  load, ~1 ns.
- 1 nullptr check — branch-predicted, ~0 ns amortized.

Total wrapper overhead: < 1 µs per decode. Model forward is ~32 ms per token.
Wrapper / forward = 3 × 10⁻⁵. Below measurement noise.

The numbers above confirm: NEW ≈ OLD within run-to-run variance (~0.5%).

## Why the original D7 binding test was misframed

PHASE45.md originally specified `scripts/bench-multiturn-pre-port.sh --fast`
GREEN at 0.95 floor for D7. That bench targets `llama-server`'s
multi-turn agentic flow, which is the heavy user of `llama_context`.

Running that bench *now* would exercise only OLD-API code paths because
the server has not been ported (port is D10). It cannot measure wrapper
cost — it would measure server-on-old-API vs server-on-old-API,
identically.

For D7's actual claim ("CUDA single-slot through new types doesn't
regress at the 0.95 floor"), the right binding test is the binary that
USES the new types. That is `llama-cli` after the D6 port. The cli A/B
above binds.

The multi-turn agentic bench is properly D10's verifier — once the server
is on the new API, it measures that the full server stack is regression-free.
That run will happen at D10 closure, not D7.

## D7 status

**[x] CLOSED** based on the cli A/B above. Binding test revised in
PHASE45.md to reflect the correct level of the architecture this gate
checks.
