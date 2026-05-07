# Phase 37: Re-opening Phase 36 — the chain-residual gap and what
actually broke fused

## The closure that wasn't

`PHASE36-CLOSURE.md` was written after Step 3's inline-KV hook landed
the +12.3% MTP-vs-nomtp win at d=1. The closure said the fused d≥3
path was broken on "Issue G" — a padded-vocab argmax problem — and
that the GGUF would need re-quantization to fix it.

That story turned out to be wrong on every count. The padded-vocab
theory was wrong. The actual bugs lived nowhere near the fused
builder. And once the real bugs were fixed, the fused path didn't
just work — it beat per-step on throughput at production context.

This phase is the record of finding out.

## What the data refuted

The closure had three load-bearing claims. Phase 37's empirical work
broke each.

### "Fused argmax picks padded vocab indices"

We instrumented the fused path with `LLAMA_MTP_FUSED_STATS=1` and
binned every argmax pick by vocab band. Across the whole production
sweep:

| Band | Picks | Share |
|---|---:|---:|
| `[0, 152064)` trained content tokens | 382 (d=3) / 636 (d=5) | **98.71% / 97.85%** |
| `[152064, 248044)` trained extension | 1 / 2 | 0.26% / 0.31% |
| `[248044, 248077)` added control tokens | 4 / 12 | 1.03% / 1.85% |
| `[248077, 248320)` truly unused | 0 / 0 | 0.00% / 0.00% |

99% real-token picks. The control-token tail was a footnote, not the
bug. Whatever was driving the 18% accept rate at d=3, it wasn't the
vocab.

### "Per-step survives because its sampler chain (top_k/top_p) suppresses bad logits"

We followed the per-step path through the codebase and found it uses
the *same* on-device argmax kernel that fused uses
(`ggml_backend_cuda_mtp_argmax_with_prob_to_host`). No top_k. No
top_p. No softmax-then-truncate. Just argmax over the full vocab.

So both paths were doing identical argmax over identical lm_head
output. If their argmax values disagreed, the *logits* disagreed —
which meant the bug was upstream of the head, in the chain compute or
in the inputs to the chain compute.

### "GGUF re-quantization will fix it"

We re-quantized the GGUF anyway — emitting `output.weight` as BF16
(the source dtype, instead of the lossy F16 cast) and zeroing the
243 truly-unused rows. The fix was correct and small (+5 pp
acceptance at d=3). But it left ~36 pp on the table, confirming the
remaining gap was elsewhere.

## The four bugs

What followed was a debugging arc — each fix exposed the next layer
of the problem. Each closed a measurable amount of the gap. None of
them lived in the fused builder.

### Bug 1: lm_head precision (vocab fix)

**Symptom:** small but real residual noise on padded rows.

**Root cause:** Tool 1 (`scripts/autoround_to_q4_0_gguf.py`) cast
`output.weight` from BF16 (the AutoRound source) to F16 during emit.
F16's 5-bit exponent loses range vs BF16's 8-bit; tail residuals on
unused rows could occasionally outshine real-token logits.

**Fix:** force BF16 emit for `output.weight` and zero rows that the
tokenizer's `vocab` + `added_tokens` don't reach.

**Lift:** d=3 fused 18% → 22.6% accept on X02 256K. Throughput
unchanged.

**Lesson:** the closure's instinct that something around the lm_head
needed cleaning was right. The *theory* was wrong (padded indices
were almost never the picks). The *fix* still helped marginally.

### Bug 2: chain steps wrote K/V to the same cell (KV cell offset)

**Symptom:** unverifiable corruption between chain steps.

**Root cause:** `build_qwen35_mtp_fused`'s chain steps each called
`build_std_attention` with the build-context's `kv_head` member —
set once to `kv_self.head`. All N steps wrote K/V to the same cell.
Step k+1 overwrote step k's write before step k+1's attention could
read it.

**Fix:** thread `kv_head_offset = k` through `build_std_attention`
and `build_qwen35_mtp_kv_only`. Step k now writes to cell `head + k`.

**Lift:** 0 pp on its own. The semantic was wrong but the symptom
was masked by Bugs 3 and 4 — the chain was already broken at step 0
because the seed itself was zeroed before the chain even started.
Fixing this preemptively unblocks pipelining work later.

**Lesson:** a fix can be necessary without being *measurable* until
the deeper bugs are out of the way. A clean architectural fix is
worth landing even when its lift is invisible.

### Bug 3: seed pointer was dangling (`set_draft_input_hidden_state` copy-on-set)

**Symptom:** every fused decode received zeros as its chain seed.

**Root cause:** `llama_set_draft_input_hidden_state` stored a host
pointer into `lctx.draft_input_hidden_state`. The next
`llama_decode`'s `llama_output_reserve` re-pointed `lctx.embd` based
on the new `n_outputs_max` (which differs between verify and fused
decodes — verify produces N+1 logit rows, fused produces N). The
pointer captured before that re-pointing now aimed at stale or
zeroed memory inside `buf_output`.

The diagnostic that exposed it:

```
[mtp-input-chk] op=DRAFT_GEN_FUSED nbytes=20480 first8=[0.000000 0.000000 0.000000 ...]
```

Eight zeros in a row, every fused decode, no exceptions. The seed
the chain had been told was its h_pre_norm was a void.

**Fix:** copy `n_embd` floats into a context-owned `std::vector<float>`
at set-time. The pointer survives any subsequent `llama_output_reserve`.

**Lift:** d=3 fused 22.6% → 35.4% accept on X02 256K. Throughput
22.08 → 26.94 t/s. *The first big lift.*

**Lesson:** tools beat hypotheses. Until we dumped the bytes that
were actually being copied to the device, the bug looked like a
chain-math problem. It was a pointer-staleness problem, three layers
above the chain.

### Bug 4: the seed source was the wrong context (`mtp_speculative_gen_draft`)

**Symptom:** fused chain still drifted relative to per-step despite
the seed-copy fix.

**Root cause:** `common/speculative.cpp`'s `mtp_speculative_gen_draft`
re-read the seed from `llama_get_embeddings_ith(ctx, 0)` immediately
before calling `llama_mtp_fused_draft_invoke`, then passed it back
through `llama_set_draft_input_hidden_state`. But the `ctx` here is
`ctx_mtp` (the MTP-target context — a *separate* `llama_context`
from the verify context `ctx_tgt`), and `ctx_mtp->embd` is never
populated by verify. The runner had already staged the correct seed
via `slot.mtp_hidden_state` and called `set_draft_input_hidden_state`
on `mtp_target` upstream. The re-read was overwriting the correct
seed with whatever happened to be at `ctx_mtp->embd[0]` — typically
zeros from a freshly-allocated buffer.

**Fix:** in `mtp_speculative_gen_draft`, pass `nullptr` to
`llama_mtp_fused_draft_invoke`. The invoke now skips the
`set_draft_input_hidden_state` call when given `nullptr`, so the
runner's earlier set persists.

**Lift:** d=3 fused 35.4% → **52.0%** accept on X02 256K. Throughput
26.94 → 34.30 t/s. *The dominant lift.*

**Lesson:** speculative-decoding architectures with multiple llama_context
instances need plumbing-level discipline. The runner sets state on
context A; downstream code reads from context A's neighbour B. If A
and B share storage names but not storage, you get a class of bugs
that look like compute bugs and aren't.

## Cumulative measurement, X02 256K, post all four fixes

| | nomtp | d=1 | d=3 | d=5 |
|---|---:|---:|---:|---:|
| Per-step (target) | 31.10 | 32.04 / 88.2% | 28.78 / 58.4% | 28.77 / 58.4% |
| Fused (Phase 0) | 31.10 | 35.41 / 84.5% | 22.08 / 22.6% | 18.44 / 15.7% |
| Fused (post-fix) | 31.31 | 34.34 / 80.2% | **34.30 / 52.0%** | 23.44 / 27.1% |

Phase 36 net at production context: **+18% throughput over per-step at
d=3**, 6 pp accept gap remaining. The d=5 path still trails per-step
on tg, indicating the chain-depth divergence compounds.

## The no-op refactor (#3 attempt)

We hypothesised the residual gap was from divergent *primitives*
between per-step's `build_qwen35_mtp` and fused's
`build_qwen35_mtp_kv_only` + manual FFN/norm. The primitives have
different `cb` tags, slightly different code paths around inp_out_ids,
and the optimizer might pick different kernels for them.

We added a unified `build_qwen35_mtp_chain_residual` helper that does
exactly what per-step's primitive does, ending at result_norm.
Routed the fused chain through it.

Result: **0 pp lift.** Fused acceptance ratio unchanged at 0.6674
small-prompt / 0.890 X02. The refactor was equivalent at the op
level — same ops, same kernel selection, same numerical result.

This was the most informative *negative* result of Phase 37. It
proved the per-step semantic at the *chain-step level* is already
implemented correctly in fused. The gap is somewhere cross-step.

## The d=2 probe

Two hypotheses for the cross-step gap:

- **H1 — KV cache write/read race within the graph.**
  `ggml_cpy(K_cur, k_cache_view)` for step k and `ggml_view_3d(kv.k_l, ...)`
  for step k+1's attention are sibling leaves on the underlying
  K-cache buffer. ggml's CUDA scheduler topologically orders ops
  *connected* by tensor flow; the cpy and the view aren't connected
  (the view sees the underlying buffer, not the cpy's output). Step
  k+1's read could see uninitialised cells.

- **H2 — cumulative numerical drift.**
  Step 0 + step 1 are correct, but the chain residual passing on-device
  through the single graph compounds enough numerical noise by step 2
  to flip argmax frequently.

The discriminator: fused d=2 accept ratio. If H1, the d=2 ratio looks
like the d=3 ratio (low). If H2, the d=2 ratio is close to per-step
parity and only d=3 shows the drift.

```
================ Phase 36 #3 d=2 probe ================
  per-step  d=2   accept=0.80292   tg=36.88 t/s
  fused     d=2   accept=0.74375   tg=39.76 t/s
  accept ratio (fused/perstep) = 0.9263
  (Reference: d=3 ratio = 0.6674 small synthetic)
============================================================
```

**Verdict: H2 with a small step-1 contribution.** The d=2 ratio
(0.93) is much closer to per-step than the d=3 ratio (0.67). Step 1
is mostly fine. Step 2 onwards is where divergence compounds.

A side observation: fused d=2 already beat per-step on throughput
(39.76 vs 36.88 t/s, +8%) despite the lower acceptance. The
single-compute architecture is throughput-positive at any chain
depth; closing the accept gap is pure additional headroom.

## The ggml_dup intervention

H2 implies the right intervention is to break ggml's cross-step
kernel fusion at chain-step boundaries. ggml_dup is identity in F32;
its purpose is to force the optimizer to materialize the chain
residual rather than fuse step k+1's input-side ops into step k's
output-side kernels.

```cpp
if (k + 1 < n_draft) {
    normed = ggml_dup(ctx0, normed);
    cb(normed, "mtp_chain_residual", il_mtp);
}
prev_residual = normed;
```

**Lift:** d=3 small-synthetic ratio 0.6674 → 0.7103, throughput ratio
0.9537 → 0.9861.

Partial. ggml_dup helped, but didn't bridge to the gate (0.97).

## Per-step matching probability — what it actually shows

Solving for the per-chain-step matching probability `p` from the
acceptance rate (`expected_accepts/cycle = p + p² + ... + pᴺ`,
`accept_rate = that_sum / N`):

| Path | p at d=2 | p at d=3 |
|---|---:|---:|
| Per-step | 0.87 | 0.90 |
| Fused (post-ggml_dup) | 0.82 | 0.72 |

Per-step's per-chain-step probability is roughly depth-independent
(≈0.88). Fused's degrades with chain depth (0.82 at d=2, 0.72 at
d=3). This isn't a uniform "every step has slightly worse precision"
— it's a *step-specific* degradation that gets worse the deeper into
the chain you go. Step 2 carries most of the gap.

ggml_dup at every interior step boundary closes some cumulative
fusion-driven drift but not the step-2-specific factor. That points
at remaining work in either F32 attention precision (ratchets all
chain ops up to F32, costs ~5%) or KV-read graph-dependency
injection (forces step k+1's read to depend on step k's write
through the graph, ~30 lines of plumbing).

## Why this is the architectural cost of single-compute

Per-step gets its high `p` because each chain step is a *separate*
graph compute. The output of step k is forced to host (D2H), then
fed back to step k+1 (H2D). That round-trip materializes the
residual at full F32 precision in well-defined memory, isolated from
any cross-step optimization the graph compiler might want to do.

Fused keeps everything on-device through the chain. The single
graph_compute saves dispatches and reduces wall-clock time. But it
also exposes the chain to whatever graph-level optimizations ggml
chooses — and those optimizations were tuned for monolithic
forwards, not for short chains where cross-step memoryfences matter.

That tradeoff is real. The four bug fixes recovered most of the
intended gain. The residual gap reflects the architectural choice;
it can be closed further but at increasing cost per pp.

## What the harness teaches us

`scripts/test-fused-harness.sh` runs per-step and fused on the same
workload and asserts ratios to gate.yaml thresholds. Currently:

```
fast (small synthetic, ctx=4K, ~3 min):
  accept_d3_ratio  threshold: 0.97   measured: 0.7103   FAIL
  tg_d3_ratio      threshold: 1.10   measured: 0.9861   FAIL

slow (X02 ctx=256K, ~40 min):
  accept_d3_ratio  threshold: 0.97   last measured: 0.890  FAIL
  tg_d3_ratio      threshold: 1.45   last measured: 1.192  FAIL
```

The thresholds are aspirational — set to the targets after #3 (per-
step parity) and #2 (pipelining) land. They are *meant* to be RED
right now. Each subsequent fix moves the ratios toward green; the
gate is what tells us we've actually closed the gap rather than
just believed we did.

## Remaining schedule

Each step is test-first against the harness — implement, run --fast
(~3 min), watch the ratios move, commit if they did, move on.

1. **#3 deeper — F32 attention precision in chain attention.**
   Ratchet `ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32)` inside
   `build_qwen35_mtp_chain_residual`. Tests whether the step-2
   precision-loss factor is in attention specifically. ~5% perf cost
   upper bound; lift if precision is the cause.
2. **#3 deeper — KV write/read explicit graph dependency.**
   Inject step k+1's K read view through step k's cpy output so the
   graph orders them. Closes any remaining KV-stale contribution.
3. **#4 — adaptive chain depth.** Once #3 is at its ceiling, pick
   chain depth dynamically: shrink when accept probability is low,
   extend when high. Compounds with #3.
4. **#5 — graph reuse across cycles.** Cache one graph per chain
   depth, swap inputs only on subsequent cycles. Removes ~600 µs
   build cost per cycle.
5. **#2 — pipelining verify(k+1) with fused(k).** The largest
   remaining headroom (+30% throughput). Two-stream architecture
   with explicit KV fences. Bigger rewrite.
6. **--slow harness pass.** Validate at production context (X02
   256K) that all gates are GREEN.
7. **Production swap** — Phase 36 Step 4 against the new vocab-fix
   GGUF. Profile path update + service restart.
8. **Close Phase 36 properly** — update `PHASE36-CLOSURE.md` to
   point at this document; supersede any wrong claims.

## Schedule progress

Per-item harness measurements as the schedule lands. Format: code
change → build → `--fast` → numbers below → keep/revert decision.
Both gate ratios in absolute form; baseline references `gate.yaml`'s
Phase 0 row unless noted.

### #3a — F32 attention precision in chain attention

Threaded a new `fa_prec_f32` parameter through `build_std_attention`
(default `false` everywhere) and set it `true` from
`build_qwen35_mtp_chain_residual` only. The fused chain's
FlashAttention reduction now runs at F32; per-step
(`build_qwen35_mtp`) and KV-only (`build_qwen35_mtp_kv_only`)
unchanged.

Harness `--fast` measurement on the vocab-fix GGUF (synthetic prompt,
ctx=4K):

| | per-step | fused | ratio | gate threshold |
|---|---:|---:|---:|---:|
| accept_d3 | 0.77181 | 0.53275 | **0.6903** | 0.97 — RED |
| tg_d3 t/s | 37.97 | 36.87 | **0.9710** | 1.10 — RED |

Per-chain-step matching probability (solved from
`accept_rate = (p + p² + p³)/3`):

| Path | p (this measurement) | p (post-ggml_dup, PHASE37 prior table) |
|---|---:|---:|
| Per-step | 0.875 | 0.90 (small-synthetic) |
| Fused | 0.717 | 0.72 (small-synthetic) |

Net gate movement: ratio **0.515 → 0.6903** vs `gate.yaml`'s Phase 0
baseline (which pre-dates ggml_dup as well as #3a). Absolute fused
accept moved 0.3975 → 0.53275 (+0.135). Per-step fused-`p` is
within rounding of the post-ggml_dup small-synthetic value, so #3a's
incremental contribution on top of ggml_dup is consistent with
"≈0 — within noise"; most of the +0.175 gate-ratio lift is
attributable to ggml_dup, which post-dated the `gate.yaml` baseline.

**Decision: KEEP.** Either-ratio rule satisfied (both move up vs the
only directly comparable baseline). The `fa_prec_f32` parameter is a
clean seam — if a future ablation shows F32-prec is actually neutral,
flip the call site to `false` without further plumbing.

Both gates still RED. Proceeding to #3b (KV write/read graph
dependency injection).

## Pickup brief — for the session that picks this up after compaction

### Live state

- **Production:** healthy, running on the *old* GGUF
  (`/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless.gguf`)
  via `systemctl --user`. The new GGUF
  (`/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf`)
  is on disk but not yet swapped in.
- **Build dir:** `/home/llm/yarn-agentic/ik_llama.cpp/build-profile/`
  (CUDA, IK_PRINT_TIMING=1). Rebuild via
  `cd /home/llm/yarn-agentic/ik_llama.cpp && cmake --build build-profile -j 32 --target llama-server`.
- **Harness:** `scripts/test-fused-harness.sh --fast` (3 min, RED).
  `scripts/test-fused-harness.sh --slow` (40 min, RED).
- **Probe:** `scripts/probe-fused-d2.sh` (single fused vs per-step
  d=2 comparison; H1/H2 discriminator).

### Code changes landed (all in ik_llama.cpp submodule, uncommitted)

1. `scripts/autoround_to_q4_0_gguf.py` — vocab fix (BF16 output.weight,
   zero unused rows).
2. `src/llama-build-context.{h,cpp}`, `src/graphs/build_qwen35.cpp` —
   `kv_head_offset` parameter threaded through `build_std_attention` and
   `build_qwen35_mtp_kv_only`. New helper `build_qwen35_mtp_chain_residual`.
   Fused chain uses helper + `ggml_dup` at step boundaries.
3. `src/llama-context.h`, `src/llama.cpp` — `draft_input_hidden_state_buf`
   copy-on-set; cycle counter (`mtp_cycle_counter`) for diagnostics;
   env-gated stats prints `LLAMA_MTP_FUSED_STATS`, `LLAMA_MTP_INPUT_CHECKSUM`,
   `LLAMA_MTP_CYCLE_DBG`.
4. `common/speculative.cpp` — fused invoke called with `nullptr` seed
   so the runner's `set_draft_input_hidden_state` persists.
5. `src/llama.cpp` — `llama_mtp_fused_draft_invoke` accepts `nullptr`
   seed; only calls set when seed is given.

### Test artefacts (yarn-agentic root, uncommitted)

- `PHASE37.md` — this document.
- `SUMMARY.md` — updated with PHASE37 + missing PHASE36 entries.
- `scripts/test-fused-harness.sh` — quality gate.
- `scripts/probe-fused-d2.sh` — H1/H2 discriminator (used).
- `tests/mtp-fused/gate.yaml` — thresholds.
- `tests/mtp-fused/test-mtp-fused-chain-residual.cpp` — unit test
  (compiles; needs an MTP-init fixture before it runs end-to-end —
  worst-case-init triggers a KV-cache assert that production avoids
  via its specific init sequence).

### Resume the schedule

The next concrete action is **#3 deeper — F32 attention precision**.
The change is one line inside `build_qwen35_mtp_chain_residual`'s
attention call (or right after it):

```cpp
ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
```

…then `cd /home/llm/yarn-agentic/ik_llama.cpp && cmake --build build-profile -j 32 --target llama-server`,
then `bash scripts/test-fused-harness.sh --fast`. Note the new
`accept_d3_ratio` and `tg_d3_ratio` and write them into PHASE37.md
under a new section heading.

Then proceed down the schedule list.
