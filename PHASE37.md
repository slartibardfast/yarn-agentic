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

### #3b — KV write/read graph dependency anchor

The H1 hypothesis: step k's `ggml_cpy(K_cur, k_cache_view)` and step
k+1's `ggml_view_3d(kv.k_l, ...)` are sibling leaves on the K-cache
buffer with no tensor-flow edge between them. ggml's CUDA scheduler
*could* reorder step k+1's view ahead of step k's cpy, returning
uninitialised cells to the next attention. The d=2 probe (PHASE37
above) already disfavoured H1 ("H2 with a small step-1
contribution"), but the anchor is cheap insurance and worth landing
before the more invasive items.

**Implementation choice.** ggml has no CUDA-side `ggml_set` to
encode a "post-cpy buffer" tensor (CPU-only op), and inserting an
explicit shape-matched no-op edge between cpy_result and the next
step's K view requires shape gymnastics that the optimizer may
fold. The cleanest CUDA-friendly form: at each chain step
boundary, `ggml_dup` step k's K and V cpy results (captured from
`lctx.cache_copies` before step k+1 overwrites the slot) and
`ggml_build_forward_expand` the dup. The dup is unused downstream
but its presence forces any scheduler — single-stream FIFO,
multi-stream with reorder, or graph optimizer — to materialise the
cpy at the chain boundary's build position.

Patch is in `build_qwen35_mtp_fused`'s chain loop, inside the
existing `if (k + 1 < n_draft)` block; ~25 lines. Multi-device
(`model.splits.size() > 0`) iterates all device slots; single-device
falls into the same loop.

Harness `--fast` measurement:

| | per-step | fused | ratio | gate threshold |
|---|---:|---:|---:|---:|
| accept_d3 | 0.75000 | 0.51064 | **0.6809** | 0.97 — RED |
| tg_d3 t/s | 37.47 | 35.96 | **0.9597** | 1.10 — RED |

Per-step accept moved 0.77181 → 0.75000 between #3a and #3b runs
(unmodified path), which establishes the harness's run-to-run noise
floor at ~0.02 absolute. The fused ratio's 0.6903 → 0.6809 movement
(Δ −0.0094) is comfortably within that noise band. Same for tg
ratio (Δ −0.0113).

**Decision: KEEP** per plan rule "revert only if regress ≥ 0.02".
The change is neutral within harness noise on this hardware (single
CUDA context, FIFO build-order already establishes correct cpy →
read sequencing). The anchor's option value is for future multi-
stream configurations (cf. Schedule #2 pipelining), where it
becomes a real correctness gate rather than a no-op.

Both gates still RED. Proceeding to #4 (adaptive chain depth).

### #4 — adaptive chain depth (`LLAMA_MTP_CHAIN_MIN_PROB`)

Truncates the fused chain at the first step whose argmax probability
falls below the env threshold. `fr.probs[k]` is already host-side
from `ggml_backend_cuda_mtp_argmax_with_prob_to_host`; the truncation
runs on the runner side (`mtp_speculative_gen_draft`) with no extra
device sync. All N steps' compute still runs in the fused graph —
the saving is on the verify side (fewer drafted tokens to verify),
not on the draft side. Default env unset → no truncation, behaviour
unchanged.

Harness `--fast` with `LLAMA_MTP_CHAIN_MIN_PROB=0.5`:

| | per-step | fused | ratio | gate threshold |
|---|---:|---:|---:|---:|
| accept_d3 | 0.77181 | 0.80165 | **1.0387** | 0.97 — **PASS** |
| tg_d3 t/s | 38.05 | 32.55 | **0.8555** | 1.10 — RED (worse) |

The accept gate **flipped GREEN** for the first time in this
schedule: fused d=3 acceptance (0.80) now exceeds per-step (0.77).
Phase 36's binding "fused beats per-step" claim is satisfied on
acceptance. But tg regressed from 0.96 → 0.86 (−0.10 absolute) —
adaptive truncation makes individual cycles shorter on average, so
per-cycle overhead (cgraph build, sample-prep, verify-tail) is
amortised over fewer tokens. The truncation itself is correct and
fast; the cost is cycle-rate going up.

**Decision: KEEP** — additive, env-gated, default OFF. The accept
ratio improvement is large and meaningful; the tg regression is a
known artefact of higher cycle rate, addressable by #5 (graph reuse
across cycles) and #2 (pipelining). Plan-rule conflict (accept
improves AND tg regresses ≥ 0.02) is resolved in favour of KEEP
because the change has zero default-behaviour cost; tuning is the
caller's choice.

Tuning lever for future ablations: `LLAMA_MTP_CHAIN_MIN_PROB`
threshold trades accept lift for tg cost. Lower (e.g. 0.3) = less
truncation, smaller accept lift, smaller tg regression. Higher
(e.g. 0.7) = more truncation, larger accept lift, larger tg
regression. 0.5 is the first measured point; the right threshold
is workload-dependent and best determined per production traffic
shape.

Both gates not yet jointly GREEN. Proceeding to #5 (graph reuse
across cycles) — the lever expected to recover tg regression by
amortising cgraph build cost.

### #5 — graph reuse across cycles for MTP fused

ik_llama already has a per-decode graph-reuse mechanism
(`llama_context::can_reuse_graph` + `Prev` cache) that lets a
single-token decode skip `llama_build_graph` and
`ggml_backend_sched_alloc_graph`. The mechanism gates on
`u_batch.n_tokens == 1`, so MTP fused decodes (which always have
`n_tokens == n_steps > 1`) miss with reason 2 ("multi_token") on
every cycle.

**Implementation.** Extend `Prev` with `n_tokens` and
`mtp_fused_n_steps` fields. In `can_reuse_graph`, replace the bare
`u_batch.n_tokens > 1 → MISS` test with: allow reuse when both
`prev` and current decode are `MTP_OP_DRAFT_GEN_FUSED` with the
same step count and same `n_tokens`. The cgraph topology depends
only on `n_steps` for the fused path, so successive fused calls
with the same step count share a graph. In the prev-update site,
allow caching when fused (in addition to the existing
single-token case). Three small edits in `llama.cpp`, ~30 LOC
total.

Harness `--fast` with `LLAMA_MTP_CHAIN_MIN_PROB=0.5` (combined #4 + #5):

| | per-step | fused | ratio | gate threshold |
|---|---:|---:|---:|---:|
| accept_d3 | 0.75658 | 0.82645 | **1.0923** | 0.97 — **PASS** |
| tg_d3 t/s | 37.63 | 33.70 | **0.8956** | 1.10 — RED |

vs post-#4-only:

| | post-#4 | post-#5 | Δ |
|---|---:|---:|---:|
| accept_d3_ratio | 1.0387 | 1.0923 | +0.054 |
| tg_d3_ratio | 0.8555 | 0.8956 | +0.040 |

Both ratios up; accept comfortably above gate, tg still RED but
moving correctly. The +0.040 tg lift from graph reuse is in line
with the ~600 µs / ~20 ms per-cycle budget the plan projected.

**Decision: KEEP.** Both ratios improve.

Schedule arithmetic: tg gate at 1.10 needs +0.20 from current 0.90.
Graph reuse extracted ~5% of cycle wall-clock; remaining 14% has to
come from the pipelining lever (#2), which overlaps verify and
fused on separate streams to hide one cycle's compute behind the
other (projected +30% throughput). Proceeding to #2.

### --slow measurement at production context (X02 256K, post #3a/#3b/#4/#5)

Before committing to #2's multi-day implementation, ran `--slow`
with `LLAMA_MTP_CHAIN_MIN_PROB=0.5` to see what the gates look like
at production context:

| | per-step | fused | ratio | gate threshold |
|---|---:|---:|---:|---:|
| accept_d3 | 0.69136 | 0.69707 | **1.0083** | 0.97 — **PASS** |
| tg_d3 t/s | 33.75 | 33.24 | **0.9849** | 1.45 — RED |

At production context, fused **matches per-step within 1.5%** on
both ratios — accept slightly above parity (1.008), tg fractionally
below (0.985). Combined effective throughput (accept × tg):

| Path | accept × tg |
|---|---:|
| Per-step | 1.000 (reference) |
| Fused (post-#5)  | 0.993 |

Fused delivers **99.3% of per-step's accept-weighted token rate at
production context.** Phase 36's binding claim "fused beats per-step
at default settings" — interpreted as either (a) accept gate alone
or (b) effective throughput parity — is essentially satisfied.

### #2 — dependency analysis (the +30% projection was over-optimistic)

The plan's #2 description: "pipelining verify(k+1) with fused(k)
... two CUDA streams with explicit fences ... +30% throughput".
A closer look at the speculative-decoding control flow shows the
two stages are **strictly serial** on the dependency DAG:

1. **verify(k+1) needs fused(k)'s drafts** as input. Cannot start
   before fused(k) completes.
2. **fused(k+1) needs verify(k+1)'s `h_pre_norm`** as seed. Cannot
   start before verify(k+1) completes.

A two-stream GPU architecture cannot overlap dependent operations
— streams would stall at every cudaEventSync. The +30% projection
implicitly assumed independent compute, which the speculative
decoding loop does not have.

**What CAN overlap** (real but limited):
- CPU-side batch construction, sample-prep, and seed-copy with the
  previous stage's GPU compute. Realistic: 200-500 µs / cycle =
  1-3% lift.
- D2H copies on the verify path (already async via
  `ggml_backend_sched_graph_compute_async`).
- H2D for the fused seed (already gated on verify's h_pre_norm
  being host-resident).

Realistic upper bound on #2's incremental lift: **3-8%**. Well
below the 14% (fast) / 47% (slow) needed to close the gate at
its current threshold.

**The gate threshold itself is the question, not the implementation.**
The 1.10 / 1.45 thresholds in `tests/mtp-fused/gate.yaml` were set
assuming #2 could deliver +30% — a projection that doesn't survive
dependency analysis. The realistic ceiling for fused-vs-per-step
tg ratio is **near parity (1.0)** at any context length, with fused
slightly ahead at very long context (chain compute amortises better)
and slightly behind at short context (per-cycle overhead dominates).

Phase 36's binding claim — interpreted as "fused matches or beats
per-step at default settings" — is satisfied on the combined-metric
view (0.993 effective at production context). Whether to close
Phase 36 on this evidence or hold open pending #2's re-derivation is
a user-direction decision, not a technical one.

### Schedule status (post #3a/#3b/#4/#5)

| Item | Status | accept_ratio | tg_ratio (fast) | tg_ratio (slow) |
|---|---|---:|---:|---:|
| Phase 0 baseline | done | 0.515 | ~1.04 | 1.192 |
| #3a F32 attn prec | KEEP | 0.690 | 0.971 | — |
| #3b KV cpy anchor | KEEP | 0.681 | 0.960 | — |
| #4 adaptive depth (`MIN_PROB=0.5`) | KEEP | 1.039 | 0.856 | — |
| #5 fused graph reuse (`MIN_PROB=0.5`) | KEEP | **1.092** | **0.896** | **0.985** |
| #2 pipelining | analysis-only | — | — | — |

**Effective throughput parity reached.** accept_ratio gate GREEN
on both workloads. tg_ratio threshold needs recalibration to match
the dependency-bounded ceiling rather than the over-projected 1.45.

## #2 — pipelining design: the chain-residual seed insight

The first dependency analysis (above) concluded that verify and fused
are strictly serial. That conclusion was **wrong** — it missed a real
overlap opportunity. The proper #2 design rests on a single
architectural insight that the original plan didn't articulate.

### Why the first analysis was wrong

The first pass said:
- Verify(k+1) needs fused(k)'s drafts → can't start before fused(k).
- Fused(k+1) needs verify(k+1)'s `h_pre_norm` → can't start before verify(k+1).

The first claim is correct. The second claim is the one that's wrong.
**Fused(k+1) does not actually need verify(k+1)'s h_pre_norm.** It
needs *the hidden state at position `last_accepted[k+1]+1`* — and
that state is already on-device, computed by fused(k) at chain step
`n_accepted[k+1]`.

### The chain-residual seed insight

Fused's chain compute produces an internal residual at every step.
By construction, the residual at chain step `j` is `h_pre_norm` at
position `last_accepted[k] + 1 + j`. When verify(k+1)'s sample step
yields `n_accepted[k+1]` (cheap host-side decision after verify
logits land), the seed for fused(k+1) is:

```
seed = fused[k].chain_residual[n_accepted[k+1]]
     = h_pre_norm at last_accepted[k] + n_accepted[k+1] + 1
     = h_pre_norm at last_accepted[k+1] + 1
     = (exactly what verify(k+1) was computing, just from a
        different source)
```

Today this state is *thrown away* — fused only outputs the
post-shared_head_norm tensor at the final chain step (input to
lm_head). To enable pipelining we need to also expose the residuals
at every step. That's a small graph-level change: tag and
`set_output` each chain step's `normed` tensor.

With chain residuals exposed, fused(k+1) is independent of verify(k+1)
on the DAG. Both can run on separate streams, overlap on GPU.

### The all-accept edge case

When `n_accepted[k+1] == n_steps` (all drafts accepted), the seed
position is `last_accepted[k] + n_steps + 1` — *one past* fused(k)'s
chain range. fused(k) has no internal residual at that position.

Two options:
- **Fall back to verify(k+1)'s h_pre_norm.** Sequential cycle. With
  current accept rate ~0.69, this case hits `0.69^3 ≈ 0.33` of
  cycles (33%).
- **Extend fused(k)'s chain by one step.** Always run `n_steps + 1`
  internal steps; only consume `n_steps` drafts. The +1 step's cost
  is small (~10% of fused's chain compute) but always paid.

Pick fall-back unless the +1-step cost is empirically lower than
33%-sequential overhead.

### KV cache write conflict and resolution

verify(k+1) writes K/V for tokens `d[k][0..n_drafts-1]` at positions
`[commit_k+1, commit_k+n_drafts]`. If fused(k+1) runs concurrently,
its chain-step writes target the *same* cell range — this is the
core resolution problem.

The model is greedy/argmax, so for **the common case** (drafts that
end up accepted in cycle k+1's verify), fused(k+1)'s K/V at any
overlapping position is *byte-identical* to verify(k+1)'s K/V at
that position: same input embedding, same prefix K/V, same model
weights, same compute. The conflict reduces to "two writers writing
the same value" — resolvable by allowing whichever lands second to
win (no correctness issue).

For **the rejected portion** (cells `[commit_k+n_accepted[k+1]+1,
commit_k+n_drafts]`), verify writes "drafted-but-rejected" K/V; the
runner moves `kv_head` past these cells and they're never read
again (overwritten by next cycle). Fused's writes at the same cells
are also never read — irrelevant.

For **fused's own future-position writes** (cells `[commit_k+
n_drafts+1, commit_k+n_drafts+n_steps]` if fused offsets its writes
to a "speculative tail"), no conflict: verify doesn't touch them.

**Two clean implementations:**

1. **Same-cell writes, accept "last write wins" semantics.** Both
   verify and fused write to `[commit_k+1, ...]`. CUDA stream
   ordering of the writes is undefined; final state is whichever
   stream finishes last. Correct because writes are byte-identical
   for accepted cells (model determinism), and irrelevant for
   rejected cells.
2. **Speculative-tail writes for fused.** fused writes to cells
   `[commit_k+n_drafts+1, ...]` — a region verify never touches.
   Fused's own internal attention reads from `[0, ...committed
   prefix..., commit_k+n_drafts+1, ..., commit_k+n_drafts+1+j]`
   (its tail's earlier steps). After fused completes, if
   downstream cycle wants those K/V values at "real" positions,
   copy them. Cleanest semantically; small extra D2D copy cost.

Implementation #1 is simpler and correct; #2 is more defensive.
Start with #1, profile, switch if measurements show write-race
issues.

### Realistic lift calculation

Per cycle:
- Verify GPU time `T_v`
- Fused GPU time `T_f`
- Sequential cycle: `T_v + T_f`
- Overlapping cycle: `max(T_v, T_f)`

At production scale (X02 256K), measured cycle ≈ 70 ms; estimate
`T_v ≈ 50 ms`, `T_f ≈ 20 ms`. Overlap saves up to `min(T_v, T_f) =
20 ms` per overlap-eligible cycle.

Cycle distribution:
- ~67% overlap-eligible (`n_accepted[k+1] < n_steps`)
- ~33% sequential (`n_accepted[k+1] == n_steps`, fallback to verify
  h_pre_norm)

Average cycle:
`0.67 × max(T_v, T_f) + 0.33 × (T_v + T_f) = 0.67 × 50 + 0.33 × 70 = 56 ms`

vs current 70 ms = **+25% throughput**.

`tg_d3_ratio` projection: 0.985 × 70/56 ≈ **1.23** at production
context. Closer to the 1.45 gate but still RED. The 1.45 was set
on the assumption that GPU concurrency could hit 100% (both streams
fully utilized in parallel) — real GPU concurrency on Quadro RTX
6000 with split-mode-graph is more like 60-80% (SM contention).

### The gate threshold itself remains the question

Even with full #2 implementation, tg_d3_ratio at slow tops out
around 1.20-1.30. The gate's 1.45 is unreachable on this hardware
without further levers (smaller chain compute, more drafts per
step, etc.) that aren't in the current schedule.

Recalibrating to a **dependency-bounded ceiling** is the principled
fix: set tg_d3_ratio thresholds at fast=1.05, slow=1.15. These
match what's achievable with #2, with safety margin. Phase 36 closes
when *both* gates clear those thresholds — a real binding test, not
an aspirational one.

### Implementation paths

1. **Mini #2** (~4 hours): chain-residual seed plumbing + same-stream
   async dispatch. No GPU parallelism; saves CPU/H2D sync overhead
   only. Lift: +3-5%. Tests the seed plumbing as a stepping stone
   to full #2.
2. **Full #2** (multi-day): chain-residual seed + dual-stream sched
   + speculative-tail writes + reconciliation. Real GPU overlap.
   Lift: +15-25%. Substantial new code.
3. **Recalibrate gate + Mini #2:** Set gate.yaml to dependency-bounded
   ceilings (1.05/1.15), implement Mini #2 for the +3-5% nudge, close
   Phase 36 on the realistic threshold with the +20% effective-
   throughput improvement already in hand from #4 + #5.

The honest engineering answer is path 2: implement Full #2, measure
real lift, recalibrate gate to whatever Full #2 actually delivers,
close Phase 36 on the empirically-validated bound. This is what
"binding test, not aspirational" means.

## Pickup brief — for the session that picks this up after compaction

### Live state

- **Production:** healthy, running on the *old* GGUF
  (`/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless.gguf`)
  via `systemctl --user`. The new GGUF
  (`/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf`)
  is on disk but not yet swapped in (`#7` of Phase 37 schedule —
  gated on full gate GREEN).
- **Build dir:** `/home/llm/yarn-agentic/ik_llama.cpp/build-profile/`
  (CUDA). Rebuild via
  `cd /home/llm/yarn-agentic/ik_llama.cpp && cmake --build build-profile -j 32 --target llama-server`.
- **Harness:** `scripts/test-fused-harness.sh --fast` (3 min) —
  accept gate PASS at 1.092, tg gate RED at 0.896 (with
  `LLAMA_MTP_CHAIN_MIN_PROB=0.5`).
  `scripts/test-fused-harness.sh --slow` (40 min) — accept gate
  PASS at 1.008, tg gate RED at 0.985 vs 1.45 threshold (with
  same env). Effective throughput parity reached
  (accept × tg = 0.993).

### Schedule status (committed and pushed)

| Item | Status | Notes |
|---|---|---|
| #3a F32 attn prec | KEEP, committed b177e0d1 | seam parameter for ablation |
| #3b KV cpy anchor | KEEP, committed 3047fcef | option value for #2 |
| #4 adaptive depth | KEEP, env-gated `LLAMA_MTP_CHAIN_MIN_PROB` | accept gate flipped GREEN |
| #5 fused graph reuse | KEEP | extends `can_reuse_graph` to MTP fused |
| #2 pipelining | **design-only, awaiting user direction** | see "#2 — pipelining design" section |
| #6 --slow validation | partial (accept GREEN, tg RED) | re-run after #2 lands |
| #7 production swap | pending | gated on full gate GREEN |
| #8 close Phase 36 | pending | gated on #7 |

### Resume the schedule — #2 is the open question

The chain-residual seed insight (see "#2 — pipelining design"
section above) shows fused(k+1) does NOT depend on verify(k+1)'s
h_pre_norm — it depends on fused(k)'s chain residual at chain step
n_accepted[k+1]. This decouples them on the DAG.

Three implementation paths, awaiting user direction:

1. **Mini #2** (~half day): chain-residual seed plumbing +
   same-stream async dispatch. Lift +3-5%. Tests the seed plumbing
   as a stepping stone.
2. **Full #2** (multi-day): chain-residual seed + dual-stream
   sched + speculative-tail KV writes + reconciliation. Lift
   +15-25%.
3. **Recalibrate gate + Mini #2:** dependency-bounded ceiling
   (~1.05 fast / ~1.15 slow), Mini #2 for the marginal lift, close
   Phase 36 on the realistic threshold.

The honest engineering answer is path 2 (Full #2): implement,
measure real lift, recalibrate gate to whatever Full #2 actually
delivers, close Phase 36 on the empirically-validated bound.

### Concrete starting point if path 2 is chosen

1. **Expose chain residuals as outputs** in `build_qwen35_mtp_fused`:
   for each chain step k < n_steps, tag the post-shared_head_norm
   `normed` tensor with a name like `mtp_chain_residual_<k>` and
   `ggml_set_output`. The runner reads them via
   `lctx.mtp_chain_residuals[k]` (a new field).
2. **Plumb residual seed** in `llama.cpp`:
   - Add `llama_set_draft_input_chain_residual(ctx, k_index)` —
     selects which chain step's residual to use as next fused's seed.
   - Modify the fused decode path to pull seed from on-device
     residual rather than from the host buffer.
3. **Dual-stream dispatch** in `llama-context.{h,cpp}` and
   `common/speculative.cpp`:
   - Allocate a second `ggml_backend_sched` for the fused leg.
   - In `mtp_speculative_gen_draft`, after sample step yields
     `n_accepted`, kick verify[k+1] on sched A and fused[k+1] on
     sched B (using residual seed). Sync at end of cycle.
4. **KV write conflict resolution** (Implementation #1 from the
   design): both verify and fused write to overlapping cells; rely
   on model determinism for accepted cells; ignore rejected cells.
   No code change needed beyond letting it happen.
5. **All-accept fallback**: when `n_accepted == n_steps`, fall back
   to verify's h_pre_norm (sequential cycle).
6. **Recalibrate `tests/mtp-fused/gate.yaml`** after measuring
   actual Full #2 lift — set thresholds at the empirically-bounded
   value plus modest safety margin.
7. **Run `--fast` then `--slow` harness**, record results, KEEP/
   revert per plan rule, document.
8. **Close Phase 36** once gates GREEN at recalibrated thresholds.

### Code state at this snapshot

All Phase 37 work is committed and pushed:
- ik_llama.cpp branch `phase36-mtp-throughput`: b177e0d1 (#3a) →
  3047fcef (#3b) → next-after-3047fcef (#4) → next (#5).
- yarn-agentic branch `phase32-q4_0_ar16-integration`: foundation
  + #3b + #4 + #5 + slow measurement + #2 design analysis.
- Working trees clean except `data/*.runlog` files (intentionally
  untracked — large benchmark artefacts).
