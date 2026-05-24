---
name: n-stream-byte-compat-tradeoff
description: "For PHASE_NSTREAM_KV's 4D K/V reshape, byte-compatible axis order [head_dim, n_head_kv, kvps, n_stream] keeps legacy graph builders untouched but PRECLUDES per-stream allocator + per-stream dispatch — the shortcut delivers only the structural foundation, not Bug C closure"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

When porting the n_stream KV 4D layout (PHASE_NSTREAM_KV_4D), there is
a load-bearing axis-order decision. Choosing
`[head_dim, n_head_kv, kvps, n_stream]` (default contig strides) makes
the byte layout byte-identical to the legacy 2D K
`[head_dim, n_head_kv*kv_size]` at every n_stream value — just
partitioned by `ne[3]`. That preserves the ~30-40 K/V view/copy sites
in `llama-build-context.cpp` and the per-arch graph builders unchanged.

**Why:** Empirical bisect 2026-05-20 on a single session attempting
N1+N2+N3 closure. Per-stream `find_slot` (allocating slot s's cells
in `[s*kvps, (s+1)*kvps)`) under byte-compatible layout: slot-1
single-request returned all-"!"-token garbage. Per-stream
`process_batch_tokens` dispatch (one llama_decode per seq_id run)
under byte-compatible layout: concurrent NP=2 R2 returned garbage
after the first few tokens, even with the decode-side prefill gate
restored. Both reverted, confirming the byte-compatible shortcut and
per-stream semantics are mutually exclusive under unchanged graph
builders.

**How to apply:** Don't attempt the "byte-compatible 4D + per-stream
semantics" combination — it cannot deliver Bug C closure or
TG-overlap recovery. To deliver structural Bug C closure (per-stream
dispatch, gate removal, TG overlap recovered), the axis order must be
the upstream-aligned non-byte-compatible
`[head_dim, kvps, n_head_kv, n_stream]` (heads outer per stream,
positions inner) AND every K/V view/copy site must be rewritten with
stream-aware base offsets (`s * nb[3]`) plus per-stream `n_kv` bounds.
That is the real ~60-100k token N2 scope the plan calls for; the
byte-compatible shortcut is a foundation-only landing, not the
closure. See [[project_production_2026q2_landing]] for the production
shape this touches and [[feedback_no_skipping_lessening]] for the
discipline against shortcuts that don't actually deliver the stated
work.

Submodule commits where N1 byte-compatible foundation landed:
`52d845e9` (initial 4D + per-stream find_slot), `c1beb104` (axis-order
fixup to byte-compatible), `38ea4127` (revert per-stream find_slot
back to legacy global flat scan).

**Update 2026-05-20 — N2 + N3 LANDED on feature branch.** After
planning, the bundle was implemented on `feature/nstream-kv-4d-n2`
(submodule). Bug C closed structurally; decode-side prefill gate
REMOVED. Verified on Qwen3.5-0.8B-BF16 at single-GPU NP=2 and
multi-GPU NP=2 concurrent under gate-removed configuration: both
slots produce coherent output. `test-dflash-np-invariance` PASSes
4/4 seeds byte-identical across N ∈ {1,2,4,8}. Bundle code-complete;
production gates G3.a–G3.h still need to run on Qwen3.6 27B before
merge to `production/2026-q2-next`.

Submodule commits: `0472275d` (N2+N3 main), `95d3c9eb` (multi-device
split + K-shift/defrag gates), `a202f4f4` (n_kv bounding + V split
factoring).

**N2 refined plan landed in PHASE_NSTREAM_KV_4D.md** (parent commit
after `48abcda`). Key decisions:
- N2.a (axis order switch) + N2.b (graph builder rewrites) cannot
  land separately — must be one bundle on a feature branch off
  `production/2026-q2-next`, merged back only when bundle gates pass.
- Per-stream `find_slot` and per-stream `process_batch_tokens`
  dispatch (both tried and reverted in N1 session) can be re-enabled
  ONCE N2.b graph builder offsets are in place.
- Entry-point graph builder helpers: `llm_build_kv_store`,
  `llm_build_kqv`, `build_std_attention` (multi-device branch),
  K-shift, defrag, `update_cache_copies`, `can_reuse_graph`, mask
  builder. ~30-40 K/V view/copy sites total.
- Estimated 120-155k tokens for full N2+N3 bundle.
