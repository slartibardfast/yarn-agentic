# PHASE45 ↔ PHASE39 integration check

Concrete map of how PHASE39's inline MTP head plugs into PHASE45's
`DRAFT_MTP` decoder role.

Verdict: **plugs in cleanly**. The dispatch surface PHASE45 needs is
already there as `cparams.mtp_op_type`. One locked PHASE45 decision
("no INLINE_KV hook needed because draft is single canonical writer of
layer N-1") needs an empirical re-check before D8 because it changes
the cost balance PHASE36/37/38 worked around.

## 1. Today's MTP graph dispatch

`cparams.mtp_op_type` (set per-`llama_context`, mutated via
`set_mtp_op_type()`) selects which graph the build path emits.
Dispatch happens inside `build_qwen35moe()` and `build_qwen35()`:

| `mtp_op_type` | Graph emitted | Producer |
|---|---|---|
| `MTP_OP_NONE` (+ `mtp_inline_kv_hook=true`) | full transformer + side-effect K/V write at layer N-1 | VERIFY today |
| `MTP_OP_NONE` (+ `mtp_inline_kv_hook=false`) | full transformer; layer N-1 stays untouched | (rare; needs separate `MTP_OP_UPDATE_ACCEPTED` decode) |
| `MTP_OP_DRAFT_GEN` | MTP tail-only graph; one draft token | DRAFT (per-step) |
| `MTP_OP_DRAFT_GEN_FUSED` | fused multi-step chain (PHASE38 C) | DRAFT (chain) |
| `MTP_OP_UPDATE_ACCEPTED` | KV-only refresh of layer N-1 after accept | (legacy, eliminated by INLINE_KV hook) |
| `MTP_OP_WARMUP` | warmup pass | factory init |

Citations:
- `src/graphs/build_qwen35.cpp:8-11` — DRAFT_GEN_FUSED early return.
- `src/graphs/build_qwen35.cpp:20-36` — non-NONE non-FUSED → MTP tail-only via `build_qwen35_mtp`.
- `src/graphs/build_qwen35.cpp:101-111` — INLINE_KV hook inside VERIFY when `mtp_inline_kv_hook=true`.
- `src/llama.cpp:566` — `llama_mtp_op_type` defined.
- `src/llama.cpp:698-701` — `set_mtp_op_type()` runtime mutator.
- `src/llama.cpp:4960-4965` — diagnostic op_label switch.

## 2. PHASE45 role → mtp_op_type mapping

Headers already carry the right knobs in `llama_decoder_params`:

```c
int   mtp_fused_n_steps;     // 0 = not fused
int   mtp_fused_n_extend;    // PHASE38 C extended chain
bool  mtp_inline_kv_hook;    // PHASE36 Step 3 KV write hook
```

The decoder constructor translates role + these flags to
`mtp_op_type` on the borrowed cparams view it uses for graph build:

| `decoder.role` | `mtp_op_type` set | Notes |
|---|---|---|
| `LLAMA_DECODER_PRIMARY` | `MTP_OP_NONE`, hook=false | standalone forward; no MTP at all |
| `LLAMA_DECODER_VERIFY` | `MTP_OP_NONE`, hook=PHASE45 lock says false | full transformer; layer N-1 NOT written here |
| `LLAMA_DECODER_DRAFT_MTP` | `MTP_OP_DRAFT_GEN` if `n_steps==0`, else `MTP_OP_DRAFT_GEN_FUSED` | tail-only or fused chain |
| `LLAMA_DECODER_TREE_BRANCH` | `MTP_OP_DRAFT_GEN` with branch-specific inputs | (D9+; not in scope for D6) |

So the existing graph dispatch is sufficient. No new graph-builder
plumbing is required for D8. The `llama_decoder_create` body sets
`session->ctx->cparams.mtp_op_type` based on role + params.

⚠ **Per-decoder `cparams` view**: today `cparams` is shared across
all forwards via `lctx`. If VERIFY and DRAFT_MTP share an internal
`llama_context` (Option A from PHASE45_D6_SPLIT.md), they must
flip `mtp_op_type` on entry to `decode()` — same as today's
`set_mtp_op_type()` pattern. PHASE45's `llama_decoder_decode` body
does this mutation as the first step.

## 3. VERIFY → DRAFT_MTP handoff

PHASE39 already implemented the data flow. The hidden state used by
the MTP head is `h_pre_norm` (input to final norm + lm_head, tagged
inside `build_qwen35moe()` when `lctx.cparams.mtp` is set):

- `src/graphs/build_qwen35.cpp:80-92` — VERIFY tags pre-final-norm
  residual as `h_pre_norm`, sets it as graph output, stashes pointer
  on `lctx.t_h_pre_norm`.
- `src/llama-context.h:358` — `lctx.t_h_pre_norm` is the addressable
  output tensor.
- `src/llama-context.h:350` — `lctx.inp_mtp_states` is the input
  tensor MTP-tail-only graphs read.

The handoff today is intra-context: VERIFY's output buffer holds
`h_pre_norm` data; the next forward (DRAFT_GEN) sets
`inp_mtp_states` via `prepare_mtp_graph_inputs` which D2Ds from
`t_h_pre_norm`'s output region.

For PHASE45:
- Both decoders share the same internal `llama_context` (Option A).
- Both `t_h_pre_norm` and `inp_mtp_states` live on that shared ctx.
- The handoff is the same D2D copy; nothing to redesign.

If we eventually move to two physical contexts (e.g., DRAFT on a
separate stream), the handoff becomes a real cross-stream copy.
Out of scope for D6/D8.

## 4. Layer N-1 single-canonical-writer concern

PHASE45.md §"PHASE39 collapsed-context MTP wrapping" locks:

> Layer N-1 (the MTP head's K/V slot) is written exclusively by the
> draft decoder — single-canonical-writer, no race, no INLINE_KV
> hook needed.

This is architecturally clean (no race) but **inverts a measured
PHASE36/37/38 win**:

- PHASE36 Step 3 introduced `mtp_inline_kv_hook` to fold the layer
  N-1 KV write into the verify forward. Before the hook, accepted
  drafts required a separate `MTP_OP_UPDATE_ACCEPTED` decode. The
  hook removed a per-accept D2H sync and a separate decode call.
- PHASE45's "no INLINE_KV hook" means: when DRAFT_MTP runs, it does
  the layer N-1 K/V writes itself (via `build_std_attention` inside
  `build_qwen35_mtp`). When the draft is accepted, the writes are
  already in place — no `UPDATE_ACCEPTED`.

That works correctly. The concern is **performance**: the writes
happen during DRAFT decode, which is on the critical path of every
speculation, while the INLINE_KV hook amortized them into the
verify forward (which runs once for all accepted tokens together).

Net effect depends on the accept rate:
- High accept (~70% measured): DRAFT writes that get committed are
  free; DRAFT writes that get rolled back are wasted. PHASE45's
  rollback via `kv_txn` drops them clean.
- Low accept: more wasted DRAFT writes; INLINE_KV hook would have
  written less (only for accepted positions in the verify forward).

**Action**: do NOT remove `mtp_inline_kv_hook` in D8 implementation.
Keep it on the decoder_params, default true for VERIFY+DRAFT
configurations, false only for tree branches. Re-measure at D8 with
the multi-turn agentic bench (MTP_OP_DRAFT_GEN with hook on vs hook
off). The PHASE45.md lock is provisional; this audit reopens it.

(Self-check against CLAUDE.md §4 "no follow-up cover": this is a
real D8 binding-test question, not a future-improvement question.
The +19% target in D8 may or may not hold depending on this
decision; we measure before declaring D8 closed.)

## 5. Recurrent state interaction

`build_qwen35moe()` standard path drives `delta_net` for hybrid
recurrent layers (DeltaNet). The MTP-tail-only graph (`MTP_OP_DRAFT_GEN`)
does NOT touch recurrent state — it runs only the MTP layer (layer
N-1), which is a standard transformer block, not a recurrent one.

- Citations: `src/graphs/build_qwen35.cpp:38, 53-58` (recurrent
  delta_net used in standard verify) vs `src/graphs/build_qwen35.cpp:36`
  (DRAFT skips delta entirely).

PHASE45 implications:
- VERIFY decoder owns the recurrent state trajectory. DRAFT_MTP
  borrows the session but does not advance recurrent state. ✓
- For multi-slot D9: recurrent state is per-(seq_id × layer), batched
  inside VERIFY's forward. DRAFT_MTP per slot runs the MTP layer
  only — no recurrent footprint. ✓
- Tree-branch D9+: still gated by the project memory entry
  `tree_fanout_hybrid_recurrent_blocker`. PHASE45 does not unlock
  this; the integration here is single-draft only.

## 6. Sampling stays external

DRAFT_MTP graph ends at `build_output` (lm_head):
- `src/graphs/build_qwen35.cpp:291-292` — `build_output` produces
  `result_output` (logits over vocab).

The decoder returns logits via `llama_decoder_get_logits_ith`. The
spec_loop applies the sampler chain externally. PHASE45's lock holds
without modification.

## 7. Concrete D8 integration plan

1. `llama_decoder_create(VERIFY)` body:
   - `cparams.mtp_op_type = MTP_OP_NONE`
   - `cparams.mtp_inline_kv_hook = params.mtp_inline_kv_hook` (default true)
2. `llama_decoder_create(DRAFT_MTP)` body:
   - Branch on `params.mtp_fused_n_steps`:
     - `0` → `cparams.mtp_op_type = MTP_OP_DRAFT_GEN`
     - `>0` → `cparams.mtp_op_type = MTP_OP_DRAFT_GEN_FUSED`,
       carry `mtp_fused_n_steps` and `mtp_fused_n_extend` through to
       `build_qwen35_mtp_fused`
3. `llama_decoder_decode(decoder, batch)` first step: flip
   `session->ctx->cparams.mtp_op_type` to the decoder's role-bound
   value before calling `llama_decode_internal`. Restore? Not
   needed — next decoder's decode flips to its own value.
4. Spec_loop step:
   - VERIFY decode → produces logits + `t_h_pre_norm` updated.
   - Apply sampler to logits; compare against draft.
   - Accept-prefix → `kv_txn_commit` for DRAFT's layer N-1 reservation.
   - Reject-tail → `kv_txn_rollback` for DRAFT's layer N-1 reservation
     (drops the K/V cells the draft wrote).
   - Repeat.
5. Multi-slot D9:
   - VERIFY runs ONE forward with batch containing all slots' tokens
     (seq_id-partitioned). Existing batched code path; no change.
   - DRAFT_MTP runs ONE forward with batch containing all slots'
     drafts. Same batched path; `build_qwen35_mtp` accepts a 2D
     `inp_mtp_states` (n_embd × n_tokens) per `build_qwen35.cpp:24-26`.
   - n_tokens = n_slots; one MTP step per slot, batched.

## 8. Required headers / params adjustments

None for D6 or D8 wrapper-mode. The current
`llama_decoder_params.mtp_*` fields cover what the dispatch needs.
At D10 extraction, the per-decoder `mtp_op_type` will live as a
`llama_decoder` member (no longer aliased on cparams).

## 9. Risks / open questions

1. **INLINE_KV hook re-measure** (§4) — D8 binding-test question.
   PHASE45.md lock is provisional pending data.
2. **`set_mtp_op_type` race** — if two decoders share an internal
   `llama_context` and run concurrently (different host threads),
   the cparams.mtp_op_type flip is racy. Today's code is single-
   threaded per ctx; PHASE45 D6/D7 keep that invariant. D9
   (multi-slot) still serializes verify and draft on the same
   ggml_backend_sched, so no concurrent flips. ✓
3. **`prepare_mtp_graph_inputs` D2D path** — depends on
   `t_h_pre_norm`'s output buffer being populated *before*
   DRAFT's decode reads it. The spec_loop step orders this
   correctly (verify completes before draft). ✓
4. **`mtp_op_type` in llama_decoder_params** — currently the dispatch
   is via internal cparams. The public `llama_decoder_params` has
   `mtp_fused_n_steps` and `mtp_fused_n_extend` but not
   `mtp_op_type` directly (it's role-derived). That's correct: role
   is the public knob, op_type is the internal mapping.

## 10. Net assessment

- PHASE39 ports the inline MTP graph builder. PHASE45 adds the
  decoder type and orchestration around it. **No port-to-PHASE45
  changes needed for the graph builder itself.**
- D8 implementation work is: write `llama_decoder_decode` body that
  flips `mtp_op_type` based on role, then forwards to
  `llama_decode_internal`. Plus spec_loop body porting from
  `common/speculative.cpp` (D8 main work, separate scope).
- Re-measure inline_kv_hook on/off at D8 to validate PHASE45's lock
  on "single-canonical-writer". Bench: existing
  `bench-multiturn-pre-port.sh` config with hook flipped.
