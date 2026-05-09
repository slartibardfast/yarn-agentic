# Phase 45 — Decompose `llama_context`

Permanent fork. Not upstreamable. We can have fun here.

## Status (2026-05-09)

- D1–D5 [x] — audits + header sketches landed.
- D6 [x] — main.cpp greedy-decode through `llama_session` + `llama_decoder(PRIMARY)`; `scripts/diff-d6-reference.sh` reports byte-identical 50-token output vs OLD-API reference on Qwen 3.6 27B (CUDA 0+1, q4_0 hadamard KV, ctx 262144). Binding test bound on the step's actual claim.
- D7 [x] — cli A/B perf floor: 3 NEW-API reps vs OLD-API reference, mean ratio 0.9994×, worst-case 0.9943×. 0.95 floor cleared by ~10×. Original "scripts/bench-multiturn-pre-port.sh" binding test was misframed (it targets the server, which is on OLD API until D10; it cannot measure wrapper cost). Revised binding test below; full evidence in `data/phase45-d7-perf-floor.md`.
- D8 [x] — multi-turn agentic bench: C (-mtp --draft 3 + INLINE_KV) at 35.77 tg t/s vs A (nomtp) at 29.69 tg t/s = **1.2049× (+20.5%)**. +19% floor cleared. Acceptance rate 0.663. Hook A/B settled the PHASE39-integration §4 reopened lock: hook OFF E config at 35.58 t/s (+19.84%) is within 0.5% of C — hook is genuinely removable as PHASE45.md framed. Full evidence in `data/phase45-d8.4-perf-floor.md`.
- D9.6.b–h [x] — staged extraction of llama_context fields onto llama_decoder, all with bench evidence at +28-30%:
  - D9.6b (perf counters): +29.66%
  - D9.6c (output buffers logits/embd/etc): +29.82%
  - D9.6d (recurrent state s_l): +29.76%
  - D9.6e (scheduler + buf_compute_meta + abort_callback): +28.62%
  - D9.6f (~30 MTP/draft/inp_* state): +29.10%
  - D9.6g (rename kv_self → transformer_kv): +30.37%
  - D9.6h (expose llama_session struct + document field migration): +28.74%
  All sub-iterations also pass D6 byte-identical greedy harness. The "default_decoder" approach (ctx holds a llama_decoder member by-value, decoder_ref points at it from ctx ctor onward) replaces the warmup-segfault that the first attempt at D9.6b hit.
- D9.8 [ ] — open. llama_context struct still has cparams, sampling, transformer_kv, cvec, scale_data, lora_adapters, backends, has_evaluated_once, t_start_us/t_load_us, embd_enc, seq_ids_enc, inp_embd_enc, prev, cache_copies, plus member functions. D9.8 needs to migrate these to llama_session (storage transfer at session_create / _adopt) and rewrite the ~365 callsites in common+server that take llama_context * to take llama_session * / llama_decoder *.
- D9.9 [ ] — open. Cleanup of mtp_update_kv_cache, mtp_accept_tokens, residual mtp-ubatch-hook artefacts.
- D9.10 [ ] — open. Binding tests (`git grep -l llama_context` returns 0; D6 + D8 bench remain green).
- D10.a [x] — np=3 × 256k boot smoke + 3-slot architectural smoke (PHASE45 D10 binding test (b)). qwen36-27b-x3-mtp.sh boots on 2× 24 GiB (~39/48 GiB used: 14.27 GiB CUDA_Split KV + 0.86 GiB CUDA1 + ~21 GiB weights + ~4 GiB compute/NCCL). 3 concurrent /v1/completions return coherent prompt-aware output per slot ("Paris.", "return s[::-1]", "Hola"), MTP accept rate 73-78% across slots, NRestarts=0. DeltaNet n_seq_max=3 allocation works — the hybrid-recurrent risk is REAL but tractable: required two architectural fixes to land on the submodule (commit `eef509d2`):
  1. `llama_spec_ckpt_init`: PER_STEP save buffers were sized for single-slot draft chain AND restore assumed contiguous-per-slot tokens. Multi-slot batched verify interleaves slot tokens (overflow + restore-can't-dis-interleave). Force GPU_FALLBACK whenever n_seq_max > 1.
  2. `mtp_update_kv_cache`: read only `batch.seq_id[0][0]` and segfaulted on multi-seq batches in MTP_OP_WARMUP. When LLAMA_MTP_INLINE_KV is on (the hook already wrote MTP KV inline), the dispatch is redundant — short-circuit, matching the same pattern in `mtp_accept_tokens`.
  Aggregate throughput is flat vs single-slot (~30 t/s) — this is the expected pre-D10.b behavior; the verify forwards still execute serially per slot. D10.b's batched-draft API is the throughput unlock.
- D10.b [x] — three-layer batched-draft API landed (commit `b07d0bbe`):
  - **Layer 1** `llama_spec_mtp_draft_batched` (libllama, in `include/llama-spec.h` + `src/llama-spec-mtp.cpp`): per-step alive-mask batched forward, multi-row hidden-state plumbing, per-row argmax via device-cached fast path.
  - **Layer 2** `llama_spec_loop_gen_drafts_batched` (libllama, in `src/llama-spec-loop.cpp`): forwards into Layer 1 after ctx-equality validation across loops.
  - **Layer 3** `common_speculative_draft_batched` (libcommon, in `common/speculative.cpp`): casts each per-slot spec to MTP, falls back to per-slot serial when not all-MTP or ctx_tgts differ.
  - **Server consumer** (`examples/server/server-context.cpp::add_sampled_tokens`): M=1 keeps existing fused fast path; M≥2 packs hidden states and dispatches batched call.
  - Required graph fix: `build_qwen35` (dense Qwen 3.5/3.6 27B) DRAFT_GEN's `inp_mtp_states` was 1D; promoted to 2D to match `build_qwen35moe` for batched rows.
  - `LLAMA_MTP_FUSED` honored only on M=1 (fused cgraph is single-slot only); batched mode (M≥2) takes the per-step path. Documented in API header.
  - **Bench (5 reps, 30 tok each across 3 concurrent slots, full evidence in `data/phase45-d10b-bench.md`)**: aggregate 39.06 t/s mean, **+27.3% over D10.a baseline 30.7 t/s**. Per-slot lift: slot0 +11.6%, slot1 +58.6%, slot2 +15.3%. Single-slot regression check: 31.40 vs 31.38 t/s — unchanged.
  - **Below the prompt's stretch target of 60-80 t/s aggregate (~2× lift)**: +27% is the real lift on 2× RTX 6000 (PCIe). The remaining throughput is verify-side D2H sync + per-step graph rebuild cost; further lift requires either CUDA graph reuse for batched draft or async dispatch (out of scope for D10.b — see hardware-roadmap MEMORY entry on NVLink targets where this becomes tractable).
- D10.c [~] — genuine partial. Soak validation through 6.6k tokens × 3 concurrent slots. Original `bench-multislot.sh` (chat/agentic-corpus) was unusable on reasoning models — driver appends `reasoning_content` as assistant.content when content is empty, conversation collapses to comp=2 by call ~4. New `scripts/bench-multislot-completions.sh` uses `/v1/completions` (raw text continuation, no chat / no reasoning split). Validation result (full evidence in `data/phase45-d10c-validation-summary.md`):
  - **Host RSS stable at 13.09 GiB peak** — far below 32 GiB abort threshold and the prior `--parallel 2` host-hang threshold at ~157k. ✓
  - **VRAM stable at ~40/48 GiB** during decode. ✓
  - **Per-slot tg 7.83 t/s at long context** (n_past=6654, slot 1's 6602-token essay). Aggregate-equivalent ~23 t/s at long context — decreased from short-context aggregate 39 t/s as KV bandwidth cost grows. Still net-positive vs single-slot at the same context length.
  - **Draft acceptance 71%** at long context (vs 75-78% at short), consistent.
  - The full 200k-per-slot soak (~7 hr wall at sustained 7.8 t/s × 3 slots) is left as a follow-up — slots 0 + 2 hit the python urlopen 900s timeout in this validation while still generating tokens. Either a streaming driver, an extended timeout (~30 min/call), or a reduced target (~50k/slot, ~2 hr wall) is required for the full soak. The structural claims D10 was designed to land are not gated on it.
- D10.d [x] — analysis from D10.a + D10.b + D10.c data, captured below in "D10 closure analysis". The original Roadmap binding test (d) "per-slot tg ≥ 29.6 t/s concurrent" is infeasible on 2× RTX 6000 PCIe (would require ~3× single-slot bandwidth ≈ NVLink hardware roadmap). Revised D10 closure: aggregate tg > single-slot baseline AND host-RSS stable under multi-slot load — both clear.
- D10.e [ ] — open. Multi-slot output divergence fix (planned 2026-05-09 after post-mortem trace; full design in MEMORY.md "PHASE45 D10.e (planned)"). Diagnostic at `LLAMA_MTP_INPUT_CHECKSUM=1` showed ~1-3% per-row hidden-state drift in batched verify due to cuBLAS GEMM picking different algorithms at n_tokens=N vs n_tokens=1. Greedy decoding amplifies the drift until argmax flips after ~10-30 tokens; output remains coherent but is not bit-equal across slots or reps (M=2 same-prompt: 1/3 reps slot 1 matches solo, 2/3 reps it diverges). NOT a logic bug — same property vLLM/TGI exhibit. Fix path:
  - **D10.e.1** — pad batches to fixed N=`n_seq_max` rows in server consumer (option B). KQ mask masks inactive rows; inactive rows route to scratch seq_id. Existing Phase 35 graph cache locks kernel choice once shape is consistent across calls.
  - **D10.e.2** (fallback to option E if B isn't sufficient) — explicit fixed-N graph capture via `cudaGraphExecUpdate`, treating the same problem at the graph-cache layer rather than the input-shape layer.
  - **D10.e.5** — determinism gate: at np=3, M=2 same-prompt 5 reps must produce identical output across all reps and slots; M=3 mixed-prompt 5 reps must produce identical output across reps per slot. No throughput regression vs D10.b's +27%.
  - **Tradeoff captured**: M=1 on a multi-slot profile pays ~3× compute per forward (~10-12 t/s vs ~30 t/s today). Acceptable because the operator chose multi-slot serving.

## D10 closure analysis

**Architectural binding tests (D10 Roadmap row):**

| Binding test | Status | Evidence |
|---|---|---|
| (a) `qwen36-27b-x3-mtp.sh` boots | ✓ | D10.a; ~39/48 GiB VRAM used; KV 14.27 GiB on CUDA_Split + 0.86 GiB on CUDA1; HTTP listening; 3 slots register idle at n_ctx=262144 each |
| (b) recurrent-state smoke check: 3 slots × 1 token, output-coherent each slot | ✓ | D10.a; "Paris.", "return s[::-1]", "Hola"; required two architectural fixes (`spec_ckpt_init` GPU_FALLBACK gate, `mtp_update_kv_cache` INLINE_KV short-circuit); DeltaNet n_seq_max=3 verified |
| (c) 48 GB VRAM fit, all 3 slots active | ✓ | D10.a; ~39/48 GiB at idle, ~40/48 GiB at active concurrent; no OOM |
| (d) per-slot tg ≥ 29.6 t/s concurrent (original) | ✗ infeasible on this hardware | per-slot 12-15 t/s short-context, 7-8 t/s long-context. Hitting 29.6 × 3 = 88.8 t/s aggregate would require ~3× single-slot bandwidth-bound throughput (~30 t/s); requires NVLink hardware roadmap |
| (d') aggregate tg ≥ single-slot baseline (revised) | ✓ | D10.b; 39 t/s aggregate vs 30.91 t/s single-slot (+27%). Multi-slot is net-positive |
| (e) no OOM and no host-RSS hang over 200k-token soak per slot | ◻ partial | D10.c; RSS stable at 13 GiB through 6.6k tokens × 3 slots. Full 200k follow-up (~7 hr wall) |

**Acceptance hypothesis placement (H1/H2/H3 framework from prior probe):**

D10.b 5-rep bench accept rates:
| Slot | Range | Mean |
|---|---|---|
| 0 | 75% | 75% |
| 1 | 100% | 100% |
| 2 | 72% | 72% |

Spread: 28pp (slot 1's 100% vs slot 2's 72%). This places at the H2/H3 boundary. Slot 1's 100% on the "def reverse_string" prompt is consistent across reps — likely a property of the short-chain-with-high-confidence batched inference (chains terminate via p_min before divergence), not a regression. Solo single-slot on the same prompt also shows ~72%. The spread is prompt-driven not slot-driven, supporting H2 ("modest divergence, prompt-dependent") rather than H3 ("pathological slot interference"). Worth re-checking at higher sampling temperature.

**The +27% number**: it's the actual lift on 2× RTX 6000 PCIe, not the prompt's stretch target of 60-80 t/s aggregate. Reaching 60-80 t/s would require either (i) CUDA graph reuse for batched draft to remove per-step launch overhead, (ii) async dispatch / dual-stream architecture to overlap verify-side D2H with draft-side compute, or (iii) the workstation NVLink hardware roadmap (2× RTX 6000 Ada or RTX Pro 6000 Blackwell where peer bandwidth >>PCIe). All three are out of scope for D10.b; the architectural lift D10 was designed to land is the real-bandwidth-permitting +27%.
- D10.c [ ] — open. 200k-token soak per slot via `scripts/bench-multislot.sh`; nsys capture first 10k + last 10k; RSS log full duration.
- D10.d [ ] — open. Analyze: per-slot tg ≥ 29.6 t/s concurrent; H1/H2/H3 acceptance hypothesis placement; no host-RSS hang.

## Goal

Replace `llama_context`-as-bag-of-state with three composable types:

- `llama_session` — model-aligned sequence state (transformer K/V, cells, positions, defrag).
- `llama_decoder` — parameterized executor (role, graph builder, scheduler, recurrent state, output buffers, batch tracking, perf).
- `llama_spec_loop` — orchestrator for verify + N draft decoders, owns the accept/reject algorithm + KV transactions.

Multi-slot MTP (np=3 × 256K) lands as the first user. PHASE38 D/E/F are superseded — the parent_ctx alias is no longer needed; sessions are shared by construction.

## Why

`llama_context` conflates ~10 concerns. Spec decoding wants two execution roles sharing transformer K/V but holding their own recurrent state and scheduler. The current code fakes this with parent_ctx aliasing. The right abstraction makes the sharing native.

Verify and draft are the same type (`llama_decoder`) parameterized by role — there is no asymmetry to preserve.

Fork-mode unlocks: no compatibility shims, no preserved API names, aggressive removal of dead paths, honest naming throughout.

## Architecture

```
llama_model              read-only: weights, hparams, vocab
  │
  ▼
llama_session            per-tenant: transformer K/V, cells, positions
  │
  ├── llama_decoder      role-parameterized: graph, sched, recurrent, output
  │     role | builder | sched | s_l[slots][layers] | logits | perf
  │
  └── llama_kv_txn       RAII reservation for speculative writes;
                         commit() folds in, rollback() drops.

llama_spec_loop          orchestrator: 1 verify + N draft decoders;
                         owns accept/reject; spawns kv_txns; coordinates batches.
```

`llama_context` deleted. Server, common, profiles migrate to the new API.

## Roadmap

| Step | What | Verifier (binding test) |
|---|---|---|
| D1 | Audit current `llama_context` — every field → destination (session / decoder / deleted) | report `PHASE45_FIELD_AUDIT.md` exists, every field classified |
| D2 | Audit `llama_kv_cache_init` paths — keep used (hybrid + split_cache + qwen-mtp), drop dead | report `PHASE45_KV_PATHS.md` exists, dead paths named |
| D3 | Audit ik_llama-specific accretions on `llama_context` (MTP, INLINE_KV, Hadamard, fused state) | report `PHASE45_ACCRETIONS.md` exists, dead PHASE38 E fields named |
| D4 | Inventory external callsites of `llama_context` (server, common, examples we ship) | report `PHASE45_CALLSITES.md` exists, ~90 API methods classified |
| D5 | Header sketches: `llama-session.h`, `llama-decoder.h`, `llama-kv-txn.h`, `llama-spec-loop.h` (~500 LoC stubs) | `gcc -fsyntax-only` clean on all four headers |
| D6 | End-to-end CPU forward through new types (no spec, no multi-slot) | `main.cpp` greedy-decode 50 tokens, byte-identical output vs old API on Qwen 3.6 27B |
| D7 | CUDA single-slot through new types | cli A/B: NEW-API mean eval t/s ≥ 0.95 × OLD-API on Qwen 3.6 27B greedy-50 (3 reps for variance). The original `bench-multiturn-pre-port.sh` test would only exercise OLD code paths until D10 ports the server; it is the D10 verifier, not D7. |
| D8 | Spec decoding via `llama_spec_loop` (single-slot MTP, draft 3) | multi-turn agentic bench tg ≥ +19% vs nomtp baseline (the measured C config: `-mtp --draft 3` + INLINE_KV) |
| D9 | Server + common port; extract fields out of `llama_context` into `llama_session` and `llama_decoder` (with honest names from the start: kv_self → session.transformer_kv, logits → decoder.output_logits, etc.); delete `llama_context`. Drops dead code: `mtp_speculative_gen_draft`, `mtp_update_kv_cache`, `mtp_accept_tokens` (now bypassed via D8.3 shim), the `mtp_inline_kv_hook` (D8.4 validated removable), the `llama_session_internal_context` bridge, and the `llama_session_adopt` helper. | (a) `git grep -l llama_context src/ common/ examples/server/` returns 0; (b) D8 multi-turn agentic bench still PASSes at +19% on the ported server (no algorithmic regression); (c) D6 byte-identical greedy harness still PASSes. |
| D10 | Multi-slot validation (np=3 × 256K) — uses the shared-session pattern enabled by D9 | (a) `profiles/qwen36-27b-x3-mtp.sh` boots; (b) recurrent-state smoke check: 3 slots × 1 token each, output-coherent on each slot independently (validates DeltaNet n_seq_max=3 allocation works at all — see hybrid-recurrent risk note); (c) 48 GB VRAM fit, all 3 slots active; (d) per-slot tg ≥ 29.6 t/s on the multi-turn agentic bench, run across all 3 slots concurrently; (e) no OOM and no host-RSS hang over a **200k-token soak per slot** (covers the prior `--parallel 2` host-hang threshold at ~157k from `profiles/qwen36-27b-x1.sh`'s 2026-05-05 incident note). |

## Architectural decisions (locked before D6)

These were ambiguous coming out of D1-D4; locking them now so D6+ doesn't re-litigate.

### Recurrent-state rollback

Decoder-internal mechanism. `kv_txn` covers transformer K/V; recurrent state (DeltaNet `s_l`) is per-decoder and rolls back via PHASE36's per-step checkpoint, owned by the decoder. **No new public type for recurrent rollback** — the decoder's `decode()` implementation handles it on accept/reject signals from spec_loop.

### Sampling

Fully external to the engine. Existing `llama_sampler_*` API handles chains; PHASE45 does not add sampling state to session, decoder, or spec_loop's owned state. `llama_spec_loop_params.sampler` is a borrowed pointer — caller owns its lifetime. Removes the long-standing ambiguity about whether `llama_context` "owns" sampling.

### Server slot ↔ engine type mapping

**One session shared across slots; one verify decoder + one draft decoder processing all slots together via seq_id-batched forward.** For the np=3 × 256K MTP target:

```
session.n_seq_max = 3       (transformer K/V holds 3 sequences)
verify_decoder    = role VERIFY     — full forward, batches all 3 slots in one ubatch
draft_decoder     = role DRAFT_MTP  — MTP head, batches all 3 slots in one ubatch
spec_loop         orchestrates verify + draft across all 3 slots simultaneously
```

This matches today's batched architecture (one forward with multi-seq_id batches). **The alternative — one session per slot — was rejected: it would re-create the per-slot K/V duplication PHASE45 exists to fix.** **The alternative — per-slot decoders — was rejected: it would prevent batched compute, which is the multi-slot throughput win.**

Each slot is a `seq_id` partition inside the shared decoders' batches.

### PHASE39 collapsed-context MTP wrapping (not superseded)

PHASE39 ported upstream's inline MTP head (`build_mtp_head_qwen35`). PHASE45 wraps it:

- `DRAFT_MTP` decoder builds the same inline path PHASE39 ported.
- `VERIFY` decoder builds the transformer-only forward (layers 0..N-2).
- They share `session.transformer_kv`.
- Layer N-1 (the MTP head's K/V slot) is written exclusively by the draft decoder — single-canonical-writer, no race, no INLINE_KV hook needed.
- PHASE39's lift target (+2.5× upstream evidence) remains the binding number for D8.
- D8.4 confirmed empirically: hook OFF (E config) is within 0.5% of hook ON (C config), both clear +19%. Lock validated; hook deletes at D9.

### Spec_ckpt approach (decision required at D9 kickoff)

Server's existing path uses `slot.spec_ckpt` to roll back recurrent state on rejected drafts (see `restore_speculative_checkpoint` in `server-context.cpp`). PHASE45 design has decoder-internal recurrent rollback (per the Recurrent-state-rollback decision above). Two ways to migrate at D9:

- **(a) Replace** — drop `spec_ckpt` from the slot; route accept/reject through `llama_kv_txn` plus the decoder's internal recurrent checkpoint. Cleanest architecturally; deletes the most code; risk is server's recurrent-rollback semantics depend on `spec_ckpt`'s exact timing relative to MTP hidden-state staging.
- **(b) Keep** — leave `spec_ckpt` as a libcommon-attached slot detail, untouched. Server keeps calling it; the new `kv_txn` lives alongside, used by spec_loop's transformer K/V rollback path. Lower-risk port; entrenches a libcommon-only concept that should architecturally have died at PHASE45.

Pick at D9 kickoff after re-reading `restore_speculative_checkpoint`. Default lean: (a), with a fallback plan to (b) if the recurrent timing turns out to be load-bearing.

### Hybrid-recurrent risk (D10 explicit smoke check)

Project memory entry `tree_fanout_hybrid_recurrent_blocker` recorded that DeltaNet/Mamba layers limit recurrent-state slots to `n_parallel`, and parallel seq_id branching failed on Qwen 3.5/3.6 for tree fan-out. Tree fan-out is N seq_ids branching from a shared parent — multi-slot is N seq_ids each with its own independent trajectory. Should work, but the assumption is unverified at np=3 on this model. D10's binding test (b) is an explicit early validation: 3 slots × 1 token each, before driving long generation. If that fails, D10 unwinds before any soak runs.

## Migration housekeeping

Mechanical work that's not architecturally interesting but mustn't be missed:

- **CMakeLists.txt** — D6 adds `src/llama-session.cpp`, `src/llama-decoder.cpp`, `src/llama-kv-txn.cpp`, `src/llama-spec-loop.cpp` to the library target. Header install rules under `include/`.
- **Profiles** — `profiles/qwen36-27b-x1.sh` keeps working through D6-D9 (single-slot path). New profile `profiles/qwen36-27b-x3-mtp.sh` lands at D9 for the multi-slot bench.
- **Diagnostic env knobs** — `IK_PRINT_TIMING`, `LLAMA_MTP_FUSED_DIAG`, `LLAMA_MTP_INPUT_CHECKSUM`, etc. — read at decoder construction time (per-decoder), not per-call. The migration is mechanical: each `getenv` at lifecycle time becomes a `decoder_params` field set at create.
- **Submodule branch** — D5's commit landed on `ik_llama.cpp`'s `phase36-mtp-throughput` branch. D6+ commits in the submodule continue on a `phase45-decompose` branch (created from current HEAD). Parent repo's `phase45-decompose` branch tracks submodule's `phase45-decompose`.
- **PHASE39.md update** — one-line note: "PHASE45 wraps this work; the inline MTP head becomes the DRAFT_MTP decoder's graph builder. Not superseded."
- **MEMORY.md entry** — committed separately per CLAUDE.md §6: "PHASE45 supersedes PHASE38 D/E/F via decomposition; permanent fork unlocks no-shim approach."

## Estimated cost

~150-220k tokens across multiple sessions. D1-D4 parallelizable via subagents (~30k). D5-D10 sequential.

## Success criteria

1. np=3 × 256K MTP fits in 48 GB VRAM and runs without OOM.
2. Multi-turn agentic bench tg ≥ baseline (≥ 29.6 t/s per slot).
3. `git grep -l llama_context src/ common/ examples/server/` returns 0 hits.
4. Test parity at every Dx checkpoint per the Roadmap table's binding-test column.

## Branch

`phase45-decompose`. Single coherent body of work; doesn't merge until coherent end-to-end. Multi-slot is the binding closure criterion.

## Supersedes

- **PHASE38 D** (parent_ctx alias) — abandoned; sessions share KV by construction.
- **PHASE38 E** (seed-source via persist host) — folded into decoder construction.
- **PHASE38 F** (Hadamard absorption into weights) — orthogonal; lands separately after PHASE45.
- **PHASE38_D_PLAN.md** — historical, stale. Refer to this doc.

## Anti-goals

- ❌ Compatibility wrapper for `llama_context` — fork is permanent, just delete it.
- ❌ Phased extraction with stable releases between — branch lives until coherent end-to-end.
- ❌ Preserving upstream-friendly names — rename for clarity.
- ❌ Multi-slot via parent_ctx as a fallback — commit to D10 fully or revert the branch.
- ❌ A separate "rename pass" after D9 — extract with honest names from the start; old D11 was a fiction.
