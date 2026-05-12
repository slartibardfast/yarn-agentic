# Phase 46 — DFlash speculative decoding port into ik_llama.cpp

## Goal

Ship DFlash speculative decoding on `production/2026-q2-next` such that:

- Target = Qwen 3.6 27B AutoRound int4 (`Q4_0_AR16`, existing
  `/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf`)
- Drafter = Qwen3.6-27B-DFlash BF16
  (`/opt/models/recast-out/qwen3.6-27b-dflash-bf16.gguf` — landed in S0)
- Output is byte-deterministic vs vanilla greedy decode at np=1.
- Measured tok/s strictly exceeds production MTP `--draft 3` baseline
  (33.5 tok/s on this hardware, 2× Quadro RTX 6000 sm_75).

GREEN closes Phase 46. RED at any stage closes the port cleanly with
the cause documented and the production MTP path untouched.

## Non-goals (out of scope for Phase 46)

- vLLM-on-sm_75 measurements. Abandoned. ik_llama.cpp is the fastest
  engine on this hardware; vLLM is reference-implementation only and
  is not a deployment target.
- Multi-slot (np > 1). Per `project_phase45_d10e_perslot_abandoned`,
  the engine is bandwidth-bound at decode batch sizes and per-slot
  concurrent dispatch does not amortise; multi-slot is parked.
- Tree fan-out. Per `project_tree_fanout_hybrid_recurrent_blocker`,
  tree-K speculation is incompatible with the hybrid recurrent
  DeltaNet layers in this target.
- AoT-build flashinfer or any other kernel work outside ik_llama.cpp.
- BF16 hardware acceleration. sm_75 has no native BF16; drafter
  weights stay BF16 on disk and cast at dispatch.

## Architecture — what the port adds

DFlash drafter is structurally a 5-layer Qwen3-style transformer with
two non-standard surfaces:

1. `dflash.fc.weight` (5120, 25600) — fuses 5 target hidden states
   (concat over the feature axis) back down to hidden_size.
2. `dflash.hidden_norm.weight` (5120,) — RMSNorm on the post-`fc`
   fused input, before the first transformer block.

No `embed_tokens`, no `lm_head` — both delegated to the paired target.
Sliding-window attention on layers 0–3 (window=2048), full attention
on layer 4. GQA 32q/8kv, head_dim 128, RMSNorm trunk + per-head
q_norm/k_norm. RoPE freq_base 10_000_000.

Inference loop per draft block:

1. Target forward over current accepted prefix. At
   `target_layer_ids = [1, 16, 31, 46, 61]` (5 layers out of 64),
   the residual stream is snapshotted into a buffer.
2. Concatenate 5 snapshots along feature axis → 25600-dim vector
   per position. Apply `fc` → 5120-dim. Apply `hidden_norm`.
3. Insert `mask_token_id = 248070` at draft positions (`block_size = 16`),
   embed via target's embed_tokens, mix into drafter input.
4. Drafter runs N denoising steps. Each step: bidirectional attention
   WITHIN the draft block, causal w.r.t. accepted prefix. Drafter
   output → target's `lm_head` → logits → argmax → refined block.
5. After N steps, propose the block to target verify (existing path,
   reused unchanged from MTP).
6. Accept on greedy match (target greedy = byte match). Reject on
   first mismatch; replay from there next iteration.

## Stages

### S0 — Drafter conversion `[x]`

- [x] `scripts/dflash_drafter_to_gguf.py` — standalone BF16 GGUF
      converter. Idiomatic llama.cpp KV via writer convenience
      methods. `attention.sliding_window_pattern` array encodes the
      4-sliding-1-full layer pattern (True=sliding, False=full).
      Custom KV: `block_size`, `mask_token_id`, `target_layer_ids`,
      `num_target_layers`.
- [x] Output: `/opt/models/recast-out/qwen3.6-27b-dflash-bf16.gguf`
      (3.22 GiB, 58 tensors, BF16 throughout). Round-trip check
      passes (tensor names, shapes, dtypes, 6 required KV keys).
- [x] Loader contract documented: drafter GGUF carries no tokenizer,
      no embed_tokens, no lm_head. Loaders MUST refuse standalone
      load with the exact error message specified in the script's
      docstring.

### S1 — Drafter arch + standalone forward `[ ]`

Land the architecture in ik_llama.cpp such that the drafter GGUF loads
and produces hidden states matching the HF reference within ε. No spec
loop yet. No target conditioning yet — feed dummy zeros into `fc`.

Submodule commit: `aa17361c` on `production/2026-q2` (slartibardfast
fork of ik_llama.cpp).

- [x] **T1.1 — Arch registration.** Added `LLM_ARCH_DFLASH_DRAFTER`
      to `src/llama-arch.{h,cpp}` + `LLM_TENSOR_DFLASH_FC` /
      `LLM_TENSOR_DFLASH_HIDDEN_NORM` tensor enum entries. Registered
      tensor schema in `src/llama-model.cpp` (no `token_embd` / no
      `output` — delegated to paired target; has `output_norm` +
      `dflash.fc` + `dflash.hidden_norm` + standard per-block
      Qwen3 surfaces). Added `ggml_tensor * dflash_fc` and
      `dflash_hidden_norm` fields on `struct llama_model`.
      Implemented `create_dflash_drafter_tensors()` in
      `src/llama-load-tensors.cpp`, dispatched from the arch switch.
      All 58 tensors covered (3 trunk + 55 per-block × 5 layers).
- [x] **T1.2 — KV key parsing.** Added 4 `LLM_KV_DFLASH_*` enum
      entries + `LLM_KV_NAMES` map entries. New `dflash` sub-struct
      on `llama_hparams` (uint32 types to match existing template
      instantiations; fixed-size `target_layer_ids` array preserves
      `llama_hparams` trivial-copyability). Load arc for
      `LLM_ARCH_DFLASH_DRAFTER` in `llama-hparams.cpp` reads the
      sliding-window scalar + sliding-window pattern array + 4
      DFlash KV via `get_arr_n` + `get_key_or_arr` for the
      `target_layer_ids` array. Added DFlash drafter to the NEOX
      RoPE-type switch.
      Verified empirically via `gguf-dump`: all 6 expected KV keys
      present and named per the parser convention.
- [x] **T1.3 — Loader-pair contract enforcement.** Added in
      `src/llama.cpp` immediately after `llm_load_arch`: if
      `model.arch == LLM_ARCH_DFLASH_DRAFTER`, throw the exact
      error message specified in
      `scripts/dflash_drafter_to_gguf.py` docstring (instructs
      caller to pass `--target-model <path>`).
      Verified: `llama-cli -m drafter.gguf -n 4 -p hi` exits rc=1
      with the documented error message; vocab load is never
      attempted (error fires before that path).
- [x] **T1.4 — Graph builder.** Created
      `src/graphs/build_dflash_drafter.cpp`. Hidden-state input
      allocated at `(n_target_pickoffs * n_embd, n_tokens)` (T1.5
      will wire runtime buffer population). Prepends
      `ggml_mul_mat(dflash.fc, inp)` → `n_embd`-projected, then
      `dflash.hidden_norm` RMSNorm. Trunk: 5 Qwen3-style blocks
      with per-head `q_norm`/`k_norm`, `ggml_rope_ext` on
      Q+K (NEOX), `llm_build_kv`. Per-layer SWA mask dispatch via
      `hparams.swa_layers[il]` — same pattern as
      `build_gemma2.cpp:28` (only condition substituted). Graph
      terminates at `output_norm`; no lm_head (delegated to target
      per S2.T2.5). Registered in `llama-build-context.{h,cpp}`
      dispatch + `CMakeLists.txt`.
      Verified: clean compile + link of `libllama.so`.
- [ ] **T1.5 — Standalone forward harness.** Add a CLI command
      `llama-dflash-drafter-forward` that loads the drafter GGUF +
      a fixed dummy input (zeros over `[seq_len=16, hidden=5120]`),
      runs forward, dumps the output hidden states to a `.npy`
      file at `data/phase46-t1.5-drafter-forward.npy`.
      Verify: harness binary builds and produces the expected output
      shape (16, 5120) in fp32.
- [ ] **T1.6 — HF reference match.** Write
      `scripts/phase46-t1.6-hf-drafter-reference.py` that loads the
      HF DFlash drafter with the same dummy input, dumps the
      reference hidden states, and computes max-abs-diff vs the
      GGUF output from T1.5.
      Verify: max abs diff < 1e-3 across the (16, 5120) tensor.
      RED at this gate is a real bug in the graph or tensor mapping
      — do not move past S1 until GREEN.

### S2 — Target hidden-state pickoff + block-diffusion sampler `[ ]`

Get the drafter producing real draft tokens, conditioned on real
target hidden states, matching the HF DFlash reference.

- [ ] **T2.1 — Multi-layer hidden-state pickoff API.** Add to
      `include/llama.h`:
      ```c
      LLAMA_API int llama_get_hidden_states_at_layers(
          struct llama_context * ctx,
          const int32_t       * layer_ids,
          size_t                n_layer_ids,
          float               * out_buffer,
          size_t                out_buffer_capacity);
      ```
      Behavior: after a target forward, copies the residual stream at
      each requested layer index into `out_buffer` in stacked
      `[n_layer_ids, n_tokens, hidden_size]` layout. Returns total
      elements written, or negative error.
      Verify: unit test in `tests/` that runs a target forward,
      collects layers [1,16,31,46,61], and confirms total elements =
      5 × n_tokens × 5120.
- [ ] **T2.2 — Target forward instrumentation.** In
      `src/llama-build-context.cpp` (or wherever the target's
      `build_qwen35.cpp` graph is constructed), add an optional
      pickoff list parameter. At each requested layer index, insert
      a `ggml_cpy` of the residual stream into the pickoff buffer.
      Verify: an A/B run with the pickoff list empty vs the full
      [1,16,31,46,61] list produces byte-identical logits at the
      end of the target forward (the pickoff is read-only, must not
      perturb compute).
- [ ] **T2.3 — Pickoff reference match.** Write
      `scripts/phase46-t2.3-hf-target-pickoff-reference.py` that
      runs the HF target on a fixed prompt, dumps hidden states at
      layers [1,16,31,46,61], compares with ik_llama.cpp's
      pickoff output via `llama_get_hidden_states_at_layers`.
      Verify: max abs diff < 1e-2 (target int4 vs HF BF16 will
      have wider tolerance than drafter BF16-vs-BF16).
- [ ] **T2.4 — Non-causal attention mask variant.** Add a mask
      mode to `build_inp_KQ_mask*` family in
      `src/llama-build-context.cpp` that produces:
      - causal between accepted-prefix positions and draft positions
      - bidirectional WITHIN the draft block (size 16)
      - causal between draft positions and accepted prefix is
        allowed for attend-back, but attend-forward only WITHIN
        the block
      Verify: a unit test in `tests/` that constructs the mask for
      a fixed (prefix_len=64, block_size=16) shape and asserts the
      expected zeroing pattern in the resulting mask tensor.
- [ ] **T2.5 — DFlash drafter forward entry.** Add to
      `include/llama-spec.h`:
      ```c
      LLAMA_API int llama_dflash_draft(
          struct llama_context * drafter_ctx,
          struct llama_context * target_ctx,
          const float          * target_hidden_states,
          size_t                 hidden_buffer_n_tokens,
          const llama_token    * prev_accepted_tokens,
          size_t                 n_prev,
          size_t                 block_size,
          size_t                 n_denoise_steps,
          llama_token          * out_block,
          float                * out_logits_optional);
      ```
      Implementation in `src/llama-spec-dflash.cpp`. For each
      denoising step: apply `fc` + `hidden_norm` over target
      hidden states, run drafter forward, project output through
      TARGET's `lm_head` (cross-context tensor reference), argmax
      → refined block. Repeat for `n_denoise_steps`.
      Verify: harness binary
      `llama-dflash-draft` that loads target+drafter, runs the
      entry point against a fixed prompt, dumps the proposed
      block to stdout.
- [ ] **T2.6 — Mask-token embedding via target.** At draft
      positions, embed `mask_token_id=248070` using the TARGET
      model's `token_embd.weight`. Cross-context tensor reference
      requires either (a) sharing the target's embedding tensor by
      pointer into the drafter's compute graph, or (b) materialising
      an embedding lookup at the spec-loop layer and passing the
      result into `llama_dflash_draft` alongside target hidden
      states.
      Decision (default): option (b) — embed mask token in the
      spec-loop layer, mix into the drafter input alongside fc(target
      hidden states). Cleaner contract, no cross-context tensor
      sharing.
      Verify: print the embedding row for token 248070 from target,
      print the drafter's first-position input at a draft position
      that's all-mask, confirm they encode the same vector after
      the spec-loop's mixing.
- [ ] **T2.7 — HF DFlash drafter token-match.** Write
      `scripts/phase46-t2.7-hf-drafter-token-reference.py`. Load
      the HF DFlash drafter + target, run a fixed prompt through
      a single draft-block cycle (N denoising steps), dump the
      block tokens. Compare to `llama-dflash-draft` output on the
      same prompt with the same seed.
      Verify: token-for-token match for all 16 positions in the
      proposed block at fixed seed. If mismatch: surface the
      first-divergent position + reference vs got vocab strings.
      RED at this gate is a sampler or denoising-step bug.

### S3 — Spec-loop integration `[ ]`

Replace the MTP draft dispatch with DFlash drafter dispatch. Verify
the existing T0 determinism fixture passes.

- [ ] **T3.1 — Drafter-kind detection at spec-loop entry.** In
      `src/llama-spec-loop.cpp`, detect the spec-loop's
      `drafter_ctx` architecture. If `LLM_ARCH_DFLASH_DRAFTER`,
      dispatch to `llama_dflash_draft` instead of
      `llama_spec_mtp_draft`.
      Verify: a server boot with `--model <target> --speculative
      <drafter>` shows debug logs indicating DFlash dispatch
      selected; an analogous boot with the existing MTP-head
      drafter still routes to the MTP path.
- [ ] **T3.2 — Hidden-state pickoff buffer plumbing in spec-loop.**
      In `src/llama-spec-loop.cpp`, allocate a per-decode pickoff
      buffer sized `[5, max_seq_len, 5120]` (≈100 MiB at
      max_seq_len=4096). Pass to `llama_get_hidden_states_at_layers`
      after each target forward; pass into `llama_dflash_draft`.
      Verify: a single decode roundtrip on the integrated path
      completes without memory errors or shape mismatches; emits
      a draft block of 16 tokens.
- [ ] **T3.3 — Verify path reuse.** Confirm that the existing
      target-verify path (`llama_decode` over `accepted_prefix ++
      draft_block`, single forward, accept on byte match)
      compiles and runs unchanged with the new DFlash drafter
      output as its `draft_block` argument.
      Verify: target verify accept rate is non-zero (sanity);
      compares first 64 generated tokens vs vanilla decode on the
      same prompt.
- [ ] **T3.4 — T0 determinism fixture.** Reuse
      `scripts/test-mtp-multislot-determinism.sh` (or a sibling
      `test-dflash-determinism.sh`) at np=1. Run the same prompt
      under vanilla decode vs DFlash spec decode; assert byte-
      identical token streams.
      Verify: GREEN = byte-identical for 1024 generated tokens
      across 8 prompts. RED at this gate is a real determinism
      bug — do not move to S4 until GREEN.

### S4 — Production landing `[ ]`

Empirical perf vs MTP `--draft 3`; soak; doc closure.

- [ ] **T4.1 — Profile.** Add `profiles/qwen36-27b-x1-dflash.sh`:
      np=1, target Q4_0_AR16, drafter BF16, ctx 262144,
      `--speculative-config method=dflash ...` (or equivalent
      llama-server CLI). Document the exact CLI in the profile
      header.
      Verify: `bash healthcheck.sh --verbose` against the active
      service reports 200 OK + DFlash dispatch confirmed in logs.
- [ ] **T4.2 — Throughput vs MTP.** Run
      `scripts/bench-multiturn-pre-port.sh` (or equivalent)
      under the new profile. Compare to MTP `--draft 3` production
      (33.5 tok/s) on the same prompt suite.
      Verify: aggregate tok/s strictly > 33.5. Cite measured
      number in `data/phase46-t4.2-perf-floor.md`. RED if not
      strictly greater — DFlash port did not earn its keep; close
      the port and keep MTP in production.
- [ ] **T4.3 — Soak.** 200 000 generated tokens at np=1 over
      multi-turn conversations. Monitor: no host hang, no OOM,
      no token drift vs the same vanilla-decode reference,
      accept rate stays in a narrow band (±10% of reported S3 mean).
      Verify: NRestarts=0 over the soak window. Soak duration is
      multi-hour; budget accordingly.
- [ ] **T4.4 — MEMORY.md + retrospective.** Append a memory entry
      naming Phase 46 landing (or the negative-result close).
      Update `MEMORY.md` in the top-level repo and write the
      private auto-memory entries that future sessions need.
- [ ] **T4.5 — SUMMARY.md sidebar.** Ensure PHASE46.md is in
      `SUMMARY.md` so the mdBook site renders it.

## Verification — phase-level gate

Phase 46 closes `[x]` only when ALL of the following hold:

1. S1.T1.6 GREEN — drafter hidden states match HF reference < 1e-3.
2. S2.T2.7 GREEN — drafter produces HF-identical draft block tokens.
3. S3.T3.4 GREEN — T0 determinism: np=1 byte-exact vs vanilla.
4. S4.T4.2 GREEN — measured tok/s strictly > 33.5 (MTP baseline).
5. S4.T4.3 GREEN — 200k soak no-incidents.

Any single RED reopens the corresponding stage. No "follow-up
covers"; per the no-follow-up-cover rule, gaps are subtasks under
the open stage, not footnotes on a closed one.

## Critical files

- `src/llama-arch.{h,cpp}` (T1.1)
- `src/llama-hparams.cpp` (T1.2)
- `src/llama-load-tensors.cpp` (T1.3)
- `src/graphs/build_dflash_drafter.cpp` — new (T1.4)
- `src/graphs/build_qwen3.cpp` — structural reference (read-only)
- `src/graphs/build_gemma2.cpp` — sliding-window-pattern reference
  (read-only; the `(il % 2 == 0)` pattern at line 28)
- `src/llama-build-context.cpp` (T2.2, T2.4)
- `include/llama.h`, `include/llama-spec.h` (T2.1, T2.5)
- `src/llama-spec-dflash.cpp` — new (T2.5)
- `src/llama-spec-loop.cpp` (T3.1, T3.2)
- `src/llama-spec-mtp.cpp` — read-only reference for MTP path
- `tests/test-dflash-*.cpp` — new (T2.1, T2.4 unit tests)
- `scripts/phase46-t1.6-hf-drafter-reference.py` — new
- `scripts/phase46-t2.3-hf-target-pickoff-reference.py` — new
- `scripts/phase46-t2.7-hf-drafter-token-reference.py` — new
- `profiles/qwen36-27b-x1-dflash.sh` — new
- `MEMORY.md` (T4.4)
- `SUMMARY.md` (T4.5)

## Estimated cost — in tokens, not days

Per CLAUDE.md §8 ("Estimate in context tokens, not days"). Rough
budgets for active session work (decision tokens + diagnostic
tokens, excluding verification cycles):

| Stage | Decision-token budget | Verification-cycle budget |
|---|---|---|
| S1 (arch + standalone forward) | ~50k | 3× ~30k = ~90k |
| S2 (sampler + ref match)       | ~80k | 4× ~30k = ~120k |
| S3 (spec-loop integration)     | ~50k | 2× ~30k = ~60k |
| S4 (perf + soak)               | ~30k | 1× ~30k = ~30k |
| **Total**                      | **~210k** | **~300k** |

Within a single 1M-token session if work is sequenced tightly; budget
for a single-session land OR two sessions split at the S2/S3 boundary.

## Risks become required tasks

Per `feedback_no_risks_only_tasks.md`, no risks-with-fallbacks section.
Every identified risk below is already a required task above.

- Non-causal block mask — T2.4 (required, not optional)
- Cross-context tensor reference for target lm_head — T2.5 decision
  recorded inline
- Determinism regression — T3.4 gate (required)
- Perf regression vs MTP — T4.2 gate (required; RED closes the port)

## Open decisions to record before S1 starts

- Cross-context `lm_head` reference: do we share the target's
  lm_head by tensor pointer (efficient, complex contract), or apply
  it in the spec-loop layer after the drafter emits hidden states
  (cleaner contract, one extra matmul per draft step)? **Default:
  spec-loop-layer application** unless perf metric in S4 forces a
  rework.
- Number of denoising steps `n_denoise_steps`: the DFlash paper
  recommends a specific value; pin from the HF reference's
  default and emit as a GGUF KV (`dflash_drafter.n_denoise_steps`)
  if not already. Decision: read from a runtime env or CLI flag
  with a documented default; revisit if accept rate is poor.
- Drafter quantisation below BF16: not in scope for Phase 46.
  Re-evaluate only if VRAM pressure forces it.
