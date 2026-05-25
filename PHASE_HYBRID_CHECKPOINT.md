# PHASE Hybrid Checkpoint — stabilise → diagnose → Phase-45-aligned decomposition

**Status**: opened 2026-05-25. Phase 1 in-flight. Phase 2 (diagnosis) and Phase 3 (decision-gated fix) pending.

**Branch**: `production/2026-q2-next` (top-level), `production/2026-q2-next` in the `ik_llama.cpp` submodule.

## 1. Context

Production wedged on 2026-05-25 07:52 UTC on `ik_llama.cpp` `llama-server` running Qwen 3.6 27B (architecture `qwen35`, a Mamba2 + attention hybrid). Two distinct bugs stacked:

- **Layer 1 — GGML_SCHED_MAX_SPLITS legacy assert.** Fixed earlier today; patch landed at `ik_llama.cpp` commit `252217d8`. Documented in [PHASE_GGML_SCHED_DYNSPLITS.md](PHASE_GGML_SCHED_DYNSPLITS.md).
- **Layer 2 — post-restore SEGV on hybrid checkpoint restore.** Today's binding test showed `apply_checkpoint()` "succeeded" (149.6 MiB blob restored via `llama_state_seq_set_data(..., LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)`), then SIGSEGV (signal 11) ~13 s later in the post-restore delta-prefill + decode. Service auto-restarts via `Restart=on-failure`, but every chat follow-up turn from a real client triggers this. **Production has been stopped pending fix** (user directive 2026-05-25).

This phase addresses Layer 2.

## 2. The orphan audit (2026-05-25)

The user's first instinct was wholesale adoption of upstream `llama_memory_hybrid` + `llama_memory_recurrent` class hierarchy. Three Explore agents audited every ik-local work-stream touching `llama_kv_cache`, `s_l`, the `seq_*` operations, `find_slot`, `server_slot`, and the server's checkpoint paths.

### High-risk code regions (would break under wholesale backport)

| Region | File:Line | LoC | Why it breaks |
|---|---|---|---|
| `find_slot` multi-seq allocator | `src/llama.cpp:1504-1700` | ~250 | Reads `v_heads[]`, `cells[]` directly; per-stream allocator; seq_id → stream mapping. Hierarchy uses block_table indirection. |
| `llama_kv_cache_seq_*` (7 fns) | `src/llama.cpp:2447-2750` | ~200 | Assume flat `cell[]` per-seq iteration. Hierarchy abstraction removes this surface. |
| `process_batch_tokens` T3.5 dispatch | `examples/server/server-context.cpp:4680-4900` | ~220 | Shape-uniform multi-seq grouping for unified-stream dispatch. find_slot signature changes propagate. |
| K/V view offset formula | `src/llama-build-context.cpp` (40+ sites) | ~300+ | `s * nb[3]` per-stream offsets baked at graph build. Hierarchy changes nb[3]'s meaning. |
| `update_cache_copies` | `src/llama.cpp:630-720` | ~90 | Stream-aware K/V view offsets. Same family as above. |
| Per-stream split tracking | `src/llama-context.h`, `src/llama-build-context.cpp` | ~150 | `split_k_l[]`, `split_v_l[]`, `split_s_l[]` per-(device, stream) multi-device tensors. **No analog in upstream's hierarchy.** |
| DFlash spec_ckpt machinery | `src/llama.cpp:1955, 1992, 2415-2416, 2137-2175`, `examples/server/server-context.cpp:4091-4168` | ~250 | `spec_ckpt.s_l_shadow` per-layer tensor snapshot for speculative rollback; per-step SSM buffer (recent `a69f19de` fix). DFlash's `restore_speculative_checkpoint` and MTP's per-step restore both depend on the current `s_l` layout. |
| Paged KV allocator (T5.1+) | `src/llama-paged-kv-allocator.h` | ~? | Custom allocator. Not 1:1 with upstream's paging. |

**Estimated total at risk:** ~1500-2000 LoC across at least 8 distinct subsystems.

### Gate-binding tests that would need re-certification

- **G3.a** — single-GPU NP-determinism byte-identity at NP∈{1,2,4,8} (`scripts/test-production-np-determinism.sh`).
- **G3.c** — Bug C absence (`scripts/r5-probe-c4.sh`). 0/20 divergences on Qwen 3.6 27B.
- **`tests/spec/test-n-stream-kv-layout.cpp`** — `KVTensorIsFourD` binding test.
- **Phase 45 D10.a** — 3-slot smoke.
- **Phase 45 D10.e** — multi-slot output determinism (OPEN).

### Phase 45 has already started a similar decomposition

- **D9.6d (closed):** *extracted `s_l` from `llama_context` onto `llama_decoder`* (+29.76% perf).
- **D9.6g (closed):** *renamed `kv_self` → `transformer_kv`* (+30.37% perf). The rename itself is preparation for a hybrid-memory split.
- **D9.8 (open):** migrate remaining `llama_context` fields to `llama_session`.

The codebase is *already moving toward* a `llama_memory_hybrid`-shaped decomposition under its own roadmap. Upstream's hierarchy is a destination overlap — but the *path* differs because the ik fork has additional concerns (per-stream KV, paged allocator, DFlash spec_ckpt, MTP buffers) that upstream never integrated.

## 3. Approach

Three-phase plan; the choice of Phase 3 sub-path is **decided by Phase 2's diagnosis**, not pre-committed.

### Phase 1 — Conservative safety patch (immediate)

`ik_llama.cpp/examples/server/server-context.cpp:3514-3517`: the validity lambda's `has_recurrent` branch returns `false` unconditionally, forcing the safe `do_reset = true` path at line 3548 (already implemented in the fork). Comment links to this PHASE doc.

Effect: every chat follow-up on hybrid models full-reprefills (slow but **safe** — no SEGV). Phase 45 D9.6d's extracted-to-decoder `s_l` is untouched. No conflicts with N-stream, T-series, MTP, DFlash, Vulkan.

Closure: re-run the 2-turn binding test from 2026-05-25 (~3 k-token prompt + small delta). Turn 2 journal must show `no usable hybrid/recurrent checkpoint; forcing full prompt re-processing` and **complete without SEGV**.

### Phase 2 — Diagnose the SEGV (1-3 days)

**Hypotheses to confirm or refute**:

- **H1.** The 149.6 MiB blob includes multi-sequence `s_l` data despite `--parallel 1` — expected ~52 MB single-seq, observed ~3×.
- **H2.** `llama_state_seq_set_data` writes back into a `s_l[il]` shape `[n_embd_v_s(), n_slots]` with `n_slots` mismatching save-time value.
- **H3.** Per-step SSM buffer (the recent `a69f19de` MTP+prefill fix) and the checkpoint blob disagree about the per-step state.
- **H4.** `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` semantics include something we don't expect (a cell-table header that desynchronises the s_l reader).

**Instrumentation work** on a dev branch (separate `llama-server-dev` binary so production stays on Phase 1):

- `SLT_WRN` traces in `llama_state_seq_get_size`, `_get_data`, `_set_data` (`src/llama.cpp:10420-10451`) — dump flags, seq_id, n_layer, s_size_row, n_slots, per-section byte counts.
- CRC32 of `s_l` payload at save; CRC at restore; flag mismatch.
- Capture core dump (set `ulimit -c unlimited`, `coredumpctl gdb` for the SEGV stack).
- Byte-compare one save payload against the immediately-following restore.

Deliverable: this PHASE doc §"Diagnosis" with the root cause chain. Locked before Phase 3 starts.

### Phase 3 — Decision-gated by Phase 2

Phase 2's diagnosis selects one of three sub-paths:

- **3a. Targeted in-place fix.** If Phase 2 finds H1, H2, or H4 — a localised indexing / sizing mismatch — the fix is a 5-50 line patch around `src/llama.cpp:9461-9505` (write) and `:9946-10014` (read). No architectural change. **Highest probability outcome.**
- **3b. Phase-45-aligned decomposition.** If Phase 2 finds the problem is structurally deeper, extend Phase 45 D9.6 with a new sub-step `D9.6h: recurrent state save/restore method on llama_decoder`. Internal to the ik fork's own decomposition roadmap. 3-7 days. Preserves all 1500-2000 LoC of ik-local work.
- **3c. Upstream `llama_memory_hybrid` backport.** **Last resort.** Only chosen if Phase 2 shows 3a/3b can't address it. 2-4 weeks of integration + 5 gate-binding tests re-certified.

## 4. Tasks

### T1 — Author this PHASE doc + SUMMARY entry

- [x] Write `PHASE_HYBRID_CHECKPOINT.md` at repo root.
- [x] Symlink into `docs/PHASE_HYBRID_CHECKPOINT.md` so mdbook picks it up.
- [x] Add entry to `docs/SUMMARY.md` under "Scheduler stability".
- [x] Commit + push (per CLAUDE.md §5).

### T2 — Phase 1 patch (apply_checkpoint validity lambda)

- [ ] Edit `ik_llama.cpp/examples/server/server-context.cpp:3514-3517`: the recurrent branch of the validity lambda returns `false` with a multi-line comment explaining the SEGV evidence and pointing to this doc.
- [ ] Commit in submodule on `production/2026-q2-next`. Do not push yet (user owns the upstream destination — push is a separate decision).

### T3 — Rebuild + redeploy (Phase 1)

- [ ] Incremental rebuild: `cmake --build /home/dconnolly/yarn-agentic/ik_llama.cpp/build -j --target llama-server`. Should be <60 s (one TU changed).
- [ ] Deploy via `scripts/deploy-llama-server.sh` (productionized path — installs binary + libggml/libllama/libmtmd `.so` files atomically, sha-verifies, restarts).
- [ ] Re-enable + start the service: `sudo systemctl enable --now llama-server.service`.

### T4 — Phase 1 binding verification

- [ ] Run the 2-turn smoking-gun test (~3 k-token prompt + ` One more word please.` delta).
- [ ] Confirm: no SEGV; turn 2 journal shows `no usable hybrid/recurrent checkpoint; forcing full prompt re-processing`; both turns return 200.
- [ ] Multi-turn soak: 5+ turns over 10 minutes; service stays `active (running)`.

### T5 — Phase 2 instrumentation (deferred)

- [ ] Set up `llama-server-dev` binary with `SLT_WRN` traces in the state-seq-data path.
- [ ] Add CRC32 of `s_l` payload to save + restore; flag mismatches.
- [ ] Configure core-dump capture (`ulimit -c unlimited` in the dev unit's environment).
- [ ] Reproduce the SEGV with the binding test (against the dev binary, while production runs the Phase 1 binary).
- [ ] `coredumpctl gdb` against the dev binary; capture the SEGV stack.
- [ ] Byte-compare one save payload against the immediately-following restore.
- [ ] Write §5 "Diagnosis" with the root-cause chain.

### T6 — Phase 3 (deferred; sub-path chosen by §5 Diagnosis)

- [ ] (3a or 3b) Apply the targeted or Phase-45-aligned fix per Phase 2's diagnosis.
- [ ] Revert the Phase 1 `return false` once the new code path is verified by §6 Verification.
- [ ] Re-run §6 binding suite end-to-end.

## 5. Diagnosis

*[populated by Phase 2]*

## 6. Verification — end-to-end binding test script

`scripts/verify-hybrid-checkpoint.sh` exercises all phases. Exit criterion of this PHASE doc is that the script passes after Phase 3 (whichever sub-path lands).

1. **GGML_SCHED stability** (already verified). 3 k-token prompt → `graph splits = 387` → no GGML_ASSERT.
2. **Phase 1 acceptance.** Two-turn same-prompt + delta. Turn 2 journal shows `no usable hybrid/recurrent checkpoint; forcing full prompt re-processing`; **both turns complete without SEGV**; turn 2 wall time ≈ turn 1 wall time.
3. **Phase 2 diagnosis evidence.** CRC mismatch or sizing log line that pinpoints the root cause. Recorded in §5.
4. **Phase 3 with-restore correctness** (if 3a or 3b). Two-turn same-prompt + delta. Turn 2 journal shows `restored context checkpoint took N ms`; **both turns complete without SEGV**; turn 2 wall time **< 50 %** of turn 1; turn 2 first-20 tokens semantically identical to a single-shot reference (at `--temp 0`).
5. **Regression — long-context fragmentation.** 35 k-token prompt; `fragmentation: 0.9+`; no GGML_ASSERT, no SEGV. Validates that the Layer 1 fix still holds.
6. **Regression — gate binding.** Re-run G3.a, G3.c, `test-n-stream-kv-layout`, Phase 45 D10.a 3-slot smoke. All must remain PASS.
7. **Multi-turn soak.** 30 turns × 60 minutes against a 5 k-token shared prefix. Service stays `active (running)`; no `Restart=on-failure` events; per-turn wall-clock variance < 20 % after turn 2.

Stretch criterion (Phase 3a or 3b only): second-turn latency drops **≥ 5×** vs Phase 1's full-reprefill baseline on a 30 k-token shared prefix.

## 7. Out of scope

- Adopting upstream's SWA `llama_kv_cache_unified_iswa` (PR #13194). Not needed for Qwen 3.6 (no SWA).
- The `causes[]` debug array cleanup (`ggml-backend.cpp:1277`). Closed as PHASE_GGML_SCHED_DYNSPLITS.md §T5.
- The `CMAKE_INSTALL_RPATH` work. Tracked in `scripts/deploy-llama-server.sh` updates + CLAUDE.md §9 — orthogonal to this PHASE.
- Wholesale `llama_memory_hybrid` backport (Phase 3c). Only if Phase 2 proves 3a/3b cannot resolve.

## 8. Cross-references

- `PHASE_GGML_SCHED_DYNSPLITS.md` — Layer 1 fix (closed).
- `MEMORY.md` 2026-05-25 entry — incident writeup.
- `docs/phases/60-llama-context-decompose/PHASE45.md` — D9.6d (`s_l` to decoder), D9.6g (`kv_self` → `transformer_kv`), D9.8 (open: remaining field migration).
- `PHASE_NSTREAM_KV.md`, `PHASE_NSTREAM_KV_PERF.md` — N-stream KV foundation and T-series tiers; affected by any KV-cache restructuring.
- ik_llama.cpp Issue **#1762** — qwen3next checkpoint restoration tracking.
- llama.cpp Issue **#22384** — upstream tracker for the same family of bug.
