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

### 5.1 Static analysis (2026-05-25, no instrumentation yet)

First pass: read the save (`src/llama.cpp:9461-9505`) and read (`:9946-10014`) blocks side-by-side, and trace the writer / reader subclasses.

**The save and read paths are symmetric for the recurrent case.**

- Both branch on `seq_id == -1` (full save / restore) vs `seq_id >= 0` (per-slot).
- Both use `llama_kv_qnext_seq_id_in_range(transformer_kv, seq_id)` as the gate (defined at `:881`: `n_slots > 0 && seq_id >= 0 && seq_id < n_slots`).
- Save (`:9494`) writes `s_rows * s_size_row` bytes from offset `src_seq_id * s_size_row`. Read (`:10006`) writes the same number of bytes to offset `s_dst_row * s_size_row`.
- The recurrent-aware writer at `:10094` and reader at `:9721` both check `model.hparams.recurrent_layer_arr[il]` and route to recurrent-only multi-device split helpers when needed (`get_tensor_data_split` 4-arg variant on the write side; `read_kv_cache_data_split` with `is_recurrent=true` on the read side).

**The one asymmetry** is that save reads from row `src_seq_id = cells[seq_id].src` (`:9489`), while read writes to row `seq_id` (`:9995`). For our production case this is harmless: `--parallel 1`, `seq_id=0`, and `cells[i].src` defaults to `0` (struct default at `src/llama-context.h:20`) or is explicitly set to `i` at several init / reset call sites (`:980, :2425, :2498, :2611, :7248`). So `src_seq_id == seq_id == 0` in steady state — the save reads row 0 and the read writes row 0. The asymmetry would only manifest if some prior `seq_cp` operation set `cells[0].src` to a different value, which doesn't happen under the current production launch flags.

### 5.2 H1 (multi-seq blob bloat) is refuted

Initial worry was that 149.6 MiB ≈ 3× the expected ~52 MB single-seq SSM-state math, suggesting a multi-seq leak despite `--parallel 1`. The 52 MB figure was wrong — it assumed a narrower element width. Recomputing with the actual element type:

- `ssm.state_size = 128`, `ssm.inner_size = 6144`, `ssm.time_step_rank = 48`, `ssm.group_count = 16`, `ssm.conv_kernel = 4`.
- `conv_state_dim = (4-1) × (2 × 128 × 16 + 6144) = 30 720` elements.
- `ssm_state_dim = (6144 / 48)² × 48 = 128² × 48 = 786 432` elements.
- `n_embd_v_s = 30 720 + 786 432 = 817 152` elements.
- `s_l[il]->type = F32` (confirmed by element-size match): `817 152 × 4 = 3 268 608` bytes ≈ 3.12 MB per row.
- `full_attention_interval = 4` → ~16 attention layers (with `s_l[il] == nullptr`, write headers only) + ~49 SSM layers (with 1 row of payload).
- Total = `16 × 16 + 49 × 3.12 MB ≈ 153 MB` ≈ **observed 149.6 MiB.**

So the blob is the **expected single-seq size**, not multi-seq bloat. H1 refuted. The 149.6 MiB is normal.

### 5.3 Implications — the bug is downstream of the restore, not in it

The restore log line (`restored context checkpoint took 46.31 ms`) is followed by `erased invalidated context checkpoint` lines and `kv cache rm [p0=2560, end)`, then `fragmentation: 0.52`, then a new `create_check 6 of 64`, then another `kv cache rm [p0=3072, end)` — and 13 s later, SIGSEGV. The restore *itself* runs to completion cleanly. The SEGV is in the **post-restore delta-prefill or first decode**, when the decoder consumes the (now-restored) `s_l[il]` state.

Updated hypotheses, ranked:

- **H3 (most likely now).** Per-step SSM buffer (`src/llama.cpp:2137-2175`, allocated post-`a69f19de` to fix the MTP+prefill abort) and the restored `s_l[il]` disagree about state-stride or per-step-allocated-count. The decoder dereferences the per-step view at a position that the restored state doesn't have valid data for.
- **H5 (new).** `s_l[il]->extra` (the multi-device split tensor descriptor) is set when allocated via the per-device split path (`:1156`). Restore via `read_kv_cache_data_split` (`:9721`) iterates per-device. If the split layout at save time differs from restore time — e.g. one device was offline / the split count changed across a service restart — the restore would write to the right per-device tensors but the resulting per-device data layout would be inconsistent. Subsequent decode reads from the wrong device-slice.
- **H4 (still possible).** `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` semantics include the cell-table metadata in the blob (the early section of `write_kv_cache_data`), and the restore desynchronises if the cell array has been resized between save and restore.
- **H2 (refuted for our case).** `n_slots` mismatch would be caught by the `s_size_row != s_size_row_ref` check at `:9978-9982` or the `s_rows_ref != s_rows` check at `:9999-10001`, both of which return false on mismatch and would log an error before any data corruption.

### 5.4 Next step — targeted instrumentation

Rather than wholesale `SLT_WRN` traces across all of save/read, the static analysis points at three high-value probe points:

1. **Per-layer save/restore byte-equality check.** Add CRC32 over each layer's `s_rows * s_size_row` payload at write time; compute CRC at read time; log mismatch with `il`, `s_rows`, `s_size_row`. Confirms whether the bytes round-trip correctly per layer.
2. **Pre-decode `s_l[il]` content check post-restore.** Hash the per-device split contents of `s_l[il]` immediately after `apply_checkpoint` completes, again before the first decode batch is committed. Confirms whether the restored state survives the intervening `kv cache rm` + `create_check` operations or whether something rewrites it in-between.
3. **Per-step SSM buffer sanity at the SEGV site.** `src/llama.cpp:2137-2175` allocates per-step SSM views. Log their shape / device / size on the first decode call after a restore. Cross-reference against `s_l[il]` shape.

(1) and (2) are tractable code changes in `llama_data_write_buffer::write_tensor_data` and `read_kv_cache_data` respectively (+ a small hash helper). (3) is a print-statement at the per-step buffer allocation site.

### 5.5 Instrumentation plan

- New branch in the `ik_llama.cpp` submodule: `dev/2026-q2-hybrid-ckpt-trace` off `production/2026-q2-next`.
- Separate build tree at `ik_llama.cpp/build-dev/` so production builds stay clean (same `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15` toolchain pin per CLAUDE.md §9).
- Instrumented binary at `ik_llama.cpp/build-dev/bin/llama-server-dev` (the `cmake --build` target stays `llama-server` but the build-dev tree is segregated; the dev binary is **not** installed to `/opt/llm-server/`).
- Run binding test against `127.0.0.1:8081` (the dev binary listens on a different port so production on `:8080` stays untouched).
- Production stays on Phase 1 throughout Phase 2.

### 5.6 Probe #1 results + the actual SEGV stack (2026-05-25 12:46 UTC)

Ran the instrumented dev binary (commit `39290fc3`, restore re-enabled, PROBE1-W/R first/last-word fingerprint at every s_l write/read) on a 1644-token prompt + 1648-token follow-up, on port 8080 (production was stopped). Turn 2 reproduced the SEGV under instrumentation.

**Concrete byte-level findings:**

- Every PROBE1-W and PROBE1-R entry shows `size=3268608` (= 3.12 MB) and `extra=1` on every SSM layer. Confirms: per-layer payload is the F32 SSM state, every s_l tensor is multi-device-split. The blob math closes: 48 SSM layers × 3.12 MB ≈ 150 MB ≈ observed 149.6 MiB. **H1 is dead — the blob is the right size.**
- 48 PROBE1-R entries (one per SSM layer) ran to completion before the crash → the restore itself was structurally complete; `apply_checkpoint()` returned `restored context checkpoint took 59.20 ms (pos_min = 1535, pos_max = 1535)` cleanly.
- Round-trip byte-equality cannot be confirmed at this granularity yet — PROBE1 doesn't tag which checkpoint a write belongs to, so we can't pair a specific PROBE1-W with its matching PROBE1-R. Probe #1.5 (add `pos_min` to the log) needed for that confirmation.

**The actual SIGSEGV stack** (`coredumpctl`, PID 10373):

```
#0  0x00007ff1ba59c08d  ggml_new_tensor_impl.constprop.1                       (libggml.so  + 0x19c08d)
#1  0x00007ff1ba5cd528  ggml_view_3d                                           (libggml.so  + 0x1cd528)
#2  0x00007ff1cf1a4433  llm_build_context::build_defrag(...)                   (libllama.so + 0x1a4433)
#3  0x00007ff1cf1b0499  llm_build_context::llama_build_graph_defrag(...)       (libllama.so + 0x1b0499)
#4  0x00007ff1cf095ee0  llama_kv_cache_update                                  (libllama.so + 0x95ee0)
#5  0x00007ff1cf0bd266  llama_decode_internal                                  (libllama.so + 0xbd266)
#6  0x00007ff1cf0bfda6  llama_decode                                           (libllama.so + 0xbfda6)
#7  0x000055994db07f08  server_context::process_batch_tokens(int&)             (llama-server)
#8  0x000055994db0b30a  server_context::update_slots()                         (llama-server)
```

**Reshapes the diagnosis completely.** The crash is **not** in `apply_checkpoint`, **not** in the post-restore prefill, and **not** in the decoder consuming `s_l`. It's in the **KV-cache defragmentation** triggered by `llama_kv_cache_update` on the first decode call after the restore. The sequence from the journal:

```
... slot apply_checkp: restored context checkpoint took 59.20 ms (pos_min=1535, pos_max=1535)
... apply_checkp: erased invalidated context checkpoint (pos_min=1644, pos_max=1644)
... batch_pending_prompt: kv cache rm [p0, end) ... p0=1536
... fragmentation: 0.50
... [SIGSEGV in build_defrag → ggml_view_3d → ggml_new_tensor_impl]
```

So: restore writes s_l rows back; `apply_checkp` then erases the now-invalidated future checkpoint (pos_max=1644 > restored pos_max=1535) which calls `kv_cache_seq_rm` at p0=1536; that wipe-from-1536-onwards rearranges the cell table; on the next decode, `llama_kv_cache_update` recomputes fragmentation, sees 0.50 and triggers defrag; the defrag graph builder constructs a `ggml_view_3d` over what it thinks is the live KV layout; **one of the source tensors is no longer valid** (most likely a freed `s_l[il]` view, or a split-tensor `->extra` pointer that the rm/restore sequence left dangling); `ggml_new_tensor_impl` crashes dereferencing it.

### 5.7 Updated hypothesis ranking

- **H6 (new, leading).** The KV-cache rm path post-restore leaves the per-layer split-tensor descriptors (`s_l[il]->extra` and/or the analogous K/V split metadata) in a state that `build_defrag` cannot construct a graph over. The 48 PROBE1-R entries showed `extra=1` for every SSM layer; the defrag builder presumably walks `extra->splits[id]` per layer and one of those entries is stale/freed.
- **H5 (still possible).** Same family — split-tensor layout mismatch between save and restore, but the failure surface is specifically the defrag graph builder.
- **H3 (demoted).** Per-step SSM buffer mismatch is no longer the leading suspect; the SEGV stack puts us in `build_defrag` which is upstream of any per-step SSM consumption.
- **H4 (still relevant).** Cell-table metadata desync is a plausible reading of the kv_cache_rm → fragmentation 0.50 → defrag-explodes sequence.

### 5.8 Phase 3 sub-path decision tightens toward 3a

The bug is **localised** to the defrag graph construction post-restore — not to the SSM state ownership architecture, the recurrent memory abstraction, or any boundary that `llama_memory_hybrid` would clean up. This pushes the Phase 3 decision strongly toward **3a (targeted in-place patch)** rather than 3b (Phase 45 D9.6h extension) or 3c (upstream backport).

Probable fix shape:

- **Option A.** Skip the defrag pass for one decode batch after a restore. Add a flag in `server_slot` / `llama_kv_cache` that suppresses `llama_kv_cache_update`'s defrag call on the first post-restore decode. Tiny patch. Doesn't address the root cause but kills the symptom.
- **Option B.** Make `build_defrag` robust to the post-restore cell layout — either rebuild the affected tensor views before defrag, or skip layers whose split-tensor descriptors point at freed memory. Slightly larger patch; addresses the root.
- **Option C.** Find the exact line in `kv_cache_seq_rm` or `apply_checkpoint` that's leaving the descriptor invalid and fix it at the source. Smallest patch if we can pinpoint it.

Next instrumentation iteration should target the defrag path:

- **Probe #4** (replaces #2, #3): log every `ggml_view_3d` call inside `build_defrag` with the source tensor identity, ne[], nb[], and `->extra` validity. The view that fails will tell us exactly which descriptor is bad.
- **Probe #1.5** (small extension): tag PROBE1-W with `pos_min`/`pos_max` of the checkpoint being saved, so we can pair the bytes round-trip if/when we want.

Probe #4 is the path to a concrete fix. Probe #1.5 is now optional given the defrag-path discovery.

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
