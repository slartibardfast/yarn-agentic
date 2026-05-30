# PHASE_PROMPT_CACHE_ISOLATION — fix cross-conversation leak in the server prompt cache

**Opened:** 2026-05-29.
**Branch:** `production/2026-q2-next`.
**Scope:** Make the `llama-server` host-memory prompt cache safe against
cross-conversation / cross-tenant leakage on the qwen35 hybrid (Mamba2), without
losing the prefix-reuse speedup that agentic workloads depend on. Two landing
changes — a **minimum-LCP guard** and **per-request `X-Prompt-Cache-Salt`
scoping** — plus an audit of the hybrid recurrent-state rewind.
**State:** A+B implemented, verified, **deployed** (submodule `eb98047f`, build 4859).
**The live leak is NOT fixed by A+B, and NO runtime flag fixes it** — see the
2026-05-30 (#2) update. Multi-sample repro proved the root cause is the qwen3next
**recurrent state slot** reused across conversations without zeroing; `--cache-ram`
and `--ctx-checkpoints` have **zero** effect (3/5 leak either way). The deployed
`--cache-ram 0 --ctx-checkpoints 0` interim is **ineffective** (and costs prefix-reuse).
Prod left as-is per user decision, pending the **only real fix: task C (qnext slot
zero-on-release)**, currently at design-review (see "Task C design" below).

---

## Update 2026-05-30 — A+B insufficient; leak vector is slot-level (C), NPC-blind

Post-deploy live smoke (this time checking **`reasoning_content`**, not just
`content`) showed the leak persists: a "capital of France" request returned
reasoning verbatim about the primed aqueduct document, with **`cached_tokens=0`** —
so the **prompt cache (A+B) did NOT reuse**; A+B work. The contamination came via the
slot's own context-switch: `apply_checkp n_past=3 ... pos_min=3041` kept only the
template prefix but the **Mamba recurrent state stayed at the end of the prior
prompt** (and a stale `--ctx-checkpoints` checkpoint restored it). This is **task C**,
the primary vector for consecutive unrelated requests on the single `--parallel 1`
slot — which I under-weighted as optional.

**Why NPC never caught it:** NPC verifies determinism (reproducibility), not
isolation. A reproducible leak passes NPC, and the harness sends independent prompts
to a fresh slot — it never exercises the cross-conversation context-switch. NPC is
structurally blind to this class of bug.

**Verification lesson:** the cache-isolation suite checked `content` (empty for this
reasoning model) and missed the leak in `reasoning_content`. Any isolation test MUST
assert on `reasoning_content` across an unrelated context-switch.

**Mitigation smoke (2026-05-30, 1-sample — SUPERSEDED, see #2 below):** `--cache-ram 0`
(ctx-ckpt 64) → LEAK; `--ctx-checkpoints 0` (cache-ram 40960) → CLEAN. Concluded
"`--ctx-checkpoints 0` is the lever" and deployed `--cache-ram 0 --ctx-checkpoints 0`
(`.bak-preleak-20260530` backup). **This conclusion was wrong — a 1-sample fluke.**

---

## Update 2026-05-30 (#2) — root cause is the qnext state slot; NO flag fixes it

**Multi-sample repro** (`data/whole-system-profile/qnextleak-20260530T164258`): 5 trials ×
2 configs, each trial a *distinct* prime topic (aqueduct / C4-photosynthesis / Byzantine
coinage / coral / guilds) then the same neutral switch ("capital of France"), re-scored
against **all** topic keywords + the "audit/document/summarize" task framing (the original
per-trial scoring undercounted cross-trial bleed):

| trial | prime | cache-ram 0 | cache-ram 40960 |
|---|---|---|---|
| 0 | aqueduct | **LEAK** | **LEAK** |
| 1 | photosynth | clean | clean |
| 2 | byzantine | **LEAK** (aqueduct bleeds) | **LEAK** |
| 3 | coral | **LEAK** (aqueduct bleeds) | **LEAK** |
| 4 | guild | clean | clean |
| | **rate** | **3/5** | **3/5 (byte-identical)** |

- `--cache-ram` has **zero** effect — identical verdicts. The 1-sample "CLEAN" was a fluke
  (stale-state influence on a short prompt is chaotic; it doesn't always *visibly* surface,
  but the stale state is present on every switch).
- `cached_tokens=0`, `prompt_n=21` on every switch → the **prompt cache (A/B) is conclusively
  not the vector**.
- The deployed interim is therefore **ineffective and pure downside** (no isolation, lost
  prefix-reuse). There is **no flag-only clean config**.

**Root cause (code-grounded).** The qwen3next Gated-DeltaNet recurrent state lives in
`s_l[il]` (shape `[n_embd_v_s, n_slots]`), indexed by `qnext_slot_alloc`
(`src/qnext-state-slot-allocator.h`) — a pool **separate** from the attention KV cells. For
`--parallel 1`, seq 0 permanently maps to slot 0. The public `llama_kv_cache_seq_rm` wrapper
(`src/llama.cpp:9296`) *does* `release(seq_id)` on a full clear (`p0<=0 && p1<0`) — but
`release()` only returns the slot **index** to the free list; it **never zeroes the slot's
`s_l` state**. The next request's `alloc(0)` pops the same slot 0 (LIFO) with the prior
conversation's recurrent state intact. `seq_rm` / `kv_cache_clear` touch only
`cells[].pos/src/seq_id`, never `s_l`. The engine comment at `llama.cpp:5612` assumes "a
freshly-allocated slot starts with zero recurrent state" — true only for init-zeroed slots,
**false for a re-allocated slot.** That gap is the leak.

---

## Incident (2026-05-29)

After deploying the H2 defrag fix (`PHASE_TU102_SPECIALIZATION`, build 4858),
post-deploy smoke prompts ("explain GPU warp scheduling") were sent to the **live**
production server. A subsequent user message of **"hi" returned a full GPU-warp
answer** — one conversation's content surfaced in an unrelated request.

**Root cause.** The fork selects cached prompts by **fuzzy Dice similarity**, not
exact prefix:

- `get_slot_similarity(lcp, plen, clen) = lcp*2/(plen+clen)` (`server-common.cpp:165`),
  threshold `cache_ram_similarity = 0.5`.
- For a short prompt the shared chat template (~30 tok) dominates, so
  `sim("hi", cached) ≈ 0.80 > 0.5` → the unrelated entry is selected
  (`server-task.cpp:1093-1115`).
- For the **hybrid** model the restored recurrent state cannot be rewound to the
  LCP, so the restore leaked the cached conversation's content rather than reusing
  only its prefix.

Operational note: smoke-testing the **live** server polluted the cache. Future
deploy verification must use a separate instance or clear the cache (restart)
afterward.

---

## Best-practice evidence (located 2026-05-29)

- **Exact-prefix, not fuzzy.** vLLM = block-hash (sha256) over exact tokens; SGLang
  RadixAttention = "exact same token sequence IDs". The 0.5-Dice match is the
  anomaly and the proximate cause.
  - <https://docs.vllm.ai/en/latest/design/prefix_caching/>
  - <https://www.lmsys.org/blog/2024-01-17-sglang/>
- **Isolation = cache salting.** vLLM `cache_salt` is injected into the first block
  hash; only same-salt requests reuse. The model for `X-Prompt-Cache-Salt` below.
- **Hybrid/SSM is the hard part.** "SSM states are updated in-place so a request's
  states cannot be rolled back to represent its prefixes"; "all-or-nothing
  reusability" (SGLang/PyTorch hybrid blog). vLLM calls SSM+prefix-caching
  experimental. SOTA = Marconi.
  - <https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/>
  - <https://github.com/vllm-project/vllm/issues/17140>
  - Marconi: <https://arxiv.org/pdf/2411.19379>
  → Recurrent reuse is valid **only at a checkpoint that is a true prefix**; else
  reprocess. Cannot rewind to an arbitrary LCP.

---

## Design

Three layers. Salt = the isolation boundary; min-LCP = require a long real prefix;
recurrent-rewind = within-boundary correctness.

### A. Minimum-LCP guard (`MIN_CACHE_REUSE_LCP = 2048`)

Workload is **agentic** (large shared system + tool-schema prefix), so reuse only
matters — and is only safe — when the shared prefix is large. Gate the selection on
the common-prefix token count:

- `server-task.cpp:1099` loop: `if (lcp_cur.first < MIN_CACHE_REUSE_LCP) continue;`
- 2048 ties reuse to meaningful prefill savings (~13 s at 157 t/s), clears the
  ~30-tok template coincidence, and aligns with `--batch-size 2048`.
- `cache_ram_similarity` can stay 0.5 behind the guard. Pragmatic re-impl of "require
  a long exact prefix" (vLLM/SGLang exact-prefix discipline).
- Cost: conversations with <2048 shared prefix lose reuse — but their prefill is
  cheap, so low cost; agentic prefixes (≫2048) are preserved.

### B. Per-request `X-Prompt-Cache-Salt` scoping (vLLM `cache_salt` model)

Salt is opaque cache metadata, **never tokenized or sent to the model** → no context
pollution, no output change. ~15 lines over 4 files:

1. `slot_params` (`server-task.h:49`): `std::string cache_salt;`
2. `server_prompt` (`server-task.h:380`): `std::string cache_salt;` + thread through
   `clone()` (and `to_json`/`from_json` only if salted disk-persisted prompts are
   wanted).
3. Handler (`server.cpp`, where auth header is read ~:663):
   `task.params.cache_salt = req.get_header_value("X-Prompt-Cache-Salt");`
4. Save: set `server_cached_prompt.cache_salt = slot.params.cache_salt;` before
   `prompt_save()` → `alloc()` (`server-task.cpp:1178`) carries it into the state.
5. Selection gate (`server-task.cpp:1099`, top of loop):
   `if (it->cache_salt != prompt.cache_salt) continue;`

Empty salt = today's behavior (backward-compatible). nginx (stock) derives it:

```nginx
location /v1/ {
    proxy_set_header X-Prompt-Cache-Salt $cookie_session;   # or $http_authorization / tenant key
    proxy_pass http://127.0.0.1:8080;
}
```

`proxy_set_header` overwrites client-supplied values (no spoofing; 8080 is
127.0.0.1-only). **Granularity:** per-tenant (auth key) preserves cross-session
agentic reuse + isolates tenants; per-session (cookie) is stronger isolation but
loses cross-session reuse.

### C. qnext slot zero-on-release — THE fix (design-review)

This is the load-bearing isolation fix. The 2026-05-30 (#2) repro localized the leak to
the qnext recurrent state slot being **re-allocated without zeroing**. The fix zeroes the
slot's `s_l` state at the exact moment the slot is returned to the free list.

**Where.** `llama_kv_cache_seq_rm` wrapper, `src/llama.cpp:9296` — the single point where a
seq's slot is released. The server's context-switch always routes here: `apply_checkpoint`
forces `n_past=0` when there's no usable prefix checkpoint, and the consume path
(`server-context.cpp:3925-3944`) then calls `llama_session_kv_seq_rm(slot.id, 0, -1)` (or the
`-1,-1` fallback), both of which satisfy the wrapper's `p0<=0 && p1<0` release condition. A
*valid* prefix checkpoint, when present, restores correct prefix state via `checkpoint_restore`
— no zeroing needed and none done.

**What.** Before `release(seq_id)`, look up the slot index and zero that slot's column in
every `s_l[il]`:

```cpp
if (ok && seq_id >= 0 && p0 <= 0 && p1 < 0 && n_slots > 0) {
    const int32_t slot = qnext_slot_alloc.lookup(seq_id);   // must read BEFORE release
    if (slot >= 0) ctx->transformer_kv.qnext_zero_slot(slot);
    qnext_slot_alloc.release(seq_id);
}
```

New method `llama_kv_cache::qnext_zero_slot(int32_t slot)`:

- `s_l[il]` is `[n_embd_v_s, n_slots]` F32; column `slot` is contiguous at
  `offset = slot * ne[0] * sizeof(float)`, `size = ne[0] * sizeof(float)`.
- **Non-split** (`->extra == nullptr`): `ggml_backend_tensor_set(s_l[il], zeros, offset, size)`.
- **Split** (`->extra`): iterate `split_info->splits[d]` (each `[size_d, n_slots]`) and
  `tensor_set` zeros into column `slot` of *each sub-tensor* — `offset = slot*size_d*4`. This
  mirrors `checkpoint_restore` (`llama.cpp:1986-1993`). **Do not** `tensor_set` the top-level
  split handle: `ggml_backend_cuda_split_buffer` aborts on it (engine comment, `llama.cpp:5621`).
- One small `std::vector<float>` of zeros sized to the largest column, reused across layers.

**Why this is correct & safe.**

- Runs at the inter-tick release site (single-threaded `update_slots`, no decode in flight for
  that slot; C1 single-dispatcher holds), not mid-graph-build like the fill-site — so it avoids
  the crash the `cells[sid].src=0 + do_copy` attempt hit (`llama.cpp:5623`).
- `ggml_backend_tensor_set` is synchronous; the zeroed state is visible to the next decode.
- Cost: one GPU memset of `n_layer × n_embd_v_s × 4` bytes per context-switch — negligible.
- Generality: per-slot (not whole-tensor) zeroing keeps concurrent slots (`--parallel > 1`)
  intact; for `--parallel 1` only slot 0 is ever touched.
- Secondary site: `llama_kv_cache_clear` (full reset, `llama.cpp:2427`) also leaves `s_l`
  stale, but the server's per-request switch does not call it — covering it is defensive,
  optional, and can zero all slots.

**Verification (build-tree, one prod stop).**

1. **Fix binds:** re-run `qnext-slot-leak-repro.sh` (5 trials × distinct topics, cross-topic
   scoring) → **0/5 LEAK** on the fixed binary, vs the 3/5 baseline above. This is the binding
   gate — it exercises the exact cross-conversation switch the bug needs.
2. **Coherence:** the switch answers are correct (Paris), and a long prime still produces a
   sensible summary (state genuinely zeroed, not corrupted).
3. **Regression — determinism:** LM NP={1,2,4,8} ×3 + SERIALIZE A/B still PASS (the new
   tensor_set must not perturb the determinism contract; governor=performance, clocks locked).
4. **Regression — reuse:** a same-prefix re-visit still reuses (zeroing fires only on full
   clear, never on a valid checkpoint restore).

Deploy via `deploy-llama-server.sh` only after 1–4 pass; then **revert the useless interim**
(`--cache-ram 0 --ctx-checkpoints 0` → restore `.bak-preleak-20260530`) to regain prefix-reuse.

**Resolved sub-questions (code-read 2026-05-30 #2):**
- *Conv-state tail?* No separate buffer. `llama-delta-net.cpp:30-33,329-360`: the qwen3next
  state row is `state_dim = conv_state_dim + ssm_state_dim`, both packed contiguously in the
  single `s_l[il]` column (`n_embd_v_s() == state_dim`). Zeroing the `s_l` column covers both.
- *Position after reset?* `keep_first(0)` empties `cache_tokens`; `pos_next()` then returns 0
  (`server-common.cpp:1610`), so the France reprocess gets `batch.pos[0] = 0`
  (`server-context.cpp:4014`).

**Crucial finding — an existing reset path is already firing and FAILING.** `llama-delta-net.cpp:719`
computes `reset_state = (batch.pos[0] == 0)` on the `all_same_seq` fast path (always taken at
`--parallel 1`), and when true the graph does `ggml_scale(state_f32, 0.0f)` ("state_reset",
line 354). By the position analysis above this **should** fire on every context-switch reprocess
— yet the leak persists. So the bug is **not** "no reset exists"; it is that the existing reset is
ineffective. Most likely (unconfirmed): the SSM kernel reads the slot's initial state from the
`s_l[slot]` storage directly (via the `inp_s_seq_qnext` slot map / `cells[].src` copy), so the
`ggml_scale(0)` on a cast/view of `state_dst` never reaches what the kernel actually consumes.

**Why zero-at-release is the robust fix regardless.** It guarantees `s_l[slot]` storage is zero at
re-allocation, so whether the kernel reads storage (hypothesis above), or the scale-reset never
fires, or the write-back re-persists — the next decode reads zero. It addresses the root
(stale storage) rather than the symptom.

**One-probe disambiguation (recommended before build, build-tree only, ~1 short run).** Add a
temporary `LOG_INF` at `llama-delta-net.cpp:719` printing `il, batch.pos[0], reset_state`, plus a
cheap checksum of `s_l[slot]` before/after the France prefill. This tells us (i) whether
`reset_state` is actually true at runtime, and (ii) whether storage is non-zero going into the
reprocess. Confirms the mechanism before committing the fix, and reveals whether the existing
reset path is a separate latent bug worth its own fix.

### Best-practice grounding (located 2026-05-30 #2)

The open design questions were resolved against how production hybrid/SSM servers handle this
**exact** bug class, not by local judgement.

- **This is a known, named bug class.** AI21's "one token to corrupt them all" post-mortem
  describes the identical failure: *"stale recurrent state leaking between requests sharing a
  cache slot … the cache slot contained garbage from a previous, completed request."* Their fix
  was **not** to memset storage on free — it was to **correctly classify the new sequence** so it
  runs the **prefill kernel, which always initializes state to zero** (`num_computed_tokens==0`
  ⇒ prefill ⇒ zero-init). Lesson quoted: *"the SSM state is loaded as a complete vector before
  updates, making isolation critical for Mamba-based models sharing cache slots."*
- **The SOTA mechanism is a kernel-honored per-sequence flag, not storage scrubbing.** vLLM's
  selective-scan takes `has_initial_state` (bool per batch row): *"if false (new sequence),
  initialize to zero; if true, load from cache,"* with the initial state addressed by
  `cache_indices` / `initial_state_idx` (storage **by index**). So (a) the kernel reads initial
  state from the slot's storage by index — confirming why our `ggml_scale(0)` on a discarded view
  is cosmetic — and (b) the idiomatic reset is a flag the **kernel** branches on, applied at the
  fresh-sequence decode, never a free-time memset.
- **SGLang** enforces isolation at the **state-storage boundary** (snapshot-copy of the matched
  state per request: *"the matched state must be fully copied as a snapshot … ensuring
  isolation"*) — again, a value operation on the state storage at sequence start, not a flag on a
  graph intermediate.

**Resolved decisions (grounded):**

1. **Repair the existing `reset_state` path — don't add a parallel free-time memset.** The ik fork
   *already has* the SOTA mechanism: `reset_state = (batch.pos[0]==0)` is precisely vLLM's
   `has_initial_state==false` / AI21's `num_computed_tokens==0`, and our position analysis shows
   it is correctly *computed* on every context-switch. It is only **mis-wired**: the zero is
   applied to a cast/view (`state_f32`) the SSM kernel never consumes, while the kernel loads the
   initial state from `s_l[slot]` storage by index (vLLM-style). The fix is to make `reset_state`
   actually zero the **storage the kernel reads** — i.e., honor the flag where vLLM/AI21 honor it
   (fresh-sequence prefill), keeping the reset on the decode path the engine already intended.
2. **Trigger at fresh-sequence use, not at slot release.** vLLM zero-inits in the prefill kernel
   at first use; AI21's fix routes new sequences through prefill. Zeroing on *free* is off-pattern
   (wasted when a slot is never reused; leaves the correctness dependent on free-time bookkeeping
   rather than the prefill classification that every fresh sequence already performs).
3. **Zero-at-release is demoted to a defensive fallback,** not the primary fix. It is still robust
   (guarantees storage is zero regardless of the flag), so it remains the contingency if the probe
   shows the `reset_state` repair is more invasive than expected — but it is not the SOTA shape.
4. **Probe first (unchanged, reinforced).** The AI21 case shows how subtle the trigger is (a
   scheduler edge-case mis-set the equivalent flag). The one-probe step (log `batch.pos[0]` /
   `reset_state`, checksum `s_l[slot]` pre/post prefill) confirms the flag's runtime value and
   that the kernel's consumed initial state is the stale storage — before any code change.
5. **`llama_kv_cache_clear` defensive zero:** optional, low-priority. Isolation is load-bearing on
   the per-sequence reset flag (1), as in vLLM/SGLang; a full-clear memset is belt-and-suspenders.

**Revised primary fix:** make `reset_state==true` zero the recurrent-state storage the SSM kernel
actually reads for the fresh sequence (`s_l[slot]` column — conv+ssm, one column), at the
fresh-sequence prefill — repairing the existing in-graph reset rather than adding a free-time
memset. Verification gates unchanged (repro → 0/5 + determinism + reuse intact).

### PROBE RESULT 2026-05-30 (#2) — recurrent state RULED OUT; fix hypothesis NEGATIVE

`data/whole-system-profile/qnextprobe-20260530T174109`, two env-gated arms on the prod-equiv
config (binary built with `[QNEXT_PROBE]`/`[QNEXT_PROBE_ZERO]`, both default-off):

- **Arm A (`QNEXT_PROBE=1`):** `reset_state=1` on **every** context-switch (`pos0=0`, confirmed
  by the log) — the existing in-graph reset IS firing — yet leak persists **3/5**. And the scaled
  state is consumed: `state_f32 = ggml_scale(state_dst, 0)` feeds `ggml_ssm_conv` (delta-net.cpp:368)
  and `build_fused_delta_net` (:410). So the DeltaNet layers process the new prompt from a **zero
  recurrent input** and aqueduct still leaks.
- **Arm B (`+QNEXT_PROBE_ZERO=1`):** zeroing `transformer_kv.s_l[slot]` at release executed
  (`zeroed s_l slot 0 across 48 layers`) — leak **unchanged: 3/5, byte-identical pattern**
  (t0/t2/t3 leak, t1/t4 clean, in both arms).

**Conclusion: the qnext recurrent state is NOT the leak vector** (ruled out by two independent
zeroing mechanisms — in-graph input scale + storage memset). The earlier root-cause framing
(qnext slot reuse) was **wrong**; reopen the diagnosis. The fix design above is **withdrawn**.

**Residual ambiguity (one untested path):** the probe did not checksum that Arm B's split-tensor
`tensor_set` actually landed, and did not confirm whether `ggml_ssm_conv` re-gathers conv state
from storage via `inp_s_seq_qnext` ignoring the zeroed arg. So a narrow recurrent-via-conv-storage
path isn't 100% excluded — but Arm A (consumed input zeroed in-graph, definitely lands) makes the
recurrent vector unlikely.

**Leading hypothesis now: the hybrid's full-attention-layer KV cache** is not evicted (or is
re-attended) on the context-switch, so the new prompt attends to the prior conversation's tokens.
`cached_tokens=0` rules out prompt-cache reuse, not stale KV cells. Next probe must (a) checksum
that a recurrent-storage zero lands, and (b) instrument whether `seq_rm` actually clears the
attention KV cells for seq 0 on the switch (cell count + surviving positions), and/or clear the
attention KV explicitly and re-test. Until then **Task C has no validated fix.**

### PROBE #2 RESULT 2026-05-30 (#3) — VECTOR DETERMINED: full `kv_cache_clear` fixes it (0/5)

`data/whole-system-profile/qnextkv-20260530T180051`, env-gated arms (`[QNEXT_PROBE_KV]` cell-survival
log + `[QNEXT_PROBE_KVCLEAR]` hard `llama_kv_cache_clear` on the full-clear switch, both default-off):

- **Arm C (log only):** **3/5 leak** (baseline). The log shows `surviving seq cells=0 used=0` after
  **every** `seq_rm(seq=0,p0=0,p1=-1) ok=1` → the attention KV cells **are** evicted. **Surviving
  stale cells are NOT the mechanism.**
- **Arm D (`+ kv_cache_clear` on switch):** **0/5 leak.** All five switches answer cleanly ("the user
  asks 'capital of France'…"), zero aqueduct bleed.

**Conclusion — vector determined.** `llama_kv_cache_clear()` on the context-switch eliminates the
leak; the per-request `seq_rm(0,-1)` the server calls does not, *despite both leaving `cells=0`*. The
vector is what `kv_cache_clear` resets **beyond** `seq_rm` (`llama.cpp:2427` vs `:2454`): `cells[i].src
= i` for **all** cells, `head = 0`, and `paged.free_seq()`. The leak rides the **recurrent-state
copy-source / allocator binding** (`cells[].src` + `head`) that `seq_rm` leaves dangling — the new
sequence inherits a stale state-row linkage and the SSM kernel pulls prior state through it. This
reconciles Arms A/B: zeroing the state *value* (in-graph input; `s_l` storage) didn't help because the
kernel re-establishes state via the stale *binding*, not the zeroed value.

**Validated fix direction (Task C):** on the recurrent/hybrid context-switch where the server forces
full reprocess (`apply_checkpoint` do_reset → `seq_rm(0,-1)`), invoke the full `kv_cache_clear` (or
replicate its `cells[].src=i` + `head=0` + `paged.free_seq()` reset) instead of/in addition to the
plain `seq_rm`. Empirically 0/5. Open precision question (optional micro-probe): which single field
(`src` vs `head` vs allocator) is load-bearing — for a minimal fix rather than the full clear.
Verification gates unchanged (repro → 0/5 + NP determinism + reuse intact).

### C — PERFORMANCE-PRESERVING FIX PLAN (2026-05-30 #4)

**Performance principle.** The binding reset is only required on the **no-reuse path** — the
`do_reset` context-switch, where the server is *already* doing a full prefill (the expensive part).
The reset itself is an O(cells-of-this-seq) host loop (~µs) dwarfed by that prefill, so it adds ≈0%.
Crucially it must **never** fire on the reuse paths (valid checkpoint restore; prompt-cache hit),
which already restore the correct binding (`checkpoint_restore` copies `cells_snapshot` incl. `src`,
`llama.cpp:1976`; `prompt_load`→`llama_state_seq_set_data` restores serialized cells). So a correctly
*scoped* fix costs nothing on the fast path. The big win is then re-enabling the reuse the interim
disabled — current prod (`--cache-ram 0 --ctx-checkpoints 0`) reprocesses **everything** and has **no**
fast-restore, so the fix is a net perf *gain*, not a tax.

**Step 1 — Minimal-field micro-probe — DONE (2026-05-30 #4, `qnextmp-20260530T182...`).** Three
env-gated arms applied one `kv_cache_clear` action each, after `seq_rm`, on the full-clear switch:
  - **SRC** (`cells[].src=i` all cells): **3/5 leak** — the SSM copy-source is NOT the binding.
  - **HEAD** (`head=0` + `v_heads=0`): **0/5 — load-bearing.** Almost certainly `v_heads`, which
    indexes the qwen3next **recurrent V-cache tail** (`llama.cpp:845` "qwen3next recurrent state is
    stored in a dedicated V-cache tail (per sequence)"). This reconciles every prior negative: the
    leaking state lives in the V-cache tail, so zeroing `s_l` (Arm B) and `src` (Arm SRC) missed it.
  - **PAGED** (`paged.free_seq` standalone): **CRASH** — `GGML_ASSERT(blk_idx < btbl.size())`
    `llama.cpp:5419` (breaks the T5.2/T5.3 block-table invariant `seq_rm` maintains via its paired
    re-alloc). `paged` reset is unsafe in isolation → **ruled out**.
  ⇒ Minimal validated reset = **`head=0` + `v_heads=0`** (O(n_stream), no cell loop, no crash).
  Open sub-question (optional, not blocking): split head-only vs v_heads-only — `v_heads` is the
  expected load-bearing one. Resetting both is cheap and validated, so the fix uses both.

**Step 2 — Minimal binding reset at `do_reset` (the fix).** Step 1 settled the field: reset
**`head` and `v_heads`** on the recurrent/hybrid full-clear switch. In `apply_checkpoint` do_reset
(or the consume-path full-clear, `server-context.cpp:3925`), after `seq_rm(0,-1)`:
  - `--parallel 1` (current prod): `transformer_kv.head = 0; v_heads[0] = 0;` — exact validated 0/5.
  - `--parallel > 1` (future-safe): scope the `v_heads` reset to the **switching seq's stream**, not
    all streams (a global `v_heads=0` could disturb other live seqs' recurrent V-tail index). `head`
    is a global allocation search-hint (find_slot checks occupancy), so resetting it is safe either
    way. The seq→stream map is available at the call site.
  - **Do NOT** touch `paged` (Step 1: standalone `free_seq` crashes the block-table invariant) or
    use the full `kv_cache_clear` (heavier, and wrong for multi-slot). Gate strictly to the
    recurrent/hybrid + do_reset path so the reuse path is untouched.
  Cost: O(n_stream) integer writes — strictly cheaper than the full clear, negligible vs the prefill
  that the do_reset path is already doing.

**Step 3 — Restore reuse (the actual performance recovery).** Revert the interim profile to
`--cache-ram 40960 --ctx-checkpoints 64` (restore `.bak-preleak-20260530`). With Step 2 making the
context-switch leak-free, the reuse mechanisms are safe and provide:
  - `--ctx-checkpoints` → hybrid fast-restore for same-conversation continuation (avoid reprocessing
    the whole conversation each turn — the dominant agentic-loop saving).
  - `--cache-ram` (prompt cache, gated by A min-LCP=2048 + B salt) → cross-request prefix reuse.
  Both restore states *including* the correct binding, so they don't reintroduce the leak (verify).

**Step 4 — Verification gates (binding on each claim).**
  1. Leak: `qnext-slot-leak-repro.sh` → **0/5** on the fixed binary (binding gate).
  2. Reuse leak-free: repro with `--ctx-checkpoints 64` AND `--cache-ram 40960` re-enabled → still
     0/5 (re-enabling reuse must not reopen the leak via the restore paths).
  3. Reuse *works*: same-conversation continuation shows checkpoint fast-restore (prefill ≪ full);
     a genuine ≥2048 shared-prefix re-visit shows prompt-cache reuse (`cached_tokens>0`, faster).
  4. Determinism: LM NP={1,2,4,8} ×3 + SERIALIZE A/B PASS (governor=performance, clocks locked).
  5. Perf: NP=1 t/s ≥ pre-leak baseline; context-switch latency ≈ unchanged (reset is µs).
  Deploy via `deploy-llama-server.sh` only after 1–5; the deploy *is* the perf restoration (interim
  off). Net effect vs today: leak-free **and** faster (reuse restored).

**Why this is max-perf, not just correct:** the only added work (the reset) lands solely on
context-switches that were already full reprocesses; every reuse path is preserved; and the fix
unblocks reverting the interim, which is where the real speed (checkpoints + prefix reuse) comes back.

Sources: AI21 "one token to corrupt them all" (vLLM Mamba state-bleed post-mortem);
PyTorch/vLLM "Hybrid Models as First-Class Citizens"; vLLM `mamba_ssm` selective-scan API
(`has_initial_state`, `cache_indices`); PyTorch/SGLang "Hybrid Models Meet SGLang" + MambaRadixCache.

---

### D. Fair-share eviction across salts

A/B give *isolation* but not *fairness*: the cache is one shared pool with a global
byte limit, and `server_prompt_cache::update()` (`server-task.cpp:1208`) evicts the
**global oldest** (`states.pop_front()`) when over `limit_size`. New entries
`emplace_back()` to the tail, so a busy salt continually pushes other tenants'
entries to the front where they're evicted first → **starvation** of less-active
salts.

Fix — max-min fair eviction: when over limit, evict the oldest entry of the salt
with the **largest current footprint**, not the global oldest:

```cpp
while (states.size() > 1 && size() > limit_size) {
    // tally bytes per cache_salt; pick the salt with the most bytes;
    // evict that salt's oldest entry (front-most with that salt).
}
```

This converges each contending salt toward an equal share without hard quotas
(which would need a dynamic active-salt count), and protects small/idle tenants.
O(states) per eviction; `states` is small. Emergent fair-share, not a strict quota;
optional refinement: a per-salt floor. The no-salt ("") pool competes as just
another salt.

## Verification

1. **Isolation:** req A (salt=a, long prompt) then req B (salt=b, same prefix,
   different tail) → no "found better prompt" for A; B reprocesses.
2. **Reuse preserved:** A then B same salt + shared ≥2048 prefix → "found better
   prompt", B faster.
3. **Backward-compat:** no salt → greedy byte-identical to today.
4. **min-LCP:** short same-salt prompt (LCP<2048) → no reuse.
5. **Recurrent:** a same-salt B sharing a prefix with A never emits A's later content.

---

## Tasks

- [~] A — `MIN_CACHE_REUSE_LCP = 2048` guard in `server_prompt_cache::load()`.
      **Implemented + verified + DEPLOYED** 2026-05-29 (submodule `eb98047f`, build 4859). Verified: short "hi" `reuse_delta=0` + topic-free; long
      re-visit `reuse_delta=1` (prefill 3035→475 tok).
- [~] B — `X-Prompt-Cache-Salt` plumbing (4 files) + nginx `proxy_set_header`.
      **Implemented + verified + DEPLOYED** (build 4859). Verified:
      `Y@beta reuse_delta=0` (isolation) vs `Y@alpha reuse_delta=1` (same-salt
      reuse). nginx `proxy_set_header` line still to be added at deploy.
- [~] **C — IMPLEMENTED + gates 1–3 PASS (2026-05-30 #5).** Fix in `seq_rm` wrapper
      (`llama.cpp:9296`): on the hybrid full-clear context-switch, after `seq_rm`+slot-release, reset
      `head=0` + the seq's stream `v_heads[stream]=0` (the recurrent V-cache-tail binding). Verify
      `qnextverify-...`: FIX0 (prod-equiv) **0/5**, REUSE (cache-ram 40960 + ctx-ckpt 64) **0/5**,
      reuse-works (same prompt 2×: pass2 `cached=3072 prompt_n=208` vs pass1 3280 — fast path intact).
      Remaining to close `[x]`: gate 4 NP determinism + gate 5 perf, then deploy + revert interim.
- [ ] (superseded) VECTOR DETERMINED (2026-05-30 #3). Probe #2
      (`qnextkv-20260530T180051`) proved `llama_kv_cache_clear` on the context-switch → **0/5 leak**
      vs `seq_rm` alone → 3/5, with attention cells already `=0` either way. Vector = the
      recurrent-state copy-source / allocator binding (`cells[].src` + `head` + paged free) that
      `seq_rm` leaves stale. Fix: do a full `kv_cache_clear` (or replicate src/head/allocator reset)
      on the recurrent context-switch. Recurrent-state-value and attention-cell-survival hypotheses
      both disproven (Arms A/B/C). Implement + verify (repro 0/5 + determinism + reuse) next.
- [ ] D — fair-share (max-min) eviction across salts so a busy tenant can't starve
      others; replaces the global-FIFO `update()` evict.
- [ ] Verify C (build-tree): `qnext-slot-leak-repro.sh` 0/5 + NP determinism + reuse intact;
      deploy via `deploy-llama-server.sh`; then revert the interim.
- [x] ~~Interim mitigation~~ **INEFFECTIVE** — `--cache-ram 0 --ctx-checkpoints 0` deployed
      2026-05-30 on a 1-sample fluke; the #2 multi-sample repro proved it does **not** fix the
      leak (3/5 either way) and only costs prefix-reuse. Prod left as-is per user decision
      pending C; **revert `.bak-preleak-20260530` when C deploys.**

## Out of scope

- Replacing the similarity cache with a radix/hash exact-prefix cache (vLLM/SGLang
  architecture) — larger rewrite; min-LCP + salt approximate it.
- DFlash / speculative-decoding cache paths.
