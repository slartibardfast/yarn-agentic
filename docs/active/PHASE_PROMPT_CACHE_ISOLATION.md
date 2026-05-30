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

**Open design questions for review:**
- Zero-on-release vs zero-on-recycled-alloc — release is simpler (have the seq→slot map) and off
  the hot decode path; chosen. (Release only fires on full-clear = seq done; no path re-allocs a
  released slot expecting its old state.)
- Fix the existing `reset_state` path *as well* (make the kernel consume the zeroed state), or
  rely solely on zero-at-release? The probe decides — if the scale-reset is structurally
  cosmetic, zero-at-release is the clean single fix; if it's a near-miss, repairing it may be
  cheaper and keeps the reset on the decode path where the engine intended it.
- Should `llama_kv_cache_clear` also zero (defensive)? Leaning yes, cheap.

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
- [ ] **C — CRITICAL (the only real fix)** — qnext slot zero-on-release (see "Task C design").
      Root cause localized 2026-05-30 (#2): `release()` returns the slot index without zeroing
      its `s_l` recurrent state; re-alloc reuses the stale state. Fix = `qnext_zero_slot(slot)`
      called before `release()` in the `seq_rm` wrapper (`llama.cpp:9296`). **Design-review
      stage** (user-requested), not yet implemented. Verify: repro → 0/5 + determinism intact.
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
