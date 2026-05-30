# PHASE_PROMPT_CACHE_ISOLATION — fix cross-conversation leak in the server prompt cache

**Opened:** 2026-05-29.
**Branch:** `production/2026-q2-next`.
**Scope:** Make the `llama-server` host-memory prompt cache safe against
cross-conversation / cross-tenant leakage on the qwen35 hybrid (Mamba2), without
losing the prefix-reuse speedup that agentic workloads depend on. Two landing
changes — a **minimum-LCP guard** and **per-request `X-Prompt-Cache-Salt`
scoping** — plus an audit of the hybrid recurrent-state rewind.
**State:** Designed + evidence-gathered. **Not implemented.** Live leak
mitigation (restart / `--cache-ram 0`) recommended, not yet authorized.

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

### C. Recurrent-rewind correctness (audit)

Even within a salt, two same-salt agentic sessions share the big prefix and WILL
match — safe only if the restore rewinds recurrent state to the LCP (a checkpoint at
that boundary) or reprocesses. The "hi" leak shows the "no usable checkpoint →
reprocess" path (logged) is not always honored. Audit `prompt_load` /
`llama_state_seq_set_data` + the hybrid checkpoint apply so a match never inherits
the cached session's **end** state.

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

- [ ] A — `MIN_CACHE_REUSE_LCP = 2048` guard in `server_prompt_cache::load()`.
- [ ] B — `X-Prompt-Cache-Salt` plumbing (4 files) + nginx `proxy_set_header`.
- [ ] C — audit + fix the hybrid recurrent-state rewind on prompt-cache restore.
- [ ] D — fair-share (max-min) eviction across salts so a busy tenant can't starve
      others; replaces the global-FIFO `update()` evict.
- [ ] Verify (1–5 above) on the build-tree binary; deploy via
      `deploy-llama-server.sh`.
- [ ] Live-leak interim until A/B land: restart to clear, or `--cache-ram 0` /
      raised `cache_ram_similarity`.

## Out of scope

- Replacing the similarity cache with a radix/hash exact-prefix cache (vLLM/SGLang
  architecture) — larger rewrite; min-LCP + salt approximate it.
- DFlash / speculative-decoding cache paths.
