# DFlash Phase 5 — server adapter glue pickup brief (2026-05-18)

Continuation of `data/dflash-multi-slot-api-brief-2026-05-18.md`,
`data/dflash-multi-slot-impl-plan-2026-05-18.md`,
`data/dflash-phase3-pickup-2026-05-18.md`. Phases 1–4 are landed and
verified; this brief covers Phase 5, the server-orchestrator glue that
makes the multi-slot C API actually reachable from `llama-server`.

## State at handover

**Landed (production/2026-q2-next):**
- Submodule HEAD: `c3...` (Phase 4 drafter_forward guard) → `0dbc23b3` (Phase 4 dispatch) → `c33f75da` (Phase 3 cb_eval demux) → `fa3e50c7` (Phase 2 scratch) → `8008feaf` (Phase 1 surface)
- Parent HEAD: `a13c0ef` (harness clock-check + drafter_forward guard bump) → `ffa9419` (Phase 4 spec/tooling) → `32f5c69` (Phase 3 bump) → `7d997a5` (Phase 2 bump)

**Gates all green (locked clocks 1455 MHz, dual Quadro RTX 6000):**
- `test-dflash-closure` 8/8 prompts argmax-equivalent
- `test-dflash-np-invariance` (T7) N∈{1,2,4,8} byte-identical FNV-1a64 hashes, 4 seeds
- `test-dflash-extract-multi-seq` (Phase 3) per-seq buffer counts correct
- `test-dflash-batch-vs-serial` (Phase 4 binding) `n_slots=2 ≡ two serial n_slots=1` byte-identical
- `scripts/verify-production-determinism.sh` NPC PASS NP∈{1,2,4,8} multi-GPU

**Tooling:**
- `scripts/gpu-clocks.sh lock|unlock|status` — required before NPC, locks both GPUs to 1455 MHz
- NPC harness now has built-in clock precheck; `SKIP_CLOCK_CHECK=1` to bypass

## The Phase 5 problem in one paragraph

`llama_dflash_draft_batch(ctx, n_slots, ...)` is now the kernel-pipeline
entrypoint, but `common_speculative_state_dflash::draft()` at
`common/speculative.cpp:1012` still calls the single-slot
`llama_dflash_draft`. The server orchestrator
(`common_speculative_draft_batched` at line 1442) has an all-MTP fan-out
path but no all-DFlash equivalent. Phase 5 adds the all-DFlash fan-out so
that `llama-server --spec dflash --parallel N` dispatches each draft
cycle through ONE `llama_dflash_draft_batch` call instead of N serial
`llama_dflash_draft` calls.

## Key architectural constraint (read first)

`llama_set_dflash(ctx_tgt, drafter)` errors on the second call to the
same context (`src/llama-dflash.cpp:407`: "called twice on same
context"). The drafter binds ONCE per context. This means Phase 5 must:

- Construct ONE `common_speculative_state_dflash` that owns the drafter
  binding for ctx_tgt.
- Each per-slot `common_speculative_state_dflash` shares the same ctx_tgt
  but cannot rebind the drafter. The current per-slot construction at
  `common/speculative.cpp:1281` would crash at the second slot because
  it calls `llama_set_dflash` again.

This is the architectural difference from MTP, where each slot's state
captures the shared ctx but doesn't try to bind a drafter — MTP's
"drafter" is the target model itself.

**Two viable approaches:**

### Option A — share-the-bind via dispatcher gating
The first per-slot state to be constructed binds the drafter. Subsequent
per-slot states detect that the ctx already has a DFlash binding and
SKIP `llama_set_dflash`, recording a non-owning pointer to the same
drafter. Teardown: only the owning state frees the drafter.

Pros: minimal new types; the per-slot state still implements the
`common_speculative_state::draft()` contract (each slot's `draft()` calls
`llama_dflash_draft_batch(n_slots=1, seq_ids=[slot.seq_id])`).
The batched fan-out is purely an orchestrator optimization; per-slot
fallback works without it.

Cons: ownership-confusing — the "first" state owns the drafter and the
others don't, which is brittle when callers free states in arbitrary
order.

### Option B — shared drafter handle on the ctx, refcounted
Add a refcount to `llama_dflash_state` so multiple per-slot wrappers can
co-own the drafter binding. First binder increments to 1; each
additional `common_speculative_state_dflash` calls a new
`llama_dflash_attach_drafter(ctx)` that increments the refcount without
re-binding. Destructor decrements; the last out releases.

Pros: clean ownership model; no "first state is special" footgun.

Cons: new C API surface (`_attach_drafter`); broader change.

**Recommendation:** Option A. The dispatcher logic is gated to "one
per-slot DFlash state per ctx" by inspection — `common_speculative_init`
is called once per slot in the server, so the construction sequence is
known. The owning vs non-owning distinction can be a single bool on the
state struct.

## Files to read first (sub-5k tokens)

1. **`common/speculative.cpp:1437-1640`** — `common_speculative_draft_batched`
   in full. Three things to note:
   - Lines 1452-1493: the per-slot bookkeeping + all-MTP gate.
   - Lines 1495-1506: serial fallback (per-slot `common_speculative_draft`).
   - Lines 1508+: all-MTP batched path. Phase 5 mirrors this for all-DFlash.

2. **`common/speculative.cpp:969-1057`** — `common_speculative_state_dflash`
   definition. Note:
   - Line 974: constructor binds drafter via `llama_set_dflash` — this is
     where the second-bind-error fires at multi-slot construction.
   - Line 1012: `draft()` calls single-slot `llama_dflash_draft`. Phase 5
     must either keep this for the fallback path OR refactor to call
     `llama_dflash_draft_batch(n_slots=1, ...)` (byte-identical, see Phase 4
     trampoline).

3. **`common/speculative.cpp:1262-1276`** — MTP per-slot construction
   pattern. The seq_id flows from `common_speculative_init`'s seq_id
   parameter into `common_speculative_state_mtp`. DFlash currently
   discards seq_id at construction (line 1281-1293); Phase 5 fixes this.

4. **`include/llama.h:1808-1840`** — `llama_dflash_draft_batch` signature.
   Phase 4 already accepts `seq_ids[]` per slot. The fan-out at
   `common_speculative_draft_batched` packs `n_slots × (anchor_id,
   anchor_pos, seq_id)` arrays and calls this once.

5. **`src/llama-dflash.cpp:632-820`** — the multi-slot dispatch
   implementation (just to confirm what the fan-out sees).

## Phase 5 changes (concrete)

### 1. seq_id flow through state construction

`common_speculative_state_dflash` constructor: add `llama_seq_id seq_id`
parameter, store it. Pattern matches MTP at line 1262-1265.

Touches `common/speculative.cpp:969-997`:
```cpp
struct common_speculative_state_dflash : public common_speculative_state {
    llama_context             * ctx_tgt = nullptr;
    llama_dflash_drafter      * drafter = nullptr;
    int32_t                     block_size = 0;
    llama_seq_id                seq_id     = 0;   // NEW
    bool                        owns_drafter = false;  // NEW (Option A)

    common_speculative_state_dflash(
            enum common_speculative_type type,
            llama_context              * ctx_tgt_,
            const std::string          & drafter_path,
            llama_seq_id                 seq_id_)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt_)
        , seq_id(seq_id_)
    { ... }
```

Dispatcher at `common/speculative.cpp:1281-1293` passes seq_id (currently
just `seq_id` in scope from `common_speculative_init`):
```cpp
case COMMON_SPECULATIVE_TYPE_DFLASH: {
    const std::string & drafter_path = ...;
    auto state = std::make_unique<common_speculative_state_dflash>(
        config.type, ctx_tgt, drafter_path, seq_id);   // ← add seq_id
    ...
}
```

### 2. Share-the-bind via Option A

Inside the constructor, detect whether the ctx already has a DFlash
binding. If not, call `llama_set_dflash` and set `owns_drafter = true`.
If yes, record a non-owning pointer to the existing drafter (read from
`ctx_tgt->dflash_state->drafter` — but this is internal; cleanest is to
add a small public getter):

Either:
(a) Add `LLAMA_API struct llama_dflash_drafter * llama_get_dflash_drafter(struct llama_context * ctx);` returning the bound drafter (or nullptr).
(b) Probe by attempting `llama_set_dflash` and checking the error code, then on second-bind-error fetch via a getter.

(a) is cleaner. Add to `include/llama.h` near the existing DFlash API.

Destructor only calls `llama_dflash_drafter_free` if `owns_drafter`.

### 3. Per-slot `draft()` route through `_batch`

Refactor `common_speculative_state_dflash::draft()` (line 1012) to call
`llama_dflash_draft_batch(ctx_tgt, /*n_slots*/ 1, &id_last,
&anchor_pos, &seq_id, result.data(), (int32_t) result.size())`. Phase 4
already makes the trampoline byte-identical to `llama_dflash_draft`, so
this is a pure call-site cleanup that also exercises the multi-slot
path at n_slots=1 — useful regression coverage.

### 4. All-DFlash fan-out in `common_speculative_draft_batched`

Mirror the all-MTP gate (`common/speculative.cpp:1488-1506`) for DFlash:

```cpp
std::vector<common_speculative_state_dflash *> df_states(inputs.size(), nullptr);
bool all_dflash = true;
for (size_t i = 0; i < inputs.size(); ++i) {
    common_speculative * spec = inputs[i].spec;
    if (spec == nullptr || spec->impls.empty()) { all_dflash = false; break; }
    common_speculative_state_dflash * df = nullptr;
    for (auto & impl : spec->impls) {
        if (impl->type == COMMON_SPECULATIVE_TYPE_DFLASH) {
            df = dynamic_cast<common_speculative_state_dflash *>(impl.get());
            break;
        }
    }
    if (df == nullptr || df->drafter == nullptr) { all_dflash = false; break; }
    df_states[i] = df;
}

// Shared ctx invariant
if (all_dflash && df_states.size() > 1) {
    llama_context * ctx0 = df_states[0]->ctx_tgt;
    for (size_t i = 1; i < df_states.size() && all_dflash; ++i) {
        if (df_states[i]->ctx_tgt != ctx0) all_dflash = false;
    }
}

if (all_dflash && df_states.size() > 1) {
    // Pack arrays
    const size_t N = df_states.size();
    std::vector<llama_token>  anchor_ids(N);
    std::vector<int32_t>      anchor_ps(N);
    std::vector<llama_seq_id> seq_ids(N);
    for (size_t i = 0; i < N; ++i) {
        anchor_ids[i] = inputs[i].id_last;
        anchor_ps[i]  = (int32_t) inputs[i].prompt_tgt.size();
        seq_ids[i]    = df_states[i]->seq_id;
    }
    const int32_t BS = df_states[0]->block_size;
    std::vector<llama_token> flat_out(N * BS, 0);
    int32_t rc = llama_dflash_draft_batch(
            df_states[0]->ctx_tgt,
            (int32_t) N,
            anchor_ids.data(), anchor_ps.data(), seq_ids.data(),
            flat_out.data(), (int32_t) flat_out.size());
    if (rc >= 0) {
        for (size_t i = 0; i < N; ++i) {
            out[i].assign(flat_out.begin() + i*BS, flat_out.begin() + (i+1)*BS);
            // Respect per-slot n_max truncation if needed (see MTP path).
        }
        return out;
    }
    // rc<0 → fall through to serial fallback.
}

// Serial fallback (existing all-MTP-fail path) handles all-DFlash-fail too.
```

The fallback path (lines 1497-1506) ALREADY uses per-slot
`common_speculative_draft`, which calls per-slot
`state_dflash::draft()`. With change #3 in place, that calls
`llama_dflash_draft_batch(n_slots=1, ...)` → correct per-slot output. ✓

## Verification gates for Phase 5

1. **Build clean** — `llama`, `llama-server`, `llama-batched-bench`,
   all dflash tests.

2. **All existing gates re-run** — closure 8/8, T7 4/4 seeds, Phase 3
   multi-seq demux, Phase 4 batch-vs-serial. Each must still PASS.

3. **NEW Phase 5 binding test** — `tests/dflash-speculative/test-dflash-spec-batched-fanout.cpp`:
   - Construct two `common_speculative` with seq_ids [0, 1] over a shared ctx (`n_seq_max=2`).
   - Prefill both seqs with the same prompt.
   - Reference: call `common_speculative_draft_batched(inputs=[A, B])` → expect both slots' candidates byte-identical to two serial `common_speculative_draft` calls (one per seq).
   - Asymmetric variant: prefill seq 0 with prompt P0, seq 1 with prompt P1 (DIFFERENT) — confirm fan-out's per-slot outputs differ (proves seq_id flows correctly through to `llama_dflash_draft_batch`).

4. **NPC harness** — must still PASS at NP={1,2,4,8} multi-GPU (Phase 5
   doesn't touch the production decode path, but the binary changed).
   Run `scripts/verify-production-determinism.sh` (clock-locked).

5. **Server smoke at `--parallel 2` with `--spec dflash`**:
   - Start `llama-server -m <target> --parallel 2 --spec dflash --draft <drafter.gguf>`.
   - Issue two concurrent `/completion` requests.
   - Confirm both complete (non-zero token output).
   - Confirm logs show "all-DFlash batched fan-out" debug line firing
     (add LOG_DBG inside the fan-out gate).

## Out-of-scope reminders for Phase 5

- DFlash kernel optimization (lm_head / GEMM rewrites): still future work
  per PHASE_DFLASH item 3.
- Profile flip: production stays on the current deterministic profile
  (`profiles/active.sh -> qwen36-27b-x8-deterministic.sh`) until Phase 5
  passes ALL gates including the new fan-out test.
- DFlash + cont-batching with VARIABLE per-slot `anchor_pos`: Phase 4's
  multi-slot dispatch already handles variable MAL by per-slot
  combine/inject + shared drafter_forward. No further work needed; the
  fan-out just packs the per-slot anchor_pos array.

## How to actually start Phase 5

1. Lock clocks: `sudo bash /home/llm/yarn-agentic/scripts/gpu-clocks.sh lock`
2. Read the 5 files listed above.
3. Confirm by inspection: would Option A's "first state owns drafter"
   pattern work in `llama-server`'s current
   `common_speculative_init`-per-slot construction sequence? (Yes, but
   confirm with a grep of server.cpp.)
4. Add the `llama_get_dflash_drafter` public getter (5 lines).
5. Refactor `common_speculative_state_dflash` constructor +
   `owns_drafter` flag.
6. Refactor `state_dflash::draft()` to call
   `llama_dflash_draft_batch(n_slots=1, ...)`. Run closure test —
   should still 8/8 PASS.
7. Add all-DFlash gate + fan-out in `common_speculative_draft_batched`.
8. Add the new diag test
   `test-dflash-spec-batched-fanout.cpp` and register it in
   `tests/CMakeLists.txt`.
9. Build, run all gates serially (NPC, closure, T7, Phase 3 multi-seq,
   Phase 4 batch-vs-serial, NEW fan-out test, server smoke).
10. Commit submodule + parent bumps + push.

## Useful references

- **Project memory:** `[[dflash-multislot-phase4-landed]]` — what Phase 4 shipped + the N_slots/n_slots_cap split.
- **Feedback memory:** `[[drafter-forward-n-slots-cap]]` — kernel-storage-stride lesson learned.
- **Spec:** `specs/dflash/kernel-design.md` §6.1 (signature + clarification #4).
- **Impl plan:** `data/dflash-multi-slot-impl-plan-2026-05-18.md` §5.
- **Prior phase pickup:** `data/dflash-phase3-pickup-2026-05-18.md`.
