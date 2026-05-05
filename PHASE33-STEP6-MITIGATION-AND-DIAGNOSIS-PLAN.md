# Phase 33 Step 6 — multi-slot crash mitigation + diagnosis plan

## Mitigation (shipped)

`profiles/qwen36-27b-x1.sh` (NEW) — same flags as the x4
profile minus `--parallel 4` → `--parallel 1`. Slot 0 alone now
gets the full 1 M-token `--ctx-size`, strictly more than the
per-slot 256 K cap of the x4 layout.

`profiles/active.sh` symlink repointed:

```
qwen36-27b-x4.sh  →  qwen36-27b-x1.sh
```

Server restarted, `slots_idle=1`, `slots_processing=0`. nginx 502
rate from the snoop run should drop to ~0% — single-slot path
has been live-tested at 35 t/s for the full Phase 32 run.

## Loss accepted

- No multi-session concurrency. Two OpenCode sessions will now
  serialise (request 2 waits for request 1 to finish).
- Sub-agent parallelism feature deferred until concat fix lands.

## Why this is reversible

When the concat assertion is fixed, flip the symlink back to
`qwen36-27b-x4.sh` and restart. No rebuild, no other state.

## Diagnosis next step (read-only probe)

Top suspect from the concat dive:
`src/llama-delta-net.cpp:689` (Phase 32 per-block dispatch).

Why it fits:

- Guarded by `!all_same_seq` — the exact condition that
  distinguishes "slot 0 alone" (clean) from "slot 1 enters"
  (crash).
- `inp_out_ids` is batch-global but applied per-block; the
  first slot-1 block (fresh prompt, no committed out_ids) takes
  the branch at `delta-net.cpp:551-557` differently from
  sibling blocks, producing `out_blk` tensors with divergent
  dtypes. Concat at `:689` then trips the CUDA-side assert.
- Phase 32 multi-slot bench used uniform prompts → same seq id
  per step → `all_same_seq == true` → block split never runs →
  bench passes. Real OpenCode traffic with two distinct
  sessions arriving in the same scheduler step → block split
  runs → crash.

Probe (does not fix; localises the site):

1. Add a non-fatal log in `ggml/src/ggml-cuda/concat.cu` at
   line 200, right before the existing assert:

   ```c
   if (!(src0->type == src1->type && src0->type == dst->type)) {
       fprintf(stderr,
           "[concat-mismatch] dst='%s' op_dim=%d "
           "src0='%s' t=%d src1='%s' t=%d dst_t=%d\n",
           dst->name, ((int32_t*)dst->op_params)[0],
           src0->name, src0->type, src1->name, src1->type, dst->type);
       fflush(stderr);
   }
   ```

   Leave the assert in place — we want the same crash, just
   with a stderr line preceding it.

2. Rebuild ik_llama.cpp.

3. Flip active.sh back to qwen36-27b-x4.sh temporarily; start
   the snoop recorder; drive heterogeneous 2-slot traffic
   (one short prompt, one long prompt). The crash will
   reproduce and the stderr line names the offending tensor.

4. Cross-reference the tensor name (Phase 32 cb-tags) against:
   - `:689` — output of attn_output cb, joined across blocks
   - `:407` — new_state_flat, recurrent-state writeback
   - `:233-234` — qkv_mixed
   The `cb(...)` tag in the name resolves the site uniquely.

5. Flip active.sh back to qwen36-27b-x1.sh post-probe (under
   30 s on this hardware); we know it crashes, so don't run
   real OpenCode traffic against the probe build.

## Fix plan (after the probe localises the site)

Conditional on the probe confirming `:689`. Two candidate
shapes:

- **A — make `inp_out_ids` per-block.** Slice `inp_out_ids` to
  the block range before passing into
  `build_layer_attn_linear_core`. All blocks then take the
  same branch and produce same-dtype outputs. (This was the
  prior strand's Step 1 candidate fix — already considered for
  the multi-token-block bug.)
- **B — force a unified dtype on `out_blk`.** Insert a
  `ggml_cast(out_blk, GGML_TYPE_F32)` before the per-block
  concat. Cheaper to land but adds a needless cast on the
  fast path; should only be used if A is non-trivial.

A is the cleaner fix; defer B to a fallback.

## Open back-burner: invalidation cause (Step 5)

The 193 invalidations the snoop captured (94% mid-prefix) are
still real and unrelated to the crash. The Layer-2 prompt
diffs at `~/snoop-runs/snoop-20260505T111847/prompts/` haven't
been analysed yet. That's a separate investigation, not
blocked by the concat fix; can run in parallel.

## Files

| Path | Status |
|------|--------|
| `profiles/qwen36-27b-x1.sh` | NEW — single-slot mitigation |
| `profiles/active.sh` | symlink repointed |
| `ggml/src/ggml-cuda/concat.cu` | unchanged (probe pending) |
| `src/llama-delta-net.cpp` | unchanged (suspect site) |
