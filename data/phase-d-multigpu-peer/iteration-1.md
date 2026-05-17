# CY.F.19 Investigation — iteration 1

## Findings

1. **NP=1 is byte-identical across 3 runs** (SHA `3683a5f83ea2f90b...`). Single-slot decode is deterministic. Rules out general GPU/CUDA timing.

2. **Race scales with prefill size**:
   - Short prompt (~15 tok): 3/5 PASS
   - Long prompt (~200 tok): 0/5 PASS

3. **Three-isocluster signature** (matches prior CY.F.14 finding):
   - {NP=1}: stable
   - {NP=2, NP=4}: drift to "fundamentally statistical pattern matchers"
   - {NP=8}: subtler drift ("inaccurate or misleading outputs")

4. **Server has a context-checkpoint feature** (`server-context.cpp:3567 create_checkpoint`, 3470 `create_checkpoint_at_interval`) called from:
   - Line 4287 (in some path)
   - Line 4687 / 4689 (cycle close + interval check)

   Both NP=1 and NP=2 logs show checkpoint creation at pos 204 (post-prefill) and pos 272 (mid-decode). NP=1 still deterministic with checkpoints, so checkpoints themselves aren't deterministic-violating UNLESS multi-slot.

5. **Slot 0 and slot 1 prefill in parallel** in NP=2 (same timestamp, same thread tid). This suggests they're in the same ubatch.

## Probes for iteration 2

- **P1**: Run NP=2 with `--no-context-shift` and add `--ctx-shift-threshold 999999` or equivalent to suppress checkpoint creation. Check if 5/5.
- **P2**: Force `cache_prompt=true` so the same prompt is shared across slots (KV cache reuse). Probably won't help because slots ARE getting separate prefills.
- **P3**: Audit `server-context.cpp` line 4687/4689 — what triggers checkpoint creation? Is it deterministic per-prompt?
- **P4**: Disable cont-batching: `--no-cont-batching` flag if available.
- **P5**: Capture concurrent NP=2 slot 0 and a serial-decode "fake NP=2" (run slot 0 alone, then slot 1 alone, on a 2-slot server) — see if serial slot 0 matches NP=1 baseline.
