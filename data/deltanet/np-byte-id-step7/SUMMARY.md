# Step 7 — empirical NP-cross byte-identity (P2 dispatch)

Date: 2026-05-15
Branch: production/2026-q2-next
Target: Qwen 3.6 27B production GGUF, Q4_0 + Hadamard KV
Harness: `tests/dflash-speculative/test-np-validity-vanilla.cpp`
Env: `LLAMA_FATTN_PER_SLOT_KV_ENABLE=1` (routes FA through pb=1 wmma_f16)

## Closure

slot-0 generated token sequence byte-identical across NP ∈ {2, 4, 8} at
n_gen=64. Same prompt input (p0.txt) for slot 0 in each NP run.

```
NP=2 slot-0 hash = -2490288469305313494
NP=4 slot-0 hash = -2490288469305313494
NP=8 slot-0 hash = -2490288469305313494

NP=2 == NP=4 → True
NP=4 == NP=8 → True
NP=2 == NP=8 → True
```

## Implication

The original PLAN.md problem ("vanilla decode non-deterministic at np>1 on
Qwen 3.6 27B hybrid arch") is closed at the slot-0 token level under the
P2 dispatch. By forcing `cols_per_block=8` + `parallel_blocks=1`, the
wmma_f16 kernel has shape-independent grid/template selection — the same
slot's K range produces the same output regardless of batch shape.

Per-slot validity (the 5 assertions) GREEN at all NP values. PPL band:
NP=2 [1.58, 1.65], NP=4 [1.32, 3.13], NP=8 [1.32, 9.70]. No NaN, no decode
failure, no glitch tokens.

## Files

- `np2.json` / `np4.json` / `np8.json` — full per-slot per-run state
  (generated_tokens, PPL, validity flags) from the harness.
