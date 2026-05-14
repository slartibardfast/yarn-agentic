# NP-validity binding — production target with new FA op

Date: 2026-05-14
Branch: production/2026-q2-next
Binary: `test-np-validity-vanilla` (post Hadamard cparams patch)
Config: Q4_0 KV + Hadamard rotation + `cparams.flash_attn = true`, n_gen=64,
        production target GGUF.

The new `GGML_OP_FLASH_ATTN_EXT_PER_SLOT_KV` op dispatches at Qwen 3.5/3.6
production shape (Dq=Dv=256, gqa=6, no sinks). 4-warp Approach C decode
pack + single-pass (no split-K; split-K deferred to follow-up).

## Per-slot validity results (each slot must pass all 5 asserts)

### NP=2

| slot | term | decode | no_nan | in_vocab | ppl |
|---|---|---|---|---|---:|
| 0 | ✓ | ✓ | ✓ | 64/64 | 4.12 |
| 1 | ✓ | ✓ | ✓ | 64/64 | 3.94 |

**2/2 PASS**

### NP=4

| slot | term | decode | no_nan | in_vocab | ppl |
|---|---|---|---|---|---:|
| 0 | ✓ | ✓ | ✓ | 64/64 | 5.99 |
| 1 | ✓ | ✓ | ✓ | 64/64 | 10.39 |
| 2 | ✓ | ✓ | ✓ | 64/64 | 8.50 |
| 3 | ✓ | ✓ | ✓ | 64/64 | 5.99 |

**4/4 PASS**

### NP=8

| slot | term | decode | no_nan | in_vocab | ppl |
|---|---|---|---|---|---:|
| 0 | ✓ | ✓ | ✓ | 64/64 | 5.02 |
| 1 | ✓ | ✓ | ✓ | 64/64 | 6.11 |
| 2 | ✓ | ✓ | ✓ | 64/64 | 16.86 |
| 3 | ✓ | ✓ | ✓ | 64/64 | 5.75 |
| 4 | ✓ | ✓ | ✓ | 64/64 | 6.65 |
| 5 | ✓ | ✓ | ✓ | 64/64 | 3.98 |
| 6 | ✓ | ✓ | ✓ | 64/64 | 4.49 |
| 7 | ✓ | ✓ | ✓ | 64/64 | 19.87 |

**8/8 PASS**

## What this shows

- New FA op runs without crash/abort at np ∈ {2, 4, 8} on production model.
- Output is coherent per slot — PPL within the [1, 50] production-coherence
  band; ≥95% in-vocab; no NaN; no decode failures.
- Different per-slot PPL across NP values reflects different prompts assigned
  to each slot (prompts p0..pN-1 picked from prompt_dir); not a determinism
  signal.

## What this DOESN'T show (future work)

- **Byte-identity across NP**: slot 0's tokens at NP=2 vs NP=4 vs NP=8 are
  NOT compared. The original DeltaNet bug (NP=1≡NP=2 byte-identical, NP=4/8
  drift) needs a separate harness that drives the SAME prompt across all NPs
  and compares slot-0 outputs token-by-token. The unit test (Stage 2.2b
  scenario C at NP={1,2,4,8}) DID prove kernel-level batch invariance with
  the same input — that closure stands. End-to-end byte-identity at
  production shape is the follow-up gate.

- **Stage 3 closure**: the per-slot validity here is the PLAN.md Stage 2.6
  binding. Stage 3 (NP=4 ↔ NP=8 byte-identity) was the secondary determinism
  gate. End-to-end production byte-id needs a token-comparison harness.

## Data files

- `np2.json` — per-slot results, NP=2
- `np4.json` — per-slot results, NP=4
- `np8.json` — per-slot results, NP=8
