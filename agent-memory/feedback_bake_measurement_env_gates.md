---
name: bake-measurement-env-gates
description: "Once a measurement-period env knob (LLAMA_*_ENABLE, LLAMA_*_DISPATCH, etc.) has been verified to fix a bug, remove the knob and bake the behavior. Leaving the knob around dilutes the codebase and forces re-discovery in later sessions."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

When an env-gated experimental code path is verified to fix a bug, remove
the env check and make the behavior unconditional in the same commit that
records the verification. Do not leave the knob behind as a "toggle".

**Why:** Two prior incidents on this repo:
1. `LLAMA_FATTN_PER_SLOT_KV_ENABLE` — used as a measurement scaffold during
   A.1'. Once verified, baked as always-on (the source comment now reads:
   "Always-on now; the prior LLAMA_FATTN_PER_SLOT_KV_ENABLE env-gate was
   a measurement-period scaffold, not a feature.")
2. `LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH` — found 2026-05-17 during NPC.4.
   It was the EXACT fix needed for the remaining NP=8 sub-ULP drift, but
   nobody remembered we already had the code. Discovery cost ~30k tokens
   of source-reading that could have been avoided if the previous session
   had baked it in instead of leaving an env knob.

**How to apply:** when verification of an env-gated path passes:
- Delete the env check from the dispatch logic.
- Delete the env var name from any docs / specs / READMEs.
- Update the comment block at the call site from "opt-in" to "baked, always-on".
- Note the prior knob in MEMORY.md so historical context is preserved.

The default workflow shape is **measure → verify → bake → record**, not
**measure → verify → toggle**.

Related: [[feedback_no_workarounds]], [[feedback_dont_rename_on_abandon]].
