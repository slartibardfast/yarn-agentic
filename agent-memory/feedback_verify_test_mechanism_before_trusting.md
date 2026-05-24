---
name: Verify the test mechanism works before trusting its result, especially before irreversible cleanups
description: A passing A/B test isn't proof unless you've also verified that the test's two legs are actually different. Before deletion based on "no measurable effect" results, re-run the test mechanism with explicit `unset` / sentinel values to confirm the variable actually exercises both branches. Strengthens "test-first discipline on negative claims".
type: feedback
originSessionId: 84c2467d-f2f1-4f7b-8d55-149618ff875c
---
When a measurement says "X is removable / harmless / equivalent to Y",
the burden of proof is higher than for a positive measurement, because
acting on the negative is irreversible (delete the code, change the
default). Before removing or pivoting on such a result:

1. Verify the test's two legs actually differ. The "they look the
   same" reading might mean they ARE the same path. Common gotchas:
   - Env-gate `getenv("VAR") != nullptr` treats `VAR=0` as set, not
     unset. The bash spelling `VAR=0 cmd` runs both legs of an "A/B"
     in the same hook-ON path. Use `unset VAR` (or `env -u VAR`) to
     actually disable.
   - Build-time `#define`s baked into a stale `.o` survive across
     "rebuilds" if cmake doesn't see the trigger change.
   - Profile/runtime flags that the harness silently overrides.

2. Re-run the test, even if you trust the prior author. Authors miss
   gotchas all the time — see PHASE45 D8.4's hook A/B that was
   structurally a no-op for nine months before D9.9a caught it.

3. If irreversible (deleting code, removing a flag), do the
   measurement YOURSELF immediately before the destructive action.
   "Trust but verify" only works if you actually verify.

**Why:** PHASE45 D9.9a deleted the INLINE_KV hook based on D8.4's
"hook A/B within 0.5%". The A/B was broken (`LLAMA_MTP_INLINE_KV=0`
still triggered the gate). Hook actually saves a per-accept
UPDATE_ACCEPTED decode — load-bearing post-D9.5. Deletion measured
+17.5% (below +19% floor). Reverted. Cost: ~30 minutes of
implementation + bench cycle + revert. Cheap because reversible —
but if I'd been deleting bigger surface area or pushing to main, it
would have been much more expensive.

**How to apply:** Add to the pre-flight checklist for any code
deletion or default flip that's justified by "this measurement shows
no effect": confirm the measurement actually exercised what it
claimed to exercise. Especially for env-gated flags. Especially
when the effect is supposed to be small (because small effects can
hide structural test bugs).

This refines the existing rule
`feedback_test_first_negative_claims.md` ("no fix exists requires an
actual test of each untested candidate"). The new sub-rule:
**verify the test itself, not just that there is one.**
