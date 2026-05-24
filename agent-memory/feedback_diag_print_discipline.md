---
name: Diagnostic prints — gate at construction or don't add
description: Discipline for adding diag prints to perf-critical code so they don't leak into "default state" and pollute future sessions
type: feedback
originSessionId: e17bdbd7-e0a3-4ec7-9818-e8f46eed5283
---
When adding a diagnostic print to perf-critical code (mtp pipeline, server-context, hot loops), gate it at construction time with a per-call env knob that defaults OFF, AND keep the format to a single line by default. Verbose multi-line diagnostic blocks must be inside a separate verbose-only env tier or not added at all.

**Why:** During Phase 38 E debug I added several diagnostic prints (`[full2-fr]`, `[full2-disp-gate]`, `[full2-cmp]`, multi-line `[mtp-input-chk]` with persist+host first4 floats). Each grew during chasing. At session-close I had to systematically revert every one — and even after revert, a verbose form persisted in `[mtp-input-chk]` that needed a second pass. Repeated revert work is friction every closure pass; harder to spot once they ship.

**How to apply:**
- Before adding a diag, ask: is there an existing env knob that fits? If yes, extend it. If no, name a new env knob (e.g. `LLAMA_MTP_INPUT_CHECKSUM`) and gate at the top of the function via `static const bool _flag = (getenv(...) != nullptr);`.
- Default form: ONE LINE, structured key=value. Never multi-line.
- If verbose multi-line output is needed (e.g. dumping float arrays), gate behind a SEPARATE `_VERBOSE` env knob inside the same flag block.
- Before commit: grep added prints; verify each is env-gated; verify each defaults OFF.
- At session close: a quick `git diff | grep fprintf` audit catches leaked diag.
