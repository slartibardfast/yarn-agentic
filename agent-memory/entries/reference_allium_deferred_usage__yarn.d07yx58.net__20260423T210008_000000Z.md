---
name: Allium `deferred` is for cross-`.allium` references only
description: `deferred Name` declarations point to other `.allium` files where logic is fully specified. Using `deferred` for black-box math helpers or C-source pointers is a misuse that `allium check` warns on.
type: reference
originSessionId: 60b8f2a3-4018-43ac-ae61-4b83f88e6a1b
---
Per the Allium language reference: `deferred Name -- see: other.allium` references Allium logic that is fully specified in another spec file. It is **not** a declaration slot for:

- Black-box math helpers (`sum_of_squares`, `sqrt`, `scaled`, `dot_product`, `max`, `abs`). These are called inline in expressions with no declaration required — the reference spec calls them "black box functions" and says they "always use free-standing call syntax."
- C-source pointers (`-- see: some_file.c function_name`). C code is not an Allium spec.
- Alternative implementations of a rule (SIMD variants, GPU backends). Those are implementation detail and usually excluded from the spec via a scope comment, not declared.

`allium check` emits `allium.deferred.missingLocationHint` on any `deferred` without a proper `.allium` target. This is an info-level warning, not an error, so files with the misuse still parse; but a clean spec has zero of them.

**Idiomatic alternatives:**

- For black-box helpers: just use them inline (`let n = sqrt(norm_sq)`). No declaration.
- For cross-spec references: `deferred OtherModule.rule -- see: path/to/other.allium`.
- For implementation variants excluded from the spec: a scope comment at the top of the file (e.g. `-- Excludes: SIMD implementations — implementation detail`).
- For unresolved design questions: `open question "..."`.

**Reference context:** encountered in `yarn-agentic/turbo-kv-4b.allium` on 2026-04-23 — a prior spec had ~20 `deferred X -- black box:` declarations that all warned. The fix was to delete them; the rule bodies' inline calls continued to parse cleanly.
