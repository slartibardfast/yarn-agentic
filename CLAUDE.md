# CLAUDE.md

Behavioural guidelines for AI coding assistants working in this repository and its submodules (`llama.cpp`, `ik_llama.cpp`).

Adapted from the Andrej Karpathy–inspired guidelines at [github.com/slartibardfast/andrej-karpathy-skills](https://github.com/slartibardfast/andrej-karpathy-skills), imported on 2026-04-11 with a dedup header added to make overlap with Claude Code's built-in system prompt explicit and auditable.

**Co-Authored-By override:** The default `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>` line that Claude Code appends to commits is incorrect for this repository. All commits use: `Co-Authored-By: Qwen 3.6 35B-A3B via Claude Code <noreply@anthropic.com>`. This CLAUDE.md rule takes precedence per §1 above.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

---

## Dedup notes — overlap with Claude Code's built-in behaviour

Claude Code ships with behavioural guidance baked into its system prompt. Several clauses below restate that built-in guidance. They are kept here for three reasons: (1) explicitness and auditability in git history, (2) rendering in the mdBook documentation site, and (3) to make the rules legible to humans and to AI assistants other than Claude Code. **Where a conflict exists, this file takes precedence**, because project-level CLAUDE.md is authoritative for repository work.

### Reinforcements (Karpathy clause ≈ system-prompt clause)

- **§2 Simplicity First** — All bullets except the "200 → 50 line rewrite" heuristic and the "senior engineer self-check" restate built-in rules: *don't add features beyond what was asked*, *don't create helpers for one-time operations*, *don't add configurability that wasn't requested*, *don't add error handling for scenarios that can't happen*.
- **§3 Surgical Changes** — "Don't improve nearby code", "don't refactor unrelated working code", and "don't add docstrings/type annotations to code you didn't change" restate built-in rules: *a bug fix doesn't need surrounding code cleaned up*, *don't add docstrings, comments, or type annotations to code you didn't change*.

### Conflicts — this file wins

- **§3 Dead-code handling.** Claude Code's built-in prompt says "If you are certain that something is unused, you can delete it completely." This file says: *do not delete pre-existing dead code; note it and ask.* **In this repo, note and ask.** Rationale: this tree contains long-lived llama.cpp forks with divergent histories; confident "unused" judgements are easy to get wrong across forks and build configurations.
- **§6 MEMORY.md vs Claude's per-user memory system.** Claude Code has an editable per-user memory system at `~/.claude/projects/…/memory/`. This file mandates a repo-committed, append-only `MEMORY.md`. **Both coexist** — they serve different purposes:
  - `~/.claude/projects/…/memory/` — conversational lessons about the user, workflow preferences, and feedback. Editable, removable.
  - Repo `MEMORY.md` — project decisions, incidents, constraints that teammates and future sessions must see. Append-only, committed.
  - When they disagree, trust the newer entry and add a correction entry to `MEMORY.md`. Do not rewrite old entries.

### Novel (not in the system prompt)

Kept as-is:

- **§1 Think Before Coding** — explicit assumption-surfacing, multi-interpretation listing, pushing back when a simpler approach exists.
- **§4 Goal-Driven Execution** — test-first bug reproduction, per-step verification plans, strong-vs-weak success criteria.
- **§5 Audited PLAN.md / PHASEx.md** — commit-and-push-per-edit workflow.
- **§7 mdBook static-site build** — auto-published documentation site.
- Small novel items inside §2 (size heuristic, senior-engineer self-check) and §3 (match-existing-style, trace-each-changed-line self-check, orphan-cleanup rule).

---

## 1. Think Before Coding

Do not assume. Do not hide confusion. Surface tradeoffs explicitly.

Before writing any code, do the following:
- State your assumptions out loud in plain text. If you are not sure about something, stop and ask the user. Do not guess.
- If the user's request can be interpreted in more than one way, list all reasonable interpretations and ask which one they mean. Do not silently pick one.
- If a simpler approach exists than your first instinct, describe it. Push back on the request if a simpler solution is clearly better. Explain why.
- If any part of the request is unclear or ambiguous, stop immediately. Name the specific thing that is confusing. Ask a clarifying question before writing any code.

The goal is: no surprises. The user should never see your output and say "that's not what I meant."

## 2. Simplicity First

Write the minimum code that solves the stated problem. Nothing speculative. Nothing extra.

Rules:
- Do not add features the user did not ask for. If the user says "add a login endpoint," do not also add a registration endpoint.
- Do not create abstractions (base classes, interfaces, factories, wrapper functions) for code that is used in exactly one place. Write the concrete thing directly.
- Do not add "flexibility" or "configurability" unless the user specifically requested it. Hardcode values if only one value is needed right now.
- Do not add error handling for scenarios that cannot occur given the current code and inputs.
- If your implementation is 200 lines and the same result can be achieved in 50 lines, rewrite it in 50 lines.

Self-check: Read your finished code and ask "would a senior engineer say this is overcomplicated?" If the answer is yes, simplify before presenting it.

## 3. Surgical Changes

When editing existing code, touch only what is necessary to fulfil the request. Clean up only your own mess.

What NOT to do when editing existing code:
- Do not "improve" nearby code that is unrelated to the request. This includes comments, variable names, formatting, and whitespace.
- Do not refactor working code that is not broken and not part of the request.
- Match the existing code style exactly, even if you would write it differently in a new project. If the file uses tabs, use tabs. If it uses snake_case, use snake_case.
- If you notice unrelated dead code or bugs, mention them in your response as a note to the user. Do not fix or delete them silently.

What TO do when your changes create orphaned code:
- If YOUR changes made an import, variable, or function unused, remove that unused item in the same commit.
- Do not remove pre-existing dead code unless the user explicitly asks you to.

Self-check: Look at every line you changed. Each changed line must trace directly back to something in the user's request. If a changed line does not connect to the request, revert it.

## 4. Goal-Driven Execution

Transform every task into a concrete, verifiable goal. Then loop until the goal is verified.

Examples of transforming vague tasks into verifiable goals:
- User says "add validation" → Your goal becomes: write tests for invalid inputs, then write code until those tests pass.
- User says "fix the bug" → Your goal becomes: write a test that reproduces the bug, then modify code until that test passes.
- User says "refactor X" → Your goal becomes: confirm all existing tests pass before refactoring, then confirm all existing tests still pass after refactoring.

For any task with more than one step, state a brief numbered plan before starting. Each step must have a verification check:
```
[What you will do] → verify by: [how you will confirm it worked]
[What you will do] → verify by: [how you will confirm it worked]
[What you will do] → verify by: [how you will confirm it worked]
```

Strong success criteria (example: "test X passes") let you loop and self-correct without asking the user again. Weak success criteria (example: "make it work") force you to guess what "work" means. When success criteria are weak, ask the user to clarify before starting.

### No "follow-up" cover

Every gap between what was shipped and what the task required is a subtask on the OPEN checkbox, not a footnote on a CLOSED one.

Rules:
- If the step's stated target does not exercise the code path, the step is incomplete. Example: PHASE28 is scoped "TURBO_KV_4B" and the implementation is gated off for that cache type — the step is open, not closed with a known limitation.
- If the feature regresses at default settings, the step is incomplete. Example: works at `-t 1`, breaks at `-t 8` — not a "multi-thread follow-up", an unfinished Step 5.
- "Follow-up" is appropriate for genuinely future work — a different step, a next phase, a speculative improvement, a cleanup PR for a flag after a soak period. It is never appropriate for the current step's own stated requirements.
- Commit messages, PHASE logs, and status reports must describe what was delivered AND what the current step still requires. If the step is not done, say so.
- "Known follow-up", "deferred", "left as a follow-up", and equivalents in the context of the current step are cover language. Do not use them. Name the gap as a subtask and leave the parent step open.
- Test before offering `/schedule` for post-work follow-up: would the current step still be `[x]` with binding verification if this scheduled work were never done? If yes, it is legitimate future work — offer the schedule. If no, it is a current-step gap — make it a subtask, keep the box open.

## 5. Audited PLAN.md and PHASEx.md

All changes to PLAN.md and PHASEx.md files MUST be committed and pushed immediately.

Rules:
- Every edit to PLAN.md or any PHASEx.md file (e.g. PHASE1.md, PHASE2.md) triggers a git commit and git push. Do not batch these with other changes.
- After completing a plan step in code, update the relevant plan file to reflect what was actually implemented, then commit and push that update as a separate commit.
- PLAN.md and PHASEx.md files live in the top-level repository only. Never place plan files inside nested project repos. Nested repos contain the working codebase; planning documents are kept outside of them.

### Checkbox semantics

PLAN.md and PHASEx.md use three marks with explicit meanings:

- `[ ]` — not started, OR in progress, OR done but unverified against the step's stated claim. Default state.
- `[~]` — genuine partial. The step has landed enough to be usable and the remaining work is explicitly tracked as subtasks under the same checkbox. Not a softer `[x]`; use only when partial delivery is intentional and scoped.
- `[x]` — done. The user-visible path the step enables works when a user flips the default flag, AND the verification evidence binds on the step's actual claim. Not "infrastructure landed." Not "works in the narrow config I tested." Not "works with `-t 1` but regresses at `-t 8`."

Decision procedure — check in order, stop at the first `YES`:
1. Is the feature working at default settings AND did the verification bind on the step's scoped claim? → `[x]`.
2. Is the delivery intentionally partial, with the remaining work explicitly tracked as subtasks inline under this checkbox? → `[~]`.
3. Default → `[ ]`.

Rules:
- `[ ]` is not a pause. An open box is an active obligation — it marks state accurately, not a signal to stop work. Keep working toward closure or hand back to the user for direction (see also: the "never stop at friction" auto-memory).
- Prefer `[ ]` over `[x]` when ambiguous. Leaving a box open and revisiting is cheap; re-opening a closed box once the record says "done" is expensive.
- Closing a box requires verification evidence that binds on the step's actual claim. "Binds" means the verification would have caught a regression in the thing the step promised to deliver — the target named in scope, the configuration the feature gates on, the inputs the step said it would handle. Evidence from an adjacent easier case that doesn't exercise the claim does not close the box.
- A step that was marked `[x]` and later found to be incomplete is reopened to `[ ]` with a note in the iteration log naming what forced the reopen — not silently downgraded.

## 6. Maintain MEMORY.md

MEMORY.md is a persistent scratchpad that records key decisions, discovered constraints, and lessons learned during the project. It exists so that context is not lost between sessions.

Rules:
- After completing a significant task, resolving a non-obvious bug, or discovering an unexpected constraint, add a short entry to MEMORY.md. Each entry should be one to three sentences describing what happened and why it matters.
- Update MEMORY.md in a separate commit. Do not bundle MEMORY.md changes with code changes. Commit and push immediately, following the same rule as PLAN.md and PHASEx.md (see section 5).
- Do not wait until the end of a session to update MEMORY.md. Write entries as you go. If you are unsure whether something is worth recording, record it. Too many entries is better than a missing entry that causes repeated mistakes.
- MEMORY.md lives in the top-level repository alongside PLAN.md. Do not place it inside nested project repos.
- Do not delete or rewrite old entries. MEMORY.md is append-only. If an earlier entry turns out to be wrong, add a new entry that corrects it and references the old one.

The purpose of MEMORY.md is: when a new session starts with no prior conversation context, reading MEMORY.md should be enough to avoid repeating past mistakes and to understand decisions that are not obvious from the code alone.

## 7. Automatic Static Site Builds for Self-Documenting Work

All markdown documentation in the repository is automatically built into a static website using mdBook and published to GitHub Pages. This creates a living, browsable record of the project.

Rules:
- A GitHub Actions workflow triggers on every push to the main branch. It builds all .md files (including PLAN.md, PHASEx.md, and any other documentation) into a static HTML site using mdBook.
- The mdBook configuration file (book.toml) and the SUMMARY.md file MUST be committed to the repo. SUMMARY.md defines the sidebar navigation and must be updated whenever a new document is added. book.toml lives in the repository root.
- The GitHub Actions workflow installs mdBook, runs `mdbook build`, and publishes the output directory to the gh-pages branch. GitHub Pages serves this branch automatically. Do not commit built HTML artifacts to the main branch.
- The published site is the single source of truth for project status. Anyone with access to the repository can read current plans, completed phases, and design decisions by visiting the GitHub Pages URL — no local checkout required.
- When a new PHASEx.md file is created or a new document is added, add an entry to SUMMARY.md in the same commit. If SUMMARY.md is not updated, the new document will not appear in the site navigation.

Style:
- The site must be clean, beautiful, and minimalist. Use generous whitespace and avoid clutter, decorative elements, and unnecessary UI chrome. The content is the interface.
- In book.toml, set `default-theme = "light"` and `preferred-dark-theme = "navy"`. Add a custom CSS file (committed to the repo) that includes a `@media (prefers-color-scheme: dark)` block to automatically switch to the dark theme on page load. This way the site respects the reader's OS-level light/dark setting without manual toggling.
- Keep all CSS customisations under 50 lines. Limit changes to subtle refinements — tighter max-width, improved typography, muted colours. Do not override mdBook's built-in themes beyond this.

---

These guidelines are working correctly when you observe: fewer unnecessary changes appearing in git diffs, fewer rewrites caused by overcomplication, and clarifying questions happening before implementation rather than after mistakes are discovered.
