---
name: Use `git commit -- <pathspec>` when the index has pre-staged items
description: Avoid bundling pre-existing staged changes into a focused commit — pass pathspec explicitly instead of running `git add X && git commit`.
type: feedback
originSessionId: 60b8f2a3-4018-43ac-ae61-4b83f88e6a1b
---
Before running `git add X && git commit`, check `git status` for anything already in the index. If there are pre-staged items you don't want in this commit, use `git commit -- <pathspec>` (or `git commit <file>`) to restrict the commit to the paths you actually intend.

**Why:** on 2026-04-23 in `yarn-agentic` I ran `git add PHASE25.md && git commit` without noticing six `D  turbo-*.allium` deletions that were already staged from a prior session. The commit bundled them in, which violates CLAUDE.md §5 ("Every edit to PHASEx.md file triggers a git commit and git push. Do not batch these with other changes"). Had to `git reset --soft HEAD~1` and re-commit with pathspec. The user had to choose the split explicitly.

**How to apply:** whenever you're about to `git add X && git commit` in a repo with any kind of per-file commit rule (CLAUDE.md §5 in yarn-agentic, or anywhere the user is running a disciplined commit flow), first read `git status` and look for capital-letter staged entries (`M `, `D `, `A `) in the first column. If any are unrelated to your intent, use `git commit -- <file>` to restrict the commit. Also applies in reverse: if you just want your files included without touching what was pre-staged, pathspec is the safe move.
