---
name: yarn-agentic repo layout
description: Canonical structure of /home/llm/yarn-agentic after 2026-04-11 consolidation — flat phase docs, top-level submodules, mdBook → GitHub Pages.
type: project
originSessionId: a5a02c13-3d1d-4998-badf-36c74b87c680
---
`/home/llm/yarn-agentic/` is the authoritative repo (published as github.com/slartibardfast/yarn-agentic, public) consolidated on 2026-04-11 from scattered wrapper repos.

**Layout (flat):**
- `CLAUDE.md`, `PLAN.md`, `MEMORY.md`, `SUMMARY.md`, `PHASE0.md..PHASE22.md` at repo root
- `BENCHMARKS.md`, `BENCHMARK_EXPECTATIONS.md`, `DISPATCH_COMPARISON.md`, `MTP_INSTRUCTIONS.md`, `benchmark.sh`
- `benchmarks/` — test artefacts
- `book.toml`, `theme/custom.css`, `docs/index.html` — mdBook config
- `.github/workflows/pages.yml` — mdBook → GitHub Pages CI (working, verified 19s build)
- `llama.cpp/` — submodule → `slartibardfast/llama.cpp`, upstream remote = `ggml-org/llama.cpp`
- `ik_llama.cpp/` — submodule → `slartibardfast/ik_llama.cpp`, upstream remote = `ikawrakow/ik_llama.cpp`

**Why:** The previous mess had PLAN/PHASE/MEMORY/mdBook duplicated inside wrapper fork repos (`radv_llama.cpp`, `radv_ik_llama.cpp`) which made upstream tracking noisy. Consolidation hoisted docs to the top level and retired the wrappers. Both wrappers are still on GitHub as historical archives.

**How to apply:**
- Plan, phase, and memory docs live at repo root. Do not create planning docs inside submodules — §5 of the repo CLAUDE.md is explicit about this.
- When adding a second conceptual branch of work, create `phases/<branch>/` as a subdirectory; the current work stays flat.
- To update a submodule from upstream: `cd <submodule> && git fetch upstream && git rebase upstream/master && git push origin`. Never force-push upstream.
- mdBook site auto-publishes on push to `main`. Site URL: https://slartibardfast.github.io/yarn-agentic/
- `.claude/settings.local.json` is ignored globally (user gitignore); local allow-list for test-backend-ops commands. Paths in it still reference the pre-consolidation `/home/llm/radv_llama.cpp/...` layout — may need refreshing when Claude Code asks to re-approve commands.
