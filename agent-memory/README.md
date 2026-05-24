# agent-memory/

Version-controlled snapshot of the agent's auto-memory. This is the **shared knowledge layer** across multiple host machines that work on `yarn-agentic`.

The live, per-host directory at `~/.claude/projects/-home-llm-yarn-agentic/memory/` is the agent's own working memory. This directory in the repo is its **synced mirror**, so that:

1. Knowledge survives host changes (a new host clones the repo and gets the full history)
2. Multiple hosts working on the same project share their lessons learned
3. The full audit trail of project decisions, feedback rules, and references is version-controlled

See `PROTOCOL.md` for the sync rules.

## Layout

```
agent-memory/
├── README.md                  ← this file
├── PROTOCOL.md                ← sync protocol + conflict resolution
├── MEMORY.md                  ← index (one line per entry, hand-maintained)
├── project_*.md               ← per-project memories
├── feedback_*.md              ← user feedback that shapes agent behaviour
├── reference_*.md             ← pointers to external systems
└── user_*.md                  ← facts about the user
```

## Quick commands

```bash
# At session start: pull repo into live dir (sync down)
bash scripts/agent-memory-pull.sh

# At session end (or after meaningful new entries): push live to repo
bash scripts/agent-memory-push.sh
```

Both commands are documented in `PROTOCOL.md`.

## Snapshot provenance

This snapshot at first repo-commit:

| field | value |
|---|---|
| date | 2026-05-24 |
| committed from host | yarn.d07yx58.net |
| file count | 130 (MEMORY.md + 129 entries) |
| size | ~720 KB |
| MEMORY.md entries | 129 |
| audit | clean — 1:1 file ↔ index entry, 0 orphans, 3 SUPERSEDED entries preserved as audit records |
