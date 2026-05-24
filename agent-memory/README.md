# agent-memory/

Event-sourced agent auto-memory with CRDT semantics. Multi-host safe.

## What this is

The agent's working memory at `~/.claude/projects/-home-llm-yarn-agentic/memory/` is a **per-host current-view** of a **shared event log** stored here under `agent-memory/entries/`. Each memory write becomes a new event file with provenance encoded in the filename (slug + host + timestamp).

This directory carries the shared history across hosts. Pull at session start to materialize the latest current-view into your live directory; push at session end to commit your session's edits as new events.

## Layout

```
agent-memory/
├── README.md                  ← this file
├── PROTOCOL.md                ← full CRDT design + workflow + recovery
├── MEMORY.md                  ← DERIVED index (do not hand-edit)
└── entries/                   ← G-Set of memory events
    └── <slug>__<host>__<ts>.md
```

## Quick start

```bash
# At session start — refresh live from the shared event log:
bash scripts/agent-memory-pull.sh

# At session end — turn live edits into new events, commit + push:
bash scripts/agent-memory-push.sh

# Audit:
bash scripts/agent-memory-audit.sh --repo   # event log
bash scripts/agent-memory-audit.sh --live   # local materialization
```

## CRDT properties

- **G-Set on event identities**: events are append-only; never modified or deleted. Filenames carry host + microsecond timestamp = always unique.
- **LWW for derived view**: when multiple events exist for a slug, the newest wins for the current view. Older events stay in `entries/` as audit records.
- **Deterministic derivation**: `MEMORY.md` is regenerated from the event set on every push; no merge conflicts on the index.
- **Concurrent multi-host writes safe**: two hosts writing the same slug at overlapping times produce two distinct event files, both preserved. No data loss.

See `PROTOCOL.md` for the full design, including workflow details, recovery scenarios, and migration history.

## Snapshot

| field | value |
|---|---|
| schema | event-sourced G-Set CRDT |
| migration | from flat layout 2026-05-24 |
| event count at first commit | 129 |
| unique slugs | 129 |
| audit at first commit | clean — 124 of 129 have valid frontmatter (5 older entries with malformed YAML; non-blocking) |
