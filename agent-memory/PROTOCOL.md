# agent-memory sync PROTOCOL

## Roles

- **Live** = `~/.claude/projects/-home-llm-yarn-agentic/memory/` (per-host; agent reads + writes here every session)
- **Repo** = `agent-memory/` in this git tree (committed; shared across hosts)
- **Direction**: bidirectional, but **never simultaneous on multiple hosts**

## Workflow (single host, simple case)

```
session start  →  pull   (repo → live)
session work   →  agent writes to live (normal CLAUDE.md memory tooling)
session end    →  push   (live → repo, commit + push)
```

Commands:

```bash
bash scripts/agent-memory-pull.sh   # at session start
bash scripts/agent-memory-push.sh   # at session end
```

## Conflict rules (file-level, since git doesn't merge agent memories naturally)

### Pull (`agent-memory-pull.sh`)

Sync direction: **repo → live**.

- Per file: `rsync --update` — if the file in the repo has a NEWER mtime than the file in live, repo overwrites live. If live is newer, live is kept.
- Files only in repo (not in live) are created in live.
- Files only in live (not in repo) are NOT deleted by pull. Live-only files survive (they will be picked up by the next push).
- `MEMORY.md`: same `rsync --update` semantics. **Caveat**: if both repo and live have edits since the last sync, last-mtime wins and the loser's MEMORY.md changes are lost. To prevent this, **always pull before writing**.

### Push (`agent-memory-push.sh`)

Sync direction: **live → repo**.

- The script auto-pulls latest repo first (`git pull --rebase`) before pushing.
- Per file: `rsync` from live to repo (no `--update`; live ALWAYS wins on push — that's the agent's authoritative state for THIS host's session).
- `--delete` is OFF: if repo has files not in live, they are KEPT in repo (so other hosts' memories aren't erased).
- After rsync: `git add agent-memory/ && git commit && git push`.

### What goes wrong + how to recover

| problem | symptom | fix |
|---|---|---|
| Two hosts edit the same file in overlapping sessions | one host's edit is lost in the push | rsync log shows which file was overwritten. Inspect git log on the file; cherry-pick or hand-merge. |
| One host forgets to pull before writing | local MEMORY.md diverges from repo; push overwrites repo's MEMORY.md changes | Pre-push, the push script runs `git pull --rebase`; if MEMORY.md was edited in the repo since last sync, rebase will surface a conflict. Resolve by hand-merging both edits. |
| MEMORY.md index drifts from actual files | entries reference missing files, or files exist without index entries | Run the audit: `bash scripts/agent-memory-audit.sh`. Reports orphan files + missing entries. |
| Sync corrupted (partial rsync) | live dir has half files updated | Rerun pull from a clean repo state. |

## Multi-host coordination (CRITICAL)

If multiple hosts are likely to write memories at the same time, the simplest discipline is **sequential sessions**:

1. Host A starts a session, pulls latest, works, pushes.
2. Host B starts AFTER A's push, pulls, works, pushes.
3. Host A starts again, pulls, ...

This keeps conflicts impossible. Realistically, for this project, 2-3 hosts may be active over different time windows; conflicts are rare and resolvable when they happen.

For genuinely concurrent multi-host work (say, two engineers each running their own agent against `yarn-agentic`), a per-host subdirectory layout would be the next iteration:

```
agent-memory/
├── _shared/                       ← cross-host curated memories
└── hosts/
    ├── yarn.d07yx58.net/         ← one host's full memory dir
    │   └── memory/...
    └── another.host.example/
        └── memory/...
```

That layout is **not yet implemented**. If it becomes needed, document the migration in this file before doing it.

## Append-only semantics

The agent's CLAUDE.md guidance for the memory system says: do not delete or rewrite old entries — add corrections via new entries that reference the old. This protocol upholds that:

- Push does NOT delete files in repo that are missing from live.
- Pull does NOT delete files in live that are missing from repo (`rsync` without `--delete`).
- The only way a file is removed is by **explicit user / agent action** (manual `rm` + manual git commit).
- If a memory becomes wrong, the convention is to add a new entry that supersedes it. The old entry is preserved as audit record. (Three SUPERSEDED entries from earlier sessions are examples of this — see MEMORY.md.)

## Bootstrap (new host first sync)

On a brand-new host with an empty live dir:

```bash
git clone --recurse-submodules <yarn-agentic remote>
cd yarn-agentic
mkdir -p ~/.claude/projects/-home-llm-yarn-agentic/
bash scripts/agent-memory-pull.sh    # populates live from repo
# Verify:
ls ~/.claude/projects/-home-llm-yarn-agentic/memory/MEMORY.md
grep -c '^- \[' ~/.claude/projects/-home-llm-yarn-agentic/memory/MEMORY.md
# expected ~129+ entries
```

The agent's next session will read live as usual; the project history is now visible.

## What does NOT go in this directory

- `~/.claude/CLAUDE.md` (user-global agent guidance) — belongs at the user-level path, not project-shared
- `~/.claude/RTK.md` (rtk reference) — also user-level
- Project-scope project memories like the repo root `MEMORY.md` — that's a DIFFERENT thing (project decisions log, append-only by humans / agents writing about the project, not the agent's per-incident memory)

## Audit + integrity

At any time:

```bash
bash scripts/agent-memory-audit.sh
```

Reports:
- Orphan files (exist on disk but no entry in MEMORY.md)
- Missing files (entry in MEMORY.md but no file on disk)
- File count vs entry count
- Most recently modified files
- Files containing `SUPERSEDED` / `OVERRIDDEN` (legitimate; should be preserved as audit records)

A clean snapshot (as of 2026-05-24) has: 130 files (MEMORY.md + 129 entries), 1:1 mapping, 0 orphans, 0 missing.
