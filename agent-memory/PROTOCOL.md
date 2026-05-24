# agent-memory PROTOCOL — event-sourced G-Set CRDT

## Design summary

Agent-memory is stored as an **append-only event log** under `agent-memory/entries/`. Each event is a single markdown file. Filenames encode provenance:

```
<slug>__<host>__<rfc3339-microsecond-ts>.md

  e.g.  project_t6_3_j_1m_ctx_ceiling__yarn.d07yx58.net__20260524T120512_000000Z.md
```

The double-underscore `__` is the field separator. All current slugs use single underscores only.

The event set is a **G-Set CRDT**: events are only ever added; never modified, never deleted. Filenames are unique by construction (host + microsecond timestamp), so concurrent writes from multiple hosts never collide on disk.

## Derived state

Two views are derived from the event set on demand:

| view | content | derivation |
|---|---|---|
| **Live `<slug>.md`** at `~/.claude/projects/-home-llm-yarn-agentic/memory/<slug>.md` | full content of the newest event for that slug | written by `agent-memory-pull.sh` |
| **`agent-memory/MEMORY.md`** (repo) | derived markdown index; links point to `entries/<event-filename>.md` | regenerated on every push |
| **Live `MEMORY.md`** (host-local) | derived markdown index; links point to `<slug>.md` (matching the agent's read pattern) | regenerated on every pull |

The repo's `agent-memory/MEMORY.md` is **never hand-edited** — edits are overwritten on the next `agent-memory-push.sh`. The same goes for the live `MEMORY.md`: it's derived on pull.

The **live `<slug>.md`** files are what the agent reads + writes during a session. From the agent's perspective, the layout looks like a flat directory (which is the existing CLAUDE.md memory system convention). The event-log machinery is invisible during a session.

## Concurrency properties (multi-agent / multi-host)

Two agents on different hosts (or on the same host) running **concurrent sessions**:

| event order | outcome |
|---|---|
| A writes a NEW slug; B writes a different NEW slug | Both events added on push. Both visible after pull. |
| A writes slug X; B writes slug Y (different) | Both events added. |
| A updates slug X; B updates slug X | Both events added (different timestamps, different host = different filename). Pull picks newest by timestamp; older event preserved in entries/ as audit record. NO DATA LOSS. |
| A and B write to same slug at same timestamp same host | Microsecond resolution; collision rare in practice. If it happens, second write fails to create the file (filesystem reports EEXIST); script reports + the user re-runs push. |

Compared to the previous linear protocol, this allows **genuinely concurrent multi-host sessions** without data loss.

## Workflow

### Single host (most common)

```bash
# At session start:
bash scripts/agent-memory-pull.sh

# Agent works as normal — reads + writes <slug>.md files in
# ~/.claude/projects/-home-llm-yarn-agentic/memory/

# At session end (or after meaningful writes):
bash scripts/agent-memory-push.sh
```

### Multiple hosts working concurrently

Each host follows the same pull → work → push cycle. The events accumulate in the event log. Pull surfaces other hosts' newest writes; push uploads this host's session work.

Recommended cadence: **pull at session start AND push at session end** on every host. Pulling mid-session is safe — it materializes the latest newest-per-slug from any host, overwriting live files. But agent in-memory state may not reflect the new content until the next read.

### Concurrent edits to the same slug

If host A and host B both update `project_X.md` during overlapping sessions:

1. Host A pushes first → event `project_X__hostA__ts1.md` added
2. Host B pushes second → `git pull --rebase` brings A's event in → no conflict (different filename) → B's event `project_X__hostB__ts2.md` added → MEMORY.md regenerated with B's event as newest (assuming ts2 > ts1)

A's update is preserved in the event log but no longer "current" (it's history). The agent can still find it by listing `entries/project_X__*`.

To **explicitly merge** two concurrent edits to the same slug (rather than let the newest win), the user would have to manually craft a new event combining both. This is rare and outside scope of automation.

## What gets created on push

`scripts/agent-memory-push.sh` does:

1. `git pull --rebase` — incorporate other hosts' new events. Conflicts in `entries/` are impossible because event filenames are uniquely keyed.
2. `python3 agent_memory_lib.py diff-live` — for each `<slug>.md` in live, check if content matches the newest event for that slug.
3. For each changed slug: write a NEW event `<slug>__$(uname -n)__$(date)Z.md` to `entries/`.
4. Regenerate `agent-memory/MEMORY.md` from the post-write event set.
5. `git add agent-memory/ && git commit && git push`.

If no live edits since last pull and no new events from other hosts, the push is a no-op (no commit).

## What gets materialized on pull

`scripts/agent-memory-pull.sh` does:

1. `git pull --rebase` — incorporate other hosts' events.
2. `python3 agent_memory_lib.py materialize-live` — for each slug in `entries/`, find the newest event by timestamp, and write its content to the corresponding `<slug>.md` in live. Skip writes if content matches (preserves mtime).
3. Regenerate live `MEMORY.md` (links to `<slug>.md`).

## Audit

```bash
bash scripts/agent-memory-audit.sh --repo    # event log
bash scripts/agent-memory-audit.sh --live    # derived live view
```

Reports:
- Event count, unique slug count
- Frontmatter validation (some older entries have malformed frontmatter; tracked but not blocking)
- Per-type breakdown (project / feedback / reference / user / unknown)
- Slugs with multiple events (history depth)

## Migration from flat layout (one-off)

`scripts/agent-memory-migrate.sh` was the one-off conversion of the previous flat layout to events. Each existing `<slug>.md` was moved to `entries/<slug>__<host>__<mtime>.md` with the file's last mtime as the initial event timestamp. Author host defaults to whichever host ran the migration (acknowledged loss of original write provenance, since it wasn't recorded under the flat scheme).

If a fresh host wants to rebuild from the event log:

```bash
git clone --recurse-submodules <repo>
cd yarn-agentic
bash scripts/agent-memory-pull.sh   # materializes 129 slug files into live
```

The migration script is not idempotent in a useful way and shouldn't be re-run. If you need to reset the event log, that's a destructive operation that requires explicit thought (and a fresh git commit).

## Frontmatter conventions

Two coexisting formats (the lib accepts both):

**Old format** (used by entries written before strict YAML discipline):

```yaml
---
name: Short title
description: One-line description
type: feedback
originSessionId: <uuid>
---
```

**New format** (preferred for new memories):

```yaml
---
name: kebab-case-slug
description: "Possibly multi-paragraph; quoted strings ok"
metadata:
  node_type: memory
  type: project
  originSessionId: <uuid>
---
```

When MEMORY.md is regenerated, `description` is the line shown next to each entry. Older entries with malformed/missing frontmatter still appear (with the description "(no description in frontmatter)") but aren't broken — they're just less informative in the index. Cleanup is opportunistic: when the agent revises one of these entries, the new event uses the new format.

## Operational guarantees + non-guarantees

**Guarantees:**
- G-Set: no data loss on concurrent writes (events are uniquely-named).
- Deterministic derivation: same event set → same MEMORY.md content (modulo the timestamp header).
- Append-only history: every memory write is preserved as a historical event, even if a later event supersedes it.
- File-system safe: filename schema is portable (no special characters beyond `_`, `-`, `.`, ASCII).

**Non-guarantees:**
- No transactionality across multiple slugs: if push is interrupted mid-loop, some events may be committed and others not. Re-running push from the affected host completes the rest (the diff would only show the remaining unwritten slugs).
- No automatic merge of concurrent same-slug edits: LWW for the derived view; both events kept in entries/. Manual merge is the user's responsibility.
- No deletion semantics: if a memory becomes wrong, the convention is to add a new event for that slug whose content explicitly says "this entry is SUPERSEDED by [other slug]". The audit reports SUPERSEDED markers but doesn't filter them from MEMORY.md.

## Recovery scenarios

| problem | fix |
|---|---|
| Live dir corrupted / partially deleted | `bash scripts/agent-memory-pull.sh` rebuilds from event log |
| Event log corrupted in a specific file | `git checkout <file>` restores; if it's a true rewrite, hand-fix via a new event |
| MEMORY.md drift | `python3 scripts/agent_memory_lib.py derive <entries-dir> <out-path> events` |
| `git pull --rebase` conflicts | Only possible on MEMORY.md (regenerable) — discard local, regenerate, commit. On entry files, impossible (unique filenames). |
| Lost write provenance after migration | Acknowledged. The host tag for pre-migration entries is whichever host ran `agent-memory-migrate.sh`. The original author isn't recovered. New writes carry correct provenance. |
