---
name: Dream-flow memory consolidation procedure
description: How to run the periodic memory-consolidation pass on this project's private auto-memory — when to trigger, the four-phase pattern, what to preserve vs distil, index regeneration
type: feedback
originSessionId: 9b52ee2b-d969-442c-83ba-98396525f6fa
---
The user calls this the "dream flow" (after the `autoDreamEnabled` Claude Code setting, which is a related concept). It's a periodic memory-consolidation pass on the private auto-memory at `/home/llm/.claude/projects/-home-llm-yarn-agentic/memory/`. First run was 2026-05-11; the user has indicated this will be run again.

## When to run

Trigger conditions, any of which is sufficient:
- Total entry count in the memory dir exceeds ~80, OR
- `MEMORY.md` index approaches the auto-memory truncation limit (~200 lines), OR
- A major workstream just terminated (success or abandonment) and produced overlap with prior step-snapshots, OR
- The user explicitly asks to "consolidate memories" / "run the dream flow."

## The four-phase pattern

Each phase does the same thing: identify a cluster of overlapping entries → write one consolidated archival entry that preserves the durable knowledge → delete the constituent files → regenerate the index. Work in clusters, not entry-by-entry.

### Phase 1 — Project lineage consolidation

Project entries that document step-by-step progress through a workstream are the primary target. When the workstream has reached a terminal entry (shipped, abandoned, deferred), the intermediate state-snapshots are superseded.

Example clusters: MTP-IR step snapshots from initial port through phases; TURBO/HARP_2B research lineage; any "Phase N step M findings" entries once Phase N+ has closed.

Keep the terminal entry intact. Distill the intermediate entries into a single `*_history_archived.md` or `*_abandoned.md` entry that:
- Records the lineage at a glance (numbered chronological steps).
- Lists the durable knowledge produced (rules, ceilings, hardware-specific facts).
- Cites preserved artifacts (branches, tags, test fixtures, SHAs).
- References the terminal entries that remain authoritative.

### Phase 2 — Redundant feedback rules

Feedback entries that articulate the same principle from different angles can be merged into one entry with multiple sections labelled as facets. Example: three test-first rules merged into `feedback_test_first_discipline.md` with three sub-sections.

Do not merge feedback entries that are genuinely distinct rules even if topically adjacent. The bar is: "Would deleting one of these and keeping the other lose information a future session needs?"

### Phase 3 — Index regeneration

After deletions, the index has broken pointers. Regenerate it from the surviving entries' frontmatter. The user-approved script:

```python
import os, re, glob

files = sorted(
    [f for f in glob.glob('*.md') if f != 'MEMORY.md'],
    key=lambda f: os.path.getmtime(f),
)

def parse_frontmatter(path):
    with open(path) as f:
        text = f.read()
    m = re.match(r'^---\n(.*?)\n---\n', text, re.DOTALL)
    if not m: return None, None
    fm = m.group(1)
    name = re.search(r'^name:\s*(.+)$', fm, re.MULTILINE)
    desc = re.search(r'^description:\s*(.+)$', fm, re.MULTILINE)
    return (
        name.group(1).strip() if name else path,
        desc.group(1).strip() if desc else '',
    )

lines = []
for f in files:
    n, d = parse_frontmatter(f)
    if n is None:
        continue
    lines.append(f'- [{n}]({f}) — {d}')

with open('MEMORY.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')
```

Sort key is mtime ascending (entries appear in roughly the order they were added, since consolidated entries are newest and end up at the bottom — which is correct).

### Phase 4 — Public MEMORY.md note (separate commit)

The public yarn-agentic `MEMORY.md` is append-only per CLAUDE.md §6. Do NOT rewrite or delete entries there. Append a brief note recording:

- Date and trigger (entry count, user-requested, etc.)
- Reduction stats (before / after entry counts)
- Which clusters were consolidated
- Which entries remain authoritative
- Confirmation that public MEMORY.md was not rewritten

Commit the note separately from any other changes per CLAUDE.md §6, and push.

## What to preserve

- **All durable rules** (the feedback corpus). Rules are the most valuable memory; only merge true facets-of-one-principle, never delete a rule.
- **Terminal-state project entries** (production landing, abandonment writeups, dead-end investigations).
- **Reference entries** (system facts, repo layout, protocols).
- **Branch / tag / SHA references** inside archive entries so prior states are reconstructable.

## What to distil away

- Intermediate phase snapshots ("Phase 36 Step 0 findings", "Phase 39 actual outcome", etc.) once a terminal entry summarises the workstream.
- Step-by-step progress reports that have been superseded by closure documentation.
- Multiple entries for the same milestone written in sequence as the work progressed.
- Plan files (`~/.claude/plans/*.md`) that have outcome sections — they're not in the memory dir but they capture the work; reference them from the archive entry.

## Verification

After the pass, the index should be:
- Self-consistent (every pointer resolves to an existing file).
- Sorted in a sensible order (mtime ascending works).
- Well under 200 lines.

Run:
```
ls *.md | grep -v '^MEMORY.md$' | wc -l   # entry count
wc -l MEMORY.md                            # index length
for f in *.md; do
  grep '^name:' "$f" 2>/dev/null | head -1
done | sort | uniq -c | sort -rn           # detect duplicate names
```

The user reads memory by scanning the index; if the index is over ~80 lines, future sessions should consider another dream pass.
