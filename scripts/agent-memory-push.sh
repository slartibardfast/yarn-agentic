#!/usr/bin/env bash
# agent-memory-push — push host-local memory edits as new events in the
# repo's event log, then commit + push.
#
# For each <slug>.md in ~/.claude/projects/-home-llm-yarn-agentic/memory/:
#   - if content differs from newest event for that slug (or no event
#     exists), append a NEW event file:
#         agent-memory/entries/<slug>__<host>__<rfc3339-ts>.md
#   - if content matches newest event, do nothing
#
# After events are written, the repo's MEMORY.md is regenerated from
# the event set (links point to the newest event filename per slug).
#
# Concurrency: per-slug events are uniquely-named so concurrent writes
# never overwrite each other on disk. `git pull --rebase` is run first
# to incorporate other hosts' events; rebase should NOT conflict since
# event filenames are disjoint across hosts. MEMORY.md is regenerated
# from the post-rebase event set, never hand-merged.
#
# See agent-memory/PROTOCOL.md.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
ENTRIES_DIR="$REPO_ROOT/agent-memory/entries"
LIVE_DIR="$HOME/.claude/projects/-home-llm-yarn-agentic/memory"
PY="${PY:-/home/llm/venv/bin/python3}"
HOST_TAG="${HOST_TAG:-$(uname -n)}"

if [ ! -d "$LIVE_DIR" ]; then
    echo "FAIL: $LIVE_DIR does not exist (run agent-memory-pull.sh first?)"
    exit 2
fi
if [ ! -d "$REPO_ROOT/agent-memory" ]; then
    echo "FAIL: $REPO_ROOT/agent-memory does not exist"
    exit 2
fi
mkdir -p "$ENTRIES_DIR"

echo "=== agent-memory push (live → event log) ==="
echo "from: $LIVE_DIR"
echo "to:   $ENTRIES_DIR"
echo "host tag: $HOST_TAG"

echo "[1/5] git pull --rebase ..."
cd "$REPO_ROOT"
git pull --rebase 2>&1 | tail -5

echo "[2/5] diff live vs event log ..."
# Capture into temp file (stderr has summary; stdout has CHANGED lines).
DIFF_TMP="$(mktemp)"
"$PY" "$HERE/agent_memory_lib.py" diff-live "$LIVE_DIR" "$ENTRIES_DIR" \
    > "$DIFF_TMP" 2>&1
# Show the summary line (last line that begins with '#')
grep '^# ' "$DIFF_TMP" || true
# Show changed entries
n_changed=$(grep -cE '^(NEW|UPDATE)' "$DIFF_TMP" 2>/dev/null || true)
n_changed=${n_changed:-0}
echo "[3/5] changed slugs: $n_changed"
if [ "$n_changed" = "0" ]; then
    echo "[done] no live edits since last push (no new events to write)"
    # Still regenerate MEMORY.md in case event log was rebased.
    "$PY" "$HERE/agent_memory_lib.py" derive \
        "$ENTRIES_DIR" "$REPO_ROOT/agent-memory/MEMORY.md" events
    if git diff --quiet agent-memory/MEMORY.md; then
        echo "[done] MEMORY.md unchanged; nothing to commit"
        rm -f "$DIFF_TMP"
        exit 0
    fi
fi

# Write new events
echo "[4/5] write events ..."
grep -E '^(NEW|UPDATE)' "$DIFF_TMP" | while IFS=$'\t' read -r status slug live_path; do
    out=$("$PY" "$HERE/agent_memory_lib.py" \
        write-event "$ENTRIES_DIR" "$slug" "$HOST_TAG" "$live_path")
    echo "  $status $slug → $(basename "$out")"
done
rm -f "$DIFF_TMP"

echo "[5/5] regenerate repo MEMORY.md + git commit + push ..."
"$PY" "$HERE/agent_memory_lib.py" derive \
    "$ENTRIES_DIR" "$REPO_ROOT/agent-memory/MEMORY.md" events

git add agent-memory/

if git diff --cached --quiet agent-memory/; then
    echo "[done] no staged changes (rare; check status manually)"
    exit 0
fi

n_added=$(git diff --cached --name-only --diff-filter=A agent-memory/ | wc -l)
n_modified=$(git diff --cached --name-only --diff-filter=M agent-memory/ | wc -l)
git commit -m "agent-memory sync from ${HOST_TAG}: ${n_added} new event(s), MEMORY.md regenerated

Live → event log via scripts/agent-memory-push.sh.
$n_added new event files, $n_modified files modified (typically MEMORY.md).
Per agent-memory/PROTOCOL.md (CRDT, G-Set on events).
"

git push 2>&1 | tail -3

echo ""
echo "[done] event log: $(ls "$ENTRIES_DIR"/*.md 2>/dev/null | wc -l) total events"
