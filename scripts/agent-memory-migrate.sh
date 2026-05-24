#!/usr/bin/env bash
# agent-memory-migrate — one-off migration of flat agent-memory/<slug>.md
# into event-sourced layout agent-memory/entries/<slug>__<host>__<ts>.md.
#
# For each existing entry file:
#   - skip docs (README.md, PROTOCOL.md, MEMORY.md)
#   - read mtime; format as RFC3339-ish UTC %Y%m%dT%H%M%S_000000Z
#   - move to agent-memory/entries/<slug>__<hostname-of-author>__<ts>.md
# Author host defaults to the current $(uname -n), since the original
# host-of-write isn't recorded anywhere; this is a historical bootstrap,
# acknowledged loss.
#
# Run once. Idempotent (if entries/ already populated, the corresponding
# flat files don't exist anymore, so no-op).
#
# After migration:
#   - agent-memory/entries/ holds the event log
#   - agent-memory/MEMORY.md is regenerated from the events
#   - agent-memory/README.md and PROTOCOL.md remain in place
#
# See agent-memory/PROTOCOL.md.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
MEM_DIR="$REPO_ROOT/agent-memory"
ENTRIES_DIR="$MEM_DIR/entries"

HOST_TAG="${HOST_TAG:-$(uname -n)}"

if [ ! -d "$MEM_DIR" ]; then
    echo "FAIL: $MEM_DIR does not exist"
    exit 2
fi

mkdir -p "$ENTRIES_DIR"

echo "=== agent-memory migration: flat → event log ==="
echo "src: $MEM_DIR (flat)"
echo "dst: $ENTRIES_DIR (event log)"
echo "author host tag: $HOST_TAG"
echo ""

migrated=0
skipped=0
for f in "$MEM_DIR"/*.md; do
    [ -f "$f" ] || continue
    base=$(basename "$f")
    case "$base" in
        README.md|PROTOCOL.md|MEMORY.md)
            echo "  skip (doc):     $base"
            skipped=$((skipped + 1))
            continue
            ;;
    esac
    slug="${base%.md}"
    # mtime as RFC3339-ish; pad microseconds to 0
    ts=$(date -u -r "$f" +"%Y%m%dT%H%M%S_000000Z" 2>/dev/null || date -u +"%Y%m%dT%H%M%S_000000Z")
    target="$ENTRIES_DIR/${slug}__${HOST_TAG}__${ts}.md"
    if [ -f "$target" ]; then
        echo "  target exists:  $base → already migrated, removing flat"
        rm -f "$f"
    else
        mv "$f" "$target"
        echo "  migrated:       $base → entries/${slug}__${HOST_TAG}__${ts}.md"
    fi
    migrated=$((migrated + 1))
done

echo ""
echo "migrated:  $migrated"
echo "skipped:   $skipped"
echo ""

# Regenerate MEMORY.md from the event set
echo "[regen] MEMORY.md from event set..."
/home/llm/venv/bin/python3 "$HERE/agent_memory_lib.py" \
    derive "$ENTRIES_DIR" "$MEM_DIR/MEMORY.md" events

echo "[audit]"
/home/llm/venv/bin/python3 "$HERE/agent_memory_lib.py" audit "$ENTRIES_DIR"

echo ""
echo "[done] now: $(ls "$ENTRIES_DIR"/*.md 2>/dev/null | wc -l) events; agent-memory/ root has README+PROTOCOL+MEMORY"
