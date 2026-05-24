#!/usr/bin/env bash
# agent-memory-pull — refresh the host-local live directory from the
# event-log in the repo.
#
# Reads:  agent-memory/entries/<slug>__<host>__<ts>.md (G-Set, append-only)
# Writes: ~/.claude/projects/-home-llm-yarn-agentic/memory/<slug>.md  (newest event per slug)
# Writes: ~/.claude/projects/-home-llm-yarn-agentic/memory/MEMORY.md  (derived index)
#
# Semantics:
#   - This is a derived materialization; the entries/ directory is the
#     authoritative store. The live dir is a per-host current-view.
#   - The pull replaces files in live with the newest-event-per-slug
#     content. If live has been edited since last push, those edits
#     will be overwritten ONLY IF a newer event exists; otherwise the
#     live changes survive (no live file is created without a matching
#     event, and unchanged-content writes don't touch mtime).
#   - Files in live that have no corresponding event are NOT deleted
#     (live-only changes are preserved until push).
#
# See agent-memory/PROTOCOL.md.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
ENTRIES_DIR="$REPO_ROOT/agent-memory/entries"
LIVE_DIR="$HOME/.claude/projects/-home-llm-yarn-agentic/memory"
PY="${PY:-/home/llm/venv/bin/python3}"

if [ ! -d "$REPO_ROOT/agent-memory" ]; then
    echo "FAIL: $REPO_ROOT/agent-memory does not exist (is the repo checked out?)"
    exit 2
fi

mkdir -p "$LIVE_DIR" "$ENTRIES_DIR"

echo "=== agent-memory pull (event log → live) ==="
echo "from: $ENTRIES_DIR"
echo "to:   $LIVE_DIR"

echo "[1/2] git pull --rebase ..."
cd "$REPO_ROOT"
git pull --rebase 2>&1 | tail -5

echo "[2/2] materialize live from event log ..."
"$PY" "$HERE/agent_memory_lib.py" materialize-live "$ENTRIES_DIR" "$LIVE_DIR"

echo ""
echo "[done] live: $(ls "$LIVE_DIR"/*.md 2>/dev/null | wc -l) files"
echo "[done] MEMORY.md entries: $(grep -c '^- \[' "$LIVE_DIR/MEMORY.md" 2>/dev/null || echo 0)"
