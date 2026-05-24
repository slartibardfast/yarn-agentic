#!/usr/bin/env bash
# agent-memory-pull — sync the repo's agent-memory/ into the host-local
# live directory ~/.claude/projects/-home-llm-yarn-agentic/memory/.
#
# Run this at session start to pick up memories written by other hosts.
#
# Semantics:
#   - rsync --update: per file, only overwrite live if repo has a newer mtime.
#     Live-side edits since last push are preserved.
#   - No --delete: files only in live are kept (they will go to repo at next push).
#   - MEMORY.md gets the same per-file update rule. If both sides edited it
#     since last sync, the newer mtime wins. To avoid losing local edits,
#     ALWAYS pull before writing.
#
# See agent-memory/PROTOCOL.md for the full protocol.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
REPO_MEM="$REPO_ROOT/agent-memory"
LIVE_MEM="$HOME/.claude/projects/-home-llm-yarn-agentic/memory"

if [ ! -d "$REPO_MEM" ]; then
    echo "FAIL: $REPO_MEM does not exist (is the repo checked out?)"
    exit 2
fi

mkdir -p "$LIVE_MEM"

echo "=== agent-memory pull ==="
echo "from: $REPO_MEM"
echo "to:   $LIVE_MEM"

# First, refresh repo from upstream so we have the latest committed memories.
echo "[1/2] git pull --rebase agent-memory/ ..."
cd "$REPO_ROOT"
git pull --rebase 2>&1 | tail -5

# Now rsync repo → live with --update (newer mtime wins, per file).
# --include='*.md' to be paranoid (only markdown), --exclude='*' otherwise.
echo "[2/2] rsync --update repo → live..."
rsync -av --update \
    --include='*.md' --exclude='*' \
    "$REPO_MEM/" "$LIVE_MEM/" 2>&1 | tail -10

echo ""
echo "[done] live now has $(ls "$LIVE_MEM"/*.md 2>/dev/null | wc -l) memory files"
echo "[done] MEMORY.md entries: $(grep -c '^- \[' "$LIVE_MEM/MEMORY.md" 2>/dev/null || echo 0)"
