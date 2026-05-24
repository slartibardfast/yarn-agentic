#!/usr/bin/env bash
# agent-memory-push — sync the host-local live directory
# ~/.claude/projects/-home-llm-yarn-agentic/memory/ into the repo's
# agent-memory/, commit, push.
#
# Run this at session end (or after writing meaningful new entries).
#
# Semantics:
#   - Auto-`git pull --rebase` first to incorporate other hosts' commits.
#   - rsync from live → repo (without --update; this host's session is
#     authoritative for the files it edited).
#   - No --delete: files in repo but not live are KEPT (so other hosts'
#     memories aren't erased).
#   - git add + commit + push agent-memory/ only.
#
# Conflict surface:
#   - If another host pushed an edit to MEMORY.md since this host's last
#     pull, `git pull --rebase` will surface the conflict. Resolve by
#     hand (typical: merge both hosts' new index entries).
#
# See agent-memory/PROTOCOL.md for the full protocol.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
REPO_MEM="$REPO_ROOT/agent-memory"
LIVE_MEM="$HOME/.claude/projects/-home-llm-yarn-agentic/memory"

if [ ! -d "$LIVE_MEM" ]; then
    echo "FAIL: $LIVE_MEM does not exist (run agent-memory-pull.sh first?)"
    exit 2
fi

if [ ! -d "$REPO_MEM" ]; then
    echo "FAIL: $REPO_MEM does not exist (is the repo checked out?)"
    exit 2
fi

echo "=== agent-memory push ==="
echo "from: $LIVE_MEM"
echo "to:   $REPO_MEM (then commit + push)"

# Pre-flight: pull any new commits from upstream so we rebase cleanly.
cd "$REPO_ROOT"
echo "[1/4] git pull --rebase ..."
git pull --rebase 2>&1 | tail -5

# rsync live → repo. Do NOT use --delete (preserve other hosts' files).
echo "[2/4] rsync live → repo..."
rsync -av \
    --include='*.md' --exclude='*' \
    "$LIVE_MEM/" "$REPO_MEM/" 2>&1 | tail -10

# Stage changes.
echo "[3/4] git add + commit + push agent-memory/..."
git add agent-memory/

# Bail if nothing changed.
if git diff --cached --quiet agent-memory/; then
    echo "[done] no changes to commit"
    exit 0
fi

# Build a useful commit message from the changed files.
N_CHANGED=$(git diff --cached --name-only agent-memory/ | wc -l)
NEW_FILES=$(git diff --cached --name-only --diff-filter=A agent-memory/ | wc -l)
MOD_FILES=$(git diff --cached --name-only --diff-filter=M agent-memory/ | wc -l)
SAMPLE=$(git diff --cached --name-only agent-memory/ | head -5 | sed 's|agent-memory/||' | tr '\n' ' ')

git commit -m "agent-memory sync from $(hostname): ${NEW_FILES} new, ${MOD_FILES} modified

Files touched ($N_CHANGED total, sample): $SAMPLE

Pushed via scripts/agent-memory-push.sh; per agent-memory/PROTOCOL.md.
"

echo "[4/4] git push..."
git push 2>&1 | tail -3

echo ""
echo "[done] repo has $(ls "$REPO_MEM"/*.md 2>/dev/null | wc -l) memory files"
echo "[done] MEMORY.md entries: $(grep -c '^- \[' "$REPO_MEM/MEMORY.md" 2>/dev/null || echo 0)"
