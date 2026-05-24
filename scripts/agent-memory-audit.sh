#!/usr/bin/env bash
# agent-memory-audit — sanity check the event log.
#
# Targets agent-memory/entries/ by default; --live to audit the live dir
# (which is a derived materialization and should mirror the event-log's
# newest-per-slug content).
#
# See agent-memory/PROTOCOL.md.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
PY="${PY:-/home/llm/venv/bin/python3}"

case "${1:-}" in
    --live)
        TARGET="$HOME/.claude/projects/-home-llm-yarn-agentic/memory"
        LABEL="host live (derived view)"
        ;;
    --repo|--events|"")
        TARGET="$REPO_ROOT/agent-memory/entries"
        LABEL="repo event log"
        ;;
    *)
        echo "usage: $0 [--repo|--events|--live]" >&2
        exit 2
        ;;
esac

if [ ! -d "$TARGET" ]; then
    echo "FAIL: $TARGET does not exist"
    exit 2
fi

echo "=== agent-memory audit ($LABEL) ==="
echo "target: $TARGET"
echo ""

if [[ "$TARGET" == *"/entries" ]]; then
    # Event-log audit
    "$PY" "$HERE/agent_memory_lib.py" audit "$TARGET"
else
    # Live dir audit: count slug-files vs MEMORY.md entries
    if [ ! -f "$TARGET/MEMORY.md" ]; then
        echo "(no MEMORY.md in live; run agent-memory-pull.sh first)"
        exit 1
    fi
    n_files=$(ls "$TARGET"/*.md 2>/dev/null | wc -l)
    # MEMORY.md is one of those .md files; subtract.
    n_entries=$(grep -c '^- \[' "$TARGET/MEMORY.md" 2>/dev/null || echo 0)
    echo "live files:        $n_files (including MEMORY.md)"
    echo "MEMORY.md entries: $n_entries"
    echo ""
    echo "(use --repo for an event-log audit)"
fi
