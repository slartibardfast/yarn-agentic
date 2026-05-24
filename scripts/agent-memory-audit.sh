#!/usr/bin/env bash
# agent-memory-audit — sanity check the agent-memory directory.
#
# Checks for:
#   - orphan files (exist on disk but not referenced from MEMORY.md)
#   - missing files (referenced from MEMORY.md but not on disk)
#   - file count vs entry count
#   - SUPERSEDED / OVERRIDDEN markers (legitimate; should be preserved)
#
# Targets the LIVE directory by default (since that's what the agent
# reads). Pass --repo to audit the repo's snapshot instead.
#
# See agent-memory/PROTOCOL.md for context.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

case "${1:-}" in
    --repo)
        TARGET="$REPO_ROOT/agent-memory"
        LABEL="repo snapshot"
        ;;
    --live|"")
        TARGET="$HOME/.claude/projects/-home-llm-yarn-agentic/memory"
        LABEL="host live"
        ;;
    *)
        echo "usage: $0 [--live|--repo]" >&2
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

if [ ! -f "$TARGET/MEMORY.md" ]; then
    echo "FAIL: $TARGET/MEMORY.md missing — cannot audit"
    exit 1
fi

n_files=$(ls "$TARGET"/*.md 2>/dev/null | wc -l)
n_entries=$(grep -c '^- \[' "$TARGET/MEMORY.md" 2>/dev/null || echo 0)
echo "file count:       $n_files (= MEMORY.md + $((n_files - 1)) entry files)"
echo "MEMORY.md entries: $n_entries"
echo ""

# Orphan files (disk file with no MEMORY.md reference).
echo "=== orphans (file present, no MEMORY.md ref) ==="
orphan_count=0
for f in "$TARGET"/*.md; do
    base=$(basename "$f")
    if [ "$base" = "MEMORY.md" ] || [ "$base" = "README.md" ] || [ "$base" = "PROTOCOL.md" ]; then
        continue
    fi
    slug="${base%.md}"
    if ! grep -q "$slug" "$TARGET/MEMORY.md"; then
        echo "  $base"
        orphan_count=$((orphan_count + 1))
    fi
done
if [ "$orphan_count" -eq 0 ]; then echo "  (none)"; fi
echo ""

# Missing files (MEMORY.md references something not on disk).
echo "=== missing (MEMORY.md ref, no file on disk) ==="
missing_count=0
while read -r ref; do
    # Match (filename.md) in markdown link.
    refbase=$(echo "$ref" | grep -oE '\([a-z_0-9-]+\.md\)' | tr -d '()' | head -1)
    [ -z "$refbase" ] && continue
    if [ ! -f "$TARGET/$refbase" ]; then
        echo "  $refbase (referenced in MEMORY.md but not on disk)"
        missing_count=$((missing_count + 1))
    fi
done < <(grep '^- \[' "$TARGET/MEMORY.md")
if [ "$missing_count" -eq 0 ]; then echo "  (none)"; fi
echo ""

# SUPERSEDED markers.
echo "=== SUPERSEDED / OVERRIDDEN markers (legitimate audit records) ==="
super_count=$(grep -l -iE 'SUPERSEDED|OVERRIDDEN' "$TARGET"/*.md 2>/dev/null | wc -l)
echo "  $super_count files contain these markers"
grep -l -iE 'SUPERSEDED|OVERRIDDEN' "$TARGET"/*.md 2>/dev/null | head -5 | sed 's|^|  |'
echo ""

# Most-recently-modified.
echo "=== 5 most-recently-modified files ==="
ls -lt "$TARGET"/*.md 2>/dev/null | head -5 | awk '{print "  " $6 " " $7 " " $8 " " $NF}'
echo ""

# Verdict.
if [ "$orphan_count" -eq 0 ] && [ "$missing_count" -eq 0 ]; then
    echo "[ok] audit clean: $n_files files, $n_entries entries, 1:1 mapping"
    exit 0
else
    echo "[warn] audit found: $orphan_count orphans, $missing_count missing"
    exit 1
fi
