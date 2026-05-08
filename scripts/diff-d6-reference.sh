#!/usr/bin/env bash
# PHASE45 D6 byte-identical diff harness.
#
# Runs bench-d6-byte-identical.sh on the given binary, then diffs the
# resulting tokens.txt against the saved reference (header stripped).
# Exits 0 if byte-identical, non-zero otherwise.
#
# Usage:
#   diff-d6-reference.sh <BIN> [REFERENCE]
#
#     BIN        path to llama-cli to test
#     REFERENCE  reference token file (default: data/phase45-d6-reference.txt)
#
# The reference file's first lines are `# ...` header comments (model
# SHA, profile, binary, flags). Those are filtered out before diff so
# they don't pollute the comparison. Token-data lines are non-comment.

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <BIN> [REFERENCE]" >&2
    exit 1
fi

BIN="$1"
REFERENCE="${2:-/home/llm/yarn-agentic/data/phase45-d6-reference.txt}"
HARNESS="/home/llm/yarn-agentic/scripts/bench-d6-byte-identical.sh"

if [ ! -f "$REFERENCE" ]; then
    echo "ERROR: reference file not found: $REFERENCE" >&2
    exit 2
fi

if [ ! -x "$HARNESS" ]; then
    echo "ERROR: harness not found or not executable: $HARNESS" >&2
    exit 3
fi

OUTDIR=$(mktemp -d -t d6-diff-XXXXXX)
trap 'rm -rf "$OUTDIR"' EXIT

"$HARNESS" "$BIN" "$OUTDIR" >/dev/null

CANDIDATE="$OUTDIR/tokens.txt"

if [ ! -f "$CANDIDATE" ]; then
    echo "ERROR: harness did not produce $CANDIDATE" >&2
    exit 4
fi

# Strip header comment lines (lines starting with '#') from the reference
# before diffing.
REF_DATA=$(mktemp -t d6-ref-XXXXXX)
trap 'rm -rf "$OUTDIR" "$REF_DATA"' EXIT
grep -v -E '^#' "$REFERENCE" > "$REF_DATA"

if diff -q "$REF_DATA" "$CANDIDATE" >/dev/null; then
    echo "PASS: $BIN matches $REFERENCE byte-identical (50 tokens)"
    exit 0
else
    echo "FAIL: $BIN differs from $REFERENCE"
    echo "--- diff (reference -> candidate) ---"
    diff "$REF_DATA" "$CANDIDATE" || true
    exit 5
fi
