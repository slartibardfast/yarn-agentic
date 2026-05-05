#!/usr/bin/env bash
# Stop recorders for a snoop run-id and emit findings.md.
set -euo pipefail

RUN_ID="${1:?usage: snoop-stop.sh <run-id>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${HOME}/snoop-runs/${RUN_ID}"
PIDS="${OUT}/pids.txt"

if [[ ! -f "${PIDS}" ]]; then
    echo "no pids.txt at ${PIDS}" >&2
    exit 1
fi

while read -r pid; do
    [[ -z "${pid}" ]] && continue
    if kill -0 "${pid}" 2>/dev/null; then
        kill -TERM "${pid}" 2>/dev/null || true
    fi
done < "${PIDS}"

sleep 2

while read -r pid; do
    [[ -z "${pid}" ]] && continue
    if kill -0 "${pid}" 2>/dev/null; then
        kill -KILL "${pid}" 2>/dev/null || true
    fi
done < "${PIDS}"

echo "recorders stopped, summarising ..."
python3 "${SCRIPT_DIR}/snoop-summarise.py" "${OUT}"
echo "findings: ${OUT}/findings.md"
