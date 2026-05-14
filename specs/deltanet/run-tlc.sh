#!/usr/bin/env bash
# Run TLC with the throughput-optimized parallel GC (per TLC's startup
# recommendation) for the DeltaNet batch-invariance spec configs.
#
# Usage:
#   ./run-tlc.sh ModalP3
#   ./run-tlc.sh OutlierP0
#
# Add new configs by dropping a PipelineDeterminism_<name>.cfg in this
# directory.

set -euo pipefail

CONFIG="${1:-ModalP3}"
SPEC_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG_FILE="${SPEC_DIR}/PipelineDeterminism_${CONFIG}.cfg"
TLA_FILE="${SPEC_DIR}/PipelineDeterminism.tla"
TLA_TOOLS="/home/llm/Specula/lib/tla2tools.jar"

if [[ ! -f "${CFG_FILE}" ]]; then
    echo "Config file not found: ${CFG_FILE}" >&2
    exit 1
fi

cd "${SPEC_DIR}"
exec java -XX:+UseParallelGC -cp "${TLA_TOOLS}" tlc2.TLC \
    -nowarning \
    -config "${CFG_FILE}" \
    "${TLA_FILE}"
