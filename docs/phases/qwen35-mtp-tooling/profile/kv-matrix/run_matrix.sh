#!/usr/bin/env bash
# Run the KV-V quant matrix. For each V type, start a server, grab the
# K/V buffer sizes from startup, drive 3x workload via drive.py, kill.
# Emits kv-<type>-<UTC>.{stderr,json,drive.json} per type.
#
# Usage: run_matrix.sh <UTC_stem>
#
# Driven by the overnight run externally — this script expects a
# running bash with llama-server available and the profile dir tree
# already created.
set -uo pipefail

TS="${1:-$(date -u +%Y-%m-%dT%H%M%SZ)}"
MODEL=/home/llm/models/Qwen3.5-9B-mtp-q4km.gguf
DEST=/home/llm/yarn-agentic/phases/qwen35-mtp/profile/kv-matrix
TMPDIR=/tmp

for V in f16 q4_0 q4_1 q5_0 q5_1 q8_0 iq4_nl tq_v_4b; do
    echo "=== $V ==="
    LOG="${TMPDIR}/kv-${V}-${TS}.stderr"
    DRV="${TMPDIR}/kv-${V}-${TS}.drive.txt"

    # Set perf logger; keep concurrent mode so timings are non-blocking.
    GGML_VK_VISIBLE_DEVICES=1 \
    GGML_VK_PERF_LOGGER=1 \
    GGML_VK_PERF_LOGGER_CONCURRENT=1 \
    GGML_VK_PERF_LOGGER_FREQUENCY=1 \
        /home/llm/src/qwen35-mtp/build-vk/bin/llama-server \
            -m "$MODEL" \
            -c 4096 -ngl 99 -np 1 -fa on \
            --cache-type-v "$V" \
            --host 127.0.0.1 --port 9099 \
            --no-warmup > "$LOG" 2>&1 &
    PID=$!
    echo "  pid=$PID"

    # Wait for readiness.
    ready=0
    for i in $(seq 1 60); do
        if grep -q 'all slots are idle' "$LOG" 2>/dev/null; then
            ready=1
            break
        fi
        sleep 1
    done
    if [ "$ready" -ne 1 ]; then
        echo "  FAILED: server not ready after 60s; grep for errors:"
        grep -iE 'error|abort|failed|insufficient' "$LOG" | head -3
        kill -9 "$PID" 2>/dev/null
        wait "$PID" 2>/dev/null
        continue
    fi

    # Drive 3 runs.
    python3 /home/llm/yarn-agentic/phases/qwen35-mtp/profile/drive.py 3 \
        --label "kv-$V" > "$DRV" 2>&1
    echo "  drive done"

    # Kill server, let perf logger flush via SIGTERM.
    kill "$PID" 2>/dev/null
    wait "$PID" 2>/dev/null

    # Move outputs into the destination dir.
    cp "$LOG" "${DEST}/kv-${V}-${TS}.stderr"
    cp "$DRV" "${DEST}/kv-${V}-${TS}.drive.txt"

    # Parse perf logger output.
    python3 /home/llm/yarn-agentic/phases/qwen35-mtp/profile/parse_vk_perf.py \
        "${DEST}/kv-${V}-${TS}.stderr" > "${DEST}/kv-${V}-${TS}.perf.json"

    # Tear down the temp files.
    rm -f "$LOG" "$DRV"
done

echo "=== done matrix ==="
