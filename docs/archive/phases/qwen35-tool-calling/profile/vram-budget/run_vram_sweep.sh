#!/usr/bin/env bash
# VRAM budget sweep. For each (n_ctx, --cache-type-v) combo, attempt to
# start the server and capture the llama_memory_breakdown_print output.
# If startup fails (OOM, -fit off rejected, etc.), log the error and
# continue to the next combo.
#
# The point is to find the OOM cliff on Vega 64 (8 GB).
#
# Usage: run_vram_sweep.sh <TS>
set -uo pipefail

TS="${1:-$(date -u +%Y-%m-%dT%H%M%SZ)}"
MODEL=/home/llm/models/Qwen3.5-9B-mtp-q4km.gguf
DEST=/home/llm/yarn-agentic/phases/qwen35-mtp/profile/vram-budget
TMPDIR=/tmp

CTX_LIST="4096 8192 12288 16384 24576 32768"
V_LIST="f16 q4_0 tq_v_4b"

for V in $V_LIST; do
    for CTX in $CTX_LIST; do
        echo "=== ctx=$CTX v=$V ==="
        LOG="${TMPDIR}/vram-${V}-${CTX}-${TS}.stderr"

        GGML_VK_VISIBLE_DEVICES=1 \
        GGML_VK_MEMORY_LOGGER=1 \
            /home/llm/src/qwen35-mtp/build-vk/bin/llama-server \
                -m "$MODEL" \
                -c "$CTX" -ngl 99 -np 1 -fa on \
                --cache-type-v "$V" \
                --host 127.0.0.1 --port 9099 \
                --no-warmup > "$LOG" 2>&1 &
        PID=$!

        ready=0
        failed=0
        for i in $(seq 1 60); do
            if grep -q 'all slots are idle' "$LOG" 2>/dev/null; then
                ready=1
                break
            fi
            if grep -qiE 'failed to fit|error|abort|insufficient|ENOMEM' "$LOG" 2>/dev/null; then
                failed=1
                break
            fi
            sleep 1
        done

        # Graceful kill — gives llama_memory_breakdown_print a chance to fire.
        kill "$PID" 2>/dev/null
        for i in $(seq 1 10); do
            if ! kill -0 "$PID" 2>/dev/null; then break; fi
            sleep 1
        done
        kill -9 "$PID" 2>/dev/null
        wait "$PID" 2>/dev/null

        cp "$LOG" "${DEST}/vram-${V}-ctx${CTX}-${TS}.stderr"
        rm -f "$LOG"

        if [ "$ready" -eq 1 ]; then
            echo "  ok"
        elif [ "$failed" -eq 1 ]; then
            echo "  FAILED to start"
            grep -iE 'failed|error|abort|insufficient|ENOMEM' \
                "${DEST}/vram-${V}-ctx${CTX}-${TS}.stderr" | head -3
            # If we failed at this ctx, larger ctx will also fail — skip to next V.
            break
        else
            echo "  TIMEOUT"
        fi
    done
done

echo "=== done vram sweep ==="
