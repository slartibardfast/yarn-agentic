#!/usr/bin/env bash
# 5-run bench-mtp average for a single 35B-A3B GGUF (one cell at a time
# because each is 67 GiB and only one fits in /opt headroom).
# Greedy temp=0, n_predict=256, dual-CUDA + cpu-moe.
#
# Usage:
#   bench_tiers_35b.sh <gguf-path> <variant> <tier>
#
# Output rows appended to /opt/models/recast-out/bench-tiers-35b.tsv

set -uo pipefail

GGUF=${1:-}
VARIANT=${2:-?}
TIER=${3:-?}

if [ -z "$GGUF" ] || [ ! -f "$GGUF" ]; then
    echo "usage: $0 <gguf-path> <variant> <tier>" >&2
    exit 2
fi

OUT_DIR=/opt/models/recast-out
TSV=$OUT_DIR/bench-tiers-35b.tsv
BIN=/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-server
PORT=${PORT:-18181}
PROMPT="The history of artificial intelligence began in earnest in"
N_PREDICT=${N_PREDICT:-256}
RUNS=${RUNS:-5}
NCMOE=${BENCH_NCMOE:-25}

mkdir -p "$OUT_DIR/logs"
[ -f "$TSV" ] || echo -e "variant\ttier\tmtp\trun_avg_tg\trun_avg_pp\taccept_rate\tcoherent" > "$TSV"

if curl -fsS --max-time 1 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "port $PORT busy; aborting" >&2
    exit 2
fi

run_mode() {
    local mtp_flag=$1 mtp_label=$2
    local log="$OUT_DIR/logs/bench-35b-${VARIANT}-${TIER}-${mtp_label}.log"
    echo "  starting $VARIANT.$TIER $mtp_label (log $log)"
    "$BIN" -m "$GGUF" \
        --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
        -ngl 999 -fa on $mtp_flag \
        -ncmoe "$NCMOE" \
        --no-mmap \
        --draft 1 -c 4096 \
        --threads 16 --batch-size 2048 --ubatch-size 512 \
        --no-context-shift --metrics --port "$PORT" --host 127.0.0.1 \
        > "$log" 2>&1 &
    local SRV=$!
    # wait up to 8 min for /health (35B FP16 load is slow)
    local ok=0
    for i in $(seq 1 960); do
        if curl -fsS --max-time 1 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then ok=1; break; fi
        sleep 0.5
    done
    if [ "$ok" -eq 0 ]; then
        echo "  $VARIANT.$TIER $mtp_label  server failed to start (see $log)"
        kill -9 "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
        printf "%s\t%s\t%s\tFAIL\tFAIL\tFAIL\tFAIL\n" "$VARIANT" "$TIER" "$mtp_label" >> "$TSV"
        return 1
    fi
    # warmup
    curl -fsS -H "Content-Type: application/json" \
        -d '{"prompt":"warmup","n_predict":4,"temperature":0,"stream":false}' \
        "http://127.0.0.1:${PORT}/completion" > /dev/null

    local total_tg=0 total_pp=0
    local out_first=""
    for r in $(seq 1 "$RUNS"); do
        local resp tg pp content
        resp=$(curl -fsS -H "Content-Type: application/json" -d "{
            \"prompt\":\"${PROMPT}\",
            \"n_predict\":${N_PREDICT},
            \"temperature\":0,
            \"stream\":false
        }" "http://127.0.0.1:${PORT}/completion")
        tg=$(echo "$resp" | /home/llm/venv/bin/python -c "import sys,json; t=json.load(sys.stdin)['timings']; print(f\"{t['predicted_per_second']:.4f}\")")
        pp=$(echo "$resp" | /home/llm/venv/bin/python -c "import sys,json; t=json.load(sys.stdin)['timings']; print(f\"{t.get('prompt_per_second',0):.4f}\")")
        content=$(echo "$resp" | /home/llm/venv/bin/python -c "import sys,json; print(json.load(sys.stdin)['content'][:120])")
        [ "$r" = "1" ] && out_first="$content"
        total_tg=$(echo "$total_tg + $tg" | bc -l)
        total_pp=$(echo "$total_pp + $pp" | bc -l)
    done
    local avg_tg avg_pp
    avg_tg=$(echo "scale=4; $total_tg / $RUNS" | bc -l)
    avg_pp=$(echo "scale=4; $total_pp / $RUNS" | bc -l)
    local accept
    accept=$(grep -E "draft acceptance rate" "$log" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "")
    local coh="?"
    if [ -n "$out_first" ] && [ "${#out_first}" -ge 8 ]; then coh=Y; else coh=N; fi
    printf "  %s %s %-7s tg=%6.2f pp=%6.2f accept=%s\n" "$VARIANT" "$TIER" "$mtp_label" "$avg_tg" "$avg_pp" "${accept:-?}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$VARIANT" "$TIER" "$mtp_label" "$avg_tg" "$avg_pp" "${accept:-?}" "$coh" >> "$TSV"

    kill -9 "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
    sleep 5
}

echo "=== bench $VARIANT.$TIER on $GGUF ==="
run_mode "-no-mtp" "nomtp"
run_mode "-mtp"    "mtp"

echo
column -s $'\t' -t < "$TSV"
