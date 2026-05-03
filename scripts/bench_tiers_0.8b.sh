#!/usr/bin/env bash
# 5-run bench-mtp average for V-F1a at all five tiers, side-by-side TSV.
# Greedy temp=0, n_predict=256, single-CUDA0 to avoid cross-device noise.

set -uo pipefail

OUT_DIR=/opt/models/recast-out
VARIANT=${VARIANT:-V-F1a}
TSV=$OUT_DIR/bench-tiers-0.8b-${VARIANT}.tsv
BIN=/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-server
PORT=${PORT:-18181}
PROMPT="The history of artificial intelligence began in earnest in"
N_PREDICT=${N_PREDICT:-256}
RUNS=${RUNS:-5}

mkdir -p "$OUT_DIR/logs"

# We never pkill any other server — refuse if port busy.
if curl -fsS --max-time 1 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "port $PORT is busy; aborting" >&2
    exit 2
fi

echo -e "tier\tmtp\trun_avg_tg\trun_avg_pp\taccept_rate" > "$TSV"

run_mode() {
    local gguf=$1 tier=$2 mtp_flag=$3 mtp_label=$4
    local log="$OUT_DIR/logs/bench-${VARIANT}-$tier-$mtp_label.log"
    "$BIN" -m "$gguf" \
        --device "${BENCH_DEV:-CUDA0}" -ngl 999 -fa on $mtp_flag \
        --draft 1 -c 4096 \
        --no-mmap \
        --threads 16 --batch-size 2048 --ubatch-size 512 \
        --no-context-shift --metrics --port "$PORT" --host 127.0.0.1 \
        > "$log" 2>&1 &
    local SRV=$!
    # health-wait
    local ok=0
    for i in $(seq 1 180); do
        if curl -fsS --max-time 1 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then ok=1; break; fi
        sleep 0.5
    done
    if [ "$ok" -eq 0 ]; then
        echo "  $tier $mtp_label  server failed to start (see $log)"
        kill -9 "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
        return 1
    fi
    # warmup
    curl -fsS -H "Content-Type: application/json" \
        -d '{"prompt":"warmup","n_predict":4,"temperature":0,"stream":false}' \
        "http://127.0.0.1:${PORT}/completion" > /dev/null

    local total_tg=0 total_pp=0
    for r in $(seq 1 "$RUNS"); do
        local resp
        resp=$(curl -fsS -H "Content-Type: application/json" -d "{
            \"prompt\":\"${PROMPT}\",
            \"n_predict\":${N_PREDICT},
            \"temperature\":0,
            \"stream\":false
        }" "http://127.0.0.1:${PORT}/completion")
        local tg pp
        tg=$(echo "$resp" | /home/llm/venv/bin/python -c "import sys,json; t=json.load(sys.stdin)['timings']; print(f\"{t['predicted_per_second']:.4f}\")")
        pp=$(echo "$resp" | /home/llm/venv/bin/python -c "import sys,json; t=json.load(sys.stdin)['timings']; print(f\"{t.get('prompt_per_second',0):.4f}\")")
        total_tg=$(echo "$total_tg + $tg" | bc -l)
        total_pp=$(echo "$total_pp + $pp" | bc -l)
    done
    local avg_tg avg_pp
    avg_tg=$(echo "scale=4; $total_tg / $RUNS" | bc -l)
    avg_pp=$(echo "scale=4; $total_pp / $RUNS" | bc -l)
    local accept
    accept=$(grep -E "draft acceptance rate" "$log" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "")
    printf "  %s %-7s tg=%6.2f pp=%6.2f accept=%s\n" "$tier" "$mtp_label" "$avg_tg" "$avg_pp" "${accept:-?}"
    printf "%s\t%s\t%s\t%s\t%s\n" "$tier" "$mtp_label" "$avg_tg" "$avg_pp" "${accept:-?}" >> "$TSV"

    kill -9 "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
    sleep 2
}

for tier in T1 T2 T3 T4 T5; do
    gguf="$OUT_DIR/Qwen3.5-0.8B-${VARIANT}-$tier.gguf"
    if [ ! -f "$gguf" ]; then echo "missing $gguf"; continue; fi
    echo "=== bench ${VARIANT:-V-F1a}.$tier ==="
    run_mode "$gguf" "$tier" "-no-mtp" "nomtp"
    run_mode "$gguf" "$tier" "-mtp"    "mtp"
done

echo
echo "==> results: $TSV"
column -s $'\t' -t < "$TSV"
