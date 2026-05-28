#!/usr/bin/env bash
# perf-r3-followup-phase1.sh — Phase 1 of PHASE_PERF_R3_FOLLOWUP.
#
# R2 inflection sweep at ctx=262144, NP=1, ubatch=256 + full RT chain.
# Sweeps prompt depths {3k,4k,5k,6k,8k,10k,12k}, 3 reps each, N_PREDICT=128,
# greedy decode. Localizes the 3k->12k TG cliff observed in Phase E.
#
# Usage:
#     OUT=<dir> bash perf-r3-followup-phase1.sh

set -uo pipefail

BIN=/home/dconnolly/yarn-agentic/ik_llama.cpp/build/bin/llama-server
GGUF=/opt/models/recast-out/qwen3.6-27b-V-F1.T1.lm_head-f16.gguf
PROMPT_DIR=/tmp/perf-r3-prompts
OUT="${OUT:-/tmp/perf-r3-followup-phase1-$(date -u +%Y%m%dT%H%M%S)}"
PORT="${PORT:-18294}"
mkdir -p "$OUT"

DEPTHS=(3000 4000 5000 6000 8000 10000 12000)
N_REPS=3

start_server() {
    sudo env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        "$BIN" -m "$GGUF" \
        --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
        -ngl 999 -fa on \
        --ctx-size 262144 --parallel 1 \
        --threads 4 --batch-size 2048 --ubatch-size 256 \
        --cache-type-k q4_0 --cache-type-v q4_0 \
        --k-cache-hadamard --v-cache-hadamard \
        --cache-ram 40960 \
        --no-context-shift --ctx-checkpoints 64 \
        --mlockall --rt-prio 50 --cpu-mask 0xF0 \
        --port "$PORT" --host 127.0.0.1 \
        > "$OUT/server.log" 2>&1 &
    SERVER_PID=$!
    for i in $(seq 1 120); do
        if curl -fsS --max-time 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

shutdown_server() {
    sudo kill -TERM "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    sleep 3
}

echo "=== PHASE_PERF_R3_FOLLOWUP Phase 1 — R2 inflection sweep ==="
echo "OUT=$OUT"
echo "Config: ctx=262144 np=1 ubatch=256 + RT chain"
echo

start_server || { echo "SERVER FAILED — see $OUT/server.log"; exit 1; }
echo "Server up on port $PORT"

# CSV header
echo "depth,rep,prompt_n,pp_tps,tg_tps,predicted_n" > "$OUT/results.csv"

trap 'shutdown_server' EXIT

for d in "${DEPTHS[@]}"; do
    PROMPT_FILE="$PROMPT_DIR/prompt-${d}.txt"
    if [[ ! -f "$PROMPT_FILE" ]]; then
        echo "MISSING prompt file: $PROMPT_FILE" >&2
        continue
    fi
    python3 -c "import json; print(json.dumps({'prompt': open('$PROMPT_FILE').read(), 'n_predict':128, 'temperature':0.0, 'top_p':1.0, 'top_k':0, 'seed':1, 'cache_prompt':False}))" > "$OUT/req-d${d}.json"

    echo
    echo "--- depth=$d ---"
    for r in $(seq 1 $N_REPS); do
        curl -sf -o "$OUT/resp-d${d}-r${r}.json" --max-time 300 -X POST \
            "http://127.0.0.1:$PORT/completion" \
            -H "Content-Type: application/json" \
            -d @"$OUT/req-d${d}.json"
        rc=$?
        if [[ $rc -ne 0 ]]; then
            echo "  d=$d r=$r: curl rc=$rc"
            echo "$d,$r,NA,NA,NA,NA" >> "$OUT/results.csv"
            continue
        fi
        python3 -c "
import json
d = json.load(open('$OUT/resp-d${d}-r${r}.json'))
t = d.get('timings', {})
pp = t.get('prompt_per_second', 0)
tg = t.get('predicted_per_second', 0)
pn = t.get('prompt_n', 0)
gn = t.get('predicted_n', 0)
print(f'  d=$d r=$r: PP={pp:.1f} t/s TG={tg:.2f} t/s n_pp={pn} n_tg={gn}')
print(f'$d,$r,{pn},{pp:.2f},{tg:.3f},{gn}', file=open('$OUT/results.csv', 'a'))
" 2>&1
    done
done

echo
echo "=== Sweep complete — results.csv ==="
cat "$OUT/results.csv"

# Summary: mean TG per depth
echo
echo "=== Mean TG per depth ==="
python3 -c "
import csv
from collections import defaultdict
rows = list(csv.DictReader(open('$OUT/results.csv')))
by_depth = defaultdict(list)
for r in rows:
    try:
        by_depth[int(r['depth'])].append(float(r['tg_tps']))
    except (ValueError, KeyError):
        pass
print('depth | mean_TG | min_TG | max_TG | n')
print('-----+--------+--------+--------+---')
for d in sorted(by_depth):
    vals = by_depth[d]
    print(f'{d:>5} | {sum(vals)/len(vals):>6.2f} | {min(vals):>6.2f} | {max(vals):>6.2f} | {len(vals)}')
"
