#!/usr/bin/env bash
# perf-r3-followup-phase3.sh — Phase 3 of PHASE_PERF_R3_FOLLOWUP.
#
# R1 ctx-allocation tax decomposition + T5.9 paged-KV effectiveness
# sub-test. Same 200t prompt at every config; the only thing that varies
# is --ctx-size and the (--cache-ram, --ctx-checkpoints) pair.
#
# Primary sweep: ctx in {8192, 32768, 131072, 262144} with production
# allocator (--cache-ram 40960 --ctx-checkpoints 64).
#
# T5.9 sub-test: at ctx=8192 and ctx=262144 only, also run with
# --cache-ram 0 --ctx-checkpoints 0 to measure whether the paged-KV
# backing is paying back the allocation tax for sparse usage.
#
# Usage:
#     OUT=<dir> bash perf-r3-followup-phase3.sh

set -uo pipefail

BIN=/home/dconnolly/yarn-agentic/ik_llama.cpp/build/bin/llama-server
GGUF=/opt/models/recast-out/qwen3.6-27b-V-F1.T1.lm_head-f16.gguf
PROMPT_FILE=/tmp/perf-r3-prompts/prompt-210.txt
OUT="${OUT:-/tmp/perf-r3-followup-phase3-$(date -u +%Y%m%dT%H%M%S)}"
PORT="${PORT:-18295}"
mkdir -p "$OUT"

N_REPS=3

# Pre-write request body (200t prompt, deterministic, n_predict=128).
python3 -c "import json; print(json.dumps({'prompt': open('$PROMPT_FILE').read(), 'n_predict':128, 'temperature':0.0, 'top_p':1.0, 'top_k':0, 'seed':1, 'cache_prompt':False}))" > "$OUT/req-body.json"

start_server() {
    local label="$1"
    local ctx="$2"
    local cache_ram="$3"
    local ctx_checkpoints="$4"
    sudo env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        "$BIN" -m "$GGUF" \
        --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
        -ngl 999 -fa on \
        --ctx-size "$ctx" --parallel 1 \
        --threads 4 --batch-size 2048 --ubatch-size 256 \
        --cache-type-k q4_0 --cache-type-v q4_0 \
        --k-cache-hadamard --v-cache-hadamard \
        --cache-ram "$cache_ram" \
        --no-context-shift --ctx-checkpoints "$ctx_checkpoints" \
        --mlockall --rt-prio 50 --cpu-mask 0xF0 \
        --port "$PORT" --host 127.0.0.1 \
        > "$OUT/$label-server.log" 2>&1 &
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

run_one_config() {
    local label="$1"
    local ctx="$2"
    local cache_ram="$3"
    local ctx_checkpoints="$4"
    echo
    echo "--- $label: ctx=$ctx cache_ram=$cache_ram ctx_checkpoints=$ctx_checkpoints ---"
    if ! start_server "$label" "$ctx" "$cache_ram" "$ctx_checkpoints"; then
        echo "$label SERVER FAILED — see $OUT/$label-server.log"
        return 1
    fi
    for r in $(seq 1 $N_REPS); do
        curl -sf -o "$OUT/$label-resp$r.json" --max-time 300 -X POST \
            "http://127.0.0.1:$PORT/completion" \
            -H "Content-Type: application/json" \
            -d @"$OUT/req-body.json"
        rc=$?
        if [[ $rc -ne 0 ]]; then
            echo "  $label rep$r: curl rc=$rc"
            echo "$ctx,$cache_ram,$ctx_checkpoints,$r,NA,NA,NA" >> "$OUT/results.csv"
            continue
        fi
        python3 -c "
import json
d = json.load(open('$OUT/$label-resp$r.json'))
t = d.get('timings', {})
pp = t.get('prompt_per_second', 0)
tg = t.get('predicted_per_second', 0)
pn = t.get('prompt_n', 0)
gn = t.get('predicted_n', 0)
print(f'  $label r$r: PP={pp:.1f} t/s TG={tg:.2f} t/s n_pp={pn} n_tg={gn}')
print(f'$ctx,$cache_ram,$ctx_checkpoints,$r,{pn},{pp:.2f},{tg:.3f}', file=open('$OUT/results.csv', 'a'))
" 2>&1
    done
    shutdown_server
}

echo "=== PHASE_PERF_R3_FOLLOWUP Phase 3 — R1 ctx tax + T5.9 effectiveness ==="
echo "OUT=$OUT"
echo "Prompt: $PROMPT_FILE (200t target)"
echo "Per-config: $N_REPS reps, N_PREDICT=128, greedy"

echo "ctx,cache_ram,ctx_checkpoints,rep,prompt_n,pp_tps,tg_tps" > "$OUT/results.csv"

# --- Primary sweep: production allocator at 4 ctx points ---
echo
echo "## Primary sweep (--cache-ram 40960 --ctx-checkpoints 64)"
run_one_config "P-ctx8k"    8192   40960 64
run_one_config "P-ctx32k"   32768  40960 64
run_one_config "P-ctx128k"  131072 40960 64
run_one_config "P-ctx256k"  262144 40960 64

# --- T5.9 effectiveness sub-test: cache-ram=0 at endpoints ---
echo
echo "## T5.9 effectiveness sub-test (--cache-ram 0 --ctx-checkpoints 0)"
run_one_config "Z-ctx8k"    8192   0 0
run_one_config "Z-ctx256k"  262144 0 0

echo
echo "=== All configs done — results.csv ==="
cat "$OUT/results.csv"

# Summary tables
echo
echo "=== R1 primary sweep — mean TG by ctx ==="
python3 -c "
import csv
from collections import defaultdict
rows = list(csv.DictReader(open('$OUT/results.csv')))
prod = defaultdict(list)
zero = defaultdict(list)
for r in rows:
    try:
        ctx = int(r['ctx']); tg = float(r['tg_tps'])
        if int(r['cache_ram']) > 0:
            prod[ctx].append(tg)
        else:
            zero[ctx].append(tg)
    except (ValueError, KeyError):
        pass

print('R1 primary sweep (cache-ram=40960, ctx-checkpoints=64):')
print('  ctx     mean_TG   min     max    n')
baseline = None
for c in sorted(prod):
    v = prod[c]
    m = sum(v)/len(v)
    if baseline is None: baseline = m
    delta = (m - baseline) / baseline * 100
    print(f'  {c:>6}  {m:>6.2f}   {min(v):>6.2f}  {max(v):>6.2f}  {len(v)}   vs ctx=8k: {delta:+.1f}%')

print()
print('T5.9 effectiveness (cache-ram=0, ctx-checkpoints=0):')
print('  ctx     mean_TG   min     max    n')
for c in sorted(zero):
    v = zero[c]
    m = sum(v)/len(v)
    pv = prod.get(c, [])
    if pv:
        pm = sum(pv)/len(pv)
        delta = (pm - m) / m * 100
        print(f'  {c:>6}  {m:>6.2f}   {min(v):>6.2f}  {max(v):>6.2f}  {len(v)}   T5.9 payback vs zero: {delta:+.1f}%')
    else:
        print(f'  {c:>6}  {m:>6.2f}   {min(v):>6.2f}  {max(v):>6.2f}  {len(v)}')
"
