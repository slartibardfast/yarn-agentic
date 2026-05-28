#!/usr/bin/env bash
# perf-r3-lever-pulls.sh — Phase G of PHASE_PERF_R3_NP1.
#
# Tests four lever-pulls relative to the A2 baseline (PD4 shape on
# current binary + RT flags). Each lever runs ×3 reps, no nsys (fast),
# greedy decode. Output: per-config t/s and a comparison table.
#
# Usage:
#     OUT=<dir> bash perf-r3-lever-pulls.sh

set -uo pipefail

BIN=/home/dconnolly/yarn-agentic/ik_llama.cpp/build/bin/llama-server
GGUF=/opt/models/recast-out/qwen3.6-27b-V-F1.T1.lm_head-f16.gguf
PROMPT_FILE=/tmp/perf-r3-prompts/prompt-210.txt
OUT="${OUT:-/tmp/perf-r3-lever-$(date -u +%Y%m%dT%H%M%S)}"
mkdir -p "$OUT"

# Pre-write request body (small prompt fits in argv easily, but keep
# consistent with E5/E6 method).
python3 -c "import json; print(json.dumps({'prompt': open('$PROMPT_FILE').read(), 'n_predict':128, 'temperature':0.0, 'top_p':1.0, 'top_k':0, 'seed':1, 'cache_prompt':False}))" > "$OUT/req-body.json"

start_server() {
    local label="$1"
    shift
    local extra_args="$*"
    sudo env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        $extra_env \
        "$BIN" -m "$GGUF" \
        --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
        -ngl 999 -fa on \
        --ctx-size 8192 --parallel 1 \
        --threads 4 --batch-size 2048 --ubatch-size 512 \
        --cache-type-k q4_0 --cache-type-v q4_0 \
        --k-cache-hadamard --v-cache-hadamard \
        --no-context-shift --ctx-checkpoints 3 \
        --mlockall --rt-prio 50 --cpu-mask 0xF0 \
        $extra_args \
        --port 18292 --host 127.0.0.1 \
        > "$OUT/$label-server.log" 2>&1 &
    SERVER_PID=$!
    for i in $(seq 1 60); do
        if curl -fsS --max-time 1 http://127.0.0.1:18292/health >/dev/null 2>&1; then return 0; fi
        sleep 1
    done
    return 1
}

shutdown_server() {
    sudo kill -TERM "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    sleep 2
}

run_reps() {
    local label="$1"
    local n_reps="${2:-3}"
    for r in $(seq 1 "$n_reps"); do
        curl -sf -o "$OUT/$label-resp$r.json" --max-time 120 -X POST \
            http://127.0.0.1:18292/completion \
            -H "Content-Type: application/json" \
            -d @"$OUT/req-body.json"
        python3 -c "
import json
d = json.load(open('$OUT/$label-resp$r.json'))
t = d.get('timings', {})
print(f'  $label rep$r: PP={t.get(\"prompt_per_second\",0):.1f} t/s TG={t.get(\"predicted_per_second\",0):.2f} t/s n_pp={t.get(\"prompt_n\")}')
" 2>/dev/null || echo "  $label rep$r: parse failed"
    done
}

echo "=== G — Lever-pull experiments ==="
echo "OUT=$OUT"

# G0 — baseline reference (matches A1 shape; sanity)
echo
echo "--- G0 — baseline (ctx 8k, default ubatch 512) ---"
extra_env=""
start_server G0 || { echo "G0 server failed"; exit 1; }
run_reps G0 3
shutdown_server

# G1 — CUDA graph cache OFF
echo
echo "--- G1 — GGML_CUDA_GRAPH_MAX=0 (graph cache disabled) ---"
extra_env="GGML_CUDA_GRAPH_MAX=0"
start_server G1 || { echo "G1 server failed"; exit 1; }
run_reps G1 3
shutdown_server

# G2a — ubatch 256
echo
echo "--- G2a — ubatch 256 ---"
extra_env=""
start_server G2a --ubatch-size 256 || { echo "G2a failed"; exit 1; }
run_reps G2a 3
shutdown_server

# G2b — ubatch 1024
echo
echo "--- G2b — ubatch 1024 ---"
extra_env=""
start_server G2b --ubatch-size 1024 || { echo "G2b failed"; exit 1; }
run_reps G2b 3
shutdown_server

# G3 — threads 2 (drop from 4 to 2)
echo
echo "--- G3 — --threads 2 (within 0xF0 mask) ---"
extra_env=""
start_server G3 --threads 2 || { echo "G3 failed"; exit 1; }
run_reps G3 3
shutdown_server

# G4 — cuBLAS workspace 16384 (vs production 4096:8)
echo
echo "--- G4 — CUBLAS_WORKSPACE_CONFIG=:16384:8 ---"
extra_env="CUBLAS_WORKSPACE_CONFIG=:16384:8"
start_server G4 || { echo "G4 failed"; exit 1; }
run_reps G4 3
shutdown_server

echo
echo "=== G summary ==="
for cfg in G0 G1 G2a G2b G3 G4; do
    for r in 1 2 3; do
        rf="$OUT/$cfg-resp$r.json"
        [ ! -s "$rf" ] && continue
        python3 -c "
import json
d = json.load(open('$rf'))
t = d.get('timings', {})
print(f\"  $cfg rep$r: PP={t.get('prompt_per_second',0):6.1f}  TG={t.get('predicted_per_second',0):6.2f}\")
" 2>/dev/null
    done
done
