#!/usr/bin/env bash
# T4.7.M3-staggered — NP=8 perf-gate bench with staggered prompt arrivals.
#
# Identical server configuration to bench-t3.8-m3.sh (NP=8,
# Hadamard K/V, --no-context-shift, locked clocks) but the 8 prompts
# fire at staggered offsets (default 5s apart) instead of all at t=0.
# This exercises the T4 chunked-prefill admission path where a new
# slot arrives while other slots are decoding — the case the pre-T4
# active_pp_slot_id PrefillSerialisationGate held idle, and that
# Sarathi-Serve admission unlocks.
#
# Usage:
#   bash scripts/bench-t4-m3-staggered.sh [OUTDIR]
#
# Env knobs:
#   N_PREDICT       default 200
#   N_PARALLEL      default 8
#   N_RUNS          default 3
#   PORT            default 18182
#   CTX_PER_SLOT    default 4096
#   STAGGER_S       default 5 (seconds between prompt arrivals)
#   MODEL           default the production lm_head-f16 GGUF
#   K_BUDGET        default 0 (== n_ubatch = 512)
#
# Output (in OUTDIR):
#   server-runN.log     server stdout/stderr
#   run-N.json          per-run aggregate result with per-slot timing
#   summary.txt         mean ± stddev across N_RUNS

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUTDIR="${1:-/home/llm/yarn-agentic/data/t4-c1-staggered-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUTDIR"

N_PREDICT="${N_PREDICT:-200}"
N_PARALLEL="${N_PARALLEL:-8}"
N_RUNS="${N_RUNS:-3}"
PORT="${PORT:-18182}"
CTX_PER_SLOT="${CTX_PER_SLOT:-4096}"
STAGGER_S="${STAGGER_S:-5}"
K_BUDGET="${K_BUDGET:-0}"
MODEL="${MODEL:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.lm_head-f16.gguf}"
SERVER="${SERVER:-/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-server}"
PROMPT="${PROMPT:-The quick brown fox jumps over the lazy dog. }"

CTX_TOTAL=$(( CTX_PER_SLOT * N_PARALLEL ))

existing_consumers=$(pgrep -x "llama-server|llama-batched-bench" 2>/dev/null || true)
if [ -n "$existing_consumers" ]; then
    echo "FAIL: existing GPU consumer detected — aborting." >&2
    ps -fp $existing_consumers >&2
    exit 2
fi

start_server() {
    local logfile=$1
    "$SERVER" \
        -m "$MODEL" \
        --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
        -ngl 999 -fa on --threads 16 \
        --batch-size 2048 --ubatch-size 512 \
        --cache-type-k q4_0 --cache-type-v q4_0 \
        --k-cache-hadamard --v-cache-hadamard \
        --ctx-size "$CTX_TOTAL" --parallel "$N_PARALLEL" \
        --prefill-chunk-budget "$K_BUDGET" \
        --no-context-shift \
        --port "$PORT" --host 127.0.0.1 \
        > "$logfile" 2>&1 &
    echo $!
}

wait_for_health() {
    local pid=$1
    for i in $(seq 1 240); do
        if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then return 0; fi
        if ! kill -0 "$pid" 2>/dev/null; then echo "server died" >&2; return 1; fi
        sleep 0.5
    done
    echo "server health timeout" >&2
    return 1
}

fire_staggered() {
    local outfile=$1
    local n_pred=$2
    local n_par=$3
    local stagger=$4
    local prompt_text=$5
    /home/llm/venv/bin/python - <<PYEOF
import json
import urllib.request
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

n_pred = $n_pred
n_par = $n_par
stagger = $stagger
prompt = $(python3 -c "import json; print(json.dumps('$prompt_text'))")
url = "http://127.0.0.1:$PORT/v1/completions"

def one(slot):
    # Stagger by slot index — slot N fires at t = slot * stagger seconds
    # from the bench wall clock origin.
    if slot > 0:
        time.sleep(slot * stagger)
    payload = {
        "prompt": prompt,
        "n_predict": n_pred,
        "temperature": 0.0,
        "seed": slot,
        "cache_prompt": False,
        "stream": False,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type":"application/json"})
    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=900)
    obj = json.loads(resp.read())
    t1 = time.time()
    tokens = obj.get("tokens_predicted", obj.get("usage", {}).get("completion_tokens", 0))
    return {"slot": slot, "t0": t0, "t1": t1, "tokens": tokens}

t_start = time.time()
results = []
with ThreadPoolExecutor(max_workers=n_par) as ex:
    futures = [ex.submit(one, s) for s in range(n_par)]
    for f in as_completed(futures):
        results.append(f.result())
t_end = time.time()

total_tokens = sum(r["tokens"] for r in results)
# Wall = from earliest t0 (first arrival) to latest t1 (last completion).
# Aggregate t/s under staggered arrival is the harmonic-style throughput
# of the system over the full observation window.
wall = max(r["t1"] for r in results) - min(r["t0"] for r in results)
agg_tps = total_tokens / wall if wall > 0 else 0.0

out = {
    "n_predict": n_pred,
    "n_parallel": n_par,
    "stagger_s": stagger,
    "wall_s": wall,
    "total_tokens": total_tokens,
    "aggregate_tps": agg_tps,
    "per_slot": results,
}
with open("$outfile", "w") as f:
    json.dump(out, f, indent=2)
print(f"AGGREGATE_TPS={agg_tps:.2f}  wall={wall:.2f}s  total_tokens={total_tokens}")
PYEOF
}

stop_server() {
    local pid=$1
    kill -INT "$pid" 2>/dev/null || true
    for i in $(seq 1 60); do
        if ! kill -0 "$pid" 2>/dev/null; then return 0; fi
        sleep 0.5
    done
    kill -KILL "$pid" 2>/dev/null || true
}

declare -a agg_tps_array=()

for r in $(seq 1 "$N_RUNS"); do
    echo "=== M3-staggered run $r/$N_RUNS (stagger=${STAGGER_S}s) ==="
    server_log="$OUTDIR/server-run$r.log"
    result_json="$OUTDIR/run-$r.json"
    pid=$(start_server "$server_log")
    if ! wait_for_health "$pid"; then
        echo "FAIL: server didn't come healthy in run $r" >&2
        kill -9 "$pid" 2>/dev/null || true
        wait 2>/dev/null
        continue
    fi

    # warmup
    curl -fsS -H "Content-Type: application/json" \
        -d '{"prompt":"warm","n_predict":4,"temperature":0,"cache_prompt":false}' \
        "http://127.0.0.1:${PORT}/v1/completions" > /dev/null

    fire_staggered "$result_json" "$N_PREDICT" "$N_PARALLEL" "$STAGGER_S" "$PROMPT"
    cat "$result_json" | head -8 || true

    stop_server "$pid"
    wait 2>/dev/null || true

    if [ -f "$result_json" ]; then
        tps=$(grep -o '"aggregate_tps": [0-9.]*' "$result_json" | awk '{print $2}')
        agg_tps_array+=("$tps")
    fi
done

echo
echo "=== summary ==="
{
    echo "N_RUNS=$N_RUNS"
    echo "N_PREDICT=$N_PREDICT"
    echo "N_PARALLEL=$N_PARALLEL"
    echo "STAGGER_S=$STAGGER_S"
    echo "K_BUDGET=$K_BUDGET (0 == n_ubatch)"
    echo "agg_tps_per_run: ${agg_tps_array[*]}"
    if [ "${#agg_tps_array[@]}" -gt 0 ]; then
        /home/llm/venv/bin/python - <<PYEOF
import statistics
vals = [$(IFS=,; echo "${agg_tps_array[*]}")]
m = statistics.mean(vals)
sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
cv = (sd / m * 100) if m > 0 else 0.0
print(f"mean={m:.2f} stddev={sd:.3f} cv={cv:.2f}%")
PYEOF
    fi
    echo "dispatch_multi_seq_count lines from server logs:"
    grep -h "multi-seq dispatch counter\|dispatch counter" "$OUTDIR"/server-run*.log 2>/dev/null | tail -5
    echo "VRAM probe lines (~ggml_backend_cuda_context):"
    grep -h "~ggml_backend_cuda_context" "$OUTDIR"/server-run*.log 2>/dev/null | tail -8
} | tee "$OUTDIR/summary.txt"
