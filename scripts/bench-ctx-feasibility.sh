#!/usr/bin/env bash
# T5.9 — high-ctx feasibility + admission gate bench.
#
# One-command driver for GP5.9.feasibility. Stands up a sibling
# llama-server with --ctx-size CTX_PER_SLOT × N_PARALLEL and the
# paged-BACKING pool sized to a VRAM-fitting capacity via
# --kv-pool-blocks POOL_BLOCKS, then fires N_PROMPTS concurrent
# completions to over-subscribe the pool.
#
# Pass criteria (binding):
#   1. At least one /v1/completions returns HTTP 200 with a non-empty
#      completion (feasibility — paged BACKING successfully allocates
#      KV at this ctx).
#   2. At least one /v1/completions returns HTTP 503 + Retry-After
#      (admission gate — pool exhaustion produces a clean rejection).
#   3. /health returns 200 after the burst (server stays up).
#
# Usage:
#   bash scripts/bench-ctx-feasibility.sh [OUTDIR]
#
# Env knobs:
#   CTX_PER_SLOT    default 1048576 (1M)
#   N_PARALLEL      default 8
#   POOL_BLOCKS     default $(( CTX_PER_SLOT * N_PARALLEL / 64 / 16 ))
#                   = 1/16th nominal — guaranteed admission rejection
#   N_PROMPTS       default 16 (> N_PARALLEL to exercise admission)
#   N_PREDICT       default 64
#   PORT            default 18193
#   MODEL           default the production lm_head-f16 GGUF
#   PROFILE         default qwen36-27b-x2-dflash-bigctx.sh
#                   (sibling profile; production untouched)
#
# Output (in OUTDIR):
#   server.log         server stdout/stderr for the single bench session
#   responses.json     per-request status code, latency, tokens
#   health-after.txt   /health response body + http status post-burst
#   summary.txt        gate verdict (PASS/FAIL) + counts per status

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUTDIR="${1:-/home/llm/yarn-agentic/data/t5.9-ctx-feasibility-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUTDIR"

CTX_PER_SLOT="${CTX_PER_SLOT:-1048576}"
N_PARALLEL="${N_PARALLEL:-8}"
DEFAULT_POOL_BLOCKS=$(( CTX_PER_SLOT * N_PARALLEL / 64 / 16 ))
POOL_BLOCKS="${POOL_BLOCKS:-$DEFAULT_POOL_BLOCKS}"
N_PROMPTS="${N_PROMPTS:-16}"
N_PREDICT="${N_PREDICT:-64}"
PORT="${PORT:-18193}"
MODEL="${MODEL:-/opt/models/recast-out/qwen3.6-27b-V-F1.T1.lm_head-f16.gguf}"
PROFILE="${PROFILE:-/home/llm/profiles/qwen36-27b-x2-dflash-bigctx.sh}"
PROMPT="${PROMPT:-The quick brown fox jumps over the lazy dog. }"

CTX_TOTAL=$(( CTX_PER_SLOT * N_PARALLEL ))

existing_consumers=$(pgrep -x "llama-server|llama-batched-bench" 2>/dev/null || true)
if [ -n "$existing_consumers" ]; then
    echo "FAIL: existing GPU consumer detected — aborting." >&2
    ps -fp $existing_consumers >&2
    exit 2
fi

if [ ! -x "$PROFILE" ]; then
    echo "FAIL: PROFILE=$PROFILE not executable" >&2
    exit 2
fi

start_server() {
    local logfile=$1
    # Sibling profile exports its own engine + flags; the bench overrides
    # PORT/MODEL/CTX/POOL_BLOCKS at the env-substitution layer by exporting
    # before exec. The profile reads them with default fallbacks.
    export LLAMA_BENCH_PORT="$PORT"
    export LLAMA_BENCH_MODEL="$MODEL"
    export LLAMA_BENCH_CTX="$CTX_TOTAL"
    export LLAMA_BENCH_PARALLEL="$N_PARALLEL"
    export LLAMA_BENCH_POOL_BLOCKS="$POOL_BLOCKS"
    bash "$PROFILE" > "$logfile" 2>&1 &
    echo $!
}

wait_for_health() {
    local pid=$1
    for i in $(seq 1 360); do
        if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then return 0; fi
        if ! kill -0 "$pid" 2>/dev/null; then echo "server died" >&2; return 1; fi
        sleep 0.5
    done
    echo "server health timeout" >&2
    return 1
}

fire_oversubscribe() {
    local outfile=$1
    /home/llm/venv/bin/python - <<PYEOF
import json
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

n_prompts  = $N_PROMPTS
n_predict  = $N_PREDICT
url        = "http://127.0.0.1:$PORT/v1/completions"
prompt     = $(python3 -c "import json,sys; print(json.dumps('$PROMPT'))")

def one(idx):
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "seed": idx,
        "cache_prompt": False,
        "stream": False,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=900)
        t1 = time.time()
        obj = json.loads(resp.read())
        tokens = obj.get("tokens_predicted",
                        obj.get("usage", {}).get("completion_tokens", 0))
        return {
            "idx": idx,
            "status": resp.status,
            "retry_after": resp.headers.get("Retry-After"),
            "tokens": tokens,
            "latency_s": t1 - t0,
            "content_chars": len((obj.get("choices") or [{}])[0].get("text", "") or "")
                              if obj.get("choices") else len(obj.get("content", "") or ""),
        }
    except urllib.error.HTTPError as e:
        t1 = time.time()
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:512]
        except Exception:
            pass
        return {
            "idx": idx,
            "status": e.code,
            "retry_after": e.headers.get("Retry-After") if e.headers else None,
            "tokens": 0,
            "latency_s": t1 - t0,
            "error_body": body,
        }
    except Exception as e:
        return {"idx": idx, "status": 0, "tokens": 0, "error": repr(e)}

t_start = time.time()
results = []
with ThreadPoolExecutor(max_workers=n_prompts) as ex:
    futures = [ex.submit(one, i) for i in range(n_prompts)]
    for f in as_completed(futures):
        results.append(f.result())
t_end = time.time()

results.sort(key=lambda r: r["idx"])
out = {
    "n_prompts": n_prompts,
    "wall_s": t_end - t_start,
    "per_request": results,
}
with open("$outfile", "w") as f:
    json.dump(out, f, indent=2)

status_counts = {}
for r in results:
    status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
print(f"STATUS_COUNTS={status_counts}  wall={t_end - t_start:.2f}s")
PYEOF
}

stop_server() {
    local pid=$1
    kill -INT "$pid" 2>/dev/null || true
    for i in $(seq 1 90); do
        if ! kill -0 "$pid" 2>/dev/null; then return 0; fi
        sleep 0.5
    done
    kill -KILL "$pid" 2>/dev/null || true
}

echo "=== T5.9 ctx-feasibility bench ==="
echo "CTX_PER_SLOT=$CTX_PER_SLOT  N_PARALLEL=$N_PARALLEL  CTX_TOTAL=$CTX_TOTAL"
echo "POOL_BLOCKS=$POOL_BLOCKS  (= $(( POOL_BLOCKS * 64 )) physical token capacity)"
echo "N_PROMPTS=$N_PROMPTS  (oversubscription = $(( N_PROMPTS - N_PARALLEL )))"
echo "OUTDIR=$OUTDIR"

server_log="$OUTDIR/server.log"
result_json="$OUTDIR/responses.json"
health_post="$OUTDIR/health-after.txt"

pid=$(start_server "$server_log")
if ! wait_for_health "$pid"; then
    echo "FAIL: server didn't come healthy" >&2
    kill -9 "$pid" 2>/dev/null || true
    wait 2>/dev/null
    echo "GATE_VERDICT=FAIL_BOOTSTRAP" | tee "$OUTDIR/summary.txt"
    exit 1
fi

fire_oversubscribe "$result_json"

# Post-burst /health probe
{
    code=$(curl -s -o "$health_post.body" -w "%{http_code}" "http://127.0.0.1:${PORT}/health" || echo "000")
    echo "http_code=$code"
    cat "$health_post.body" 2>/dev/null
} > "$health_post"

stop_server "$pid"
wait 2>/dev/null || true

# Verdict
verdict_py=$(/home/llm/venv/bin/python - <<PYEOF
import json
with open("$result_json") as f:
    obj = json.load(f)
rs = obj["per_request"]
n_200 = sum(1 for r in rs if r["status"] == 200)
n_503 = sum(1 for r in rs if r["status"] == 503)
n_other = sum(1 for r in rs if r["status"] not in (200, 503))
has_200 = n_200 >= 1
has_503 = n_503 >= 1
has_retry_after = any(r["status"] == 503 and r.get("retry_after") for r in rs)
print(f"n_200={n_200} n_503={n_503} n_other={n_other} has_retry_after={int(has_retry_after)}")
gate_a = has_200
gate_b = has_503 and has_retry_after
print(f"gate_a_feasibility={int(gate_a)} gate_b_admission_503_with_retry_after={int(gate_b)}")
PYEOF
)

health_code=$(grep -oE 'http_code=[0-9]+' "$health_post" | head -1 | sed 's/http_code=//')

{
    echo "=== T5.9 ctx-feasibility bench — summary ==="
    echo "CTX_PER_SLOT=$CTX_PER_SLOT"
    echo "N_PARALLEL=$N_PARALLEL"
    echo "POOL_BLOCKS=$POOL_BLOCKS  (physical_capacity_tokens=$(( POOL_BLOCKS * 64 )))"
    echo "N_PROMPTS=$N_PROMPTS"
    echo "$verdict_py"
    echo "health_after_http_code=$health_code"
    if echo "$verdict_py" | grep -q "gate_a_feasibility=1" \
       && echo "$verdict_py" | grep -q "gate_b_admission_503_with_retry_after=1" \
       && [ "$health_code" = "200" ]; then
        echo "GATE_VERDICT=PASS"
    else
        echo "GATE_VERDICT=FAIL"
    fi
} | tee "$OUTDIR/summary.txt"
