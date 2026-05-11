#!/usr/bin/env bash
# Run a short bench with the IK_PRINT_TIMING build and capture histogram.
set -uo pipefail

BIN=/home/llm/yarn-agentic/ik_llama.cpp/build-timing/bin/llama-server
MODEL=/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf
TEMPLATE=/home/llm/profiles/qwen36-fixed-template.jinja
CORPUS=/home/llm/yarn-agentic/scripts/agentic-multiturn-corpus.json
PORT=18181
LOG=/home/llm/yarn-agentic/data/bench-timing-C.serverlog

pkill -f "llama-server" 2>/dev/null || true
sleep 3

LLAMA_MTP_INLINE_KV=1 "$BIN" \
    -m "$MODEL" \
    --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
    -ngl 999 -fa on \
    -mtp --draft 3 \
    --ctx-size 262144 --parallel 1 --threads 16 \
    --batch-size 2048 --ubatch-size 512 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --k-cache-hadamard --v-cache-hadamard \
    --no-context-shift --jinja \
    --chat-template-file "$TEMPLATE" \
    --metrics --port "$PORT" --host 127.0.0.1 \
    > "$LOG" 2>&1 &
SRV_PID=$!

for i in $(seq 1 240); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then break; fi
    sleep 0.5
done

if ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "server failed to start"
    kill -9 "$SRV_PID" 2>/dev/null || true
    exit 1
fi

# warmup
curl -fsS -H "Content-Type: application/json" -d '{
    "messages":[{"role":"user","content":"warmup"}],
    "max_tokens":8,"temperature":0,"stream":false
}' "http://127.0.0.1:${PORT}/v1/chat/completions" > /dev/null

payload=$(/home/llm/venv/bin/python -c "
import json
with open('$CORPUS') as f:
    c = json.load(f)
print(json.dumps({
    'messages': c['messages'],
    'max_tokens': 200,
    'temperature': 0,
    'stream': False,
    'cache_prompt': False,
}))")

resp=$(curl -fsS --max-time 600 -H "Content-Type: application/json" -d "$payload" "http://127.0.0.1:${PORT}/v1/chat/completions")
echo "$resp" | /home/llm/venv/bin/python -c "
import sys, json
d=json.load(sys.stdin); t=d.get('timings',{})
print(f'tg={t.get(\"predicted_per_second\",0):.2f} t/s predicted_n={t.get(\"predicted_n\",0)}')" || echo "(parse failed)"

kill -TERM "$SRV_PID" 2>/dev/null || true
sleep 2
kill -9 "$SRV_PID" 2>/dev/null || true

echo ""
echo "===== HIT/MISS histogram (last 5) ====="
grep "can_reuse_graph: HIT" "$LOG" | tail -5
echo ""
echo "===== build_graph + sched_alloc_graph counts ====="
echo -n "  build_graph instances:        "
grep -c "build_graph(\.\.\.)" "$LOG" || true
echo -n "  sched_alloc_graph instances:  "
grep -c "sched_alloc_graph(\.\.\.)" "$LOG" || true
echo ""
echo "Server log: $LOG"
