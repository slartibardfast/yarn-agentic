#!/usr/bin/env bash
# Compare output text from GraphCache build vs the baseline (no-fix
# would require checkout — but we can at least sanity-check that
# A nomtp output and C mtp output are coherent and align on the
# initial prefix).
set -uo pipefail

BIN=/home/llm/yarn-agentic/ik_llama.cpp/build/bin/llama-server
MODEL=/opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf
TEMPLATE=/home/llm/profiles/qwen36-fixed-template.jinja
CORPUS=/home/llm/yarn-agentic/scripts/agentic-multiturn-corpus.json
PORT=18181

cleanup() { pkill -f "llama-server.*$PORT" 2>/dev/null || true; sleep 3; }

run_and_capture() {
    local label=$1
    local mtp_flag=$2
    local extra_env=$3

    cleanup
    local logfile="/tmp/parity-${label}.log"
    local outfile="/tmp/parity-${label}.txt"
    rm -f "$logfile" "$outfile"

    # shellcheck disable=SC2086
    env $extra_env "$BIN" \
        -m "$MODEL" --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
        -ngl 999 -fa on $mtp_flag \
        --ctx-size 262144 --parallel 1 --threads 16 \
        --batch-size 2048 --ubatch-size 512 \
        --cache-type-k q4_0 --cache-type-v q4_0 \
        --k-cache-hadamard --v-cache-hadamard \
        --no-context-shift --jinja \
        --chat-template-file "$TEMPLATE" \
        --metrics --port "$PORT" --host 127.0.0.1 \
        > "$logfile" 2>&1 &
    local pid=$!

    for i in $(seq 1 240); do
        if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then break; fi
        sleep 0.5
    done

    payload=$(/home/llm/venv/bin/python -c "
import json
with open('$CORPUS') as f:
    c = json.load(f)
print(json.dumps({
    'messages': c['messages'],
    'max_tokens': 128,
    'temperature': 0,
    'stream': False,
    'cache_prompt': False,
}))")

    resp=$(curl -fsS --max-time 600 -H "Content-Type: application/json" \
                -d "$payload" "http://127.0.0.1:${PORT}/v1/chat/completions")
    echo "$resp" | /home/llm/venv/bin/python -c "
import sys, json
d = json.load(sys.stdin)
print(d['choices'][0]['message']['content'])" > "$outfile"

    kill -TERM "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    echo "  output -> $outfile ($(wc -c < $outfile) chars)"
}

echo "=== capturing outputs (greedy temp=0, max_tokens=128) ==="
run_and_capture "A_nomtp" "-no-mtp" ""
run_and_capture "C_mtp_d3_ikv" "-mtp --draft 3" "LLAMA_MTP_INLINE_KV=1"

cleanup
echo ""
echo "=== first 400 chars: A_nomtp ==="
head -c 400 /tmp/parity-A_nomtp.txt
echo ""
echo ""
echo "=== first 400 chars: C_mtp_d3_ikv ==="
head -c 400 /tmp/parity-C_mtp_d3_ikv.txt
echo ""
echo ""
echo "=== diff (greedy decoding should produce identical token sequences) ==="
diff /tmp/parity-A_nomtp.txt /tmp/parity-C_mtp_d3_ikv.txt > /tmp/parity-diff.txt
diff_lines=$(wc -l < /tmp/parity-diff.txt)
if [[ "$diff_lines" -eq 0 ]]; then
    echo "IDENTICAL — full parity"
else
    echo "DIFF ($diff_lines lines):"
    head -50 /tmp/parity-diff.txt
fi
