#!/usr/bin/env bash
# T6.3 1M overnight stress test for qwen36-27b-x1-yarn-1m-mtp.
#
# 7-phase validation of the proposed 1M-Yarn-MTP-single-slot production
# config:
#   1. Boot smoke         (10 min)
#   2. Cold 1M prefill    (15 min)  — measure cold prefill, NIAH @1M
#   3. Cache validation   (30 min)  — 3 requests with shared prefix
#   4. NIAH depth sweep   (~2 hr)   — needle at 10/25/50/75/90% depth
#   5. War & Peace QA     (~1 hr)   — qualitative comprehension
#   6. Stability soak     (~4 hr)   — mixed-length cycle, VRAM watch
#   7. Summary            (10 min)  — aggregate metrics → SUMMARY.md
# Total: ~9 hours.
#
# Pre-requisites:
#   - Clocks locked 1455 MHz (caller responsibility)
#   - Production servicestopped
#   - Coord BUSY claim on both GPUs
#   - data/t6.3-1m-overnight-prep/war-and-peace.txt present (853K Qwen tokens)
#
# Usage:
#   bash scripts/run-t6.3-1m-overnight.sh [OUTDIR]

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
OUTDIR="${1:-$REPO_ROOT/data/t6.3-1m-overnight-$(date +%Y%m%dT%H%M%S)}"
mkdir -p "$OUTDIR"

COORD_DIR="${COORD_DIR:-$REPO_ROOT/coord}"
PORT="${PORT:-8080}"
PROFILE="${PROFILE:-/home/llm/profiles/qwen36-27b-x1-yarn-1m-mtp.sh}"
PREP_DIR="${PREP_DIR:-$REPO_ROOT/data/t6.3-1m-overnight-prep}"
WARMUP_MAX_S="${WARMUP_MAX_S:-600}"

WAR_PEACE="$PREP_DIR/war-and-peace.txt"
[ -f "$WAR_PEACE" ] || { echo "FAIL: $WAR_PEACE not found — run preflight"; exit 2; }

# ---------- GPU coord ----------
claim_gpus() {
    flock -w 5 "$COORD_DIR/gpu-0.lock" -c "echo BUSY > '$COORD_DIR/gpu-0.state'" || { echo "[FAIL] GPU 0 busy"; exit 1; }
    flock -w 5 "$COORD_DIR/gpu-1.lock" -c "echo BUSY > '$COORD_DIR/gpu-1.state'" || { echo "[FAIL] GPU 1 busy"; exit 1; }
}
release_gpus() {
    flock -w 5 "$COORD_DIR/gpu-0.lock" -c "echo IDLE > '$COORD_DIR/gpu-0.state'" 2>/dev/null || true
    flock -w 5 "$COORD_DIR/gpu-1.lock" -c "echo IDLE > '$COORD_DIR/gpu-1.state'" 2>/dev/null || true
}
stop_systemd() { systemctl --user is-active llama-server.service >/dev/null 2>&1 && { systemctl --user stop llama-server.service; sleep 2; }; }
start_systemd() { systemctl --user start llama-server.service; }

wait_for_health() {
    local pid=$1
    local deadline=$((SECONDS + WARMUP_MAX_S))
    until curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
        kill -0 "$pid" 2>/dev/null || return 1
        [ "$SECONDS" -ge "$deadline" ] && return 1
        sleep 5
    done
}

# ---------- HTTP completion helpers ----------
# Fire a /v1/completions with a long prompt; return wall, prefill_est, decode_toks.
# Uses streaming so we can capture TTFT (time to first token).
fire_completion() {
    local label=$1
    local prompt_file=$2
    local max_tokens=$3
    local extra_json=${4:-'""'}
    local outfile="$OUTDIR/${label}.json"
    /home/llm/venv/bin/python3 - "$PORT" "$prompt_file" "$max_tokens" "$outfile" "$extra_json" <<'PYEOF'
import json, sys, time, urllib.request

port, pf, mtok, outfile, extra_json = sys.argv[1:6]
mtok = int(mtok)

with open(pf, 'r') as f:
    prompt = f.read()

payload = {
    "prompt": prompt,
    "n_predict": mtok, "max_tokens": mtok,
    "temperature": 0.0, "seed": 42,
    "cache_prompt": True, "stream": True, "ignore_eos": False,
}
body = json.dumps(payload).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                              data=body, headers={"Content-Type": "application/json"})

t0 = time.time()
ttft = None
toks = []
gen_text = []
try:
    resp = urllib.request.urlopen(req, timeout=3600)
    for raw in resp:
        line = raw.decode('utf-8', errors='replace').strip()
        if not line.startswith('data:'):
            continue
        data = line[5:].strip()
        if data == '[DONE]':
            break
        try:
            obj = json.loads(data)
        except Exception:
            continue
        ch = obj.get('choices') or []
        if ch and ch[0].get('text'):
            if ttft is None:
                ttft = time.time() - t0
            gen_text.append(ch[0]['text'])
            toks.append(time.time())
    t1 = time.time()
except Exception as e:
    print(f"FAIL: {e}", file=sys.stderr)
    json.dump({"error": repr(e)}, open(outfile, 'w'))
    sys.exit(1)

result = {
    "label": "",
    "wall_s": t1 - t0,
    "ttft_s": ttft,
    "decode_s": (t1 - (t0 + (ttft or 0))) if ttft else None,
    "n_predicted": len(gen_text),
    "first_tokens": ''.join(gen_text)[:200],
    "last_tokens":  ''.join(gen_text)[-200:],
}
json.dump(result, open(outfile, 'w'), indent=2)
print(json.dumps({k: v for k, v in result.items()
                  if k not in ("first_tokens","last_tokens")}))
PYEOF
}

vram_snapshot() {
    nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv,noheader \
        > "$OUTDIR/vram-$(date +%s).csv"
}

# ---------- Phase machinery ----------
echo "=== T6.3 1M overnight ===" | tee "$OUTDIR/run.log"
echo "OUTDIR=$OUTDIR" | tee -a "$OUTDIR/run.log"
echo "PROFILE=$(basename "$PROFILE")" | tee -a "$OUTDIR/run.log"
echo "Started $(date -Is)" | tee -a "$OUTDIR/run.log"

stop_systemd
claim_gpus
cleanup() {
    kill -INT "$SERVER_PID" 2>/dev/null || true
    sleep 5
    pkill -INT -x llama-server 2>/dev/null || true
    sleep 2
    pkill -KILL -x llama-server 2>/dev/null || true
    release_gpus
    start_systemd
    echo "Ended $(date -Is)" >> "$OUTDIR/run.log"
}
trap cleanup EXIT INT TERM

# Spawn server
nohup bash "$PROFILE" > "$OUTDIR/server.log" 2>&1 &
SERVER_PID=$!
echo "[server] pid=$SERVER_PID; waiting for /health (timeout ${WARMUP_MAX_S}s)" | tee -a "$OUTDIR/run.log"
if ! wait_for_health "$SERVER_PID"; then
    echo "[FATAL] server did not become healthy" | tee -a "$OUTDIR/run.log"
    tail -100 "$OUTDIR/server.log" | tee -a "$OUTDIR/run.log"
    exit 1
fi
echo "[server] healthy" | tee -a "$OUTDIR/run.log"
vram_snapshot

# ============================
# Phase 1: boot smoke
# ============================
echo "" | tee -a "$OUTDIR/run.log"
echo "=== Phase 1: boot smoke (short prompt) ===" | tee -a "$OUTDIR/run.log"
echo "Tell me about the role of telomeres in cellular aging in two sentences." > "$OUTDIR/phase1-prompt.txt"
fire_completion phase1-boot "$OUTDIR/phase1-prompt.txt" 128 | tee -a "$OUTDIR/run.log"
vram_snapshot

# ============================
# Phase 2: cold 1M prefill — feed War and Peace + a tail question
# ============================
echo "" | tee -a "$OUTDIR/run.log"
echo "=== Phase 2: cold 1M prefill (War and Peace) ===" | tee -a "$OUTDIR/run.log"
cp "$WAR_PEACE" "$OUTDIR/phase2-prompt.txt"
cat >> "$OUTDIR/phase2-prompt.txt" <<'EOF'

---
Question: Briefly describe the major events of the second half of this novel. Cite at least two specific minor characters by name.
Answer:
EOF
fire_completion phase2-cold-1m "$OUTDIR/phase2-prompt.txt" 384 | tee -a "$OUTDIR/run.log"
vram_snapshot

# ============================
# Phase 3: cache validation — re-fire identical prefix with different tails
# ============================
echo "" | tee -a "$OUTDIR/run.log"
echo "=== Phase 3: cache validation (same 853K prefix, varied tails) ===" | tee -a "$OUTDIR/run.log"
for i in 1 2 3; do
    cp "$WAR_PEACE" "$OUTDIR/phase3-${i}-prompt.txt"
    case $i in
        1) Q="Name three central characters in this novel.";;
        2) Q="In what year does this novel begin?";;
        3) Q="What is the title of the book?";;
    esac
    cat >> "$OUTDIR/phase3-${i}-prompt.txt" <<EOF

---
Question: $Q
Answer:
EOF
    fire_completion "phase3-cache-${i}" "$OUTDIR/phase3-${i}-prompt.txt" 96 | tee -a "$OUTDIR/run.log"
    vram_snapshot
done

# ============================
# Phase 4: NIAH depth sweep — inject a needle at varied depths
# ============================
echo "" | tee -a "$OUTDIR/run.log"
echo "=== Phase 4: NIAH depth sweep ===" | tee -a "$OUTDIR/run.log"
NEEDLE="The secret passphrase for the Petersburg garrison is BRAVO-LIMA-7-EAGLE."
for depth in 10 25 50 75 90; do
    /home/llm/venv/bin/python3 - "$WAR_PEACE" "$depth" "$NEEDLE" "$OUTDIR/phase4-d${depth}-prompt.txt" <<'PYEOF'
import sys
src, depth, needle, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
text = open(src).read()
i = int(len(text) * (depth / 100.0))
# Snap to nearest sentence-end for cleaner injection
while i < len(text) and text[i] != '.':
    i += 1
i = min(i + 1, len(text))
injected = text[:i] + f"\n\n[STAFF MEMO]: {needle}\n\n" + text[i:]
question = "\n\n---\nQuestion: What is the secret passphrase for the Petersburg garrison?\nAnswer:"
open(out, 'w').write(injected + question)
PYEOF
    fire_completion "phase4-niah-d${depth}" "$OUTDIR/phase4-d${depth}-prompt.txt" 32 | tee -a "$OUTDIR/run.log"
    vram_snapshot
done

# ============================
# Phase 5: War and Peace QA — qualitative
# ============================
echo "" | tee -a "$OUTDIR/run.log"
echo "=== Phase 5: War and Peace qualitative QA ===" | tee -a "$OUTDIR/run.log"
QUESTIONS=(
    "What happens to Prince Andrei Bolkonsky at the Battle of Austerlitz?"
    "Who is Platon Karataev and what is his role in the novel?"
    "Describe the relationship between Natasha Rostova and Pierre Bezukhov by the end of the novel."
    "What is Tolstoy's argument about the role of 'great men' in history, as presented in the epilogue?"
    "Name three peripheral characters who appear only in the second half of the novel."
)
for j in $(seq 0 4); do
    cp "$WAR_PEACE" "$OUTDIR/phase5-q${j}-prompt.txt"
    cat >> "$OUTDIR/phase5-q${j}-prompt.txt" <<EOF

---
Question: ${QUESTIONS[$j]}
Answer:
EOF
    fire_completion "phase5-qa-${j}" "$OUTDIR/phase5-q${j}-prompt.txt" 256 | tee -a "$OUTDIR/run.log"
    vram_snapshot
done

# ============================
# Phase 6: stability soak — cycle mixed lengths
# ============================
echo "" | tee -a "$OUTDIR/run.log"
echo "=== Phase 6: stability soak (4 hours) ===" | tee -a "$OUTDIR/run.log"
SOAK_END=$((SECONDS + 14400))
iter=0
while [ "$SECONDS" -lt "$SOAK_END" ]; do
    iter=$((iter + 1))
    # Rotate prompt size: 100K, 300K, 500K, 850K
    case $((iter % 4)) in
        0) MAX_TOKENS=128; head -c 600000 "$WAR_PEACE" > "$OUTDIR/phase6-i${iter}-prompt.txt" ;;
        1) MAX_TOKENS=128; head -c 1500000 "$WAR_PEACE" > "$OUTDIR/phase6-i${iter}-prompt.txt" ;;
        2) MAX_TOKENS=128; head -c 2200000 "$WAR_PEACE" > "$OUTDIR/phase6-i${iter}-prompt.txt" ;;
        3) MAX_TOKENS=128; cp "$WAR_PEACE" "$OUTDIR/phase6-i${iter}-prompt.txt" ;;
    esac
    echo "\n\n---\nQuestion: Summarise the preceding text in one sentence.\nAnswer:" >> "$OUTDIR/phase6-i${iter}-prompt.txt"
    echo "[soak] iter=$iter remaining=$((SOAK_END - SECONDS))s" | tee -a "$OUTDIR/run.log"
    fire_completion "phase6-soak-i${iter}" "$OUTDIR/phase6-i${iter}-prompt.txt" "$MAX_TOKENS" | tee -a "$OUTDIR/run.log"
    vram_snapshot
    # Light pause to avoid saturating server log
    sleep 5
done
echo "[soak] complete: $iter iterations" | tee -a "$OUTDIR/run.log"

# ============================
# Phase 7: summary
# ============================
echo "" | tee -a "$OUTDIR/run.log"
echo "=== Phase 7: aggregate summary ===" | tee -a "$OUTDIR/run.log"
"$HERE/aggregate-t6.3-1m-overnight.py" "$OUTDIR" 2>&1 | tee -a "$OUTDIR/run.log" || \
    echo "[warn] aggregator not yet present" | tee -a "$OUTDIR/run.log"

echo "=== T6.3 1M overnight complete ===" | tee -a "$OUTDIR/run.log"
