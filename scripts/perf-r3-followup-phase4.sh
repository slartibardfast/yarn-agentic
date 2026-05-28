#!/usr/bin/env bash
# perf-r3-followup-phase4.sh — Phase 4 of PHASE_PERF_R3_FOLLOWUP.
#
# nsys kernel diff at ctx=8192 vs ctx=262144, identical 200t prompt
# workload, to identify which kernel(s) account for the R1 -25.9% TG
# gap measured in Phase 3.
#
# Per config:
#   - server up (nsys-wrapped, full traces on)
#   - 1 untraced warmup request (uses /completion same as Phase 3)
#   - 1 traced rep (the one we read)
#   - SIGINT to inner llama-server PID -> nsys parent finalizes the trace
#
# Output: two .nsys-rep files + their cuda_gpu_kern_sum reports + diff.
#
# Usage:
#     OUT=<dir> bash perf-r3-followup-phase4.sh

set -uo pipefail

BIN=/home/dconnolly/yarn-agentic/ik_llama.cpp/build/bin/llama-server
GGUF=/opt/models/recast-out/qwen3.6-27b-V-F1.T1.lm_head-f16.gguf
PROMPT_FILE=/tmp/perf-r3-prompts/prompt-210.txt
OUT="${OUT:-/tmp/perf-r3-followup-phase4-$(date -u +%Y%m%dT%H%M%S)}"
PORT="${PORT:-18298}"
mkdir -p "$OUT"

# Pre-write request body
python3 -c "import json; print(json.dumps({'prompt': open('$PROMPT_FILE').read(), 'n_predict':128, 'temperature':0.0, 'top_p':1.0, 'top_k':0, 'seed':1, 'cache_prompt':False}))" > "$OUT/req-body.json"

run_traced() {
    local label="$1"
    local ctx="$2"
    echo
    echo "=== $label: ctx=$ctx — launching nsys-wrapped server ==="

    # nsys profile launches the server. We send SIGINT to the inner
    # llama-server PID to give nsys clean finalization.
    sudo env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        nsys profile \
            --output "$OUT/$label.nsys-rep" \
            -t cuda,nvtx,osrt \
            --gpu-metrics-devices=all \
            --gpu-metrics-frequency=1000 \
            --sample=none --cpuctxsw=none \
            --capture-range=none \
            --force-overwrite=true \
            "$BIN" -m "$GGUF" \
            --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
            -ngl 999 -fa on \
            --ctx-size "$ctx" --parallel 1 \
            --threads 4 --batch-size 2048 --ubatch-size 256 \
            --cache-type-k q4_0 --cache-type-v q4_0 \
            --k-cache-hadamard --v-cache-hadamard \
            --cache-ram 40960 \
            --no-context-shift --ctx-checkpoints 64 \
            --mlockall --rt-prio 50 --cpu-mask 0xF0 \
            --port "$PORT" --host 127.0.0.1 \
            > "$OUT/$label-server.log" 2>&1 &
    NSYS_PID=$!
    echo "  nsys parent PID=$NSYS_PID"

    # Wait for server health.
    for i in $(seq 1 180); do
        if curl -fsS --max-time 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    if ! curl -fsS --max-time 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "  SERVER FAILED — see $OUT/$label-server.log"
        sudo kill -TERM "$NSYS_PID" 2>/dev/null
        return 1
    fi
    echo "  server up"

    # Warmup (untraced from analysis standpoint — included in trace but
    # we filter to the second response window via timings).
    echo "  warmup rep..."
    curl -sf -o "$OUT/$label-warmup.json" --max-time 300 -X POST \
        "http://127.0.0.1:$PORT/completion" \
        -H "Content-Type: application/json" \
        -d @"$OUT/req-body.json" >/dev/null
    echo "  warmup done"

    # Measured rep
    echo "  measured rep..."
    curl -sf -o "$OUT/$label-traced.json" --max-time 300 -X POST \
        "http://127.0.0.1:$PORT/completion" \
        -H "Content-Type: application/json" \
        -d @"$OUT/req-body.json" >/dev/null
    python3 -c "
import json
d = json.load(open('$OUT/$label-traced.json'))
t = d.get('timings', {})
print(f'  $label TRACED: PP={t.get(\"prompt_per_second\",0):.1f} t/s TG={t.get(\"predicted_per_second\",0):.2f} t/s n_pp={t.get(\"prompt_n\")} n_tg={t.get(\"predicted_n\")}')
"

    # Find inner llama-server PID (child of sudo->nsys->llama-server)
    LLAMA_PID=$(pgrep -P $(pgrep -P $NSYS_PID) -f llama-server | head -1)
    if [[ -z "$LLAMA_PID" ]]; then
        # alt: anything matching llama-server with our port
        LLAMA_PID=$(pgrep -f "llama-server.*port $PORT" | head -1)
    fi
    echo "  llama-server inner PID=$LLAMA_PID; sending SIGINT for clean nsys finalize"
    sudo kill -INT "$LLAMA_PID" 2>/dev/null
    # Wait for nsys parent to exit naturally (it finalizes the report).
    wait "$NSYS_PID" 2>/dev/null
    sleep 5

    if [[ -f "$OUT/$label.nsys-rep" ]]; then
        echo "  trace: $OUT/$label.nsys-rep ($(du -h "$OUT/$label.nsys-rep" | cut -f1))"
    else
        echo "  WARNING: trace file missing"
    fi
}

echo "=== PHASE_PERF_R3_FOLLOWUP Phase 4 — R1 nsys kernel diff ==="
echo "OUT=$OUT"
echo

# Run both configs
run_traced "ctx8k"   8192   || { echo "ctx8k failed"; exit 1; }
run_traced "ctx256k" 262144 || { echo "ctx256k failed"; exit 1; }

# Export kern-sum reports
echo
echo "=== Exporting cuda_gpu_kern_sum reports ==="
for label in ctx8k ctx256k; do
    echo "  $label..."
    nsys stats --report cuda_gpu_kern_sum --format csv \
        --output "$OUT/$label-kern-sum.csv" \
        "$OUT/$label.nsys-rep" \
        > "$OUT/$label-kern-sum.stdout.log" 2>&1
done

# Build the diff
echo
echo "=== Diffing top kernels ==="
python3 << 'PYEOF'
import csv, os
OUT = os.environ.get("OUT")

def load(p):
    # nsys stats writes its CSV with a header preamble; skip until we find
    # the column header line.
    rows = []
    with open(p) as f:
        text = f.read()
    # find the line starting with "Time(%)" or similar
    lines = text.splitlines()
    hdr_idx = None
    for i, l in enumerate(lines):
        if l.startswith("Time (%)") or l.startswith("Time(%)"):
            hdr_idx = i
            break
    if hdr_idx is None:
        return []
    rdr = csv.DictReader(lines[hdr_idx:])
    return list(rdr)

# Look for the kern-sum CSV produced as the actual output file
import glob
ck8  = glob.glob(f"{OUT}/ctx8k*kern_sum*.csv") + glob.glob(f"{OUT}/ctx8k-kern-sum.csv")
ck25 = glob.glob(f"{OUT}/ctx256k*kern_sum*.csv") + glob.glob(f"{OUT}/ctx256k-kern-sum.csv")
print("ctx8k candidates:", ck8)
print("ctx256k candidates:", ck25)

if not ck8 or not ck25:
    print("Could not locate kern-sum CSVs; check nsys stats stdout logs")
    exit(0)

r8  = load(ck8[0])
r25 = load(ck25[0])
print(f"  loaded {len(r8)} kernels (ctx=8k), {len(r25)} kernels (ctx=256k)")

# Build name -> total ns
def to_map(rows):
    m = {}
    for r in rows:
        name = r.get("Name") or r.get("Kernel Name") or r.get("Style")
        tot  = r.get("Total Time (ns)") or r.get("Total Time")
        if name and tot:
            try:
                m[name] = float(str(tot).replace(",", ""))
            except ValueError:
                pass
    return m

m8 = to_map(r8); m25 = to_map(r25)

# Diff
diff = []
for name in set(m8.keys()) | set(m25.keys()):
    t8 = m8.get(name, 0.0); t25 = m25.get(name, 0.0)
    diff.append((name, t8, t25, t25 - t8, (t25/t8 - 1) if t8 > 0 else float("inf")))

# Sort by absolute delta (ns)
diff.sort(key=lambda x: -abs(x[3]))

print()
print("Top 30 kernels by |Δns| (ctx=256k − ctx=8k):")
print(f"  {'kernel':<60}  {'ctx8k_ns':>14}  {'ctx256k_ns':>14}  {'Δns':>14}  {'Δ%':>8}")
for name, t8, t25, dns, pct in diff[:30]:
    nm = name[:60]
    p = f"{pct*100:+.1f}%" if pct != float("inf") else "  NEW"
    print(f"  {nm:<60}  {int(t8):>14,}  {int(t25):>14,}  {int(dns):>14,}  {p:>8}")

# Save diff CSV
with open(f"{OUT}/kern-diff.csv", "w") as f:
    w = csv.writer(f)
    w.writerow(["kernel", "ctx8k_ns", "ctx256k_ns", "delta_ns", "delta_pct"])
    for row in diff:
        w.writerow([row[0], row[1], row[2], row[3], row[4]])
print(f"\nFull diff: {OUT}/kern-diff.csv")
PYEOF
