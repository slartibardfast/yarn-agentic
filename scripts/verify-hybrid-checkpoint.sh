#!/usr/bin/env bash
# PHASE_HYBRID_CHECKPOINT §6 binding test.
#
# Verifies hybrid/recurrent checkpoint RESTORE works without the post-restore
# defrag SEGV (root-caused in §5.6 to build_defrag -> ggml_view_3d; fixed by the
# skip_next_defrag fence, d67da398). A 2-turn shared-prefix conversation forces a
# checkpoint restore on turn 2, then a short multi-turn soak. PASS requires:
# restore actually fired (>=1), no crash markers, server stays alive, turns 200.
#
# Stops production, runs the build-tree binary on :8080 in isolation, restart-trap
# restores production on the untouched /opt binary. No deploy.
set -uo pipefail
cd /home/dconnolly/yarn-agentic

RUN_ID="hybridckpt-$(date +%Y%m%dT%H%M%S)"
EV="data/hybrid-checkpoint/$RUN_ID"; mkdir -p "$EV"
BIN=${BIN:-/home/dconnolly/yarn-agentic/ik_llama.cpp/build/bin/llama-server}
MODEL=/opt/models/recast-out/qwen3.6-27b-V-F1.T1.lm_head-f16.gguf
PORT=8080
HEALTH="http://127.0.0.1:$PORT/health"
SUM="$EV/summary.txt"; SRV="$EV/server.stderr"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

restart_prod(){
  log "trap: restarting production llama-server.service"
  sudo -n systemctl start llama-server.service 2>&1 || log "trap: start nonzero"
  for i in $(seq 1 60); do
    curl -s --max-time 3 "$HEALTH" 2>/dev/null | grep -q '"status":"ok"' && { log "trap: production /health ok"; echo RESTARTED_OK>>"$SUM"; return 0; }
    sleep 2
  done
  log "trap: WARNING production /health NOT confirmed"; echo RESTART_UNCONFIRMED>>"$SUM"
}
trap restart_prod EXIT

echo "RUN_ID=$RUN_ID  BIN=$BIN" > "$SUM"
log "stopping production"
sudo -n systemctl stop llama-server.service 2>&1 | tee -a "$SUM"
for i in $(seq 1 30); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); [ "$u" -lt 2000 ] && break; sleep 2; done
log "post-stop GPU0 used=${u} MiB"
sudo -n bash scripts/gpu-clocks.sh lock >/dev/null 2>&1 || true

log "launching test server (build-tree) on :$PORT"
"$BIN" -m "$MODEL" \
  --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 \
  -ngl 999 -fa on --ctx-size 262144 --parallel 1 --threads 4 \
  --batch-size 2048 --ubatch-size 512 \
  --cache-type-k q4_0 --cache-type-v q4_0 --k-cache-hadamard --v-cache-hadamard \
  --cache-ram 40960 --ctx-checkpoints 64 --no-context-shift --jinja \
  --temp 0 --top-p 1.0 --top-k 1 --port $PORT \
  > "$EV/server.stdout" 2> "$SRV" &
SPID=$!
log "waiting up to 240s for /health (pid $SPID)"
HOK=0
for i in $(seq 1 120); do
  kill -0 $SPID 2>/dev/null || { log "server died during startup"; echo STARTUP_FAIL>>"$SUM"; tail -5 "$SRV"|tee -a "$SUM"; exit 1; }
  curl -s --max-time 3 "$HEALTH" 2>/dev/null | grep -q '"status":"ok"' && { HOK=1; break; }
  sleep 2
done
[ "$HOK" = 1 ] || { log "health timeout"; echo HEALTH_TIMEOUT>>"$SUM"; exit 1; }
log "server healthy"

# ~3500-token deterministic prompt (recurrent state accumulates over the prefix).
PROMPT=$(python3 -c "print('You are auditing a system log; read it then answer. ' + ' '.join('Line %d: subsystem %d nominal, queue depth %d, latency %dms.'%(i,i%11,i%5,i%9) for i in range(600)))")
DELTA="One more word please."

chat(){ curl -sS -m 180 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$1" -o "$2" -w "%{http_code}"; }

# Turn 1 — establishes the prefix + creates checkpoints during prefill.
T1=$(python3 -c "import json,sys;print(json.dumps({'model':'q','max_tokens':24,'temperature':0,'messages':[{'role':'user','content':sys.argv[1]+' Summarize in one sentence.'}]}))" "$PROMPT")
log "turn 1 (establish prefix)"; C1=$(chat "$T1" "$EV/resp1.json"); log "turn1 http=$C1"
A1=$(python3 -c "import json;print((json.load(open('$EV/resp1.json')).get('choices',[{}])[0].get('message',{}).get('content') or '')[:240])" 2>/dev/null)

# Turn 2 — shares the (prefix + assistant) tokens => forces a checkpoint RESTORE,
# which is what exercises the post-restore defrag fence.
T2=$(python3 -c "import json,sys;print(json.dumps({'model':'q','max_tokens':24,'temperature':0,'messages':[{'role':'user','content':sys.argv[1]+' Summarize in one sentence.'},{'role':'assistant','content':sys.argv[2]},{'role':'user','content':sys.argv[3]}]}))" "$PROMPT" "$A1" "$DELTA")
log "turn 2 (expect RESTORE + post-restore defrag)"; C2=$(chat "$T2" "$EV/resp2.json"); log "turn2 http=$C2"

# Soak — 4 more shared-prefix delta turns.
for k in 3 4 5 6; do
  TK=$(python3 -c "import json,sys;print(json.dumps({'model':'q','max_tokens':16,'temperature':0,'messages':[{'role':'user','content':sys.argv[1]+' Summarize in one sentence.'},{'role':'assistant','content':sys.argv[2]},{'role':'user','content':'Turn '+sys.argv[3]+': one more word.'}]}))" "$PROMPT" "$A1" "$k")
  CK=$(chat "$TK" "$EV/resp$k.json"); log "turn $k http=$CK"
done

sleep 2
alive=0; kill -0 $SPID 2>/dev/null && alive=1
restored=$(grep -c "restored context checkpoint took" "$SRV" 2>/dev/null); restored=${restored:-0}
created=$(grep -c "created context checkpoint" "$SRV" 2>/dev/null); created=${created:-0}
nousable=$(grep -c "no usable hybrid/recurrent checkpoint" "$SRV" 2>/dev/null); nousable=${nousable:-0}
crash=0; grep -qiE "SIGSEGV|signal 11|GGML_ASSERT|Segmentation fault|core dumped|CUDA error" "$SRV" && crash=1
{
  echo "TURN1_HTTP=$C1  TURN2_HTTP=$C2"
  echo "SERVER_ALIVE=$alive  CRASH_MARKERS=$crash"
  echo "CHECKPOINTS_CREATED=$created  CHECKPOINTS_RESTORED=$restored  NO_USABLE=$nousable"
} >> "$SUM"
if [ "$alive" = 1 ] && [ "$crash" = 0 ] && [ "$C1" = 200 ] && [ "$C2" = 200 ] && [ "${restored:-0}" -ge 1 ]; then
  echo "HYBRID_CKPT_RESTORE=PASS (restore fired ${restored}x, post-restore defrag clean, server alive, soak ok)" >> "$SUM"
elif [ "${restored:-0}" -lt 1 ] && [ "$crash" = 0 ] && [ "$alive" = 1 ]; then
  echo "HYBRID_CKPT_RESTORE=INCONCLUSIVE (no restore fired — prompt/turn shape did not trigger a checkpoint match; defrag fence not exercised)" >> "$SUM"
else
  echo "HYBRID_CKPT_RESTORE=FAIL (alive=$alive crash=$crash t1=$C1 t2=$C2 restored=$restored)" >> "$SUM"
fi
kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
echo "=== SUMMARY ($RUN_ID) ==="; cat "$SUM"
