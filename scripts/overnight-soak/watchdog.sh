#!/usr/bin/env bash
# Static watchdog: append one CSV row per fire to a per-run log file.
# Sampled by a systemd user timer every 60 s — no live observer needed.
#
# Auto-recovery: if /health fails for HEALTH_FAIL_THRESHOLD consecutive
# fires, swap profiles/active.sh symlink to the safe single-slot
# profile and restart llama-server. Writes a RECOVERED_AT_<ts> marker
# row before doing so. Manual review of the CSV the next morning is
# expected.
#
# Outputs:
#   $LOG_ROOT/<run-id>.csv             — per-fire metrics
#   $LOG_ROOT/<run-id>.events.log      — narrative events (start, recover, etc.)
#
# Idempotent: safe to invoke as a oneshot every minute.

set -uo pipefail

LOG_ROOT=${LOG_ROOT:-/opt/models/profiling/overnight-soak}
RUN_ID_FILE=$LOG_ROOT/.run-id
HEALTH_FAIL_FILE=$LOG_ROOT/.health-fail-count
HEALTH_FAIL_THRESHOLD=${HEALTH_FAIL_THRESHOLD:-5}
SAFE_PROFILE=${SAFE_PROFILE:-qwen36-27b-x1.sh}
JOURNAL_CURSOR_FILE=$LOG_ROOT/.journal-cursor

mkdir -p "$LOG_ROOT"

# Establish run id (one per active llama-server start).
SERVICE_START=$(systemctl --user show llama-server --property=ActiveEnterTimestamp --value 2>/dev/null \
                | tr ' ' '_' | tr -d ':')
[ -z "$SERVICE_START" ] && SERVICE_START=unknown
RUN_ID=$(cat "$RUN_ID_FILE" 2>/dev/null || true)
if [ -z "$RUN_ID" ] || [ "$RUN_ID" != "$SERVICE_START" ]; then
    RUN_ID=$SERVICE_START
    echo "$RUN_ID" > "$RUN_ID_FILE"
    echo "0" > "$HEALTH_FAIL_FILE"
    rm -f "$JOURNAL_CURSOR_FILE"
    {
        echo "ts,uptime_s,host_rss_mb,host_swap_mb,host_load_1m,gpu0_mem_used_mb,gpu0_mem_free_mb,gpu0_util_pct,gpu1_mem_used_mb,gpu1_mem_free_mb,gpu1_util_pct,health_ok,slots_idle,slots_processing,journal_5xx,journal_alloc_failed,journal_concat_probe,journal_abort,recovered"
    } > "$LOG_ROOT/$RUN_ID.csv"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) START run_id=$RUN_ID" >> "$LOG_ROOT/$RUN_ID.events.log"
fi

CSV=$LOG_ROOT/$RUN_ID.csv
EVTS=$LOG_ROOT/$RUN_ID.events.log

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# Host metrics
RSS=$(ps -C llama-server -o rss= 2>/dev/null | awk '{s+=$1} END {print s/1024}')
[ -z "$RSS" ] && RSS=0
SWAP=$(awk '/SwapFree/ {free=$2} /SwapTotal/ {tot=$2} END {if (tot>0) printf "%d", (tot-free)/1024; else print 0}' /proc/meminfo)
LOAD1=$(awk '{print $1}' /proc/loadavg)

# GPU metrics
read GPU0_USED GPU0_FREE GPU0_UTIL <<<$(nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits -i 0 2>/dev/null | tr ',' ' ')
read GPU1_USED GPU1_FREE GPU1_UTIL <<<$(nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits -i 1 2>/dev/null | tr ',' ' ')
GPU0_USED=${GPU0_USED:-0}; GPU0_FREE=${GPU0_FREE:-0}; GPU0_UTIL=${GPU0_UTIL:-0}
GPU1_USED=${GPU1_USED:-0}; GPU1_FREE=${GPU1_FREE:-0}; GPU1_UTIL=${GPU1_UTIL:-0}

# Health probe
HEALTH=0
SLOTS_IDLE=0
SLOTS_PROCESSING=0
HEALTH_JSON=$(curl -fsS --max-time 3 http://127.0.0.1:8080/health 2>/dev/null)
if [ -n "$HEALTH_JSON" ]; then
    HEALTH=1
    SLOTS_IDLE=$(echo "$HEALTH_JSON" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("slots_idle",0))' 2>/dev/null || echo 0)
    SLOTS_PROCESSING=$(echo "$HEALTH_JSON" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("slots_processing",0))' 2>/dev/null || echo 0)
fi

# Uptime of the llama-server process
UPTIME=$(ps -C llama-server -o etimes= 2>/dev/null | awk '{print $1; exit}')
[ -z "$UPTIME" ] && UPTIME=0

# Journal-based event counters since last sample
JOURNAL_5XX=0
JOURNAL_ALLOC=0
JOURNAL_CONCAT=0
JOURNAL_ABORT=0

# Use --since / cursor to bound the lookup window to last 70s (a bit
# > the 60s timer to guarantee no gap).
SINCE=$(date -u -d '70 seconds ago' '+%Y-%m-%d %H:%M:%S')
JBLOCK=$(journalctl --user -u llama-server --since "$SINCE" --no-pager 2>/dev/null)
JOURNAL_5XX=$(echo "$JBLOCK" | grep -cE 'status=(5[0-9]{2})' 2>/dev/null || echo 0)
JOURNAL_ALLOC=$(echo "$JBLOCK" | grep -ci 'alloc_failed\|GGML_STATUS_ALLOC_FAILED\|recoverable OOM' 2>/dev/null || echo 0)
JOURNAL_CONCAT=$(echo "$JBLOCK" | grep -c 'CONCAT-PROBE' 2>/dev/null || echo 0)
JOURNAL_ABORT=$(echo "$JBLOCK" | grep -ciE 'GGML_ABORT|ggml_abort|aborted \(core dumped\)' 2>/dev/null || echo 0)

# Auto-recovery decision
HEALTH_FAILS=$(cat "$HEALTH_FAIL_FILE" 2>/dev/null || echo 0)
RECOVERED=0
if [ "$HEALTH" = "0" ]; then
    HEALTH_FAILS=$((HEALTH_FAILS + 1))
    echo "$HEALTH_FAILS" > "$HEALTH_FAIL_FILE"
    if [ "$HEALTH_FAILS" -ge "$HEALTH_FAIL_THRESHOLD" ]; then
        echo "$(now) RECOVER threshold=$HEALTH_FAIL_THRESHOLD; flipping active.sh -> $SAFE_PROFILE; restarting" >> "$EVTS"
        ln -sfn "$SAFE_PROFILE" /home/llm/profiles/active.sh
        systemctl --user restart llama-server
        RECOVERED=1
        echo "0" > "$HEALTH_FAIL_FILE"
    fi
else
    [ "$HEALTH_FAILS" != "0" ] && echo "$(now) RECOVER cleared (health restored)" >> "$EVTS"
    echo "0" > "$HEALTH_FAIL_FILE"
fi

# Append the row
printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$(now)" "$UPTIME" "$RSS" "$SWAP" "$LOAD1" \
    "$GPU0_USED" "$GPU0_FREE" "$GPU0_UTIL" \
    "$GPU1_USED" "$GPU1_FREE" "$GPU1_UTIL" \
    "$HEALTH" "$SLOTS_IDLE" "$SLOTS_PROCESSING" \
    "$JOURNAL_5XX" "$JOURNAL_ALLOC" "$JOURNAL_CONCAT" "$JOURNAL_ABORT" \
    "$RECOVERED" \
    >> "$CSV"

exit 0
