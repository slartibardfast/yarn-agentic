#!/usr/bin/env bash
# Start passive recorders for a llama-server snoop run.
# Prints the run-id, the stop command, and exits. Recorders run
# in the background until snoop-stop.sh is invoked.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="snoop-$(date +%Y%m%dT%H%M%S)"
OUT="${HOME}/snoop-runs/${RUN_ID}"
mkdir -p "${OUT}/prompts"

# 1) journal stream (ISO timestamps so summarise can sort)
journalctl --user -u llama-server -f -o short-iso \
    --since "now" \
    > "${OUT}/journal.log" 2>&1 &
echo $! >> "${OUT}/pids.txt"

# 2) nginx access log filtered to OpenCode
( tail -F /var/log/nginx/llm.log 2>/dev/null \
    | grep --line-buffered opencode ) \
    > "${OUT}/nginx.log" 2>&1 &
echo $! >> "${OUT}/pids.txt"

# 3) slots poll (compact JSONL, no full prompt text)
python3 - "${OUT}/slots.jsonl" <<'PY' &
import json, sys, time, urllib.request, datetime
out = open(sys.argv[1], "w", buffering=1)
url = "http://127.0.0.1:8080/slots"
while True:
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            slots = json.load(r)
        rec = {
            "ts": datetime.datetime.now().isoformat(timespec="milliseconds"),
            "slots": [
                {
                    "id": s.get("id"),
                    "state": s.get("state"),
                    "id_task": s.get("id_task"),
                    "n_decoded": s.get("n_decoded"),
                    "prompt_chars": len(s.get("prompt") or ""),
                    "checkpoint_count": len(s.get("context_checkpoints") or []),
                    "checkpoint_pos_max": max(
                        (c.get("pos_max", 0) for c in (s.get("context_checkpoints") or [])),
                        default=0,
                    ),
                }
                for s in slots
            ],
        }
        out.write(json.dumps(rec) + "\n")
    except Exception as e:
        out.write(json.dumps({"ts": datetime.datetime.now().isoformat(), "err": str(e)}) + "\n")
    time.sleep(1.0)
PY
echo $! >> "${OUT}/pids.txt"

# 4) GPU dmon, 1 Hz
nvidia-smi dmon -s pucm -d 1 -o T > "${OUT}/gpu.tsv" 2>&1 &
echo $! >> "${OUT}/pids.txt"

# 5) prompt-diff capturer
python3 "${SCRIPT_DIR}/snoop-prompt-diff.py" "${OUT}" \
    > "${OUT}/prompt-diff.log" 2>&1 &
echo $! >> "${OUT}/pids.txt"

cat <<EOF
run_id: ${RUN_ID}
output: ${OUT}
stop:   bash ${SCRIPT_DIR}/snoop-stop.sh ${RUN_ID}
pids:   $(tr '\n' ' ' < "${OUT}/pids.txt")
EOF
