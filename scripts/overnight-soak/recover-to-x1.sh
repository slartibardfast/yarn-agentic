#!/usr/bin/env bash
# Manual fallback to single-slot safe profile. Use if the watchdog
# auto-recovery didn't fire and you need to revert immediately.
set -euo pipefail
ln -sfn qwen36-27b-x1.sh /home/llm/profiles/active.sh
ls -la /home/llm/profiles/active.sh
systemctl --user restart llama-server
sleep 4
systemctl --user is-active llama-server
/home/llm/.local/bin/llama-healthcheck 8080 60 2>&1 | tail -1
