#!/usr/bin/env bash
set -uo pipefail
pkill -f "llama-server" 2>/dev/null || true
sleep 3
mkdir -p /home/llm/yarn-agentic/data/nsys-graphcache
export OUT_ROOT=/home/llm/yarn-agentic/data/nsys-graphcache
exec bash /home/llm/yarn-agentic/scripts/nsys-revisit-pre-port.sh
