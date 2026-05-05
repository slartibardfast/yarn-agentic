#!/usr/bin/env bash
# CUDA graph cache: cache-collapse assertion.
#
# Drives a probe-enabled llama-server with the agentic prompt corpus
# and asserts that the per-device cuda_graphs cache size stays small.
# Pre-collapse (today): cache holds one entry per distinct shape (~200
# entries observed under real OpenCode traffic).
# Post-collapse (after Phase B): cache holds one entry per topology
# class (11–15 observed under same workload).
#
# Threshold: cache_size <= 30 per device. Chosen above the observed
# topology-class count (15) with margin for transient cache state but
# well below the pre-B shape-entry count (200+).
#
# Usage: scripts/cuda-graph-probe/run-topology-class-count.sh
set -euo pipefail

THRESHOLD=${THRESHOLD:-30}
TARGET_TOKENS=${TARGET_TOKENS:-3000}
NP=${NP:-1}
DUMP_ROOT="$HOME/cuda-graph-probe/run-collapse-$(date +%Y%m%dT%H%M%S)"

bash /home/llm/yarn-agentic/scripts/cuda-graph-probe/run-soak.sh \
    "$NP" "$TARGET_TOKENS" "$DUMP_ROOT" >&2

echo
echo "=== parsing cache_size from $DUMP_ROOT ==="
python3 -c "
import json, sys
from pathlib import Path
from collections import defaultdict

root = Path('$DUMP_ROOT')
# Each hit_counter flush emits one record per cache entry. Count the
# distinct shape_key values per backend at the latest flush — that's
# the steady-state cache size.
latest_ts = defaultdict(int)
entries = defaultdict(set)

for f in root.glob('cuda*-hit_counter.jsonl'):
    with f.open() as fp:
        for line in fp:
            try:
                r = json.loads(line)
                b = r['backend']
                ts = int(r['ts_ns'])
                if ts > latest_ts[b]:
                    latest_ts[b] = ts
                    entries[b] = set()
                if ts == latest_ts[b]:
                    entries[b].add(r['shape_key'])
            except Exception:
                continue

threshold = $THRESHOLD
fail = False
for b in sorted(entries.keys()):
    n = len(entries[b])
    verdict = 'PASS' if n <= threshold else 'FAIL'
    if verdict == 'FAIL': fail = True
    print(f'  {b}: cache_size={n} (threshold <= {threshold}) {verdict}')

if not entries:
    print('  no hit_counter records found; probe may not be active')
    sys.exit(1)

print()
if fail:
    print('RESULT: FAIL — cache size exceeds threshold; topology collapse not in effect')
    sys.exit(1)
print('RESULT: PASS — cache_size <= threshold on all devices')
"
