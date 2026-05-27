#!/usr/bin/env bash
# scripts/gen-test-vision-payload.sh — PHASE 46 B.8 helper.
#
# Generates scripts/test-vision-payload.json from the in-tree test-1.jpeg
# fixture. The payload is the OpenAI-compatible chat/completions request
# shape used for ad-hoc deploy-smoke checks against live prod (B.8 step 3
# and the rollback drill at step 4).
#
# The JSON itself is ~165 KB (mostly base64 image data) and is
# gitignored — regenerate on demand from test-1.jpeg, which IS in tree.
#
# Usage:
#   bash scripts/gen-test-vision-payload.sh
#   # then:
#   curl -fsS -X POST http://127.0.0.1:8085/v1/chat/completions \
#       -H "Authorization: Bearer $BEARER" \
#       -H "Content-Type: application/json" \
#       -d @scripts/test-vision-payload.json
#
# Env overrides:
#   IMG=/path/to/jpeg    (default: ik_llama.cpp/examples/mtmd/test-1.jpeg)
#   OUT=/path/to.json    (default: scripts/test-vision-payload.json)

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMG=${IMG:-"$REPO_ROOT/ik_llama.cpp/examples/mtmd/test-1.jpeg"}
OUT=${OUT:-"$REPO_ROOT/scripts/test-vision-payload.json"}

if [[ ! -f "$IMG" ]]; then
    echo "ABORT: image fixture missing: $IMG" >&2
    exit 2
fi

python3 - "$IMG" "$OUT" <<'PY'
import base64, json, sys
img_path, out_path = sys.argv[1], sys.argv[2]
with open(img_path, 'rb') as f:
    b64 = base64.b64encode(f.read()).decode('ascii')
payload = {
    'model': 'qwen',
    'max_tokens': 16,
    'temperature': 0,
    'messages': [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'What is in this image? One short sentence.'},
        {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
    ]}],
}
with open(out_path, 'w') as f:
    json.dump(payload, f, indent=2)
print(f"wrote {len(b64)} b64 chars → {out_path}")
PY
