#!/usr/bin/env bash
# T6.0.a — cross-engine bench (ik_llama / vllm / any /v1/completions backend).
#
# Fires the 8 reference prompts (from scripts/gate0-dflash-speedup.py) at
# a /v1/completions endpoint and produces a schema-conformant JSON cell
# per PHASE_T6_CHARACTERISATION.md T6.0.b.
#
# This is the harness — it does NOT start the server. The caller stands
# up the engine (systemctl --user start <service> OR direct profile
# invocation) and points the bench at the resulting endpoint.
#
# Usage:
#   bash scripts/cross-engine-bench.sh [OUTDIR]
#
# Env knobs (all optional; defaults match the gate0 reference workload):
#   PORT            default 8080  (ik_llama production)
#   ENGINE          default "ik_llama"  (label for the cell JSON)
#   ENGINE_BUILD    default "$(cd ik_llama.cpp && git rev-parse --short HEAD)"
#   CELL_ID         default "<engine>-np${NP}-${fire_pattern}-$(date +%Y%m%dT%H%M%S)"
#   MAX_TOKENS      default 256  (matches gate0)
#   TEMP            default 0.0  (matches gate0)
#   SEED            default 42  (matches gate0)
#   IGNORE_EOS      default 1
#   FIRE_PATTERN    default "concurrent"  ("concurrent" | "sequential")
#   NP              default 8  (number of prompts fired together when concurrent)
#   GPU_MHZ         default 1455  (locked-clocks discipline)
#   NOTES           default ""  (freeform string appended to cell)
#
# Outputs (in OUTDIR):
#   cell.json         schema-conformant cell per PHASE_T6_CHARACTERISATION.md
#   responses.json    per-request raw timings + content
#   summary.txt       human-readable one-screen summary

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
OUTDIR="${1:-$REPO_ROOT/data/t6-cell-$(date +%Y%m%dT%H%M%S)}"
mkdir -p "$OUTDIR"

PORT="${PORT:-8080}"
ENGINE="${ENGINE:-ik_llama}"
ENGINE_BUILD="${ENGINE_BUILD:-$(cd "$REPO_ROOT/ik_llama.cpp" && git rev-parse --short HEAD 2>/dev/null || echo unknown)}"
MAX_TOKENS="${MAX_TOKENS:-256}"
TEMP="${TEMP:-0.0}"
SEED="${SEED:-42}"
IGNORE_EOS="${IGNORE_EOS:-1}"
FIRE_PATTERN="${FIRE_PATTERN:-concurrent}"
NP="${NP:-8}"
GPU_MHZ="${GPU_MHZ:-1455}"
NOTES="${NOTES:-}"
CELL_ID="${CELL_ID:-${ENGINE}-np${NP}-${FIRE_PATTERN}-$(date +%Y%m%dT%H%M%S)}"

ENDPOINT="http://127.0.0.1:${PORT}/v1/completions"
HEALTH="http://127.0.0.1:${PORT}/health"

# Pre-flight: engine must be reachable
if ! curl -fsS --max-time 5 "$HEALTH" >/dev/null 2>&1; then
    echo "FAIL: $HEALTH not reachable. Start the engine first." >&2
    exit 2
fi

# Pre-flight: no overlap with concurrent bench (per feedback_no_overlapping_benchmarks)
existing_consumers=$(pgrep -x "llama-batched-bench" 2>/dev/null || true)
if [ -n "$existing_consumers" ]; then
    echo "FAIL: existing GPU bench consumer detected — aborting." >&2
    ps -fp $existing_consumers >&2
    exit 2
fi

/home/llm/venv/bin/python - "$OUTDIR" "$CELL_ID" "$ENGINE" "$ENGINE_BUILD" \
    "$PORT" "$MAX_TOKENS" "$TEMP" "$SEED" "$IGNORE_EOS" \
    "$FIRE_PATTERN" "$NP" "$GPU_MHZ" "$NOTES" <<'PYEOF'
import datetime
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

(outdir, cell_id, engine, engine_build, port,
 max_tokens, temp, seed, ignore_eos,
 fire_pattern, np_count, gpu_mhz, notes) = sys.argv[1:14]

max_tokens   = int(max_tokens)
temp         = float(temp)
seed         = int(seed)
ignore_eos   = bool(int(ignore_eos))
np_count     = int(np_count)
gpu_mhz      = int(gpu_mhz)

# T6.0 reference workload — 8 prompts lifted from gate0-dflash-speedup.py
PROMPTS = [
    "Explain the difference between latent diffusion and pixel-space diffusion in two sentences.",
    "Summarize the plot of King Lear in one paragraph.",
    "Write Python code that fits a 2nd-degree polynomial to a list of (x, y) pairs.",
    "What are the main causes of the Peloponnesian War?",
    "Translate to French: The early-morning fog lingered over the harbour until the trawlers cut through it.",
    "List five practical steps for reducing memory allocations in a hot inner loop in Rust.",
    "Describe the role of telomeres in cellular aging.",
    "Write a haiku about a printing press.",
]

url = f"http://127.0.0.1:{port}/v1/completions"

def fire_one(idx, prompt):
    payload = {
        "prompt": prompt,
        "n_predict":   max_tokens,   # ik_llama
        "max_tokens":  max_tokens,   # vLLM
        "temperature": temp,
        "seed": seed + idx,
        "cache_prompt": False,
        "stream": False,
        "ignore_eos": ignore_eos,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=1800)
        t1  = time.time()
        obj = json.loads(resp.read())
        tokens = obj.get("tokens_predicted",
                        obj.get("usage", {}).get("completion_tokens", 0))
        if not tokens and obj.get("choices"):
            text = obj["choices"][0].get("text", "")
            tokens = len(text.split())  # rough fallback
        return {
            "idx": idx,
            "status": resp.status,
            "tokens": tokens,
            "latency_s": t1 - t0,
            "ttft_s": None,  # not measured without streaming
            "decode_s": None,
        }
    except urllib.error.HTTPError as e:
        t1 = time.time()
        return {
            "idx": idx,
            "status": e.code,
            "tokens": 0,
            "latency_s": t1 - t0,
            "ttft_s": None,
            "decode_s": None,
            "error_body": (e.read().decode("utf-8", errors="replace")[:300]
                           if e.fp else ""),
        }
    except Exception as e:
        return {"idx": idx, "status": 0, "tokens": 0,
                "latency_s": time.time() - t0,
                "ttft_s": None, "decode_s": None,
                "error": repr(e)}

# Warm one prompt to avoid first-call init in measurement
print(f"  [{cell_id}] warmup (1 prompt, 32 tokens)...", flush=True)
warm_payload = {
    "prompt": PROMPTS[0],
    "n_predict": 32, "max_tokens": 32,
    "temperature": 0.0, "seed": 0,
    "cache_prompt": False, "stream": False,
    "ignore_eos": True,
}
warm_body = json.dumps(warm_payload).encode()
warm_req = urllib.request.Request(url, data=warm_body, headers={"Content-Type": "application/json"})
urllib.request.urlopen(warm_req, timeout=600).read()

# Workload — np_count prompts × max_tokens
prompts_used = PROMPTS[:np_count] if np_count <= len(PROMPTS) else PROMPTS * ((np_count // len(PROMPTS)) + 1)
prompts_used = prompts_used[:np_count]

print(f"  [{cell_id}] timed: np={np_count} {fire_pattern}, {max_tokens} tokens each, ignore_eos={ignore_eos}", flush=True)
t_start = time.time()
results = []
if fire_pattern == "sequential":
    for i, p in enumerate(prompts_used):
        results.append(fire_one(i, p))
else:  # concurrent
    with ThreadPoolExecutor(max_workers=np_count) as ex:
        futures = [ex.submit(fire_one, i, p) for i, p in enumerate(prompts_used)]
        for f in as_completed(futures):
            results.append(f.result())
t_end = time.time()

results.sort(key=lambda r: r["idx"])
total_output_tokens = sum(r["tokens"] for r in results if r["status"] == 200)
wall = t_end - t_start
agg_tps = total_output_tokens / wall if wall > 0 else 0.0
latencies = sorted(r["latency_s"] for r in results if r["status"] == 200)
per_slot_tps = [r["tokens"] / r["latency_s"]
                for r in results if r["status"] == 200 and r["latency_s"] > 0]

def p(arr, frac):
    if not arr:
        return None
    k = max(0, min(len(arr) - 1, int(round(frac * (len(arr) - 1)))))
    return arr[k]

cell = {
    "cell_id": cell_id,
    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    "engine": engine,
    "engine_build": engine_build,
    "model": {
        "path": os.environ.get("LLAMA_BENCH_MODEL_HINT", "<see profile>"),
        "weights_dtype": os.environ.get("LLAMA_BENCH_WEIGHTS_DTYPE", "<see profile>"),
        "n_params": None,
        "n_layers": None,
        "n_head_kv": None,
    },
    "config": {
        "ctx_per_slot":      int(os.environ.get("LLAMA_BENCH_CTX_PER_SLOT", "0")) or None,
        "parallel":          int(os.environ.get("LLAMA_BENCH_PARALLEL", str(np_count))),
        "batch_size":        int(os.environ.get("LLAMA_BENCH_BATCH",      "0")) or None,
        "ubatch_size":       int(os.environ.get("LLAMA_BENCH_UBATCH",     "0")) or None,
        "kv_type_k":         os.environ.get("LLAMA_BENCH_KV_TYPE_K", None),
        "kv_type_v":         os.environ.get("LLAMA_BENCH_KV_TYPE_V", None),
        "k_cache_hadamard":  (os.environ.get("LLAMA_BENCH_K_HADAMARD", "") in ("1", "true", "True")) if os.environ.get("LLAMA_BENCH_K_HADAMARD") is not None else None,
        "v_cache_hadamard":  (os.environ.get("LLAMA_BENCH_V_HADAMARD", "") in ("1", "true", "True")) if os.environ.get("LLAMA_BENCH_V_HADAMARD") is not None else None,
        "flash_attn":        (os.environ.get("LLAMA_BENCH_FLASH_ATTN", "") in ("1", "true", "True")) if os.environ.get("LLAMA_BENCH_FLASH_ATTN") is not None else None,
        "dflash":            (os.environ.get("LLAMA_BENCH_DFLASH",     "") in ("1", "true", "True")) if os.environ.get("LLAMA_BENCH_DFLASH")     is not None else None,
        "draft_max":         int(os.environ.get("LLAMA_BENCH_DRAFT_MAX",      "0")) or None,
        "kv_pool_blocks":    int(os.environ.get("LLAMA_BENCH_KV_POOL_BLOCKS", "0")) or None,
        "defrag_thold":      float(os.environ["LLAMA_BENCH_DEFRAG_THOLD"]) if os.environ.get("LLAMA_BENCH_DEFRAG_THOLD") not in (None, "") else None,
        "ctx_checkpoints":   int(os.environ.get("LLAMA_BENCH_CTX_CHECKPOINTS", "0")) or None,
        "cache_ram":         int(os.environ.get("LLAMA_BENCH_CACHE_RAM",       "0")) or None,
        "device":            os.environ.get("LLAMA_BENCH_DEVICE", None),
        "split_mode":        os.environ.get("LLAMA_BENCH_SPLIT_MODE", None),
        "tensor_split":      os.environ.get("LLAMA_BENCH_TENSOR_SPLIT", None),
    },
    "clocks": {
        "gpu_mhz": gpu_mhz,
        "locked": True,
    },
    "workload": {
        "label": f"gate0-prompts-np{np_count}-{fire_pattern}",
        "prompts": len(PROMPTS),
        "max_tokens": max_tokens,
        "temp": temp,
        "seed": seed,
        "fire_pattern": fire_pattern,
        "ignore_eos": ignore_eos,
    },
    "results": {
        "wall_secs": wall,
        "total_input_toks": None,  # not measured client-side
        "total_output_toks": total_output_tokens,
        "tok_per_sec_aggregate": agg_tps,
        "tok_per_sec_per_slot_mean": statistics.mean(per_slot_tps) if per_slot_tps else None,
        "tok_per_sec_per_slot_p50":  p(sorted(per_slot_tps), 0.50)  if per_slot_tps else None,
        "tok_per_sec_per_slot_p99":  p(sorted(per_slot_tps), 0.99)  if per_slot_tps else None,
        "ttft_s_mean": None,
        "ttft_s_p50":  None,
        "ttft_s_p99":  None,
        "tpot_s_mean": None,
        "per_request": results,
    },
    "instrumentation": {
        "defrag_events":            None,
        "admission_events_503":     None,
        "admission_events_413":     None,
        "dispatch_multi_seq_count": None,
        "vram_kv_buffer_mib":       None,
        "vram_compute_buffer_mib":  None,
        "graph_pool_size":          None,
    },
    "notes": notes,
}

with open(os.path.join(outdir, "cell.json"), "w") as f:
    json.dump(cell, f, indent=2)
with open(os.path.join(outdir, "responses.json"), "w") as f:
    json.dump({"per_request": results, "wall_secs": wall}, f, indent=2)

summary = (
    f"=== T6 cell summary: {cell_id} ===\n"
    f"engine={engine} build={engine_build}\n"
    f"workload={cell['workload']['label']} max_tokens={max_tokens} ignore_eos={ignore_eos}\n"
    f"clocks={gpu_mhz} MHz locked\n"
    f"wall_secs={wall:.2f}\n"
    f"total_output_toks={total_output_tokens}\n"
    f"tok_per_sec_aggregate={agg_tps:.2f}\n"
    f"tok_per_sec_per_slot_mean={cell['results']['tok_per_sec_per_slot_mean']:.2f}\n"
    if per_slot_tps else
    f"=== T6 cell summary: {cell_id} ===\n"
    f"engine={engine} build={engine_build}\n"
    f"workload={cell['workload']['label']}\n"
    f"FAIL: no successful (200) responses\n"
)
status_counts = {}
for r in results:
    status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
summary += f"status_counts={status_counts}\n"

with open(os.path.join(outdir, "summary.txt"), "w") as f:
    f.write(summary)

print(summary)
PYEOF
