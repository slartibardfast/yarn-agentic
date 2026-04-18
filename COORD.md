# GPU sharing protocol

File-based coordination for parallel agents / processes sharing the two local GPUs. No daemon — all state is plain files under `coord/`, so you can inspect with `cat` / `ls` and it survives restarts. Uses `flock(1)` for mutual exclusion.

## Devices

| Device | Label | Target class |
|---|---|---|
| AMD Radeon RX 6800 XT (RADV NAVI21, 16 GB) | **gpu-0** | Main work: 35B-A3B quantize, production perplexity, long-running benchmarks |
| AMD Radeon RX Vega (RADV VEGA10, 8 GB) | **gpu-1** | Smoke tests, 0.8B-scale perplexity, short-lived probes |

Tasks that request the "wrong" device by default still acquire the correct lock first; the device choice is advisory. Avoid memory-spilling multi-GPU splits unless the task explicitly needs both (reference: `project_mtp_ir_status.md` incident).

## Files

All under `coord/` (repo root). Recreated on first use if absent.

```
coord/
  gpu.lock            # flock advisory file — single point of serialisation
  gpu-0.state         # IDLE | <agent-id>:<task>:<last-heartbeat-ts>
  gpu-1.state         # IDLE | <agent-id>:<task>:<last-heartbeat-ts>
  gpu.queue           # append-only pending-claim log
  gpu.log             # append-only event log (claims, releases, steals)
  abort               # optional kill switch (see §7)
```

## Rules

### 1. Claim

Before launching any GPU work, agent:
1. Appends to `coord/gpu.queue`: `<timestamp> <agent-id> <task-name> <est-minutes> <priority>`
2. Acquires `flock` on `coord/gpu.lock`
3. Reads `coord/gpu-<N>.state`; if `IDLE`, writes `<agent-id>:<task>:<ts>` and releases the flock
4. If not `IDLE`, releases flock, then `tail -f coord/gpu.log | grep RELEASED` and retries when the target device frees
5. Logs `CLAIMED <agent-id> gpu-<N> <ts> <task>` to `coord/gpu.log`

### 2. No overlap on the same GPU

Per memory `feedback_no_overlapping_benchmarks.md`: **never run concurrent inference or benchmarks on the same GPU**. Quantize and perplexity both count as exclusive work; only one agent writes to each `gpu-N.state` at a time. Cross-GPU parallelism (gpu-0 quantize while gpu-1 smoke test) is allowed and encouraged.

### 3. Release

On exit (normal or signal), the owner:
1. Writes `RELEASED <agent-id> gpu-<N> <ts> <exit-status>` to `coord/gpu.log`
2. Writes `IDLE` to `coord/gpu-<N>.state`

Signal handler is mandatory:

```bash
release_gpu() {
    echo "RELEASED ${AGENT_ID} ${GPU_DEV} $(date -u +%s) ${1:-0}" >> coord/gpu.log
    echo "IDLE" > coord/gpu-${GPU_DEV#gpu-}.state
}
trap 'release_gpu 130' EXIT INT TERM
```

### 4. Heartbeat

While running, agent updates `coord/gpu-<N>.state` every 30 seconds with `<agent-id>:<task>:<now>`. A peer that sees a stale heartbeat (> 5 min old) audits via `gpu.log`, writes `STOLEN_STALE <stealer-id> gpu-<N> <prev-owner> <ts>` to the log, and takes over.

### 5. Priorities

At dequeue time, higher priority wins; equal priority is FIFO.

| Priority | Class |
|---|---|
| **P0** (highest) | Correctness gate runs (test-backend-ops, T1 NMSE harness) |
| P1 | T3 PPL on qwen35-0.8b (small, ~40–90 min) |
| P2 | Research variant quantize + PPL on qwen35-0.8b |
| P3 | T4 PPL on 35B-A3B (6–8 h) |
| P4 (lowest) | Long-running 35B-A3B quantize (non-critical variants) |

A P3/P4 task **must** release the GPU if a P0/P1 task claims: the owner checks `gpu.queue` every heartbeat and if a higher-priority task is waiting for the same device, it aborts cleanly, writes `PREEMPTED <owner> <by> <ts>` to `gpu.log`, and re-queues itself.

### 6. Two-GPU partitioning

Default targets (advisory):

| Task | Default device |
|---|---|
| test-backend-ops | gpu-0 (main target), but can run CPU-only so GPU acquisition is optional |
| T1/T2 NMSE harness | CPU-only — no lock needed |
| quantize (0.8B) | gpu-1 if idle, else gpu-0 |
| perplexity (0.8B) | gpu-1 by default |
| quantize (35B-A3B) | gpu-0 |
| perplexity (35B-A3B) | gpu-0 |
| Vulkan shader development | gpu-0 (NAVI21 has richer extensions) |

Memory-spilling multi-GPU splits: **only when the task explicitly needs both** (e.g., 35B-A3B inference that spills on gpu-0 alone). Claim *both* locks in the same `flock` block to avoid deadlock:

```bash
flock coord/gpu.lock -c '
  check gpu-0.state and gpu-1.state both IDLE
  else release and wait
  else mark both with heartbeat
'
```

### 7. Kill switch

If `coord/abort` exists, every agent immediately:
1. Stops its active GPU task
2. Runs its release handler
3. Exits with status 137

User-operable: `touch coord/abort` halts everything; `rm coord/abort` re-enables claims.

## Agent prompt boilerplate

Every agent prompt that may run GPU work gets this paragraph appended:

> Before any GPU work: follow `/home/llm/yarn-agentic/COORD.md`. Acquire via `coord/gpu-{device}.state`, release on exit via `trap`. Never run concurrent inference/bench on the same device. If `coord/abort` exists, release and exit immediately.

## Reference shell fragments

Claim script (POSIX sh):

```sh
AGENT_ID="${CLAUDE_AGENT_ID:-$$}"
COORD=/home/llm/yarn-agentic/coord
mkdir -p "$COORD"

claim_gpu() {
    local dev=$1 task=$2 est_min=$3 prio=${4:-P2}
    [ -f "$COORD/abort" ] && { echo "abort set"; exit 137; }
    echo "$(date -u +%s) $AGENT_ID $task $est_min $prio" >> "$COORD/gpu.queue"
    (
        flock 9
        while :; do
            state=$(cat "$COORD/gpu-${dev#gpu-}.state" 2>/dev/null || echo IDLE)
            [ "$state" = "IDLE" ] && { echo "$AGENT_ID:$task:$(date -u +%s)" > "$COORD/gpu-${dev#gpu-}.state"; echo "CLAIMED $AGENT_ID $dev $(date -u +%s) $task" >> "$COORD/gpu.log"; return 0; }
            flock -u 9
            sleep 10
            flock 9
        done
    ) 9>"$COORD/gpu.lock"
    GPU_DEV=$dev
    export GPU_DEV AGENT_ID
    trap "echo 'RELEASED $AGENT_ID $GPU_DEV \$(date -u +%s) \$?' >> $COORD/gpu.log; echo IDLE > $COORD/gpu-${dev#gpu-}.state" EXIT INT TERM
}
```

Heartbeat loop (run in background):

```sh
heartbeat_loop() {
    while :; do
        [ -f "$COORD/abort" ] && exit 137
        echo "$AGENT_ID:$TASK:$(date -u +%s)" > "$COORD/gpu-${GPU_DEV#gpu-}.state"
        sleep 30
    done &
    HB_PID=$!
    trap "kill $HB_PID 2>/dev/null; release_gpu \$?" EXIT INT TERM
}
```

## Example session

```sh
# Agent A: quantize 35B-A3B on gpu-0
claim_gpu gpu-0 "harp-2b-35b-quantize" 360 P4
heartbeat_loop
./bin/llama-quantize ... HARP_2B_S
# on exit, release_gpu fires via trap

# Agent B: concurrent smoke test on gpu-1
claim_gpu gpu-1 "0.8b-smoke" 3 P0
heartbeat_loop
./bin/llama-cli -m ... -p "hello"
```

## Inspecting state

```sh
# Who owns what right now
cat coord/gpu-0.state coord/gpu-1.state

# What's queued
tail coord/gpu.queue

# Recent activity
tail coord/gpu.log

# Abort state
ls coord/abort 2>/dev/null && echo "ABORT SET" || echo "running"
```
