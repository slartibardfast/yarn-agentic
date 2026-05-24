---
name: Multi-agent GPU sharing protocol (file-based, COORD.md)
description: How agents coordinate exclusive GPU access on this host — claim via flock+queue file, release via trap, heartbeat every 30s, priority-aware FIFO, 6800 XT for main / Vega for smoke, abort kill switch
type: reference
originSessionId: e06c78f3-43de-46c4-b7cf-81ea2dbd7d8f
---
Follow `/home/llm/yarn-agentic/COORD.md` before any GPU work. File-based
protocol — no daemon; state survives restarts and is inspectable via
`cat` / `ls`. The flock mechanic is POSIX-portable across shell-spawned
agents, git worktrees, and independent processes.

## Rules

1. **Claim**: before launching GPU work, append
   `<agent-id> <task-name> <est-duration> <priority>` to `coord/gpu.queue`,
   then try `flock coord/gpu.lock` on the device file
   (`coord/gpu-0.state` or `coord/gpu-1.state`). If blocked, `tail -f` the
   log and wait for a `RELEASED` event.

2. **Never overlap inference/bench on the same GPU** — per the existing
   `feedback_no_overlapping_benchmarks.md` memory. Quantize and
   perplexity both need exclusive access. Only one agent writes to each
   `gpu-N.state` at a time.

3. **Release**: on exit (normal or signal), the owner writes
   `RELEASED <agent-id> <device> <timestamp> <exit-status>` to
   `coord/gpu.log` and `echo IDLE > coord/gpu-N.state`. Use
   `trap 'release_gpu' EXIT INT TERM` so release happens even on crash.

4. **Heartbeat**: the running agent updates `coord/gpu-N.state` every
   30 s with `<agent-id>:<task>:<last-heartbeat-ts>`. Peers that see a
   stale heartbeat (> 5 min) may reclaim via a `STOLEN_STALE` audit entry
   in `coord/gpu.log`.

5. **Priorities**: T3 PPL runs (small, fast) outrank T4 35B-A3B quantize
   (long-running). Queue respects priority; equal priority is FIFO.

6. **Two-GPU explicit partitioning**: 6800 XT (16 GB) for main work;
   Vega (8 GB) for smoke tests. `COORD.md` lists which device each task
   class defaults to — avoids memory-spilling multi-GPU splits (see
   `project_mtp_ir_status.md` for the concrete regression that triggered
   this rule).

7. **Kill switch**: `coord/abort` — if the file is present, all agents
   release immediately and exit. User-operable emergency brake.

## Boilerplate to include in every GPU-using agent's prompt

```
Before any GPU work: follow /home/llm/yarn-agentic/COORD.md.
Acquire via coord/gpu-{device}.state, release on exit via trap.
Never run concurrent inference/bench on the same device.
If coord/abort exists, release and exit immediately.
```

## How to apply

- Any agent that runs `llama-server`, `llama-cli`, `llama-perplexity`,
  quantize, bench scripts, or any `-ngl > 0` binary MUST acquire before
  starting and release on exit.
- For quick reads (nvidia-smi / rocm-smi style checks) no claim needed.
- If you see a stale heartbeat on a device you need, inspect
  `coord/gpu.queue` first to confirm no higher-priority work is queued,
  then reclaim with a `STOLEN_STALE` log entry and proceed.
- CPU-only work (no GPU backend activated) doesn't use the protocol,
  but if you're unsure whether a binary touches GPU, claim anyway.
