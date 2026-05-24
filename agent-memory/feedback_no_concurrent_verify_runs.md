---
name: no-concurrent-verify-runs
description: Never launch a second verify-production-determinism / llama-batched-bench / similar GPU test before stopping the previous one. Concurrent runs produce within-NP even/odd slot divergence that masquerades as a real bug.
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

Never launch a second `verify-production-determinism.sh` (or any other
multi-process GPU benchmark/test) while a previous instance is still
running. They claim the SAME devices (`--device CUDA0,CUDA1`) AND the
same port (18292) AND issue `pkill -x llama-server` between phases.
Concurrent runs catastrophically interfere: one process's verify server
gets killed mid-decode by the other process's `pkill`, OR both end up
running servers on the same GPUs simultaneously, OR Run 2's pkill races
with Run 1's port bind.

**Why:** It already caused a false positive on 2026-05-21. I launched
`bau0n9ple` (background verify) then a minute later, before checking
its progress, launched `bvlepp0f3` (foreground retry). Both ran. The
first reported `divergence vs NP=1 at NP in {4}` with slots 0,2 in
Chinese text and slots 1,3 correct. I spent ~30k tokens hunting a
"stochastic NP=4 bug" before realising the runs had overlapped.

The diagnostic signature this produces — within-NP even/odd slot
divergence (slots {0,2} ≢ slots {1,3} **within one NP value**) — is
distinct from the legitimate cluster-partition signature in
[[feedback_np_cluster_partition_signature]] (which is across-NP-values:
`NP=A ≡ NP=B ≢ NP=C`). If a "stochastic NP failure" appears AND a
second test had been launched in the same time window, the contamination
explanation comes first.

**How to apply:**

- Before launching any verify-production-determinism /
  llama-batched-bench / DFlash-server-multi-slot test, run
  `pgrep -af "verify-production|llama-server|llama-batched"`. If the
  list isn't empty, `TaskStop` any tracked background task or `kill`
  the stray process, then verify GPUs idle with `nvidia-smi`.
- For the sequential N-iter case (measuring failure rate of a test),
  use a bash for-loop that runs serially — like
  `scripts/probe-np4-stochastic.sh` — rather than firing concurrent
  background tasks.
- Per `feedback_no_overlapping_benchmarks`, this is a one-strike rule.
  A script-level safeguard (have verify-production-determinism claim
  the same `coord/gpu-*.lock` files the rest of the harness uses)
  would make the mistake harder to make.

Related: [[feedback_no_overlapping_benchmarks]],
[[feedback_np_cluster_partition_signature]].
