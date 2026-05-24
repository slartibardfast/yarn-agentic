---
name: Zero waste mantra for performance work
description: 100% CPU, 100% memory bandwidth utilization, not one wasted or repeated byte — core engineering principle for all MTP and inference optimization work
type: feedback
originSessionId: e6381887-6047-47bb-a3a3-a7bfce8e6af4
---
100% CPU utilization, 100% memory bandwidth utilization, and not one wasted or repeated byte.

**Why:** User's explicit engineering mantra for multi-GPU MTP draft throughput (Phase 36) and all performance-critical inference work. Every graph rebuild that could be reused, every PCIe transfer that could stay on-device, every scheduler alloc that could be skipped — these are the enemy.

**How to apply:** When designing or reviewing inference optimizations, audit every operation for: (1) is this work repeated? (2) does this data cross a bus it doesn't need to? (3) is this CPU/GPU idle when it could be doing useful work? No operation gets a pass for being "small" — cumulative waste is the bottleneck. This applies to draft generation, verify/accept paths, state checkpoint/restore, graph building, scheduler allocation, and any new MTP code paths.
