---
name: feedback-split-equal-contiguous-seqs
description: "split_equal multi-seq dispatch requires batch seq_ids to be contiguous-from-0 in order ([0, n_seq_in_batch)). Non-contiguous sets fall back to single-seq dispatch. K/V view ne[3] = n_seq_in_batch, view offset is 0, stride is parent->nb[3]."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 4f2638d6-977b-48ae-9380-29a5b41d1c93
---

# split_equal grouping rule: contiguous-from-0 seq_ids

**Rule:** In `process_batch_tokens`, a multi-seq group is formed only when:
1. The first run's seq_id is 0.
2. Subsequent runs have seq_ids 1, 2, 3, ... in strict numerical order.
3. Per-seq token count is uniform across all runs in the group.
4. seq_ids do not repeat within the group.

If ANY of these fail, the dispatcher falls back to single-seq dispatch for the leading run and continues from the next run.

**Why:** The K/V view in the 4D build path has ne[3]=n_seq_in_batch, offset=0, stride=parent->nb[3]. This means ne[3] index `i` maps to stream `i` of the K/V cache. For the mapping to be correct, the batch's `i`-th seq must be stream `i`. Since slot_id==stream_id in our design, this requires batch seq_ids = [0, n_seq_in_batch) in order.

If you allowed seq_ids {0, 2, 5}, the FA kernel would read K from streams 0, 1, 2 instead of 0, 2, 5 — silent data corruption (no assertion, just wrong outputs).

**How to apply:** When adding any code that constructs multi-seq batches OR forms multi-seq dispatch groups, verify the seqs span `[0, n_seq_in_batch)`. The verify scenario at NP={1,2,4,8} satisfies this naturally because all slots prefill+decode together with slot_id=seq_id=stream_id mapping.

Production scenarios where this might NOT hold:
- A request finishes early; one slot becomes IDLE while others continue. The batch then has gaps (e.g., {0, 2, 3} if slot 1 finished). Multi-seq dispatch falls back to single-seq for those ticks. Acceptable — correctness preserved at the cost of the unified dispatch path being temporarily off until the IDLE slot is rebound to a new request.

**Future relaxation path (not landed):** Pass a per-seq stream-id table to FA so the K view can pick arbitrary streams. Requires kernel ABI change.
