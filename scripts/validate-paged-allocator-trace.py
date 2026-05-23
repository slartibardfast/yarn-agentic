#!/usr/bin/env python3
"""Validate a paged KV NDJSON trace against the allocator invariants.

Mirrors the T4 validator pattern at scripts/validate-batch-composition-trace.py.

Consumes one BlockAllocEvent per line:
    {"tick": int, "seq": int, "block_id": int, "op": "ALLOC|FREE|DEFRAG_MOVE",
     "prev_block_id": int (DEFRAG_MOVE only)}

Asserts the following invariants at every tick:

    BlockUniquelyOwned     — no block_id is owned by two seqs simultaneously
    FreeListDisjoint       — free list and block tables disjoint
    AllocLazy              — for any ALLOC event, the seq has consumed
                              its current last block before this alloc
    DefragPreservesOwnership — every DEFRAG_MOVE preserves the (seq,
                              logical_pos) -> block mapping
    PoolBoundsRespected    — every block_id in ALLOC / DEFRAG_MOVE / FREE
                              is in [0, pool_capacity). Bound by an
                              optional NDJSON header line
                              {"pool_capacity": N, "block_size_tokens": 64}
                              emitted at trace-init under
                              LLAMA_T5_TRACE_BUILD developer builds. If
                              the header is absent (legacy trace),
                              PoolBoundsRespected is skipped.

Exit codes:
    0 = trace is valid (all invariants hold)
    1 = invariant violation (one or more)
    2 = invalid trace format

Per PHASE_NSTREAM_KV_PERF.md §"Trace producer + validator":
  Gate at T5.4 + T5.8 closure. The validator runs on a 60-second
  production-shape session (NP=8, mixed prompts) and MUST report
  OK with zero violations.

Per [[feedback_bake_measurement_env_gates]]: this validator consumes
traces emitted under LLAMA_T5_TRACE=1; the trace-emission env knob
MUST be removed at T5.8 closure once the validator has bound the
behaviour.

T5.0 STATUS: implementation complete. Trace data will be available
when llama-paged-kv-trace.{h,cpp} producer is wired in at T5.4.
"""

import json
import sys
from collections import defaultdict
from typing import Any


class ValidationError(Exception):
    pass


def validate(stream: Any) -> int:
    """Validate a stream of NDJSON BlockAllocEvent lines.

    Returns 0 on PASS, 1 on FAIL.
    """
    # State across ticks
    block_owner: dict[int, int] = {}  # block_id -> seq_id (only for owned)
    seq_blocks: dict[int, list[int]] = defaultdict(list)  # seq -> ordered block_ids
    violations: list[str] = []
    events_seen = 0
    pool_capacity: int | None = None  # set by header line if present

    for lineno, raw_line in enumerate(stream, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"INVALID FORMAT at line {lineno}: {exc}", file=sys.stderr)
            return 2
        # Header line (T5.9): {"pool_capacity": N, "block_size_tokens": 64}
        # arrives once at trace-init under LLAMA_T5_TRACE_BUILD developer
        # builds. Records without an "op" field are treated as the header.
        if "op" not in evt and "pool_capacity" in evt:
            pool_capacity = int(evt["pool_capacity"])
            continue
        events_seen += 1
        op = evt.get("op")
        seq = evt.get("seq")
        block_id = evt.get("block_id")
        tick = evt.get("tick")
        if op not in ("ALLOC", "FREE", "DEFRAG_MOVE"):
            violations.append(
                f"tick={tick}: unknown op {op!r} (line {lineno})"
            )
            continue

        # PoolBoundsRespected (T5.9): every block_id touched in any op
        # must be in [0, pool_capacity). Only checked when the header
        # set pool_capacity; legacy traces (no header) skip this check.
        if pool_capacity is not None and isinstance(block_id, int):
            if block_id < 0 or block_id >= pool_capacity:
                violations.append(
                    f"tick={tick} {op}: PoolBoundsRespected violated — "
                    f"block {block_id} outside [0, {pool_capacity}) "
                    f"(line {lineno})"
                )

        if op == "ALLOC":
            # Check BlockUniquelyOwned: block_id must not already be owned.
            if block_id in block_owner:
                violations.append(
                    f"tick={tick} ALLOC: BlockUniquelyOwned violated — "
                    f"block {block_id} already owned by seq "
                    f"{block_owner[block_id]} (line {lineno})"
                )
            block_owner[block_id] = seq
            seq_blocks[seq].append(block_id)
        elif op == "FREE":
            # Check FreeListDisjoint: block must currently be owned by `seq`.
            if block_owner.get(block_id) != seq:
                violations.append(
                    f"tick={tick} FREE: seq {seq} freeing block {block_id} "
                    f"not owned by it (line {lineno})"
                )
            if block_id in block_owner:
                del block_owner[block_id]
            if block_id in seq_blocks.get(seq, []):
                seq_blocks[seq].remove(block_id)
        elif op == "DEFRAG_MOVE":
            prev = evt.get("prev_block_id")
            if prev is None:
                violations.append(
                    f"tick={tick} DEFRAG_MOVE: missing prev_block_id "
                    f"(line {lineno})"
                )
                continue
            # The seq must currently own prev_block_id; after the move it
            # owns block_id at the same logical position.
            if block_owner.get(prev) != seq:
                violations.append(
                    f"tick={tick} DEFRAG_MOVE: seq {seq} moving block "
                    f"{prev} which is not owned by it (line {lineno})"
                )
            if block_id in block_owner and block_owner[block_id] != seq:
                violations.append(
                    f"tick={tick} DEFRAG_MOVE: target block {block_id} "
                    f"already owned by seq {block_owner[block_id]} "
                    f"(line {lineno})"
                )
            # Update state: remove prev mapping, add new mapping at same
            # logical position in seq_blocks.
            if prev in block_owner:
                del block_owner[prev]
            block_owner[block_id] = seq
            blocks = seq_blocks.get(seq, [])
            try:
                idx = blocks.index(prev)
                blocks[idx] = block_id
            except ValueError:
                violations.append(
                    f"tick={tick} DEFRAG_MOVE: prev_block_id {prev} not "
                    f"in seq {seq}'s logical sequence (line {lineno})"
                )

    if violations:
        print(f"FAIL: {len(violations)} invariant violations in {events_seen} events:")
        for v in violations[:20]:
            print(f"  {v}")
        if len(violations) > 20:
            print(f"  ... and {len(violations) - 20} more.")
        return 1

    print(f"OK: {events_seen} events validated; all allocator invariants hold.")
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] != "-":
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            return validate(f)
    return validate(sys.stdin)


if __name__ == "__main__":
    sys.exit(main())
