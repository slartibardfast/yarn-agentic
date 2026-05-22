#!/usr/bin/env python3
"""
validate-batch-composition-trace.py

T4 — NDJSON trace validator for the chunked-prefill admission scheduler
invariants.

Reads an NDJSON file (or stdin) produced by llama-server when run with
LLAMA_TRACE_NDJSON_DIR=<dir> set, and verifies the invariants from
/home/llm/yarn-agentic/specs/scheduler/batch_composition.allium and
/home/llm/yarn-agentic/specs/multislot/BatchComposition.tla (T4 form):

  - BatchCompositionInvariant: every Tick's batch satisfies the budget
    cap (sum of prefill chunk counts + len(decode set) <= K) AND
    per-token flag exclusivity (no slot contributes both prefill and
    decode in the same Tick).
  - TokenBudgetRespected: per-tick total tokens <= budget_k from
    the TickDispatch record header.
  - DecodePriorityAdmission: if any prefill chunk was admitted in a
    Tick, every PROCESSING slot listed in processing_set_at_start_of_tick
    is also in decode_slots.
  - PrefillCarryProgresses: across consecutive Ticks where a slot id S
    appears in prefill_counts with count > 0, the slot's
    cumulative n_prompt_processed is monotonically non-decreasing
    (carry never regresses).
  - PerTokenFlagExclusivity: a slot is in EITHER prefill_counts with
    count > 0 OR decode_slots, never both.

The pre-Tier-4 MixedBatchProhibition / AtMostOnePrefillSlotPerBatch /
DecodeHoldImpliedByPendingPrefill checks are gone; under T4 those
properties do NOT hold (mixed batches are explicitly admitted).

P0.B.S5 extension — graph-cache events from
/home/llm/yarn-agentic/specs/graphs/cuda_graph_reuse.allium and
/home/llm/yarn-agentic/specs/graphs/CUDAGraphReuse.tla:

  - CacheBounded: cache_size_post is bounded by GGML_CUDA_GRAPH_MAX
    (default 128) at every CaptureGraph / UpdateGraphExec /
    EvictGraphCacheEntry record.
  - DtypeStrictness: every UpdateGraphExec record's dtype matches the
    dtype of the original CaptureGraph record for the same
    topology_hash. (Mismatched dtype reuse is impossible —
    re-instantiate would be emitted as a fresh CaptureGraph.)
  - FifoEvictionOrdering: EvictGraphCacheEntry always drops the
    longest-resident entry. The validator tracks insertion order
    against the trace and verifies eviction picks the head.

P0.B.S5 extension — WarmUpRunIndex records for the GP3.g
byte-identity gate. When the trace contains WarmUpRunIndex markers,
the validator partitions the record stream into runs and compares
the post-WarmUp record sequence across runs; R1 ≡ R2 ≡ R3 must
match byte-identically.

NDJSON record schema (one JSON object per line):

    {"action": "ArrivePrompt",      "tick": N, "slot": S, "n_prompt": K}
    {"action": "ContributePrefill", "tick": N, "slot": S, "n_tokens":  K,
     "n_prompt_processed": K_prior, "loading_prompt_set": [S, ...],
     "processing_set": [S, ...]}
    {"action": "ContributeDecode",  "tick": N, "slot": S,
     "loading_prompt_set": [S, ...]}
    {"action": "TickDispatch",      "tick": N,
     "prefill_counts": {S: count, ...}, "decode_slots": [S, ...],
     "loading_prompt_set_at_start_of_tick": [S, ...],
     "processing_set_at_start_of_tick": [S, ...],
     "budget_k": K}

    All TickDispatch fields are REQUIRED under T4. Missing
    prefill_counts, processing_set_at_start_of_tick, or budget_k
    is a schema error (exit 2) — this catches trace-producer
    regressions loudly rather than silently relaxing the validator.
    {"action": "CompletePrefill",   "tick": N, "slot": S}
    {"action": "Release",           "tick": N, "slot": S}
    {"action": "CaptureGraph",      "tick": N,
     "topology_hash": H, "dtype": "F16|F32|...",
     "cache_size_post": K}
    {"action": "UpdateGraphExec",   "tick": N,
     "topology_hash": H, "dtype": "F16|F32|...",
     "cache_index": K}
    {"action": "EvictGraphCacheEntry", "tick": N,
     "evicted_topology_hash": H, "cache_size_post": K}
    {"action": "WarmUpRunIndex",    "run_index": R}

Note: WarmUpRunIndex carries no "tick" field — it's a stream-level
marker, not a tick-level event.

Exit code: 0 on PASS, 1 on any invariant violation, 2 on schema error.

Usage:
    validate-batch-composition-trace.py trace.ndjson
    cat trace.ndjson | validate-batch-composition-trace.py -
    validate-batch-composition-trace.py --max-cache-entries 128 trace.ndjson
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import Counter
from typing import Any


@dataclasses.dataclass
class Violation:
    invariant: str
    tick: int
    detail: str


def check_record_schema(rec: dict[str, Any], lineno: int) -> None:
    if "action" not in rec:
        raise ValueError(f"line {lineno}: missing 'action': {rec!r}")
    # WarmUpRunIndex is a stream-level marker and carries no tick.
    if rec["action"] != "WarmUpRunIndex" and "tick" not in rec:
        raise ValueError(
            f"line {lineno}: missing 'tick': {rec!r}"
        )


def validate(
    stream,
    max_cache_entries: int = 128,
) -> tuple[list[Violation], int]:
    violations: list[Violation] = []
    n_records = 0
    n_ticks = 0
    contrib_counter: Counter[int] = Counter()

    # Graph-cache state for CacheBounded + DtypeStrictness + FIFO order.
    # captured_dtype: topology_hash -> str (dtype at capture time).
    # cache_order: list of topology_hash in insertion order (oldest at index 0).
    captured_dtype: dict[int, str] = {}
    cache_order: list[int] = []

    # PrefillCarryProgresses tracking: per-slot last-seen
    # n_prompt_processed across the trace. Updated from ContributePrefill
    # records (which carry n_prompt_processed: K_prior + n_tokens admitted
    # this tick). Used to assert monotonic non-decrease across ticks.
    slot_last_n_prompt_processed: dict[int, int] = {}

    # WarmUpRunIndex partitioning: list[list[dict]] indexed by run_index.
    # records_per_run[i] = records observed during run i (excluding the
    # WarmUpRunIndex markers themselves).
    records_per_run: list[list[dict[str, Any]]] = []
    current_run_records: list[dict[str, Any]] | None = None

    for lineno, line in enumerate(stream, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"schema error line {lineno}: {e}", file=sys.stderr)
            sys.exit(2)
        check_record_schema(rec, lineno)
        n_records += 1

        action = rec["action"]

        # WarmUpRunIndex is a stream-level partition marker. Close out
        # the previous run (if any), open a fresh one, AND reset the
        # cache-tracking state — each run corresponds to a fresh server
        # boot (cache starts empty).
        if action == "WarmUpRunIndex":
            if current_run_records is not None:
                records_per_run.append(current_run_records)
            current_run_records = []
            captured_dtype.clear()
            cache_order.clear()
            contrib_counter["WarmUpRunIndex"] += 1
            continue

        # Append to the current run (if any is open). The
        # WarmUpRunIndex partition cross-cuts the per-record invariants
        # — the latter still run independently of run boundaries.
        if current_run_records is not None:
            current_run_records.append(rec)

        tick = rec["tick"]

        if action == "TickDispatch":
            n_ticks += 1
            # T4 required fields — hard-fail on absence so trace-producer
            # regressions surface loudly.
            for required in ("prefill_counts", "decode_slots",
                             "processing_set_at_start_of_tick", "budget_k"):
                if required not in rec:
                    print(
                        f"schema error line {lineno}: TickDispatch missing "
                        f"required field {required!r}: {rec!r}",
                        file=sys.stderr,
                    )
                    sys.exit(2)
            prefill_counts: dict[int, int] = {
                int(s): int(c) for s, c in rec["prefill_counts"].items()
            }
            decode_slots = list(rec["decode_slots"])
            processing_set = list(rec["processing_set_at_start_of_tick"])
            budget_k = int(rec["budget_k"])

            # TokenBudgetRespected — sum of prefill counts + decode set
            # cardinality bounded by K.
            total_tokens = sum(prefill_counts.values()) + len(decode_slots)
            if total_tokens > budget_k:
                violations.append(Violation(
                    invariant="TokenBudgetRespected",
                    tick=tick,
                    detail=(
                        f"sum(prefill_counts)={sum(prefill_counts.values())} + "
                        f"len(decode_slots)={len(decode_slots)} = "
                        f"{total_tokens} > budget_k={budget_k}"
                    ),
                ))

            # PerTokenFlagExclusivity — no slot in both prefill_counts
            # (count > 0) and decode_slots.
            decode_set = set(int(s) for s in decode_slots)
            for s, c in prefill_counts.items():
                if c > 0 and s in decode_set:
                    violations.append(Violation(
                        invariant="PerTokenFlagExclusivity",
                        tick=tick,
                        detail=(
                            f"slot={s} appears in both prefill_counts "
                            f"(count={c}) and decode_slots"
                        ),
                    ))

            # DecodePriorityAdmission — if any prefill admitted, every
            # PROCESSING slot at tick start is in decode_slots.
            any_prefill = any(c > 0 for c in prefill_counts.values())
            if any_prefill and processing_set:
                missing_decodes = [
                    s for s in processing_set if int(s) not in decode_set
                ]
                if missing_decodes:
                    violations.append(Violation(
                        invariant="DecodePriorityAdmission",
                        tick=tick,
                        detail=(
                            f"prefill admitted (sum="
                            f"{sum(prefill_counts.values())}) but PROCESSING "
                            f"slots {missing_decodes} from "
                            f"processing_set_at_start_of_tick are NOT in "
                            f"decode_slots={decode_slots}"
                        ),
                    ))

            contrib_counter["TickDispatch"] += 1

        elif action == "ContributePrefill":
            # PrefillCarryProgresses — monotonic non-decrease of per-slot
            # n_prompt_processed across the trace. Each ContributePrefill
            # carries n_prompt_processed (post-tick value: prior + this
            # tick's admission count).
            slot = rec.get("slot")
            n_tokens = rec.get("n_tokens", 0)
            n_prompt_processed = rec.get("n_prompt_processed")
            if slot is not None and n_prompt_processed is not None:
                slot_id = int(slot)
                last = slot_last_n_prompt_processed.get(slot_id)
                if last is not None and int(n_prompt_processed) < last:
                    violations.append(Violation(
                        invariant="PrefillCarryProgresses",
                        tick=tick,
                        detail=(
                            f"slot={slot_id} n_prompt_processed regressed "
                            f"from {last} to {n_prompt_processed} "
                            f"(this tick admitted {n_tokens} tokens)"
                        ),
                    ))
                slot_last_n_prompt_processed[slot_id] = int(n_prompt_processed)
            contrib_counter[action] += 1

        elif action == "CaptureGraph":
            topo = rec["topology_hash"]
            dtype = rec["dtype"]
            size_post = rec["cache_size_post"]

            # CacheBounded — never exceed the cap.
            if size_post > max_cache_entries:
                violations.append(Violation(
                    invariant="CacheBounded",
                    tick=tick,
                    detail=(
                        f"cache_size_post={size_post} exceeds "
                        f"max_cache_entries={max_cache_entries}"
                    ),
                ))

            # No-duplicate-topology — CaptureGraph fires only when
            # topology_hash isn't already in the cache. If we see a
            # duplicate, the dispatcher should have routed to
            # UpdateGraphExec or REINSTANTIATE instead.
            if topo in captured_dtype and topo in cache_order:
                violations.append(Violation(
                    invariant="AddressToleranceScopedToViewCpy",
                    tick=tick,
                    detail=(
                        f"duplicate CaptureGraph for topology_hash={topo} "
                        f"(should be UpdateGraphExec or REINSTANTIATE)"
                    ),
                ))
            # Even on the duplicate-violation path, update tracking so
            # subsequent records validate.
            captured_dtype[topo] = dtype
            if topo not in cache_order:
                cache_order.append(topo)

            contrib_counter["CaptureGraph"] += 1

        elif action == "UpdateGraphExec":
            topo = rec["topology_hash"]
            dtype = rec["dtype"]
            cache_index = rec.get("cache_index", -1)

            # DtypeStrictness — UpdateGraphExec dtype must match the
            # dtype the original CaptureGraph recorded for this topology.
            captured = captured_dtype.get(topo)
            if captured is None:
                violations.append(Violation(
                    invariant="ReuseImpliesPropertyMatch",
                    tick=tick,
                    detail=(
                        f"UpdateGraphExec for topology_hash={topo} "
                        f"with no prior CaptureGraph"
                    ),
                ))
            elif captured != dtype:
                violations.append(Violation(
                    invariant="DtypeStrictness",
                    tick=tick,
                    detail=(
                        f"UpdateGraphExec dtype={dtype!r} differs from "
                        f"captured dtype={captured!r} for "
                        f"topology_hash={topo}"
                    ),
                ))
            # cache_index sanity: must be in [0, current cache size).
            if cache_index < 0 or cache_index >= len(cache_order):
                violations.append(Violation(
                    invariant="CacheBounded",
                    tick=tick,
                    detail=(
                        f"UpdateGraphExec cache_index={cache_index} "
                        f"out of range [0, {len(cache_order)})"
                    ),
                ))
            contrib_counter["UpdateGraphExec"] += 1

        elif action == "EvictGraphCacheEntry":
            evicted = rec["evicted_topology_hash"]
            size_post = rec["cache_size_post"]

            # FIFO ordering — eviction picks the oldest (head) entry.
            if not cache_order:
                violations.append(Violation(
                    invariant="EventualEviction",
                    tick=tick,
                    detail="EvictGraphCacheEntry on empty cache",
                ))
            elif cache_order[0] != evicted:
                violations.append(Violation(
                    invariant="FifoEvictionOrdering",
                    tick=tick,
                    detail=(
                        f"evicted={evicted} but FIFO head is "
                        f"{cache_order[0]}"
                    ),
                ))
            else:
                cache_order.pop(0)
                captured_dtype.pop(evicted, None)

            # cache_size_post must match our tracking.
            if size_post != len(cache_order):
                violations.append(Violation(
                    invariant="CacheBounded",
                    tick=tick,
                    detail=(
                        f"cache_size_post={size_post} disagrees with "
                        f"tracked cache size {len(cache_order)}"
                    ),
                ))
            contrib_counter["EvictGraphCacheEntry"] += 1

        elif action in ("ContributeDecode",
                         "ArrivePrompt", "Release"):
            contrib_counter[action] += 1

        elif action == "CompletePrefill":
            # Reset carry tracking when the slot transitions to
            # PROCESSING — subsequent prompt arrivals on the same slot id
            # start fresh.
            slot = rec.get("slot")
            if slot is not None:
                slot_last_n_prompt_processed.pop(int(slot), None)
            contrib_counter[action] += 1

        else:
            print(
                f"warning line {lineno}: unknown action {action!r}",
                file=sys.stderr,
            )

    # Close out the final run, if any.
    if current_run_records is not None:
        records_per_run.append(current_run_records)

    # WarmUpRunIndex byte-identity check (GP3.g). Only fires when the
    # trace actually contains WarmUpRunIndex markers; absent that, the
    # validator is silent on this gate.
    if len(records_per_run) >= 2:
        ref = records_per_run[0]
        for i, run in enumerate(records_per_run[1:], start=1):
            if len(run) != len(ref):
                violations.append(Violation(
                    invariant="WarmUpRunByteIdentity",
                    tick=-1,
                    detail=(
                        f"run {i} has {len(run)} records; "
                        f"ref run 0 has {len(ref)}"
                    ),
                ))
                continue
            for j, (a, b) in enumerate(zip(ref, run)):
                if a != b:
                    violations.append(Violation(
                        invariant="WarmUpRunByteIdentity",
                        tick=a.get("tick", -1),
                        detail=(
                            f"run {i} record #{j} differs from ref run 0: "
                            f"{a!r} vs {b!r}"
                        ),
                    ))
                    break

    return violations, n_records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        help="NDJSON trace file. Pass '-' to read from stdin.",
    )
    parser.add_argument(
        "--max-cache-entries",
        type=int,
        default=128,
        help="GGML_CUDA_GRAPH_MAX bound (default 128) for CacheBounded check.",
    )
    args = parser.parse_args()

    if args.path == "-":
        stream = sys.stdin
    else:
        stream = open(args.path, "r", encoding="utf-8")

    try:
        violations, n_records = validate(
            stream, max_cache_entries=args.max_cache_entries
        )
    finally:
        if stream is not sys.stdin:
            stream.close()

    print(f"records: {n_records}", file=sys.stderr)
    if violations:
        for v in violations:
            print(
                f"VIOLATION {v.invariant} @ tick={v.tick}: {v.detail}",
                file=sys.stderr,
            )
        print(f"FAIL: {len(violations)} violation(s)", file=sys.stderr)
        return 1
    print("PASS: all batch-composition invariants hold", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
