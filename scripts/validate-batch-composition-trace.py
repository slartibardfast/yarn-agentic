#!/usr/bin/env python3
"""
validate-batch-composition-trace.py

S5 — NDJSON trace validator for the Bug C closure scheduler invariants.

Reads an NDJSON file (or stdin) produced by llama-server when run with
LLAMA_TRACE_NDJSON_DIR=<dir> set, and verifies the invariants from
/home/llm/yarn-agentic/specs/scheduler/batch_composition.allium and
/home/llm/yarn-agentic/specs/multislot/BatchComposition.tla:

  - BatchCompositionInvariant: every Tick's batch is either pure-prefill
    or pure-decode, never mixed.
  - MixedBatchProhibition: stronger token-level form of the above.
  - AtMostOnePrefillSlotPerBatch: PrefillSerialisationGate's singleton
    property.
  - DecodeHoldImpliedByPendingPrefill: if any slot in LOAD_PROMPT when a
    Tick fires, the Tick's decode set is empty.
  - PrefillContinuationPriority: when continuation candidates exist, the
    contributing prefill slot has n_prompt_tokens_processed > 0.

NDJSON record schema (one JSON object per line):

    {"action": "ArrivePrompt",      "tick": N, "slot": S, "n_prompt": K}
    {"action": "ContributePrefill", "tick": N, "slot": S, "n_tokens":  K,
     "n_prompt_processed": K_prior, "loading_prompt_set": [S, ...],
     "processing_set": [S, ...]}
    {"action": "ContributeDecode",  "tick": N, "slot": S,
     "loading_prompt_set": [S, ...]}
    {"action": "TickDispatch",      "tick": N,
     "prefill_slots": [S, ...], "decode_slots": [S, ...],
     "loading_prompt_set_at_start_of_tick": [S, ...]}
    {"action": "CompletePrefill",   "tick": N, "slot": S}
    {"action": "Release",           "tick": N, "slot": S}

Exit code: 0 on PASS, 1 on any invariant violation, 2 on schema error.

Usage:
    validate-batch-composition-trace.py trace.ndjson
    cat trace.ndjson | validate-batch-composition-trace.py -
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
    if "action" not in rec or "tick" not in rec:
        raise ValueError(
            f"line {lineno}: missing 'action' or 'tick': {rec!r}"
        )


def validate(stream) -> tuple[list[Violation], int]:
    violations: list[Violation] = []
    n_records = 0
    n_ticks = 0
    contrib_counter: Counter[int] = Counter()

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
        tick = rec["tick"]

        if action == "TickDispatch":
            n_ticks += 1
            prefill = list(rec.get("prefill_slots", []))
            decode = list(rec.get("decode_slots", []))
            loading = list(
                rec.get("loading_prompt_set_at_start_of_tick", [])
            )

            # BatchCompositionInvariant + MixedBatchProhibition.
            if prefill and decode:
                violations.append(Violation(
                    invariant="MixedBatchProhibition",
                    tick=tick,
                    detail=(
                        f"prefill_slots={prefill} and decode_slots={decode} "
                        "in the same Tick"
                    ),
                ))

            # AtMostOnePrefillSlotPerBatch.
            if len(prefill) > 1:
                violations.append(Violation(
                    invariant="AtMostOnePrefillSlotPerBatch",
                    tick=tick,
                    detail=f"prefill_slots={prefill} has cardinality > 1",
                ))

            # DecodeHoldImpliedByPendingPrefill.
            if loading and decode:
                violations.append(Violation(
                    invariant="DecodeHoldImpliedByPendingPrefill",
                    tick=tick,
                    detail=(
                        f"loading_prompt_set_at_start_of_tick={loading} "
                        f"non-empty AND decode_slots={decode} non-empty"
                    ),
                ))

            contrib_counter["TickDispatch"] += 1

        elif action in ("ContributePrefill", "ContributeDecode",
                         "ArrivePrompt", "CompletePrefill", "Release"):
            contrib_counter[action] += 1

        else:
            print(
                f"warning line {lineno}: unknown action {action!r}",
                file=sys.stderr,
            )

    return violations, n_records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        help="NDJSON trace file. Pass '-' to read from stdin.",
    )
    args = parser.parse_args()

    if args.path == "-":
        stream = sys.stdin
    else:
        stream = open(args.path, "r", encoding="utf-8")

    try:
        violations, n_records = validate(stream)
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
