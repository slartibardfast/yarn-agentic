#!/usr/bin/env python3
# T6.1 cell validator — checks every cell.json against the schema-shape
# contract enumerated in specs/t6-characterisation-cell.allium.
#
# Usage:
#   scripts/validate-t6-cell.py data/t6.1-matrix/<cell_dir>/cell.json [...]
#   scripts/validate-t6-cell.py data/t6.1-matrix/   # walks all cell.json
#
# Exit code:
#   0 = every cell binds
#   1 = at least one cell fails
#
# This is a lightweight shape+enum checker — the authoritative spec is
# specs/t6-characterisation-cell.allium. The script encodes the same
# contracts (SchemaShapeBound, EnumValuesRespected, BuildIsCommitHash,
# BindingClocksLocked, AggregatesDerivedFromPerRequest) in Python so a
# cell can be validated standalone without booting allium.

import json
import re
import sys
from pathlib import Path

VALID_ENGINES = {"ik_llama", "vllm"}
VALID_FIRE_PATTERNS = {"concurrent", "staggered_Ns", "sequential"}
BINDING_CLOCK_MHZ = 1455
HASH_RE = re.compile(r"^[0-9a-f]{7,40}$")

TOP_LEVEL_KEYS = {
    "cell_id", "timestamp", "engine", "engine_build",
    "model", "config", "clocks", "workload",
    "results", "instrumentation", "notes",
}
RESULTS_PER_REQUEST_KEY = "per_request"  # nested under results in current harness

NESTED_OBJECTS = ("model", "config", "clocks", "workload", "results", "instrumentation")


def check_cell(path):
    errs = []
    warns = []
    try:
        with open(path) as f:
            cell = json.load(f)
    except Exception as e:
        return [f"{path}: cannot read/parse: {e}"], []

    # SchemaShapeBound::AllTopLevelFieldsPresent
    missing = TOP_LEVEL_KEYS - set(cell.keys())
    if missing:
        errs.append(f"missing top-level keys: {sorted(missing)}")

    # SchemaShapeBound::NestedEntitiesNonNull
    for k in NESTED_OBJECTS:
        if cell.get(k) is None:
            errs.append(f"nested entity '{k}' is null/missing at top level")
        elif not isinstance(cell[k], dict):
            errs.append(f"'{k}' is not an object")

    # per_request lives nested under results in the harness output
    results = cell.get("results", {}) or {}
    per_req = results.get(RESULTS_PER_REQUEST_KEY)
    if per_req is None or not isinstance(per_req, list):
        errs.append("results.per_request missing or not a list")

    # EnumValuesRespected
    eng = cell.get("engine")
    if eng not in VALID_ENGINES:
        errs.append(f"engine '{eng}' not in {sorted(VALID_ENGINES)}")
    fp = (cell.get("workload") or {}).get("fire_pattern")
    if fp not in VALID_FIRE_PATTERNS:
        errs.append(f"workload.fire_pattern '{fp}' not in {sorted(VALID_FIRE_PATTERNS)}")

    # BuildIsCommitHash
    eb = cell.get("engine_build") or ""
    if not HASH_RE.match(eb):
        errs.append(f"engine_build '{eb}' is not a hex commit hash")

    # BindingClocksLocked
    clocks = cell.get("clocks") or {}
    if clocks.get("locked") is True:
        if clocks.get("gpu_mhz") != BINDING_CLOCK_MHZ:
            errs.append(f"clocks.locked=true but gpu_mhz={clocks.get('gpu_mhz')} != {BINDING_CLOCK_MHZ}")
    else:
        notes = cell.get("notes") or ""
        if "unlocked" not in notes.lower():
            warns.append("clocks.locked is not true and notes don't mention unlocked-clocks intent")

    # AggregatesDerivedFromPerRequest::TokenCountsSum (when per_request non-empty)
    if isinstance(per_req, list) and per_req:
        sum_tokens = sum(r.get("tokens", 0) for r in per_req if r.get("status") == 200)
        tot = results.get("total_output_toks")
        if tot is not None and sum_tokens != tot:
            errs.append(f"total_output_toks={tot} != sum(per_request.tokens for 200)={sum_tokens}")

    return errs, warns


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)

    targets = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            targets.extend(sorted(p.glob("**/cell.json")))
        elif p.is_file():
            targets.append(p)
        else:
            print(f"WARN: '{arg}' not found, skipping", file=sys.stderr)

    if not targets:
        print("no cell.json files to validate", file=sys.stderr)
        sys.exit(2)

    n_pass = 0
    n_fail = 0
    for path in targets:
        errs, warns = check_cell(path)
        if errs:
            n_fail += 1
            print(f"FAIL  {path}")
            for e in errs:
                print(f"      {e}")
        else:
            n_pass += 1
            print(f"PASS  {path}")
        for w in warns:
            print(f"WARN  {path}")
            print(f"      {w}")

    print(f"\n{n_pass} pass, {n_fail} fail of {n_pass + n_fail} cells")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
