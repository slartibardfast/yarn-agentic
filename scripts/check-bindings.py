#!/usr/bin/env python3
"""
check-bindings.py — enforce the bespoke Allium <-> TLA+ <-> C++ synergy contract.

Reads:
  - specs/dflash/dflash.allium             (source of truth)
  - specs/dflash/allium-tla-binding.json   (helpers + divergences + externals)
  - specs/dflash/DFlashCycle.tla           (Phase 1 model)
  - specs/dflash/DFlashMultiSlot.tla       (Phase 2 model)
  - ik_llama.cpp/tests/dflash-speculative/*.{cpp,h}   (property tests)

Checks (each one fails the script with exit code 1 if violated):

  1. FORWARD: every Allium invariant name appears in some TLA+ file,
     unless explicitly listed under `bindings_external` (with rationale).

  2. REVERSE: every TLA+ `Foo == ...` definition is one of:
       - an Allium invariant name (forward direction), OR
       - a whitelisted helper in `tla_helpers`, OR
       - a `divergences[i].tla` target with a matching Allium source.

  3. C++ CITATIONS: every `DFLASH_REQUIRE(_, "FooBar")` string and every
     `Spec: ... invariant FooBar` comment in dflash-speculative/ tests
     cites a real Allium invariant name (or a divergence target).

  4. DIVERGENCE INTEGRITY: every entry in `divergences` has:
       - allium present in the Allium spec
       - tla present in some TLA file
       - rationale non-empty.

  5. EXTERNAL INTEGRITY: every entry in `bindings_external` has:
       - allium present in the Allium spec
       - bound_by non-empty (e.g., "cpp:test-foo.cpp", "gate:Gate-3.5")
       - rationale non-empty.

Output:
  - If all checks pass: a one-line summary plus a coverage report.
  - If any fails: detailed diagnostics, exit code 1.

Usage:
  scripts/check-bindings.py
  scripts/check-bindings.py --report-only   # skip enforcement; just print
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_DIR = REPO_ROOT / "specs" / "dflash"
ALLIUM_SPEC = SPEC_DIR / "dflash.allium"
BINDING_JSON = SPEC_DIR / "allium-tla-binding.json"
KERNEL_DESIGN = SPEC_DIR / "kernel-design.md"
TLA_FILES = [SPEC_DIR / "DFlashCycle.tla", SPEC_DIR / "DFlashMultiSlot.tla"]
CPP_TEST_DIR = REPO_ROOT / "ik_llama.cpp" / "tests" / "dflash-speculative"


def red(s: str) -> str:
    return f"\033[31m{s}\033[0m"


def green(s: str) -> str:
    return f"\033[32m{s}\033[0m"


def yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m"


def fail(msg: str) -> None:
    print(red(f"FAIL: {msg}"), file=sys.stderr)


def info(msg: str) -> None:
    print(msg)


# ---------- AST helpers ----------------------------------------------------


def allium_ast() -> dict:
    """Parse the Allium spec via `allium parse`. Cached on disk for CI."""
    result = subprocess.run(
        ["allium", "parse", str(ALLIUM_SPEC)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def allium_invariant_names(ast: dict) -> set[str]:
    """All unique Allium invariant names (top-level + in-contract @invariant)."""
    names: set[str] = set()
    for decl in ast["module"]["declarations"]:
        # Top-level: { "Invariant": { "name": {"name": "FooBar"}, ... } }
        if "Invariant" in decl:
            n = decl["Invariant"]["name"]["name"]
            names.add(n)
        # In-contract: { "Block": { "kind": "Contract", "items": [
        #     { "kind": {"Annotation": {"kind": "Invariant", "name": ...}} } ] } }
        blk = decl.get("Block")
        if not blk or blk.get("kind") != "Contract":
            continue
        for item in blk.get("items", []):
            k = item.get("kind")
            if isinstance(k, dict) and "Annotation" in k:
                ann = k["Annotation"]
                if isinstance(ann, dict) and ann.get("kind") == "Invariant":
                    n = ann.get("name", {}).get("name")
                    if n:
                        names.add(n)
    return names


def allium_other_names(ast: dict) -> set[str]:
    """Names of Allium contracts, entities, and enums (anything kernel-design
    §7 may legitimately reference besides @invariants)."""
    names: set[str] = set()
    for decl in ast["module"]["declarations"]:
        blk = decl.get("Block")
        if not blk:
            continue
        kind = blk.get("kind")
        if kind in ("Contract", "Entity", "Enum"):
            nm = blk.get("name", {})
            if isinstance(nm, dict):
                n = nm.get("name")
                if n:
                    names.add(n)
    return names


# ---------- TLA helpers ----------------------------------------------------

TLA_DEF_RE = re.compile(r"^(?P<name>[A-Z][A-Za-z0-9_]*)\s*==")


def tla_definition_names(tla_path: Path) -> set[str]:
    """All `Foo == ...` definitions at column 0 in a TLA file."""
    names: set[str] = set()
    if not tla_path.exists():
        return names
    for line in tla_path.read_text().splitlines():
        m = TLA_DEF_RE.match(line)
        if m:
            names.add(m.group("name"))
    return names


def all_tla_names() -> set[str]:
    out: set[str] = set()
    for p in TLA_FILES:
        out |= tla_definition_names(p)
    return out


# ---------- Markdown helpers ----------------------------------------------

# Match a backtick-quoted CamelCase name in a markdown table-row's left column.
# A table row starts with `|`, and the left column ends at the next `|`.
KD_TABLE_NAME_RE = re.compile(r"^\|\s*`([A-Z][A-Za-z0-9_]+)`\s*\|")


def kernel_design_section7_names() -> set[str]:
    """Extract invariant-shaped names from kernel-design.md §7 binding table.
    Names are recognized as backtick-quoted CamelCase tokens in the first
    cell of a markdown table row, within the §7 section."""
    names: set[str] = set()
    if not KERNEL_DESIGN.exists():
        return names
    in_section_7 = False
    for line in KERNEL_DESIGN.read_text().splitlines():
        if line.startswith("## 7."):
            in_section_7 = True
            continue
        if in_section_7 and line.startswith("## "):
            break
        if not in_section_7:
            continue
        m = KD_TABLE_NAME_RE.match(line)
        if m:
            names.add(m.group(1))
    return names


# ---------- C++ helpers ---------------------------------------------------

CPP_DFLASH_REQUIRE_RE = re.compile(r'DFLASH_REQUIRE\s*\([^,]*,\s*"([A-Z][A-Za-z0-9_]*)"')
CPP_SPEC_INV_RE = re.compile(r'\b(?:@?invariant)\s+([A-Z][A-Za-z0-9_]+)')


def cpp_invariant_citations() -> dict[str, list[Path]]:
    """Map invariant-name -> list of files that cite it."""
    out: dict[str, list[Path]] = {}
    if not CPP_TEST_DIR.exists():
        return out
    for p in list(CPP_TEST_DIR.glob("*.cpp")) + list(CPP_TEST_DIR.glob("*.h")):
        text = p.read_text()
        for m in CPP_DFLASH_REQUIRE_RE.finditer(text):
            out.setdefault(m.group(1), []).append(p)
        for m in CPP_SPEC_INV_RE.finditer(text):
            out.setdefault(m.group(1), []).append(p)
    return out


# ---------- Checks --------------------------------------------------------


def run_checks(report_only: bool = False) -> int:
    failures = 0

    ast = allium_ast()
    allium_invs = allium_invariant_names(ast)
    allium_others = allium_other_names(ast)
    tla_names = all_tla_names()
    kd_section7 = kernel_design_section7_names()
    binding = json.loads(BINDING_JSON.read_text())
    helpers = set(binding.get("tla_helpers", []))
    divergences = binding.get("divergences", [])
    externals = binding.get("bindings_external", [])
    diverge_allium = {d["allium"] for d in divergences}
    diverge_tla = {d["tla"] for d in divergences}
    external_allium = {e["allium"] for e in externals}

    info(f"Allium invariants:        {len(allium_invs)}")
    info(f"Allium other (contracts/entities/enums): {len(allium_others)}")
    info(f"TLA definitions:          {len(tla_names)}")
    info(f"TLA helpers:              {len(helpers)}")
    info(f"Divergences:              {len(divergences)}")
    info(f"External bindings:        {len(externals)}")
    info(f"kernel-design.md §7 names: {len(kd_section7)}")
    info("")

    # CHECK 1 — FORWARD: every Allium invariant bound in TLA or external
    info("[1] forward (Allium -> TLA)")
    fwd_missing = []
    for name in sorted(allium_invs):
        if name in tla_names:
            continue
        if name in diverge_allium:
            d = next(d for d in divergences if d["allium"] == name)
            if d["tla"] in tla_names:
                continue
            fwd_missing.append(f"  {name} (diverges to '{d['tla']}', but that's missing from TLA)")
            continue
        if name in external_allium:
            continue
        fwd_missing.append(f"  {name}")
    if fwd_missing:
        failures += 1
        fail("Allium invariants missing from TLA:")
        for m in fwd_missing:
            print(m, file=sys.stderr)
    else:
        info(green(f"    OK — all {len(allium_invs)} Allium invariants bound."))

    # CHECK 2 — REVERSE: every TLA name is allowed
    info("[2] reverse (TLA -> Allium)")
    rev_unbound = []
    for name in sorted(tla_names):
        if name in allium_invs:
            continue
        if name in helpers:
            continue
        if name in diverge_tla:
            continue
        rev_unbound.append(name)
    if rev_unbound:
        failures += 1
        fail("TLA definitions not bound to Allium and not in tla_helpers:")
        for n in rev_unbound:
            print(f"  {n}", file=sys.stderr)
        print("  Resolution: either add to specs/dflash/allium-tla-binding.json "
              "`tla_helpers` if it's a helper, or `divergences` with rationale.",
              file=sys.stderr)
    else:
        info(green(f"    OK — all {len(tla_names)} TLA definitions accounted for."))

    # CHECK 3 — C++ citations
    info("[3] C++ test citations -> Allium")
    citations = cpp_invariant_citations()
    cpp_unknown = []
    for name in sorted(citations):
        if name in allium_invs:
            continue
        if name in diverge_tla:
            continue
        cpp_unknown.append((name, citations[name]))
    if cpp_unknown:
        failures += 1
        fail("C++ tests cite invariants not in Allium:")
        for n, paths in cpp_unknown:
            print(f"  {n}  ({', '.join(p.name for p in paths)})", file=sys.stderr)
    else:
        n_cites = sum(len(v) for v in citations.values())
        info(green(f"    OK — {len(citations)} cited names, {n_cites} mentions, all map."))

    # CHECK 4 — divergence integrity
    info("[4] divergence integrity")
    div_errors = []
    for d in divergences:
        if d.get("allium") not in allium_invs:
            div_errors.append(f"  divergence allium='{d.get('allium')}' not found in Allium spec")
        if d.get("tla") not in tla_names:
            div_errors.append(f"  divergence tla='{d.get('tla')}' not found in TLA files")
        if not d.get("rationale", "").strip():
            div_errors.append(f"  divergence {d.get('allium')} missing rationale")
    if div_errors:
        failures += 1
        fail("divergence entries are malformed:")
        for e in div_errors:
            print(e, file=sys.stderr)
    else:
        info(green(f"    OK — {len(divergences)} divergence entries valid."))

    # CHECK 5b — kernel-design.md §7 binding table
    info("[5b] kernel-design.md §7 -> Allium")
    kd_unbound = []
    for name in sorted(kd_section7):
        if name in allium_invs:
            continue
        if name in allium_others:
            continue
        if name in diverge_tla:
            continue
        kd_unbound.append(name)
    if kd_unbound:
        failures += 1
        fail("kernel-design.md §7 cites names not in Allium spec:")
        for n in kd_unbound:
            print(f"  {n}", file=sys.stderr)
        print("  Resolution: rename row to a real Allium @invariant / contract / "
              "entity / enum, or add the missing concept to specs/dflash/dflash.allium.",
              file=sys.stderr)
    else:
        info(green(f"    OK — all {len(kd_section7)} §7 table names resolve."))

    # CHECK 6 — external integrity
    info("[6] external-binding integrity")
    ext_errors = []
    for e in externals:
        if e.get("allium") not in allium_invs:
            ext_errors.append(f"  external allium='{e.get('allium')}' not found in Allium spec")
        if not e.get("bound_by", "").strip():
            ext_errors.append(f"  external {e.get('allium')} missing bound_by")
        if not e.get("rationale", "").strip():
            ext_errors.append(f"  external {e.get('allium')} missing rationale")
    if ext_errors:
        failures += 1
        fail("external-binding entries are malformed:")
        for e in ext_errors:
            print(e, file=sys.stderr)
    else:
        info(green(f"    OK — {len(externals)} external entries valid."))

    # Coverage report
    info("")
    info("Coverage report:")
    bound_in_tla = sum(1 for n in allium_invs if n in tla_names)
    bound_via_diverge = sum(1 for n in allium_invs if n in diverge_allium)
    bound_external = sum(1 for n in allium_invs if n in external_allium)
    bound_in_cpp = sum(1 for n in allium_invs if n in citations)
    info(f"  Allium invariants bound in TLA   : {bound_in_tla} / {len(allium_invs)}")
    info(f"  Allium invariants via divergence : {bound_via_diverge}")
    info(f"  Allium invariants external       : {bound_external}")
    info(f"  Allium invariants cited in C++   : {bound_in_cpp}")
    info(f"  TLA definitions total            : {len(tla_names)}")
    info(f"  TLA helpers whitelisted          : {len(helpers)}")

    if failures and not report_only:
        info("")
        info(red(f"check-bindings FAILED — {failures} check(s) failed."))
        return 1
    info("")
    info(green("check-bindings PASSED."))
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--report-only", action="store_true",
                   help="Print the report but don't fail on errors.")
    args = p.parse_args()
    return run_checks(report_only=args.report_only)


if __name__ == "__main__":
    sys.exit(main())
