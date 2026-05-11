#!/usr/bin/env python3
"""
allium-to-tla.py — generate TLA+ skeleton from an Allium spec.

Reads `allium parse <file>.allium` JSON AST and emits a TLA+ module
skeleton with:

  - Module preamble + EXTENDS
  - CONSTANTS (one per enum + caller-fillable bound constants)
  - VARIABLES suggestion list (commented; author selects)
  - TypeOK skeleton (commented per entity)
  - One action stub per Contract (preserving Allium contract name)
  - INVARIANT entry per top-level Allium invariant (preserving name + body)
  - Init / Next / Spec scaffolding

Naming convention (enforced by this generator):

  Allium `FooBar`     ->  TLA+ `FooBar`           (identical)
  Allium `FooBar`     ->  C++ `test-dflash-foo-bar.cpp`  (kebab-case)
  Allium `FooBar`     ->  DFLASH_REQUIRE(cond, "FooBar")

When a TLA+ invariant differs in name from the Allium invariant, that
divergence MUST be recorded in `scripts/allium-tla-binding.json` —
otherwise the spec drift goes silent.

Usage:
  allium parse spec.allium > /tmp/ast.json
  scripts/allium-to-tla.py /tmp/ast.json --module DFlashCycleGenerated \
      > PHASE_DFLASH-TLA/DFlashCycleGenerated.tla

Or pipe:
  allium parse spec.allium | scripts/allium-to-tla.py --module FooBar
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# ---- AST helpers ------------------------------------------------------

def walk_module_declarations(ast: dict) -> tuple[list, list, list, list, list]:
    """Return (entities, enums, contracts, invariants, open_questions)."""
    decls = ast["module"]["declarations"]
    entities, enums, contracts, invariants, opens = [], [], [], [], []
    for d in decls:
        if "Block" in d:
            blk = d["Block"]
            kind = blk.get("kind")
            name = blk.get("name", {}).get("name")
            items = blk.get("items", [])
            if kind == "Entity":
                entities.append({"name": name, "items": items})
            elif kind == "Enum":
                values = []
                for it in items:
                    ik = it.get("kind", {})
                    if isinstance(ik, dict):
                        if "EnumVariant" in ik:
                            v = ik["EnumVariant"].get("name", {}).get("name")
                            if v: values.append(v)
                        elif "EnumValue" in ik:  # legacy AST shape
                            v = ik["EnumValue"].get("name", {}).get("name")
                            if v: values.append(v)
                enums.append({"name": name, "values": values})
            elif kind == "Contract":
                signature = None
                inv_names = []
                for it in items:
                    ik = it.get("kind", {})
                    if isinstance(ik, dict):
                        if "Assignment" in ik:
                            a = ik["Assignment"]
                            if a.get("name", {}).get("name") == "invoke":
                                signature = "<see spec>"
                        elif "Annotation" in ik:
                            ann = ik["Annotation"]
                            if ann.get("name", {}).get("name") == "invariant":
                                pass
                contracts.append({"name": name, "items": items})
        elif "Invariant" in d:
            inv = d["Invariant"]
            invariants.append({
                "name": inv["name"]["name"],
                "body_span": inv.get("body", {}).get("Block", {}).get("span"),
            })
        elif "OpenQuestion" in d:
            opens.append(d["OpenQuestion"])
    return entities, enums, contracts, invariants, opens


def entity_field_names(entity: dict) -> list[str]:
    names: list[str] = []
    for it in entity["items"]:
        k = it.get("kind", {})
        if isinstance(k, dict) and "Assignment" in k:
            n = k["Assignment"].get("name", {}).get("name")
            if n: names.append(n)
    return names


def contract_inv_names(contract: dict) -> list[dict]:
    """In-contract @invariant annotations attached to this contract.

    AST shape per direct inspection:
      item.kind == "Annotation"  (string form)
      item.Annotation == {
          "kind": "Invariant",
          "name": { "name": "FooBar" },
          "body": [<prose lines>]
      }

    Returns list of {name, body_lines} dicts so the emitter can preserve
    the prose body as a TLA+ comment.
    """
    out: list[dict] = []
    for it in contract["items"]:
        k = it.get("kind")
        # AST shape: item.kind is dict-with-Annotation-key; the Annotation
        # value itself has {kind: "Invariant", name: ..., body: [...] }.
        if isinstance(k, dict) and "Annotation" in k:
            ann = k["Annotation"]
            if isinstance(ann, dict) and ann.get("kind") == "Invariant":
                nm = ann.get("name", {}).get("name")
                body = ann.get("body", [])
                if not isinstance(body, list):
                    body = []
                if nm:
                    out.append({"name": nm, "body_lines": body})
    return out


# ---- TLA+ emission ----------------------------------------------------

def emit(module_name: str, entities, enums, contracts, invariants, opens) -> str:
    lines: list[str] = []
    a = lines.append
    bar = "-" * 78
    a(bar)
    a(f"MODULE {module_name}".center(78))
    a(bar)
    a("(*****************************************************************************)")
    a("(* AUTO-GENERATED from Allium spec. Hand-edited bodies must preserve every    *)")
    a("(* INVARIANT name verbatim — they bind to Allium invariant names by the       *)")
    a("(* naming convention in scripts/allium-to-tla.py docstring.                   *)")
    a("(*                                                                            *)")
    a(f"(* Allium spec inputs: {len(entities)} entities, {len(enums)} enums, {len(contracts)} contracts, {len(invariants)} invariants, {len(opens)} OQs")
    a("(*****************************************************************************)")
    a("EXTENDS Integers, Sequences, FiniteSets, TLC")
    a("")

    # ---- Enums as CONSTANTS ---------------------------------------------
    a("(* ===== Enums (set-valued constants from Allium) ===== *)")
    for e in enums:
        a(f"\\* {e['name']}: {e['values']}")
        a(f"{e['name']}Values == {{ {', '.join(f'\"{v}\"' for v in e['values'])} }}")
    a("")

    # ---- Caller-fillable bound constants -------------------------------
    a("(* ===== Bound CONSTANTS (fill in .cfg) ===== *)")
    a("CONSTANTS")
    a("    MaxStep,             \\* TLC bound: cycle counter")
    a("    BlockSize            \\* From Allium DraftBlock.block_size (config: 16 for Qwen3.6-27B-DFlash)")
    a("")

    # ---- Variables suggestion (commented; author selects) --------------
    a("(* ===== State variables — author SELECTS from entity field suggestions ===== *)")
    a("(* Below: every entity field is listed as a candidate state variable.        *)")
    a("(* Uncomment + add to VARIABLES + Init those that are runtime state.          *)")
    a("(* Some entities are TYPES (e.g. Tensor, KVPair); they do not become state.   *)")
    for ent in entities:
        fields = entity_field_names(ent)
        if not fields: continue
        a(f"(* {ent['name']}:")
        for f in fields:
            a(f"     {f} (suggested var: {ent['name'].lower()}_{f})")
        a(" *)")
    a("")
    a("VARIABLES")
    a("    \\* AUTHOR: declare runtime state variables here. Example:")
    a("    step,                       \\* Nat — cycle counter")
    a("    pc                          \\* {\"draft\",\"verify\",\"accept\",\"advance\"}")
    a("")
    a("vars == << step, pc >>          \\* AUTHOR: extend as variables added")
    a("")

    # ---- TypeOK skeleton ------------------------------------------------
    a("(* ===== TypeOK ===== *)")
    a("TypeOK ==")
    a("    /\\ step \\in 0..MaxStep")
    a("    /\\ pc \\in { \"draft\", \"verify\", \"accept\", \"advance\" }")
    a("    \\* AUTHOR: add per-variable type clauses")
    a("")

    # ---- Init -----------------------------------------------------------
    a("(* ===== Init ===== *)")
    a("Init ==")
    a("    /\\ step = 0")
    a("    /\\ pc   = \"draft\"")
    a("    \\* AUTHOR: initialise remaining variables")
    a("")

    # ---- Actions from contracts ----------------------------------------
    a("(* ===== Actions (one per Allium contract) ===== *)")
    for c in contracts:
        a(f"(* Contract: {c['name']} — see Allium spec for invoke signature and invariants *)")
        ann_invs = contract_inv_names(c)
        if ann_invs:
            a(f"\\* Allium in-contract @invariants attached to {c['name']}:")
            for inv in ann_invs:
                a(f"\\*   - {inv['name']}")
        a(f"{c['name']} ==")
        a("    /\\ FALSE   \\* AUTHOR: fill pre/post conditions, primed variables")
        a("")

    a("Next ==")
    for c in contracts:
        a(f"    \\/ {c['name']}")
    a("")
    a("Spec == Init /\\ [][Next]_vars")
    a("")

    # ---- Invariants -----------------------------------------------------
    a("(* ===== Top-level Allium invariants ===== *)")
    a("(* Static deployment facts (shape, vocab, layer-count, no-tree, etc).         *)")
    a("(* Most are CONSTRUCTION-time properties, not state-machine. TLC sees TRUE.   *)")
    a("(* AUTHOR: replace TRUE only if the invariant has a state-machine form.       *)")
    for inv in invariants:
        a(f"\\* Allium top-level invariant: {inv['name']}")
        a(f"{inv['name']} ==")
        a("    TRUE   \\* AUTHOR: encode safety condition or leave TRUE for static")
        a("")

    a("(* ===== In-contract @invariants — the cycle-structure properties ===== *)")
    a("(* These ARE state-machine and need encoding. One TLA+ INVARIANT per name.    *)")
    a("(* Each preserves the Allium prose body as a comment for traceability.        *)")
    # Collect all in-contract invariants across all contracts; preserve order
    seen: set[str] = set()
    for c in contracts:
        ann_invs = contract_inv_names(c)
        for inv in ann_invs:
            if inv["name"] in seen:
                continue
            seen.add(inv["name"])
            a(f"\\* Allium @invariant ({c['name']}): {inv['name']}")
            for line in inv["body_lines"][:5]:  # cap prose at 5 lines
                a(f"\\*   {line}")
            if len(inv["body_lines"]) > 5:
                a(f"\\*   ... ({len(inv['body_lines'])} total body lines in Allium spec)")
            a(f"{inv['name']} ==")
            a("    TRUE   \\* AUTHOR: encode state-machine form of this invariant")
            a("")

    a("=" * 78)
    a("")
    return "\n".join(lines)


def emit_cfg(module_name: str, invariants, contracts) -> str:
    lines = []
    a = lines.append
    a(f"\\* {module_name}.cfg — auto-generated by allium-to-tla.py")
    a("\\* AUTHOR fills CONSTANTS values + reviews INVARIANTS list.")
    a("")
    a("SPECIFICATION Spec")
    a("")
    a("CONSTANTS")
    a("    MaxStep   = 4      \\* AUTHOR: tune per modeling brief §Bounds")
    a("    BlockSize = 3")
    a("")
    a("INVARIANTS")
    a("    TypeOK")
    a("    \\* Top-level Allium invariants (static facts):")
    for inv in invariants:
        a(f"    {inv['name']}")
    # Collect in-contract invariants for the cfg
    seen: set[str] = set()
    cycle_invs: list[str] = []
    for c in contracts:
        for inv in contract_inv_names(c):
            if inv["name"] not in seen:
                seen.add(inv["name"])
                cycle_invs.append(inv["name"])
    if cycle_invs:
        a("    \\* In-contract @invariants (cycle structure):")
        for n in cycle_invs:
            a(f"    {n}")
    return "\n".join(lines) + "\n"


# ---- CLI --------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    p.add_argument("ast_json", nargs="?", help="Path to allium parse JSON; default stdin.")
    p.add_argument("--module", required=True, help="TLA+ module name.")
    p.add_argument("--cfg", help="Optional: write companion .cfg to this path.")
    args = p.parse_args()

    if args.ast_json:
        ast = json.loads(Path(args.ast_json).read_text())
    else:
        ast = json.loads(sys.stdin.read())

    entities, enums, contracts, invariants, opens = walk_module_declarations(ast)
    tla = emit(args.module, entities, enums, contracts, invariants, opens)
    sys.stdout.write(tla)

    if args.cfg:
        Path(args.cfg).write_text(emit_cfg(args.module, invariants, contracts))
        print(f"\\* wrote {args.cfg}", file=sys.stderr)

    print(
        f"# generated {args.module}: {len(entities)} entities, {len(enums)} enums, "
        f"{len(contracts)} actions, {len(invariants)} invariants, {len(opens)} OQs ref",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
