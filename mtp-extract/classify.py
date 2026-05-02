#!/usr/bin/env python3
"""Classify the 118 fork commits between 847e1919..a0d0e06e for MTP extraction.

Tags each commit:
- mtp_core    : touches MTP graph/runtime/loader/quantize files
- mtp_test    : touches tests/mtp-*
- mtp_doc     : doc-only and references MTP
- out_scope   : touches only TURBO_KV_4B / Vulkan / Mesa / batch-invariance / FWHT etc.
- mixed       : touches both MTP-relevant AND out-of-scope files
- sneaky_dep  : touches none of the MTP files but changes shared infra MTP relies on
- merge       : merge commit (no files in non-merge inspection)

Output: writes mtp-extract/inventory.md (markdown table).
"""
import subprocess, re, sys, os
from pathlib import Path

REPO = Path("/home/llm/yarn-agentic/ik_llama.cpp")

# File-name predicates
def is_mtp_core_file(f: str) -> bool:
    if "tests/mtp" in f or "tests/mtp-" in f:
        return False  # test, not core
    keys = (
        "src/llama-build-context.cpp",
        "src/llama-context.cpp",
        "src/llama-context.h",
        "src/llama-delta-net.cpp",
        "src/llama-delta-net.h",
        "src/llama-load-tensors.cpp",
        "src/llama-model.cpp",
        "src/llama-model.h",
        "src/llama-quantize.cpp",
        "src/llama-hparams.cpp",
        "src/llama-hparams.h",
        "src/llama.cpp",
        "examples/server/server-context.cpp",
        "examples/server/server-context.h",
        "examples/server/server.cpp",
        "common/common.cpp",
        "common/common.h",
        "common/sampling/common.cpp",
        "common/sampling.cpp",
        "common/sampling.h",
        "examples/main/main.cpp",
        "examples/llama-bench/",
        "include/llama.h",
        "ggml/src/ggml-cuda.cu",
    )
    return any(f == k or f.startswith(k.rstrip('/') + '/') for k in keys) or f in keys

def is_mtp_test_file(f: str) -> bool:
    return "tests/mtp" in f or "tests/mtp-" in f or "/mtp-matrix/" in f or "/mtp-rollout/" in f

def is_mtp_doc_signal(subject: str, files: list[str]) -> bool:
    s = subject.lower()
    if any(s.startswith(p) for p in ("docs:", "doc:", "readme")):
        return True
    if any("README.md" in f or "/docs/" in f for f in files):
        return True
    return False

OUT_OF_SCOPE_KEYWORDS = (
    "turbo_kv", "turbo-kv", "turbo_kv_4b",
    "vulkan", "mesa", "fwht", "aco", "radv",
    "mesa-repro",
    "batch-invariance", "batch invariance",
    "ipu", "vega", "rdna", "amdgpu",
)

def subject_is_out_scope(subject: str) -> bool:
    s = subject.lower()
    return any(k in s for k in OUT_OF_SCOPE_KEYWORDS)

def file_is_out_of_scope(f: str) -> bool:
    f_lower = f.lower()
    if "vulkan" in f_lower:
        return True
    if "tools/mesa" in f_lower:
        return True
    if "ggml/src/ggml-vulkan" in f_lower:
        return True
    if "/turbo_kv" in f_lower or "/turbo-kv" in f_lower:
        return True
    if "ggml-aco" in f_lower or "ggml-radv" in f_lower:
        return True
    return False

def file_is_mtp_signal(subject: str) -> bool:
    s = subject.lower()
    return any(k in s for k in ("mtp", "qwen35", "nextn", "delta-net", "deltanet", "speculative"))


def main():
    out_path = REPO.parent / "mtp-extract" / "inventory.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Get all commits in topo order, oldest first
    cmd = ["git", "-C", str(REPO), "log", "--reverse", "--pretty=format:%H|%an|%s", "847e1919..a0d0e06e"]
    out = subprocess.check_output(cmd).decode()
    commits = [line.split("|", 2) for line in out.strip().split("\n") if line]

    rows = []
    for sha, author, subject in commits:
        # Get files touched
        try:
            files_out = subprocess.check_output(
                ["git", "-C", str(REPO), "show", "--no-renames", "--pretty=", "--name-only", sha]
            ).decode()
        except subprocess.CalledProcessError:
            files_out = ""
        files = [f.strip() for f in files_out.strip().split("\n") if f.strip()]

        # Detect merge by checking parent count
        try:
            pcount = subprocess.check_output(
                ["git", "-C", str(REPO), "show", "-s", "--pretty=%P", sha]
            ).decode().strip().split()
        except subprocess.CalledProcessError:
            pcount = []
        is_merge = len(pcount) > 1

        if is_merge:
            cls = "merge"
        else:
            mtp_core_hit = any(is_mtp_core_file(f) for f in files)
            mtp_test_hit = any(is_mtp_test_file(f) for f in files)
            out_scope_hit = any(file_is_out_of_scope(f) for f in files) or subject_is_out_scope(subject)
            mtp_signal = file_is_mtp_signal(subject) or mtp_core_hit or mtp_test_hit

            if mtp_core_hit and out_scope_hit:
                cls = "mixed"
            elif mtp_test_hit and not mtp_core_hit and not out_scope_hit:
                cls = "mtp_test"
            elif mtp_core_hit and not out_scope_hit:
                cls = "mtp_core"
            elif out_scope_hit and not mtp_core_hit and not mtp_test_hit:
                cls = "out_scope"
            elif mtp_signal and not files:
                cls = "mtp_doc"
            elif is_mtp_doc_signal(subject, files) and mtp_signal:
                cls = "mtp_doc"
            elif not mtp_signal and not out_scope_hit:
                # No signal either way — could be sneaky-dep or just out-of-scope
                # Default to out_scope unless it touches one of the shared files MTP relies on
                shared_infra = {
                    "src/llama-build-context.cpp",
                    "src/llama-context.h",
                    "src/llama-context.cpp",
                    "src/llama-model.cpp",
                    "src/llama-load-tensors.cpp",
                    "src/llama.cpp",
                    "src/llama-hparams.cpp",
                    "src/llama-quantize.cpp",
                    "ggml/src/ggml-cuda.cu",
                }
                if any(f in shared_infra for f in files):
                    cls = "sneaky_dep"
                else:
                    cls = "out_scope"
            else:
                cls = "mixed"

        rows.append({
            "sha": sha[:10],
            "subject": subject,
            "author": author,
            "files": files,
            "class": cls,
        })

    # Build markdown
    md = ["# MTP Extraction Inventory\n",
          f"Total commits: **{len(rows)}** between merge-base `847e1919` and fork tip `a0d0e06e`.\n",
          "\n## Classification counts\n"]
    counts = {}
    for r in rows:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    for k in sorted(counts):
        md.append(f"- **{k}**: {counts[k]}")
    md.append("")

    md.append("\n## Commit table (chronological, oldest first)\n")
    md.append("| Class | SHA | Subject | Author | Files (count) |")
    md.append("|---|---|---|---|---|")
    for r in rows:
        s = r["subject"].replace("|", "\\|")
        a = r["author"].replace("|", "\\|")
        md.append(f"| {r['class']} | `{r['sha']}` | {s[:90]} | {a[:25]} | {len(r['files'])} |")
    md.append("")

    # Detail listings per class
    for tag in ("mtp_core", "mtp_test", "mtp_doc", "mixed", "sneaky_dep"):
        md.append(f"\n## {tag}: full file list per commit\n")
        for r in rows:
            if r["class"] == tag:
                md.append(f"\n### `{r['sha']}` — {r['subject']}\n")
                if r["files"]:
                    for f in r["files"]:
                        md.append(f"- `{f}`")
                else:
                    md.append("(no files reported)")

    out_path.write_text("\n".join(md))
    print(f"Wrote {out_path}")
    print()
    print("Counts:")
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}")

if __name__ == "__main__":
    main()
