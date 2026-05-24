---
name: No yarn-agentic nomenclature in code, scripts, tests, directories, or branch names
description: Phase/Track/Stage/A.X nomenclature is for planning docs only — never in source files, harness scripts, test files, directory names, file names, or branch names, in either the submodule codebases OR the agentic-harness layer of the parent repo
type: feedback
originSessionId: a5a02c13-3d1d-4998-badf-36c74b87c680
---

yarn-agentic is the outer "agentic host" that organizes work into phases, tracks, stages, and step IDs (A.0, A.T1, B.5, etc.). These are planning/documentation constructs only. They have no meaning inside submodule codebases, and they leak host concerns into anything that may eventually be upstreamed or read by external contributors. They are equally inappropriate in the parent repo's harness layer (scripts/, tests/, profiles/, anywhere code lives), even though `PHASE*.md` planning documents themselves use them.

**Why:** Mixing host-level organizational metadata into code/scripts/tests makes them less portable, less readable, and uglier to upstream. Phase/track references belong in `PHASE*.md` and `PLAN.md` — full stop. Anywhere else, they're noise. The 2026-05-05 cuda graph cache instrumentation work landed `probe35.{cuh,cu}`, `phase35::` namespace, `phase35_probe_state` struct, four `test-phase35-A-*.cpp` files, and `scripts/phase35/run-A-*.sh`. The user called it "just terrible." Don't repeat.

**Where phase/track/stage nomenclature is ALLOWED (the only places):**

- `/PHASE*.md`, `/PLAN.md` in the yarn-agentic root.
- `/MEMORY.md` and the entries it indexes.
- Memory files under `~/.claude/projects/-home-llm-yarn-agentic/memory/` (this file).
- The running TaskList (subjects/descriptions can reference phases for human navigation).
- Commit subjects in the **parent repo only** — `PHASE34: amend RCA`, `PHASE35: land plan` follow the existing yarn-agentic commit-message convention. Submodule commit subjects must NOT.

**Where it is FORBIDDEN:**

- Any source file in the submodule codebases (`ik_llama.cpp/`, `llama.cpp/`, `text-embeddings-inference/`, `mesa-fix/`).
- Any source file, header, or build script in the **parent repo's code/harness/tests layer**: `scripts/`, `tests/`, `profiles/`, `litellm-qwen/`, etc.
- Comments, docstrings, function/struct/namespace names, env-var names, log strings.
- File names and directory names. (Examples of violations: `probe35.cuh`, `cuda_graph_probe35.cu`, `test-phase35-A-monotonic.cpp`, `scripts/phase35/`, `tests/phase35/`.)
- Commit subjects in any submodule.
- Branch names in any repo (submodule or parent). Branches are technical artifacts. Use feature-descriptive names: `cuda-graph-probe`, `vulkan-turbo-kv-4b`, `split-k-cache` — never `phase33-concat-probe`, `vulkan-phase4`, `track3-wip`.
- Test ID labels in source/scripts: `A.T1`, `B.T3`, `T4`, `Step 5`, `iter 24`, `substep 6.3` — these inherit the problem and must not appear.

**How to apply:**

- Before committing, run `git diff --staged | grep -iE 'phase ?[0-9]|track [0-9]|stage [0-9]|substep|iter [0-9]|^\+.*\bA\.[0-9T]|^\+.*\bB\.[0-9T]|^\+.*\bT[0-9]\b|PHASE[0-9]'` against your changes. Any match in code/scripts/tests/dirs is a violation; strip and rewrite.
- Choose names that describe the feature: `cuda_graph_probe.cuh`, `test-cuda-graph-probe-hit-monotonic.cpp`, `scripts/cuda-graph-probe/run-overhead-canary.sh`. The reader should know what the file does without needing the planning doc.
- For tests, name by the assertion they make, not by the test ID in the planning doc: `test-cuda-graph-probe-hit-monotonic` (asserts hit counter monotonicity), not `test-phase35-A-T3` or `test-A-T3-monotonic`.
- For scripts, name by the action they perform, not by the milestone they belong to: `run-overhead-canary.sh`, `run-flush-trigger.sh` — not `run-A-overhead.sh`, `run-A-flush.sh`.
- For commits in submodules: descriptive subject lines that stand alone. "cuda graph cache: instrumentation surface + RED tests" — not "PHASE35 A.0: …".
- For commits in the parent repo: keep the `PHASE\d+:` prefix because the parent's history already uses it; that's the planning convention. But the BODY of the parent commit must not propagate phase nomenclature into descriptions of code being added — describe the code descriptively.
- When pointing readers at design context inside source/script files, describe inline in technical terms or point to a CLAUDE.md or repo-level docs file — never to `PHASE*.md` doc names.
- References to in-repo issues/PRs (`// See #12147`) are fine.
- Audit existing references and clean them up when encountered, even in committed code; create a follow-up commit that strips them out.

**The grep self-check is mandatory** before any commit that touches submodule code or the parent harness layer. The rule is enforced by you grepping yourself, not by anything external. Skipping it is the failure mode.
