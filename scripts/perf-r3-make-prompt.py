#!/usr/bin/env python3
"""Generate synthetic prompts of approximate token-count targets for
PHASE_PERF_R3 context-depth scaling tests.

Uses a simple text-repeat strategy. Approximate token count is computed
assuming ~3.8 bytes/token for Qwen 3.x BPE on English prose — empirically
verified on the production model in earlier work.

Usage:
    perf-r3-make-prompt.py --target-tokens 16384 > /tmp/prompt-16k.txt

The output is plain text suitable for the PROMPT env var of
test-production-np-determinism.sh.
"""
from __future__ import annotations

import argparse
import sys

# Base paragraph: English content with mixed punctuation, sentences, and
# multi-syllable words — matches the token-density profile of typical
# completion benches. Avoids special characters that might collide with
# Qwen template parsing or trigger thinking-mode prompt patterns.
BASE = (
    "The history of artificial intelligence began in earnest with the work of Alan "
    "Turing, who in 1950 published the influential paper Computing Machinery and "
    "Intelligence, introducing the imitation game now widely known as the Turing "
    "test. Following Turings pioneering ideas, the field saw rapid growth during "
    "the 1956 Dartmouth workshop organized by John McCarthy, Marvin Minsky, "
    "Nathaniel Rochester, and Claude Shannon. McCarthy coined the term artificial "
    "intelligence for the workshop. Through the 1960s and 1970s, researchers "
    "developed expert systems, theorem provers, and natural language interfaces, "
    "though hardware limitations of the era constrained the scale at which these "
    "systems could operate. Funding cycles produced two notable AI winters before "
    "deep learning, building on three decades of neural network research, "
    "transformed the field starting in the 2010s. "
)

# Empirical: ~3.8 bytes/token for Qwen 3.x BPE on this prose.
BYTES_PER_TOKEN = 3.8

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-tokens", type=int, required=True,
                   help="Approximate target token count (Qwen 3.x BPE)")
    p.add_argument("--tail-question", default=" Summarize the development of "
                   "AI in one paragraph.",
                   help="Final tail appended after the bulk text — gives the "
                        "model something specific to continue.")
    args = p.parse_args()

    target_bytes = int(args.target_tokens * BYTES_PER_TOKEN)
    n_repeats = max(1, target_bytes // len(BASE))

    bulk = (BASE * n_repeats).rstrip()
    out = bulk + args.tail_question
    sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
