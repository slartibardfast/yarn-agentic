# Phase 2: Tool-Calling Accuracy Results

Raw reports live in `harness/results/<model-label>-<UTC>.json`; this document
is the human-readable summary. Updated as each model run lands.

## Test set

92 hand-written cases across 8 tool schemas. Each schema has an "easy" file
(`<schema>.jsonl`) and a "hard" file (`<schema>.hard.jsonl`). Totals:

- 49 positives (40 easy + 9 hard)
- 43 negatives (17 easy + 26 hard)

Hard positives push the model on indirect references ("LAX" → Los Angeles,
"Big Apple" → New York), compositional reasoning (JSON body embedded in a
`write_file` call, relative date math from a fixed "today" anchor), and
dirty/underspecified input ("Weather."). Hard negatives include queries that
look tool-shaped but lack required information ("Draft an email to Bob.",
"Send an email.", "Schedule something for next week."), and queries about a
schema's domain that are not tool calls ("Explain what /etc/passwd is used
for.", "How do timers work on Android?").

## Scoring

Per case, four booleans:

- `function_selection` — positives only: called the correct function.
- `required_arg_presence` — positives only: all schema-required args present.
- `arg_type_correctness` — positives only: emitted args match schema types.
- `abstention` — negatives only: did NOT call any function.

Rates per axis are reported as `passes / eligible`. A case can pass one axis
and fail another (e.g. correct function call with a wrong arg type).

## Ranking

| Rank | Model | Fn selection | Required args | Arg types | Abstention |
|------|-------|-------------:|--------------:|----------:|-----------:|
| 1    | `Qwen3.5-9B-mtp-q4km` | 48/49 (98%) | 48/49 (98%) | 48/49 (98%) | 43/43 (100%) |
| —    | _(comparison models pending)_ | — | — | — | — |

Reports:

- `harness/results/qwen35-9b-q4km-hard-2026-04-11T215245Z.json`

## Per-model notes

### `Qwen3.5-9B-mtp-q4km` (Vega 64, Vulkan, F16 K/V, `--jinja` tool templating)

First baseline. Strong on positives, perfect on negatives.

**Positive failures:**

- `create_calendar_event.hard.jsonl:1` — asked to book "next Monday at 3 PM
  for 'Code review'" with an explicit `"Today is 2026-04-11"` anchor (a
  Saturday). Expected a call with `start` resolved to `2026-04-13T15:00:00`.
  The model abstained instead of attempting the relative-date computation,
  i.e. it was over-conservative.

The pattern is a mild abstention bias rather than hallucinated calls or
wrong-tool selection. Qwen3.5-9B is very reliable at recognising when it has
enough information, and at extracting cleanly when it does — including cases
that require light transformation ("ten-second" → `{seconds: 10}`, "5-minute
timer" → `{seconds: 300}`, "Fahrenheit" → `{units: "imperial"}`).

The `draft_n_accepted / draft_n` ratio is not captured per-case in the
report yet (Open Question 3 in `PHASE2.md`). Worth adding in a later
iteration if it turns out to matter for ranking.

## Observations

_Filled in after all model runs complete._

## Final recommendation

_Filled in after all model runs complete._
