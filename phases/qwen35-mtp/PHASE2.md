# Phase 2: Tool-Calling Accuracy Harness

## Goal

Rank candidate models by **tool-calling accuracy** on a small, hand-written eval set. The ranking drives downstream decisions about which model to deploy for agent work, and whether the MTP path is worth building more spec-decode infrastructure around.

Per §12 of Phase 1:

> The mission is tool-calling accuracy, not throughput. Do not build more spec-decode infrastructure until the accuracy ranking is done. That's the lesson the previous host paid for.

## Why now

Phase 1 confirmed that the MTP-enabled Qwen3.5-9B server runs correctly on Vega 64 Vulkan (byte-identical outputs vs non-spec, 77.78 % draft acceptance). What it did **not** measure is whether the model is actually any good at picking the right tool, filling in the right arguments, or abstaining when no tool is appropriate. Phase 2 answers that — and only that.

## Non-goals

- Throughput measurement (Phase 1 covered it; the revised §9 performance note is honest about what we saw).
- Production-quality evals. This is a hand-written exploratory set, not a formal benchmark.
- Multi-turn agent loops. Single-turn tool invocation only.
- Model training or fine-tuning.

## Pass criteria

1. A reproducible harness that takes a model (by server URL + label), runs a fixed test set, and emits a JSON score report.
2. An initial test set of 5–10 tool schemas, each with 5–10 positive queries + 2–3 negative queries. Hand-written, version-controlled, reviewed by a human for ambiguity before use.
3. Scoring along four axes, per case:
   - **Function selection accuracy** — positives: called the correct function name. Negatives: did not call any function.
   - **Required-arg presence** — all required args from the schema appear in the emitted call (positives only).
   - **Arg type correctness** — all emitted args match the declared JSON Schema types (no stringified numbers, etc.).
   - **Abstention correctness** — on negatives, the model did not invoke a tool.
4. Per-model totals for each axis, and an overall pass rate.
5. Ranked comparison against at least three alternative models on the same test set.
6. A short writeup as `RESULTS.md` inside this phase folder: ranking table, per-axis observations, and commentary on failure modes.

## Harness architecture

Target layout inside this phase folder:

```
phases/qwen35-mtp/
├── PHASE1.md
├── PHASE2.md                  (this document)
├── harness/
│   ├── run_eval.py            (main entry point)
│   ├── schemas/
│   │   ├── get_weather.json
│   │   ├── search_web.json
│   │   ├── read_file.json
│   │   └── ...
│   ├── cases/
│   │   ├── get_weather.jsonl
│   │   ├── search_web.jsonl
│   │   └── ...
│   ├── grammars/
│   │   └── tool_call.gbnf.tpl (template — per-schema grammars generated at runtime)
│   └── results/
│       └── <model-label>-<UTC-timestamp>.json
└── RESULTS.md                 (committed writeup)
```

Single Python entry point, roughly:

```python
# pseudocode
cases = load_cases(schemas_dir="schemas", cases_dir="cases")
for case in cases:
    grammar = build_grammar_for(case.schema)
    resp = post_completion(
        server_url,
        prompt=build_prompt(case.schema, case.query, model_chat_template),
        grammar=grammar,
        temperature=0,
        seed=42,
        n_predict=256,
    )
    parsed = parse_tool_call(resp)     # -> (name, args_dict) | None
    result = score(case, parsed)
    report.cases.append(result)

report.totals = aggregate(report.cases)
write_json(report, f"results/{label}-{utc_now()}.json")
```

No framework. No pytest. One script, one JSON report.

## Tool schemas — initial proposed set

Hand-written, OpenAI-style function specs. Start with three for Phase 2-a, expand to eight for Phase 2-d.

1. **`get_weather`** — `city` (required string), `units` (optional enum `{metric, imperial}`, default `metric`)
2. **`search_web`** — `query` (required string), `max_results` (optional integer, default 5)
3. **`read_file`** — `path` (required string)
4. **`write_file`** — `path` (required string), `contents` (required string)
5. **`list_directory`** — `path` (optional string, default `"."`)
6. **`send_email`** — `to` (required string), `subject` (required string), `body` (required string)
7. **`create_calendar_event`** — `title` (required string), `start` (required ISO 8601 string), `duration_minutes` (optional integer, default 60)
8. **`set_timer`** — `seconds` (required integer)

Negative-case patterns to include for every schema:

- Pure greeting / small talk (`"hi there"`) — no tool needed.
- Fact questions the model should answer without tools (`"what is 7 * 8?"`).
- Ambiguous near-miss requests that look tool-shaped but lack crucial information (`"send an email"` with no recipient, subject, or body — the model should either ask a clarifying question or abstain, not fabricate).

## GBNF grammar

Per-schema GBNF generated at runtime from the JSON Schema. The grammar constrains the model to emit exactly one of:

```json
{"name": "<function_name>", "arguments": { ... }}
```

or

```json
{"tool_call": null}
```

A single catch-all "any JSON with a name field" grammar is too loose — it defeats the correctness floor that the §11 Phase 1 gotcha calls for. Per-schema strictness means a model that emits `{"city": 30}` for `get_weather` is rejected at decode time, not in scoring.

Abstention token is an explicit `{"tool_call": null}` so the grammar has one uniform output shape. Alternative considered — let the model free-form on negatives and treat "no valid tool-call JSON" as abstention — is cleaner against models that don't know about the abstention token but makes scoring ambiguous; rejected in favour of the explicit token.

## Scoring

Per case, booleans along the four axes:

| Axis                 | Positive case                            | Negative case                      |
| -------------------- | ---------------------------------------- | ---------------------------------- |
| Function selection   | Called the correct function name         | Did NOT call any function          |
| Required-arg presence| All required args from schema are present| (not scored)                       |
| Arg-type correctness | All emitted arg values match schema types| (not scored)                       |
| Abstention           | (not scored)                             | Emitted `{"tool_call": null}`      |

Per-model totals: count passes per axis out of eligible cases. Rough quality floor: > 80 % on each axis. **The ranking, not the floor, is what drives decisions** — even a 60 % model can win the comparison if it's the best of a bad lot.

## Models to compare

Per §12 of Phase 1, at least three candidates alongside the current build:

1. **`Qwen3.5-9B-mtp-q4km`** — this phase's build. Running on Vega 64 via Vulkan. Base variant (see note below).
2. **`Qwen3-14B-Q6_K`** — larger, older architecture, no MTP. Control for "does size help?".
3. **`Hermes-3-8B-Q6_K`** (Nous) — tuned specifically for tool-use. Control for "does tool-tuning help?".
4. **`Functionary-Small-Q6_K`** (MeetKai) — specifically fine-tuned for function calling. Control for "does the sharpest tool-specific model help?".

**High-value addition:** if a `Qwen3.5-9B-Instruct` variant exists, add it as slot 1.5. The Phase 1 build's `general.name` is `Qwen3.5 9B HF` and `general.base_model.0.name` is `Qwen3.5 9B Base` — this is very likely the **base** variant, and base models are not usually meant for instruction-following. Comparing only against an instruct sibling isolates "is it the architecture/size or the instruct-tune that matters?".

Each comparison model needs a GGUF with the same tokenizer family if possible (to keep prompt-template differences from dominating the comparison). Q6_K for non-MTP candidates keeps the quant ceiling consistent; no custom tensor-type overrides needed.

## Implementation order

Four tasks, linear. Each is small enough to land in one sitting.

### 1. Scaffold + one tool, end-to-end

Write `run_eval.py`, `schemas/get_weather.json`, `cases/get_weather.jsonl` (5 positives + 2 negatives), `grammars/` builder. Hit the already-running Vega 64 server. Emit a JSON report. Human sanity-check the report.

**Verify:** script runs end-to-end against `localhost:9099`, emits a JSON file with one entry per case, each entry has the four scoring axes. No framework, no external deps beyond `requests` and the stdlib.

### 2. Full initial test set

Expand to all 8 schemas, 5–10 positives + 2–3 negatives each. Have the test set reviewed by a human before trusting scores (ambiguity in the queries will skew results more than model differences).

**Verify:** run the full set against the Vega 64 server, get a report with ~60–100 cases, no crashes, no obviously-ambiguous queries. The harness spends most of its time in server latency, not Python.

### 3. Comparison model runs

Download and quantize the three comparison models (Qwen3-14B, Hermes 3 8B, Functionary Small). Run the harness against each. All three can share one llama-server invocation per model — start server, run harness, kill server, swap model, repeat. No need to modify the harness per model except the server URL / label string.

**Verify:** four `results/*.json` reports, one per model, each emitted against the identical test set.

### 4. Writeup

Commit `RESULTS.md` with:

- A ranking table: model × axis pass rate.
- Per-axis observations: which axis separates the models most? Where does each model fail?
- Commentary on failure modes: does the Qwen3.5-9B base variant stubbornly free-form instead of calling tools? Does Hermes 3 call the wrong function when the schema is ambiguous? Does Functionary refuse to abstain?
- A final recommendation: which model to use for agent work, or "none of these are good enough; go hunt for candidates."

## Open questions

1. **Prompt format.** Qwen3.5 ships with a `chat_template.jinja` from the HF download. Do we feed the schema via the OpenAI-style `tools` array (if llama-server supports it), or embed the schema in the system prompt? Different models may require different prompting strategies — this may need to be per-model, not uniform.
2. **Abstention via grammar vs. trust.** Proposed grammar includes an explicit `{"tool_call": null}` token. Alternative: let the model free-form on negatives, score "didn't emit a valid tool call" as correct abstention. Explicit token is cleaner but assumes the model understands the convention. Decide in 2-a after seeing actual Qwen3.5 outputs.
3. **Spec-decode acceptance during tool calls.** Tool-call outputs are structured JSON, which tokens should be more predictable than free text, which *should* raise MTP acceptance. Worth logging `draft_n_accepted / draft_n` per case as a bonus metric and reporting the ratio between tool-call and free-text cases at the end of the phase.
4. **Where does the harness code live.** Proposal: inside `phases/qwen35-mtp/harness/`, not at the repo root. It's phase-specific scaffolding, not a top-level tool. If it turns out to be generally useful across workstreams, promote to `tools/` later.
5. **Test-set licence.** Hand-written queries are ours, no licence issue, but make it explicit in a `harness/LICENSE` header so it's obvious what can be reused.
