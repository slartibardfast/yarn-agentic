#!/usr/bin/env python3
"""
Tool-calling accuracy harness.

Loads tool schemas from schemas/, per-schema test cases from cases/,
and scores a server along four axes:

    function_selection     — positives: called the correct function name
    required_arg_presence  — positives: all schema-required args present
    arg_type_correctness   — positives: emitted args match schema types
    abstention             — negatives: did NOT call any function

Talks to a llama.cpp server on its OpenAI-compatible /v1/chat/completions
endpoint, passing the schema in the `tools` field and letting the server
handle model-specific tool-call templating. Outputs a JSON report per run
to results/<model-label>-<UTC>.json.

Usage:

    python run_eval.py --model-label qwen35-9b-f16 --server-url http://127.0.0.1:9099

No external dependencies beyond the stdlib.
"""
import argparse
import json
import pathlib
import re
import sys
import time
import urllib.request
import urllib.error


HARNESS_DIR = pathlib.Path(__file__).parent
SCHEMAS_DIR = HARNESS_DIR / "schemas"
CASES_DIR = HARNESS_DIR / "cases"
RESULTS_DIR = HARNESS_DIR / "results"


def load_schemas(schemas_dir):
    """Load every *.json under schemas_dir. Returns name -> schema dict."""
    schemas = {}
    for path in sorted(schemas_dir.glob("*.json")):
        with open(path) as f:
            schema = json.load(f)
        schemas[schema["name"]] = schema
    return schemas


def load_cases(cases_dir):
    """Load every *.jsonl under cases_dir.

    Files named `<schema>.jsonl` or `<schema>.<tag>.jsonl` both belong to the
    schema `<schema>`. The optional `<tag>` allows a single schema to have
    multiple case files (e.g. `get_weather.jsonl` + `get_weather.hard.jsonl`).
    A case may also override the schema association by setting a `schema`
    field in the JSON record.
    """
    cases = []
    for path in sorted(cases_dir.glob("*.jsonl")):
        # Filename schema: <schema>[.tag].jsonl — the schema is everything
        # before the first dot.
        schema_name = path.name.split(".", 1)[0]
        with open(path) as f:
            for line_num, raw in enumerate(f, 1):
                line = raw.strip()
                if not line:
                    continue
                case = json.loads(line)
                case.setdefault("schema_name", case.pop("schema", schema_name))
                case["source"] = f"{path.name}:{line_num}"
                cases.append(case)
    return cases


def build_tool(schema):
    """Wrap our schema format into an OpenAI-compatible tool dict."""
    return {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema.get("description", ""),
            "parameters": schema["parameters"],
        },
    }


def _http_post(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


# ---------- Prompt / parsing strategies ----------
#
# A strategy is a pair (build_request, parse_response). Different models need
# different tool-calling conventions:
#
# - `openai`: server handles the chat template (via `--jinja`) and tool
#   injection; response arrives with a `tool_calls` field. Works for any
#   model whose chat_template already knows about tools (Qwen3.5, Qwen3,
#   Mistral v0.3, llama-3.1-instruct-tools, etc.).
#
# - `hermes`: Nous Research format — the tools are embedded in the system
#   prompt inside <tools></tools> XML tags, and the model is told to emit
#   calls inside <tool_call>...</tool_call> tags. We parse the raw content
#   text for the first <tool_call> block. Used for Hermes-3 and anything
#   trained on Nous's function-calling data. See
#   https://github.com/NousResearch/Hermes-Function-Calling.


HERMES_SYSTEM_PROMPT = """You are a function-calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Do not make assumptions about what values to plug into functions. Here are the available tools:
<tools>
{tools_json}
</tools>

For each function call return a JSON object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>

If the user query does not require any tool, answer in plain text without emitting a <tool_call> block."""


def build_request_openai(server_url, schema, query, seed, temperature, n_predict):
    """OpenAI tools-field request. Returns (url, payload)."""
    url = server_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that can call tools when the user's request requires one. If no tool fits, answer in plain text instead of calling a tool.",
            },
            {"role": "user", "content": query},
        ],
        "tools": [build_tool(schema)],
        "tool_choice": "auto",
        "temperature": temperature,
        "seed": seed,
        "max_tokens": n_predict,
    }
    return url, payload


def parse_response_openai(response):
    """OpenAI-style: inspect choices[0].message.tool_calls."""
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        return None
    call = tool_calls[0]
    func = call.get("function", {})
    name = func.get("name")
    raw_args = func.get("arguments", "{}")
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    except json.JSONDecodeError:
        args = None
    return (name, args)


def build_request_hermes(server_url, schema, query, seed, temperature, n_predict):
    """Nous Hermes format — inject schema into system prompt, parse <tool_call>."""
    url = server_url.rstrip("/") + "/v1/chat/completions"
    tool_spec = {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema.get("description", ""),
            "parameters": schema["parameters"],
        },
    }
    system = HERMES_SYSTEM_PROMPT.format(tools_json=json.dumps(tool_spec))
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ],
        "temperature": temperature,
        "seed": seed,
        "max_tokens": n_predict,
    }
    return url, payload


# <tool_call> ... </tool_call> — greedy on the inner body but non-greedy
# on end tag, to tolerate trailing whitespace. Multiline.
_HERMES_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
)


def parse_response_hermes(response):
    """Hermes: look at content text for the first <tool_call> JSON block."""
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None
    content = message.get("content") or ""
    match = _HERMES_TOOL_CALL_RE.search(content)
    if not match:
        return None
    blob = match.group(1).strip()
    # Some models wrap in ```json ... ``` inside the tool_call block.
    blob = re.sub(r"^```(?:json)?\s*|\s*```$", "", blob, flags=re.MULTILINE)
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        return None
    name = parsed.get("name")
    args = parsed.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = None
    return (name, args)


STRATEGIES = {
    "openai": (build_request_openai, parse_response_openai),
    "hermes": (build_request_hermes, parse_response_hermes),
}


def _matches_json_type(value, expected):
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        # bool is a subclass of int in Python — exclude it
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "null":
        return value is None
    # Unknown schema type — don't flag as mismatch.
    return True


def score_case(case, schema, parsed):
    """Score a single case along the four axes. Returns a dict."""
    is_positive = case["kind"] == "positive"
    expected_name = case.get("expected_name")

    result = {
        "case_source": case["source"],
        "kind": case["kind"],
        "query": case["query"],
        "expected_name": expected_name,
        "parsed": (
            {
                "name": parsed[0] if parsed else None,
                "args": parsed[1] if parsed else None,
            }
            if parsed
            else None
        ),
    }

    if is_positive:
        func_selected = parsed is not None and parsed[0] == expected_name
        result["function_selection"] = func_selected

        required = (schema["parameters"].get("required") or [])
        if parsed and parsed[1] is not None:
            emitted_keys = set(parsed[1].keys())
            result["required_arg_presence"] = all(r in emitted_keys for r in required)
        else:
            result["required_arg_presence"] = False

        props = schema["parameters"].get("properties") or {}
        if parsed and parsed[1] is not None:
            ok = True
            for k, v in parsed[1].items():
                prop_schema = props.get(k)
                if prop_schema is None:
                    ok = False
                    break
                if not _matches_json_type(v, prop_schema.get("type")):
                    ok = False
                    break
            result["arg_type_correctness"] = ok
        else:
            result["arg_type_correctness"] = False

        result["abstention"] = None  # not scored on positives
    else:
        # Negative case — only abstention axis is scored.
        result["function_selection"] = None
        result["required_arg_presence"] = None
        result["arg_type_correctness"] = None
        result["abstention"] = parsed is None

    return result


def aggregate(results):
    """Count passes along each axis, excluding None entries."""
    def rate(axis):
        eligible = [r for r in results if r.get(axis) is not None]
        if not eligible:
            return None
        passes = sum(1 for r in eligible if r[axis])
        return {
            "passed": passes,
            "total": len(eligible),
            "rate": round(passes / len(eligible), 4),
        }

    return {
        "function_selection": rate("function_selection"),
        "required_arg_presence": rate("required_arg_presence"),
        "arg_type_correctness": rate("arg_type_correctness"),
        "abstention": rate("abstention"),
    }


def run(args):
    schemas = load_schemas(SCHEMAS_DIR)
    cases = load_cases(CASES_DIR)

    if args.schema:
        cases = [c for c in cases if c["schema_name"] in args.schema]
    if args.limit:
        cases = cases[: args.limit]

    if not cases:
        print("No cases selected.", file=sys.stderr)
        return 2

    if args.strategy not in STRATEGIES:
        print(
            f"Unknown strategy {args.strategy!r}. Known: {sorted(STRATEGIES)}",
            file=sys.stderr,
        )
        return 2
    build_request, parse_response = STRATEGIES[args.strategy]

    results = []
    for i, case in enumerate(cases, 1):
        schema = schemas[case["schema_name"]]
        t0 = time.time()
        try:
            url, payload = build_request(
                args.server_url,
                schema,
                case["query"],
                seed=args.seed,
                temperature=args.temperature,
                n_predict=args.n_predict,
            )
            response = _http_post(url, payload)
            parsed = parse_response(response)
            error = None
        except Exception as e:  # noqa: BLE001
            response = None
            parsed = None
            error = f"{type(e).__name__}: {e}"
        elapsed = round(time.time() - t0, 3)

        scored = score_case(case, schema, parsed)
        scored["elapsed_s"] = elapsed
        scored["error"] = error
        results.append(scored)

        # One-line status per case.
        if case["kind"] == "positive":
            status = "OK" if scored["function_selection"] else "FAIL-fn"
        else:
            status = "OK" if scored["abstention"] else "FAIL-abs"
        print(f"[{i}/{len(cases)}] {case['source']}: {status} ({elapsed:.1f}s)", flush=True)

    totals = aggregate(results)
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H%M%SZ", time.gmtime())
    out_path = RESULTS_DIR / f"{args.model_label}-{ts}.json"
    report = {
        "model_label": args.model_label,
        "strategy": args.strategy,
        "server_url": args.server_url,
        "seed": args.seed,
        "temperature": args.temperature,
        "timestamp_utc": ts,
        "n_cases": len(results),
        "cases": results,
        "totals": totals,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Totals ===")
    for axis, value in totals.items():
        if value is None:
            print(f"  {axis}: (no eligible cases)")
            continue
        print(
            f"  {axis}: {value['passed']}/{value['total']} "
            f"({value['rate']*100:.0f}%)"
        )
    print(f"\nReport: {out_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Tool-calling accuracy harness")
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:9099",
        help="llama.cpp server base URL",
    )
    parser.add_argument(
        "--model-label",
        required=True,
        help="short label for the model run (goes into the report filename)",
    )
    parser.add_argument(
        "--schema",
        action="append",
        help="filter to these schema names (repeatable)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n-predict", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="cap to first N cases")
    parser.add_argument(
        "--strategy",
        choices=sorted(STRATEGIES),
        default="openai",
        help="prompt/parse strategy — `openai` (server handles tools via chat template) "
        "or `hermes` (Nous <tool_call> XML format injected in the system prompt)",
    )
    args = parser.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
