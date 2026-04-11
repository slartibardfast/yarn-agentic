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


def post_chat_completion(server_url, messages, tools, seed, temperature, n_predict):
    """POST /v1/chat/completions. Returns the decoded JSON response."""
    url = server_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": temperature,
        "seed": seed,
        "max_tokens": n_predict,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def parse_tool_call(response):
    """Extract (name, args_dict) or None from a chat completion response."""
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

    results = []
    for i, case in enumerate(cases, 1):
        schema = schemas[case["schema_name"]]
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can call tools when the user's request requires one. If no tool fits, answer in plain text instead of calling a tool.",
            },
            {"role": "user", "content": case["query"]},
        ]
        tools = [build_tool(schema)]
        t0 = time.time()
        try:
            response = post_chat_completion(
                args.server_url,
                messages,
                tools,
                seed=args.seed,
                temperature=args.temperature,
                n_predict=args.n_predict,
            )
            parsed = parse_tool_call(response)
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
    args = parser.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
