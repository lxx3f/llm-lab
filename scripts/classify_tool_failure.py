"""P1-05 failure-layer classification evaluator.

Evaluates a model's tool-calling transcript against a D0/D1 sample through
five failure layers:

    parse_success      model output parsed into structured tool calls
    schema_valid       each call matches the tool's declared schema
    execution_success  MockExecutor ran the calls (all dependencies OK)
    result_grounded    expected tool result equals mock result
    task_success       expected_answer is contained in the final answer

Each layer's pass/fail is recorded; the first failing layer identifies where
the model failed (per open-issue P1-05).

Usage:
    .venv/python.exe scripts/classify_tool_failure.py \\
        --sample datasets/tool-calling-d1/train/d1-0001.json \\
        --transcript artifacts/transcript.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jsonschema import Draft202012Validator  # noqa: E402


def classify(sample: dict[str, Any], transcript: dict[str, Any]) -> dict[str, Any]:
    """Classify a model transcript against the sample through 5 layers."""
    layers: dict[str, bool] = {}

    # Layer 1: parse_success — transcript contains a tool-calls list.
    calls = transcript.get("tool_calls", [])
    layers["parse_success"] = isinstance(calls, list) and len(calls) > 0

    # Layer 2: schema_valid — every call matches the tool's arguments schema.
    schema_valid = True
    if layers["parse_success"]:
        for call in calls:
            tool = next((t for t in sample["tools"] if t["function"]["name"] == call.get("name")), None)
            if tool is None:
                schema_valid = False
                break
            validator = Draft202012Validator(tool["function"]["parameters"])
            if list(validator.iter_errors(call.get("arguments", {}))):
                schema_valid = False
                break
    layers["schema_valid"] = schema_valid

    # Layer 3: execution_success — every call ran without mock error.
    layers["execution_success"] = all(
        call.get("execution_outcome") == "success" for call in calls
    ) if layers["parse_success"] else False

    # Layer 4: result_grounded — actual mock result == expected result.
    expected = sample.get("expected_tool_calls", [])
    grounded = True
    if layers["execution_success"]:
        for idx, call in enumerate(calls):
            exp = next((e for e in expected if e.get("call_id") == call.get("call_id")), None)
            if exp is None or "expected_result" not in exp:
                continue
            actual = call.get("result")
            if actual != exp["expected_result"]:
                grounded = False
                break
    layers["result_grounded"] = grounded

    # Layer 5: task_success — expected answer appears in final assistant text.
    expected_answer = sample.get("expected_answer")
    final_text = transcript.get("final_answer", "") or ""
    layers["task_success"] = bool(expected_answer) and str(expected_answer) in final_text

    # First failing layer (or None if all pass).
    first_failure = next((name for name, ok in layers.items() if not ok), None)
    return {"layers": layers, "first_failure": first_failure}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    transcript = json.loads(args.transcript.read_text(encoding="utf-8"))
    result = classify(sample, transcript)
    if args.output:
        payload = {
            "schema_version": "1.0",
            "sample_id": sample.get("id"),
            "layers": result["layers"],
            "first_failure": result["first_failure"],
        }
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[classify_tool_failure] wrote {args.output}")
    print(f"[classify_tool_failure] layers={json.dumps(result['layers'])} first_failure={result['first_failure']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
