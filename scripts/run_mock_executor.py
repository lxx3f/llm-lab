"""Execute a mock tool-calling sample and write a validated execution record.

Loads a ``tool_calling_sample`` JSON (``--input``) plus a mock registry JSON
(``--mocks``), registers the mocks (importing the Python functions via
``importlib``), runs the ``expected_tool_calls`` through ``MockExecutor``,
and writes an array of ``tool_execution_result`` objects to ``--output``.

Usage:
    .venv/python.exe scripts/run_mock_executor.py \\
        --input examples/mock_execution/sample-mock-001.json \\
        --mocks examples/mock_execution/sample-mock-001.mocks.json \\
        --output artifacts/mock-execution-sample-001-result.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture_lab.execution import MockExecutor, validate_execution_result  # noqa: E402


def _resolve_function(path: str) -> Any:
    module_name, _, attr = path.partition(":")
    if not attr:
        raise ValueError(f"function_path must be 'module:attr', got {path!r}")
    module = importlib.import_module(module_name)
    fn = getattr(module, attr)
    if not callable(fn):
        raise ValueError(f"{path!r} does not resolve to a callable")
    return fn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mocks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sample = json.loads(args.input.read_text(encoding="utf-8"))
    registry = json.loads(args.mocks.read_text(encoding="utf-8"))
    executor = MockExecutor()
    for tool_name, spec in registry.items():
        executor.register_mock(
            tool_name,
            _resolve_function(spec["function_path"]),
            spec["arguments_schema"],
        )
    calls: list[dict[str, Any]] = []
    for call in sample.get("expected_tool_calls", []):
        calls.append({
            "tool_name": call["name"],
            "call_id": call["call_id"],
            "arguments": call["arguments"],
            "depends_on": call.get("depends_on", []),
        })
    results = executor.execute_sequence(calls)
    for result in results:
        validate_execution_result(result)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[run_mock_executor] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())