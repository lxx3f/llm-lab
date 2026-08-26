"""P1-01 mock tool executor tests."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from architecture_lab.execution import MockExecutor, validate_execution_result
from architecture_lab.execution.mock_executor import _load_schema

ADD_SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
    "required": ["a", "b"],
}


def _add(a: float, b: float) -> float:
    return a + b


def _boom(a: float) -> float:
    raise ValueError("mock exploded")


class MockExecutorTests(unittest.TestCase):
    def test_register_mock_and_execute_success(self) -> None:
        ex = MockExecutor()
        ex.register_mock("add", _add, ADD_SCHEMA)
        result = ex.execute({"tool_name": "add", "call_id": "call-1", "arguments": {"a": 2, "b": 3}})
        self.assertEqual(result["outcome"], "success")
        self.assertEqual(result["result"], 5)
        self.assertEqual(result["tool_name"], "add")
        self.assertEqual(result["call_id"], "call-1")
        self.assertIsNone(result["error"])
        validate_execution_result(result)

    def test_execute_unknown_tool_returns_mock_not_found(self) -> None:
        ex = MockExecutor()
        ex.register_mock("add", _add, ADD_SCHEMA)
        result = ex.execute({"tool_name": "nope", "call_id": "call-1", "arguments": {}})
        self.assertEqual(result["outcome"], "mock_not_found")
        self.assertIsNone(result["result"])
        self.assertIn("no mock registered", result["error"])
        validate_execution_result(result)

    def test_execute_invalid_arguments_returns_argument_invalid(self) -> None:
        ex = MockExecutor()
        ex.register_mock("add", _add, ADD_SCHEMA)
        result = ex.execute({"tool_name": "add", "call_id": "call-1", "arguments": {"a": "not-a-number", "b": 3}})
        self.assertEqual(result["outcome"], "argument_invalid")
        self.assertIsNone(result["result"])
        self.assertIn("argument schema violation", result["error"])
        validate_execution_result(result)

    def test_execute_mock_function_exception_returns_mock_exception(self) -> None:
        ex = MockExecutor()
        ex.register_mock("boom", _boom, {"type": "object", "properties": {"a": {"type": "number"}}, "required": ["a"]})
        result = ex.execute({"tool_name": "boom", "call_id": "call-1", "arguments": {"a": 1}})
        self.assertEqual(result["outcome"], "mock_exception")
        self.assertIsNone(result["result"])
        self.assertIn("mock exploded", result["error"])
        validate_execution_result(result)

    def test_execute_sequence_respects_depends_on(self) -> None:
        ex = MockExecutor()
        ex.register_mock("add", _add, ADD_SCHEMA)
        calls = [
            {"tool_name": "add", "call_id": "call-1", "arguments": {"a": 2, "b": 3}},
            {"tool_name": "add", "call_id": "call-2", "arguments": {"a": 5, "b": 4}, "depends_on": ["call-1"]},
        ]
        results = ex.execute_sequence(calls)
        self.assertEqual([r["outcome"] for r in results], ["success", "success"])
        self.assertEqual(results[1]["result"], 9)

    def test_execute_sequence_skips_when_dependency_fails(self) -> None:
        ex = MockExecutor()
        ex.register_mock("add", _add, ADD_SCHEMA)
        calls = [
            {"tool_name": "add", "call_id": "call-1", "arguments": {"a": "bad", "b": 3}},
            {"tool_name": "add", "call_id": "call-2", "arguments": {"a": 5, "b": 4}, "depends_on": ["call-1"]},
        ]
        results = ex.execute_sequence(calls)
        self.assertEqual(results[0]["outcome"], "argument_invalid")
        self.assertEqual(results[1]["outcome"], "mock_not_found")
        self.assertIn("did not complete", results[1]["error"])

    def test_schema_rejects_invalid_outcome(self) -> None:
        schema = _load_schema()
        bad = {
            "schema_version": "1.0",
            "tool_name": "add",
            "call_id": "call-1",
            "outcome": "made_up_outcome",
            "arguments": {},
            "result": None,
            "error": None,
            "execution_time_ms": 0.0,
            "mock_metadata": {},
        }
        errors = list(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(bad))
        self.assertTrue(errors)

    def test_cli_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "result.json"
            proc = subprocess.run(
                [sys.executable, "scripts/run_mock_executor.py",
                 "--input", "examples/mock_execution/sample-mock-001.json",
                 "--mocks", "examples/mock_execution/sample-mock-001.mocks.json",
                 "--output", str(out)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=f"stderr: {proc.stderr}")
            results = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["outcome"], "success")
            self.assertEqual(results[0]["result"], 5)
            self.assertEqual(results[1]["outcome"], "success")
            self.assertEqual(results[1]["result"], 20)


if __name__ == "__main__":
    unittest.main()