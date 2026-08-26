"""In-process mock tool executor for the evaluation pipeline.

The mock executor runs registered Python functions with **zero external
side effects**: no subprocess, no filesystem writes, no network. It is the
P1-01 building block; P1-02 (sandboxed subprocess execution) is separate.

Each tool call produces a ``tool_execution_result`` object validated against
``schemas/tool_execution_result.schema.json`` v1.0.

Outcome vocabulary (P1-01 subset of the eventual failure taxonomy):
- ``success``: the registered mock function returned without raising.
- ``mock_not_found``: no mock is registered under ``tool_name``.
- ``argument_invalid``: ``arguments`` fail the mock's declared JSON Schema.
- ``mock_exception``: the mock function raised (message captured as ``error``).
- ``execution_timeout``: reserved for P1-02; not emitted by this class.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft202012Validator, FormatChecker

_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "tool_execution_result.schema.json"


def _load_schema() -> dict[str, Any]:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_execution_result(result: dict[str, Any]) -> None:
    """Raise ValueError if ``result`` does not satisfy the execution schema."""
    errors = sorted(
        Draft202012Validator(_load_schema(), format_checker=FormatChecker()).iter_errors(result),
        key=lambda error: list(error.path),
    )
    if errors:
        location = ".".join(str(part) for part in errors[0].path) or "$"
        raise ValueError(f"invalid tool execution result at {location}: {errors[0].message}")


class MockExecutor:
    """Registers mock tools and executes tool-call dicts in-process."""

    def __init__(self) -> None:
        self._mocks: dict[str, tuple[Callable[..., Any], dict[str, Any], str]] = {}

    def register_mock(self, tool_name: str, fn: Callable[..., Any], arguments_schema: dict[str, Any]) -> None:
        """Register a pure-Python mock under ``tool_name`` with its argument schema."""
        if not tool_name:
            raise ValueError("tool_name must be non-empty")
        if not callable(fn):
            raise ValueError(f"fn must be callable for tool {tool_name!r}")
        if not isinstance(arguments_schema, dict):
            raise ValueError(f"arguments_schema must be a dict for tool {tool_name!r}")
        self._mocks[tool_name] = (fn, arguments_schema, datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))

    def registered_tools(self) -> list[str]:
        return sorted(self._mocks)

    def execute(self, call: dict[str, Any]) -> dict[str, Any]:
        """Execute a single tool call dict and return a validated result.

        ``call`` must contain at least ``tool_name``, ``call_id`` and
        ``arguments`` (mirroring ``expected_tool_calls`` in the
        tool_calling_sample schema).
        """
        tool_name = str(call.get("tool_name", ""))
        call_id = str(call.get("call_id", ""))
        arguments = call.get("arguments", {})
        started = time.monotonic()
        entry = self._mocks.get(tool_name)
        if entry is None:
            return {
                "schema_version": "1.0",
                "tool_name": tool_name,
                "call_id": call_id,
                "outcome": "mock_not_found",
                "arguments": arguments,
                "result": None,
                "error": f"no mock registered for tool {tool_name!r}",
                "execution_time_ms": round((time.monotonic() - started) * 1000.0, 3),
                "mock_metadata": {},
            }
        fn, arguments_schema, registered_at = entry
        errors = sorted(
            Draft202012Validator(arguments_schema).iter_errors(arguments),
            key=lambda error: list(error.path),
        )
        if errors:
            location = ".".join(str(part) for part in errors[0].path) or "$"
            return {
                "schema_version": "1.0",
                "tool_name": tool_name,
                "call_id": call_id,
                "outcome": "argument_invalid",
                "arguments": arguments,
                "result": None,
                "error": f"argument schema violation at {location}: {errors[0].message}",
                "execution_time_ms": round((time.monotonic() - started) * 1000.0, 3),
                "mock_metadata": {"mock_id": tool_name, "registered_at": registered_at},
            }
        try:
            output = fn(**arguments)
        except Exception as exc:  # noqa: BLE001 - mock errors are captured, not propagated
            return {
                "schema_version": "1.0",
                "tool_name": tool_name,
                "call_id": call_id,
                "outcome": "mock_exception",
                "arguments": arguments,
                "result": None,
                "error": f"{type(exc).__name__}: {exc}",
                "execution_time_ms": round((time.monotonic() - started) * 1000.0, 3),
                "mock_metadata": {"mock_id": tool_name, "registered_at": registered_at},
            }
        result = {
            "schema_version": "1.0",
            "tool_name": tool_name,
            "call_id": call_id,
            "outcome": "success",
            "arguments": arguments,
            "result": output,
            "error": None,
            "execution_time_ms": round((time.monotonic() - started) * 1000.0, 3),
            "mock_metadata": {"mock_id": tool_name, "registered_at": registered_at},
        }
        validate_execution_result(result)
        return result

    def execute_sequence(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Execute calls in dependency order.

        ``depends_on`` is a list of earlier ``call_id`` values (as in the
        tool_calling_sample schema). A call whose dependency failed is
        reported as ``mock_not_found`` with a note (simple topological
        order, no parallelism).
        """
        by_id = {str(call.get("call_id", "")): call for call in calls}
        results: list[dict[str, Any]] = []
        done: set[str] = set()
        for call in calls:
            deps = call.get("depends_on") or []
            failed_dep = next((dep for dep in deps if dep in by_id and dep not in done), None)
            if failed_dep is not None:
                results.append({
                    "schema_version": "1.0",
                    "tool_name": str(call.get("tool_name", "")),
                    "call_id": str(call.get("call_id", "")),
                    "outcome": "mock_not_found",
                    "arguments": call.get("arguments", {}),
                    "result": None,
                    "error": f"dependency {failed_dep!r} did not complete; skipped",
                    "execution_time_ms": 0.0,
                    "mock_metadata": {},
                })
                continue
            result = self.execute(call)
            results.append(result)
            if result["outcome"] == "success":
                done.add(str(call.get("call_id", "")))
        return results
