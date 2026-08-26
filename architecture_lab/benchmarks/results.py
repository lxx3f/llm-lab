"""Result validation and writing for N2 benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


def validate_routing_result(result: dict[str, Any]) -> None:
    schema_path = Path(__file__).parents[2] / "schemas/n2_routing_stats.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(result), key=lambda error: list(error.path))
    if errors:
        location = ".".join(str(part) for part in errors[0].path) or "$"
        raise ValueError(f"invalid N2 routing result at {location}: {errors[0].message}")


def write_routing_result(result: dict[str, Any], path: str | Path) -> None:
    validate_routing_result(result)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_n2_result(result: dict[str, Any]) -> None:
    schema_path = Path(__file__).parents[2] / "schemas/n2_benchmark_result.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(result),
        key=lambda error: list(error.path),
    )
    if errors:
        location = ".".join(str(part) for part in errors[0].path) or "$"
        raise ValueError(f"invalid N2 result at {location}: {errors[0].message}")


def write_n2_result(result: dict[str, Any], path: str | Path) -> None:
    validate_n2_result(result)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
