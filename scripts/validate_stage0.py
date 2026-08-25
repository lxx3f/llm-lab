"""Validate Stage 0 JSON examples against the project schemas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker, RefResolver


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate(document_path: Path, schema_path: Path) -> list[str]:
    schema = load_json(schema_path)
    document = load_json(document_path)
    resolver = RefResolver(schema_path.as_uri(), schema)
    validator = Draft202012Validator(
        schema,
        resolver=resolver,
        format_checker=FormatChecker(),
    )
    return [error.message for error in sorted(validator.iter_errors(document), key=str)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Validate all committed Stage 0 example files.",
    )
    args = parser.parse_args()

    if not args.examples:
        parser.error("当前仅支持 --examples")

    checks = [
        (
            ROOT / "examples/tool_calling/sample-001.json",
            ROOT / "schemas/tool_calling_sample.schema.json",
        ),
        (
            ROOT / "examples/tool_calling/sample-002-no-tool.json",
            ROOT / "schemas/tool_calling_sample.schema.json",
        ),
        (
            ROOT / "examples/tool_calling/sample-003-multi-tool.json",
            ROOT / "schemas/tool_calling_sample.schema.json",
        ),
        (
            ROOT / "examples/model_outputs/sample-001.json",
            ROOT / "schemas/model_output.schema.json",
        ),
        (
            ROOT / "examples/evaluation_results/sample-001.json",
            ROOT / "schemas/evaluation_result.schema.json",
        ),
    ]

    failed = False
    for document_path, schema_path in checks:
        errors = validate(document_path, schema_path)
        if errors:
            failed = True
            print(f"FAIL {document_path.relative_to(ROOT)}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"PASS {document_path.relative_to(ROOT)}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
