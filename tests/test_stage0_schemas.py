"""Positive and negative tests for the Stage 0 JSON schemas."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"
EXAMPLES = ROOT / "examples"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def errors_for(document: Any, schema_name: str) -> list[str]:
    schema = load_json(SCHEMAS / schema_name)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    return [error.message for error in validator.iter_errors(document)]


class Stage0SchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.single_tool = load_json(EXAMPLES / "tool_calling/sample-001.json")
        self.evaluation_result = load_json(
            EXAMPLES / "evaluation_results/sample-001.json"
        )

    def assert_invalid(self, document: Any, schema_name: str) -> None:
        errors = errors_for(document, schema_name)
        self.assertTrue(errors, "负例意外通过 schema 校验")

    def test_all_committed_examples_are_valid(self) -> None:
        groups = {
            "tool_calling_sample.schema.json": EXAMPLES / "tool_calling",
            "model_output.schema.json": EXAMPLES / "model_outputs",
            "evaluation_result.schema.json": EXAMPLES / "evaluation_results",
        }
        for schema_name, directory in groups.items():
            for path in sorted(directory.glob("*.json")):
                if path.name == "MANIFEST.json":
                    continue  # D0 manifest is not a sample document
                with self.subTest(path=path.relative_to(ROOT)):
                    self.assertEqual([], errors_for(load_json(path), schema_name))

    def test_missing_tool_name_is_rejected(self) -> None:
        document = copy.deepcopy(self.single_tool)
        del document["tools"][0]["function"]["name"]
        self.assert_invalid(document, "tool_calling_sample.schema.json")

    def test_non_object_expected_arguments_are_rejected(self) -> None:
        document = copy.deepcopy(self.single_tool)
        document["expected_tool_calls"][0]["arguments"] = '{"city":"北京"}'
        self.assert_invalid(document, "tool_calling_sample.schema.json")

    def test_invalid_failure_type_is_rejected(self) -> None:
        document = copy.deepcopy(self.evaluation_result)
        document["failures"] = [
            {
                "sample_id": "sample-invalid",
                "input": "input",
                "expected": "expected",
                "prediction": "prediction",
                "failure_type": "unknown_failure_type",
                "raw_output": "raw",
                "analysis": "用于验证非法失败类型会被拒绝。"
            }
        ]
        self.assert_invalid(document, "evaluation_result.schema.json")

    def test_missing_source_is_rejected(self) -> None:
        document = copy.deepcopy(self.single_tool)
        del document["metadata"]["source"]
        self.assert_invalid(document, "tool_calling_sample.schema.json")

    def test_missing_license_is_rejected(self) -> None:
        document = copy.deepcopy(self.single_tool)
        del document["metadata"]["license"]
        self.assert_invalid(document, "tool_calling_sample.schema.json")


if __name__ == "__main__":
    unittest.main()
