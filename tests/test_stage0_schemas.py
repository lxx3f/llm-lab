"""Positive and negative tests for the Stage 0 JSON schemas."""

from __future__ import annotations

import copy
import importlib.util
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
        self.assertTrue(errors, "反向样例意外通过 schema 校验")

    def test_all_committed_examples_are_valid(self) -> None:
        groups = {
            "tool_calling_sample.schema.json": EXAMPLES / "tool_calling",
            "model_output.schema.json": EXAMPLES / "model_outputs",
            "evaluation_result.schema.json": EXAMPLES / "evaluation_results",
            "reward_signal.schema.json": EXAMPLES / "reward_signals",
        }
        for schema_name, directory in groups.items():
            for path in sorted(directory.glob("*.json")):
                if path.name == "MANIFEST.json":
                    continue
                with self.subTest(path=path.relative_to(ROOT)):
                    self.assertEqual([], errors_for(load_json(path), schema_name))

        d2_paths = sorted((EXAMPLES / "d2_multi_turn").glob("sample-positive-*.json"))
        self.assertGreaterEqual(len(d2_paths), 2)
        for path in d2_paths:
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertEqual([], errors_for(
                    load_json(path), "d2_multi_turn_sample.schema.json"))

    def test_d2_multi_turn_positive_examples_are_valid(self) -> None:
        paths = sorted((EXAMPLES / "d2_multi_turn").glob("sample-positive-*.json"))
        self.assertGreaterEqual(len(paths), 2)
        for path in paths:
            with self.subTest(path=path.name):
                self.assertEqual([], errors_for(
                    load_json(path), "d2_multi_turn_sample.schema.json"))

    def test_d2_multi_turn_negative_examples_fail_semantic_contract(self) -> None:
        generator_path = ROOT / "scripts" / "generate_d2_dataset.py"
        spec = importlib.util.spec_from_file_location("d2_generator_for_stage0", generator_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        paths = sorted((EXAMPLES / "d2_multi_turn").glob("sample-negative-*.json"))
        self.assertGreaterEqual(len(paths), 2)
        for path in paths:
            with self.subTest(path=path.name):
                document = load_json(path)
                self.assertEqual([], errors_for(
                    document, "d2_multi_turn_sample.schema.json"),
                    "negative fixture should be structurally schema-valid")
                self.assertTrue(module.validate_semantics(document))

    def test_d2_schema_rejects_noncanonical_task_type(self) -> None:
        document = load_json(
            EXAMPLES / "d2_multi_turn/sample-positive-001-multi-tool-sequential.json")
        document["metadata"]["task_type"] = "multi_turn_tool_chain"
        self.assert_invalid(document, "d2_multi_turn_sample.schema.json")

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
        document["failures"] = [{
            "sample_id": "sample-invalid",
            "input": "input",
            "expected": "expected",
            "prediction": "prediction",
            "failure_type": "unknown_failure_type",
            "raw_output": "raw",
            "analysis": "用于验证非法失败类型会被拒绝。",
        }]
        self.assert_invalid(document, "evaluation_result.schema.json")

    def test_missing_source_is_rejected(self) -> None:
        document = copy.deepcopy(self.single_tool)
        del document["metadata"]["source"]
        self.assert_invalid(document, "tool_calling_sample.schema.json")

    def test_missing_license_is_rejected(self) -> None:
        document = copy.deepcopy(self.single_tool)
        del document["metadata"]["license"]
        self.assert_invalid(document, "tool_calling_sample.schema.json")

    def test_reward_signal_full_pass_is_valid(self) -> None:
        reward = load_json(EXAMPLES / "reward_signals/reward-sample-001.json")
        self.assertEqual([], errors_for(reward, "reward_signal.schema.json"))

    def test_reward_signal_parse_fail_is_valid(self) -> None:
        reward = load_json(
            EXAMPLES / "reward_signals/reward-sample-002-parse-fail.json")
        self.assertEqual([], errors_for(reward, "reward_signal.schema.json"))

    def test_reward_signal_missing_reward_type_is_rejected(self) -> None:
        document = load_json(
            EXAMPLES / "reward_signals/reward-sample-001.json")
        del document["reward_type"]
        self.assert_invalid(document, "reward_signal.schema.json")

    def test_reward_signal_unknown_reward_type_is_rejected(self) -> None:
        document = copy.deepcopy(load_json(
            EXAMPLES / "reward_signals/reward-sample-001.json"))
        document["reward_type"] = "unknown_reward_type"
        self.assert_invalid(document, "reward_signal.schema.json")

    def test_reward_signal_binary_out_of_range_is_rejected(self) -> None:
        document = copy.deepcopy(load_json(
            EXAMPLES / "reward_signals/reward-sample-001.json"))
        document["reward_binary"] = 1.5
        self.assert_invalid(document, "reward_signal.schema.json")

    def test_reward_signal_accepts_d2_task_types(self) -> None:
        """reward_offline can now tag signals with the six canonical D2
        multi-turn task types; the schema must accept them."""
        base = copy.deepcopy(load_json(
            EXAMPLES / "reward_signals/reward-sample-001.json"))
        for task_type in ("tool_not_available", "tool_error_response",
                          "insufficient_result_search", "req_change_city",
                          "multi_tool_sequential", "error_recovery"):
            with self.subTest(task_type=task_type):
                document = copy.deepcopy(base)
                document["task_type"] = task_type
                self.assertEqual([], errors_for(
                    document, "reward_signal.schema.json"))

    def test_reward_signal_rejects_unknown_task_type(self) -> None:
        document = copy.deepcopy(load_json(
            EXAMPLES / "reward_signals/reward-sample-001.json"))
        document["task_type"] = "multi_turn_tool_chain"
        self.assert_invalid(document, "reward_signal.schema.json")


if __name__ == "__main__":
    unittest.main()
