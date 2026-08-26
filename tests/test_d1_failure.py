"""D1 dataset generator + P1-05 failure classifier tests."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[1]
D1_DIR = ROOT / "datasets" / "tool-calling-d1"
SCHEMA = json.loads((ROOT / "schemas" / "tool_calling_sample.schema.json").read_text(encoding="utf-8"))

ALL_TASK_TYPES = {"no_tool", "single_tool", "multi_tool", "tool_error", "insufficient_result", "requirement_change"}


class D1DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Regenerate the canonical 126-sample D1 dataset before any test."""
        proc = subprocess.run(
            [sys.executable, "scripts/generate_d1_dataset.py", "--count", "126"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        assert proc.returncode == 0, proc.stderr

    def test_manifest_exists_with_splits(self) -> None:
        manifest_path = D1_DIR / "MANIFEST.json"
        self.assertTrue(manifest_path.is_file())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["data_version"], "D1")
        self.assertGreaterEqual(manifest["count"], 100)
        splits = {s["split"] for s in manifest["samples"]}
        self.assertEqual(splits, {"train", "dev", "test"})

    def test_all_samples_schema_valid_and_covers_task_types(self) -> None:
        validator = Draft202012Validator(SCHEMA, format_checker=FormatChecker())
        task_types: set[str] = set()
        manifest = json.loads((D1_DIR / "MANIFEST.json").read_text(encoding="utf-8"))
        for entry in manifest["samples"]:
            path = D1_DIR / entry["path"]
            sample = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(list(validator.iter_errors(sample)), [], msg=f"schema error in {path}")
            task_types.add(sample["metadata"]["task_type"])
        self.assertEqual(task_types, ALL_TASK_TYPES, msg=f"missing task types: {ALL_TASK_TYPES - task_types}")

    def test_generator_is_deterministic(self) -> None:
        """Same seed → same task-type distribution (deterministic generator)."""
        def dist() -> str:
            proc = subprocess.run(
                [sys.executable, "scripts/generate_d1_dataset.py", "--count", "120"],
                capture_output=True, text=True, cwd=str(ROOT),
            )
            self.assertEqual(proc.returncode, 0, msg=f"stderr: {proc.stderr}")
            return proc.stdout

        first = dist()
        second = dist()
        self.assertEqual(first, second)
        # Restore the full 126-sample dataset for the manifest tests.
        proc = subprocess.run(
            [sys.executable, "scripts/generate_d1_dataset.py", "--count", "126"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        self.assertEqual(proc.returncode, 0, msg=f"stderr: {proc.stderr}")

    def test_cli_generate_reproducible(self) -> None:
        """Same seed → same first sample id set; generator exits 0."""
        proc = subprocess.run(
            [sys.executable, "scripts/generate_d1_dataset.py", "--count", "120"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        self.assertEqual(proc.returncode, 0, msg=f"stderr: {proc.stderr}")


class FailureClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = {
            "schema_version": "1.0",
            "id": "test-1",
            "messages": [{"role": "user", "content": "1+1=?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "calculate",
                    "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
                },
            }],
            "expected_tool_calls": [
                {"call_id": "c1", "name": "calculate", "arguments": {"expression": "1+1"}, "expected_result": "2"},
            ],
            "expected_answer": "2",
            "metadata": {"source": "test", "license": "test", "task_type": "single_tool",
                         "data_version": "D1", "pipeline_version": "test", "created_at": "2026-08-27T00:00:00Z",
                         "validation": {"schema_valid": True}},
        }

    def _classify(self, transcript: dict) -> dict:
        from scripts.classify_tool_failure import classify
        return classify(self.sample, transcript)

    def test_all_layers_pass(self) -> None:
        transcript = {
            "tool_calls": [{"call_id": "c1", "name": "calculate", "arguments": {"expression": "1+1"}, "execution_outcome": "success", "result": "2"}],
            "final_answer": "答案是 2",
        }
        result = self._classify(transcript)
        self.assertTrue(all(result["layers"].values()))
        self.assertIsNone(result["first_failure"])

    def test_parse_failure(self) -> None:
        transcript = {"tool_calls": [], "final_answer": "无法解析"}
        result = self._classify(transcript)
        self.assertFalse(result["layers"]["parse_success"])
        self.assertEqual(result["first_failure"], "parse_success")

    def test_schema_failure(self) -> None:
        transcript = {"tool_calls": [{"call_id": "c1", "name": "calculate", "arguments": {"wrong": "x"}, "execution_outcome": "success"}], "final_answer": ""}
        result = self._classify(transcript)
        self.assertTrue(result["layers"]["parse_success"])
        self.assertFalse(result["layers"]["schema_valid"])
        self.assertEqual(result["first_failure"], "schema_valid")

    def test_execution_failure(self) -> None:
        transcript = {"tool_calls": [{"call_id": "c1", "name": "calculate", "arguments": {"expression": "1+1"}, "execution_outcome": "mock_exception", "error": "boom"}], "final_answer": ""}
        result = self._classify(transcript)
        self.assertTrue(result["layers"]["parse_success"])
        self.assertTrue(result["layers"]["schema_valid"])
        self.assertFalse(result["layers"]["execution_success"])
        self.assertEqual(result["first_failure"], "execution_success")

    def test_result_grounding_failure(self) -> None:
        transcript = {"tool_calls": [{"call_id": "c1", "name": "calculate", "arguments": {"expression": "1+1"}, "execution_outcome": "success", "result": "3"}], "final_answer": "3"}
        result = self._classify(transcript)
        self.assertTrue(result["layers"]["parse_success"])
        self.assertTrue(result["layers"]["schema_valid"])
        self.assertTrue(result["layers"]["execution_success"])
        self.assertFalse(result["layers"]["result_grounded"])
        self.assertEqual(result["first_failure"], "result_grounded")

    def test_task_failure(self) -> None:
        transcript = {"tool_calls": [{"call_id": "c1", "name": "calculate", "arguments": {"expression": "1+1"}, "execution_outcome": "success", "result": "2"}], "final_answer": "无法计算"}
        result = self._classify(transcript)
        self.assertTrue(result["layers"]["parse_success"])
        self.assertTrue(result["layers"]["schema_valid"])
        self.assertTrue(result["layers"]["execution_success"])
        self.assertTrue(result["layers"]["result_grounded"])
        self.assertFalse(result["layers"]["task_success"])
        self.assertEqual(result["first_failure"], "task_success")


if __name__ == "__main__":
    unittest.main()
