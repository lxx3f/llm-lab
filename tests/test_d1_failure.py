"""D1 dataset generator + P1-05 failure classifier tests."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from scripts.classify_tool_failure import classify

ROOT = Path(__file__).resolve().parents[1]
D1_DIR = ROOT / "datasets" / "tool-calling-d1"
SCHEMA = json.loads((ROOT / "schemas" / "tool_calling_sample.schema.json").read_text(encoding="utf-8"))

ALL_TASK_TYPES = {"no_tool", "single_tool", "multi_tool", "tool_error", "insufficient_result", "requirement_change"}
CANONICAL = {"train": 100, "dev": 13, "test": 13}


class D1DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Regenerate the canonical 126-sample D1 dataset before any test."""
        proc = subprocess.run(
            [sys.executable, "scripts/generate_d1_dataset.py"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        assert proc.returncode == 0, proc.stderr

    def test_manifest_matches_ondisk_layout(self) -> None:
        """On-disk files must exactly match the manifest (no stale files)."""
        manifest = json.loads((D1_DIR / "MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["count"], 126)
        manifest_ids = {Path(e["path"]).name for e in manifest["samples"]}
        on_disk = {p.name for p in D1_DIR.glob("*/d1-*.json")}
        self.assertEqual(on_disk, manifest_ids, msg="stale/extra files on disk")
        # Split counts must be exactly 100/13/13.
        split_counts = {}
        for entry in manifest["samples"]:
            split_counts[entry["split"]] = split_counts.get(entry["split"], 0) + 1
        self.assertEqual(split_counts, CANONICAL)

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

    def test_manifest_hashes_match_files(self) -> None:
        import hashlib
        manifest = json.loads((D1_DIR / "MANIFEST.json").read_text(encoding="utf-8"))
        for entry in manifest["samples"]:
            path = D1_DIR / entry["path"]
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                entry["sha256"],
                msg=f"hash mismatch for {entry['path']}",
            )

    def test_generator_default_is_126(self) -> None:
        proc = subprocess.run(
            [sys.executable, "scripts/generate_d1_dataset.py"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        self.assertEqual(proc.returncode, 0, msg=f"stderr: {proc.stderr}")
        self.assertIn("126 samples", proc.stdout)

    def test_generator_is_deterministic(self) -> None:
        """Same seed → same manifest (byte-identical) across two runs."""
        first = (D1_DIR / "MANIFEST.json").read_bytes()
        subprocess.run(
            [sys.executable, "scripts/generate_d1_dataset.py"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        second = (D1_DIR / "MANIFEST.json").read_bytes()
        self.assertEqual(first, second)


class FailureClassifierTests(unittest.TestCase):
    """P1-05 five/six-layer classification across all D1 task types."""

    def _base_sample(self, task_type: str, expected_calls: list, expected_answer: str | None) -> dict:
        return {
            "schema_version": "1.0",
            "id": f"test-{task_type}",
            "messages": [{"role": "user", "content": "q"}],
            "tools": [
                {"type": "function", "function": {
                    "name": "calculate",
                    "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
                }},
                {"type": "function", "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
                }},
            ],
            "expected_tool_calls": expected_calls,
            "expected_answer": expected_answer,
            "metadata": {"source": "test", "license": "test", "task_type": task_type,
                         "data_version": "D1", "pipeline_version": "test", "created_at": "2026-08-27T00:00:00Z",
                         "validation": {"schema_valid": True}},
        }

    def test_no_tool_correct_output_passes_all(self) -> None:
        """no_tool: empty tool_calls + matching final answer = full pass."""
        sample = self._base_sample("no_tool", [], "直接回答，不调用工具")
        transcript = {"tool_calls": [], "final_answer": "直接回答，不调用工具"}
        result = classify(sample, transcript)
        self.assertTrue(result["layers"]["parse_success"])
        self.assertTrue(result["layers"]["schema_valid"])
        self.assertTrue(result["layers"]["call_plan_matches"])
        self.assertTrue(result["layers"]["execution_success"])
        self.assertEqual(result["layers"]["result_grounded"], None)  # no expected_result declared
        self.assertTrue(result["layers"]["task_success"])
        self.assertIsNone(result["first_failure"])

    def test_no_tool_calling_a_tool_fails_plan(self) -> None:
        sample = self._base_sample("no_tool", [], "直接回答")
        transcript = {"tool_calls": [{"call_id": "c1", "name": "calculate", "arguments": {"expression": "1+1"}}], "final_answer": "直接回答"}
        result = classify(sample, transcript)
        self.assertTrue(result["layers"]["parse_success"])
        self.assertFalse(result["layers"]["call_plan_matches"])
        self.assertEqual(result["first_failure"], "call_plan_matches")

    def test_single_tool_null_answer_task_layer_not_applicable(self) -> None:
        """single_tool with expected_answer null: task_success is None, not False."""
        sample = self._base_sample("single_tool", [{"call_id": "c1", "name": "calculate", "arguments": {"expression": "1+1"}}], None)
        transcript = {"tool_calls": [{"call_id": "c1", "name": "calculate", "arguments": {"expression": "1+1"}, "execution_outcome": "success", "result": "2"}], "final_answer": ""}
        result = classify(sample, transcript)
        self.assertTrue(result["layers"]["parse_success"])
        self.assertTrue(result["layers"]["schema_valid"])
        self.assertTrue(result["layers"]["call_plan_matches"])
        self.assertTrue(result["layers"]["execution_success"])
        self.assertEqual(result["layers"]["result_grounded"], None)  # no expected_result
        self.assertIsNone(result["layers"]["task_success"])  # not applicable, NOT failure
        self.assertIsNone(result["first_failure"])

    def test_wrong_arguments_fail_schema(self) -> None:
        sample = self._base_sample("single_tool", [{"call_id": "c1", "name": "calculate", "arguments": {"expression": "1+1"}}], "2")
        transcript = {"tool_calls": [{"call_id": "c1", "name": "calculate", "arguments": {"wrong": "x"}}], "final_answer": "2"}
        result = classify(sample, transcript)
        self.assertTrue(result["layers"]["parse_success"])
        self.assertFalse(result["layers"]["schema_valid"])
        self.assertEqual(result["first_failure"], "schema_valid")

    def test_wrong_tool_name_fails_plan(self) -> None:
        sample = self._base_sample("single_tool", [{"call_id": "c1", "name": "calculate", "arguments": {"expression": "1+1"}}], "2")
        transcript = {"tool_calls": [{"call_id": "c1", "name": "get_weather", "arguments": {"city": "北京"}}], "final_answer": "2"}
        result = classify(sample, transcript)
        self.assertTrue(result["layers"]["parse_success"])
        self.assertTrue(result["layers"]["schema_valid"])
        self.assertFalse(result["layers"]["call_plan_matches"])
        self.assertEqual(result["first_failure"], "call_plan_matches")

    def test_wrong_order_fails_plan(self) -> None:
        expected = [
            {"call_id": "a", "name": "calculate", "arguments": {"expression": "1+1"}},
            {"call_id": "b", "name": "get_weather", "arguments": {"city": "北京"}, "depends_on": ["a"]},
        ]
        sample = self._base_sample("multi_tool", expected, None)
        # Reversed order: weather called first.
        transcript = {"tool_calls": [
            {"call_id": "b", "name": "get_weather", "arguments": {"city": "北京"}, "depends_on": ["a"], "execution_outcome": "success"},
            {"call_id": "a", "name": "calculate", "arguments": {"expression": "1+1"}, "execution_outcome": "success"},
        ], "final_answer": ""}
        result = classify(sample, transcript)
        self.assertTrue(result["layers"]["parse_success"])
        self.assertTrue(result["layers"]["schema_valid"])
        self.assertFalse(result["layers"]["call_plan_matches"])
        self.assertEqual(result["first_failure"], "call_plan_matches")

    def test_execution_failure(self) -> None:
        sample = self._base_sample("single_tool", [{"call_id": "c1", "name": "calculate", "arguments": {"expression": "1+1"}}], None)
        transcript = {"tool_calls": [{"call_id": "c1", "name": "calculate", "arguments": {"expression": "1+1"}, "execution_outcome": "mock_exception", "error": "boom"}], "final_answer": ""}
        result = classify(sample, transcript)
        self.assertTrue(result["layers"]["parse_success"])
        self.assertTrue(result["layers"]["schema_valid"])
        self.assertTrue(result["layers"]["call_plan_matches"])
        self.assertFalse(result["layers"]["execution_success"])
        self.assertEqual(result["first_failure"], "execution_success")

    def test_result_grounding_mismatch(self) -> None:
        sample = self._base_sample(
            "single_tool",
            [{"call_id": "c1", "name": "calculate", "arguments": {"expression": "1+1"}, "expected_result": "2"}],
            "2",
        )
        transcript = {"tool_calls": [{"call_id": "c1", "name": "calculate", "arguments": {"expression": "1+1"}, "execution_outcome": "success", "result": "3"}], "final_answer": "2"}
        result = classify(sample, transcript)
        self.assertTrue(result["layers"]["parse_success"])
        self.assertTrue(result["layers"]["schema_valid"])
        self.assertTrue(result["layers"]["call_plan_matches"])
        self.assertTrue(result["layers"]["execution_success"])
        self.assertFalse(result["layers"]["result_grounded"])
        self.assertEqual(result["first_failure"], "result_grounded")

    def test_task_failure_with_answer(self) -> None:
        sample = self._base_sample("single_tool", [{"call_id": "c1", "name": "calculate", "arguments": {"expression": "1+1"}}], "2")
        transcript = {"tool_calls": [{"call_id": "c1", "name": "calculate", "arguments": {"expression": "1+1"}, "execution_outcome": "success", "result": "2"}], "final_answer": "我不知道"}
        result = classify(sample, transcript)
        self.assertTrue(result["layers"]["result_grounded"] is None or result["layers"]["result_grounded"])
        self.assertFalse(result["layers"]["task_success"])
        self.assertEqual(result["first_failure"], "task_success")

    def test_every_d1_task_type_has_working_sample(self) -> None:
        """Each D1 task type must have a representative sample that can be
        classified without crashing, and a correct-output transcript must not
        fail any layer."""
        manifest = json.loads((D1_DIR / "MANIFEST.json").read_text(encoding="utf-8"))
        seen: set[str] = set()
        for entry in manifest["samples"]:
            ttype = entry["task_type"]
            if ttype in seen:
                continue
            seen.add(ttype)
            sample = json.loads((D1_DIR / entry["path"]).read_text(encoding="utf-8"))
            # Build a perfectly-correct transcript from the sample's own
            # expected calls (empty for no_tool / tool_error).
            calls = [
                {"call_id": e.get("call_id", f"c{i}"), "name": e["name"], "arguments": e["arguments"],
                 "depends_on": e.get("depends_on", []), "execution_outcome": "success",
                 "result": e.get("expected_result")}
                for i, e in enumerate(sample["expected_tool_calls"])
            ]
            transcript = {"tool_calls": calls, "final_answer": sample.get("expected_answer") or "ok"}
            result = classify(sample, transcript)
            self.assertIsNone(result["first_failure"], msg=f"{ttype}: correct output failed: {result}")
        self.assertEqual(seen, ALL_TASK_TYPES)


if __name__ == "__main__":
    unittest.main()
