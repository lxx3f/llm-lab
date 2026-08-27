"""D1 dataset generator + P1-05 failure classifier tests."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any

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

    def test_canonical_expected_call_count_is_117(self) -> None:
        """Defensive contract (auditor round 11): canonical D1 has exactly
        117 expected tool calls across all 126 samples, all with
        deterministic expected_result. Counting by task_type:
            single_tool (36 samples × 1 call) = 36
            multi_tool (18 samples × 2 calls) = 36
            tool_error_response (9 samples × 1 call) = 9
            insufficient_result (18 samples × 1 call) = 18
            requirement_change (18 samples × 1 call) = 18
            total = 117
        If the generator changes (e.g. new sub-scenario added), this test
        forces the documentation, protocols, and completion summary to be
        updated to match."""
        manifest = json.loads((D1_DIR / "MANIFEST.json").read_text(encoding="utf-8"))
        counts_by_type: dict[str, int] = {}
        total_calls = 0
        calls_with_expected_result = 0
        for entry in manifest["samples"]:
            sample = json.loads((D1_DIR / entry["path"]).read_text(encoding="utf-8"))
            calls = sample.get("expected_tool_calls", [])
            total_calls += len(calls)
            counts_by_type[sample["metadata"]["task_type"]] = (
                counts_by_type.get(sample["metadata"]["task_type"], 0) + len(calls)
            )
            for c in calls:
                if "expected_result" in c:
                    calls_with_expected_result += 1
        self.assertEqual(
            counts_by_type,
            {"single_tool": 36, "multi_tool": 36, "tool_error": 9,
             "insufficient_result": 18, "requirement_change": 18, "no_tool": 0},
            msg=f"call counts by task_type drifted: {counts_by_type}",
        )
        self.assertEqual(total_calls, 117,
                         msg=f"canonical expected call count drifted from 117 to {total_calls}")
        self.assertEqual(calls_with_expected_result, 117,
                         msg=f"expected_result coverage drifted: {calls_with_expected_result}/117")

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
                    "name": "d1_calculate",
                    "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
                }},
                {"type": "function", "function": {
                    "name": "d1_get_weather",
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
        self.assertTrue(result["layers"]["result_grounded"])  # trivially passes (nothing to ground)
        self.assertTrue(result["layers"]["task_success"])
        self.assertIsNone(result["first_failure"])

    def test_no_tool_calling_a_tool_fails_plan(self) -> None:
        sample = self._base_sample("no_tool", [], "直接回答")
        transcript = {"tool_calls": [{"call_id": "c1", "name": "d1_calculate", "arguments": {"expression": "1+1"}}], "final_answer": "直接回答"}
        result = classify(sample, transcript)
        self.assertTrue(result["layers"]["parse_success"])
        self.assertFalse(result["layers"]["tool_name_correct"])
        self.assertFalse(result["layers"]["call_plan_matches"])
        self.assertEqual(result["first_failure"], "tool_name_correct")

    def test_single_tool_null_answer_task_layer_not_applicable(self) -> None:
        """single_tool with expected_answer null: task_success is None, not False."""
        sample = self._base_sample("single_tool", [{"call_id": "c1", "name": "d1_calculate", "arguments": {"expression": "1+1"}}], None)
        transcript = {"tool_calls": [{"call_id": "c1", "name": "d1_calculate", "arguments": {"expression": "1+1"}, "execution_outcome": "success", "result": "2"}], "final_answer": ""}
        result = classify(sample, transcript)
        self.assertTrue(result["layers"]["parse_success"])
        self.assertTrue(result["layers"]["schema_valid"])
        self.assertTrue(result["layers"]["call_plan_matches"])
        self.assertTrue(result["layers"]["execution_success"])
        self.assertEqual(result["layers"]["result_grounded"], None)  # no expected_result
        self.assertIsNone(result["layers"]["task_success"])  # not applicable, NOT failure
        self.assertIsNone(result["first_failure"])

    def test_wrong_arguments_fail_schema(self) -> None:
        sample = self._base_sample("single_tool", [{"call_id": "c1", "name": "d1_calculate", "arguments": {"expression": "1+1"}}], "2")
        transcript = {"tool_calls": [{"call_id": "c1", "name": "d1_calculate", "arguments": {"wrong": "x"}}], "final_answer": "2"}
        result = classify(sample, transcript)
        self.assertTrue(result["layers"]["parse_success"])
        self.assertFalse(result["layers"]["schema_valid"])
        self.assertEqual(result["first_failure"], "schema_valid")

    def test_wrong_tool_name_fails_plan(self) -> None:
        sample = self._base_sample("single_tool", [{"call_id": "c1", "name": "d1_calculate", "arguments": {"expression": "1+1"}}], "2")
        transcript = {"tool_calls": [{"call_id": "c1", "name": "d1_get_weather", "arguments": {"city": "北京"}}], "final_answer": "2"}
        result = classify(sample, transcript)
        self.assertTrue(result["layers"]["parse_success"])
        self.assertTrue(result["layers"]["schema_valid"])
        self.assertFalse(result["layers"]["tool_name_correct"])
        self.assertFalse(result["layers"]["call_plan_matches"])
        self.assertEqual(result["first_failure"], "tool_name_correct")

    def test_wrong_order_fails_plan(self) -> None:
        expected = [
            {"call_id": "a", "name": "d1_calculate", "arguments": {"expression": "1+1"}},
            {"call_id": "b", "name": "d1_get_weather", "arguments": {"city": "北京"}, "depends_on": ["a"]},
        ]
        sample = self._base_sample("multi_tool", expected, None)
        # Reversed order: weather called first.
        transcript = {"tool_calls": [
            {"call_id": "b", "name": "d1_get_weather", "arguments": {"city": "北京"}, "depends_on": ["a"], "execution_outcome": "success"},
            {"call_id": "a", "name": "d1_calculate", "arguments": {"expression": "1+1"}, "execution_outcome": "success"},
        ], "final_answer": ""}
        result = classify(sample, transcript)
        self.assertTrue(result["layers"]["parse_success"])
        self.assertTrue(result["layers"]["schema_valid"])
        self.assertFalse(result["layers"]["call_plan_matches"])
        self.assertEqual(result["first_failure"], "call_plan_matches")

    def test_dangling_dependency_reference_fails_plan(self) -> None:
        """A depends_on id that does not exist in the call sequence is a
        dangling reference and must fail call_plan_matches."""
        expected = [
            {"call_id": "a", "name": "d1_calculate", "arguments": {"expression": "1+1"}},
            {"call_id": "b", "name": "d1_get_weather", "arguments": {"city": "北京"}, "depends_on": ["missing-id"]},
        ]
        sample = self._base_sample("multi_tool", expected, None)
        # The transcript faithfully reproduces the dangling reference.
        transcript = {"tool_calls": [
            {"call_id": "a", "name": "d1_calculate", "arguments": {"expression": "1+1"}, "execution_outcome": "success"},
            {"call_id": "b", "name": "d1_get_weather", "arguments": {"city": "北京"}, "depends_on": ["missing-id"], "execution_outcome": "success"},
        ], "final_answer": ""}
        result = classify(sample, transcript)
        self.assertTrue(result["layers"]["parse_success"])
        self.assertTrue(result["layers"]["schema_valid"])
        self.assertFalse(result["layers"]["call_plan_matches"])
        self.assertEqual(result["first_failure"], "call_plan_matches")

    def test_self_dependency_reference_fails_plan(self) -> None:
        """A call depending on itself is invalid."""
        expected = [
            {"call_id": "a", "name": "d1_calculate", "arguments": {"expression": "1+1"}},
        ]
        sample = self._base_sample("multi_tool", expected, None)
        transcript = {"tool_calls": [
            {"call_id": "a", "name": "d1_calculate", "arguments": {"expression": "1+1"}, "depends_on": ["a"], "execution_outcome": "success"},
        ], "final_answer": ""}
        result = classify(sample, transcript)
        self.assertFalse(result["layers"]["call_plan_matches"])
        self.assertEqual(result["first_failure"], "call_plan_matches")

    def test_execution_failure(self) -> None:
        sample = self._base_sample("single_tool", [{"call_id": "c1", "name": "d1_calculate", "arguments": {"expression": "1+1"}}], None)
        transcript = {"tool_calls": [{"call_id": "c1", "name": "d1_calculate", "arguments": {"expression": "1+1"}, "execution_outcome": "mock_exception", "error": "boom"}], "final_answer": ""}
        result = classify(sample, transcript)
        self.assertTrue(result["layers"]["parse_success"])
        self.assertTrue(result["layers"]["schema_valid"])
        self.assertTrue(result["layers"]["call_plan_matches"])
        self.assertFalse(result["layers"]["execution_success"])
        self.assertEqual(result["first_failure"], "execution_success")

    def test_result_grounding_mismatch(self) -> None:
        sample = self._base_sample(
            "single_tool",
            [{"call_id": "c1", "name": "d1_calculate", "arguments": {"expression": "1+1"}, "expected_result": "2"}],
            "2",
        )
        transcript = {"tool_calls": [{"call_id": "c1", "name": "d1_calculate", "arguments": {"expression": "1+1"}, "execution_outcome": "success", "result": "3"}], "final_answer": "2"}
        result = classify(sample, transcript)
        self.assertTrue(result["layers"]["parse_success"])
        self.assertTrue(result["layers"]["schema_valid"])
        self.assertTrue(result["layers"]["call_plan_matches"])
        self.assertTrue(result["layers"]["execution_success"])
        self.assertFalse(result["layers"]["result_grounded"])
        self.assertEqual(result["first_failure"], "result_grounded")

    def test_task_failure_with_answer(self) -> None:
        sample = self._base_sample("single_tool", [{"call_id": "c1", "name": "d1_calculate", "arguments": {"expression": "1+1"}}], "2")
        transcript = {"tool_calls": [{"call_id": "c1", "name": "d1_calculate", "arguments": {"expression": "1+1"}, "execution_outcome": "success", "result": "2"}], "final_answer": "我不知道"}
        result = classify(sample, transcript)
        self.assertTrue(result["layers"]["result_grounded"] is None or result["layers"]["result_grounded"])
        self.assertFalse(result["layers"]["task_success"])
        self.assertFalse(result["layers"]["final_answer_correct"])
        # 8-level taxonomy reports the canonical name for the answer layer.
        self.assertEqual(result["first_failure"], "final_answer_correct")

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

    def test_wrong_argument_value_localized_to_argument_value_correct(self) -> None:
        """8-level extension (auditor round 1 for this stage): a transcript
        with the right tool NAME but a WRONG argument value must fail at
        ``argument_value_correct``, not at ``call_plan_matches``."""
        sample = self._base_sample(
            "single_tool",
            [{"call_id": "c1", "name": "d1_calculate",
              "arguments": {"expression": "1+1"}, "expected_result": 2}],
            "2",
        )
        transcript = {"tool_calls": [{
            "call_id": "c1", "name": "d1_calculate",
            "arguments": {"expression": "1+2"},  # wrong value, right name
            "execution_outcome": "success", "result": 3,
        }], "final_answer": "3"}
        result = classify(sample, transcript)
        self.assertTrue(result["layers"]["parse_success"])
        self.assertTrue(result["layers"]["schema_valid"])  # 1+2 is schema-valid
        self.assertTrue(result["layers"]["tool_name_correct"])  # name is right
        self.assertFalse(result["layers"]["argument_value_correct"])  # args wrong
        self.assertEqual(result["first_failure"], "argument_value_correct")

    def test_wrong_order_still_localized_to_call_plan_matches(self) -> None:
        """8-level extension: a reversed order (right names/args, wrong
        sequence) must fail at ``call_plan_matches`` — tool_name_correct and
        argument_value_correct stay True because the name/arg multisets
        match."""
        expected = [
            {"call_id": "a", "name": "d1_calculate", "arguments": {"expression": "1+1"}},
            {"call_id": "b", "name": "d1_get_weather", "arguments": {"city": "北京"}, "depends_on": ["a"]},
        ]
        sample = self._base_sample("multi_tool", expected, None)
        transcript = {"tool_calls": [
            {"call_id": "b", "name": "d1_get_weather", "arguments": {"city": "北京"}, "depends_on": ["a"], "execution_outcome": "success"},
            {"call_id": "a", "name": "d1_calculate", "arguments": {"expression": "1+1"}, "execution_outcome": "success"},
        ], "final_answer": ""}
        result = classify(sample, transcript)
        self.assertTrue(result["layers"]["tool_name_correct"])
        self.assertTrue(result["layers"]["argument_value_correct"])
        self.assertFalse(result["layers"]["call_plan_matches"])
        self.assertEqual(result["first_failure"], "call_plan_matches")

    def test_final_answer_correct_alias_matches_task_success(self) -> None:
        """8-level extension: ``final_answer_correct`` is the canonical name
        of the final-answer layer and must equal ``task_success`` (alias)."""
        sample = self._base_sample(
            "single_tool",
            [{"call_id": "c1", "name": "d1_calculate",
              "arguments": {"expression": "1+1"}, "expected_result": 2}],
            "2",
        )
        good = classify(sample, {"tool_calls": [{
            "call_id": "c1", "name": "d1_calculate", "arguments": {"expression": "1+1"},
            "execution_outcome": "success", "result": 2}], "final_answer": "结果是 2"})
        self.assertIs(good["layers"]["final_answer_correct"], True)
        self.assertIs(good["layers"]["task_success"], True)
        self.assertEqual(good["layers"]["final_answer_correct"], good["layers"]["task_success"])
        bad = classify(sample, {"tool_calls": [{
            "call_id": "c1", "name": "d1_calculate", "arguments": {"expression": "1+1"},
            "execution_outcome": "success", "result": 2}], "final_answer": "结果是 3"})
        self.assertIs(bad["layers"]["final_answer_correct"], False)
        self.assertIs(bad["layers"]["task_success"], False)
        self.assertEqual(bad["first_failure"], "final_answer_correct")

    def test_unknown_tool_name_localized_to_tool_name_correct(self) -> None:
        """Regression (auditor round 1 for this stage): a transcript call
        whose tool name does NOT exist in the declared tool registry (or in
        the expected plan) must be classified first as ``tool_name_correct``,
        NOT ``schema_valid``. Previously the schema layer set
        schema_valid=False for an unknown tool, masking the name error."""
        sample = self._base_sample(
            "single_tool",
            [{"call_id": "c1", "name": "d1_calculate",
              "arguments": {"expression": "1+1"}, "expected_result": 2}],
            "2",
        )
        transcript = {"tool_calls": [{
            "call_id": "c1", "name": "unknown_tool",
            "arguments": {"x": 1},
            "execution_outcome": "success",
        }], "final_answer": "2"}
        result = classify(sample, transcript)
        self.assertTrue(result["layers"]["parse_success"])
        self.assertIsNone(result["layers"]["schema_valid"])  # not checkable
        self.assertFalse(result["layers"]["tool_name_correct"])  # unknown name
        self.assertEqual(result["first_failure"], "tool_name_correct")

    def test_mixed_nonstring_names_do_not_crash(self) -> None:
        """Regression (auditor round 1 for this stage): a structurally valid
        list of dicts with mixed/missing/non-string ``name`` values must
        classify without crashing sorted() — non-string names are normalized
        to a stable placeholder and always mismatch the expected plan."""
        sample = self._base_sample(
            "single_tool",
            [{"call_id": "c1", "name": "d1_calculate",
              "arguments": {"expression": "1+1"}, "expected_result": 2}],
            "2",
        )
        for names in ([None], [123], ["d1_calculate", None], [None, 1, "x"], [{}]):
            calls = []
            for i, n in enumerate(names):
                call: dict[str, Any] = {"call_id": f"c{i}"}
                if n == "d1_calculate":
                    # Valid name with schema-valid arguments so any failure
                    # must come from the name layer, not the schema layer.
                    call["name"] = n
                    call["arguments"] = {"expression": "1+1"}
                elif n == {}:
                    pass  # empty dict: no name, no arguments
                else:
                    call["name"] = n
                calls.append(call)
            result = classify(sample, {"tool_calls": calls, "final_answer": "2"})
            self.assertIn("tool_name_correct", result["layers"], msg=f"names={names!r}")
            self.assertFalse(result["layers"]["tool_name_correct"], msg=f"names={names!r}")
            self.assertEqual(result["first_failure"], "tool_name_correct", msg=f"names={names!r}")

    def test_mixed_nonstring_arguments_do_not_crash(self) -> None:
        """Regression (auditor round 1 for this stage): arguments that are
        not plain JSON dicts (None, str, list, dict with non-string keys)
        must not crash argument_value_correct serialization."""
        sample = self._base_sample(
            "single_tool",
            [{"call_id": "c1", "name": "d1_calculate",
              "arguments": {"expression": "1+1"}, "expected_result": 2}],
            "2",
        )
        for args in (None, "x", 1, [1, 2], {1: "a"}):
            result = classify(sample, {"tool_calls": [{
                "call_id": "c1", "name": "d1_calculate", "arguments": args,
            }], "final_answer": "2"})
            self.assertIn("argument_value_correct", result["layers"],
                          msg=f"args={args!r}")
            self.assertIsInstance(result["layers"]["argument_value_correct"], bool,
                                  msg=f"args={args!r}")

    def test_mixed_dependency_id_types_do_not_crash(self) -> None:
        """Regression (auditor round 2 for this stage): ``depends_on``
        values with mixed types (str + int) previously crashed sorted(). The
        plan layer must normalize ids and report call_plan_matches=False."""
        sample = self._base_sample(
            "multi_tool",
            [{"call_id": "a", "name": "d1_calculate", "arguments": {"expression": "1+1"}},
             {"call_id": "b", "name": "d1_get_weather", "arguments": {"city": "北京"}, "depends_on": ["a"]}],
            None,
        )
        transcript = {"tool_calls": [
            {"call_id": "a", "name": "d1_calculate", "arguments": {"expression": "1+1"},
             "depends_on": ["x", 1], "execution_outcome": "success"},
            {"call_id": "b", "name": "d1_get_weather", "arguments": {"city": "北京"},
             "depends_on": ["a"], "execution_outcome": "success"},
        ], "final_answer": ""}
        result = classify(sample, transcript)
        self.assertTrue(result["layers"]["tool_name_correct"])
        self.assertTrue(result["layers"]["argument_value_correct"])
        self.assertFalse(result["layers"]["call_plan_matches"])
        self.assertEqual(result["first_failure"], "call_plan_matches")

    def test_unhashable_call_id_does_not_crash(self) -> None:
        """Regression (auditor round 2 for this stage): a dict ``call_id``
        is unhashable and previously crashed set construction in
        call_plan_matches. The id must be normalized to a string and the
        plan mismatch reported without raising."""
        sample = self._base_sample(
            "multi_tool",
            [{"call_id": "a", "name": "d1_calculate", "arguments": {"expression": "1+1"}},
             {"call_id": "b", "name": "d1_get_weather", "arguments": {"city": "北京"}, "depends_on": ["a"]}],
            None,
        )
        transcript = {"tool_calls": [
            {"call_id": {"k": 1}, "name": "d1_calculate", "arguments": {"expression": "1+1"},
             "execution_outcome": "success"},
            {"call_id": "b", "name": "d1_get_weather", "arguments": {"city": "北京"},
             "depends_on": ["a"], "execution_outcome": "success"},
        ], "final_answer": ""}
        result = classify(sample, transcript)
        self.assertFalse(result["layers"]["call_plan_matches"])
        self.assertEqual(result["first_failure"], "call_plan_matches")

    def test_all_eight_layers_present_in_ok_transcript(self) -> None:
        """8-level extension: a fully-correct transcript populates all eight
        layers with booleans (no None for an answer-declared sample)."""
        sample = self._base_sample(
            "single_tool",
            [{"call_id": "c1", "name": "d1_calculate",
              "arguments": {"expression": "1+1"}, "expected_result": 2}],
            "2",
        )
        transcript = {"tool_calls": [{
            "call_id": "c1", "name": "d1_calculate", "arguments": {"expression": "1+1"},
            "execution_outcome": "success", "result": 2}], "final_answer": "2"}
        result = classify(sample, transcript)
        expected_keys = {"parse_success", "schema_valid", "tool_name_correct",
                         "argument_value_correct", "call_plan_matches",
                         "execution_success", "result_grounded",
                         "task_success", "final_answer_correct"}
        self.assertEqual(set(result["layers"].keys()), expected_keys)
        for name, ok in result["layers"].items():
            self.assertIsInstance(ok, bool, msg=f"{name} should be bool for ok transcript")
        self.assertIsNone(result["first_failure"])


class D1SemanticIntegrationTests(unittest.TestCase):
    """End-to-end semantic checks across all canonical D1 samples.

    These run MockExecutor over every sample's ``expected_tool_calls`` and
    assert that (a) every call executes successfully, (b) every declared
    ``expected_result`` matches the mock's actual return value, and (c)
    every expected ``depends_on`` reference points at a call_id that exists
    in the same sample's plan.
    """

    def _load_samples(self) -> list[dict]:
        manifest = json.loads((D1_DIR / "MANIFEST.json").read_text(encoding="utf-8"))
        return [json.loads((D1_DIR / e["path"]).read_text(encoding="utf-8")) for e in manifest["samples"]]

    def test_every_expected_call_has_mock_executable_expected_result(self) -> None:
        """Every expected call must have a declared expected_result, and the
        mock must actually return it when invoked with the declared arguments."""
        import examples.d1_mocks as d1_mocks
        registry = {"d1_calculate": d1_mocks.d1_calculate,
                    "d1_get_weather": d1_mocks.d1_get_weather,
                    "d1_web_search": d1_mocks.d1_web_search,
                    "d1_translate": d1_mocks.d1_translate}
        samples = self._load_samples()
        missing: list[str] = []
        mismatched: list[str] = []
        for sample in samples:
            for call in sample.get("expected_tool_calls", []):
                if "expected_result" not in call:
                    missing.append(f"{sample['id']}:{call.get('call_id')}")
                    continue
                fn = registry.get(call["name"])
                if fn is None:
                    mismatched.append(f"{sample['id']}:{call['call_id']} unknown tool")
                    continue
                actual = fn(**call["arguments"])
                if actual != call["expected_result"]:
                    mismatched.append(f"{sample['id']}:{call['call_id']} mock={actual!r} expected={call['expected_result']!r}")
        self.assertEqual(missing, [], msg=f"samples without expected_result: {missing}")
        self.assertEqual(mismatched, [], msg=f"mock/expected mismatches: {mismatched[:5]}")

    def test_every_expected_call_runs_through_mock_executor_successfully(self) -> None:
        from architecture_lab.execution import MockExecutor
        import examples.d1_mocks as d1_mocks
        registry = {"d1_calculate": d1_mocks.d1_calculate,
                    "d1_get_weather": d1_mocks.d1_get_weather,
                    "d1_web_search": d1_mocks.d1_web_search,
                    "d1_translate": d1_mocks.d1_translate}
        samples = self._load_samples()
        failures: list[str] = []
        for sample in samples:
            executor = MockExecutor()
            for tool in sample.get("tools", []):
                fn = registry.get(tool["function"]["name"])
                if fn is not None:
                    executor.register_mock(tool["function"]["name"], fn, tool["function"]["parameters"])
            calls = [
                {"tool_name": e["name"], "call_id": e["call_id"],
                 "arguments": e["arguments"], "depends_on": e.get("depends_on", [])}
                for e in sample.get("expected_tool_calls", [])
            ]
            results = executor.execute_sequence(calls)
            for exp, r in zip(sample.get("expected_tool_calls", []), results):
                if r["outcome"] != "success":
                    failures.append(f"{sample['id']}:{exp['call_id']} outcome={r['outcome']!r}")
        self.assertEqual(failures, [], msg=f"mock-execution failures: {failures[:5]}")

    def test_every_depends_on_reference_resolves(self) -> None:
        """Every depends_on id in a sample's expected_tool_calls must refer
        to a call_id present in the same sample."""
        samples = self._load_samples()
        dangling: list[str] = []
        for sample in samples:
            ids = {e["call_id"] for e in sample.get("expected_tool_calls", [])}
            for call in sample.get("expected_tool_calls", []):
                for dep in call.get("depends_on") or []:
                    if dep not in ids or dep == call["call_id"]:
                        dangling.append(f"{sample['id']}:{call['call_id']} -> {dep}")
        self.assertEqual(dangling, [], msg=f"dangling dependencies: {dangling[:5]}")

    def test_no_tool_and_tool_error_unavailable_samples_have_zero_expected_calls(self) -> None:
        """no_tool and the ``tool_not_available`` sub-scenario must have empty
        expected_tool_calls. Note: ``tool_error_response`` (the other
        tool_error sub-scenario) DOES have one call — the model is expected
        to invoke the tool, observe the ERROR response, and report the
        failure (auditor round 10)."""
        samples = self._load_samples()
        for sample in samples:
            ttype = sample["metadata"]["task_type"]
            template = sample["metadata"].get("task_template", "")
            if ttype == "no_tool" or (ttype == "tool_error" and template == "tool_not_available"):
                self.assertEqual(
                    sample["expected_tool_calls"], [],
                    msg=f"{sample['id']} ({ttype}/{template}) must have empty expected_tool_calls",
                )

    def test_expected_answer_consistent_with_task_type(self) -> None:
        """no_tool/tool_error/insufficient_result/requirement_change samples
        must have a non-null expected_answer; single_tool/multi_tool may or
        may not (computed results are typically included)."""
        samples = self._load_samples()
        require_answer = {"no_tool", "tool_error", "insufficient_result", "requirement_change"}
        missing: list[str] = []
        for sample in samples:
            ttype = sample["metadata"]["task_type"]
            if ttype in require_answer and not sample.get("expected_answer"):
                missing.append(f"{sample['id']} ({ttype})")
        self.assertEqual(missing, [], msg=f"samples missing expected_answer: {missing}")

    def test_canonical_samples_activate_result_grounded_layer(self) -> None:
        """result_grounded must be either True or False (not None) for every
        sample with at least one expected call, so the layer is actually
        exercised on canonical data."""
        samples = self._load_samples()
        undetermined: list[str] = []
        for sample in samples:
            if not sample.get("expected_tool_calls"):
                continue
            layers = classify(sample, {
                "tool_calls": [
                    {"call_id": e["call_id"], "name": e["name"], "arguments": e["arguments"],
                     "execution_outcome": "success", "result": e.get("expected_result")}
                    for e in sample["expected_tool_calls"]
                ],
                "final_answer": sample.get("expected_answer") or "",
            })["layers"]
            if layers["result_grounded"] is None:
                undetermined.append(f"{sample['id']}")
        self.assertEqual(undetermined, [], msg=f"result_grounded never activated: {undetermined}")


class MalformedTranscriptRegressionTests(unittest.TestCase):
    """Reverse-assertion tests: classify() must not crash and must report
    parse_success=False (with first_failure='parse_success') when the
    transcript's ``tool_calls`` field is missing, null, or non-list."""

    def _sample(self) -> dict:
        return {
            "schema_version": "1.0",
            "id": "malformed-test",
            "messages": [{"role": "user", "content": "q"}],
            "tools": [{"type": "function", "function": {
                "name": "d1_calculate",
                "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
            }}],
            "expected_tool_calls": [{"call_id": "c1", "name": "d1_calculate", "arguments": {"expression": "1+1"}}],
            "expected_answer": "2",
            "metadata": {"source": "test", "license": "test", "task_type": "single_tool",
                         "data_version": "D1", "pipeline_version": "test", "created_at": "2026-08-27T00:00:00Z",
                         "validation": {"schema_valid": True}},
        }

    def test_missing_tool_calls_field_fails_parse(self) -> None:
        """No ``tool_calls`` key at all → parse_success=False, no crash."""
        sample = self._sample()
        result = classify(sample, {"final_answer": "2"})
        self.assertFalse(result["layers"]["parse_success"])
        self.assertEqual(result["first_failure"], "parse_success")
        # All downstream layers are N/A when the transcript is unparseable.
        for name in ("schema_valid", "tool_name_correct", "argument_value_correct",
                     "call_plan_matches", "execution_success", "result_grounded"):
            self.assertIsNone(result["layers"][name], msg=f"{name} should be None when parse fails")

    def test_null_tool_calls_fails_parse(self) -> None:
        """``tool_calls: None`` → parse_success=False (regression: previously
        crashed with TypeError on len(None))."""
        sample = self._sample()
        result = classify(sample, {"tool_calls": None, "final_answer": "2"})
        self.assertFalse(result["layers"]["parse_success"])
        self.assertEqual(result["first_failure"], "parse_success")

    def test_non_list_tool_calls_fails_parse(self) -> None:
        """``tool_calls: 'not a list'`` (string), ``{}`` (dict), ``42`` (int)
        are all malformed parses."""
        sample = self._sample()
        for bad in ("not a list", {"oops": True}, 42, 3.14):
            result = classify(sample, {"tool_calls": bad, "final_answer": "2"})
            self.assertFalse(result["layers"]["parse_success"], msg=f"input={bad!r}")
            self.assertEqual(result["first_failure"], "parse_success", msg=f"input={bad!r}")

    def test_task_success_still_evaluated_when_parse_fails(self) -> None:
        """Even with a malformed tool_calls, the final_answer can be checked
        against expected_answer (task_success is independent of parsing)."""
        sample = self._sample()
        result = classify(sample, {"tool_calls": None, "final_answer": "2"})
        self.assertTrue(result["layers"]["task_success"])
        result = classify(sample, {"tool_calls": None, "final_answer": "nope"})
        self.assertFalse(result["layers"]["task_success"])

    def test_null_tool_calls_does_not_crash_execution_layer(self) -> None:
        """Regression: previously, ``execution_success``/``result_grounded``
        blocks short-circuited correctly, but ``call_plan_matches`` did not.
        Verify the whole pipeline survives a null tool_calls."""
        sample = self._sample()
        # If this raises, the regression is back.
        result = classify(sample, {"tool_calls": None, "final_answer": "anything"})
        self.assertIn("layers", result)
        self.assertIn("first_failure", result)
        self.assertIsInstance(result["layers"], dict)
        self.assertEqual(set(result["layers"].keys()),
                         {"parse_success", "schema_valid", "tool_name_correct",
                          "argument_value_correct", "call_plan_matches",
                          "execution_success", "result_grounded", "task_success",
                          "final_answer_correct"})

    def test_malformed_list_members_fail_parse_without_crash(self) -> None:
        """Regression (auditor round 8): ``[None]``, ``[1]``, ``["x"]``
        previously crashed with AttributeError on ``call.get(...)``. The
        classifier must now detect a list whose members are not all
        structured mappings and report ``parse_success=False``."""
        sample = self._sample()
        for bad in ([None], [1], ["x"], [None, {"call_id": "c1"}], [True, False]):
            result = classify(sample, {"tool_calls": bad, "final_answer": "2"})
            self.assertFalse(result["layers"]["parse_success"], msg=f"input={bad!r}")
            self.assertEqual(result["first_failure"], "parse_success", msg=f"input={bad!r}")
            # Downstream layers must be N/A (None), not crash.
            for name in ("schema_valid", "tool_name_correct", "argument_value_correct",
                         "call_plan_matches", "execution_success", "result_grounded"):
                self.assertIsNone(result["layers"][name], msg=f"{name} for {bad!r}")

    def test_empty_dict_member_parses_but_fails_name(self) -> None:
        """``[{}]`` is technically a list of dicts, so parse_success is True,
        but the empty mapping has no ``name`` — it cannot be matched to any
        declared tool, so schema_valid is N/A (None) and tool_name_correct
        reports the name mismatch (auditor round 1 for this stage)."""
        sample = self._sample()
        result = classify(sample, {"tool_calls": [{}], "final_answer": "2"})
        self.assertTrue(result["layers"]["parse_success"])
        self.assertIsNone(result["layers"]["schema_valid"])
        self.assertFalse(result["layers"]["tool_name_correct"])
        self.assertEqual(result["first_failure"], "tool_name_correct")

    def test_mixed_list_with_malformed_member_fails_parse(self) -> None:
        """If even one member is non-dict, parse fails entirely — never
        partial-parse, never crash."""
        sample = self._sample()
        result = classify(sample, {
            "tool_calls": [
                {"call_id": "c1", "name": "d1_calculate", "arguments": {"expression": "1+1"}},
                None,  # malformed member
            ],
            "final_answer": "2",
        })
        self.assertFalse(result["layers"]["parse_success"])
        self.assertEqual(result["first_failure"], "parse_success")

    def test_call_id_mismatch_does_not_produce_false_pass(self) -> None:
        """Regression (auditor round 10): a transcript with the correct tool
        name/arguments/result but a DIFFERENT call_id must still be grounded
        against the expected result by POSITION. Previously, result_grounded
        matched by call_id only and silently returned None (N/A), allowing a
        transcript with a wrong result to report ``first_failure=None``."""
        sample = {
            "schema_version": "1.0",
            "id": "test-callid-mismatch",
            "messages": [{"role": "user", "content": "q"}],
            "tools": [{"type": "function", "function": {
                "name": "d1_calculate",
                "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
            }}],
            "expected_tool_calls": [{"call_id": "expected-call-id",
                                     "name": "d1_calculate",
                                     "arguments": {"expression": "1+1"},
                                     "expected_result": 2}],
            "expected_answer": "2",
            "metadata": {"source": "test", "license": "test", "task_type": "single_tool",
                         "data_version": "D1", "pipeline_version": "test", "created_at": "2026-08-27T00:00:00Z",
                         "validation": {"schema_valid": True}},
        }
        # Transcript: same tool name/arguments/result as the expected call,
        # but a totally different call_id. Under the old (buggy) behavior
        # result_grounded could not find a match and returned None.
        transcript = {
            "tool_calls": [{
                "call_id": "different-call-id",
                "name": "d1_calculate",
                "arguments": {"expression": "1+1"},
                "execution_outcome": "success",
                "result": 2,
            }],
            "final_answer": "2",
        }
        result = classify(sample, transcript)
        # Positional pairing now correctly matches expected_result=2 with
        # actual result=2; result_grounded is True (not None) and there is
        # no first_failure.
        self.assertIs(result["layers"]["result_grounded"], True,
                      msg=f"expected True (positional match), got {result['layers']!r}")
        self.assertIsNone(result["first_failure"])

    def test_call_id_mismatch_with_wrong_result_fails_grounding(self) -> None:
        """Companion regression (auditor round 10): when call_id differs
        BUT the actual result also differs from expected_result, the
        classifier must catch the wrong result (positional pairing), not
        silently return None."""
        sample = {
            "schema_version": "1.0",
            "id": "test-callid-wrong-result",
            "messages": [{"role": "user", "content": "q"}],
            "tools": [{"type": "function", "function": {
                "name": "d1_calculate",
                "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
            }}],
            "expected_tool_calls": [{"call_id": "expected-call-id",
                                     "name": "d1_calculate",
                                     "arguments": {"expression": "1+1"},
                                     "expected_result": 2}],
            "expected_answer": "2",
            "metadata": {"source": "test", "license": "test", "task_type": "single_tool",
                         "data_version": "D1", "pipeline_version": "test", "created_at": "2026-08-27T00:00:00Z",
                         "validation": {"schema_valid": True}},
        }
        transcript = {
            "tool_calls": [{
                "call_id": "different-call-id",
                "name": "d1_calculate",
                "arguments": {"expression": "1+1"},
                "execution_outcome": "success",
                "result": 999,  # wrong result
            }],
            "final_answer": "2",
        }
        result = classify(sample, transcript)
        self.assertIs(result["layers"]["result_grounded"], False)
        self.assertEqual(result["first_failure"], "result_grounded")


if __name__ == "__main__":
    unittest.main()
