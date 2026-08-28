"""Tests for the P2 offline reward evaluator (``scripts/reward_offline.py``).

Layout: one ``_BaseCase`` helper that builds a tool-calling sample, then a
suite of tests organised by the eight P1-05 layers, plus a direct
consistency test against ``scripts/classify_tool_failure.classify``. The
goal is to cover the layer matrix required by the P2 objective:

> 8 级各样本（3-5 个）× reward 校验（正例、负例、边界）；
> 与 P1-05 分类器输出一致性测试。

Layers covered here:

- parse_success: positive (full pass), negative (non-list / non-dict), boundary (empty list)
- schema_valid: positive (declared tool with matching args), negative (unknown tool), boundary (missing tool registry)
- tool_name_correct: positive (single name), negative (wrong name), boundary (extra names)
- argument_value_correct: positive (matches expected), negative (mismatch), boundary (no expected calls)
- call_plan_matches: positive (right order), negative (wrong order), boundary (with depends_on)
- execution_success: positive (success outcome), negative (mock_exception), boundary (no calls)
- result_grounded: positive (result == expected), negative (mismatch), boundary (no expected_result)
- final_answer_correct: positive (answer substring), negative (wrong answer), boundary (no expected_answer)
"""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REWARD = ROOT / "scripts" / "reward_offline.py"
CLASSIFIER = ROOT / "scripts" / "classify_tool_failure.py"
TMP = ROOT / "artifacts" / "test-reward-offline-tmp"


def _tool_decl(name: str = "d1_get_weather",
              *, prop: str = "city", required: tuple[str, ...] = ("city",)) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {prop: {"type": "string"}},
                "required": list(required),
            },
        },
    }


def _ok_call(call_id: str, *, name: str = "d1_get_weather",
            city: str = "北京", expected_result: str = "北京 22C") -> dict:
    return {
        "call_id": call_id,
        "name": name,
        "arguments": {"city": city},
        "expected_result": expected_result,
        "execution_outcome": "success",
        "result": expected_result,
    }


def _row(sample_id: str, *, calls: Any = None, generated: str = "") -> dict:
    return {
        "sample_id": sample_id,
        "extracted_calls": calls if calls is not None else [],
        "first_failure": None,
        "layers": {},
        "generated": generated,
    }


def _sample(sample_id: str, *, expected_calls: list[dict],
            answer: str | None = None, task_type: str = "single_tool",
            tools: list[dict] | None = None) -> dict:
    sample: dict[str, Any] = {
        "schema_version": "1.0",
        "id": sample_id,
        "expected_tool_calls": expected_calls,
        "metadata": {"task_type": task_type},
    }
    if answer is not None:
        sample["expected_answer"] = answer
    if tools is not None:
        sample["tools"] = tools
    return sample


def _load_compute_reward() -> Any:
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("_reward_offline", REWARD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_reward


class ParseSuccessLayerTests(unittest.TestCase):
    """parse_success: transcript must be a list of dicts."""

    def test_positive_full_pass(self) -> None:
        compute_reward = _load_compute_reward()
        sample = _sample("d1-parse-pos", expected_calls=[
            _ok_call("c1").copy(),
        ], answer="北京 22C", tools=[_tool_decl()])
        sig = compute_reward(sample, _row("d1-parse-pos",
            calls=[_ok_call("c1")], generated="北京 22C"),
            transcript_kind="mock_transcript")
        self.assertTrue(sig["layers"]["parse_success"])
        self.assertEqual(sig["reward_type"], "execution_correct")
        self.assertEqual(sig["reward_binary"], 1.0)

    def test_negative_non_list(self) -> None:
        compute_reward = _load_compute_reward()
        sample = _sample("d1-parse-neg1", expected_calls=[_ok_call("c1")],
                         tools=[_tool_decl()])
        sig = compute_reward(sample, _row("d1-parse-neg1",
            calls="not-a-list", generated=""))
        self.assertFalse(sig["layers"]["parse_success"])
        self.assertEqual(sig["reward_type"], "parse_success")
        self.assertEqual(sig["first_failure"], "parse_success")
        self.assertEqual(sig["reward_binary"], 0.0)

    def test_negative_list_of_non_dicts(self) -> None:
        compute_reward = _load_compute_reward()
        sample = _sample("d1-parse-neg2", expected_calls=[_ok_call("c1")])
        sig = compute_reward(sample, _row("d1-parse-neg2",
            calls=["raw-text", 42, None], generated=""))
        self.assertFalse(sig["layers"]["parse_success"])
        self.assertEqual(sig["reward_type"], "parse_success")

    def test_boundary_empty_list_for_no_tool(self) -> None:
        compute_reward = _load_compute_reward()
        sample = _sample("d1-parse-bnd", expected_calls=[],
                         task_type="no_tool")
        sig = compute_reward(sample, _row("d1-parse-bnd", calls=[]))
        self.assertTrue(sig["layers"]["parse_success"])
        self.assertEqual(sig["reward_binary"], 1.0)


class SchemaValidLayerTests(unittest.TestCase):
    """schema_valid: arguments must match the declared tool schema."""

    def test_positive_args_match_schema(self) -> None:
        compute_reward = _load_compute_reward()
        sample = _sample("d1-schema-pos", expected_calls=[
            _ok_call("c1").copy()
        ], answer="北京 22C", tools=[_tool_decl()])
        sig = compute_reward(sample, _row("d1-schema-pos",
            calls=[_ok_call("c1")], generated="北京 22C"),
            transcript_kind="mock_transcript")
        self.assertTrue(sig["layers"]["schema_valid"])

    def test_negative_unknown_tool_name(self) -> None:
        compute_reward = _load_compute_reward()
        sample = _sample("d1-schema-neg", expected_calls=[
            _ok_call("c1", name="d1_unknown").copy()
        ], tools=[_tool_decl()])
        sig = compute_reward(sample, _row("d1-schema-neg",
            calls=[_ok_call("c1", name="d1_unknown")]))
        self.assertIsNone(sig["layers"]["schema_valid"])

    def test_boundary_no_tool_registry(self) -> None:
        compute_reward = _load_compute_reward()
        sample = _sample("d1-schema-bnd", expected_calls=[
            _ok_call("c1").copy()
        ], answer="北京 22C", tools=[])
        sig = compute_reward(sample, _row("d1-schema-bnd",
            calls=[_ok_call("c1")], generated="北京 22C"),
            transcript_kind="mock_transcript")
        self.assertIsNone(sig["layers"]["schema_valid"])


class ToolNameCorrectLayerTests(unittest.TestCase):
    """tool_name_correct: transcript names must match expected names."""

    def test_positive_name_matches(self) -> None:
        compute_reward = _load_compute_reward()
        sample = _sample("d1-name-pos", expected_calls=[
            _ok_call("c1", name="d1_get_weather").copy()
        ], tools=[_tool_decl()], answer="北京 22C")
        sig = compute_reward(sample, _row("d1-name-pos",
            calls=[_ok_call("c1")], generated="北京 22C"),
            transcript_kind="mock_transcript")
        self.assertTrue(sig["layers"]["tool_name_correct"])

    def test_negative_wrong_name(self) -> None:
        compute_reward = _load_compute_reward()
        sample = _sample("d1-name-neg", expected_calls=[
            _ok_call("c1", name="d1_get_weather").copy()
        ], tools=[_tool_decl()])
        sig = compute_reward(sample, _row("d1-name-neg",
            calls=[_ok_call("c1", name="d1_get_news")]))
        self.assertFalse(sig["layers"]["tool_name_correct"])
        self.assertEqual(sig["first_failure"], "tool_name_correct")

    def test_boundary_extra_predicted_calls(self) -> None:
        compute_reward = _load_compute_reward()
        sample = _sample("d1-name-bnd", expected_calls=[
            _ok_call("c1").copy()
        ], tools=[_tool_decl()])
        # Two predicted calls with one extra: the classifier still
        # evaluates tool_name_correct against the position-paired expected.
        sig = compute_reward(sample, _row("d1-name-bnd",
            calls=[_ok_call("c1"), _ok_call("c2", name="d1_get_news")]))
        self.assertTrue(sig["layers"]["parse_success"])


class ArgumentValueCorrectLayerTests(unittest.TestCase):
    """argument_value_correct: pairwise argument comparison (multiset)."""

    def test_positive_args_match(self) -> None:
        compute_reward = _load_compute_reward()
        sample = _sample("d1-arg-pos", expected_calls=[
            _ok_call("c1", city="北京").copy()
        ], tools=[_tool_decl()], answer="北京 22C")
        sig = compute_reward(sample, _row("d1-arg-pos",
            calls=[_ok_call("c1", city="北京")],
            generated="北京 22C"), transcript_kind="mock_transcript")
        self.assertTrue(sig["layers"]["argument_value_correct"])

    def test_negative_wrong_city(self) -> None:
        compute_reward = _load_compute_reward()
        sample = _sample("d1-arg-neg", expected_calls=[
            _ok_call("c1", city="北京").copy()
        ], tools=[_tool_decl()])
        sig = compute_reward(sample, _row("d1-arg-neg",
            calls=[_ok_call("c1", city="上海")]))
        self.assertFalse(sig["layers"]["argument_value_correct"])
        self.assertEqual(sig["first_failure"], "argument_value_correct")
        self.assertEqual(sig["reward_type"], "argument_correct")

    def test_boundary_no_expected_calls(self) -> None:
        compute_reward = _load_compute_reward()
        sample = _sample("d1-arg-bnd", expected_calls=[], task_type="no_tool")
        sig = compute_reward(sample, _row("d1-arg-bnd", calls=[]))
        # For no_tool samples (zero expected and zero predicted calls) the
        # intermediate layers are trivially True (the classifier agrees).
        self.assertTrue(sig["layers"]["argument_value_correct"])


class CallPlanMatchesLayerTests(unittest.TestCase):
    """call_plan_matches: position order + call_id dependency check."""

    def test_positive_order_matches(self) -> None:
        compute_reward = _load_compute_reward()
        sample = _sample("d1-plan-pos", expected_calls=[
            {"call_id": "c1", "name": "d1_search",
             "arguments": {"q": "x"}, "expected_result": "y"},
            {"call_id": "c2", "name": "d1_calc",
             "arguments": {"expr": "1+1"}, "expected_result": "2",
             "depends_on": ["c1"]},
        ], tools=[_tool_decl("d1_search", prop="q", required=("q",)),
                  _tool_decl("d1_calc", prop="expr", required=("expr",))],
           answer="2")
        sig = compute_reward(sample, _row("d1-plan-pos",
            calls=[
                {"call_id": "c1", "name": "d1_search",
                 "arguments": {"q": "x"}, "expected_result": "y",
                 "execution_outcome": "success", "result": "y"},
                {"call_id": "c2", "name": "d1_calc",
                 "arguments": {"expr": "1+1"}, "expected_result": "2",
                 "execution_outcome": "success", "result": "2",
                 "depends_on": ["c1"]},
            ], generated="2"), transcript_kind="mock_transcript")
        self.assertTrue(sig["layers"]["call_plan_matches"])

    def test_negative_swapped_order(self) -> None:
        compute_reward = _load_compute_reward()
        sample = _sample("d1-plan-neg", expected_calls=[
            {"call_id": "c1", "name": "d1_search",
             "arguments": {"q": "x"}, "expected_result": "y"},
            {"call_id": "c2", "name": "d1_calc",
             "arguments": {"expr": "1+1"}, "expected_result": "2",
             "depends_on": ["c1"]},
        ], tools=[_tool_decl("d1_search", prop="q", required=("q",)),
                  _tool_decl("d1_calc", prop="expr", required=("expr",))])
        sig = compute_reward(sample, _row("d1-plan-neg",
            calls=[
                {"call_id": "c2", "name": "d1_calc",
                 "arguments": {"expr": "1+1"}, "expected_result": "2",
                 "execution_outcome": "success", "result": "2",
                 "depends_on": ["c1"]},
                {"call_id": "c1", "name": "d1_search",
                 "arguments": {"q": "x"}, "expected_result": "y",
                 "execution_outcome": "success", "result": "y"},
            ], generated="2"), transcript_kind="mock_transcript")
        self.assertFalse(sig["layers"]["call_plan_matches"])
        self.assertEqual(sig["first_failure"], "call_plan_matches")

    def test_boundary_no_expected_calls(self) -> None:
        compute_reward = _load_compute_reward()
        sample = _sample("d1-plan-bnd", expected_calls=[], task_type="no_tool")
        sig = compute_reward(sample, _row("d1-plan-bnd", calls=[]))
        # no_tool: call_plan_matches trivially True (nothing to compare).
        self.assertTrue(sig["layers"]["call_plan_matches"])


class ExecutionSuccessLayerTests(unittest.TestCase):
    """execution_success: MockExecutor reported no error per call."""

    def test_positive_success_outcome(self) -> None:
        compute_reward = _load_compute_reward()
        sample = _sample("d1-exec-pos", expected_calls=[
            _ok_call("c1").copy()
        ], tools=[_tool_decl()], answer="北京 22C")
        sig = compute_reward(sample, _row("d1-exec-pos",
            calls=[_ok_call("c1")], generated="北京 22C"),
            transcript_kind="mock_transcript")
        self.assertTrue(sig["layers"]["execution_success"])

    def test_negative_mock_exception(self) -> None:
        compute_reward = _load_compute_reward()
        sample = _sample("d1-exec-neg", expected_calls=[
            _ok_call("c1").copy()
        ], tools=[_tool_decl()])
        bad_call = _ok_call("c1")
        bad_call["execution_outcome"] = "mock_exception"
        sig = compute_reward(sample, _row("d1-exec-neg",
            calls=[bad_call]))
        self.assertFalse(sig["layers"]["execution_success"])

    def test_boundary_no_calls(self) -> None:
        compute_reward = _load_compute_reward()
        sample = _sample("d1-exec-bnd", expected_calls=[], task_type="no_tool")
        sig = compute_reward(sample, _row("d1-exec-bnd", calls=[]))
        self.assertTrue(sig["layers"]["execution_success"])


class ResultGroundedLayerTests(unittest.TestCase):
    """result_grounded: transcript result == expected_result (position pair)."""

    def test_positive_results_match(self) -> None:
        compute_reward = _load_compute_reward()
        sample = _sample("d1-res-pos", expected_calls=[
            _ok_call("c1", expected_result="北京 22C").copy()
        ], tools=[_tool_decl()], answer="北京 22C")
        sig = compute_reward(sample, _row("d1-res-pos",
            calls=[_ok_call("c1", expected_result="北京 22C")],
            generated="北京 22C"), transcript_kind="mock_transcript")
        self.assertTrue(sig["layers"]["result_grounded"])

    def test_negative_result_mismatch(self) -> None:
        compute_reward = _load_compute_reward()
        sample = _sample("d1-res-neg", expected_calls=[
            _ok_call("c1", expected_result="北京 22C").copy()
        ], tools=[_tool_decl()])
        sig = compute_reward(sample, _row("d1-res-neg",
            calls=[_ok_call("c1", expected_result="上海 25C")]))
        self.assertFalse(sig["layers"]["result_grounded"])

    def test_boundary_no_expected_result(self) -> None:
        compute_reward = _load_compute_reward()
        sample = _sample("d1-res-bnd", expected_calls=[
            {"call_id": "c1", "name": "d1_get_weather",
             "arguments": {"city": "北京"}}  # no expected_result
        ], tools=[_tool_decl()])
        sig = compute_reward(sample, _row("d1-res-bnd",
            calls=[{"call_id": "c1", "name": "d1_get_weather",
                    "arguments": {"city": "北京"},
                    "execution_outcome": "success"}]))
        self.assertIsNone(sig["layers"]["result_grounded"])


class FinalAnswerCorrectLayerTests(unittest.TestCase):
    """final_answer_correct: substring containment in generated text."""

    def test_positive_answer_substring(self) -> None:
        compute_reward = _load_compute_reward()
        sample = _sample("d1-final-pos", expected_calls=[
            _ok_call("c1").copy()
        ], tools=[_tool_decl()], answer="北京 22C")
        sig = compute_reward(sample, _row("d1-final-pos",
            calls=[_ok_call("c1")], generated="北京 22C"),
            transcript_kind="mock_transcript")
        self.assertTrue(sig["layers"]["final_answer_correct"])
        self.assertEqual(sig["reward_binary"], 1.0)

    def test_negative_wrong_answer(self) -> None:
        compute_reward = _load_compute_reward()
        sample = _sample("d1-final-neg", expected_calls=[
            _ok_call("c1").copy()
        ], tools=[_tool_decl()], answer="北京 22C")
        sig = compute_reward(sample, _row("d1-final-neg",
            calls=[_ok_call("c1")], generated="我不知道"))
        self.assertFalse(sig["layers"]["final_answer_correct"])
        self.assertEqual(sig["first_failure"], "final_answer_correct")
        self.assertEqual(sig["reward_type"], "final_answer_correct")

    def test_boundary_no_expected_answer(self) -> None:
        compute_reward = _load_compute_reward()
        sample = _sample("d1-final-bnd", expected_calls=[
            _ok_call("c1").copy()
        ], tools=[_tool_decl()])  # no expected_answer
        sig = compute_reward(sample, _row("d1-final-bnd",
            calls=[_ok_call("c1")], generated="随便"),
            transcript_kind="mock_transcript")
        self.assertIsNone(sig["layers"]["final_answer_correct"])


class ClassifierConsistencyTests(unittest.TestCase):
    """Direct comparison against ``scripts.classify_tool_failure.classify``."""

    @staticmethod
    def _classify() -> Any:
        spec = importlib.util.spec_from_file_location(
            "_p105", CLASSIFIER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.classify

    def _run_through_both(self, sample: dict, row: dict) -> tuple[dict, dict]:
        compute_reward = _load_compute_reward()
        sig = compute_reward(sample, row)
        classify = self._classify()
        transcript = {
            "tool_calls": row.get("extracted_calls", []),
            "final_answer": row.get("generated", ""),
        }
        cls = classify(sample, transcript)
        return sig, cls

    def test_first_failure_and_layers_match_full_pass(self) -> None:
        sample = _sample("d1-cs-fp", expected_calls=[
            _ok_call("c1").copy()
        ], tools=[_tool_decl()], answer="北京 22C")
        row = _row("d1-cs-fp", calls=[_ok_call("c1")], generated="北京 22C")
        sig, cls = self._run_through_both(sample, row)
        for k in ("parse_success", "schema_valid", "tool_name_correct",
                  "argument_value_correct", "call_plan_matches",
                  "execution_success", "result_grounded",
                  "final_answer_correct"):
            self.assertEqual(sig["layers"][k], cls["layers"][k],
                             f"layer {k} mismatch")
        self.assertEqual(sig["first_failure"], cls["first_failure"])

    def test_first_failure_and_layers_match_parse_fail(self) -> None:
        sample = _sample("d1-cs-pf", expected_calls=[
            _ok_call("c1").copy()
        ], tools=[_tool_decl()])
        row = _row("d1-cs-pf", calls=["raw-text", 1], generated="")
        sig, cls = self._run_through_both(sample, row)
        self.assertEqual(sig["first_failure"], cls["first_failure"])
        # Only compare the eight canonical reward layers (excluding the
        # backward-compat alias ``task_success`` emitted by classify()).
        for k in ("parse_success", "schema_valid", "tool_name_correct",
                  "argument_value_correct", "call_plan_matches",
                  "execution_success", "result_grounded",
                  "final_answer_correct"):
            self.assertEqual(sig["layers"][k], cls["layers"][k],
                             f"layer {k} mismatch")

    def test_reward_binary_equals_one_iff_no_first_failure(self) -> None:
        sample = _sample("d1-cs-bin", expected_calls=[
            _ok_call("c1").copy()
        ], tools=[_tool_decl()], answer="北京 22C")
        ok_row = _row("d1-cs-bin", calls=[_ok_call("c1")], generated="北京 22C")
        sig_ok, cls_ok = self._run_through_both(sample, ok_row)
        self.assertEqual(sig_ok["reward_binary"], 1.0)
        self.assertIsNone(cls_ok["first_failure"])
        bad_row = _row("d1-cs-bin", calls=["raw"], generated="")
        sig_bad, cls_bad = self._run_through_both(sample, bad_row)
        self.assertEqual(sig_bad["reward_binary"], 0.0)
        self.assertIsNotNone(cls_bad["first_failure"])

    def test_layered_reward_invariant(self) -> None:
        sample = _sample("d1-cs-lay", expected_calls=[
            _ok_call("c1").copy()
        ], tools=[_tool_decl()], answer="北京 22C")
        row = _row("d1-cs-lay", calls=[_ok_call("c1", city="上海")],
                   generated="上海 25C")
        sig, _ = self._run_through_both(sample, row)
        if sig["layer_pass_total"] > 0:
            self.assertAlmostEqual(
                sig["reward_layered"],
                sig["layer_pass_count"] / sig["layer_pass_total"],
                places=6,
            )


class CLIIntegrationTests(unittest.TestCase):
    """Smoke test the CLI end-to-end through subprocess."""

    def setUp(self) -> None:
        if TMP.exists():
            shutil.rmtree(TMP)
        TMP.mkdir(parents=True)

    def tearDown(self) -> None:
        if TMP.exists():
            shutil.rmtree(TMP)

    def test_cli_aggregate_writes_json(self) -> None:
        sample_dir = TMP / "samples"
        sample_dir.mkdir()
        (sample_dir / "d1-0101.json").write_text(json.dumps(_sample(
            "d1-0101", expected_calls=[_ok_call("c1")],
            answer="北京 22C", tools=[_tool_decl()]),
            ensure_ascii=False), encoding="utf-8")
        (sample_dir / "d1-0102.json").write_text(json.dumps(_sample(
            "d1-0102", expected_calls=[], task_type="no_tool"),
            ensure_ascii=False), encoding="utf-8")
        transcript_path = TMP / "transcripts.json"
        transcript_path.write_text(json.dumps({
            "summary": {"total": 2},
            "rows": [
                _row("d1-0101", calls=[_ok_call("c1")],
                     generated="北京 22C"),
                _row("d1-0102", calls=[], generated="hello"),
            ],
        }, ensure_ascii=False), encoding="utf-8")
        out_path = TMP / "rewards.json"
        result = subprocess.run(
            [sys.executable, str(REWARD),
             "--samples-dir", str(sample_dir),
             "--transcripts", str(transcript_path),
             "--output", str(out_path),
             "--checkpoint", "fake-checkpoint"],
            capture_output=True, text=True, check=True,
        )
        self.assertIn("[reward-offline] wrote", result.stdout)
        data = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(data["aggregate"]["n"], 2)
        self.assertEqual(data["aggregate"]["reward_binary_mean"], 1.0)
        self.assertEqual(data["aggregate"]["reward_layered_mean"], 1.0)
        # reward_type must be present and conform to the schema enum
        for sig in data["signals"]:
            self.assertIn("reward_type", sig)
            self.assertIn(sig["reward_type"],
                          {"parse_success", "argument_correct",
                           "final_answer_correct", "execution_correct"})


if __name__ == "__main__":
    unittest.main()