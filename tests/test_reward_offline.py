"""Tests for the P2 offline reward evaluator (``scripts/reward_offline.py``)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REWARD = ROOT / "scripts" / "reward_offline.py"
TMP = ROOT / "artifacts" / "test-reward-offline-tmp"


def _make_sample(path: Path, *, sample_id: str, task_type: str,
                 calls: list[dict], answer: str | None = None) -> None:
    sample = {
        "schema_version": "1.0",
        "id": sample_id,
        "messages": [{"role": "user", "content": "请帮我"}],
        "tools": [],
        "expected_tool_calls": calls,
        "metadata": {
            "task_type": task_type,
            "source": "test",
            "data_version": "test",
            "pipeline_version": "test",
            "created_at": "2026-08-28T00:00:00Z",
            "validation": {"schema_valid": True},
        },
    }
    if answer is not None:
        sample["expected_answer"] = answer
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sample, ensure_ascii=False), encoding="utf-8")


def _make_transcript(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": {"total": len(rows)}, "rows": rows}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _row(sample_id: str, *, calls: list[dict] | None = None,
         generated: str = "", with_execution_outcome: bool = False) -> dict:
    """Build a transcript row.

    ``with_execution_outcome`` injects ``execution_outcome="success"`` and
    a matching ``result`` on every call so the P1-05 classifier passes the
    execution_success and result_grounded layers when the expected
    ``expected_result`` matches.
    """
    if calls is not None and with_execution_outcome:
        calls = [{
            **c,
            "execution_outcome": "success",
            "result": c.get("expected_result", c.get("arguments")),
        } for c in calls]
    return {
        "sample_id": sample_id,
        "extracted_calls": calls if calls is not None else [],
        "first_failure": None,
        "layers": {},
        "generated": generated,
    }


class RewardOfflineTests(unittest.TestCase):
    def setUp(self) -> None:
        if TMP.exists():
            shutil.rmtree(TMP)
        TMP.mkdir(parents=True)

    def tearDown(self) -> None:
        if TMP.exists():
            shutil.rmtree(TMP)

    def test_full_pass_yields_binary_one(self) -> None:
        from scripts.reward_offline import compute_reward
        sample = {
            "schema_version": "1.0",
            "id": "d1-fullpass",
            "tools": [{
                "type": "function",
                "function": {
                    "name": "d1_get_weather",
                    "parameters": {"type": "object",
                                   "properties": {"city": {"type": "string"}},
                                   "required": ["city"]},
                },
            }],
            "expected_tool_calls": [
                {"call_id": "c1", "name": "d1_get_weather",
                 "arguments": {"city": "北京"},
                 "expected_result": "ok"}
            ],
            "expected_answer": "北京 22C",
            "metadata": {"task_type": "single_tool"},
        }
        transcript_row = _row(
            "d1-fullpass",
            calls=[{"call_id": "c1", "name": "d1_get_weather",
                    "arguments": {"city": "北京"},
                    "expected_result": "ok"}],
            generated="北京 22C",
            with_execution_outcome=True,
        )
        sig = compute_reward(sample, transcript_row,
                             transcript_kind="mock_transcript")
        self.assertEqual(sig["reward_binary"], 1.0)
        self.assertEqual(sig["reward_layered"], 1.0)
        self.assertEqual(sig["first_failure"], None)
        self.assertEqual(sig["expected_calls_count"], 1)
        self.assertEqual(sig["predicted_calls_count"], 1)
        self.assertEqual(sig["layer_pass_count"], 8)
        self.assertEqual(sig["layer_pass_total"], 8)
        for v in sig["layers"].values():
            self.assertTrue(v, f"expected layer True, got {v}")

    def test_parse_fail_yields_zero_and_distribution(self) -> None:
        from scripts.reward_offline import compute_reward
        sample = {
            "id": "d1-parsefail",
            "expected_tool_calls": [
                {"call_id": "c1", "name": "d1_get_weather",
                 "arguments": {"city": "上海"}, "expected_result": "ok"}
            ],
            "metadata": {"task_type": "single_tool"},
        }
        # Missing ``name`` (None) and a string instead of dict: invalid list.
        bad_calls = [None, "raw-text"]
        sig = compute_reward(sample, _row("d1-parsefail", calls=bad_calls))
        self.assertEqual(sig["reward_binary"], 0.0)
        self.assertEqual(sig["first_failure"], "parse_success")
        self.assertEqual(sig["reward_layered"], 0.0)
        self.assertFalse(sig["layers"]["parse_success"])
        self.assertIsNone(sig["layers"]["schema_valid"])

    def test_no_tool_with_no_calls_is_binary_one(self) -> None:
        from scripts.reward_offline import compute_reward
        sample = {
            "id": "d1-notool",
            "expected_tool_calls": [],
            "metadata": {"task_type": "no_tool"},
        }
        sig = compute_reward(sample, _row("d1-notool", calls=[]))
        self.assertEqual(sig["reward_binary"], 1.0)
        self.assertEqual(sig["first_failure"], None)

    def test_layered_reward_consistent_with_first_failure(self) -> None:
        from scripts.reward_offline import compute_reward
        sample = {
            "id": "d1-partial",
            "expected_tool_calls": [
                {"call_id": "c1", "name": "d1_calculate",
                 "arguments": {"expr": "1+1"}, "expected_result": "2"}
            ],
            "metadata": {"task_type": "single_tool"},
        }
        transcript_row = _row(
            "d1-partial",
            calls=[{"call_id": "c1", "name": "d1_calculate",
                    "arguments": {"expr": "1+2"}}],
            generated="答案是 3",
        )
        sig = compute_reward(sample, transcript_row)
        self.assertLess(sig["reward_binary"], 1.0)
        self.assertGreater(sig["reward_layered"], 0.0)
        self.assertNotEqual(sig["first_failure"], None)
        self.assertGreaterEqual(sig["layer_pass_count"], 1)

    def test_cli_aggregate_writes_json_and_aggregate(self) -> None:
        sample_dir = TMP / "samples"
        sample_dir.mkdir()
        _make_sample(sample_dir / "d1-0101.json", sample_id="d1-0101",
                     task_type="single_tool", calls=[
                         {"call_id": "c1", "name": "d1_get_weather",
                          "arguments": {"city": "北京"},
                          "expected_result": "ok"}], answer="北京 22C")
        _make_sample(sample_dir / "d1-0102.json", sample_id="d1-0102",
                     task_type="no_tool", calls=[])
        transcript_path = TMP / "transcripts.json"
        _make_transcript(transcript_path, [
            _row("d1-0101", calls=[{"call_id": "c1",
                                    "name": "d1_get_weather",
                                    "arguments": {"city": "北京"},
                                    "expected_result": "ok"}],
                 generated="北京 22C",
                 with_execution_outcome=True),
            _row("d1-0102", calls=[], generated="hello"),
        ])
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
        self.assertEqual(data["aggregate"]["first_failure_distribution"],
                         {"null": 2})


if __name__ == "__main__":
    unittest.main()