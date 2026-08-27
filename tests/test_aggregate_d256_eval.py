"""Tests for scripts/aggregate_d256_eval.py aggregator.

The aggregator uses ``Path(__file__).resolve().parents[1].glob(...)``,
so the test fake eval files must live under the project root. We use
``artifacts/test-aggregate-tmp/`` and clean up after ourselves (the
directory is gitignored, see ``.gitignore``).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGGREGATOR = ROOT / "scripts" / "aggregate_d256_eval.py"
TMP = ROOT / "artifacts" / "test-aggregate-tmp"


def _make_eval(path: Path, *, total: int, parse_success_rate: float,
               first_failure_distribution: dict[str, int]) -> None:
    no_failure = total - sum(first_failure_distribution.values())
    summary = {
        "checkpoint": "fake.pt",
        "samples_dir": "fake",
        "total": total,
        "first_failure_distribution": first_failure_distribution,
        "no_failure": no_failure,
        "parse_success_rate": parse_success_rate,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": summary, "rows": []}, ensure_ascii=False),
                    encoding="utf-8")


class AggregateD256EvalTests(unittest.TestCase):
    def setUp(self) -> None:
        if TMP.exists():
            shutil.rmtree(TMP)
        TMP.mkdir(parents=True)

    def tearDown(self) -> None:
        if TMP.exists():
            shutil.rmtree(TMP)

    def test_aggregator_handles_zero_parse_rate(self) -> None:
        for seed in (42, 123, 7):
            _make_eval(TMP / f"seed{seed}.json", total=13,
                       parse_success_rate=0.0,
                       first_failure_distribution={"parse_success": 13})
        out = TMP / "agg.json"
        result = subprocess.run(
            [sys.executable, str(AGGREGATOR),
             "--eval-glob", str(TMP / "seed*.json"),
             "--output", str(out),
             "--label", "test"],
            capture_output=True, text=True, check=True,
        )
        self.assertIn("[aggregate] wrote", result.stdout)
        data = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(data["n_seeds"], 3)
        self.assertEqual(data["total_evaluations"], 39)
        self.assertEqual(data["aggregate_metrics"]["parse_success_rate"]["mean"], 0.0)
        self.assertEqual(data["aggregate_metrics"]["parse_success_rate"]["pop_std"], 0.0)
        self.assertEqual(
            data["aggregate_metrics"]["first_failure_layer_distribution_count"],
            {"parse_success": 39},
        )

    def test_aggregator_with_nonzero_parse_rate(self) -> None:
        _make_eval(TMP / "s1.json", total=10, parse_success_rate=0.2,
                   first_failure_distribution={"parse_success": 8})
        _make_eval(TMP / "s2.json", total=10, parse_success_rate=0.6,
                   first_failure_distribution={"parse_success": 4})
        _make_eval(TMP / "s3.json", total=10, parse_success_rate=0.4,
                   first_failure_distribution={"parse_success": 6})
        out = TMP / "agg.json"
        subprocess.run(
            [sys.executable, str(AGGREGATOR),
             "--eval-glob", str(TMP / "s*.json"),
             "--output", str(out),
             "--label", "test"],
            capture_output=True, text=True, check=True,
        )
        data = json.loads(out.read_text(encoding="utf-8"))
        self.assertAlmostEqual(
            data["aggregate_metrics"]["parse_success_rate"]["mean"],
            0.4, places=4,
        )
        # pop std of [0.2, 0.4, 0.6] = sqrt((0.04 + 0 + 0.04)/3) ≈ 0.1633
        self.assertAlmostEqual(
            data["aggregate_metrics"]["parse_success_rate"]["pop_std"],
            0.1633, places=3,
        )


if __name__ == "__main__":
    unittest.main()