"""Sweep doc/artifact drift regression tests."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SweepDocConsistencyTests(unittest.TestCase):
    def test_verify_script_passes(self) -> None:
        """The verify_sweep_doc_consistency script must report zero drift."""
        proc = subprocess.run(
            [sys.executable, "scripts/verify_sweep_doc_consistency.py"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        self.assertEqual(proc.returncode, 0, msg=f"stderr: {proc.stderr}")
        self.assertIn("verified", proc.stdout)

    def test_n9_dff256_readme_row_matches_artifact(self) -> None:
        """Regression for the N9 d_ff=256 doc/artifact drift.

        The N9 README's d_ff=256 row must report ``val_min=7.5573 @ step 3200``
        and a non-zero delta_val_loss, matching the artifact's curve_summary.
        """
        artifact = json.loads(
            (ROOT / "artifacts/dense-owt-formal-curve-medium-dff-256-result.json").read_text(encoding="utf-8")
        )
        summary = artifact["metrics"]["curve_summary"]
        self.assertEqual(summary["val_loss_min_step"], 3200)
        self.assertNotEqual(summary["delta_val_loss"], 0.0)

        # N9 README may live under docs/experiments/ or docs/archive/sweep-experiments/
        for rel in (
            "docs/experiments/n9-dense-dff-sweep/README.md",
            "docs/archive/sweep-experiments/n9-dense-dff-sweep/README.md",
        ):
            path = ROOT / rel
            if path.exists():
                readme = path.read_text(encoding="utf-8")
                break
        else:
            self.fail("N9 README not found in docs/experiments/ or docs/archive/sweep-experiments/")
        dff_256_row = next(
            (line for line in readme.splitlines() if line.startswith("| **256**")),
            None,
        )
        self.assertIsNotNone(dff_256_row, msg="d_ff=256 row not found in N9 README")
        self.assertRegex(dff_256_row, r"7\.5573\b\**\s*@\s*\**\s*3200\b")
        self.assertRegex(dff_256_row, r"0\.0348")
        self.assertNotRegex(dff_256_row, r"\|\s*0\.0\s*\|")

    def test_n9_review_row_matches_artifact(self) -> None:
        """The N9 stage review must also record d_ff=256's actual min step."""
        artifact = json.loads(
            (ROOT / "artifacts/dense-owt-formal-curve-medium-dff-256-result.json").read_text(encoding="utf-8")
        )
        summary = artifact["metrics"]["curve_summary"]
        # N9 stage review may live under docs/plans/reviews/ or docs/plans/reviews/archive/
        for rel in (
            "docs/plans/reviews/stage-n9-dense-dff-sweep.md",
            "docs/plans/reviews/archive/stage-n9-dense-dff-sweep.md",
        ):
            path = ROOT / rel
            if path.exists():
                review = path.read_text(encoding="utf-8")
                break
        else:
            self.fail("N9 stage review not found in active or archive")
        self.assertIn(f"{summary['val_loss_min']:.4f}", review)
        self.assertRegex(review, r"7\.5573\b\**\s*@\s*\**\s*3200\b")
        self.assertRegex(review, r"step\s+3200")


if __name__ == "__main__":
    unittest.main()
