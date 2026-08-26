"""D0 tool-calling manifest validation tests."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from scripts.build_d0_manifest import MANIFEST_PATH, SAMPLES_DIR, _sha256_file, build_manifest

REQUIRED_TASK_TYPES = {"no_tool", "single_tool", "multi_tool"}


class D0ManifestTests(unittest.TestCase):
    def test_build_manifest_is_valid(self) -> None:
        manifest = build_manifest()
        self.assertGreaterEqual(len(manifest["samples"]), 3)
        self.assertEqual(manifest["data_version"], "D0")
        task_types = {s["task_type"] for s in manifest["samples"]}
        self.assertTrue(REQUIRED_TASK_TYPES.issubset(task_types), msg=f"missing task types: {REQUIRED_TASK_TYPES - task_types}")

    def test_manifest_hashes_match_files(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        for sample in manifest["samples"]:
            path = SAMPLES_DIR / sample["path"]
            self.assertTrue(path.is_file(), msg=f"sample file missing: {sample['path']}")
            self.assertEqual(
                _sha256_file(path),
                sample["sha256"],
                msg=f"sha256 mismatch for {sample['path']}",
            )
        # Aggregate hash = sha256 of concatenated per-file hashes (ascii).
        import hashlib
        aggregate = hashlib.sha256("".join(s["sha256"] for s in manifest["samples"]).encode("ascii")).hexdigest()
        self.assertEqual(aggregate, manifest["samples_sha256"])

    def test_all_samples_use_synthetic_source(self) -> None:
        for sample_path in SAMPLES_DIR.glob("sample-*.json"):
            payload = json.loads(sample_path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["metadata"]["source"],
                "synthetic",
                msg=f"{sample_path.name}: source must be synthetic",
            )

    def test_cli_validate_passes(self) -> None:
        proc = subprocess.run(
            [sys.executable, "scripts/build_d0_manifest.py"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=f"stderr: {proc.stderr}")
        self.assertIn("D0 manifest valid", proc.stdout)


if __name__ == "__main__":
    unittest.main()
