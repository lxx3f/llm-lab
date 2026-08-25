"""CLI and metadata tests for tokenizer artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/train_bpe_tokenizer.py"


class TokenizerArtifactCliTests(unittest.TestCase):
    def test_cli_creates_fixed_artifact_layout_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.txt"
            source.write_text(
                "Once upon a time, a model learned. <|endoftext|>\n",
                encoding="utf-8",
            )
            artifact_root = root / "artifacts" / "tokenizers"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(source),
                    "--artifact-root",
                    str(artifact_root),
                    "--name",
                    "tiny-bpe",
                    "--version",
                    "v0.1.0",
                    "--vocab-size",
                    "270",
                    "--special-token",
                    "<|endoftext|>",
                    "--source-path",
                    "data/owt-sample/train.txt",
                    "--data-version",
                    "D0",
                    "--license",
                    "CC0-1.0",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            artifact_dir = artifact_root / "tiny-bpe" / "v0.1.0"
            tokenizer_path = artifact_dir / "tokenizer.json"
            metadata_path = artifact_dir / "metadata.json"
            self.assertTrue(tokenizer_path.is_file())
            self.assertTrue(metadata_path.is_file())

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["metadata_version"], "1.0")
            self.assertEqual(metadata["tokenizer"]["name"], "tiny-bpe")
            self.assertEqual(metadata["tokenizer"]["version"], "v0.1.0")
            self.assertEqual(metadata["source"]["path"], "data/owt-sample/train.txt")
            self.assertEqual(metadata["source"]["data_version"], "D0")
            self.assertEqual(metadata["source"]["license"], "CC0-1.0")
            self.assertEqual(
                metadata["source"]["sha256"],
                hashlib.sha256(source.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                metadata["artifacts"]["tokenizer_sha256"],
                hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
            )

    def test_cli_refuses_non_empty_artifact_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.txt"
            source.write_text("hello hello hello", encoding="utf-8")
            artifact_root = root / "artifacts"
            command = [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(source),
                "--artifact-root",
                str(artifact_root),
                "--name",
                "tiny-bpe",
                "--version",
                "v0.1.0",
                "--vocab-size",
                "260",
            ]
            first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("not empty", second.stderr)


if __name__ == "__main__":
    unittest.main()
