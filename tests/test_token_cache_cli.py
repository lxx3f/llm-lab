"""CLI tests for streaming token cache encoding."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/encode_token_cache.py"
TOKENIZER_SCRIPT = ROOT / "scripts/train_bpe_tokenizer.py"


class TokenCacheCliTests(unittest.TestCase):
    def test_cli_encodes_cache_and_records_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.txt"
            source.write_bytes(b"alpha alpha\nbeta beta\n")
            tokenizer_root = root / "artifacts"
            tokenizer_result = subprocess.run(
                [
                    sys.executable,
                    str(TOKENIZER_SCRIPT),
                    "--input",
                    str(source),
                    "--artifact-root",
                    str(tokenizer_root),
                    "--name",
                    "cache-bpe",
                    "--version",
                    "v0.1.0",
                    "--vocab-size",
                    "270",
                    "--special-token",
                    "<|endoftext|>",
                    "--data-version",
                    "TEST-v1",
                    "--license",
                    "test-terms",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(tokenizer_result.returncode, 0, tokenizer_result.stderr)
            tokenizer = tokenizer_root / "cache-bpe" / "v0.1.0" / "tokenizer.json"
            output_root = root / "cache"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(source),
                    "--tokenizer",
                    str(tokenizer),
                    "--output-root",
                    str(output_root),
                    "--split",
                    "train",
                    "--max-bytes",
                    "12",
                    "--chunk-tokens",
                    "2",
                    "--data-version",
                    "TEST-v1",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads(
                (output_root / "train.metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["cache"]["split"], "train")
            self.assertEqual(metadata["source"]["requested_max_bytes"], 12)
            self.assertEqual(metadata["data_version"], "TEST-v1")
            self.assertTrue((output_root / "train.tokens.uint16").is_file())

    def test_cli_uses_default_data_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.txt"
            source.write_bytes(b"alpha alpha\n")
            tokenizer_root = root / "artifacts"
            tokenizer_result = subprocess.run(
                [
                    sys.executable,
                    str(TOKENIZER_SCRIPT),
                    "--input", str(source),
                    "--artifact-root", str(tokenizer_root),
                    "--name", "cache-bpe", "--version", "v0.1.0",
                    "--vocab-size", "270",
                ], cwd=ROOT, capture_output=True, text=True,
            )
            self.assertEqual(tokenizer_result.returncode, 0, tokenizer_result.stderr)
            tokenizer = tokenizer_root / "cache-bpe" / "v0.1.0" / "tokenizer.json"
            result = subprocess.run(
                [
                    sys.executable, str(SCRIPT),
                    "--input", str(source), "--tokenizer", str(tokenizer),
                    "--output-root", str(root / "cache"), "--split", "train",
                ], cwd=ROOT, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads((root / "cache" / "train.metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["data_version"], "OWT-SAMPLE-v1")

    def test_cli_refuses_existing_cache_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.txt"
            source.write_text("alpha alpha\n", encoding="utf-8")
            tokenizer_root = root / "artifacts"
            tokenizer_result = subprocess.run(
                [
                    sys.executable,
                    str(TOKENIZER_SCRIPT),
                    "--input",
                    str(source),
                    "--artifact-root",
                    str(tokenizer_root),
                    "--name",
                    "cache-bpe",
                    "--version",
                    "v0.1.0",
                    "--vocab-size",
                    "270",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(tokenizer_result.returncode, 0, tokenizer_result.stderr)
            tokenizer = tokenizer_root / "cache-bpe" / "v0.1.0" / "tokenizer.json"
            command = [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(source),
                "--tokenizer",
                str(tokenizer),
                "--output-root",
                str(root / "cache"),
                "--split",
                "train",
            ]
            first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("already exist", second.stderr)


if __name__ == "__main__":
    unittest.main()
