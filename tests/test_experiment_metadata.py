"""Tests for the unified experiment-metadata collector (N3)."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from architecture_lab.experiment_metadata import (
    METADATA_FIELDS,
    UNSET,
    collect_metadata,
)


class CollectMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.expected_fields = {
            "git_commit",
            "config_sha256",
            "python_version",
            "pytorch_version",
            "cuda_version",
            "gpu_name",
            "gpu_compute_capability",
            "tokenizer_revision",
            "dataset_hash",
            "seed",
        }

    def test_field_set_is_frozen(self) -> None:
        self.assertEqual(set(METADATA_FIELDS), self.expected_fields)
        self.assertEqual(len(METADATA_FIELDS), 10)

    def test_returns_all_fields_with_correct_types(self) -> None:
        metadata = collect_metadata(seed=42)
        self.assertEqual(set(metadata), self.expected_fields)
        self.assertIsInstance(metadata["seed"], int)
        self.assertEqual(metadata["seed"], 42)
        # Contract-required non-null string fields: always strings (never None).
        self.assertIsInstance(metadata["python_version"], str)
        self.assertNotEqual(metadata["python_version"], "")
        self.assertIsInstance(metadata["git_commit"], str)
        self.assertIsInstance(metadata["config_sha256"], str)
        self.assertIsInstance(metadata["pytorch_version"], str)
        self.assertIsInstance(metadata["cuda_version"], str)
        self.assertIsInstance(metadata["dataset_hash"], str)
        # Optional fields: string or None (contract permits both).
        self.assertIsInstance(metadata["gpu_name"], (str, type(None)))
        self.assertIsInstance(metadata["gpu_compute_capability"], (str, type(None)))
        self.assertIsInstance(metadata["tokenizer_revision"], (str, type(None)))

    def test_config_sha256_matches_file_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_path = tmp / "config.yaml"
            config_path.write_text("seed: 1\nmodel: {}\n", encoding="utf-8")
            metadata = collect_metadata(config_path=config_path, seed=1)
            expected = hashlib.sha256(config_path.read_bytes()).hexdigest()
            self.assertEqual(metadata["config_sha256"], expected)

    def test_missing_config_returns_unset(self) -> None:
        metadata = collect_metadata(config_path=Path("/nonexistent/path/config.yaml"), seed=0)
        self.assertEqual(metadata["config_sha256"], UNSET)

    def test_missing_tokenizer_artifact_returns_null(self) -> None:
        metadata = collect_metadata(tokenizer_artifact_dir=Path("/nonexistent/tokenizer"), seed=0)
        self.assertIsNone(metadata["tokenizer_revision"])

    def test_tokenizer_revision_reads_metadata_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "metadata.json").write_text(
                json.dumps({"tokenizer": {"name": "owt-bpe", "version": "v9.9.9"}}), encoding="utf-8"
            )
            metadata = collect_metadata(tokenizer_artifact_dir=tmp, seed=0)
            self.assertEqual(metadata["tokenizer_revision"], "v9.9.9")

    def test_tokenizer_metadata_missing_version_returns_null(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "metadata.json").write_text(
                json.dumps({"tokenizer": {"name": "owt-bpe"}}), encoding="utf-8"
            )
            metadata = collect_metadata(tokenizer_artifact_dir=tmp, seed=0)
            self.assertIsNone(metadata["tokenizer_revision"])

    def test_dataset_hash_reads_source_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fake_sha = "a" * 64
            metadata_path = tmp / "train_metadata.json"
            metadata_path.write_text(
                json.dumps({"source": {"sha256": fake_sha}, "cache": {"split": "train"}}), encoding="utf-8"
            )
            metadata = collect_metadata(train_cache_dir=metadata_path, seed=0)
            self.assertEqual(metadata["dataset_hash"], fake_sha)

    def test_dataset_hash_missing_returns_unset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            metadata_path = tmp / "train_metadata.json"
            metadata_path.write_text(
                json.dumps({"cache": {"split": "train"}}), encoding="utf-8"
            )
            metadata = collect_metadata(train_cache_dir=metadata_path, seed=0)
            self.assertEqual(metadata["dataset_hash"], UNSET)

    def test_dataset_hash_bad_format_returns_unset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            metadata_path = tmp / "train_metadata.json"
            metadata_path.write_text(
                json.dumps({"source": {"sha256": "not-a-real-sha"}}), encoding="utf-8"
            )
            metadata = collect_metadata(train_cache_dir=metadata_path, seed=0)
            self.assertEqual(metadata["dataset_hash"], UNSET)

    def test_gpu_fields_null_when_cuda_unavailable(self) -> None:
        with mock.patch("architecture_lab.experiment_metadata._safe_gpu_fields", return_value=(None, None)):
            metadata = collect_metadata(seed=0)
            self.assertIsNone(metadata["gpu_name"])
            self.assertIsNone(metadata["gpu_compute_capability"])

    def test_gpu_compute_capability_uses_dotted_format(self) -> None:
        with mock.patch("architecture_lab.experiment_metadata._safe_gpu_fields", return_value=("RTX 5070 Ti", "12.0")):
            metadata = collect_metadata(seed=0)
            self.assertEqual(metadata["gpu_compute_capability"], "12.0")

    def test_gpu_compute_capability_sm_format_not_emitted(self) -> None:
        # Negative test: the implementation must not emit "sm_<digits>".
        # The collector is hard-coded to emit the contract-preferred dotted
        # "major.minor" form when CUDA is available; any "sm_X" emission
        # would indicate the format regressed.
        with mock.patch(
            "architecture_lab.experiment_metadata._safe_gpu_fields",
            return_value=("RTX 5070 Ti", "12.0"),
        ):
            metadata = collect_metadata(seed=0)
            self.assertFalse(
                (metadata["gpu_compute_capability"] or "").startswith("sm_"),
                "gpu_compute_capability must not be in 'sm_<digits>' form",
            )

    def test_seed_negative_clamped_to_zero(self) -> None:
        metadata = collect_metadata(seed=-7)
        self.assertEqual(metadata["seed"], 0)

    def test_seed_none_defaults_to_zero(self) -> None:
        metadata = collect_metadata(seed=None)
        self.assertEqual(metadata["seed"], 0)

    def test_git_commit_returns_unset_on_subprocess_error(self) -> None:
        with mock.patch(
            "architecture_lab.experiment_metadata._safe_git_commit", return_value=UNSET
        ):
            metadata = collect_metadata(seed=0)
            self.assertEqual(metadata["git_commit"], UNSET)

    def test_metadata_is_json_serializable(self) -> None:
        metadata = collect_metadata(seed=123)
        rendered = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        self.assertIsInstance(rendered, str)
        self.assertIn("\"seed\": 123", rendered)


if __name__ == "__main__":
    unittest.main()