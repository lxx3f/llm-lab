"""Smoke tests for scripts/plot_dense_curve.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


SAMPLE_RESULT: dict = {
    "experiment_id": "test-plot-run",
    "schema_version": "1.1",
    "result_type": "dense_training",
    "timestamp": "2026-08-26T00:00:00Z",
    "status": "completed",
    "metadata": {
        "git_commit": "0" * 40,
        "config_sha256": "0" * 64,
        "python_version": "3.11.0",
        "pytorch_version": "2.1.0",
        "cuda_version": "unset",
        "gpu_name": None,
        "gpu_compute_capability": None,
        "tokenizer_revision": "v0.0.0",
        "dataset_hash": "0" * 64,
        "seed": 42,
    },
    "model": {
        "name": "t",
        "architecture": "DenseTransformer",
        "config": {"d_model": 64, "n_layers": 2, "d_ff": 256, "vocab_size": 280, "max_seq_len": 8, "n_heads": 4, "dropout": 0.0, "rope_base": 10000.0},
        "parameter_count": 1000,
    },
    "data": {
        "data_version": None,
        "tokenizer": {"path": "x", "artifact_sha256": "0" * 64, "vocab_size": 280},
        "train_cache": {"split": "train", "token_count": 1000, "encoded_bytes": 2000, "metadata_sha256": "0" * 64, "token_file_sha256": "0" * 64},
        "validation_cache": {"split": "validation", "token_count": 100, "encoded_bytes": 200, "metadata_sha256": "0" * 64, "token_file_sha256": "0" * 64},
    },
    "training": {
        "seed": 42, "device": "cpu", "model_dtype": "float32", "optimizer_steps": 3, "epoch": 0,
        "micro_batch_size": 2, "gradient_accumulation_steps": 1, "effective_batch_size": 2, "sequence_length": 4,
        "optimizer": {"name": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0, "gradient_clip_norm": 1.0},
        "scheduler": {"name": "warmup_cosine", "warmup_steps": 0, "total_steps": 3, "min_lr_ratio": 0.0},
        "amp": {"enabled": False, "dtype": "float16", "scaler_enabled": False},
    },
    "metrics": {
        "last_train_loss": 5.0,
        "validation_losses": {"100": 6.0, "200": 5.5, "300": 5.2},
        "train_losses": [
            {"step": 50, "loss": 7.0, "lr": 0.0003},
            {"step": 100, "loss": 6.5, "lr": 0.0002},
            {"step": 150, "loss": 6.0, "lr": 0.0001},
        ],
        "curve_summary": {
            "train_loss_first": 7.0, "train_loss_last": 6.0,
            "val_loss_min": 5.2, "val_loss_min_step": 300, "val_loss_last": 5.2,
            "delta_train_loss": -1.0, "delta_val_loss": 0.0,
            "train_loss_sample_count": 3, "val_loss_count": 3,
        },
    },
    "artifacts": {"checkpoint_path": "x", "checkpoint_sha256": None},
    "generation": {"prompt": "", "max_new_tokens": 0, "text": None},
}


def _write_input(directory: Path, name: str, payload: dict) -> Path:
    path = directory / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _validate_sample_against_schema(payload: dict) -> None:
    """Helper to ensure the test sample is schema-valid (so it tests the plot
    script with realistic data, not garbage)."""
    schema = json.loads((Path(__file__).parents[1] / "schemas/dense_training_result.schema.json").read_text(encoding="utf-8"))
    errors = list(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(payload))
    if errors:
        raise AssertionError(f"sample is not schema-valid: {errors}")


class PlotDenseCurveTests(unittest.TestCase):
    def test_sample_passes_schema(self) -> None:
        # Smoke check that the test fixture itself is realistic
        _validate_sample_against_schema(SAMPLE_RESULT)

    def test_plot_dense_curve_single_input(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            inp = _write_input(directory, "in.json", SAMPLE_RESULT)
            out = directory / "out.png"
            proc = subprocess.run(
                [sys.executable, "scripts/plot_dense_curve.py",
                 "--input", str(inp), "--output", str(out)],
                capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=f"stderr: {proc.stderr}")
            self.assertTrue(out.exists(), msg="PNG was not created")
            self.assertGreater(out.stat().st_size, 5000, msg="PNG suspiciously small")

    def test_plot_dense_curve_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            inp1 = _write_input(directory, "in1.json", SAMPLE_RESULT)
            payload2 = json.loads(json.dumps(SAMPLE_RESULT))
            payload2["experiment_id"] = "test-plot-run-2"
            payload2["model"]["parameter_count"] = 2000
            inp2 = _write_input(directory, "in2.json", payload2)
            out = directory / "overlay.png"
            proc = subprocess.run(
                [sys.executable, "scripts/plot_dense_curve.py",
                 "--input", str(inp1), "--input", str(inp2),
                 "--output", str(out), "--overlay"],
                capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=f"stderr: {proc.stderr}")
            self.assertTrue(out.exists(), msg="PNG was not created")
            self.assertGreater(out.stat().st_size, 5000, msg="PNG suspiciously small")

    def test_plot_dense_curve_missing_input_errors(self) -> None:
        proc = subprocess.run(
            [sys.executable, "scripts/plot_dense_curve.py"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(proc.returncode, 0, msg="missing --input should fail")


class CurveSummaryUnitTests(unittest.TestCase):
    """Tiny unit tests for ``compute_curve_summary`` that live next to the
    plot tests so the metric-plot pair stays in one place."""

    def test_compute_curve_summary_empty(self) -> None:
        from architecture_lab.training.dense_training import compute_curve_summary
        result = compute_curve_summary([], {})
        self.assertEqual(result["train_loss_sample_count"], 0)
        self.assertEqual(result["val_loss_count"], 0)
        self.assertIsNone(result["train_loss_first"])
        self.assertIsNone(result["val_loss_min"])

    def test_compute_curve_summary_typical(self) -> None:
        from architecture_lab.training.dense_training import compute_curve_summary
        train = [
            {"step": 1, "loss": 10.0, "lr": 0.001},
            {"step": 2, "loss": 8.0, "lr": 0.0008},
            {"step": 3, "loss": 6.0, "lr": 0.0005},
        ]
        val = {"100": 7.0, "200": 5.5, "300": 6.0}
        result = compute_curve_summary(train, val)
        self.assertEqual(result["train_loss_first"], 10.0)
        self.assertEqual(result["train_loss_last"], 6.0)
        self.assertEqual(result["delta_train_loss"], -4.0)
        self.assertEqual(result["val_loss_min"], 5.5)
        self.assertEqual(result["val_loss_min_step"], 200)
        self.assertEqual(result["val_loss_last"], 6.0)
        self.assertEqual(result["delta_val_loss"], 0.5)
        self.assertEqual(result["train_loss_sample_count"], 3)
        self.assertEqual(result["val_loss_count"], 3)


if __name__ == "__main__":
    unittest.main()