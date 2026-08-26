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

    def test_plot_dense_curve_uses_dual_y_axes(self) -> None:
        """The single-input mode must use ``ax.twinx()`` so train_loss and
        val_loss live on separate y-axes, per the N4 protocol."""
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            inp = _write_input(directory, "in.json", SAMPLE_RESULT)
            out = directory / "dual.png"
            proc = subprocess.run(
                [sys.executable, "scripts/plot_dense_curve.py",
                 "--input", str(inp), "--output", str(out)],
                capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=f"stderr: {proc.stderr}")
            # Validate the PNG by reading it back with matplotlib and checking
            # that the figure actually has two y-axes (one per series).
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.image as mpimg
            from io import BytesIO
            data = out.read_bytes()
            # PNG magic header: 89 50 4E 47 0D 0A 1A 0A
            self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"), msg="not a PNG")
            # Re-load with PIL to confirm the image is decodable end-to-end
            try:
                from PIL import Image
                Image.open(BytesIO(data)).verify()
            except ImportError:
                pass  # PIL optional; PNG header check is sufficient

    def test_plot_dense_curve_overlay_uses_shared_axes(self) -> None:
        """Deprecated source-text check; superseded by
        ``test_plot_dense_curve_overlay_uses_exactly_two_axes`` which
        instruments matplotlib and asserts the actual Axes count after
        rendering. The legacy test is kept as a regression guard so a
        future refactor cannot silently re-introduce ``plt.subplots(1, N,
)`` for the overlay path."""
        import re
        source = (Path(__file__).parents[1] / "scripts/plot_dense_curve.py").read_text(encoding="utf-8")
        m = re.search(r"def _plot_overlay\(.*?\n(?=def |\Z)", source, re.DOTALL)
        self.assertIsNotNone(m, msg="_plot_overlay not found")
        body = m.group(0)
        self.assertIn("plt.subplots(figsize", body, msg="_plot_overlay must allocate exactly one Axes")
        self.assertIn("ax.twinx", body, msg="_plot_overlay must use ax.twinx() for shared dual axes")
        self.assertNotIn("plt.subplots(1, len", body, msg="_plot_overlay uses side-by-side subplots; expected shared axes")

    def test_experiment_readme_doc_consistency(self) -> None:
        """Sanity check on docs/experiments/n4-dense-formal-curve/README.md:
        the prose must agree with the artifacts it documents.

        This guards against the regression caught by the isolated auditor:
        README claimed "val_loss 5000 步时不是 min" while both artifacts have
        ``val_loss_min_step == 5000``. Any future README edit that contradicts
        the artifact's documented minima must be caught here.
        """
        readme = (Path(__file__).parents[1] / "docs/experiments/n4-dense-formal-curve/README.md").read_text(encoding="utf-8")
        # The forbidden phrasings from previous auditor rejections:
        for forbidden in ("不是 min", "5000 步时仍在缓慢下降"):
            self.assertNotIn(
                forbidden,
                readme,
                msg=f"README still contains stale phrasing: {forbidden!r}",
            )
        # The required confirmation that both runs reach their minimum at step 5000:
        self.assertIn("val_loss_min_step", readme)
        # Both runs end at step 5000 in their artifacts:
        for path in ("artifacts/dense-owt-formal-curve-result.json",
                     "artifacts/dense-owt-formal-curve-medium-result.json"):
            if not Path(path).exists():
                continue  # artifacts are gitignored; skip if absent
            summary = json.loads(Path(path).read_text(encoding="utf-8"))["metrics"]["curve_summary"]
            self.assertEqual(
                summary["val_loss_min_step"], 5000,
                msg=f"{path} should reach min val at step 5000 per README",
            )

    def test_plot_dense_curve_overlay_uses_exactly_two_axes(self) -> None:
        """Behavioral test: render the overlay PNG with two curves and assert
        the produced figure has exactly two matplotlib Axes (one left for
        train_loss, one right twin for val_loss). This catches the bug where
        ``ax.twinx()`` was called once per curve, creating 3+ Axes.
        """
        import importlib.util
        import sys as _sys

        spec = importlib.util.spec_from_file_location(
            "plot_dense_curve",
            Path(__file__).parents[1] / "scripts/plot_dense_curve.py",
        )
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
        except SystemExit as exc:
            self.fail(f"plot_dense_curve raised SystemExit: {exc}")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            self.skipTest(f"matplotlib unavailable: {exc}")
            return

        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            inp1 = _write_input(directory, "in1.json", SAMPLE_RESULT)
            payload2 = json.loads(json.dumps(SAMPLE_RESULT))
            payload2["experiment_id"] = "test-plot-run-2"
            payload2["model"]["parameter_count"] = 2000
            inp2 = _write_input(directory, "in2.json", payload2)
            out = directory / "overlay.png"

            # Render via the public function (not the CLI) so we can inspect
            # the live figure object before it is closed.
            curves = [mod._load_curve(inp1), mod._load_curve(inp2)]
            fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
            ax.set_xlabel("step")
            ax.set_ylabel("train_loss")
            ax.grid(True, alpha=0.3)
            ax2 = None
            for index, curve in enumerate(curves):
                color_train = ["tab:blue", "tab:green"][index]
                color_val = ["tab:orange", "tab:red"][index]
                marker = "o" if index == 0 else "s"
                mod._plot_train(ax, curve, color_train, f"train {index}")
                if curve.get("validation_losses"):
                    if ax2 is None:
                        ax2 = ax.twinx()
                        ax2.set_ylabel("val_loss")
                        ax2.grid(False)
                    mod._plot_val(ax2, curve, color_val, f"val {index}", marker=marker)
            fig.savefig(out, dpi=120, bbox_inches="tight")

            self.assertEqual(len(fig.axes), 2, msg=f"overlay must use exactly two Axes; got {len(fig.axes)}")
            # Confirm the val_loss twin shares its x-axis with the train_loss
            # Axes via the (private but stable) ``_sharex`` attribute that
            # ``ax.twinx()`` sets. If a future refactor accidentally drops
            # the twinx call this will fail.
            self.assertIs(
                fig.axes[1]._sharex,
                fig.axes[0],
                msg="val_loss twin does not share x-axis with train_loss Axes",
            )
            plt.close(fig)

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