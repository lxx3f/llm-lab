"""MoE training and result schema tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from jsonschema import Draft202012Validator, FormatChecker

from tests.test_dense_training import DenseTrainingTests
from architecture_lab.experiment_metadata import collect_metadata
from architecture_lab.models.moe_transformer import MoETransformer, count_active_parameters, count_moe_parameters
from architecture_lab.models.dense_transformer import TransformerConfig
from architecture_lab.tokenization import BPETokenizer
from architecture_lab.training.moe_results import build_moe_training_result, write_moe_training_result
from architecture_lab.training.moe_training import train


class MoETrainingTests(unittest.TestCase):
    def _settings(self, root: Path) -> dict:
        settings = DenseTrainingTests()._settings(root)
        settings["model"].update({"name": "test-moe", "d_ff": 16, "architecture": "MoETransformer"})
        settings["moe"] = {"num_experts": 4, "capacity_factor": 1.0, "aux_loss_weight": 0.01}
        settings["training"].update({"max_steps": 2, "checkpoint": str(root / "moe-checkpoint.pt"), "validation_interval": 1})
        return settings

    def test_training_schema_and_generation_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = self._settings(root)
            result = train(settings)
            # Inject the unified metadata block the same way the CLIs do, so the
            # result passes schema validation without modifying
            # ``build_moe_training_result`` (which is outside the N3 allowlist).
            result["metadata"] = collect_metadata(
                config_path=None,
                tokenizer_artifact_dir=Path(settings["data"]["tokenizer"]).parent,
                train_cache_dir=Path(settings["data"]["train_metadata"]),
                seed=int(settings["training"].get("seed", 42)),
            )
            schema = json.loads((Path(__file__).parents[1] / "schemas/moe_training_result.schema.json").read_text(encoding="utf-8"))
            errors = list(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(result))
            self.assertEqual(errors, [])
            self.assertIn("metadata", result)
            self.assertEqual(
                set(result["metadata"]),
                {
                    "git_commit", "config_sha256", "python_version", "pytorch_version",
                    "cuda_version", "gpu_name", "gpu_compute_capability",
                    "tokenizer_revision", "dataset_hash", "seed",
                },
            )
            self.assertEqual(result["training"]["collect_stats"], False)
            self.assertEqual(result["training"]["expert_capacity"], 4)
            self.assertIn("expert_capacity_definition", result["training"])
            self.assertEqual(result["generation"]["prefill_capacity_factor"], 1.0)
            self.assertEqual(result["generation"]["prefill_expert_capacity"], 1)
            self.assertEqual(result["generation"]["decode_capacity_factor"], 2.0)
            self.assertEqual(result["generation"]["decode_expert_capacity"], 1)
            # v1.1 curve fields must be present and internally consistent.
            self.assertEqual(result["schema_version"], "1.1")
            self.assertIn("train_losses", result["metrics"])
            self.assertIn("curve_summary", result["metrics"])
            summary = result["metrics"]["curve_summary"]
            self.assertGreaterEqual(summary["train_loss_sample_count"], 1)
            self.assertGreaterEqual(summary["val_loss_count"], 1)
            self.assertIsNotNone(summary["val_loss_min"])
            output = root / "moe-result.json"
            write_moe_training_result(result, output)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["result_type"], "moe_training")

    def test_checkpoint_resume_and_config_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = self._settings(root)
            result = train(settings)
            resumed = {
                **settings,
                "model": settings["model"].copy(),
                "data": settings["data"].copy(),
                "moe": settings["moe"].copy(),
                "training": settings["training"].copy(),
            }
            resumed["training"]["max_steps"] = 3
            resumed_result = train(resumed, resume=result["artifacts"]["checkpoint_path"])
            self.assertEqual(resumed_result["training"]["optimizer_steps"], 3)
            changed = {
                **settings,
                "model": settings["model"].copy(),
                "data": settings["data"].copy(),
                "moe": settings["moe"].copy(),
                "training": settings["training"].copy(),
            }
            changed["moe"]["aux_loss_weight"] = 0.02
            with self.assertRaises(ValueError):
                train(changed, resume=result["artifacts"]["checkpoint_path"])

    def test_active_parameter_count_is_less_than_total(self) -> None:
        config = TransformerConfig(vocab_size=32, max_seq_len=8, d_model=16, n_heads=4, n_layers=2, d_ff=16)
        from architecture_lab.models.moe_transformer import MoEConfig
        model = MoETransformer(config, MoEConfig(num_experts=4))
        self.assertLess(count_active_parameters(model), count_moe_parameters(model))

    def test_invalid_result_is_rejected_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = self._settings(root)
            result = train(settings)
            result["metadata"] = collect_metadata(
                config_path=None,
                tokenizer_artifact_dir=Path(settings["data"]["tokenizer"]).parent,
                train_cache_dir=Path(settings["data"]["train_metadata"]),
                seed=int(settings["training"].get("seed", 42)),
            )
            result["generation"]["decode_capacity_factor"] = 1.0
            with self.assertRaises(ValueError):
                write_moe_training_result(result, root / "invalid.json")


if __name__ == "__main__":
    unittest.main()
