"""Schema and result-record tests for Dense training experiments."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

import tests.test_dense_training as dense_training_tests
from architecture_lab.experiment_metadata import collect_metadata
from architecture_lab.training.results import build_training_result, write_training_result


def _inject_metadata(result: dict, settings: dict, config_path=None) -> None:
    """Inject the unified metadata block the same way the CLIs do, so the
    result passes schema validation without modifying
    ``build_training_result`` (which is outside the N3 allowlist)."""
    result["metadata"] = collect_metadata(
        config_path=config_path,
        tokenizer_artifact_dir=Path(settings["data"]["tokenizer"]).parent,
        train_cache_dir=Path(settings["data"]["train_metadata"]),
        seed=int(settings["training"].get("seed", 42)),
    )


class DenseResultSchemaTests(unittest.TestCase):
    def test_synthetic_result_validates_and_roundtrips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = dense_training_tests.DenseTrainingTests()
            settings = helper._settings(root)
            from architecture_lab.models.dense_transformer import DenseTransformer, TransformerConfig
            import torch
            from architecture_lab.tokenization import BPETokenizer
            model = DenseTransformer(TransformerConfig(vocab_size=280, max_seq_len=8, d_model=16, n_heads=4, n_layers=1, d_ff=32))
            tokenizer_path = Path(settings["data"]["tokenizer"])
            result = build_training_result(
                settings=settings,
                model=model,
                tokenizer_path=tokenizer_path,
                train_token_path=Path(settings["data"]["train_tokens"]),
                train_metadata_path=Path(settings["data"]["train_metadata"]),
                validation_token_path=Path(settings["data"]["validation_tokens"]),
                validation_metadata_path=Path(settings["data"]["validation_metadata"]),
                device=torch.device("cpu"),
                model_dtype=torch.float32,
                optimizer_steps=2,
                epoch=0,
                last_train_loss=1.0,
                validation_losses={"1": 1.2, "2": 1.1},
                checkpoint_path=root / "missing.pt",
                prompt="alpha",
                max_new_tokens=2,
                generated_text="alpha beta",
                gradient_accumulation_steps=2,
                scheduler_config={"name": "warmup_cosine", "warmup_steps": 1, "total_steps": 2, "min_lr_ratio": 0.1},
                amp_config={"enabled": True, "dtype": "bfloat16", "scaler_enabled": False},
            )
            _inject_metadata(result, settings)
            schema = json.loads((Path(__file__).parents[1] / "schemas/dense_training_result.schema.json").read_text(encoding="utf-8"))
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
            self.assertEqual(result["metadata"]["seed"], 7)
            output = root / "result.json"
            write_training_result(result, output)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8"))["schema_version"], "1.0"
            )

    def test_invalid_result_is_rejected_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = dense_training_tests.DenseTrainingTests()
            settings = helper._settings(root)
            from architecture_lab.models.dense_transformer import DenseTransformer, TransformerConfig
            import torch
            result = build_training_result(
                settings=settings,
                model=DenseTransformer(TransformerConfig(vocab_size=280, max_seq_len=8, d_model=16, n_heads=4, n_layers=1, d_ff=32)),
                tokenizer_path=Path(settings["data"]["tokenizer"]),
                train_token_path=Path(settings["data"]["train_tokens"]),
                train_metadata_path=Path(settings["data"]["train_metadata"]),
                validation_token_path=Path(settings["data"]["validation_tokens"]),
                validation_metadata_path=Path(settings["data"]["validation_metadata"]),
                device=torch.device("cpu"), model_dtype=torch.float32,
                optimizer_steps=1, epoch=0, last_train_loss=1.0,
                validation_losses={"1": 1.0}, checkpoint_path=root / "missing.pt",
                prompt="", max_new_tokens=0, generated_text=None,
                gradient_accumulation_steps=1,
                scheduler_config={"name": "warmup_cosine", "warmup_steps": 0, "total_steps": 1, "min_lr_ratio": 0.0},
                amp_config={"enabled": False, "dtype": "float16", "scaler_enabled": False},
            )
            _inject_metadata(result, settings)
            result["metrics"].pop("validation_losses")
            with self.assertRaises(ValueError):
                write_training_result(result, root / "invalid.json")


if __name__ == "__main__":
    unittest.main()