"""Correctness tests for the Dense training MVP."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from architecture_lab.data.token_cache import encode_token_cache
from architecture_lab.models.dense_transformer import DenseTransformer
from architecture_lab.tokenization import BPETokenizer, train_bpe
from architecture_lab.training.dense_training import (
    causal_loss,
    load_settings,
    train,
)


class DenseTrainingTests(unittest.TestCase):
    def _settings(self, root: Path) -> dict:
        text = ("alpha beta gamma delta epsilon zeta eta theta\n" * 8)
        source = root / "source.txt"
        source.write_bytes(text.encode("utf-8"))
        vocab, merges = train_bpe(text, vocab_size=280, special_tokens=("<|endoftext|>",))
        tokenizer_path = root / "tokenizer.json"
        BPETokenizer(vocab, merges, ("<|endoftext|>",)).save(tokenizer_path)
        train_info = encode_token_cache(
            input_path=source,
            tokenizer_path=tokenizer_path,
            output_root=root / "cache",
            split="train",
            chunk_tokens=4,
            force=True,
        )
        validation_info = encode_token_cache(
            input_path=source,
            tokenizer_path=tokenizer_path,
            output_root=root / "cache",
            split="validation",
            chunk_tokens=4,
            force=True,
        )
        return {
            "model": {
                "name": "test-dense",
                "max_seq_len": 8,
                "d_model": 16,
                "n_heads": 4,
                "n_layers": 1,
                "d_ff": 32,
                "dropout": 0.0,
            },
            "data": {
                "tokenizer": str(tokenizer_path),
                "train_tokens": str(train_info.token_path),
                "train_metadata": str(train_info.metadata_path),
                "train_source": str(source),
                "train_max_bytes": None,
                "validation_tokens": str(validation_info.token_path),
                "validation_metadata": str(validation_info.metadata_path),
                "validation_source": str(source),
                "validation_max_bytes": None,
            },
            "training": {
                "seed": 7,
                "device": "cpu",
                "dtype": "float32",
                "batch_size": 2,
                "sequence_length": 8,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "max_steps": 2,
                "log_interval": 100,
                "validation_interval": 1,
                "validation_batches": 1,
                "checkpoint": str(root / "checkpoint.pt"),
                "prompt": "alpha",
                "max_new_tokens": 2,
            },
        }

    def test_causal_loss_uses_every_explicit_target_position(self) -> None:
        logits = torch.zeros(1, 3, 5)
        targets = torch.tensor([[1, 2, 3]])
        expected = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 5), targets.reshape(-1)
        )
        self.assertEqual(float(causal_loss(logits, targets)), float(expected))

    def test_train_checkpoint_resume_and_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = self._settings(root)
            result = train(settings)
            self.assertEqual(result["step"], 2)
            self.assertTrue(Path(result["checkpoint"]).is_file())
            self.assertIsInstance(result["generated_text"], str)

            resumed_settings = settings.copy()
            resumed_settings["training"] = settings["training"].copy()
            resumed_settings["training"]["max_steps"] = 3
            resumed = train(resumed_settings, resume=result["checkpoint"])
            self.assertEqual(resumed["step"], 3)

    def test_checkpoint_rejects_cache_metadata_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = self._settings(root)
            result = train(settings)
            metadata = Path(settings["data"]["train_metadata"])
            metadata.write_text(metadata.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                train(settings, resume=result["checkpoint"])
    def test_model_vocab_size_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = self._settings(root)
            settings["model"]["vocab_size"] = 999
            with self.assertRaises(ValueError):
                train(settings)


if __name__ == "__main__":
    unittest.main()
