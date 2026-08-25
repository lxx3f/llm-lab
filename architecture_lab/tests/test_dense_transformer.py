"""Correctness tests for the Dense Transformer baseline."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture_lab.models.dense_transformer import (  # noqa: E402
    DenseTransformer,
    TransformerConfig,
    count_parameters,
)


class DenseTransformerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.config = TransformerConfig(
            vocab_size=32,
            max_seq_len=16,
            d_model=32,
            n_heads=4,
            n_layers=2,
            d_ff=64,
            dropout=0.0,
        )
        self.model = DenseTransformer(self.config).eval()

    def test_forward_shape_and_loss(self) -> None:
        input_ids = torch.randint(0, self.config.vocab_size, (2, 6))
        logits, loss = self.model(input_ids, labels=input_ids)
        self.assertEqual(logits.shape, (2, 6, self.config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss).item())

    def test_generate_respects_max_sequence_length(self) -> None:
        input_ids = torch.randint(0, self.config.vocab_size, (2, 4))
        generated = self.model.generate(input_ids, max_new_tokens=32)
        self.assertEqual(generated.shape, (2, self.config.max_seq_len))

    def test_full_forward_matches_incremental_decode(self) -> None:
        input_ids = torch.randint(0, self.config.vocab_size, (2, 8))
        full_logits, _ = self.model(input_ids)
        cache = [dict() for _ in range(self.config.n_layers)]
        pieces = []
        for position in range(input_ids.size(1)):
            logits, _ = self.model(
                input_ids[:, position : position + 1],
                start_pos=position,
                kv_cache=cache,
            )
            pieces.append(logits)
        incremental_logits = torch.cat(pieces, dim=1)
        torch.testing.assert_close(full_logits, incremental_logits, rtol=1e-5, atol=1e-5)

    def test_cache_shapes_and_length(self) -> None:
        input_ids = torch.randint(0, self.config.vocab_size, (2, 5))
        cache = [dict() for _ in range(self.config.n_layers)]
        self.model(input_ids, kv_cache=cache)
        for layer_cache in cache:
            self.assertEqual(layer_cache["k"].shape, (2, 4, 5, 8))
            self.assertEqual(layer_cache["v"].shape, (2, 4, 5, 8))
        self.model(input_ids[:, :1], start_pos=5, kv_cache=cache)
        for layer_cache in cache:
            self.assertEqual(layer_cache["k"].shape[-2], 6)

    def test_backward_produces_gradients(self) -> None:
        input_ids = torch.randint(0, self.config.vocab_size, (2, 6))
        _, loss = self.model(input_ids, labels=input_ids)
        assert loss is not None
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in self.model.parameters()))

    def test_tied_embedding_and_lm_head(self) -> None:
        self.assertIs(self.model.token_embedding.weight, self.model.lm_head.weight)
        untied_count = count_parameters(self.model)
        self.assertGreater(untied_count, 0)

    def test_invalid_config_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TransformerConfig(d_model=30, n_heads=4)


if __name__ == "__main__":
    unittest.main()
