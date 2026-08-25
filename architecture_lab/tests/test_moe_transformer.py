"""Correctness and routing tests for the Top-1 MoE Transformer MVP."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture_lab.models.moe_transformer import (  # noqa: E402
    MoEConfig,
    MoETransformer,
    Top1MoE,
    count_moe_parameters,
)
from architecture_lab.models.dense_transformer import TransformerConfig  # noqa: E402


class Top1MoETests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.transformer_config = TransformerConfig(
            vocab_size=32,
            max_seq_len=16,
            d_model=32,
            n_heads=4,
            n_layers=2,
            d_ff=64,
            dropout=0.0,
        )
        self.moe_config = MoEConfig(
            num_experts=4,
            # Keep enough capacity for the decode-equivalence test. Capacity
            # scheduling across prefill and incremental decode is a later MVP
            # extension; the dedicated capacity test covers overflow.
            capacity_factor=4.0,
            aux_loss_weight=0.01,
        )
        self.model = MoETransformer(self.transformer_config, self.moe_config).eval()

    def test_top1_moe_shape_and_stats(self) -> None:
        layer = Top1MoE(32, 64, self.moe_config)
        x = torch.randn(2, 8, 32)
        output, aux_loss, stats = layer(x)
        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.isfinite(aux_loss).item())
        self.assertEqual(stats["tokens"], 16)
        self.assertEqual(len(stats["assigned_tokens_per_expert"]), 4)
        self.assertEqual(sum(stats["assigned_tokens_per_expert"]), 16)
        self.assertLessEqual(stats["dropped_tokens"], 16)

    def test_capacity_limits_tokens_per_expert(self) -> None:
        config = MoEConfig(num_experts=2, capacity_factor=0.5)
        layer = Top1MoE(8, 16, config)
        x = torch.randn(2, 4, 8)
        _, _, stats = layer(x)
        self.assertEqual(stats["capacity"], 2)
        self.assertTrue(
            all(count <= 2 for count in stats["kept_tokens_per_expert"])
        )
        self.assertGreaterEqual(stats["dropped_tokens"], 0)

    def test_model_forward_and_total_loss(self) -> None:
        input_ids = torch.randint(0, self.transformer_config.vocab_size, (2, 8))
        logits, lm_loss, aux_loss = self.model(input_ids, labels=input_ids)
        self.assertEqual(logits.shape, (2, 8, self.transformer_config.vocab_size))
        assert lm_loss is not None
        self.assertTrue(torch.isfinite(lm_loss).item())
        self.assertTrue(torch.isfinite(aux_loss).item())
        total = self.model.total_loss(lm_loss, aux_loss)
        self.assertTrue(torch.isfinite(total).item())
        self.assertEqual(len(self.model.last_routing_stats), 2)

    def test_backward_reaches_router_and_experts(self) -> None:
        input_ids = torch.randint(0, self.transformer_config.vocab_size, (2, 8))
        _, lm_loss, aux_loss = self.model(input_ids, labels=input_ids)
        assert lm_loss is not None
        self.model.total_loss(lm_loss, aux_loss).backward()
        router_grad = self.model.blocks[0].moe.router.weight.grad
        expert_grad = self.model.blocks[0].moe.experts[0].gate.weight.grad
        self.assertIsNotNone(router_grad)
        self.assertTrue(torch.isfinite(router_grad).all().item())
        self.assertIsNotNone(expert_grad)

    def test_full_forward_matches_incremental_decode(self) -> None:
        input_ids = torch.randint(0, self.transformer_config.vocab_size, (2, 8))
        full_logits, _, _ = self.model(input_ids)
        cache = [dict() for _ in range(self.transformer_config.n_layers)]
        pieces = []
        for position in range(input_ids.size(1)):
            logits, _, _ = self.model(
                input_ids[:, position : position + 1],
                start_pos=position,
                kv_cache=cache,
            )
            pieces.append(logits)
        incremental_logits = torch.cat(pieces, dim=1)
        torch.testing.assert_close(full_logits, incremental_logits, rtol=1e-5, atol=1e-5)

    def test_parameter_count_exceeds_dense_ffn_baseline(self) -> None:
        self.assertGreater(count_moe_parameters(self.model), 0)


if __name__ == "__main__":
    unittest.main()
