"""MoE SFT pipeline tests (data + smoke; no full training)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from architecture_lab.models.dense_transformer import TransformerConfig  # noqa: E402
from architecture_lab.models.moe_transformer import MoEConfig, MoETransformer  # noqa: E402
from architecture_lab.tokenization import BPETokenizer  # noqa: E402
from architecture_lab.training.sft_moe_training import build_moe_model  # noqa: E402
from architecture_lab.training.sft_training import (  # noqa: E402
    SFTBatcher,
    build_sft_examples,
    masked_causal_loss,
)

TOKENIZER = ROOT / "artifacts" / "tokenizers" / "owt-bpe" / "v0.2.0" / "tokenizer.json"


def _sample() -> dict:
    return {
        "schema_version": "1.0",
        "id": "sft-moe-test",
        "messages": [{"role": "user", "content": "请帮我查北京天气"}],
        "tools": [{"type": "function", "function": {
            "name": "d1_get_weather",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}},
                           "required": ["city"]},
        }}],
        "expected_tool_calls": [{"call_id": "c1", "name": "d1_get_weather",
                                 "arguments": {"city": "北京"},
                                 "expected_result": "北京 晴 22C"}],
        "expected_answer": "北京今天晴，22度。",
        "metadata": {"source": "test", "task_type": "single_tool",
                     "data_version": "D1.1", "pipeline_version": "test",
                     "created_at": "2026-08-27T00:00:00Z",
                     "validation": {"schema_valid": True}},
    }


class SftMoeDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = BPETokenizer.load(TOKENIZER)

    def test_build_moe_model_returns_moe_transformer(self) -> None:
        model = build_moe_model(
            {"vocab_size": 8192, "max_seq_len": 256, "d_model": 128,
             "n_heads": 4, "n_layers": 4, "d_ff": 512, "dropout": 0.0,
             "rope_base": 10000.0, "moe_num_experts": 4,
             "moe_aux_loss_coeff": 0.01},
            self.tokenizer.vocab_size,
        )
        self.assertIsInstance(model, MoETransformer)
        self.assertEqual(model.moe_config.num_experts, 4)
        self.assertEqual(model.moe_config.aux_loss_weight, 0.01)

    def test_moe_forward_returns_logits_loss_aux(self) -> None:
        torch.manual_seed(0)
        model = build_moe_model(
            {"vocab_size": 8192, "max_seq_len": 256, "d_model": 64,
             "n_heads": 4, "n_layers": 2, "d_ff": 256, "dropout": 0.0,
             "rope_base": 10000.0, "moe_num_experts": 2},
            self.tokenizer.vocab_size,
        ).to("cpu")
        ids = torch.randint(0, self.tokenizer.vocab_size, (1, 8))
        logits, ce, aux = model(ids, labels=ids)
        self.assertEqual(logits.shape, (1, 8, self.tokenizer.vocab_size))
        self.assertEqual(ce.ndim, 0)  # scalar
        self.assertGreaterEqual(float(aux.detach()), 0.0)
        # Masked loss: only verifies the pipeline runs and returns a finite
        # tensor with correct shape (the value depends on random init).
        masked = masked_causal_loss(logits, ids, torch.ones_like(ids))
        self.assertEqual(masked.ndim, 0)
        self.assertTrue(torch.isfinite(masked.detach()))

    def test_sft_examples_cover_assistant_span(self) -> None:
        exs = build_sft_examples([_sample()], self.tokenizer, max_len=256)
        self.assertEqual(len(exs), 1)
        ex = exs[0]
        self.assertGreater(sum(ex.mask), 0)


if __name__ == "__main__":
    unittest.main()