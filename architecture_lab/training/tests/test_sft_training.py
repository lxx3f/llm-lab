"""SFT training pipeline unit tests (no GPU, no API calls).

Verifies the masked-sequence construction, batcher shapes, and masked loss
behavior without running an actual training loop.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from architecture_lab.tokenization import BPETokenizer  # noqa: E402
from architecture_lab.training.sft_training import (  # noqa: E402
    SFTBatcher,
    build_sft_examples,
    masked_causal_loss,
    render_plan_json,
    render_result_summary,
)

TOKENIZER = ROOT / "artifacts" / "tokenizers" / "owt-bpe" / "v0.2.0" / "tokenizer.json"


def _sample() -> dict:
    return {
        "schema_version": "1.0",
        "id": "sft-test",
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


class SftDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = BPETokenizer.load(TOKENIZER)

    def test_example_mask_covers_only_assistant_span(self) -> None:
        ex = build_sft_examples([_sample()], self.tokenizer, max_len=256)[0]
        self.assertEqual(len(ex.input_ids), len(ex.mask))
        self.assertEqual(len(ex.target_ids), len(ex.input_ids))
        # The prefix (user + separators) must be masked off.
        user_ids = self.tokenizer.encode("### User\n请帮我查北京天气\n### Assistant\n")
        self.assertEqual(ex.mask[: len(user_ids)], [0] * len(user_ids))
        # Assistant span must be active.
        self.assertGreater(sum(ex.mask), 0)
        # Targets are next-token ids.
        self.assertEqual(ex.target_ids[:-1], ex.input_ids[1:])

    def test_render_plan_and_result(self) -> None:
        sample = _sample()
        plan = render_plan_json(sample["expected_tool_calls"])
        self.assertIn("d1_get_weather", plan)
        # Newline-separated objects (no enclosing array): each line is a
        # standalone {"call_id"...} JSON object.
        obj = json.loads(plan)
        self.assertEqual(obj["arguments"]["city"], "北京")
        result = render_result_summary(sample["expected_tool_calls"])
        self.assertIn("d1_get_weather", result)

    def test_batcher_shapes_and_dynamic_padding(self) -> None:
        examples = build_sft_examples([_sample()] * 5, self.tokenizer, max_len=256)
        batcher = SFTBatcher(examples, batch_size=2, seed=1)
        seen = 0
        sizes = []
        for inputs, targets, mask in batcher.iter_epoch():
            self.assertEqual(inputs.shape, targets.shape)
            self.assertEqual(inputs.shape, mask.shape)
            sizes.append(inputs.shape[0])
            seen += 1
        # 5 examples / batch 2 -> 3 batches of sizes [2, 2, 1].
        self.assertEqual(seen, 3)
        self.assertEqual(sizes, [2, 2, 1])

    def test_masked_loss_ignores_zero_mask(self) -> None:
        # All-mask-zero: loss must not be NaN and must have a grad_fn.
        logits = torch.randn(2, 8, 128)
        targets = torch.randint(0, 128, (2, 8))
        mask = torch.zeros(2, 8, dtype=torch.long)
        loss = masked_causal_loss(logits, targets, mask)
        self.assertEqual(float(loss), 0.0)

    def test_masked_loss_only_counts_masked_positions(self) -> None:
        logits = torch.randn(1, 4, 100)
        targets = torch.tensor([[1, 2, 3, 4]])
        # Only the last position is masked -> loss = CE at position 3 only.
        mask = torch.tensor([[0, 0, 0, 1]])
        loss = masked_causal_loss(logits, targets, mask)
        expected = torch.nn.functional.cross_entropy(
            logits[0, 3:4], targets[0, 3:4]
        )
        self.assertAlmostEqual(float(loss), float(expected), places=6)

    def test_no_tool_sample_builds(self) -> None:
        sample = _sample()
        sample["expected_tool_calls"] = []
        sample["expected_answer"] = "你好！"
        ex = build_sft_examples([sample], self.tokenizer, max_len=256)[0]
        self.assertGreater(sum(ex.mask), 0)

    def test_system_message_is_not_used_as_user_turn(self) -> None:
        """D1 template samples carry a leading English system message; the
        user turn must be the last role=user content, not messages[0]."""
        sample = _sample()
        sample["messages"] = [
            {"role": "system", "content": "You are a helpful assistant with tool access."},
            {"role": "user", "content": "今天北京天气怎么样？"},
        ]
        ex = build_sft_examples([sample], self.tokenizer, max_len=256)[0]
        prefix = self.tokenizer.encode("### User\n今天北京天气怎么样？\n### Assistant\n")
        # The user span must match the true user turn (Chinese), not the
        # English system message.
        self.assertEqual(ex.input_ids[: len(prefix)], prefix)
        self.assertEqual(ex.mask[: len(prefix)], [0] * len(prefix))


if __name__ == "__main__":
    unittest.main()
