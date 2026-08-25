"""Tests for the rewritten llm-lab BPE tokenizer."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture_lab.tokenization import BPETokenizer, train_bpe, train_bpe_from_file  # noqa: E402


class BPETokenizerTests(unittest.TestCase):
    def setUp(self) -> None:
        text = (
            "Once upon a time, a small model learned. "
            "Once upon a time, a tool was called. "
            "<|endoftext|>"
        )
        vocab, merges = train_bpe(
            text,
            vocab_size=280,
            special_tokens=("<|endoftext|>",),
        )
        self.tokenizer = BPETokenizer(
            vocab=vocab,
            merges=merges,
            special_tokens=("<|endoftext|>",),
        )

    def test_roundtrip_text_and_special_token(self) -> None:
        text = "Once upon a time, a small model. <|endoftext|>"
        ids = self.tokenizer.encode(text)
        self.assertEqual(self.tokenizer.decode(ids), text)
        self.assertIn("<|endoftext|>", self.tokenizer.special_token_ids)

    def test_vocab_size_and_bpe_merges(self) -> None:
        self.assertGreater(self.tokenizer.vocab_size, 256)
        self.assertGreater(len(self.tokenizer.merges), 0)
        self.assertLessEqual(self.tokenizer.vocab_size, 280)

    def test_artifact_save_load_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            self.tokenizer.save(path)
            loaded = BPETokenizer.load(path)
            text = "Once upon a time. <|endoftext|>"
            self.assertEqual(loaded.encode(text), self.tokenizer.encode(text))
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["artifact_type"], "llm-lab-bpe-tokenizer")
            self.assertIn("artifact_sha256", payload)

    def test_tampered_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            self.tokenizer.save(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["special_tokens"] = []
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                BPETokenizer.load(path)

    def test_train_bpe_iterable_is_deterministic(self) -> None:
        from architecture_lab.tokenization import train_bpe_iterable

        chunks = ["Once upon a time, ", "a small model learned."]
        first = train_bpe_iterable(
            chunks,
            vocab_size=280,
            special_tokens=("<|endoftext|>",),
        )
        second = train_bpe_iterable(
            chunks,
            vocab_size=280,
            special_tokens=("<|endoftext|>",),
        )
        self.assertEqual(first, second)

    def test_train_bpe_from_file_supports_newline_aligned_prefix(self) -> None:
        from architecture_lab.tokenization import train_bpe_from_file

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.txt"
            path.write_text("alpha alpha\nbeta beta\ngamma gamma\n", encoding="utf-8")
            tokenizer = train_bpe_from_file(
                path,
                vocab_size=270,
                special_tokens=("<|endoftext|>",),
                chunk_size_bytes=5,
                max_training_bytes=13,
            )
            self.assertEqual(tokenizer.decode(tokenizer.encode("alpha alpha")), "alpha alpha")

    def test_train_bpe_from_file_preserves_special_token_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.txt"
            text = "alpha alpha<|endoftext|>beta beta<|endoftext|>"
            path.write_text(text, encoding="utf-8")
            streamed = train_bpe_from_file(
                path,
                vocab_size=275,
                special_tokens=("<|endoftext|>",),
                chunk_size_bytes=1,
            )
            direct_vocab, direct_merges = train_bpe(
                text,
                vocab_size=275,
                special_tokens=("<|endoftext|>",),
            )
            self.assertEqual(streamed.vocab, direct_vocab)
            self.assertEqual(streamed.merges, direct_merges)

    def test_invalid_vocab_size_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            train_bpe("hello", vocab_size=256, special_tokens=("<eos>",))


if __name__ == "__main__":
    unittest.main()
