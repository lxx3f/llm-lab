"""Tests for token-stream loading and causal batch sampling."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from architecture_lab.data.batching import TokenStreamBatcher, load_token_cache
from architecture_lab.data.token_cache import encode_token_cache
from architecture_lab.tokenization import BPETokenizer, train_bpe


class TokenBatchingTests(unittest.TestCase):
    def _cache(self, root: Path, text: str = "abcdefghij\nklmnopqrst\nuvwxyz\n") -> tuple[Path, Path, Path]:
        source = root / "train.txt"
        source.write_bytes(text.encode("utf-8"))
        vocab, merges = train_bpe(text, vocab_size=280, special_tokens=("<|endoftext|>",))
        tokenizer_path = root / "tokenizer.json"
        BPETokenizer(vocab, merges, ("<|endoftext|>",)).save(tokenizer_path)
        info = encode_token_cache(
            input_path=source,
            tokenizer_path=tokenizer_path,
            output_root=root / "cache",
            split="train",
            chunk_tokens=3,
        )
        return info.token_path, info.metadata_path, source

    def test_load_validates_and_returns_long_batches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            token_path, metadata_path, source = self._cache(Path(directory))
            tokens, info = load_token_cache(
                token_path=token_path,
                metadata_path=metadata_path,
                source_path=source,
                split="train",
                mmap=False,
            )
            self.assertEqual(tokens.dtype, torch.uint16)
            self.assertEqual(tokens.numel(), info.token_count)
            batcher = TokenStreamBatcher(tokens, batch_size=2, sequence_length=4, seed=7)
            inputs, targets = batcher.sample()
            self.assertEqual(inputs.dtype, torch.long)
            self.assertEqual(targets.dtype, torch.long)
            torch.testing.assert_close(targets[:, :-1], inputs[:, 1:])

    def test_sample_is_reproducible_for_same_seed(self) -> None:
        tokens = torch.arange(40, dtype=torch.int64).to(torch.uint16)
        first = TokenStreamBatcher(tokens, batch_size=3, sequence_length=5, seed=123)
        second = TokenStreamBatcher(tokens, batch_size=3, sequence_length=5, seed=123)
        first_batch = first.sample()
        second_batch = second.sample()
        torch.testing.assert_close(first_batch[0], second_batch[0])
        torch.testing.assert_close(first_batch[1], second_batch[1])

    def test_iter_epoch_preserves_contiguous_stream_and_drops_remainder(self) -> None:
        tokens = torch.arange(25, dtype=torch.int64).to(torch.uint16)
        batcher = TokenStreamBatcher(tokens, batch_size=2, sequence_length=4)
        batches = list(batcher.iter_epoch())
        self.assertEqual(len(batches), 3)
        inputs, targets = batches[0]
        torch.testing.assert_close(inputs, torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]))
        torch.testing.assert_close(targets, torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]))
        self.assertEqual(batches[-1][0][0, -1].item(), 19)

    def test_invalid_cache_and_short_stream_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TokenStreamBatcher(
                torch.arange(5, dtype=torch.int64).to(torch.uint16),
                batch_size=1,
                sequence_length=5,
            )
        with self.assertRaises(ValueError):
            TokenStreamBatcher(torch.arange(10, dtype=torch.int64), batch_size=1, sequence_length=4)
        with tempfile.TemporaryDirectory() as directory:
            token_path, metadata_path, source = self._cache(Path(directory))
            source.write_bytes(b"changed\n")
            with self.assertRaises(ValueError):
                load_token_cache(
                    token_path=token_path,
                    metadata_path=metadata_path,
                    source_path=source,
                    split="train",
                )


if __name__ == "__main__":
    unittest.main()
