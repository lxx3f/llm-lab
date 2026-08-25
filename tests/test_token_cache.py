"""Tests for streaming token cache encoding and validation."""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture_lab.data.token_cache import (  # noqa: E402
    encode_token_cache,
    validate_token_cache,
)
from architecture_lab.tokenization import BPETokenizer, train_bpe  # noqa: E402


class TokenCacheTests(unittest.TestCase):
    def _make_tokenizer(self, root: Path) -> Path:
        vocab, merges = train_bpe(
            "alpha alpha beta beta <|endoftext|> gamma gamma",
            vocab_size=280,
            special_tokens=("<|endoftext|>",),
        )
        path = root / "tokenizer.json"
        BPETokenizer(vocab, merges, ("<|endoftext|>",)).save(path)
        return path

    def test_encode_and_validate_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.txt"
            source.write_bytes(
                "alpha alpha<|endoftext|>\nβeta beta\nignored after limit\n".encode("utf-8")
            )
            tokenizer = self._make_tokenizer(root)
            output_root = root / "cache"
            info = encode_token_cache(
                input_path=source,
                tokenizer_path=tokenizer,
                output_root=output_root,
                split="train",
                max_bytes=len("alpha alpha<|endoftext|>\nβeta beta\n".encode("utf-8")),
                chunk_tokens=2,
            )
            self.assertGreater(info.token_count, 0)
            self.assertEqual(info.token_path.stat().st_size, info.token_count * 2)
            metadata = json.loads(info.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["cache"]["dtype"], "uint16")
            self.assertTrue(metadata["source"]["newline_aligned_prefix"])
            self.assertEqual(metadata["source"]["encoded_bytes"], info.encoded_bytes)

            ids = [value[0] for value in struct.iter_unpack("<H", info.token_path.read_bytes())]
            loaded = BPETokenizer.load(tokenizer)
            self.assertEqual(loaded.decode(ids), "alpha alpha<|endoftext|>\nβeta beta\n")
            validated = validate_token_cache(
                token_path=info.token_path,
                metadata_path=info.metadata_path,
                source_path=source,
                tokenizer_path=tokenizer,
                split="train",
                max_bytes=len("alpha alpha<|endoftext|>\nβeta beta\n".encode("utf-8")),
            )
            self.assertEqual(validated.token_count, info.token_count)

    def test_source_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.txt"
            source.write_text("alpha alpha\n", encoding="utf-8")
            tokenizer = self._make_tokenizer(root)
            info = encode_token_cache(
                input_path=source,
                tokenizer_path=tokenizer,
                output_root=root / "cache",
                split="train",
            )
            source.write_text("changed\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_token_cache(
                    token_path=info.token_path,
                    metadata_path=info.metadata_path,
                    source_path=source,
                    tokenizer_path=tokenizer,
                    split="train",
                )

    def test_tokenizer_artifact_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.txt"
            source.write_text("alpha alpha\n", encoding="utf-8")
            tokenizer = self._make_tokenizer(root)
            info = encode_token_cache(
                input_path=source,
                tokenizer_path=tokenizer,
                output_root=root / "cache",
                split="train",
            )
            tokenizer.write_text(tokenizer.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_token_cache(
                    token_path=info.token_path,
                    metadata_path=info.metadata_path,
                    source_path=source,
                    tokenizer_path=tokenizer,
                    split="train",
                )


    def test_crlf_source_preserves_newline_bytes_and_roundtrips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.txt"
            text = "alpha alpha\r\nbeta beta\r\n"
            source.write_bytes(text.encode("utf-8"))
            tokenizer = self._make_tokenizer(root)
            info = encode_token_cache(
                input_path=source,
                tokenizer_path=tokenizer,
                output_root=root / "cache",
                split="train",
                max_bytes=len(text.encode("utf-8")),
            )
            ids = [value[0] for value in struct.iter_unpack("<H", info.token_path.read_bytes())]
            self.assertEqual(BPETokenizer.load(tokenizer).decode(ids), text)
            self.assertEqual(info.encoded_bytes, len(text.encode("utf-8")))

    def test_validator_rejects_split_or_max_bytes_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.txt"
            source.write_text("alpha alpha\nbeta beta\n", encoding="utf-8")
            tokenizer = self._make_tokenizer(root)
            info = encode_token_cache(
                input_path=source,
                tokenizer_path=tokenizer,
                output_root=root / "cache",
                split="train",
                max_bytes=12,
            )
            with self.assertRaises(ValueError):
                validate_token_cache(
                    token_path=info.token_path,
                    metadata_path=info.metadata_path,
                    source_path=source,
                    tokenizer_path=tokenizer,
                    split="validation",
                    max_bytes=12,
                )
            with self.assertRaises(ValueError):
                validate_token_cache(
                    token_path=info.token_path,
                    metadata_path=info.metadata_path,
                    source_path=source,
                    tokenizer_path=tokenizer,
                    split="train",
                    max_bytes=13,
                )

    def test_existing_cache_requires_force(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.txt"
            source.write_text("alpha alpha\n", encoding="utf-8")
            tokenizer = self._make_tokenizer(root)
            kwargs = {
                "input_path": source,
                "tokenizer_path": tokenizer,
                "output_root": root / "cache",
                "split": "train",
            }
            encode_token_cache(**kwargs)
            with self.assertRaises(FileExistsError):
                encode_token_cache(**kwargs)


if __name__ == "__main__":
    unittest.main()
