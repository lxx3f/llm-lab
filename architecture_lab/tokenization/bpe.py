"""BPE tokenizer utilities adapted from a CS336 Assignment 1 implementation.

Source reviewed:
https://github.com/lxx3f/assignment1-basics

The implementation is rewritten for llm-lab's interfaces and JSON artifacts;
it is not a runtime dependency on the source repository.
"""

from __future__ import annotations

import hashlib
import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import regex


GPT2_PRETOKENIZATION_PATTERN = (
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


ByteToken = bytes
Merge = tuple[bytes, bytes]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _split_special_tokens(text: str, special_tokens: Sequence[str]) -> list[str]:
    if not special_tokens:
        return regex.findall(GPT2_PRETOKENIZATION_PATTERN, text)
    ordered = sorted(set(special_tokens), key=len, reverse=True)
    union = "|".join(regex.escape(token) for token in ordered)
    parts = regex.split(f"({union})", text)
    special_set = set(special_tokens)
    result: list[str] = []
    for part in parts:
        if not part:
            continue
        if part in special_set:
            result.append(part)
        else:
            result.extend(regex.findall(GPT2_PRETOKENIZATION_PATTERN, part))
    return result


def _iter_split_for_training(
    text: str, special_tokens: Sequence[str]
) -> Iterator[bytes]:
    parts = _split_special_tokens(text, special_tokens)
    special_set = set(special_tokens)
    for part in parts:
        if part not in special_set:
            yield part.encode("utf-8")


def _word_to_bytes(word: bytes) -> tuple[bytes, ...]:
    return tuple(bytes([value]) for value in word)


def _get_pairs(word: Sequence[bytes]) -> set[Merge]:
    return set(zip(word, word[1:]))


def _validate_training_args(vocab_size: int, special_tokens: Sequence[str]) -> None:
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size must fit 256 byte tokens and special tokens")
    if len(set(special_tokens)) != len(special_tokens):
        raise ValueError("special_tokens must be unique")


def _merge_word(word: Sequence[bytes], pair: Merge, merged: bytes) -> tuple[bytes, ...]:
    result: list[bytes] = []
    index = 0
    while index < len(word):
        if index + 1 < len(word) and (word[index], word[index + 1]) == pair:
            result.append(merged)
            index += 2
        else:
            result.append(word[index])
            index += 1
    return tuple(result)


def _train_bpe_from_frequencies(
    token_frequencies: dict[tuple[bytes, ...], int],
    vocab_size: int,
    special_tokens: Sequence[str],
) -> tuple[dict[int, bytes], list[Merge]]:
    """Run BPE merges with incrementally maintained pair counts.

    The frequency convention intentionally matches the original implementation:
    each pair contributes once per distinct pre-tokenized word, weighted by that
    word's frequency. A lazy heap avoids rescanning all pairs to find the next
    merge, while ``pair_to_words`` limits updates to words containing the merge.
    """
    vocab = {index: bytes([index]) for index in range(256)}
    merges: list[Merge] = []
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    pair_counts: dict[Merge, int] = {}
    pair_to_words: dict[Merge, set[tuple[bytes, ...]]] = {}
    heap: list[tuple[int, Merge]] = []

    def add_word(word: tuple[bytes, ...], frequency: int) -> None:
        token_frequencies[word] = token_frequencies.get(word, 0) + frequency
        for pair in _get_pairs(word):
            pair_counts[pair] = pair_counts.get(pair, 0) + frequency
            pair_to_words.setdefault(pair, set()).add(word)
            heapq.heappush(heap, (-pair_counts[pair], pair))

    def remove_word(word: tuple[bytes, ...], frequency: int) -> None:
        for pair in _get_pairs(word):
            pair_counts[pair] -= frequency
            words = pair_to_words[pair]
            words.remove(word)
            if not words:
                del pair_to_words[pair]
                del pair_counts[pair]

    for word, frequency in list(token_frequencies.items()):
        for pair in _get_pairs(word):
            pair_counts[pair] = pair_counts.get(pair, 0) + frequency
            pair_to_words.setdefault(pair, set()).add(word)
            heapq.heappush(heap, (-pair_counts[pair], pair))

    def pop_best_pair() -> Merge | None:
        candidates: list[Merge] = []
        best_count: int | None = None
        while heap:
            neg_count, pair = heapq.heappop(heap)
            count = -neg_count
            if pair_counts.get(pair) != count:
                continue
            best_count = count
            candidates.append(pair)
            while heap and -heap[0][0] == best_count:
                next_neg_count, next_pair = heapq.heappop(heap)
                if pair_counts.get(next_pair) == best_count:
                    candidates.append(next_pair)
            chosen = max(candidates)
            for candidate in candidates:
                if candidate != chosen and pair_counts.get(candidate) == best_count:
                    heapq.heappush(heap, (-best_count, candidate))
            return chosen
        return None

    while len(vocab) < vocab_size:
        best_pair = pop_best_pair()
        if best_pair is None:
            break
        merged = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab[len(vocab)] = merged

        affected_words = list(pair_to_words.get(best_pair, ()))
        replacement_frequencies: dict[tuple[bytes, ...], int] = {}
        for old_word in affected_words:
            frequency = token_frequencies.pop(old_word)
            remove_word(old_word, frequency)
            new_word = _merge_word(old_word, best_pair, merged)
            replacement_frequencies[new_word] = replacement_frequencies.get(new_word, 0) + frequency
        for new_word, frequency in replacement_frequencies.items():
            add_word(new_word, frequency)

    return vocab, merges


def train_bpe(
    text: str,
    vocab_size: int,
    special_tokens: Sequence[str] = (),
) -> tuple[dict[int, bytes], list[Merge]]:
    """Train a byte-level BPE vocabulary on a text string."""
    return train_bpe_iterable((text,), vocab_size, special_tokens)


def train_bpe_iterable(
    chunks: Iterable[str],
    vocab_size: int,
    special_tokens: Sequence[str] = (),
) -> tuple[dict[int, bytes], list[Merge]]:
    """Train BPE from chunks without concatenating the complete input."""
    _validate_training_args(vocab_size, special_tokens)

    token_frequencies: dict[tuple[bytes, ...], int] = {}
    for chunk in chunks:
        for word in _iter_split_for_training(chunk, special_tokens):
            token = _word_to_bytes(word)
            token_frequencies[token] = token_frequencies.get(token, 0) + 1

    return _train_bpe_from_frequencies(token_frequencies, vocab_size, special_tokens)


@dataclass
class BPETokenizer:
    """Byte-level BPE tokenizer with JSON serialization."""

    vocab: dict[int, bytes]
    merges: list[Merge]
    special_tokens: tuple[str, ...] = ()
    version: str = "llm-lab-bpe-v1"

    def __post_init__(self) -> None:
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("special_tokens must be unique")
        self.vocab = {int(index): bytes(value) for index, value in self.vocab.items()}
        self._byte_to_id = {value: index for index, value in self.vocab.items()}
        self._merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._special_to_id: dict[str, int] = {}
        for token in self.special_tokens:
            encoded = token.encode("utf-8")
            if encoded not in self._byte_to_id:
                raise ValueError(f"special token missing from vocabulary: {token}")
            self._special_to_id[token] = self._byte_to_id[encoded]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def special_token_ids(self) -> dict[str, int]:
        return dict(self._special_to_id)

    def _apply_merges(self, word: tuple[bytes, ...]) -> tuple[bytes, ...]:
        current = word
        while len(current) > 1:
            pairs = _get_pairs(current)
            available = [pair for pair in pairs if pair in self._merge_ranks]
            if not available:
                break
            pair = min(available, key=self._merge_ranks.__getitem__)
            current = _merge_word(current, pair, pair[0] + pair[1])
        return current

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for part in _split_special_tokens(text, self.special_tokens):
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
                continue
            for byte_word in regex.findall(GPT2_PRETOKENIZATION_PATTERN, part):
                merged = self._apply_merges(_word_to_bytes(byte_word.encode("utf-8")))
                try:
                    ids.extend(self._byte_to_id[token] for token in merged)
                except KeyError as error:
                    raise ValueError("tokenizer vocabulary does not cover encoded bytes") from error
        return ids

    def encode_iterable(self, chunks: Iterable[str]) -> Iterator[int]:
        for chunk in chunks:
            yield from self.encode(chunk)

    def decode(self, ids: Iterable[int]) -> str:
        try:
            data = b"".join(self.vocab[int(index)] for index in ids)
        except KeyError as error:
            raise ValueError(f"unknown token id: {error.args[0]}") from error
        return data.decode("utf-8", errors="replace")

    def artifact_dict(self) -> dict[str, object]:
        vocab_items = [
            {"id": index, "bytes_hex": value.hex()}
            for index, value in sorted(self.vocab.items())
        ]
        merge_items = [[left.hex(), right.hex()] for left, right in self.merges]
        payload: dict[str, object] = {
            "artifact_type": "llm-lab-bpe-tokenizer",
            "version": self.version,
            "vocab": vocab_items,
            "merges": merge_items,
            "special_tokens": list(self.special_tokens),
        }
        payload["artifact_sha256"] = _sha256_json(payload)
        return payload

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.artifact_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        expected_hash = payload.pop("artifact_sha256", None)
        if expected_hash != _sha256_json(payload):
            raise ValueError("tokenizer artifact hash mismatch")
        vocab = {
            int(item["id"]): bytes.fromhex(item["bytes_hex"])
            for item in payload["vocab"]
        }
        merges = [
            (bytes.fromhex(pair[0]), bytes.fromhex(pair[1]))
            for pair in payload["merges"]
        ]
        return cls(
            vocab=vocab,
            merges=merges,
            special_tokens=tuple(payload["special_tokens"]),
            version=str(payload["version"]),
        )


def training_prefix_bytes(input_path: str | Path, max_training_bytes: int | None = None) -> int:
    """Return bytes included by a newline-aligned training prefix."""
    path = Path(input_path)
    if max_training_bytes is None:
        return path.stat().st_size
    if max_training_bytes <= 0:
        raise ValueError("max_training_bytes must be positive when provided")

    consumed = 0
    with path.open("r", encoding="utf-8", newline="") as file:
        for line in file:
            line_bytes = len(line.encode("utf-8"))
            if consumed + line_bytes > max_training_bytes:
                break
            consumed += line_bytes
    return consumed


def train_bpe_from_file(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: Sequence[str] = (),
    *,
    chunk_size_bytes: int = 8 * 1024 * 1024,
    max_training_bytes: int | None = None,
) -> BPETokenizer:
    """Train BPE from a UTF-8 file using newline-aligned chunks."""
    path = Path(input_path)
    if chunk_size_bytes <= 0:
        raise ValueError("chunk_size_bytes must be positive")
    if max_training_bytes is not None and max_training_bytes <= 0:
        raise ValueError("max_training_bytes must be positive when provided")

    def chunks() -> Iterator[str]:
        bytes_seen = 0
        pending: list[str] = []
        pending_bytes = 0
        special_set = set(special_tokens)
        special_union = "|".join(regex.escape(token) for token in sorted(special_set, key=len, reverse=True))
        splitter = regex.compile(f"({special_union})") if special_union else None

        def emit_pending() -> Iterator[str]:
            nonlocal pending, pending_bytes
            if pending:
                yield "".join(pending)
                pending = []
                pending_bytes = 0

        with path.open("r", encoding="utf-8", newline="") as file:
            for line in file:
                line_bytes = len(line.encode("utf-8"))
                if max_training_bytes is not None and bytes_seen + line_bytes > max_training_bytes:
                    break
                bytes_seen += line_bytes

                if splitter is None:
                    pending.append(line)
                    pending_bytes += line_bytes
                    if pending_bytes >= chunk_size_bytes:
                        yield from emit_pending()
                    continue

                for part in splitter.split(line):
                    if not part:
                        continue
                    if part in special_set:
                        yield from emit_pending()
                    else:
                        pending.append(part)
                        pending_bytes += len(part.encode("utf-8"))

            yield from emit_pending()

    vocab, merges = train_bpe_iterable(chunks(), vocab_size, special_tokens)
    return BPETokenizer(vocab, merges, tuple(special_tokens))
