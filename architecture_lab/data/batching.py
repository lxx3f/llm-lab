"""Batch sampling utilities for contiguous causal-language-model token streams."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .token_cache import TokenCacheInfo, validate_token_cache


@dataclass(frozen=True)
class BatchSamplerConfig:
    batch_size: int
    sequence_length: int
    seed: int = 42

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")


class TokenStreamBatcher:
    """Sample independent contiguous windows from one validated token cache.

    For every sampled start offset ``s``, ``inputs`` contains ``tokens[s:s+T]``
    and ``targets`` contains ``tokens[s+1:s+T+1]``. Windows may overlap across
    batches, but never cross the end of the selected split.
    """

    def __init__(
        self,
        tokens: Tensor,
        *,
        batch_size: int,
        sequence_length: int,
        seed: int = 42,
    ) -> None:
        if tokens.ndim != 1:
            raise ValueError("tokens must be a one-dimensional tensor")
        if tokens.dtype != torch.uint16:
            raise ValueError("tokens must have dtype torch.uint16")
        config = BatchSamplerConfig(batch_size, sequence_length, seed)
        if tokens.numel() < config.sequence_length + 1:
            raise ValueError("token cache is too short for one causal batch")
        self.tokens = tokens
        self.config = config
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(config.seed)
        self._max_start = tokens.numel() - config.sequence_length - 1

    @property
    def token_count(self) -> int:
        return self.tokens.numel()

    @property
    def batches_per_epoch(self) -> int:
        return max(
            0,
            (self.token_count - 1)
            // (self.config.batch_size * self.config.sequence_length),
        )

    def _batch_from_starts(self, starts: list[int], *, device: torch.device | str | None) -> tuple[Tensor, Tensor]:
        input_ids = torch.stack(
            [self.tokens[start : start + self.config.sequence_length] for start in starts]
        )
        target_ids = torch.stack(
            [
                self.tokens[start + 1 : start + self.config.sequence_length + 1]
                for start in starts
            ]
        )
        input_ids = input_ids.to(dtype=torch.long)
        target_ids = target_ids.to(dtype=torch.long)
        if device is not None:
            input_ids = input_ids.to(device=device)
            target_ids = target_ids.to(device=device)
        return input_ids, target_ids

    def sample(self, *, device: torch.device | str | None = None) -> tuple[Tensor, Tensor]:
        """Return one deterministic-by-seed batch of input and next-token targets."""
        starts = torch.randint(
            0,
            self._max_start + 1,
            (self.config.batch_size,),
            generator=self._generator,
            dtype=torch.int64,
        )
        return self._batch_from_starts([int(start) for start in starts], device=device)

    def iter_epoch(self, *, device: torch.device | str | None = None):
        """Yield deterministic, non-overlapping input windows for one split pass."""
        batch_span = self.config.batch_size * self.config.sequence_length
        batch_count = (self.token_count - 1) // batch_span
        for step in range(batch_count):
            starts = [
                step * batch_span + offset * self.config.sequence_length
                for offset in range(self.config.batch_size)
            ]
            yield self._batch_from_starts(starts, device=device)


def load_token_cache(
    *,
    token_path: str | Path,
    metadata_path: str | Path,
    source_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    split: str | None = None,
    max_bytes: int | None = None,
    mmap: bool = True,
) -> tuple[Tensor, TokenCacheInfo]:
    """Validate and load a uint16 token cache, optionally memory-mapped."""
    info = validate_token_cache(
        token_path=token_path,
        metadata_path=metadata_path,
        source_path=source_path,
        tokenizer_path=tokenizer_path,
        split=split,
        max_bytes=max_bytes,
    )
    if mmap:
        tokens = torch.from_file(
            str(info.token_path),
            shared=False,
            size=info.token_count,
            dtype=torch.uint16,
        )
    else:
        raw = info.token_path.read_bytes()
        tokens = torch.frombuffer(bytearray(raw), dtype=torch.uint16).clone()
    if tokens.numel() != info.token_count:
        raise ValueError("loaded token count does not match cache metadata")
    return tokens, info
