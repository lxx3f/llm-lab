"""Versioned streaming token caches for language-model text data."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from architecture_lab.tokenization import BPETokenizer

CACHE_METADATA_VERSION = "1.0"
TOKEN_DTYPE = "uint16"
TOKEN_MAX = 2**16 - 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _iter_lines_prefix(path: Path, max_bytes: int | None) -> Iterator[tuple[str, int]]:
    if max_bytes is not None and max_bytes <= 0:
        raise ValueError("max_bytes must be positive when provided")
    consumed = 0
    with path.open("r", encoding="utf-8", newline="") as file:
        for line in file:
            line_bytes = len(line.encode("utf-8"))
            if max_bytes is not None and consumed + line_bytes > max_bytes:
                break
            consumed += line_bytes
            yield line, line_bytes


def _write_uint16_chunk(file: Any, ids: list[int]) -> int:
    if not ids:
        return 0
    for token_id in ids:
        if not 0 <= token_id <= TOKEN_MAX:
            raise ValueError(f"token id {token_id} cannot be stored as uint16")
    payload = b"".join(token_id.to_bytes(2, "little") for token_id in ids)
    file.write(payload)
    return len(ids)


def _validate_tokenizer_dtype(tokenizer: BPETokenizer) -> None:
    if tokenizer.vocab_size - 1 > TOKEN_MAX:
        raise ValueError(
            f"tokenizer vocab size {tokenizer.vocab_size} exceeds uint16 range"
        )


@dataclass(frozen=True)
class TokenCacheInfo:
    split: str
    token_path: Path
    metadata_path: Path
    token_count: int
    source_sha256: str
    tokenizer_sha256: str
    encoded_bytes: int


def encode_token_cache(
    *,
    input_path: str | Path,
    tokenizer_path: str | Path,
    output_root: str | Path,
    split: str,
    max_bytes: int | None = None,
    chunk_tokens: int = 65536,
    data_version: str = "OWT-SAMPLE-v1",
    force: bool = False,
) -> TokenCacheInfo:
    """Stream a UTF-8 text split into a hash-bound little-endian uint16 cache."""
    if not split or "/" in split or "\\" in split or split in {".", ".."}:
        raise ValueError("split must be a non-empty path component")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")

    source = Path(input_path)
    tokenizer_file = Path(tokenizer_path)
    if not source.is_file():
        raise FileNotFoundError(f"source file does not exist: {source}")
    if not tokenizer_file.is_file():
        raise FileNotFoundError(f"tokenizer artifact does not exist: {tokenizer_file}")

    tokenizer = BPETokenizer.load(tokenizer_file)
    _validate_tokenizer_dtype(tokenizer)
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    token_path = output_dir / f"{split}.tokens.uint16"
    metadata_path = output_dir / f"{split}.metadata.json"
    if (token_path.exists() or metadata_path.exists()) and not force:
        raise FileExistsError(
            f"cache files already exist for split {split}; use --force to overwrite"
        )

    temp_token_path = token_path.with_suffix(token_path.suffix + ".tmp")
    metadata_temp_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    source_sha256 = sha256_file(source)
    tokenizer_sha256 = sha256_file(tokenizer_file)
    token_count = 0
    encoded_bytes = 0
    pending_ids: list[int] = []
    try:
        with temp_token_path.open("wb") as output:
            for line, line_bytes in _iter_lines_prefix(source, max_bytes):
                pending_ids.extend(tokenizer.encode(line))
                encoded_bytes += line_bytes
                if len(pending_ids) >= chunk_tokens:
                    token_count += _write_uint16_chunk(output, pending_ids)
                    pending_ids = []
            token_count += _write_uint16_chunk(output, pending_ids)

        metadata: dict[str, Any] = {
            "metadata_type": "llm-lab-token-cache-metadata",
            "metadata_version": CACHE_METADATA_VERSION,
            "cache": {
                "split": split,
                "token_file": token_path.name,
                "token_file_sha256": sha256_file(temp_token_path),
                "token_count": token_count,
                "dtype": TOKEN_DTYPE,
                "endianness": "little",
            },
            "source": {
                "path": str(source),
                "sha256": source_sha256,
                "size_bytes": source.stat().st_size,
                "encoded_bytes": encoded_bytes,
                "requested_max_bytes": max_bytes,
                "newline_aligned_prefix": True,
                "encoding": "utf-8",
            },
            "tokenizer": {
                "path": str(tokenizer_file),
                "artifact_sha256": tokenizer_sha256,
                "vocab_size": tokenizer.vocab_size,
                "special_token_ids": tokenizer.special_token_ids,
                "special_tokens": list(tokenizer.special_tokens),
            },
            "data_version": data_version,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        metadata["config_sha256"] = canonical_json_hash(
            {
                "dtype": TOKEN_DTYPE,
                "endianness": "little",
                "split": split,
                "requested_max_bytes": max_bytes,
                "tokenizer_artifact_sha256": tokenizer_sha256,
            }
        )
        metadata_temp_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(metadata_temp_path, metadata_path)
        temp_token_path.replace(token_path)
    except BaseException:
        temp_token_path.unlink(missing_ok=True)
        metadata_temp_path.unlink(missing_ok=True)
        token_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        raise

    return TokenCacheInfo(
        split=split,
        token_path=token_path,
        metadata_path=metadata_path,
        token_count=token_count,
        source_sha256=source_sha256,
        tokenizer_sha256=tokenizer_sha256,
        encoded_bytes=encoded_bytes,
    )


def validate_token_cache(
    *,
    token_path: str | Path,
    metadata_path: str | Path,
    source_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    split: str | None = None,
    max_bytes: int | None = None,
) -> TokenCacheInfo:
    """Validate cache bytes and optionally validate source/tokenizer bindings."""
    token_file = Path(token_path)
    metadata_file = Path(metadata_path)
    metadata = _load_json(metadata_file)
    cache = metadata.get("cache", {})
    source = metadata.get("source", {})
    tokenizer = metadata.get("tokenizer", {})
    if metadata.get("metadata_type") != "llm-lab-token-cache-metadata":
        raise ValueError("unsupported token cache metadata type")
    if metadata.get("metadata_version") != CACHE_METADATA_VERSION:
        raise ValueError("unsupported token cache metadata version")
    actual_split = cache.get("split")
    if not isinstance(actual_split, str):
        raise ValueError("cache metadata has no valid split")
    if split is not None and actual_split != split:
        raise ValueError(f"split mismatch: expected {split}, found {actual_split}")
    if cache.get("dtype") != TOKEN_DTYPE or cache.get("endianness") != "little":
        raise ValueError("unsupported token cache dtype or endianness")
    if not token_file.is_file():
        raise FileNotFoundError(f"token cache does not exist: {token_file}")
    if sha256_file(token_file) != cache.get("token_file_sha256"):
        raise ValueError("token cache hash mismatch")
    token_count = cache.get("token_count")
    if not isinstance(token_count, int) or token_count < 0:
        raise ValueError("invalid token count in cache metadata")
    if token_file.stat().st_size != token_count * 2:
        raise ValueError("token cache byte size does not match token count")

    actual_source = Path(source_path) if source_path is not None else None
    if actual_source is not None:
        if not actual_source.is_file():
            raise FileNotFoundError(f"source file does not exist: {actual_source}")
        if source.get("encoding") != "utf-8":
            raise ValueError("source encoding metadata mismatch")
        if source.get("newline_aligned_prefix") is not True:
            raise ValueError("source newline alignment metadata mismatch")
        if source.get("size_bytes") != actual_source.stat().st_size:
            raise ValueError("source file size mismatch")
        if sha256_file(actual_source) != source.get("sha256"):
            raise ValueError("source file hash mismatch")
        if max_bytes != source.get("requested_max_bytes"):
            raise ValueError("source byte limit mismatch")
        if source.get("encoded_bytes", 0) > actual_source.stat().st_size:
            raise ValueError("encoded source bytes exceed source file size")

    actual_tokenizer = Path(tokenizer_path) if tokenizer_path is not None else None
    if actual_tokenizer is not None:
        if not actual_tokenizer.is_file():
            raise FileNotFoundError(f"tokenizer artifact does not exist: {actual_tokenizer}")
        if sha256_file(actual_tokenizer) != tokenizer.get("artifact_sha256"):
            raise ValueError("tokenizer artifact hash mismatch")
        loaded = BPETokenizer.load(actual_tokenizer)
        if loaded.vocab_size != tokenizer.get("vocab_size"):
            raise ValueError("tokenizer vocab size mismatch")
        if list(loaded.special_tokens) != tokenizer.get("special_tokens"):
            raise ValueError("tokenizer special tokens mismatch")
        if loaded.special_token_ids != tokenizer.get("special_token_ids"):
            raise ValueError("tokenizer special token IDs mismatch")

    return TokenCacheInfo(
        split=actual_split,
        token_path=token_file,
        metadata_path=metadata_file,
        token_count=token_count,
        source_sha256=str(source.get("sha256", "")),
        tokenizer_sha256=str(tokenizer.get("artifact_sha256", "")),
        encoded_bytes=int(source.get("encoded_bytes", 0)),
    )
