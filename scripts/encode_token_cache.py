"""Encode OWT text splits into hash-bound token caches.

The default mode preserves the project's uint16 BPE cache contract. The
``--hf-model-dir`` mode is for public Hugging Face/ModelScope model
tokenizers and writes an int32 cache because Qwen vocabularies exceed uint16.
Both modes stream newline-aligned source text and record source/cache hashes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hashlib
import json
import time
from typing import Any

import numpy as np

from architecture_lab.data.token_cache import encode_token_cache  # noqa: E402


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _encode_hf_token_cache(
    *,
    input_path: Path,
    model_dir: Path,
    output_root: Path,
    split: str,
    max_bytes: int | None,
    chunk_tokens: int,
    data_version: str,
    force: bool,
    model_id: str,
    tokenizer_revision: str,
    hf_expected_revision: str | None,
    expected_source_sha256: str | None,
) -> dict[str, Any]:
    """Stream a local HF tokenizer into an int32 hash-bound cache."""
    from transformers import AutoTokenizer

    if not input_path.is_file():
        raise FileNotFoundError(f"source file does not exist: {input_path}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"HF model directory does not exist: {model_dir}")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if max_bytes is not None and max_bytes <= 0:
        raise ValueError("max_bytes must be positive when provided")

    source_sha256 = _sha256_file(input_path)
    if expected_source_sha256 and source_sha256 != expected_source_sha256:
        raise ValueError(
            f"source SHA mismatch: expected {expected_source_sha256}, got {source_sha256}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    token_path = output_root / f"{split}.tokens.int32"
    metadata_path = output_root / f"{split}.metadata.json"
    if (token_path.exists() or metadata_path.exists()) and not force:
        raise FileExistsError(
            f"cache files already exist for split {split}; use --force to overwrite"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    pending: list[int] = []
    token_count = 0
    encoded_bytes = 0
    temp_token_path = token_path.with_suffix(token_path.suffix + ".tmp")
    temp_metadata_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")

    def flush(output) -> None:
        nonlocal token_count, pending
        if not pending:
            return
        ids = np.asarray(pending, dtype=np.int64)
        if ids.size and (int(ids.min()) < 0 or int(ids.max()) > np.iinfo(np.int32).max):
            raise ValueError("token id cannot be stored as int32")
        output.write(ids.astype("<i4", copy=False).tobytes())
        token_count += int(ids.size)
        pending = []

    try:
        with input_path.open("r", encoding="utf-8", newline="") as source, temp_token_path.open("wb") as output:
            for line in source:
                line_bytes = len(line.encode("utf-8"))
                if max_bytes is not None and encoded_bytes + line_bytes > max_bytes:
                    break
                pending.extend(tokenizer.encode(line, add_special_tokens=False))
                encoded_bytes += line_bytes
                if len(pending) >= chunk_tokens:
                    flush(output)
            flush(output)

        metadata = {
            "schema_version": "1.0",
            "cache_path": str(token_path),
            "metadata_path": str(metadata_path),
            "model_id": model_id,
            "model_short": output_root.name,
            "tokenizer_revision": tokenizer_revision,
            "local_snapshot_revision": tokenizer_revision,
            "hf_expected_revision": hf_expected_revision,
            "source_path": str(input_path),
            "source_sha256": source_sha256,
            "source_size_bytes": input_path.stat().st_size,
            "encoded_bytes": encoded_bytes,
            "max_bytes": max_bytes,
            "newline_aligned": True,
            "encoded_tokens": token_count,
            "dtype": "int32",
            "endianness": "little",
            "cache_sha256": _sha256_file(temp_token_path),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        metadata_blob = json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8")
        metadata["metadata_sha256"] = hashlib.sha256(metadata_blob).hexdigest()
        temp_metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temp_token_path.replace(token_path)
        temp_metadata_path.replace(metadata_path)
    except BaseException:
        temp_token_path.unlink(missing_ok=True)
        temp_metadata_path.unlink(missing_ok=True)
        raise
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="UTF-8 source text")
    parser.add_argument("--tokenizer", type=Path, help="Project tokenizer.json (default mode)")
    parser.add_argument(
        "--hf-model-dir",
        type=Path,
        help="Local HF/ModelScope snapshot containing tokenizer files; writes int32 cache",
    )
    parser.add_argument("--model-id", default="local/model", help="Model ID recorded in HF cache metadata")
    parser.add_argument("--tokenizer-revision", default="unknown", help="Actual local tokenizer snapshot revision recorded in HF cache metadata")
    parser.add_argument("--hf-expected-revision", help="Expected Hugging Face commit recorded separately for provenance")
    parser.add_argument("--expected-source-sha256", help="Reject source unless it has this SHA-256")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", required=True, help="Cache split name, e.g. train or validation")
    parser.add_argument("--max-bytes", type=int, help="Newline-aligned source prefix limit")
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=65536,
        help="Number of token IDs buffered before writing",
    )
    parser.add_argument("--data-version", default="OWT-SAMPLE-v1")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (args.tokenizer is None) == (args.hf_model_dir is None):
        print(
            "error: specify exactly one of --tokenizer or --hf-model-dir",
            file=sys.stderr,
        )
        return 2
    try:
        if args.hf_model_dir is not None:
            info = _encode_hf_token_cache(
                input_path=args.input,
                model_dir=args.hf_model_dir,
                output_root=args.output_root,
                split=args.split,
                max_bytes=args.max_bytes,
                chunk_tokens=args.chunk_tokens,
                data_version=args.data_version,
                force=args.force,
                model_id=args.model_id,
                tokenizer_revision=args.tokenizer_revision,
                hf_expected_revision=args.hf_expected_revision,
                expected_source_sha256=args.expected_source_sha256,
            )
            print(f"tokens: {info['cache_path']}")
            print(f"metadata: {info['metadata_path']}")
            print(f"token_count: {info['encoded_tokens']}")
            print(f"encoded_bytes: {info['encoded_bytes']}")
            return 0
        info = encode_token_cache(
            input_path=args.input,
            tokenizer_path=args.tokenizer,
            output_root=args.output_root,
            split=args.split,
            max_bytes=args.max_bytes,
            chunk_tokens=args.chunk_tokens,
            data_version=args.data_version,
            force=args.force,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"tokens: {info.token_path}")
    print(f"metadata: {info.metadata_path}")
    print(f"token_count: {info.token_count}")
    print(f"encoded_bytes: {info.encoded_bytes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
