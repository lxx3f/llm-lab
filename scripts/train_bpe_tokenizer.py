"""Train and persist a versioned BPE tokenizer artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture_lab.tokenization import BPETokenizer, train_bpe_from_file, training_prefix_bytes  # noqa: E402


ARTIFACT_METADATA_VERSION = "1.0"
TOKENIZER_IMPLEMENTATION = "llm-lab-bpe-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def artifact_directory(root: Path, name: str, version: str) -> Path:
    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError("name must be a non-empty path component")
    if not version or "/" in version or "\\" in version or version in {".", ".."}:
        raise ValueError("version must be a non-empty path component")
    return root / name / version


def build_metadata(
    *,
    name: str,
    version: str,
    input_path: Path,
    vocab_size_requested: int,
    special_tokens: list[str],
    tokenizer: BPETokenizer,
    tokenizer_path: Path,
    source_path_for_metadata: str,
    data_version: str,
    license_name: str,
    source_kind: str,
    training_scope: dict[str, Any],
) -> dict[str, Any]:
    source_size = input_path.stat().st_size
    metadata: dict[str, Any] = {
        "metadata_type": "llm-lab-tokenizer-metadata",
        "metadata_version": ARTIFACT_METADATA_VERSION,
        "tokenizer": {
            "name": name,
            "version": version,
            "implementation": TOKENIZER_IMPLEMENTATION,
            "algorithm": "byte-level-bpe",
            "pretokenization": "gpt2-regex-v1",
            "vocab_size_requested": vocab_size_requested,
            "vocab_size_actual": tokenizer.vocab_size,
            "merge_count": len(tokenizer.merges),
            "special_tokens": special_tokens,
        },
        "source": {
            "path": source_path_for_metadata,
            "kind": source_kind,
            "license": license_name,
            "data_version": data_version,
            "sha256": sha256_file(input_path),
            "size_bytes": source_size,
            "encoding": "utf-8",
        },
        "training": {
            "input_path_is_used_only_for_metadata": False,
            "special_tokens_excluded_from_bpe_merges": True,
            **training_scope,
        },
        "artifacts": {
            "tokenizer_file": tokenizer_path.name,
            "tokenizer_sha256": sha256_file(tokenizer_path),
        },
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    metadata["config_sha256"] = sha256_json(
        {
            "implementation": metadata["tokenizer"]["implementation"],
            "algorithm": metadata["tokenizer"]["algorithm"],
            "pretokenization": metadata["tokenizer"]["pretokenization"],
            "vocab_size_requested": vocab_size_requested,
            "special_tokens": special_tokens,
        }
    )
    return metadata


def train_and_save(
    *,
    input_path: Path,
    artifact_root: Path,
    name: str,
    version: str,
    vocab_size: int,
    special_tokens: list[str],
    source_path_for_metadata: str | None = None,
    data_version: str = "unversioned",
    license_name: str = "unspecified",
    source_kind: str = "external-dataset",
    chunk_size_bytes: int = 8 * 1024 * 1024,
    max_training_bytes: int | None = None,
    force: bool = False,
) -> tuple[Path, Path]:
    if not input_path.is_file():
        raise FileNotFoundError(f"input file does not exist: {input_path}")
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size must fit 256 byte tokens and special tokens")

    output_dir = artifact_directory(artifact_root, name, version)
    tokenizer_path = output_dir / "tokenizer.json"
    metadata_path = output_dir / "metadata.json"
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(
            f"artifact directory is not empty: {output_dir}; use --force to overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = train_bpe_from_file(
        input_path,
        vocab_size,
        tuple(special_tokens),
        chunk_size_bytes=chunk_size_bytes,
        max_training_bytes=max_training_bytes,
    )
    tokenizer.save(tokenizer_path)
    prefix_bytes = training_prefix_bytes(input_path, max_training_bytes)
    metadata = build_metadata(
        name=name,
        version=version,
        input_path=input_path,
        vocab_size_requested=vocab_size,
        special_tokens=special_tokens,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        source_path_for_metadata=source_path_for_metadata or str(input_path),
        data_version=data_version,
        license_name=license_name,
        source_kind=source_kind,
        training_scope={
            "chunk_size_bytes": chunk_size_bytes,
            "max_training_bytes": max_training_bytes,
            "training_byte_limit": max_training_bytes,
            "training_bytes": prefix_bytes,
            "newline_aligned_prefix": True,
        },
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return tokenizer_path, metadata_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="UTF-8 training text")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=ROOT / "artifacts/tokenizers",
        help="Root directory for versioned tokenizer artifacts",
    )
    parser.add_argument("--name", required=True, help="Tokenizer artifact name")
    parser.add_argument("--version", required=True, help="Tokenizer artifact version")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument(
        "--special-token",
        dest="special_tokens",
        action="append",
        default=[],
        help="Repeat for each reserved special token",
    )
    parser.add_argument(
        "--source-path",
        help="Portable source path to record in metadata; defaults to --input",
    )
    parser.add_argument(
        "--data-version",
        default="unversioned",
        help="Version of the source data recorded in metadata",
    )
    parser.add_argument(
        "--license",
        dest="license_name",
        default="unspecified",
        help="License identifier for the source data",
    )
    parser.add_argument(
        "--source-kind",
        default="external-dataset",
        help="Source kind recorded in metadata",
    )
    parser.add_argument(
        "--chunk-size-bytes",
        type=int,
        default=8 * 1024 * 1024,
        help="Approximate newline-aligned input chunk size",
    )
    parser.add_argument(
        "--max-training-bytes",
        type=int,
        help="Train on a deterministic newline-aligned prefix only",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing artifact files")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        tokenizer_path, metadata_path = train_and_save(
            input_path=args.input,
            artifact_root=args.artifact_root,
            name=args.name,
            version=args.version,
            vocab_size=args.vocab_size,
            special_tokens=args.special_tokens,
            source_path_for_metadata=args.source_path,
            data_version=args.data_version,
            license_name=args.license_name,
            source_kind=args.source_kind,
            chunk_size_bytes=args.chunk_size_bytes,
            max_training_bytes=args.max_training_bytes,
            force=args.force,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"tokenizer: {tokenizer_path}")
    print(f"metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
