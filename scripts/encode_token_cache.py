"""Encode OWT text splits into hash-bound uint16 token caches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture_lab.data.token_cache import encode_token_cache  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="UTF-8 source text")
    parser.add_argument("--tokenizer", type=Path, required=True, help="tokenizer.json")
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
    try:
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
