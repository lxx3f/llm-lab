"""Build and validate the D0 tool-calling manifest.

Scans ``examples/tool_calling/*.json``, validates each sample against
``schemas/tool_calling_sample.schema.json``, verifies the D0 protocol
requirements (source == "synthetic", task_type coverage), computes per-file
SHA-256 and the aggregate samples hash, and writes ``MANIFEST.json``.

Usage:
    .venv/python.exe scripts/build_d0_manifest.py
"""

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

from jsonschema import Draft202012Validator, FormatChecker  # noqa: E402

SAMPLES_DIR = ROOT / "examples" / "tool_calling"
SCHEMA_PATH = ROOT / "schemas" / "tool_calling_sample.schema.json"
MANIFEST_PATH = SAMPLES_DIR / "MANIFEST.json"
REQUIRED_TASK_TYPES = {"no_tool", "single_tool", "multi_tool"}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def build_manifest() -> dict[str, Any]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    sample_files = sorted(p for p in SAMPLES_DIR.glob("*.json") if p.name != "MANIFEST.json")
    samples: list[dict[str, Any]] = []
    seen_task_types: set[str] = set()
    for sample_path in sample_files:
        payload = json.loads(sample_path.read_text(encoding="utf-8"))
        errors = list(validator.iter_errors(payload))
        if errors:
            raise ValueError(f"{sample_path.name}: schema errors: {errors}")
        source = (payload.get("metadata") or {}).get("source")
        if source != "synthetic":
            raise ValueError(f"{sample_path.name}: metadata.source must be 'synthetic', got {source!r}")
        task_type = (payload.get("metadata") or {}).get("task_type")
        seen_task_types.add(task_type)
        samples.append({
            "path": sample_path.name,
            "sha256": _sha256_file(sample_path),
            "task_type": task_type,
        })
    missing = REQUIRED_TASK_TYPES - seen_task_types
    if missing:
        raise ValueError(f"D0 must cover task types {sorted(REQUIRED_TASK_TYPES)}; missing {sorted(missing)}")

    aggregate = _sha256_bytes("".join(s["sha256"] for s in samples).encode("ascii"))
    manifest = {
        "schema_version": "1.0",
        "data_version": "D0",
        "split": "none",
        "samples": samples,
        "samples_sha256": aggregate,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write MANIFEST.json (default: validate only)")
    args = parser.parse_args()
    manifest = build_manifest()
    if args.write:
        MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[build_d0_manifest] wrote {MANIFEST_PATH.relative_to(ROOT)}")
    else:
        print(f"[build_d0_manifest] D0 manifest valid: {len(manifest['samples'])} samples, "
              f"samples_sha256={manifest['samples_sha256'][:12]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())