"""One-shot MANIFEST rebuild helper.

Re-emits ``datasets/tool-calling-d1-llm/MANIFEST-train.json`` with the
schema ``generate_d1_llm.py`` now produces (per-entry ``source``,
``count`` = total on-disk, ``sources`` = sorted unique).

Usage:
    .venv/python.exe scripts/_rebuild_d1_manifest.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

D1LLM = ROOT / "datasets" / "tool-calling-d1-llm"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    train = D1LLM / "train"
    files = sorted(train.glob("d1llm-*.json"))
    entries: list[dict] = []
    for path in files:
        sample = json.loads(path.read_text(encoding="utf-8"))
        entries.append({
            "path": f"train/{path.name}",
            "sha256": sha256_file(path),
            "split": "train",
            "task_type": sample["metadata"]["task_type"],
            "source": sample["metadata"]["source"],
        })
    aggregate = hashlib.sha256()
    for e in entries:
        aggregate.update(e["sha256"].encode("utf-8"))
        aggregate.update(b"\x00")
        aggregate.update(e["source"].encode("utf-8"))
        aggregate.update(b"\x00")
    sources = sorted({e["source"] for e in entries})
    manifest = {
        "data_version": "D1.1",
        "pipeline": "d1.1-llm-generator",
        "count": len(entries),
        "split": "train",
        "aggregate_sha256": aggregate.hexdigest(),
        "sources": sources,
        "samples": entries,
    }
    out = D1LLM / "MANIFEST-train.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"wrote {out} (count={len(entries)}, sources={sources})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())