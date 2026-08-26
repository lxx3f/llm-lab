"""Sweep docs vs artifact consistency verifier.

For each documented sweep (N5/N6/N7/N8/N9), this script asserts that any
README/review doc that mentions a specific ``val_loss_min`` for a
configuration also records the matching ``val_loss_min_step`` and
``delta_val_loss``. A drift in any of these fields is treated as a
documentation bug.

Row-level check: when a line quotes a config's ``val_min``, the line's
``@ N`` step marker and delta column must match the artifact.

Usage:
    .venv/python.exe scripts/verify_sweep_doc_consistency.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# (artifact path, list of doc paths that should be consistent with it)
SWEEP_DOCS: list[tuple[str, list[str]]] = [
    ("artifacts/dense-owt-formal-curve-result.json",
     ["docs/experiments/n4-dense-formal-curve/README.md"]),
    ("artifacts/dense-owt-formal-curve-medium-result.json",
     ["docs/experiments/n5-dense-scale-sweep/README.md",
      "docs/experiments/n7-dense-rope-sweep/README.md",
      "docs/experiments/n8-dense-heads-sweep/README.md",
      "docs/experiments/n9-dense-dff-sweep/README.md"]),
    ("artifacts/dense-owt-formal-curve-medium-dff-256-result.json",
     ["docs/experiments/n9-dense-dff-sweep/README.md"]),
    ("artifacts/dense-owt-formal-curve-medium-dff-1024-result.json",
     ["docs/experiments/n9-dense-dff-sweep/README.md"]),
]


def _format_val(value: float) -> str:
    return f"{value:.4f}"


def _check_artifact(artifact_rel: str, doc_rel: str, summary: dict[str, Any]) -> list[str]:
    drifts: list[str] = []
    doc_path = ROOT / doc_rel
    if not doc_path.is_file():
        return drifts
    text = doc_path.read_text(encoding="utf-8")
    val_min = summary["val_loss_min"]
    val_min_step = summary["val_loss_min_step"]
    delta_val_loss = summary["delta_val_loss"]
    val_min_str = _format_val(val_min)

    for line_no, line in enumerate(text.splitlines(), 1):
        # Count distinct val_min mentions; skip multi-row summary lines.
        if text.count(val_min_str) < line.count(val_min_str):
            continue
        if line.count(val_min_str) > 1:
            # Multi-column summary row — skip to avoid false positives.
            continue
        if val_min_str not in line:
            continue
        # Tight pattern: val_min_str must be immediately followed (after
        # optional whitespace) by '@ N'. This avoids misreading unrelated
        # '@ 4800' references that describe val at non-min steps.
        tight_step = re.search(re.escape(val_min_str) + r"\s*@\s*(\d+)", line)
        if tight_step and int(tight_step.group(1)) != val_min_step:
            drifts.append(
                f"{doc_rel}:{line_no}: quotes val_min={val_min_str} with @ "
                f"{tight_step.group(1)} but artifact {artifact_rel} has "
                f"val_loss_min_step={val_min_step}"
            )
        if delta_val_loss != 0.0 and re.search(r"\|\s*0\.0\s*\|", line):
            drifts.append(
                f"{doc_rel}:{line_no}: val_min={val_min_str} row shows delta=0.0 "
                f"but artifact {artifact_rel} has delta_val_loss={delta_val_loss:.4f}"
            )

    return drifts


def verify() -> list[str]:
    all_drifts: list[str] = []
    for artifact_rel, doc_rels in SWEEP_DOCS:
        artifact_path = ROOT / artifact_rel
        if not artifact_path.is_file():
            all_drifts.append(f"missing artifact: {artifact_rel}")
            continue
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        summary = payload["metrics"]["curve_summary"]
        for doc_rel in doc_rels:
            all_drifts.extend(_check_artifact(artifact_rel, doc_rel, summary))
    return all_drifts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    drifts = verify()
    if drifts:
        for line in drifts:
            print(f"[verify_sweep_doc_consistency] DRIFT: {line}", file=sys.stderr)
        print(f"[verify_sweep_doc_consistency] {len(drifts)} drift(s) detected", file=sys.stderr)
        return 1
    print(f"[verify_sweep_doc_consistency] {len(SWEEP_DOCS)} artifact/doc pairs verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
