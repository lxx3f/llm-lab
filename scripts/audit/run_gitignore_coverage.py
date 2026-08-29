#!/usr/bin/env python3
"""scripts/audit/run_gitignore_coverage.py

Bounded, batched per-file .gitignore coverage audit for the final-audit.

Uses `git check-ignore --no-index --stdin --verbose --non-matching` (one
subprocess for ALL files, batched via stdin) instead of one subprocess
per file. This is fast enough to complete within the 120-second audit
window even on large artifact trees (7048+ files).

Output format from git:
    .gitignore:<linenum>:<pattern>\t<path>      # ignored (rule matched)
    ::\t<path>                                  # not ignored

Writes machine-readable JSON to artifacts/audits/gitignore-coverage.json
and exits 0 if Not ignored == 0, exits 1 otherwise.

Reproduction (non-mutating; no git checkout):
    1. Confirm clean main: git status --short  # → empty
    2. python scripts/audit/run_gitignore_coverage.py
    3. Expected: exit 0, JSON written, "Not ignored: 0"
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "artifacts" / "audits"
JSON_OUT = AUDIT_DIR / "gitignore-coverage.json"

SENSITIVE_PATHS = (
    "datasets/tool-calling-d1",
    "datasets/tool-calling-d1-llm",
    "datasets/tool-calling-d2",
    "artifacts",
    ".tmp",
)


def collect_files() -> list[Path]:
    files: list[Path] = []
    for rel in SENSITIVE_PATHS:
        abs_dir = ROOT / rel
        if not abs_dir.exists():
            continue
        for p in abs_dir.rglob("*"):
            if p.is_file():
                files.append(p)
    return files


def _normalize_path(p: str) -> str:
    """Normalize git check-ignore output path.

    git on MINGW outputs paths as `<source>\t"<path>\\r"\n` where:
      - The path is wrapped in quotes (sometimes)
      - The trailing `\\r` is a literal backslash+r (not a CR byte) used
        by git to indicate end-of-record
      - A newline terminates the line
    This function strips all of those and converts backslashes to forward
    slashes for portable matching.
    """
    s = p.strip()
    # Strip surrounding quotes
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    # Strip git's end-of-record marker (literal \r in source) + trailing
    # newline that may have survived .strip()
    while s.endswith("\\r"):
        s = s[:-2]
    s = s.rstrip("\n").rstrip("\r")
    # Convert backslashes (from Windows paths) to forward slashes
    s = s.replace("\\", "/")
    return s


def batch_check_ignore(files: list[Path]) -> tuple[set[str], set[str], list[str]]:
    """Run git check-ignore --stdin --verbose --non-matching on all files.

    Returns (ignored_set, not_ignored_set, errors).
    """
    rel_paths = [_normalize_path(str(f.relative_to(ROOT))) for f in files]
    stdin_data = ("\n".join(rel_paths) + "\n").encode("utf-8")

    proc = subprocess.run(
        [
            "git", "check-ignore",
            "--no-index",
            "--stdin",
            "--verbose",
            "--non-matching",
        ],
        cwd=str(ROOT),
        input=stdin_data,
        capture_output=True,
        timeout=180,
    )

    ignored_set: set[str] = set()
    not_ignored_set: set[str] = set()
    for line in proc.stdout.decode("utf-8", errors="replace").split("\n"):
        # Format: <source>:<linenum>:<pattern>\t<path>
        # or:      ::\t<path>  (non-matching)
        # On MINGW the path may be wrapped in quotes by git and have a
        # trailing \\r (literal backslash-r, not CR).
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        path = _normalize_path(parts[1])
        source = parts[0].strip()
        if source == "::":
            not_ignored_set.add(path)
        else:
            ignored_set.add(path)

    errors: list[str] = []
    if proc.returncode not in (0, 1):
        errors.append(f"git check-ignore returned {proc.returncode}")
    if proc.stderr.strip():
        errors.append(proc.stderr.strip().decode("utf-8", errors="replace"))

    return ignored_set, not_ignored_set, errors


def main() -> int:
    files = collect_files()
    total = len(files)

    print("=== Bounded per-file .gitignore audit ===", flush=True)
    print(f"Total sensitive files: {total}", flush=True)

    if total == 0:
        print("ERROR: no sensitive files found (something is wrong)", flush=True)
        return 2

    ignored_set, not_ignored_set, errors = batch_check_ignore(files)

    # Semantic-coverage check: verify that arbitrary datasets/ paths are
    # ignored by the broad datasets/ rule (catches future unlisted datasets).
    semantic_test_paths = [
        "datasets/unlisted-dataset/train.json",
        "datasets/unlisted-dataset/train.parquet",
        "datasets/future-experiment/data.csv",
    ]
    # Create temporary files for the test, then clean up.
    created_tmp: list[Path] = []
    for tp in semantic_test_paths:
        p = ROOT / tp
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        created_tmp.append(p)
    semantic_ignored, semantic_not_ignored, _ = batch_check_ignore(
        [ROOT / tp for tp in semantic_test_paths]
    )
    semantic_ignored_norm = {_normalize_path(p) for p in semantic_ignored}
    semantic_not_ignored_norm = {_normalize_path(p) for p in semantic_not_ignored}
    semantic_passes = all(
        tp in semantic_ignored_norm for tp in semantic_test_paths
    ) and not any(tp in semantic_not_ignored_norm for tp in semantic_test_paths)
    # Clean up
    for p in created_tmp:
        if p.exists():
            p.unlink()
        if p.parent.exists() and not any(p.parent.iterdir()):
            p.parent.rmdir()

    # Build a lookup with multiple path normalizations
    ignored: list[Path] = []
    not_ignored: list[Path] = []
    unmatched: list[Path] = []
    for f in files:
        rel_norm = _normalize_path(str(f.relative_to(ROOT)))
        if rel_norm in ignored_set:
            ignored.append(f)
        elif rel_norm in not_ignored_set:
            not_ignored.append(f)
        else:
            # File did not appear in either set — this is an audit error.
            unmatched.append(f)

    if unmatched:
        # Treat unmatched files as not ignored (safer default for the
        # invariant; any unmatched file should be investigated).
        not_ignored.extend(unmatched)

    print(f"Gitignored: {len(ignored)}", flush=True)
    print(f"Not ignored: {len(not_ignored)}", flush=True)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    # Per-category breakdown
    category_breakdown: dict[str, dict[str, int]] = {}
    for f in files:
        rel = f.relative_to(ROOT).as_posix()
        cat = rel.split("/", 1)[0] if "/" in rel else rel
        if cat not in category_breakdown:
            category_breakdown[cat] = {"total": 0, "gitignored": 0, "not_ignored": 0}
        category_breakdown[cat]["total"] += 1
        if f in ignored:
            category_breakdown[cat]["gitignored"] += 1
        else:
            category_breakdown[cat]["not_ignored"] += 1

    payload = {
        "audit_name": "final-audit per-file .gitignore coverage (bounded)",
        "audit_date": "2026-08-29",
        "methodology": "git check-ignore --no-index --stdin --verbose --non-matching (single subprocess, all files batched via stdin)",
        "totals": {
            "total_sensitive_files": total,
            "gitignored_count": len(ignored),
            "not_ignored_count": len(not_ignored),
            "coverage_percent": (len(ignored) / total * 100.0) if total else 0.0,
        },
        "categories": {
            cat: category_breakdown[cat]
            for cat in sorted(category_breakdown)
        },
        "semantic_coverage_test": {
            "test_paths": semantic_test_paths,
            "all_ignored": semantic_passes,
            "note": "verifies that arbitrary unlisted datasets/* paths are also covered by the broad datasets/ rule",
        },
        "not_ignored_paths": [
            str(f.relative_to(ROOT)) for f in not_ignored[:100]
        ],
        "result": "PASS" if not not_ignored and semantic_passes else "FAIL",
        "errors": errors,
    }

    JSON_OUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if not_ignored or not semantic_passes:
        if not_ignored:
            print("", flush=True)
            print(f"NOT IGNORED ({len(not_ignored)}):", flush=True)
            for f in not_ignored[:20]:
                print(f"  {f.relative_to(ROOT)}", flush=True)
            if len(not_ignored) > 20:
                print(f"  ... and {len(not_ignored) - 20} more", flush=True)
        if not semantic_passes:
            print("", flush=True)
            print("SEMANTIC COVERAGE TEST FAILED: arbitrary datasets/* paths not all ignored", flush=True)
        print("", flush=True)
        print(f"FAIL: {len(not_ignored)}/{total} files not gitignored", flush=True)
        print(f"JSON: {JSON_OUT}", flush=True)
        return 1

    coverage_pct = len(ignored) / total * 100.0
    print("", flush=True)
    print(f"PASS: {len(ignored)}/{total} ({coverage_pct:.4f}%, 0 exceptions)", flush=True)
    print(f"JSON: {JSON_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())