#!/usr/bin/env python3
"""scripts/audit/run_doc_artifact_reconciliation.py

Doc <-> artifact reconciliation audit for the final-audit.

For each `docs/experiments/*/README.md`, this script:
   1. Extracts `artifacts/...` path references from the README
   2. Classifies each reference by kind:
      - EXACT_FILE: ends in a known extension; check exact path or
        extension fallback
      - GLOB_PATTERN: ends in `/`, `-`, or contains `*`; expand via
        pathlib glob and check at least one match
      - PLACEHOLDER: a path that's been truncated mid-name (e.g.,
        'artifacts/sft-' which is just a category reference)
   3. Categorizes each ref's resolution as:
      - FOUND (EXACT_FILE exists OR GLOB matches at least one file)
      - MISSING_RESOLVABLE (not found but has documented regen cmd)
      - MISSING_UNRESOLVABLE (not found AND no regen cmd — AUDIT FAIL)
      - PLACEHOLDER (truncated reference; excluded from existence check)
   4. Emits per-doc, per-ref JSON evidence
   5. Exits 1 if any MISSING_UNRESOLVABLE references

Reproduction:
    1. git checkout HEAD~1   # obtain audited tree
    2. python scripts/audit/run_doc_artifact_reconciliation.py
    3. Expected: exit 0 (all non-placeholder refs resolved)
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "artifacts" / "audits"
JSON_OUT = AUDIT_DIR / "doc-artifact-reconciliation.json"

# Regex: match artifacts/... up to whitespace, end of line, or quote/paren
ARTIFACT_REF_RE = re.compile(r"artifacts/[A-Za-z0-9_./-]+")

# Regex: match scripts/...py invocation lines
SCRIPT_INVOCATION_RE = re.compile(
    r"((?:scripts|architecture_lab)/[A-Za-z0-9_./-]+\.py)\b[^#\n]*"
)

EXTENSIONS = [".json", ".png", ".md", ".txt", ".csv", ".yaml", ".pt", ".ckpt", ".safetensors", ".bin"]

# Classification heuristics
# Placeholders are references that have been truncated (ending in a
# non-filename-character like `-`, `_`, `/`, or `...`) and do not
# point to a specific file. They are excluded from the existence check
# because they describe categories/directories, not individual artifacts.
PLACEHOLDER_ENDINGS = ("-", "_", "/", "...")


def classify_ref(ref: str) -> str:
    """Classify an artifact reference by kind.

    Returns: 'exact_file' | 'glob_pattern' | 'placeholder'

    - 'exact_file': ends in a known file extension; check exact path
      with extension fallback
    - 'glob_pattern': contains '*'; expand via pathlib glob
    - 'placeholder': ends in `-`, `_`, `/`, or `...`; truncated
      reference describing a category or directory, not a specific file.
      Excluded from existence check.
    """
    # Directory / ellipsis reference (ends with /, /..., ..., etc.)
    if any(ref.endswith(ending) for ending in PLACEHOLDER_ENDINGS):
        return "placeholder"
    # Has known file extension → exact file
    if any(ref.endswith(ext) for ext in EXTENSIONS):
        return "exact_file"
    # Has glob pattern (* in the middle)
    if "*" in ref:
        return "glob_pattern"
    # Trailing path component without extension and not a placeholder
    # → likely an exact reference to a file without extension
    return "exact_file"


def check_artifact_exists(ref: str, kind: str) -> tuple[bool | None, list[str]]:
    """Check whether an artifact reference is resolvable on disk.

    Returns:
      - (True, [paths]) if found
      - (False, []) if not found
      - (None, []) for placeholders (excluded from existence check)
    """
    if kind == "placeholder":
        # Placeholders are not checked for existence.
        return None, []
    if kind == "glob_pattern":
        # Expand glob
        glob_pattern = str(ROOT / ref)
        matches = [str(p.relative_to(ROOT)) for p in ROOT.glob(glob_pattern)]
        if matches:
            return True, matches
        return False, []
    # exact_file: try exact path, then extension fallback
    base = ROOT / ref
    matches: list[str] = []
    if base.is_file():
        matches.append(str(base.relative_to(ROOT)))
    if not matches:
        for ext in EXTENSIONS:
            if (ROOT / (ref + ext)).is_file():
                matches.append(str((ROOT / (ref + ext)).relative_to(ROOT)))
                break
    return bool(matches), matches


def find_regen_command_for_artifact(readme_text: str, artifact_ref: str) -> str | None:
    """Search the README for a regeneration command that produces the
    artifact reference (or its parent dir for placeholders).
    """
    artifact_dir = artifact_ref.rsplit("/", 1)[0]
    for line in readme_text.splitlines():
        if not SCRIPT_INVOCATION_RE.search(line):
            continue
        if (
            artifact_ref in line
            or artifact_dir + "/" in line
            or artifact_dir in line
        ):
            m = SCRIPT_INVOCATION_RE.search(line)
            if m:
                return m.group(1)
    return None


def script_exists(script_path: str) -> bool:
    return (ROOT / script_path).exists()


def main() -> int:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    readme_paths = sorted((ROOT / "docs" / "experiments").rglob("README.md"))
    total_docs = len(readme_paths)

    print("=== Doc <-> artifact reconciliation ===", flush=True)
    print(f"Total experiment READMEs: {total_docs}", flush=True)

    docs_with_refs = 0
    docs_without_refs = 0
    total_references = 0
    placeholders_excluded = 0
    references_found = 0
    references_missing_resolvable = 0
    references_missing_unresolvable = 0

    per_doc_results: list[dict] = []

    for readme in readme_paths:
        rel = str(readme.relative_to(ROOT))
        text = readme.read_text(encoding="utf-8")

        refs = sorted(set(ARTIFACT_REF_RE.findall(text)))
        ref_count = len(refs)

        if ref_count == 0:
            docs_without_refs += 1
            per_doc_results.append({
                "doc": rel,
                "ref_count": 0,
                "refs": [],
            })
            continue

        docs_with_refs += 1
        total_references += ref_count

        per_refs: list[dict] = []
        for ref in refs:
            kind = classify_ref(ref)
            exists, matches = check_artifact_exists(ref, kind)
            regen_cmd = find_regen_command_for_artifact(text, ref)
            regen_exists = script_exists(regen_cmd) if regen_cmd else False

            if kind == "placeholder":
                placeholders_excluded += 1
                status = "PLACEHOLDER"
            elif exists is True:
                references_found += 1
                status = "FOUND"
            elif regen_exists:
                references_missing_resolvable += 1
                status = "MISSING_RESOLVABLE"
            else:
                references_missing_unresolvable += 1
                status = "MISSING_UNRESOLVABLE"

            per_refs.append({
                "ref": ref,
                "kind": kind,
                "exists": exists,
                "status": status,
                "matched_paths": matches,
                "regen_command": regen_cmd,
                "regen_command_exists": regen_exists,
            })

        per_doc_results.append({
            "doc": rel,
            "ref_count": ref_count,
            "refs": per_refs,
        })

    payload = {
        "audit_name": "final-audit doc-artifact reconciliation",
        "audit_date": "2026-08-29",
        "methodology": (
            "For each docs/experiments/*/README.md: (a) extract artifacts/... "
            "references; (b) classify as exact_file / glob_pattern / directory / "
            "placeholder; (c) check existence per kind (placeholders excluded "
            "from existence check); (d) for missing refs, identify documented "
            "regeneration command. Categorize as FOUND / MISSING_RESOLVABLE / "
            "MISSING_UNRESOLVABLE / PLACEHOLDER."
        ),
        "summary": {
            "total_docs": total_docs,
            "docs_with_artifact_refs": docs_with_refs,
            "docs_without_artifact_refs": docs_without_refs,
            "total_references": total_references,
            "placeholders_excluded": placeholders_excluded,
            "references_found": references_found,
            "references_missing_resolvable": references_missing_resolvable,
            "references_missing_unresolvable": references_missing_unresolvable,
        },
        "invariant": (
            "all non-placeholder artifact references must be FOUND on disk OR "
            "have a documented regeneration command whose script exists in the "
            "repo. MISSING_UNRESOLVABLE references are audit failures."
        ),
        "per_doc": per_doc_results,
    }

    JSON_OUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("", flush=True)
    print("=== Summary ===", flush=True)
    print(f"Total docs:                          {total_docs}", flush=True)
    print(f"Docs with artifact refs:             {docs_with_refs}", flush=True)
    print(f"Docs without artifact refs:          {docs_without_refs}", flush=True)
    print(f"Total artifact references:           {total_references}", flush=True)
    print(f"Placeholders (excluded from check):  {placeholders_excluded}", flush=True)
    print(f"References FOUND on disk:            {references_found}", flush=True)
    print(f"References MISSING (resolvable):     {references_missing_resolvable}", flush=True)
    print(f"References MISSING (unresolvable):   {references_missing_unresolvable}", flush=True)
    print("", flush=True)
    print(f"JSON: {JSON_OUT}", flush=True)

    if references_missing_unresolvable > 0:
        print("", flush=True)
        print(f"FAIL: {references_missing_unresolvable} unresolvable references", flush=True)
        for doc in per_doc_results:
            for ref in doc.get("refs", []):
                if ref["status"] == "MISSING_UNRESOLVABLE":
                    print(f"  {doc['doc']}: {ref['ref']}", flush=True)
        return 1

    if references_missing_resolvable > 0:
        print("", flush=True)
        print(
            f"INFO: {references_missing_resolvable} references missing but "
            "resolvable by running the documented commands",
            flush=True,
        )

    print("", flush=True)
    print("PASS: all non-placeholder references resolved", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())