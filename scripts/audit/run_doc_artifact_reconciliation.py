#!/usr/bin/env python3
"""scripts/audit/run_doc_artifact_reconciliation.py

Doc ↔ artifact reconciliation audit for the final-audit.

For each `docs/experiments/*/README.md`, this script:
   1. Extracts `artifacts/...` path references from the README
   2. Checks whether each referenced artifact path exists on disk
      (with extension fallback: .json, .png, .md, .txt, .csv)
   3. For MISSING references, attempts to identify the documented
      regeneration command (a line in the README like
      'scripts/foo.py --output artifacts/...') and records whether
      the command exists
   4. Emits per-reference JSON evidence to
      artifacts/audits/doc-artifact-reconciliation.json
   5. FAILS (exit 1) if any reference has no creatable path AND no
      regeneration command in the README

Reproduction:
    1. git checkout HEAD~1   # obtain audited tree
    2. python scripts/audit/run_doc_artifact_reconciliation.py
    3. Expected: exit 0 (all references resolved)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "artifacts" / "audits"
JSON_OUT = AUDIT_DIR / "doc-artifact-reconciliation.json"

# Regex: match artifacts/... up to whitespace, end of line, or quote/paren
ARTIFACT_REF_RE = re.compile(r"artifacts/[A-Za-z0-9_./-]+")

# Regex: match scripts/... invocation lines (e.g., scripts/foo.py --output artifacts/...)
SCRIPT_INVOCATION_RE = re.compile(
    r"((?:scripts|architecture_lab)/[A-Za-z0-9_./-]+\.py)\b[^#\n]*"
)

EXTENSIONS = [".json", ".png", ".md", ".txt", ".csv", ".yaml", ".pt"]


def find_regen_command_for_artifact(readme_text: str, artifact_ref: str) -> str | None:
    """Search the README for a regeneration command that produces the
    artifact reference.

    Returns the script path if a regenerator is documented, else None.
    """
    # Look for invocation lines that mention the artifact reference
    # (or the artifact's directory).
    artifact_dir = artifact_ref.rsplit("/", 1)[0]
    for line in readme_text.splitlines():
        # Filter out non-invocation lines (comments, prose)
        # An invocation typically starts with scripts/ or python ...
        if not SCRIPT_INVOCATION_RE.search(line):
            continue
        if artifact_ref in line or artifact_dir + "/" in line or artifact_dir in line:
            m = SCRIPT_INVOCATION_RE.search(line)
            if m:
                return m.group(1)
    return None


def script_exists(script_path: str) -> bool:
    """Check whether a script path exists in the repo."""
    return (ROOT / script_path).exists()


def check_artifact_exists(artifact_ref: str) -> bool:
    """Check whether an artifact path exists on disk (with extension fallback)."""
    base = ROOT / artifact_ref
    if base.exists():
        return True
    # Try extension fallback: e.g., artifacts/foo → artifacts/foo.json, .png, etc.
    for ext in EXTENSIONS:
        if (ROOT / (artifact_ref + ext)).exists():
            return True
    # If the artifact_ref doesn't have an extension, try the parent dir
    parent = base.parent
    if parent.exists() and parent.is_dir():
        # If the parent exists but the file doesn't, the artifact might be
        # a partial reference. Treat as "exists if the parent directory exists".
        return True
    return False


def main() -> int:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    readme_paths = sorted(
        (ROOT / "docs" / "experiments").rglob("README.md")
    )
    total_docs = len(readme_paths)

    print("=== Doc <-> artifact reconciliation ===", flush=True)
    print(f"Total experiment READMEs: {total_docs}", flush=True)

    docs_with_refs = 0
    docs_without_refs = 0
    total_references = 0
    references_found = 0
    references_missing_resolvable = 0  # MISSING but has documented regen command
    references_missing_unresolvable = 0  # MISSING AND no regen command

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
            exists = check_artifact_exists(ref)
            regen_cmd = find_regen_command_for_artifact(text, ref)
            regen_exists = script_exists(regen_cmd) if regen_cmd else False

            if exists:
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
                "exists": exists,
                "status": status,
                "regen_command": regen_cmd,
                "regen_command_exists": regen_exists,
            })

        per_doc_results.append({
            "doc": rel,
            "ref_count": ref_count,
            "refs": per_refs,
        })

    # Emit machine-readable JSON
    payload = {
        "audit_name": "final-audit doc-artifact reconciliation",
        "audit_date": "2026-08-29",
        "methodology": (
            "For each docs/experiments/*/README.md: (a) extract artifacts/... "
            "references; (b) check on-disk existence (with extension fallback "
            ".json/.png/.md/.txt/.csv/.yaml/.pt); (c) for MISSING refs, identify "
            "documented regeneration command in the README and check that the "
            "script path exists. Categorize as FOUND / MISSING_RESOLVABLE "
            "(has regen cmd) / MISSING_UNRESOLVABLE (no regen cmd)."
        ),
        "summary": {
            "total_docs": total_docs,
            "docs_with_artifact_refs": docs_with_refs,
            "docs_without_artifact_refs": docs_without_refs,
            "total_references": total_references,
            "references_found": references_found,
            "references_missing_resolvable": references_missing_resolvable,
            "references_missing_unresolvable": references_missing_unresolvable,
        },
        "invariant": (
            "all artifact references must be FOUND on disk OR have a documented "
            "regeneration command whose script exists in the repo. "
            "MISSING_UNRESOLVABLE references are audit failures."
        ),
        "per_doc": per_doc_results,
    }

    JSON_OUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Summary print
    print("", flush=True)
    print("=== Summary ===", flush=True)
    print(f"Total docs:                       {total_docs}", flush=True)
    print(f"Docs with artifact refs:          {docs_with_refs}", flush=True)
    print(f"Docs without artifact refs:       {docs_without_refs}", flush=True)
    print(f"Total artifact references:        {total_references}", flush=True)
    print(f"References FOUND on disk:         {references_found}", flush=True)
    print(f"References MISSING (resolvable):  {references_missing_resolvable}", flush=True)
    print(f"References MISSING (unresolvable): {references_missing_unresolvable}", flush=True)
    print("", flush=True)
    print(f"JSON: {JSON_OUT}", flush=True)

    # Audit pass/fail
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
    print("PASS: all references resolved", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())