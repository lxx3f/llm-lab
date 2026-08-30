#!/usr/bin/env python3
"""scripts/audit/run_doc_artifact_reconciliation.py

Doc <-> artifact reconciliation audit for the final-audit.

Scans every in-scope active documentation file (docs/**/*.md, with
explicit exclusions for license files and the audit report itself),
extracts artifact references (including wildcard, brace-expansion,
and angle-bracket placeholder forms), classifies them by kind, and
verifies existence per kind.

Classification:
  - EXACT_FILE: concrete path with extension; check exact path or
    extension fallback
  - GLOB_PATTERN: contains * OR {}; expand via glob/braces and require
    at least one match
  - DIRECTORY: ends in /; or on-disk check finds a directory at the
    path (handles 'artifacts/foo' that points to a directory even
    without trailing slash)
  - PLACEHOLDER: ends in -, _, ...; or contains <...> angle brackets
    (template placeholder); excluded from existence check

Resolution status:
  - FOUND (exact_file/glob_pattern/directory, exists on disk)
  - MISSING_RESOLVABLE (not found but has documented regen cmd)
  - MISSING_UNRESOLVABLE (not found AND no regen cmd) - AUDIT FAIL
  - PLACEHOLDER (kind=placeholder; excluded)

Per-doc/per-ref JSON evidence + exit 1 if any MISSING_UNRESOLVABLE.

Reproduction (non-mutating; no git checkout):
    1. Confirm clean main: git status --short  # → empty
    2. python scripts/audit/run_doc_artifact_reconciliation.py
    3. Expected: exit 0
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

# Active documentation scope: every .md file under docs/, except:
#   - docs/licenses/   (third-party license text, not project docs)
#   - docs/archive/    (closed / non-active records; may reference
#     removed artifacts as part of the change record)
#   - docs/plans/reviews/*   (historical stage reviews; may reference
#     removed artifacts as part of the change record. These are not
#     active contracts. The reconciliation report records this scope
#     decision in its methodology field.)
EXCLUDE_PATH_PATTERNS = (
    "docs/licenses/",
    "docs/archive/",
    "docs/plans/reviews/",
)

# Regex: match artifacts/... up to whitespace, end of line, or quote/paren
# Captures concrete paths WITH wildcards/brace-expansions/angle-brackets.
# Stops at characters that would terminate a path reference:
#   - whitespace, quotes (", '), parentheses, backticks, semicolons,
#     equal-signs (CLI flag values), or end-of-line
# Allows inside: alphanumeric, _, ., /, -, *, {, }, <, >, ,, =
# (comma and = are inside the character class so brace-expansion
# alternatives like {a,b,c} are captured fully; only spaces/punctuation
# outside the pattern terminate it)
ARTIFACT_REF_RE = re.compile(
    r"artifacts/[A-Za-z0-9_./\-*{}<>,=]+"
)

SCRIPT_INVOCATION_RE = re.compile(
    r"((?:scripts|architecture_lab)/[A-Za-z0-9_./-]+\.py)\b[^#\n]*"
)

EXTENSIONS = [".json", ".png", ".md", ".txt", ".csv", ".yaml", ".pt",
              ".ckpt", ".safetensors", ".bin"]

# Placeholder suffix chars: truncated category references like
# 'artifacts/sft-' or 'artifacts/moe-owt-formal-curve-'.
PLACEHOLDER_ENDINGS = ("-", "_", "...")


def is_excluded(rel_path: str) -> bool:
    # Normalize to forward slashes for pattern matching (Windows uses \)
    rel_norm = rel_path.replace("\\", "/")
    for pat in EXCLUDE_PATH_PATTERNS:
        pat_norm = pat.replace("\\", "/")
        if rel_norm.startswith(pat_norm) or rel_norm == pat_norm.rstrip("/"):
            return True
    return False


def expand_brace(ref: str) -> list[str]:
    """Expand a brace-expansion reference to concrete paths.

    Example: 'artifacts/{a,b,c}-foo.json' →
        ['artifacts/a-foo.json', 'artifacts/b-foo.json',
         'artifacts/c-foo.json']

    If the reference contains no braces, returns [ref] unchanged.
    """
    if "{" not in ref:
        return [ref]

    # Find the brace group(s)
    # Simple expansion: split on commas inside the first brace group,
    # then recurse if multiple groups.
    def _expand(s: str) -> list[str]:
        m = re.search(r"\{([^{}]*)\}", s)
        if not m:
            return [s]
        prefix = s[: m.start()]
        suffix = s[m.end():]
        alternatives = m.group(1).split(",")
        results: list[str] = []
        for alt in alternatives:
            results.extend(_expand(prefix + alt + suffix))
        return results

    return _expand(ref)


def classify_ref(ref: str) -> str:
    """Classify an artifact reference.

    Returns: 'exact_file' | 'glob_pattern' | 'directory' | 'placeholder'

    Strategy:
      1. Placeholder: trailing -, _, or ...
      2. Placeholder: contains <...> angle brackets (template)
      3. Directory: ends in /
      4. On-disk check: if any brace-expanded candidate exists as a
         directory → directory (handles refs like
         'artifacts/foo/{a,b}' or 'artifacts/foo/{a,b}-*/' that point
         to dirs)
      5. Contains * OR {} → glob_pattern
      6. Known file extension → exact_file
      7. Default → exact_file
    """
    # Placeholder: trailing -, _, or ...
    if any(ref.endswith(ending) for ending in PLACEHOLDER_ENDINGS):
        return "placeholder"
    # Placeholder: contains <...> angle brackets (template placeholder)
    if "<" in ref and ">" in ref:
        return "placeholder"
    # Directory: ends in /
    if ref.endswith("/"):
        return "directory"
    # On-disk check: does any brace-expanded candidate exist as a dir?
    for cand in expand_brace(ref):
        cand_clean = cand.rstrip("/")
        if (ROOT / cand_clean).is_dir():
            return "directory"
    # Glob pattern: contains * OR {}
    if "*" in ref or "{" in ref:
        return "glob_pattern"
    # Exact file: ends in known extension
    if any(ref.endswith(ext) for ext in EXTENSIONS):
        return "exact_file"
    # Default: treat as exact file
    return "exact_file"


def expand_glob(ref: str) -> list[str]:
    """Expand a brace + glob reference to concrete paths on disk.

    Handles both:
      - Brace expansion: 'artifacts/{a,b}-foo' → ['a-foo', 'b-foo']
      - Glob: 'artifacts/*-result.json' → all matching files
    """
    # First expand braces to candidate refs
    candidates = expand_brace(ref)
    matched: list[str] = []
    for cand in candidates:
        if "*" in cand:
            # Use pathlib glob with relative pattern
            for p in Path(".").glob(cand):
                if p.is_file():
                    matched.append(str(p))
        else:
            # No wildcard — check exact path
            p = Path(cand)
            if p.is_file():
                matched.append(str(p))
    return matched


def check_artifact_exists(ref: str, kind: str) -> tuple[bool | None, list[str]]:
    """Check whether an artifact reference is resolvable on disk.

    Returns:
      - (True, [paths]) if found
      - (False, []) if not found
      - (None, []) for placeholders (excluded)
    """
    if kind == "placeholder":
        return None, []

    if kind == "directory":
        # For directory refs, expand braces and check each
        candidates = expand_brace(ref)
        for cand in candidates:
            cand_clean = cand.rstrip("/")
            # Try literal path first
            p = ROOT / cand_clean
            if p.is_dir():
                return True, [str(p.relative_to(ROOT))]
            # Try with glob (handles refs like 'artifacts/run-{A,B}-*/'
            # where the expanded candidate still has a wildcard)
            if "*" in cand_clean:
                for match in Path(".").glob(cand_clean):
                    if match.is_dir():
                        return True, [str(match)]
        return False, []

    if kind == "glob_pattern":
        matches = expand_glob(ref)
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


def find_regen_command(doc_text: str, artifact_ref: str) -> str | None:
    """Search for a scripts/...py invocation that produces the artifact ref.

    Handles multi-line shell invocations by joining continuation lines
    (lines ending with `\\` are joined with the next line).
    """
    artifact_dir = artifact_ref.rstrip("/").rsplit("/", 1)[0]
    # Join continuation lines so multi-line bash invocations are scanned as one
    joined_lines: list[str] = []
    buf = ""
    for line in doc_text.splitlines():
        if line.rstrip().endswith("\\"):
            buf += line.rstrip().rstrip("\\") + " "
        else:
            buf += line
            joined_lines.append(buf)
            buf = ""
    if buf:
        joined_lines.append(buf)

    for line in joined_lines:
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

    # Scan every active doc under docs/, except exclusions.
    docs_dir = ROOT / "docs"
    all_docs = sorted(docs_dir.rglob("*.md"))
    readme_paths = [
        d for d in all_docs
        if not is_excluded(str(d.relative_to(ROOT)))
    ]
    total_docs = len(readme_paths)

    print("=== Doc <-> artifact reconciliation ===", flush=True)
    print(f"Total active documentation files: {total_docs}", flush=True)
    print(
        "Excluded: docs/licenses/, docs/archive/, "
        "docs/plans/reviews/*",
        flush=True,
    )

    docs_with_refs = 0
    docs_without_refs = 0
    total_references = 0
    placeholders_excluded = 0
    directories_checked = 0
    directories_found = 0
    directories_missing_resolvable = 0
    directories_missing_unresolvable = 0
    references_found = 0
    references_missing_resolvable = 0
    references_missing_unresolvable = 0
    globs_found = 0
    globs_missing_resolvable = 0
    globs_missing_unresolvable = 0

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
            regen_cmd = find_regen_command(text, ref)
            regen_exists = script_exists(regen_cmd) if regen_cmd else False

            if kind == "placeholder":
                placeholders_excluded += 1
                status = "PLACEHOLDER"
            elif exists is True:
                if kind == "directory":
                    directories_found += 1
                elif kind == "glob_pattern":
                    globs_found += 1
                else:
                    references_found += 1
                status = "FOUND"
            elif kind == "directory":
                directories_checked += 1
                if regen_exists:
                    directories_missing_resolvable += 1
                    status = "MISSING_RESOLVABLE"
                else:
                    directories_missing_unresolvable += 1
                    status = "MISSING_UNRESOLVABLE"
            elif kind == "glob_pattern":
                if regen_exists:
                    globs_missing_resolvable += 1
                    status = "MISSING_RESOLVABLE"
                else:
                    globs_missing_unresolvable += 1
                    status = "MISSING_UNRESOLVABLE"
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

    total_missing_unresolvable = (
        references_missing_unresolvable
        + directories_missing_unresolvable
        + globs_missing_unresolvable
    )

    payload = {
        "audit_name": "final-audit doc-artifact reconciliation",
        "audit_date": "2026-08-29",
        "methodology": (
            "Scan every active documentation file under docs/ (excluding "
            "docs/licenses/, docs/archive/, and "
            "docs/plans/reviews/* which are historical stage reviews "
            "that may reference removed artifacts as part of the change "
            "record). Extract artifacts/... references using a regex that "
            "captures concrete paths WITH wildcards (*), brace expansions "
            "({a,b,c}), and angle-bracket placeholders (<foo>). "
            "Classify each as exact_file / glob_pattern / directory / "
            "placeholder. Check existence per kind: placeholders excluded; "
            "glob_pattern uses pathlib.glob + brace expansion; directory "
            "checks on-disk directory existence. Categorize as FOUND / "
            "MISSING_RESOLVABLE / MISSING_UNRESOLVABLE / PLACEHOLDER."
        ),
        "summary": {
            "total_docs_scanned": total_docs,
            "docs_with_artifact_refs": docs_with_refs,
            "docs_without_artifact_refs": docs_without_refs,
            "total_references": total_references,
            "placeholders_excluded": placeholders_excluded,
            "references_found": references_found,
            "references_missing_resolvable": references_missing_resolvable,
            "references_missing_unresolvable": references_missing_unresolvable,
            "directories_found": directories_found,
            "directories_missing_resolvable": directories_missing_resolvable,
            "directories_missing_unresolvable": directories_missing_unresolvable,
            "globs_found": globs_found,
            "globs_missing_resolvable": globs_missing_resolvable,
            "globs_missing_unresolvable": globs_missing_unresolvable,
        },
        "invariant": (
            "all non-placeholder artifact references (exact_file / "
            "glob_pattern / directory) must be FOUND on disk OR have a "
            "documented regeneration command whose script exists in the "
            "repo. MISSING_UNRESOLVABLE references (any kind) are audit "
            "failures."
        ),
        "per_doc": per_doc_results,
    }

    JSON_OUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("", flush=True)
    print("=== Summary ===", flush=True)
    print(f"Total active docs scanned:                {total_docs}", flush=True)
    print(f"Docs with artifact refs:                  {docs_with_refs}", flush=True)
    print(f"Docs without artifact refs:               {docs_without_refs}", flush=True)
    print(f"Total artifact references:                {total_references}", flush=True)
    print(f"Placeholders (excluded):                  {placeholders_excluded}", flush=True)
    print(f"References FOUND on disk:                 {references_found}", flush=True)
    print(f"References MISSING (resolvable):          {references_missing_resolvable}", flush=True)
    print(f"References MISSING (unresolvable):        {references_missing_unresolvable}", flush=True)
    print(f"Directories FOUND on disk:                {directories_found}", flush=True)
    print(f"Directories MISSING (resolvable):         {directories_missing_resolvable}", flush=True)
    print(f"Directories MISSING (unresolvable):       {directories_missing_unresolvable}", flush=True)
    print(f"Globs FOUND on disk:                      {globs_found}", flush=True)
    print(f"Globs MISSING (resolvable):               {globs_missing_resolvable}", flush=True)
    print(f"Globs MISSING (unresolvable):             {globs_missing_unresolvable}", flush=True)
    print("", flush=True)
    print(f"JSON: {JSON_OUT}", flush=True)

    if total_missing_unresolvable > 0:
        print("", flush=True)
        print(f"FAIL: {total_missing_unresolvable} unresolvable references", flush=True)
        for doc in per_doc_results:
            for ref in doc.get("refs", []):
                if ref["status"] == "MISSING_UNRESOLVABLE":
                    print(f"  {doc['doc']}: {ref['ref']} ({ref['kind']})", flush=True)
        return 1

    total_missing_resolvable = (
        references_missing_resolvable
        + directories_missing_resolvable
        + globs_missing_resolvable
    )
    if total_missing_resolvable > 0:
        print("", flush=True)
        print(
            f"INFO: {total_missing_resolvable} references/dirs/globs missing "
            "but resolvable by running the documented commands",
            flush=True,
        )

    print("", flush=True)
    print("PASS: all non-placeholder references resolved", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())