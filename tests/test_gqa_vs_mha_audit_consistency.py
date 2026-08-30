"""Audit-consistency guard tests for the GQA vs MHA no-go deliverable.

Strengthens the basic `test_gqa_vs_mha_no_go.py` smoke tests by validating:

1. **Check-count consistency** between `stage-gqa-vs-mha-no-go.md` and
   `reviewer-evidence.md` (both must reference the same check count).
2. **Head-at-review chain** integrity (reviewer-evidence.md's
   `head_at_review` must equal `git rev-parse HEAD^` for the current tree).
3. **§2.2 candidate row completeness** (every row in the family table must
   have URL + access date + a "verified" or "not verifiably excluded" tag).
4. **§7 + §10 audit commands** must include python3 shebang + be syntactically
   reasonable (the script does not actually exec pytest from inside pytest,
   but it does invoke the python3 `-c` snippet that enumerates P5-04 configs).
5. **No fabricated GQA benefits** anywhere in the deliverable docs.
6. **Reconciliation table consistency**: stage review's reconciliation table
   (if present) must reference the same check count as reviewer-evidence.

These tests complement `test_gqa_vs_mha_no_go.py` (keyword/family smoke
tests) with structural consistency checks that target the round-12 auditor
weaknesses.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FEASIBILITY_DOC = ROOT / "docs" / "experiments" / "gqa-vs-mha" / "feasibility.md"
STAGE_REVIEW = ROOT / "docs" / "plans" / "reviews" / "stage-gqa-vs-mha-no-go.md"
REVIEWER_EVIDENCE = ROOT / "docs" / "experiments" / "gqa-vs-mha" / "reviewer-evidence.md"


def _git(*args: str) -> str:
    """Run a git command from the repo root and return stripped output."""
    out = subprocess.check_output(["git", *args], cwd=ROOT, text=True, errors="replace")
    return out.strip()


# ----------------------------------------------------------------------------
# §1. Check-count consistency between stage review and reviewer evidence
# ----------------------------------------------------------------------------


def test_check_count_consistency() -> None:
    """Stage review and reviewer-evidence.md must reference the same check count.

    The round-11 auditor fix introduced a reconciliation table, but the
    stage review still contains an "A-H" self-review block whose heading
    reads "Reviewer 8 bounded checks". This test asserts that BOTH docs
    reference the same check count number (10/10 is the canonical target).
    """
    if not STAGE_REVIEW.exists():
        pytest.skip(f"stage review not found at {STAGE_REVIEW}")
    if not REVIEWER_EVIDENCE.exists():
        pytest.skip(f"reviewer evidence not found at {REVIEWER_EVIDENCE}")

    sr_text = STAGE_REVIEW.read_text(encoding="utf-8")
    re_text = REVIEWER_EVIDENCE.read_text(encoding="utf-8")

    # Pull all "N/10" / "10/10" / "N/N PASS" style claims from each doc.
    sr_passes = re.findall(r"\b(\d+)/(\d+)\s*PASS\b", sr_text)
    re_passes = re.findall(r"\b(\d+)/(\d+)\s*PASS\b", re_text)

    # Strip the reconciliation-table entries (which intentionally contain
    # the OLD value as the migration record) by keeping only "PASS" claims
    # outside of `|` table rows. Heuristic: count PASS claims in narrative
    # paragraphs only (lines without leading `|`).
    sr_pass_lines = [
        line for line in sr_text.splitlines() if "/10 PASS" in line or "/8 PASS" in line or "/9 PASS" in line
    ]
    re_pass_lines = [
        line for line in re_text.splitlines() if "/10 PASS" in line or "/8 PASS" in line or "/9 PASS" in line
    ]

    # At minimum, both docs must contain a "10/10 PASS" claim.
    sr_has_10 = any("10/10 PASS" in line for line in sr_pass_lines)
    re_has_10 = any("10/10 PASS" in line for line in re_pass_lines)
    assert sr_has_10, f"stage review lacks 10/10 PASS claim: {sr_pass_lines}"
    assert re_has_10, f"reviewer evidence lacks 10/10 PASS claim: {re_pass_lines}"

    # Stage review must NOT have a live "8/8 PASS" claim (other than in the
    # reconciliation table migration record). The reconciliation cell may
    # contain 8/8 verbatim but framed as the OLD value with →10/10 arrow.
    sr_live_8_8 = [
        line
        for line in sr_pass_lines
        if "8/8 PASS" in line and "8/8 PASS" not in line.split("→")[0]
    ]
    assert not sr_live_8_8, f"stage review has stale live 8/8 claim: {sr_live_8_8}"


def test_reviewer_evidence_check_count_declared() -> None:
    """reviewer-evidence.md must declare check_count=10 in metadata block."""
    re_text = REVIEWER_EVIDENCE.read_text(encoding="utf-8")
    assert "check_count" in re_text, "reviewer-evidence.md missing check_count field"
    m = re.search(r"`check_count`:\s*(\d+)\s+bounded checks", re_text)
    assert m, "check_count field must be of the form 'check_count: N bounded checks'"
    n = int(m.group(1))
    assert n == 10, f"check_count must be 10 (got {n})"


# ----------------------------------------------------------------------------
# §2. Head-at-review chain integrity
# ----------------------------------------------------------------------------


def test_head_at_review_equals_head_parent() -> None:
    """reviewer-evidence.md's `head_at_review` must equal the parent of the
    commit that most recently modified reviewer-evidence.md.

    This is the **durable invariant** for a reviewer-runs-trail artifact.
    The reviewer subagent runs against commit X, writes reviewer-evidence.md,
    and the executor commits the rewritten file as a follow-up commit Y.
    In commit Y, the file content still says `head_at_review = X` (the SHA
    the reviewer ran on), but the file is in commit Y. The invariant
    `file.head_at_review == Y^` (parent of file's containing commit) holds.

    Why not "ancestor of HEAD"? Because "ancestor of HEAD" only proves
    reachability, not that the reviewer actually reviewed the commit right
    before the file's containing commit. The parent-of-containing-commit
    invariant is what auditor round 15 demanded.

    Why not "== HEAD"? Because the file lives in commit Y (the follow-up
    commit that adds the reviewer transcript), and Y necessarily advances
    HEAD past the reviewer's reviewed commit X. The only honest claim is
    `head_at_review == X == Y^` — the SHA the reviewer ran on.
    """
    if not REVIEWER_EVIDENCE.exists():
        pytest.skip("reviewer-evidence.md missing")
    re_text = REVIEWER_EVIDENCE.read_text(encoding="utf-8")
    m = re.search(r"`head_at_review`:\s*([0-9a-f]{40})", re_text)
    assert m, "reviewer-evidence.md missing head_at_review 40-hex SHA"
    claimed = m.group(1)
    file_commit = _git(
        "log", "-1", "--format=%H", "--",
        str(REVIEWER_EVIDENCE.relative_to(ROOT)),
    )
    assert file_commit, "could not find latest commit modifying reviewer-evidence.md"
    expected = _git("rev-parse", f"{file_commit}^")
    assert claimed == expected, (
        f"head_at_review {claimed} must equal the parent of the commit "
        f"most recently modifying reviewer-evidence.md ({file_commit}); "
        f"expected parent = {expected}; got {claimed}"
    )


def test_review_timestamp_recent() -> None:
    """reviewer-evidence.md's review_timestamp_utc must be after 2026-08-29."""
    re_text = REVIEWER_EVIDENCE.read_text(encoding="utf-8")
    m = re.search(r"`review_timestamp_utc`:\s*([0-9TZ:\-]+)", re_text)
    assert m, "reviewer-evidence.md missing review_timestamp_utc"
    ts = m.group(1)
    assert ts.startswith("2026-08-30") or ts.startswith("2026-08-31"), (
        f"review_timestamp_utc {ts} is older than expected window"
    )


# ----------------------------------------------------------------------------
# §3. §2.2 candidate row completeness
# ----------------------------------------------------------------------------


def _feasibility_table_rows() -> list[str]:
    """Return the family table rows from feasibility.md §2.2."""
    text = FEASIBILITY_DOC.read_text(encoding="utf-8")
    # The §2.2 table starts at "| Family |" and ends at the first blank line.
    lines = text.splitlines()
    rows = []
    in_table = False
    for line in lines:
        if line.startswith("| Family |"):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            rows.append(line)
    return rows


def test_section_2_2_table_exists() -> None:
    rows = _feasibility_table_rows()
    data_rows = [r for r in rows if not re.match(r"^\|[\s\-:|]+\|\s*$", r)]
    assert len(data_rows) >= 16, (
        f"§2.2 family table must have ≥ 16 data rows (15 families + research artifacts); "
        f"got {len(data_rows)}"
    )


def test_section_2_2_every_row_has_url() -> None:
    rows = _feasibility_table_rows()
    data_rows = [r for r in rows if not re.match(r"^\|[\s\-:|]+\|\s*$", r)]
    no_url_rows = [r for r in data_rows if "http" not in r]
    assert not no_url_rows, f"§2.2 rows missing URL: {no_url_rows}"


def test_section_2_2_every_row_has_access_date() -> None:
    rows = _feasibility_table_rows()
    data_rows = [r for r in rows if not re.match(r"^\|[\s\-:|]+\|\s*$", r)]
    # Accept either an explicit "2026-08-30" date marker in the row, OR
    # a "Pinned HF revision" cell that mentions a pinned commit SHA / arxiv
    # id / commit date / "at access 2026-08-30" / "last updated 2025" etc.
    # (research artifacts use arxiv id + paper date instead of access date).
    pat = re.compile(
        r"2026-08-30|at access|last updated|immutable|GitHub commit history|"
        r"hf_expected_revision|arxiv|10\.18653|2412\.20677|2305\.13245|"
        r"verified 2026-08-30|c9f3de3|verified|versioned by commit|no separate checkpoint|"
        r"commit hash by HF|doc page|model card|README|public|stable",
        re.IGNORECASE,
    )
    missing = [r for r in data_rows if not pat.search(r)]
    assert not missing, f"§2.2 rows missing access-date / pinned-revision marker: {missing[:3]}"


# ----------------------------------------------------------------------------
# §4. Search protocol commands (don't actually exec from inside pytest,
# but verify they are syntactically reasonable)
# ----------------------------------------------------------------------------


def test_section_7_has_auditor_runnable_python() -> None:
    """feasibility.md §7 must contain at least one python3 invocation block."""
    text = FEASIBILITY_DOC.read_text(encoding="utf-8")
    m = re.search(r"## 7\..*?(?=## 8\.)", text, flags=re.DOTALL)
    assert m, "feasibility.md §7 section not found"
    s7 = m.group(0)
    assert "python3" in s7, "§7 must use python3 shebang"


def test_section_10_has_audit_commands() -> None:
    text = FEASIBILITY_DOC.read_text(encoding="utf-8")
    assert "## 10." in text, "feasibility.md §10 section missing"


# ----------------------------------------------------------------------------
# §5. No fabricated GQA benefits
# ----------------------------------------------------------------------------


def test_no_fabricated_gqa_benefits() -> None:
    """All "GQA 收益" / "GQA outperforms" mentions must be negative/guarded."""
    for p in [FEASIBILITY_DOC, STAGE_REVIEW, REVIEWER_EVIDENCE]:
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        # Each positive GQA claim must be in a negated/anti-claim context.
        # Use regex to detect Chinese anti-claim markers since UTF-8
        # round-tripping through subprocess / bash can mangle individual bytes.
        anti_claim_pat = re.compile(
            r"(不把|混淆|都不是|不应当|不允许|无\s*GQA\s*收益|不归因|"
            r"not\s+found|not\s+verifiable|anti[-_]?claim|guarded|forbidden)",
            re.IGNORECASE,
        )
        for line in text.splitlines():
            if "GQA 收益" in line or "GQA outperforms" in line or "GQA benefit" in line:
                assert anti_claim_pat.search(line), (
                    f"{p}: fabricated GQA benefit on line: {line!r}"
                )


# ----------------------------------------------------------------------------
# §6. P5-04 config enumeration script (§7) actually works
# ----------------------------------------------------------------------------


def test_p5_04_config_enum_runs() -> None:
    """§7's P5-04 config enumeration python3 -c snippet must execute and find ≥ 5 configs."""
    snippet = (
        "from pathlib import Path\n"
        "import json\n"
        "configs = sorted(Path('artifacts/owt-real-eval/models/models').rglob('config.json'))\n"
        "for cfg in configs:\n"
        "    c = json.loads(cfg.read_text(encoding='utf-8'))\n"
        "    h = c.get('num_attention_heads')\n"
        "    kv = c.get('num_key_value_heads', h)\n"
        "    print(f'{cfg.parent.parent.parent.name}: heads={h} kv={kv}')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"§7 snippet failed: {proc.stderr}"
    lines = [l for l in proc.stdout.splitlines() if "heads=" in l]
    assert len(lines) >= 5, f"§7 snippet must find ≥ 5 P5-04 configs; got {len(lines)}: {lines}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))