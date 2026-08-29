# Round-17 Reviewer / Auditor Evidence Document

- **Round**: 17 (the implementation commit; this evidence document is
  introduced in round-18 to address auditor round-17's request for a
  durable final reviewer/auditor evidence document)
- **Date**: 2026-08-29
- **Reviewer/auditor**: detached auditor (model: `setting`)
- **Reviewer verdict status**: this document records the EXECUTOR's
  evidence (commands + outputs + objection handling) for round-17. The
  detached auditor's independent verdict on round-17 is recorded in
  the goal state (`state.goal.auditHistory[*].approved`); this
  executor-authored document must NOT be presented as that
  independent verdict.
- **Round-17 commit (the audited implementation)**:
  `7771ca65107a7381510e421207192636d27bd66c`
- **Round-17 audited tree (parent of round-17 commit)**:
  `3c9a4758070c702647669cf1c1007180f164b7b9` (round-16, which
  removed `git checkout HEAD~1` from audit-script docstrings)
- **This evidence document commit**: round-18

## Scope of round-17

Round-17 of the final-audit fixes auditor round-16's one specific
blocking defect:

> "The durable final-audit report at line 156-157 stated 'On this
> audit ... 58 active docs scanned.' This stale 58-count paragraph
> was presented as 'On this audit' (i.e., the current audited tree)
> rather than clearly labeled as historical evidence. The fresh
> audit confirms 59 active documents, not 58."

The line was actually describing the audited parent (round-14 era,
HEAD~1 before the round-14 reviewer-evidence doc was added) but was
not labeled as such.

## Files in round-17

| File | Status | Purpose |
|---|---|---|
| `docs/reports/final-audit.md` | modified | Section 4 "Active documentation scope" paragraph relabeled to explicitly distinguish historical audited-parent counts from current HEAD counts |

## Exact reproduction commands (non-mutating; no git checkout)

```bash
# Stay on the current branch (main); no checkout needed.

# 1. Test suite
.venv/python.exe scripts/run_tests.py full
# Expected: "running 28 test targets", "Ran 386 tests ...", "OK (skipped=N)"

# 2. Schema validation
.venv/python.exe scripts/validate_stage0.py --examples
# Expected: 9/9 PASS, exit 0

# 3. Working tree state
git status --short                # → empty

# 4. Tracked sensitive paths (all should be empty)
git ls-files artifacts/           # → empty
git ls-files datasets/            # → empty
git ls-files .tmp/                # → empty (if listed)
git ls-files '*.pt' '*.ckpt' '*.safetensors' '*.bin'   # → empty

# 5. Bounded per-file .gitignore audit
python scripts/audit/run_gitignore_coverage.py
# Expected: "PASS: N/N (100%, 0 exceptions)"

# 6. Doc <-> artifact reconciliation (broad scan with wildcards/braces)
python scripts/audit/run_doc_artifact_reconciliation.py
# Expected: "PASS: all non-placeholder references resolved"
```

## Reproduced outputs from the audited tree (HEAD~1 = `3c9a475`)

```
$ .venv/python.exe scripts/run_tests.py full
[test] running 28 test targets
...
Ran 386 tests in 46.470s
OK (skipped=1)

$ .venv/python.exe scripts/validate_stage0.py --examples
9/9 PASS, exit 0

$ python scripts/audit/run_gitignore_coverage.py
PASS: 7093/7093 (100.0000%, 0 exceptions)
JSON: artifacts/audits/gitignore-coverage.json

$ python scripts/audit/run_doc_artifact_reconciliation.py
Total active docs scanned:                59
Docs with artifact refs:                  49
Total artifact references:                173
Placeholders (excluded):                  19
References FOUND on disk:                 97
References MISSING (resolvable):          5
References MISSING (unresolvable):        0
Directories FOUND on disk:                19
Directories MISSING (resolvable):         0
Directories MISSING (unresolvable):       0
Globs FOUND on disk:                      33
Globs MISSING (resolvable):               0
Globs MISSING (unresolvable):             0
Exit code: 0

$ git status --short
(empty — 0 lines)
```

## Current audited-tree counts (round-17 era, current HEAD)

These are the counts the round-17 commit claims are documented in
`docs/reports/final-audit.md` Section 4 "Real results from this
audit (this host)":

```
Total active docs scanned:                59
Total artifact references:                173
Placeholders (excluded):                  19
References FOUND on disk:                 97
Directories FOUND on disk:                19
Globs FOUND on disk:                      33
MISSING_UNRESOLVABLE:                     0 (refs + dirs + globs)
```

## Round-17 verification table

| Metric | Round-16 | Round-17 | Delta |
|---|---|---|---|
| Stale "On this audit" 58-count claim | present (line 156-157) | **relabeled as historical evidence** | defect fixed |
| Total active docs (current HEAD) | 59 | 59 | 0 |
| Total artifact references | 173 | 173 | 0 |
| Globs FOUND | 33 | 33 | 0 |
| MISSING_UNRESOLVABLE | 0 | 0 | 0 |
| Exit code | 0 | 0 | 0 |

## Round-17 objection handling

### Objection 1: "Stale 58-count presented as 'On this audit'"

**Auditor evidence**: `docs/reports/final-audit.md` line 156-157
said:

> "On this audit: 92 total docs, 34 excluded ... 58 active docs
> scanned."

This stale 58-count paragraph was presented as current audited tree
rather than historical evidence. The fresh audit confirms 59 active
documents, not 58.

**Fix**: Section 4 paragraph relabeled to explicitly distinguish
historical audited-parent counts from current HEAD counts:

> "On this audit at the audited parent (`HEAD~1`, before the
> round-14 reviewer-evidence doc was added): 92 total docs, 34
> excluded (1 license + 1 final-audit + 32 stage reviews + 1 other),
> **58 active docs scanned**. The current HEAD shows 59 active docs
> (after round-14 added the round-13 reviewer-evidence doc which
> contributes one additional active doc); the round-13
> implementation itself was verified at 58 active docs, and the
> current 'Real results from this audit (this host)' section below
> uses the current 59-doc tree state."

The "Real results" section's existing note about "58/167/29 counts"
was also clarified as historical evidence (round-13 verification
of commit `afd579f`), with the current HEAD's 59/173/33 numbers
explicitly called out.

## Round-17 EXECUTOR verdict (this document)

This document records the executor's evidence that round-17 (commit
`7771ca6`) addresses auditor round-16's blocking objection (the
stale "58 active docs" claim). The detached auditor's independent
verdict on round-17 is recorded in the goal state (`auditHistory`)
and is **separate** from this executor evidence.

**No further commits to this document are expected.** The
detached-auditor verdict, when it arrives, is recorded by the
auditor itself in the goal state, not by edits to this file.

## Pointers

- **Final-audit report**: `docs/reports/final-audit.md` (round-17 section)
- **Round-13 reviewer evidence (predecessor document)**:
  `docs/reports/final-audit/round-13-reviewer-evidence.md`
- **Audit scripts**: `scripts/audit/run_doc_artifact_reconciliation.py`
  and `scripts/audit/run_gitignore_coverage.py`
- **Audit JSON evidence**: `artifacts/audits/*.json` (gitignored;
  regenerable by running the scripts)
- **Regression tests**: `tests/test_audit_reconciliation_extraction.py`