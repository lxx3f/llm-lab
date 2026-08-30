# Round-13 Reviewer / Auditor Evidence Document

- **Round**: 13 (the implementation commit; this evidence document was
  actually added in round-14, commit `e94ab02`)
- **Date**: 2026-08-29
- **Reviewer/auditor**: detached auditor (model: `setting`)
- **Reviewer verdict status**: this document records the EXECUTOR's
  evidence (commands + outputs + objection handling). The detached
  auditor's independent verdict is a separate artifact, recorded in
  the goal state (`state.goal.auditHistory[*].approved`). This document
  must NOT claim to be the detached verdict itself.
- **Round-13 commit (the audited implementation)**:
  `afd579f3a006aeb2862802a300acfa5530e933be`
- **This evidence document commit**: `e94ab02` (introduced in round-14
  to address auditor round-13's request for a durable reviewer
  evidence document)

## Scope of round-13

Round-13 of the final-audit fixes auditor round-12's three specific
blocking defects:

1. The artifact-reference regex `artifacts/[A-Za-z0-9_./-]+` excluded
   wildcards (`*`), so `artifacts/*-result.json` was truncated to
   `artifacts/`.
2. The same regex excluded brace-expansions, so
   `artifacts/{qwen2.5,huggingfacetb-smollm2}-*-eval-d2dev.json` was
   not extracted at all.
3. Angle-bracket placeholders (`<model>`, `<ckpt>`, `<safe-name>`) were
   silently dropped.

## Files in round-13

| File | Status | Purpose |
|---|---|---|
| `docs/reports/final-audit.md` | modified | Round-13 section added describing wildcard/brace/placeholder extraction fix |
| `scripts/audit/run_doc_artifact_reconciliation.py` | modified | Extended regex; new `expand_brace()` + `expand_glob()`; angle-bracket placeholders classified as placeholder |
| `scripts/run_tests.py` | modified | Added `test_audit_reconciliation_extraction.py` to `COMMON_TESTS` (27 → 28 test targets) |
| `tests/test_audit_reconciliation_extraction.py` | new | 26 regression tests for regex extraction, brace expansion, classify_ref, and integration with real docs |

## Exact reproduction commands (non-mutating — no `git checkout`)

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

# 7. Round-13 specific: regression tests for extraction
.venv/python.exe -m unittest tests.test_audit_reconciliation_extraction -v
# Expected: 26 tests, OK
```

### Why we no longer recommend `git checkout HEAD~1`

`git checkout HEAD~1` mutates the working tree's HEAD pointer, leaving
the auditor (and reproducer) on a detached HEAD that requires manual
restoration. Reproductions against `main` (current HEAD) are
non-mutating, simpler, and equally verifiable.

## Reproduced outputs from this round-13 commit (`afd579f`)

```
$ .venv/python.exe scripts/run_tests.py full
[test] running 28 test targets
...
Ran 386 tests in 49.218s
OK (skipped=1)

$ .venv/python.exe scripts/validate_stage0.py --examples
9/9 PASS, exit 0

$ python scripts/audit/run_gitignore_coverage.py
PASS: 7093/7093 (100.0000%, 0 exceptions)
JSON: artifacts/audits/gitignore-coverage.json

$ python scripts/audit/run_doc_artifact_reconciliation.py
Total active docs scanned:                58    # round-13 era (pre round-14 evidence doc)
Total artifact references:                167
Placeholders (excluded):                  18
References FOUND on disk:                 96
References MISSING (resolvable):          5
References MISSING (unresolvable):        0
Directories FOUND on disk:                19
Directories MISSING (resolvable):         0
Directories MISSING (unresolvable):       0
Globs FOUND on disk:                      29
Globs MISSING (resolvable):               0
Globs MISSING (unresolvable):             0
Exit code: 0

$ git status --short
(empty — 0 lines)
```

## Current audited-tree counts (post-round-14, current HEAD)

After round-14 added this evidence document and the `docs/reports/`
reviewer-evidence subdirectory, the counts naturally shifted because
the new doc itself contains 9 artifact references:

```
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
```

These numbers reflect the **current** tree state at HEAD. Round-13
itself (the audit script fix) was verified at commit `afd579f` with
58/167/29 counts; round-14 added this evidence document, which added
1 doc + 9 refs + 4 globs (now 59/173/33).

## Round-13 verification table (round-12 → round-13)

| Metric | Round-12 | Round-13 | Delta |
|---|---|---|---|
| Total artifact refs captured | 141 | **167** | +26 newly captured |
| Globs FOUND on disk | 0 | **29** | +29 (was unable to capture wildcards) |
| References MISSING_UNRESOLVABLE | 0 | 0 | 0 (held) |
| Directories MISSING_UNRESOLVABLE | 0 | 0 | 0 (held) |
| Globs MISSING_UNRESOLVABLE | n/a | 0 | 0 (new category) |
| Exit code | 0 | 0 | 0 (held) |
| Total test targets | 27 | **28** | +1 (test_audit_reconciliation_extraction added) |
| Total tests | 360 | **386** | +26 (new regression tests) |

## Round-13 objection handling

### Objection 1: "Wildcards truncated"

**Auditor evidence**:
- `artifacts/*-result.json` was truncated to `artifacts/` because the
  regex `[A-Za-z0-9_./-]+` excluded `*`.

**Fix**: Regex extended to `[A-Za-z0-9_./\-*{}<>,=]+`. Now captures
`*`, `{`, `}`, `<`, `>`, `,`, `=` inside artifact refs.

**Regression test** (`tests/test_audit_reconciliation_extraction.py`):
- `test_path_with_mid_wildcard`: extracts `artifacts/*-result.json` fully.
- `test_path_with_directory_wildcard`: extracts
  `artifacts/grpo-experiment/*/state.json` fully.

### Objection 2: "Brace-expansions not captured"

**Auditor evidence**:
- `artifacts/{qwen2.5,huggingfacetb-smollm2}-*-eval-d2dev.json` was
  not extracted at all.

**Fix**: New `expand_brace()` recursively expands `{a,b,c}` to
`[a, b, c]`. New `expand_glob()` uses `pathlib.glob` after brace
expansion.

**Regression test**:
- `test_brace_expansion`: brace expansion produces 3 alternatives for
  `{a,b,c}`.
- `test_brace_with_wildcard`: brace + wildcard combined.
- `test_p2_evaluator_brace_with_wildcard`: integration with the
  auditor-cited reference.

### Objection 3: "Angle-bracket placeholders dropped"

**Auditor evidence**:
- `artifacts/<model>`, `<ckpt>`, `<safe-name>` not extracted.

**Fix**: Angle-bracket placeholders `<...>` are classified as
`placeholder` (template references, excluded from existence check).
This is the correct semantic: `<model>` is a documentation template,
not a concrete path.

**Regression test**:
- `test_angle_bracket_placeholder`: regex captures it.
- `test_placeholder_angle_brackets`: classified as `placeholder`.

## Round-13 EXECUTOR verdict (this document)

This document records the executor's evidence that round-13 (commit
`afd579f`) addresses all three auditor round-12 objections. The
detached auditor's independent verdict on round-13 is recorded in the
goal state (`auditHistory`) and is **separate** from this executor
evidence.

## Pointers

- **Final-audit report**: `docs/reports/final-audit.md` (round-13 section)
- **Audit scripts**: `scripts/audit/run_doc_artifact_reconciliation.py`
  and `scripts/audit/run_gitignore_coverage.py`
- **Audit JSON evidence**: `artifacts/audits/*.json` (gitignored;
  regenerable by running the scripts)
- **Regression tests**: `tests/test_audit_reconciliation_extraction.py`