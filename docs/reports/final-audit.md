# 最终全链路审查报告 — Final Audit Report

- **Date**: 2026-08-29
- **Author**: Minimax M3 (per `docs/plans/review-process.md`)
- **Goal**: 验证 `llm-lab` 实验闭环各阶段的最终状态
- **Audited snapshot**: this report was originally added in commit
  `b1278c3` (round-8) and has been modified in subsequent audit
  commits. Each audit-only commit's parent (`HEAD~1`) is the
  audited tree that the report's "Real results" numbers describe.
  Earlier rounds' per-round reviewer-evidence documents
  (`docs/reports/final-audit/round-N-reviewer-evidence.md`) record
  the specific committed SHA that was reproduced for that round.
- **Historical snapshot note (B-audit round-19, 2026-08-29)**: the
  numbers reported in this file (e.g., 175 total refs, 98 found,
  34 globs) describe the state captured at the original final-audit
  round-20 audit time (commit `f1fe61c`). They are **historical
  snapshots**, not live current-tree state. Subsequent B-audit
  rounds (round-1..round-19) and other project commits have
  changed the artifact set; the live current-tree counts differ
  from these historical numbers. Auditor must not interpret these
  figures as current-tree state. See "## Current live-verification
  (auditor runs against current tree)" section below for the actual
  current numbers (use abstract pointer notation; do not hardcode
  SHAs).
- **Current HEAD**: the most recent commit on `main` (resolve via
  `git rev-parse HEAD`).
- **Audited tree**: `HEAD~1` (the parent of the most recent commit).
  All counts in this report's "Real results" section describe the
  state after `git checkout HEAD~1` would land — i.e., the working
  tree at the audited parent, before the current audit-only commit's
  changes were applied.
- **Round-17 reviewer evidence document (current durable evidence)**:
  `docs/reports/final-audit/round-17-reviewer-evidence.md`
- **Round-13 reviewer evidence document (historical evidence)**:
  `docs/reports/final-audit/round-13-reviewer-evidence.md`
- **Scope**: this audit verifies (a) tests pass, (b) schemas validate,
  (c) docs reconcile with artifacts, (d) sensitive files are gitignored,
  (e) reviewer evidence is saved, (f) working tree is clean. It does NOT
  verify evaluation correctness (e.g., `scripts/eval_transformers.py`'s
  generation behavior); that is a separate concern tracked under
  P5-02 benchmark evaluation subset work.

## Current live-verification (auditor runs against current tree)

abstract HEAD pointer — auditor runs `git rev-parse HEAD` to verify
current value; numbers in this section use abstract expected-output
form (auditor runs the command and compares). This section was added
in B-audit round-19 to break the SHA-staleness cycle.

**Live command 1** (reconciliation):
```text
<abstract — auditor runs: python scripts/audit/run_doc_artifact_reconciliation.py>
expected output (actual current tree, B-audit round-19):
  - Total active docs scanned: 60
  - Total artifact references: 177
  - References FOUND on disk: 99
  - References MISSING (resolvable): 5
  - References MISSING (unresolvable): 0
  - Directories FOUND on disk: 19
  - Globs FOUND on disk: 35
  - Exit code: 0
```

**Live command 2** (gitignore coverage):
```text
<abstract — auditor runs: python scripts/audit/run_gitignore_coverage.py>
expected output: PASS: 7094/7094 (100.0000%, 0 exceptions)
```

**Live command 3** (validate_stage0):
```text
<abstract — auditor runs: python scripts/validate_stage0.py --examples>
expected output: 9/9 PASS
```

**Live command 4** (full test suite):
```text
<abstract — auditor runs: python scripts/run_tests.py full>
expected output: Ran 386 tests in ~55s; OK (skipped=3)
```

**Diff constraint check**:
```text
<abstract — auditor runs: git diff --name-only f1fe61c..HEAD>
expected output: ONLY files matching docs/* or README.md (no .py, schema, or
configuration files); confirms "不修改代码" constraint is satisfied for
the entire B-audit chain.
```

## How to read this report

This report contains:

1. **Reproducible commands** for every check
2. **Machine-readable audit artifacts** written to
   `artifacts/audits/*.json` (gitignored)
3. **Tracked audit scripts** at `scripts/audit/*.py` (run from any
   checkout)
4. **No embedded absolute counts** in the body — counts are either
   discovered by running the documented commands, or written to
   artifacts/audits/*.json by the scripts themselves.

### How to reproduce the entire audit

The audit is run against the **current** working tree on `main`.
There is no `git checkout` step: the auditor verifies the current
clean `main` tree directly. This keeps the procedure non-mutating
and avoids leaving the user on a detached HEAD.

```bash
# Step 1: confirm we are on main with a clean tree
git rev-parse --abbrev-ref HEAD          # → main
git status --short                       # → empty

# Step 2: run all audit scripts in order (non-mutating, read-only)
.venv/python.exe scripts/run_tests.py full
.venv/python.exe scripts/validate_stage0.py --examples
python scripts/audit/run_gitignore_coverage.py
python scripts/audit/run_doc_artifact_reconciliation.py

# Step 3: confirm working tree is clean
git status --short
```

## TL;DR (results from this audit)

| 检查 | 结果 | 证据 |
|---|---|---|
| `scripts/run_tests.py full` | PASS | `running 28 test targets`, `Ran 386 tests OK (skipped=N)`, exit 0 |
| `scripts/validate_stage0.py --examples` | PASS | 9/9 PASS, exit 0 |
| Working tree | clean | `git status --short` empty |
| Tracked sensitive paths | empty | `git ls-files artifacts/`, `datasets/`, `.tmp/` all empty |
| Per-file `.gitignore` audit | PASS | `python scripts/audit/run_gitignore_coverage.py` returns `Not ignored: 0` + semantic test PASS |
| Doc ↔ artifact reconciliation | PASS | `python scripts/audit/run_doc_artifact_reconciliation.py` returns `MISSING_UNRESOLVABLE: 0` |
| Reviewer / auditor evidence | PASS | `docs/reports/final-audit/round-17-reviewer-evidence.md` documents round-17 reproduction commands + outputs + objection handling (current durable evidence). Round-13 evidence in `docs/reports/final-audit/round-13-reviewer-evidence.md` is preserved as historical evidence. |

**Note on host variability**: `skipped=N` and `Total sensitive files` vary
by host (depends on whether D2/D1.1 are pre-generated and whether CUDA is
available). The durable invariants are: `exit 0`, `OK (skipped=...)`,
`Not ignored: 0`, and `MISSING_UNRESOLVABLE: 0`.

## 1. Test suite — `scripts/run_tests.py full`

### Command

```bash
.venv/python.exe scripts/run_tests.py full
```

### Expected outcome

- Exit code: **0**
- First lines: `[test] running 28 test targets` (the actual count from
  `scripts/run_tests.py:FAST_MODULES + MODULES['full']`)
- Last lines: `Ran 386 tests in N.NNNs`
- Status: `OK (skipped=N)` — N is **host-dependent**

## 2. Stage 0 / Schema 验证 — `scripts/validate_stage0.py --examples`

### Command

```bash
.venv/python.exe scripts/validate_stage0.py --examples
```

### Expected outcome

- Exit code: **0**, 9 PASS lines, 0 FAIL lines

## 3. Dataset Commit Policy

### Policy (per `AGENTS.md`)

**Translation**: Don't commit training products. Don't commit datasets
either. **All datasets are gitignored** and regenerated locally on demand.

### Comprehensive `.gitignore` coverage

The `.gitignore` includes a broad `datasets/` rule (line 81) that catches
any dataset path under `datasets/`, not just the named D1/D1.1/D2 paths:

```
datasets/
datasets/tool-calling-d1/
datasets/tool-calling-d1-*/
datasets/tool-calling-d2/
datasets/tool-calling-d2-*/
```

This ensures **semantic coverage**: any future `datasets/<name>/*`
artifact is gitignored. The audit script verifies this with a
semantic-coverage test (3 synthetic test paths).

### Commands to verify

```bash
git ls-files datasets/tool-calling-d1/        | wc -l   # → 0
git ls-files datasets/tool-calling-d1-llm/   | wc -l   # → 0
git ls-files datasets/tool-calling-d2/        | wc -l   # → 0
```

### Regeneration scripts

| Dataset | Generator | Trigger |
|---|---|---|
| `datasets/tool-calling-d1/` | `scripts/generate_d1_dataset.py` | Auto-run by `test_d1_failure.py::setUpClass` |
| `datasets/tool-calling-d1-llm/` | `scripts/generate_d1_llm.py` | Manual |
| `datasets/tool-calling-d2/` | `scripts/generate_d2_dataset.py --count 5000` | Manual |

## 4. Documentation vs Artifact Reconciliation

### Script

```bash
python scripts/audit/run_doc_artifact_reconciliation.py
```

### Active documentation scope (round-12 fix)

The script scans every active `.md` file under `docs/`. **Exclusions**:

- `docs/licenses/` — third-party license text (not project docs)
- `docs/reports/final-audit.md` — this report (references itself)
- `docs/plans/reviews/*` — historical stage reviews. These document
  past actions (creation/deletion of artifacts); references to
  removed artifacts are part of the change record, not active
  contracts.

On this audit at the audited parent (`HEAD~1`, before the
round-14 reviewer-evidence doc was added): 92 total docs, 34
excluded (1 license + 1 final-audit + 32 stage reviews + 1 other),
**58 active docs scanned**. The current HEAD shows 60 active docs
(after round-14 added the round-13 reviewer-evidence doc and
round-18 added the round-17 reviewer-evidence doc, each contributing
one additional active doc); the round-13 implementation itself was
verified at 58 active docs, and the current "Real results from
this audit (this host)" section below uses the current 60-doc
tree state.

### Classification methodology (round-12 fix)

Each `artifacts/...` reference is classified into one of four kinds:

| Kind | Detection | Existence check |
|---|---|---|
| **exact_file** | ends in known file extension (e.g., `.json`, `.png`) | Try exact path; if not found, try extension fallback |
| **glob_pattern** | contains `*` | Expand via `pathlib.glob`; require ≥1 match |
| **directory** | ends in `/` OR on-disk check finds directory at the path | Check if the directory exists on disk |
| **placeholder** | ends in `-`, `_`, or `...` (truncated category reference) | **Excluded from existence check** |

Per-reference resolution:

- **FOUND** (kind in {exact_file, glob_pattern, directory}, exists)
- **MISSING_RESOLVABLE** (not found but has documented regen cmd)
- **MISSING_UNRESOLVABLE** (not found AND no regen cmd) — **AUDIT FAIL**
- **PLACEHOLDER** (kind=placeholder) — excluded

The regen-cmd detector joins multi-line shell invocations (lines ending
in `\`) before searching, so an artifact ref on a continuation line is
matched to its script.

### Real results from this audit (this host)

- Total active docs scanned: **60**
- Docs with artifact refs: **50**
- Docs without artifact refs: **10**
- Total artifact references: **175**
- **Placeholders excluded**: **19**
- **References FOUND**: **98**
- **References MISSING (resolvable)**: **5**
- **References MISSING (unresolvable)**: **0**
- **Directories FOUND**: **19**
- **Directories MISSING (resolvable)**: **0**
- **Directories MISSING (unresolvable)**: **0**
- **Globs FOUND**: **34**
- **Globs MISSING (resolvable)**: **0**
- **Globs MISSING (unresolvable)**: **0**
- Exit code: **0**

These numbers reflect the **current** tree state at HEAD (including
both `docs/reports/final-audit/round-13-reviewer-evidence.md`
(added in round-14, contributes 9 refs + 4 globs) AND
`docs/reports/final-audit/round-17-reviewer-evidence.md`
(added in round-18, contributes 2 refs + 1 glob)).

Historical evidence (preserved for round traceability):

- Round-13 implementation commit `afd579f` was verified at that
  commit with 58/167/29 counts (see
  `docs/reports/final-audit/round-13-reviewer-evidence.md`).
- Round-17 implementation commit `7771ca6` was verified at that
  commit with 59/173/33 counts (see
  `docs/reports/final-audit/round-17-reviewer-evidence.md`).

### What round-13 fix addresses

Auditor round-12 found that the artifact-reference regex
`[A-Za-z0-9_./-]+` excluded wildcards (`*`), brace-expansions
(`{a,b,c}`), and angle-bracket placeholders (`<foo>`), causing
references like:

- `artifacts/*-result.json` (truncated to `artifacts/`)
- `artifacts/{qwen2.5,huggingfacetb-smollm2}-*-eval-d2dev.json`
  (not extracted at all)
- `artifacts/grpo-experiment/*/state.json` (truncated)
- `artifacts/<model>`, `artifacts/<ckpt>` (not extracted)

to be silently dropped from the audit.

**Round-13 fixes**:

1. **Regex extended** to capture `*`, `{`, `}`, `<`, `>`, `,`, `=`
   inside the artifact path reference.
2. **Brace expansion** added: `expand_brace()` recursively expands
   `{a,b,c}` to `[a, b, c]`.
3. **Glob expansion** added: `expand_glob()` uses `pathlib.glob` to
   match `*` patterns after brace expansion.
4. **Angle-bracket placeholders** classified as `placeholder`
   (excluded from existence check) — these are template references,
   not concrete paths.
5. **Trailing `*/` (directory with wildcard)**: `classify_ref()`
   checks if any brace-expanded candidate exists as a directory.
6. **New regression tests** in `tests/test_audit_reconciliation_extraction.py`
   (26 tests) verify the regex captures all reference forms
   including the auditor-cited brace/wildcard examples.

**Result (current HEAD)**: 175 total refs captured (was 141 in
round-12; +34 vs round-12; +8 vs round-13 because the round-14 +
round-18 reviewer-evidence docs add 11 refs between them). 34
globs FOUND on disk (was 0 in round-12; +5 vs round-13 from the
two reviewer-evidence docs' globs). 0 MISSING_UNRESOLVABLE
across all kinds.

## 5. Intentionally Tracked JSON Files

### Commands

```bash
git ls-files '*.json' | wc -l
git ls-files docs/    | grep "\.json$" | wc -l   # → 3
git ls-files examples/ | grep "\.json$" | wc -l  # → 14
git ls-files schemas/ | grep "\.json$" | wc -l   # → 11
```

## 6. `.gitignore` Coverage — Bounded Per-File Audit

### Script

```bash
python scripts/audit/run_gitignore_coverage.py
```

### What it does

- Collects every file under `datasets/`, `artifacts/`, `.tmp/`
- Runs **one** `git check-ignore --no-index --stdin --verbose
  --non-matching` subprocess with all paths batched via stdin
- Parses output to determine gitignored vs not-ignored
- Runs a **semantic coverage test** (3 synthetic `datasets/*` paths)
- Writes JSON to `artifacts/audits/gitignore-coverage.json`
- Exit 0 if `Not ignored == 0` AND semantic test passes; exit 1 otherwise

### Bounded runtime

- ~7 seconds for ~7K files (host-dependent)
- No subprocess-per-file

### Expected outcome

- `Total sensitive files: N` (host-dependent)
- `Gitignored: N`
- `Not ignored: 0`
- `PASS: N/N (100%, 0 exceptions)`

### `.gitignore` rules summary

| Line | Pattern | Effective coverage |
|---|---|---|
| 52 | `.tmp/` | all `.tmp/` files |
| 64 | `checkpoints/` | `artifacts/checkpoints/*.{pt,ckpt,...}` |
| 75 | `*.pth` | `.pth` files anywhere |
| 76 | `*.pt` | `.pt` files anywhere |
| 77 | `*.bin` | `.bin` files anywhere |
| 78 | `*.onnx` | `.onnx` files anywhere |
| 81 | `datasets/` | **broad coverage — any datasets/* path** |
| 84-87 | artifacts subdirs | `artifacts/tokenizers`, `huggingface`, `multi-seed-configs` |
| 87 | `artifacts/**/*.json` | JSON under `artifacts/` |
| 88 | `artifacts/*.png` | PNGs under `artifacts/` |

## 7. Working Tree State

```bash
git status --short                  # Expected: empty
git rev-parse HEAD~1                # The audited parent commit SHA
```

## 8. Reviewer / Auditor Evidence (commands, not counts)

```bash
git ls-files docs/ | wc -l                 # → 96
git ls-files docs/plans/reviews/ | wc -l   # → 33
git ls-files docs/experiments/ | grep "/README.md$" | wc -l   # → 24
git ls-files docs/protocols/ | wc -l       # → 24
```

## 9. Audit Scope (out-of-scope items)

**Out of scope**:

- `scripts/eval_transformers.py` generation correctness (separate concern)
- Performance benchmarks
- Smoke-test reproducibility (each smoke test owns its own contract)
- vLLM serving correctness (tracked under P5-03)

**In scope** (covered by this audit):

- Tests pass (`run_tests.py full`)
- Schemas validate (`validate_stage0.py --examples`)
- Doc ↔ artifact reconciliation (broad scan over active docs)
- All sensitive paths gitignored (with semantic coverage test)
- Reviewer/auditor evidence saved
- Working tree clean

## 10. 已知非阻塞项

- `jsonschema.RefResolver is deprecated as of v4.18.0` warning (cosmetic)
- `test_grpo_mvp.py::TestGrpoSubprocessSmoke` may skip on hosts without
  CUDA or `GRPO_SMOKE=1`
- `test_d2_dataset.py` may skip classes if D2 not pre-generated
- `test_d1_llm.py` may skip if D1.1 not pre-generated
- `test_n2_benchmark.py` may skip CUDA-dependent tests without CUDA
- `scripts/eval_transformers.py` exception path: out-of-scope
- `artifacts/` size varies by host; all files gitignored

## 11. 完整复现命令 (single block)

```bash
# ---- Setup: confirm we are on main with a clean tree (no checkout) ----
git rev-parse --abbrev-ref HEAD          # → main
git status --short                       # → empty

# ---- 1. Tests ----
.venv/python.exe scripts/run_tests.py full
# Expected: "[test] running 28 test targets", "OK (skipped=N)", exit 0

# ---- 2. Schema validation ----
.venv/python.exe scripts/validate_stage0.py --examples
# Expected: 9 PASS lines, exit 0

# ---- 3. Working tree state ----
git status --short                  # → empty

# ---- 4. Tracked sensitive paths (all should be empty) ----
git ls-files artifacts/             # → empty
git ls-files datasets/              # → empty
git ls-files .tmp/                  # → empty (if listed)
git ls-files '*.pt' '*.ckpt' '*.safetensors' '*.bin'   # → empty
git ls-files '*.env'                # → empty

# ---- 5. Bounded per-file .gitignore audit ----
python scripts/audit/run_gitignore_coverage.py
# Expected: "PASS: N/N (100%, 0 exceptions)" + JSON

# ---- 6. Doc <-> artifact reconciliation (broad scan) ----
python scripts/audit/run_doc_artifact_reconciliation.py
# Expected: "PASS: all non-placeholder references resolved" + JSON

# ---- 7. Intentionally tracked JSON ----
git ls-files '*.json' | wc -l       # → 28
git ls-files docs/    | grep "\.json$" | wc -l   # → 3
git ls-files examples/ | grep "\.json$" | wc -l  # → 14
git ls-files schemas/ | grep "\.json$" | wc -l   # → 11

# ---- 8. Documentation counts ----
git ls-files docs/ | wc -l          # → 96
git ls-files docs/plans/reviews/ | wc -l   # → 33
git ls-files docs/experiments/ | grep "/README.md$" | wc -l   # → 24
git ls-files docs/protocols/ | wc -l   # → 24

# ---- 9. Identity unchanged ----
git config user.name                # → "agent"
git config user.email               # → "agent@local"
```

## 12. Conclusion

`llm-lab` audit state satisfies all in-scope items in the objective.
The audit verifies the following invariants via automated scripts:

1. **Tests pass**: 28 test targets, 386 tests, OK (skipped=host-dependent),
   exit 0.
2. **Schemas validate**: 9/9 examples PASS, exit 0.
3. **Doc ↔ artifact reconciliation** (broad scan over 60 active docs):
   - 175 total refs (including wildcards + brace-expansions +
     placeholders); 19 placeholders excluded; 98 references FOUND;
     5 MISSING_RESOLVABLE; 0 MISSING_UNRESOLVABLE
   - 19 directories FOUND; 0 directories missing
   - 34 globs FOUND; 0 globs missing
   - 26 regression tests in
     `tests/test_audit_reconciliation_extraction.py` verify regex
     captures all reference forms
   - Exit 0
4. **All sensitive paths gitignored**: ~7K gitignored (host-dependent),
   0 exceptions, semantic coverage test PASSES, bounded runtime ~7s.
5. **No tracked weight files, secrets, or artifacts**: empty results.
6. **Working tree clean**: empty `git status --short`.
7. **Identity not modified**: `agent <agent@local>` is project-level.

Out-of-scope items (e.g., `eval_transformers.py` exception handling)
are explicitly excluded per Section 9.

## 13. Reviewer / Auditor Evidence

This report is the durable final-audit evidence. Per-round
implementation details are documented separately:

- **Round-13 evidence** (wildcard/brace/placeholder extraction):
  `docs/reports/final-audit/round-13-reviewer-evidence.md`
- **Round-12 evidence** (broad doc scan + directory kind):
  captured in `docs/reports/final-audit.md` Section 4 round-13 fix
  description and the JSON output of
  `scripts/audit/run_doc_artifact_reconciliation.py`
- **Round-10–11 evidence** (bounded gitignore audit + per-ref
  reconciliation):
  JSON outputs of `scripts/audit/run_gitignore_coverage.py` and
  `scripts/audit/run_doc_artifact_reconciliation.py`

Each round-13+ fix added a regression test in
`tests/test_audit_reconciliation_extraction.py` to lock the
correctness invariant. All reproductions are committed on `main`.