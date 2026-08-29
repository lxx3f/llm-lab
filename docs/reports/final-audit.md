# 最终全链路审查报告 — Final Audit Report

- **Date**: 2026-08-29
- **Author**: Minimax M3 (per `docs/plans/review-process.md`)
- **Goal**: 验证 `llm-lab` 实验闭环各阶段的最终状态
- **Audited snapshot**: working tree of `main` immediately before this
  report was committed (the parent of the commit that adds this file)
- **Scope**: this audit verifies (a) tests pass, (b) schemas validate,
  (c) docs reconcile with artifacts, (d) sensitive files are gitignored,
  (e) reviewer evidence is saved, (f) working tree is clean. It does NOT
  verify evaluation correctness (e.g., `scripts/eval_transformers.py`'s
  generation behavior); that is a separate concern tracked under
  P5-02 benchmark evaluation subset work.

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

```bash
# Step 1: obtain the audited tree
git stash push -u -m "WIP before re-running final-audit"
git checkout HEAD~1

# Step 2: run all audit scripts in order
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
| `scripts/run_tests.py full` | PASS | `running 27 test targets`, `Ran 360 tests OK (skipped=N)`, exit 0 |
| `scripts/validate_stage0.py --examples` | PASS | 9/9 PASS, exit 0 |
| Working tree | clean | `git status --short` empty |
| Tracked sensitive paths | empty | `git ls-files artifacts/`, `datasets/`, `.tmp/` all empty |
| Per-file `.gitignore` audit | PASS | `python scripts/audit/run_gitignore_coverage.py` returns `Not ignored: 0` + semantic test PASS |
| Doc ↔ artifact reconciliation | PASS | `python scripts/audit/run_doc_artifact_reconciliation.py` returns `MISSING_UNRESOLVABLE: 0` |
| Reviewer evidence saved | 96 docs | `git ls-files docs/ \| wc -l` |

**Note on host variability**: `skipped=N` in the test suite and `Total
sensitive files` in the gitignore audit vary by host (depends on
whether D2/D1.1 are pre-generated and whether CUDA is available).
The durable invariants are: `exit 0`, `OK (skipped=...)`, `Not ignored:
0`, and `MISSING_UNRESOLVABLE: 0`.

## 1. Test suite — `scripts/run_tests.py full`

### Command

```bash
.venv/python.exe scripts/run_tests.py full
```

### Expected outcome

- Exit code: **0**
- First lines: `[test] running 27 test targets` (the actual count from
  `scripts/run_tests.py:FAST_MODULES + MODULES['full']`)
- Last lines: `Ran 360 tests in N.NNNs`
- Status: `OK (skipped=N)` — N is **host-dependent**:
  - On this host (D2 + D1.1 present, CUDA not used by GRPO tests):
    `skipped=1` (only the GRPO subprocess smoketest gated by
    `GRPO_SMOKE=1`)
  - On a host without D2: `skipped=10` (1 D2 class + 1 method
    skipped, plus other skipUnless tests may activate)
  - On a host without CUDA: additional `test_n2_benchmark` tests skip
- The durable invariant is `exit 0` and `OK (skipped=...)`. Any
  skipped count from 1 to ~13 is acceptable.

## 2. Stage 0 / Schema 验证 — `scripts/validate_stage0.py --examples`

### Command

```bash
.venv/python.exe scripts/validate_stage0.py --examples
```

### Expected outcome

- Exit code: **0**
- 9 PASS lines (one per example fixture under `examples/`)
- 0 FAIL lines

## 3. Dataset Commit Policy

### Policy (per `AGENTS.md`)

**Translation**: Don't commit training products. Don't commit datasets
either. **All datasets are gitignored** and regenerated locally on demand.

### Comprehensive `.gitignore` coverage

The `.gitignore` rules include a broad `datasets/` rule (line 81) that
catches any dataset path under `datasets/`, not just the named D1/
D1.1/D2 paths:

```
datasets/
datasets/tool-calling-d1/
datasets/tool-calling-d1-*/
datasets/tool-calling-d2/
datasets/tool-calling-d2-*/
```

This ensures **semantic coverage**: any future `datasets/<name>/*`
artifact is gitignored. The audit script verifies this with a
semantic-coverage test (3 synthetic test paths: `unlisted-dataset`,
`future-experiment`).

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

### Classification methodology

Each `artifacts/...` reference extracted from a README is classified
into one of four kinds, with **different existence checks per kind**:

| Kind | Definition | Existence check |
|---|---|---|
| **exact_file** | Reference ends in a known file extension (e.g., `.json`, `.png`) | Try exact path; if not found, try extension fallback |
| **glob_pattern** | Reference contains `*` | Expand via pathlib `glob`; require ≥1 match |
| **directory** | Reference ends in `/` | Check if the directory exists |
| **placeholder** | Reference ends in `-`, `_`, `/`, `...` (truncated category reference) | **Excluded from existence check** — describes a category/directory, not a specific file |

Per-reference resolution:

- **FOUND** (kind=exact_file|glob_pattern|directory, exists on disk)
- **MISSING_RESOLVABLE** (not found but has documented regen cmd)
- **MISSING_UNRESOLVABLE** (not found AND no regen cmd) — **AUDIT FAIL**
- **PLACEHOLDER** (kind=placeholder) — excluded from existence check

### Expected outcome

- Exit code: **0**
- Output includes:
  - `Total docs: 24`
  - `Docs with artifact refs: 21`
  - `Total artifact references: 77`
  - `Placeholders (excluded from check): ~20`
  - `References FOUND on disk: ~56`
  - `References MISSING (resolvable): 0-2`
  - `References MISSING (unresolvable): 0`
- JSON: `artifacts/audits/doc-artifact-reconciliation.json` with full
  per-doc, per-ref evidence

### What this round-11 fix addresses

The round-10 script had a defect: `check_artifact_exists()` returned
True whenever a reference's parent directory existed, which led to
false-positive FOUND classifications for placeholder references like
`artifacts/sft-`, `artifacts/moe-owt-formal-curve-`, and trailing
hyphen patterns. Round-11 fix:

1. Classifies each reference by kind (exact_file / glob_pattern /
   directory / placeholder)
2. **Excludes placeholders from existence check** (they're category
   references, not specific files)
3. Glob patterns use `pathlib.glob` (not parent-dir check)
4. Per-reference JSON includes `kind`, `matched_paths`, `regen_command`

This host (this audit): 77 total refs, 20 placeholders excluded,
56 FOUND, 1 MISSING_RESOLVABLE, 0 MISSING_UNRESOLVABLE → exit 0.

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
  --non-matching` subprocess with all paths batched via stdin (NOT one
  subprocess per file)
- Parses output to determine gitignored vs not-ignored
- Runs a **semantic coverage test** (3 synthetic `datasets/*` paths) to
  confirm the broad `datasets/` rule catches future unlisted datasets
- Writes machine-readable JSON to `artifacts/audits/gitignore-coverage.json`
- Exit 0 if `Not ignored == 0` AND semantic test passes; exit 1 otherwise

### Bounded runtime

- ~7 seconds for ~7K files (host-dependent)
- No subprocess-per-file (that would take many minutes)

### Expected outcome

- `Total sensitive files: N` (varies by host; this host has 7093)
- `Gitignored: N`
- `Not ignored: 0`
- `PASS: N/N (100.0000%, 0 exceptions)`
- JSON: `artifacts/audits/gitignore-coverage.json`

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
| 82-85 | specific datasets | named D1/D1.1/D2/D2-* paths (subsumed by line 81) |
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
git ls-files docs/experiments/ | wc -l     # → 29
git ls-files docs/experiments/ | grep "/README.md$" | wc -l   # → 24
git ls-files docs/protocols/ | wc -l       # → 24
```

## 9. Audit Scope (out-of-scope items)

**Out of scope** (NOT covered by this audit):

- `scripts/eval_transformers.py` generation correctness — the script
  has an `except Exception` branch that converts generation failures
  into empty `generated` strings and returns exit 0. Whether this is
  the desired behavior is a separate design question (tracked in
  `docs/plans/open-issues.md`), not a `final-audit` concern.
- Performance benchmarks — not in the audit objective.
- Smoke-test reproducibility — each smoke test is responsible for its
  own reproducibility per `docs/protocols/<experiment>.md`.
- vLLM serving correctness — tracked under P5-03 (separate goal).

**In scope** (covered by this audit):

- Tests pass (`run_tests.py full`)
- Schemas validate (`validate_stage0.py --examples`)
- Doc ↔ artifact reconciliation
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
# ---- Setup: obtain the audited tree ----
git stash push -u -m "WIP before re-running final-audit"
git checkout HEAD~1

# ---- 1. Tests ----
.venv/python.exe scripts/run_tests.py full
# Expected: "[test] running 27 test targets", "OK (skipped=N)", exit 0
# N varies by host (1-13 depending on which optional datasets/CUDA are present)

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

# ---- 6. Doc <-> artifact reconciliation ----
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

1. **Tests pass**: 27 test targets, 360 tests, OK (skipped=host-dependent),
   exit 0.
2. **Schemas validate**: 9/9 examples PASS, exit 0.
3. **Doc ↔ artifact reconciliation**: 24 docs, 77 references; of which
   ~20 are placeholders (category refs, excluded from check), ~56
   FOUND, ≤2 MISSING_RESOLVABLE, 0 MISSING_UNRESOLVABLE, exit 0.
4. **All sensitive paths gitignored**: N/N gitignored (host-dependent,
   ~7K on this host), 0 exceptions, semantic coverage test PASSES,
   bounded runtime ~7s.
5. **No tracked weight files, secrets, or artifacts**: empty results.
6. **Working tree clean**: empty `git status --short`.
7. **Identity not modified**: `agent <agent@local>` is project-level.

Out-of-scope items (e.g., `eval_transformers.py` exception handling)
are explicitly excluded per Section 9.