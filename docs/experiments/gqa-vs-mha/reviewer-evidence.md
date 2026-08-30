# Reviewer evidence — fresh-context rehearsal (HEAD at review time)

## Metadata

- `head_at_review`: 29c4f0f9e267af221a1db446baf0657fbe70493e
- `review_timestamp_utc`: 2026-08-30T07:57:41Z
- `reviewer_model`: minimax-cn/MiniMax-M3
- `check_count`: 10 bounded checks (A-J)
- `working_tree`: empty (`git status --short --untracked-files=all` returned no output)
- `previous_durable_run`: 2026-08-30T07:38:31Z head_at_review=08fb02f — superseded by this round (3 additional commits: 551c5b3 → c76c17b → 29c4f0f)

## Context: this is the post-fix audit after auditor round 13

Auditor round 13 identified 5 specific weaknesses; this reviewer run verifies durable fixes:

1. reviewer-evidence.md head_at_review=08fb02f but current HEAD=29c4f0f (durable reviewer didn't cover the actual final tree after 3 more commits) — **FIXED by this run**: head_at_review=29c4f0f9e267af221a1db446baf0657fbe70493e.
2. The 07:52 reviewer claim was only executor-authored with no durable report — **SUPERSEDED**: this round's timestamp 07:57:41Z is the canonical durable reviewer evidence for HEAD 29c4f0f; prior ephemeral executor-authored claims are no longer relied upon.
3. Stage review lines 120/123 still contain stale "head_at_review equals HEAD^" claims (test was weakened to "ancestor" but doc wasn't reconciled) — **PARTIALLY FIXED**: see Check C below. Line 120 still says "field equals `git rev-parse HEAD^`"; line 110 still says "the artifact file (which itself is `git rev-parse HEAD^` consistent)". Doc reconciliation still pending.
4. §2.2 blanket ✅ verified overstates auditability (some rows are absence-based or unpinned master/main) — **NOT YET FIXED**: see Check B below. Current §2.2 table still marks all 19 rows as ✅ verified. Per-row audit required.
5. README.md / protocol.md missing for the no-go path — **NOT YET FIXED**: see Check D below. Neither file exists in `docs/experiments/gqa-vs-mha/`.

## Audit checks (10 bounded)

### A. HEAD + clean tree
- status: PASS
- evidence:
  - `git rev-parse HEAD` → `29c4f0f9e267af221a1db446baf0657fbe70493e`
  - `git log --oneline -10` shows:
    ```
    29c4f0f F: reviewer's round-12 follow-up caught structural bug in head_at_review test
    c76c17b F: auditor 等十二轮 4 处 weaknesses 全部 durable 修复
    551c5b3 F: auditor 等十一轮反对修复 — reviewer-evidence.md 10/10 verbatim + check count 统一
    08fb02f F: §3 LLaMA-2 residual 'expert parallelism' + §9 5 categories bilingual keywords
    90ba2fd F: auditor 等九轮反对 4 处 weaknesses 修复（factual correction + narrowed claim + 不写死数字）
    48dbc7a F: auditor 等八轮反对 3 处 weaknesses 全部修复
    c5c4522 F: stage review §Reviewer evidence 替换 stale 04dd9a1 report 为 fresh 5385968 report
    5385968 F: §7 auditor-runnable commands 统一 python3 shebang
    648ee18 F: auditor 等六轮反对 4 处 weaknesses 全部修复
    04dd9a1 F: GQA vs MHA 公开模型对比 — no-go 结论 (可行性搜索 + selftest guard + stage review)
    ```
  - `git status --short --untracked-files=all` empty
  - `date -u` → `2026-08-30T07:57:41Z`

### B. §2.2 verification status accuracy (this round's critical fix)
- status: **FAIL (current state — table overstates auditability; needs per-row revision)**
- evidence: counted 19 family rows in §2.2 (lines 54-72 of `feasibility.md`); categorized each by whether evidence includes a pinned config.json / arxiv id / GitHub commit SHA versus only `master` + paper / absence / declaration claims.

Per-row verdict (19 rows total):

| # | Line | Family | Pinned evidence present? | Verdict |
|---|---:|---|---|---|
| 1 | 54 | LLaMA-1 / LLaMA-2 / LLaMA-3 | mirror `tjluyao/llama-2-7b-hf` config.json (JSON content reproducible per row text) — **no specific commit SHA cited for the mirror**; gated repo at `main` | ⚠ not verifiably excluded (mirror reproducible but unpinned; verifier cannot point to a specific SHA for the mirror itself) |
| 2 | 55 | Mistral-7B | claims HF docs "pinned to docs commit hash by HF" — **no specific SHA given in row**; only `master` for the gated repo | ⚠ not verifiably excluded (claimed-pinned docs without concrete commit hash) |
| 3 | 56 | Mixtral 8x7B | mirror `soprasteria/Mixtral-8x7B-Instruct-v0.1-FP8` config.json at commit `c9f3de3` — **pinned SHA + URL** | ✅ verified (mirror config.json at pinned commit `c9f3de3`) |
| 4 | 57 | Qwen / Qwen2 / Qwen2.5 | only `master`; P5-04 local snapshots at `snapshots/master/` — no specific SHA in row | ⚠ not verifiably excluded (master only; P5-04 hf_expected_revision SHAs exist in `artifacts/owt-real-eval/results/*/metadata.json` but are not cited inline) |
| 5 | 58 | Phi-1 / 1.5 / 2 / 3 / 4 | only `master` + "HF model cards 全部无 GQA 声明" (absence claim) | ⚠ not verifiably excluded (master + absence claim, no pinned config.json or paper §X) |
| 6 | 59 | Gemma / Gemma2 / Gemma3 | claims HF docs "pinned to docs commit hash by HF" — **no specific SHA given in row** | ⚠ not verifiably excluded (HF docs no concrete commit hash) |
| 7 | 60 | DeepSeek-V2 / V3 | only `master` + model card "MLA 是 GQA/MHA 之外的第类" | ⚠ not verifiably excluded (master + explicit declaration, no pinned config.json) |
| 8 | 61 | OPT | `master` + "OPT 论文 §3 + facebook/opt-* configs" — **OPT paper §3 is a canonical paper reference** (Zhang et al. 2022), but config.json is at master only | ⚠ not verifiably excluded (paper §3 canonical but config unpinned at master) |
| 9 | 62 | BLOOM | only `master` + "BLOOM 论文 §2.1 全部 MHA" — paper canonical | ⚠ not verifiably excluded (master + paper §2.1 reference, no pinned SHA) |
| 10 | 63 | GPT-NeoX | only `master` + "EleutherAI 公开声明 GPT-NeoX 全部 MHA" (declaration, no §anchor) | ⚠ not verifiably excluded (master + declaration claim) |
| 11 | 64 | Falcon | only `master` + "Falcon 180B paper §3 MQA" — paper canonical | ⚠ not verifiably excluded (master + paper §3 reference) |
| 12 | 65 | Yi / Yi-Llama | only `master` + "01-ai HF repo + Yi paper" | ⚠ not verifiably excluded (master + paper) |
| 13 | 66 | Baichuan / Baichuan2 | only `master` + "baichuan-inc HF repo + Baichuan2 paper" | ⚠ not verifiably excluded (master + paper) |
| 14 | 67 | SmolLM / SmolLM2 / SmolLM3 | P5-04 `hf_expected_revision` 40-hex commit SHAs (SmolLM2-360M `a10cc1512eabd3dde888204e902eca88bddb4951`; SmolLM2-1.7B `31b70e2e869a7173562077fd711b654946d38674`) — **specific pinned SHAs** | ✅ verified (P5-04 hf_expected_revision 40-hex pinned SHAs) |
| 15 | 68 | BEE-spoke-data smol_llama | only `master` + "HF model cards 全部声明 GQA" (declaration) | ⚠ not verifiably excluded (master + declaration claim) |
| 16 | 69 | Research: Ainslie 2023 GQA paper | arxiv 2305.13245 (immutable canonical paper id) | ✅ verified (arxiv id) |
| 17 | 70 | Research: fpcsong/mha2gqa | arxiv 2412.20677 (immutable) + aclanthology 2025.findings-emnlp.467 + GitHub repo | ✅ verified (arxiv id + GitHub repo) |
| 18 | 71 | Research: SmolLM3 blog nanotron ablation | HF blog (versioned by commit; no specific commit cited) + "no checkpoint released" — explicitly absence | ⚠ not verifiably excluded (blog no pinned commit; explicitly notes "no HF repo 发布") |
| 19 | 72 | Research: shreyansh26/multihead-latent-attention | GitHub repo commit history publicly browsable (no specific SHA in row, but GitHub commit history is itself a pinned reference) | ✅ verified (GitHub commit history publicly inspectable) |

**Tally**:
- ✅ verified (genuine pinned evidence): **5 rows** — Mixtral (#3), SmolLM (#14), Ainslie 2023 (#16), fpcsong/mha2gqa (#17), shreyansh26 (#19)
- ⚠ not verifiably excluded (master + paper / absence / declaration without pinned SHA): **14 rows** — LLaMA-1/2/3 (#1), Mistral-7B (#2), Qwen (#4), Phi (#5), Gemma (#6), DeepSeek-V2 (#7), OPT (#8), BLOOM (#9), GPT-NeoX (#10), Falcon (#11), Yi (#12), Baichuan (#13), BEE-spoke (#15), SmolLM3 blog (#18)

**Conclusion**: the current §2.2 table marks all 19 rows as "✅ verified" — this is a blanket overstatement. 14/19 rows should be re-marked to "⚠ not verifiably excluded" per the table note (line 75 "Verification status column"). The footnote at line 92 "Currently all §2.2 rows are ✅ verified" should also be updated to reflect the new tally.

### C. Stage review stale HEAD^ claim reconciliation
- status: **FAIL (current state — doc still contains stale claims)**
- evidence: `grep -nE 'HEAD\^|HEAD\^|git rev-parse HEAD\^' docs/plans/reviews/stage-gqa-vs-mha-no-go.md` returns:
  - line 110: `"The 10-check list below is a local summary; the verbatim reviewer subagent output is in the artifact file (which itself is `git rev-parse HEAD^` consistent)."`
  - line 120: `"- I. **Reviewer-evidence.md metadata preserved (not regenerated mid-cycle)** — `head_at_review` field equals `git rev-parse HEAD^`; previous reviewer run audit-trail references (e.g. `c5c4522` from prior rounds) preserved for transparency."`

  - Both lines still claim `head_at_review` equals `git rev-parse HEAD^` (parent of current HEAD). The stage-gqa-vs-mha-no-go.md was written under the (since-revised) hypothesis that the durable reviewer run targets the *parent* of the current HEAD. This contradicts the orchestrator's revised test (which weakens the requirement to "ancestor" — `head_at_review` is an ancestor of current HEAD, not specifically `HEAD^`). Both lines still need reconciliation. Lines 119 and 123 (the "8/8 PASS → 10/10 PASS" historical reconciliation cell + the post-block narrative) also reference `HEAD^` implicitly via the line 120 claim and should be updated.

### D. README.md / protocol.md existence
- status: **FAIL (current state — both files missing)**
- evidence:
  ```
  $ ls -la docs/experiments/gqa-vs-mha/README.md docs/experiments/gqa-vs-mha/protocol.md
  ls: cannot access 'docs/experiments/gqa-vs-mha/README.md': No such file or directory
  ls: cannot access 'docs/experiments/gqa-vs-mha/protocol.md': No such file or directory
  ```
  - Directory listing `docs/experiments/gqa-vs-mha/` contains only:
    - `feasibility.md` (23897 bytes)
    - `reviewer-evidence.md` (9942 bytes, this file)
  - No README.md (project-level intro / index for the no-go deliverable).
  - No protocol.md (reproducible search protocol beyond §8 inline).
  - Per auditor round 13 weakness #5, both files are missing for the no-go path. Either create README.md + protocol.md, or add an explicit "not applicable — no-go conclusion has feasibility.md as the single durable artifact, no separate README/protocol needed" note in the stage review or feasibility.md.

### E. Pytest no regression
- status: PASS
- evidence:
  ```
  $ python -m pytest tests/test_gqa_vs_mha_no_go.py tests/test_gqa_vs_mha_audit_consistency.py 2>&1 | tail -3
  collected 18 items
  tests\test_gqa_vs_mha_no_go.py .......                                   [ 38%]
  tests\test_gqa_vs_mha_audit_consistency.py ...........                   [100%]
  ============================= 18 passed in 0.14s =============================
  ```
  - 18 passed (7 from `test_gqa_vs_mha_no_go.py` + 11 from `test_gqa_vs_mha_audit_consistency.py`); 0 failed; 0 skipped. Matches expected.

### F. P5-04 E selftest no regression
- status: PASS
- evidence: ran via wsl (Ubuntu-22.04) on `/mnt/c/Users/23236/repositories/llm-lab`:
  ```
  EXIT=0
  LAST:
    [PASS] test_result_SmolLM2-360M_cache_encoded_tokens_match
    [PASS] test_result_SmolLM2-360M_validate_token_cache_passes
    [PASS] test_result_SmolLM2-360M_validate_metadata_sha_matches_json
    [PASS] test_result_SmolLM2-360M_validate_cache_sha_matches_json
  [selftest] all tests PASSED
  PASS_COUNT=97
  FAIL_COUNT=0
  ```
  - `[selftest] all tests PASSED`, PASS=97, FAIL=0 — matches expected (97 PASS / 0 FAIL).

### G. Mixtral + LLaMA-2 + §9 + §10 fixes intact
- status: PASS
- evidence:
  - `grep -nE 'Mixtral.*GQA on attention layers|soprasteria/Mixtral.*c9f3de3' docs/experiments/gqa-vs-mha/feasibility.md` → line 56 contains both phrases ("MoE with **GQA on attention layers**" + "soprasteria/Mixtral-8x7B-Instruct-v0.1-FP8` config.json at commit `c9f3de3`").
  - `grep -nE 'LLaMA-2 \*\*dense\*\*' docs/experiments/gqa-vs-mha/feasibility.md` → line 54 contains "LLaMA-2 **dense** (no MoE / no expert parallelism)".
  - `grep -nE '^## 9|^## 10' docs/experiments/gqa-vs-mha/feasibility.md` → line 193 `## 9. 结论范围（narrowed claim，避开universal overreach）` and line 221 `## 10. 实证命令清单（auditor-runnable；当前 §7 之外的补充）`.
  - `grep -cE 'closed-source / gated-only|unpublicized checkpoints|private org-internal trainings|non-English / non-mainstream platforms|paper-only ablations without released safetensors' docs/experiments/gqa-vs-mha/feasibility.md` → `5` (all 5 not-auditable categories present in §9, with bilingual EN+CN keywords).

### H. Head-at-review chain
- status: PASS (this run's `head_at_review` = HEAD)
- evidence: file.metadata `head_at_review` = `29c4f0f9e267af221a1db446baf0657fbe70493e` = `git rev-parse HEAD` (verified above in Check A). This satisfies the durable reviewer requirement: the reviewer run captures the actual final tree state, not a stale parent. Note: stage review line 120 still asserts the relationship is `HEAD^` (see Check C); the file itself correctly tracks HEAD.

### I. Reconciliation table consistency
- status: PASS (current state — 3 sites all reference 10/10)
- evidence:
  ```
  $ grep -nE '10/10|8/8' docs/plans/reviews/stage-gqa-vs-mha-no-go.md docs/experiments/gqa-vs-mha/reviewer-evidence.md
  docs/plans/reviews/stage-gqa-vs-mha-no-go.md:123:fresh-context reviewer (`reviewer` subagent) VERDICT: **10/10 PASS** — 见 `docs/experiments/gqa-vs-mha/reviewer-evidence.md` §"Verbatim reviewer output (timestamp 2026-08-30T07:38:31Z)"。auditor runs `cat docs/experiments/gqa-vs-mha/reviewer-evidence.md` (HEAD-agnostic) 拿最新 reviewer evidence：file.metadata `head_at_review` 等于 parent of current HEAD（详见 reviewer-evidence.md §"About the chicken-and-egg"）。10 bounded checks A-J 完整 verbatim 列在 reviewer-evidence.md。
  docs/experiments/gqa-vs-mha/reviewer-evidence.md:109:**VERDICT: PASS — 10/10 checks satisfied**
  docs/experiments/gqa-vs-mha/reviewer-evidence.md:115:- **verdict**: PASS — 10/10
  docs/experiments/gqa-vs-mha/reviewer-evidence.md:130:| `docs/plans/reviews/stage-gqa-vs-mha-no-go.md` line 119 | 8/8 PASS | **10/10 PASS** (this round) | ✅ |
  docs/experiments/gqa-vs-mha/reviewer-evidence.md:131:| `complete_goal` verificationSummary | 10/10 PASS | **10/10 PASS** | ✅ |
  docs/experiments/gqa-vs-mha/reviewer-evidence.md:133:All three sites now reference **10/10** bounded checks, anchored on the verbatim reviewer output captured above.
  ```
  - 3 sites (`stage-gqa-vs-mha-no-go.md:123`, `reviewer-evidence.md:109/115`, reconciliation table at `reviewer-evidence.md:130/131`) all reference **10/10**.
  - The line-130 historical cell ("8/8 PASS → 10/10 PASS") is the expected exception showing the prior-round count being upgraded.
  - This round's reviewer run reports a fresh 10-check structure (A-J), consistent with the previous durable run and the stage review's 10-check list (lines 113-122).

### J. §2.2 row count + per-row verification status
- status: PASS (snapshot taken; see Check B for the table verdict)
- evidence:
  - `wc -l docs/experiments/gqa-vs-mha/feasibility.md` → `241 lines total`.
  - §2.2 family row count: **19 rows** (lines 54-72 of `feasibility.md`; §2.1 has 5 P5-04 local config rows which are separately tabulated, not part of the §2.2 search table).
  - Per-row verdicts summarized in Check B:
    - ✅ verified (genuine pinned evidence): 5 rows — Mixtral, SmolLM, Ainslie 2023, fpcsong/mha2gqa, shreyansh26.
    - ⚠ not verifiably excluded (master + paper/absence/declaration without pinned SHA): 14 rows — LLaMA-1/2/3, Mistral-7B, Qwen, Phi, Gemma, DeepSeek-V2, OPT, BLOOM, GPT-NeoX, Falcon, Yi, Baichuan, BEE-spoke, SmolLM3 blog.

## Verbatim reviewer output

> Reviewer run started at 2026-08-30T07:57:41Z.
> HEAD at review time: 29c4f0f9e267af221a1db446baf0657fbe70493e.
> Working tree: empty.
> All 10 bounded checks (A-J) executed in order.

### A. HEAD + clean tree
PASS.
- `git rev-parse HEAD` → `29c4f0f9e267af221a1db446baf0657fbe70493e`
- `git log --oneline -10` confirms 10 commits ending in `29c4f0f F: reviewer's round-12 follow-up caught structural bug in head_at_review test`.
- `git status --short --untracked-files=all` returns empty (clean tree).
- `date -u` → `2026-08-30T07:57:41Z`.

### B. §2.2 verification status accuracy (this round's critical fix)
FAIL — current state needs per-row revision.

**Methodology**: For each of the 19 §2.2 family rows, checked the "Pinned HF revision" cell for the presence of: (i) a specific 40-hex commit SHA, (ii) an arxiv id, or (iii) a GitHub repo with browsable commit history. Rows with one of these count as ✅ verified; rows with only `master` + paper / model card / absence claims count as ⚠ not verifiably excluded.

**Per-row verdict**:

1. **LLaMA-1/2/3** (line 54): mirror `tjluyao/llama-2-7b-hf` config.json reproducible but no specific SHA cited → **⚠ not verifiably excluded**.
2. **Mistral-7B** (line 55): "HF docs pinned to docs commit hash by HF" but no specific SHA → **⚠ not verifiably excluded**.
3. **Mixtral 8x7B** (line 56): mirror at commit `c9f3de3` with URL → **✅ verified (mirror config.json at pinned commit)**.
4. **Qwen / Qwen2 / Qwen2.5** (line 57): `master` only; P5-04 hf_expected_revision SHAs exist in artifacts but not cited inline → **⚠ not verifiably excluded**.
5. **Phi-1 / 1.5 / 2 / 3 / 4** (line 58): `master` + "HF model cards 全部无 GQA 声明" absence claim → **⚠ not verifiably excluded**.
6. **Gemma / Gemma2 / Gemma3** (line 59): HF docs no specific SHA → **⚠ not verifiably excluded**.
7. **DeepSeek-V2 / V3** (line 60): `master` + "model card 明明" → **⚠ not verifiably excluded**.
8. **OPT** (line 61): `master` + paper §3 (canonical) → **⚠ not verifiably excluded** (paper is canonical reference but row cites no specific SHA or paper §anchor inline; row text says "OPT 论文 + config.json 全部 MHA" without pinning either).
9. **BLOOM** (line 62): `master` + paper §2.1 → **⚠ not verifiably excluded**.
10. **GPT-NeoX** (line 63): `master` + "EleutherAI 公开声明" (declaration, no §anchor) → **⚠ not verifiably excluded**.
11. **Falcon** (line 64): `master` + Falcon 180B paper §3 MQA → **⚠ not verifiably excluded**.
12. **Yi / Yi-Llama** (line 65): `master` + Yi paper → **⚠ not verifiably excluded**.
13. **Baichuan / Baichuan2** (line 66): `master` + Baichuan2 paper → **⚠ not verifiably excluded**.
14. **SmolLM / SmolLM2 / SmolLM3** (line 67): P5-04 `hf_expected_revision` 40-hex SHAs (specific pinned commit SHAs) → **✅ verified (hf_expected_revision is pinned SHA)**.
15. **BEE-spoke-data smol_llama** (line 68): `master` + "HF model cards 全部声明 GQA" (declaration) → **⚠ not verifiably excluded**.
16. **Research: Ainslie 2023 GQA paper** (line 69): arxiv 2305.13245 (immutable) → **✅ verified (arxiv id)**.
17. **Research: fpcsong/mha2gqa** (line 70): arxiv 2412.20677 (immutable) + GitHub repo → **✅ verified (arxiv id + GitHub)**.
18. **Research: SmolLM3 blog nanotron ablation** (line 71): HF blog no pinned commit + explicitly "no checkpoint released" → **⚠ not verifiably excluded**.
19. **Research: shreyansh26/multihead-latent-attention** (line 72): GitHub commit history publicly inspectable (no specific SHA cited but repo browseable) → **✅ verified (GitHub history public)**.

**Tally**: 5 rows ✅ verified; 14 rows ⚠ not verifiably excluded.

**Required action**: Update §2.2 table to mark the 14 ⚠ rows appropriately. The footnote at line 92 ("Currently all §2.2 rows are ✅ verified") should be updated to reflect the new tally (5 ✅ / 14 ⚠). The "Verification status" column note (line 75) already defines both states correctly; only the table cells need updating.

### C. Stage review stale HEAD^ claim reconciliation
FAIL — current state has stale claims at lines 110, 120.

- line 110: `"which itself is `git rev-parse HEAD^` consistent"` → should be revised to "which itself is `git rev-parse HEAD` consistent" or "ancestor of current HEAD" (the test was weakened to "ancestor" but doc wasn't reconciled).
- line 120: `"`head_at_review` field equals `git rev-parse HEAD^`"` → same revision needed.
- The line-119 / line-123 narrative is downstream of line 120's claim and should also be reviewed for consistency.

### D. README.md / protocol.md existence
FAIL — both files missing.

```
$ ls -la docs/experiments/gqa-vs-mha/README.md docs/experiments/gqa-vs-mha/protocol.md
ls: cannot access 'docs/experiments/gqa-vs-mha/README.md': No such file or directory
ls: cannot access 'docs/experiments/gqa-vs-mha/protocol.md': No such file or directory
```

Directory contains only `feasibility.md` (23897 bytes) and `reviewer-evidence.md` (9942 bytes). Either create README.md + protocol.md, or add an explicit "not applicable" note in stage review / feasibility.md §1.

### E. Pytest no regression
PASS.
```
$ python -m pytest tests/test_gqa_vs_mha_no_go.py tests/test_gqa_vs_mha_audit_consistency.py 2>&1 | tail -3
collected 18 items
tests\test_gqa_vs_mha_no_go.py .......                                   [ 38%]
tests\test_gqa_vs_mha_audit_consistency.py ...........                   [100%]
============================= 18 passed in 0.14s =============================
```

### F. P5-04 E selftest no regression
PASS.
```
EXIT=0
LAST:
  [PASS] test_result_SmolLM2-360M_cache_encoded_tokens_match
  [PASS] test_result_SmolLM2-360M_validate_token_cache_passes
  [PASS] test_result_SmolLM2-360M_validate_metadata_sha_matches_json
  [PASS] test_result_SmolLM2-360M_validate_cache_sha_matches_json
[selftest] all tests PASSED
PASS_COUNT=97
FAIL_COUNT=0
```

### G. Mixtral + LLaMA-2 + §9 + §10 fixes intact
PASS.
- Mixtral line 56: contains both "GQA on attention layers" and "soprasteria/Mixtral-8x7B-Instruct-v0.1-FP8` config.json at commit `c9f3de3`".
- LLaMA-2 line 54: contains "LLaMA-2 **dense** (no MoE / no expert parallelism)".
- §9 starts at line 193, §10 starts at line 221.
- 5 not-auditable categories present in §9: closed-source / gated-only, unpublicized checkpoints, private org-internal trainings, non-English / non-mainstream platforms, paper-only ablations without released safetensors.

### H. Head-at-review chain
PASS.
- file.metadata `head_at_review` = `29c4f0f9e267af221a1db446baf0657fbe70493e` = `git rev-parse HEAD`.
- This is the durable reviewer for THIS round (covering HEAD 29c4f0f). Prior durable run (2026-08-30T07:38:31Z, head_at_review=08fb02f) covered up to that commit; the 3 commits between (551c5b3, c76c17b, 29c4f0f) are now durably covered.

### I. Reconciliation table consistency
PASS.
- 3 sites all reference **10/10**:
  - `stage-gqa-vs-mha-no-go.md:123` → "10/10 PASS"
  - `reviewer-evidence.md:109/115` → "10/10 checks satisfied" / "10/10"
  - `reviewer-evidence.md:130/131` reconciliation table → both "10/10 PASS" with checkmark ✅
- The line-130 historical cell ("8/8 PASS → 10/10 PASS") is the expected historical exception showing prior-round count being upgraded.
- This round's reviewer run also reports 10 checks A-J, consistent with the previous durable run.

### J. §2.2 row count + per-row verification status
PASS — snapshot taken; see Check B for full per-row verdicts.
- Total file: 241 lines.
- §2.2 family rows: 19 (lines 54-72).
- ✅ verified (genuine pinned evidence): 5 rows.
- ⚠ not verifiably excluded: 14 rows.

## VERDICT

**FAIL** — 3 of 10 bounded checks did not pass this round:

1. **Check B (§2.2 verification status accuracy)** — FAIL: 14/19 rows currently marked "✅ verified" should be re-marked to "⚠ not verifiably excluded" per the per-row evidence audit. Table needs revision.
2. **Check C (Stage review stale HEAD^ claim reconciliation)** — FAIL: lines 110 and 120 of `stage-gqa-vs-mha-no-go.md` still assert `head_at_review` equals `git rev-parse HEAD^` instead of `git rev-parse HEAD` or "ancestor of current HEAD". Doc reconciliation pending.
3. **Check D (README.md / protocol.md existence)** — FAIL: neither file exists; either create them or add explicit "not applicable" note.

Checks A, E, F, G, H, I, J PASS.

**Required durable fixes** (for next orchestrator cycle):
- Update §2.2 table in `feasibility.md`: 14 rows marked ⚠; footnote updated to "5 ✅ / 14 ⚠".
- Update `stage-gqa-vs-mha-no-go.md` lines 110 and 120 to remove stale `HEAD^` claim.
- Either create `docs/experiments/gqa-vs-mha/README.md` and `docs/experiments/gqa-vs-mha/protocol.md`, or add a note explicitly stating they are not applicable for the no-go deliverable.

After these 3 fixes are durable, the next reviewer run should expect all 10 checks to PASS and can confirm the post-fix audit as complete.

