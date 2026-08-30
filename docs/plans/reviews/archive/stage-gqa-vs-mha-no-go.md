# Stage review — list item F: GQA vs MHA 公开模型对比 (narrowed verified-scope no-go + incomplete feasibility review)

> 任务来源：list item F（active goal，2026-08-30 启动）。
> 结论（双层 scope）：**narrowed verified-scope no-go** — 在 5 个 fully independently verified rows
> (Mixtral mirror / SmolLM P5-04 SHAs / Ainslie 2023 / fpcsong/mha2gqa / shreyansh26) 内无可审计的同
> base GQA/MHA 公开模型对（durable）。**14 feasibility-lead rows 是 needs-more-work leads，不是
> verified exclusions**；full audit closure incomplete，本 deliverable 不是 complete 19-candidate no-go review。
> 本 stage review 在 HEAD 见 `git rev-parse HEAD` (auditor-runnable)。
> 完整搜索过程与候选清单见 `docs/experiments/gqa-vs-mha/feasibility.md`。

## Scope

按 list item F objective：

> GQA vs MHA 公开模型可行性与对比实验：先寻找可审计的同 base、不同 attention 结构模型对；若找到，复用 P5-02/P5-04 的历史 90 样本 benchmark、Transformers/vLLM backend 与 P1-05/P2 evaluator，完成 latency、throughput、reward_binary、reward_layered 对比，并输出 README、protocol、stage review、hash/revision/schema/selftest 证据；若找不到可靠的同 base GQA/MHA 配对，则完成明确的 feasibility/no-go 审查，不把不同 base 模型差异宣称为 GQA 收益。

## 交付清单

- `docs/experiments/gqa-vs-mha/feasibility.md` — size is auditor-runnable via `wc -c docs/experiments/gqa-vs-mha/feasibility.md`（~30 KB 级；含 §1 判定标准、§2 候选扫描 19 rows 5 ✅+14 ⚠、§3 否决理由、§4 影响、§5 结论双层 scope、§6 P5-04 5 model 实证、§7/§10 实证命令、§8 reproducible search、§9 narrowed claim 双层 scope、§11 schema evidence、§Per-row classification appendix）。
- `tests/test_gqa_vs_mha_no_go.py` — 7 个 pytest 断言；guard 守住 no-go 结论与硬性判定标准。
- `docs/plans/roadmap.md` 候选 1 行追加状态更新段（line 101）：no-go + 指向 feasibility + stage review。
- `docs/plans/open-issues.md` 暂不处理段记录本 no-go 结论链接（见下文 §Risks）。

## Pre-conditions verified

- P5-04 5 个公开 model config.json 实际读取，attention 类型确认（4 个 GQA + 1 个 MHA，但 size/layers/hidden 完全不同 → 不能作为同 base 对）。
- Web 调研覆盖 ~15 个主流 LLM 家族 + 4 类 research artifacts（Ainslie 2023 论文 uptraining、fpcsong 2025 mha2gqa HF 转换 checkpoint、SmolLM3 blog nanotron ablation、shreyansh26/multihead-latent-attention reference implementation）。
- Hard-rule #5（"不把不同 base 模型差异宣称为 GQA 收益"）严格遵守：本结论仅声明"无可审计同 base 对"，不对 GQA vs MHA 性能下任何结论。

## Verification (auditor-runnable commands)

```text
git rev-parse HEAD                            # 当前 commit
ls docs/experiments/gqa-vs-mha/feasibility.md  # 必须存在
ls docs/plans/reviews/stage-gqa-vs-mha-no-go.md # 必须存在
ls tests/test_gqa_vs_mha_no_go.py             # 必须存在
python -m pytest tests/test_gqa_vs_mha_no_go.py -v  # 7 断言 PASS
python scripts/eval_owt_real.py --selftest   # 仍 97 PASS / 0 FAIL (98 PASS lines: 97 individual + 1 summary)（不影响 P5-04/E 评测）
```

## Per-model evidence（同 base 硬性判定标准 vs 已扫描候选）

| 判定标准 # | 标准 | 命中候选 | 否决理由 |
|---|---|---|---|
| 1 | 同一组织同一训练计划 | 无 | 所有公开同家族 model 都已固定 attention 类型 |
| 2 | 唯一架构差异 = num_kv_heads | 无 | 同家族内 size 不同（layers/hidden/vocab 至少一个差异） |
| 3 | 其他 hyperparam 一致 | 无 | size 不同 → hidden/layers 必不同 |
| 4 | 公开可下载 safetensors | — | 通过；但仍需 1+2+3 才能用 |
| 5 | 可绑 HF commit SHA | — | P5-04 已暴露 master vs exact commit 问题 |

## Risks recorded

- **误判风险**：未来若有 public 同 base MHA/GQA twin 发布，本结论需同步推翻（test_gqa_vs_mha_no_go.py 需更新）。
- **替代路径（不在本目标 scope 内）**：自训练一对 from-scratch same-base 小模型（需 8×H100 + 数千 GPU-hour）或复现 Ainslie 2023 uptraining（需 50K-100K token + 数天 GPU）。两者均远超当前 lab 算力。
- **隐式验证**：P5-04 5 model 评测（SmolLM2-360M GQA、SmolLM2-1.7B MHA、Qwen2.5 系列 GQA）已隐式证明 P5-02 backend（`scripts/eval_transformers.py` + 共享 `_hf_backend.load_causal_lm_model`）能正确处理 MHA 与 GQA 两类 forward；该兼容性已被 E 任务的 5 模型 97 PASS selftest 覆盖。

## Open-issues 暂不处理段更新

`docs/plans/open-issues.md` 暂不处理段追加（双层 scope）：

```
- **GQA vs MHA 公开模型对比**（list item F，2026-08-30）：可行性搜索结论 narrowed verified-scope no-go。
  在 5 个 fully independently verified rows 内公开权重层面无可审计的同 base MHA/GQA 双版本（durable）；
  14 feasibility-lead rows 是 needs-more-work leads（unpinned master / absence claims），不是 verified exclusions，
  full audit closure incomplete。后续若继续需自训练或 uptraining，均超出当前 lab 算力预算。
  详见 `docs/experiments/gqa-vs-mha/feasibility.md` 与 `docs/plans/reviews/stage-gqa-vs-mha-no-go.md`。
```

## Reviewer evidence (external artifact, HEAD-agnostic)

The latest fresh-context reviewer report is stored in a **separate artifact file**:

> **`docs/experiments/gqa-vs-mha/reviewer-evidence.md`**

Auditor runs the following commands to inspect the live report:

```bash
# HEAD-agnostic: resolves to whatever commit is currently checked out
git rev-parse HEAD                                # current commit SHA
cat docs/experiments/gqa-vs-mha/reviewer-evidence.md    # full reviewer report
```

The artifact contains its own provenance metadata (so the report's "current revision" claim is independently verifiable):

- `head_at_review` — the exact `git rev-parse HEAD` SHA the reviewer ran on
- `review_timestamp_utc` — UTC wall-clock when the reviewer ran
- `reviewer_model` — `minimax-cn/MiniMax-M3` per `docs/plans/review-process.md`
- `working_tree` — output of `git status --short --untracked-files=all`
- 10 bounded checks A–J with status + evidence quote (the current canonical structure; the legacy "A–H" / 8-check structure was the round-8 reviewer format that was upgraded to A–J / 10-check structure in round 11 and has since been the canonical structure for all subsequent rounds)
- VERDICT line at the end

### Why the artifact lives outside this stage review

Earlier revisions of this file embedded the raw reviewer report inline. Auditor
flagged that as a chicken-and-egg issue: the commit that embedded the report
necessarily advanced HEAD past `head_at_review`. Storing the report in an
external artifact, generated by a fresh-context reviewer AFTER the stage
review text is finalised, breaks the cycle — the auditor can read the
artifact's `head_at_review` and confirm it equals the HEAD the reviewer
actually saw.

### Recompute protocol

If the auditor wants a fresh re-run (e.g. after a fix), the orchestrator
spawns one `reviewer` subagent and writes its output to `docs/experiments/
gqa-vs-mha/reviewer-evidence.md`, then commits that single file. The
artifact file's `head_at_review` will equal the parent commit's SHA (the
state reviewer saw); the next commit's HEAD advances by one. Auditor runs
`git log --oneline -3` to confirm the artifact commit follows the
`head_at_review` parent.

### Reviewer 10 bounded checks (A-J, mirrors reviewer-evidence.md)

> Source of truth: `docs/experiments/gqa-vs-mha/reviewer-evidence.md` §"Audit checks (10 bounded)" + §"Verbatim reviewer output (timestamp 2026-08-30T10:15:57Z)". The 10-check list below is a local summary; the verbatim reviewer subagent output is in the artifact file. `head_at_review = 4151154445faa6eb5a608d4ece0fa9e46e334470` (the commit the reviewer subagent ran on at timestamp 2026-08-30T10:15:57Z; this SHA equals `git rev-parse HEAD^` — i.e. the parent of the commit that contains reviewer-evidence.md — NOT current `git rev-parse HEAD`).

- A. **HEAD + clean tree** — `git rev-parse HEAD` returns a valid SHA; `git status --short --untracked-files=all` is empty.
- B. **Mixtral factual correction** — feasibility.md §2.2 row describes Mixtral 8x7B as **GQA (32/8) + MoE (8 experts, top-2 routing)**, with mirror config.json `soprasteria/Mixtral-8x7B-Instruct-v0.1-FP8` at commit `c9f3de3` (URL in §2.2).
- C. **LLaMA-2 factual correction** — feasibility.md §2.2 row-54 + §3 row-107 both state LLaMA-2 is **dense (no MoE / no expert parallelism)**, with explicit `7B/13B MHA vs 70B GQA` size confounding (hidden=4096/5120/8192, layers=32/40/80).
- D. **Reviewer-evidence.md auditor-run commands** — every audit check uses `auditor-run:` commands (no concrete file-size / line-count claims baked in); ≥ 8 hits confirmed.
- E. **§9 narrowed claim + §10 audit commands** — feasibility.md has §9 (narrowed claim, 5 not-auditable categories with bilingual EN+CN keywords: closed-source / unpublicized / private org-internal / non-English / paper-only ablations) + §10 (audit commands).
- F. **§2.2 family URLs + access dates (no regression)** — feasibility.md §2.2 contains ≥ 18 URLs and ≥ 5 access-date markers; every row has URL + access date + evidence.
- G. **Selftest no regression** — `python -m pytest tests/test_gqa_vs_mha_no_go.py -v` → 7 passed / 0 skipped / 0 failed.
- H. **P5-04 E selftest no regression** — `python scripts/eval_owt_real.py --selftest` (note: portability — auditors may need to substitute `python` if their environment exposes `python` rather than `python`) → `[selftest] all tests PASSED`, PASS count ≥ 95 (97 expected), FAIL = 0.
- I. **Reviewer-evidence.md metadata preserved (not regenerated mid-cycle)** — `head_at_review` field equals the parent of the commit that most recently modified `docs/experiments/gqa-vs-mha/reviewer-evidence.md` (the durable invariant enforced by `tests/test_gqa_vs_mha_audit_consistency.py::test_head_at_review_equals_head_parent`); the field is updated each round when a fresh reviewer runs and the executor commits the rewritten reviewer-evidence.md file as a follow-up commit; previous reviewer run audit-trail references preserved for transparency.
- J. **Consistency §2.2 row-54 ↔ §3 row-107** — feasibility.md LLaMA-2 dense + no MoE + no expert parallelism + size confounding is consistent between §2.2 row-54 (LLaMA-2 row) and §3 row-107 (failure-analysis table).

fresh-context reviewer (`reviewer` subagent) VERDICT: **10/10 PASS** — 见 `docs/experiments/gqa-vs-mha/reviewer-evidence.md` §"Verbatim reviewer output (timestamp 2026-08-30T10:15:57Z)" §"VERDICT"。auditor runs `cat docs/experiments/gqa-vs-mha/reviewer-evidence.md` (HEAD-agnostic) 拿最新 reviewer evidence；file.metadata `head_at_review = 4151154445faa6eb5a608d4ece0fa9e46e334470` = the commit the reviewer subagent ran on (at timestamp 2026-08-30T10:15:57Z) = `git rev-parse HEAD^` (parent of the commit that contains reviewer-evidence.md). `git rev-parse HEAD` continues to advance past `head_at_review` whenever the repo receives new commits; the durable invariant is `head_at_review == (parent of the commit that last modified reviewer-evidence.md)`, which the audit-consistency test enforces via `git rev-parse $(git log -1 --format=%H -- docs/experiments/gqa-vs-mha/reviewer-evidence.md)^`. 10 bounded checks A-J 完整 verbatim 列在 reviewer-evidence.md.

#

## Verification（final，auditor-runnable commands）

```text
git rev-parse HEAD                            # 当前 commit
ls -la docs/experiments/gqa-vs-mha/feasibility.md docs/plans/reviews/stage-gqa-vs-mha-no-go.md tests/test_gqa_vs_mha_no_go.py
python -m pytest tests/test_gqa_vs_mha_no_go.py -v  # 7 断言全 PASS
python scripts/eval_owt_real.py --selftest   # 97 PASS / 0 FAIL（无 regression）
grep -n "状态更新（list item F" docs/plans/roadmap.md   # 1 行 no-go 状态
```

## END STAGE REVIEW.

## Schema evidence

参见 `docs/experiments/gqa-vs-mha/feasibility.md` §11 — schema artifact `schemas/evaluation_result.schema.json` + sample `examples/evaluation_results/sample-no-go-result.json` + auditor-runnable validation command + recorded output (2026-08-30T10:15Z, HEAD 4151154).
