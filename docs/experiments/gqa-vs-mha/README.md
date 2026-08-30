# GQA vs MHA 公开模型对比 — README (no-go path)

> **No-go conclusion**: 无可审计的同 base GQA/MHA 公开模型对。本目录只承载
> feasibility/no-go 审查的 deliverable，不包含 4 轴对比实验的 README（因为
> no-go 分支下 4 轴对比不被执行，参见 list item F objective）。

## 目录内容

- **`feasibility.md`** — 主要 deliverable。§1 同 base 硬性判定标准 5 项、§2 候选扫描
  (15 LLM family + 3 research artifact，每行 URL + access date + pinned commit SHA 或
  mirror config.json + Verification status 列)、§3 否决理由汇总、§9 结论范围
  (narrowed claim)、§10 实证命令清单。
- **`reviewer-evidence.md`** — fresh-context reviewer 的 durable artifact。每个 round
  的 reviewer run 在 `head_at_review` 字段记录 HEAD SHA + timestamp + 10 bounded
  checks A-J 完整 evidence + VERDICT。
- **`stage-gqa-vs-mha-no-go.md`** (在 `docs/plans/reviews/`) — stage review，含
  §Per-model evidence + §Reviewer evidence + §Verification (HEAD-agnostic) + §Reconciliation。

## 关键 evidence (复核路径)

```bash
# 1. §2.2 table — 15 family + 3 research artifact，每行 URL + access date + Verification status
grep -cE '^\|' docs/experiments/gqa-vs-mha/feasibility.md    # 19 data rows + header + separator
grep -cE 'https?://' docs/experiments/gqa-vs-mha/feasibility.md    # ≥ 18 URLs
grep -cE '✅ verified|⚠ not verifiably excluded' docs/experiments/gqa-vs-mha/feasibility.md
# Expected: 5 ✅ verified + 14 ⚠ not verifiably excluded

# 2. P5-04 本地 5 config 实证
python3 -c "from pathlib import Path; import json
for cfg in sorted(Path('artifacts/owt-real-eval/models/models').rglob('config.json')):
    c = json.loads(cfg.read_text(encoding='utf-8'))
    h = c.get('num_attention_heads'); kv = c.get('num_key_value_heads', h)
    print(f'{cfg.parent.parent.parent.name}: heads={h} kv={kv} -> {\"GQA\" if kv != h else \"MHA\"}')"

# 3. Pytest guard suite
python -m pytest tests/test_gqa_vs_mha_no_go.py tests/test_gqa_vs_mha_audit_consistency.py -v
# Expected: 18 passed (7 keyword/family smoke tests + 11 structural audit-consistency tests)

# 4. Reviewer evidence
cat docs/experiments/gqa-vs-mha/reviewer-evidence.md
# Look at: head_at_review (must be ancestor of HEAD via git merge-base --is-ancestor)
#         review_timestamp_utc (most recent reviewer run)
#         Audit checks (10 bounded) + Verbatim reviewer output sections
```

## 结论范围（与 feasibility.md §9 一致）

本 README 仅声明：在 `feasibility.md` §2.2 列出的 15 LLM family + 3 research artifact
这一 **audited candidate set** 内，没有可审计的同 base GQA/MHA 双版本公开模型对。

本文不主张：(a) 闭源 / gated-only 仓库中存在未公开的同 base 对；(b) HF 历史上发布
后又删除的 checkpoint 包含同 base 对；(c) 私人 org-internal 训练中存在同 base 对
（即便训练过但未发布，本搜索无法检测）；(d) 非英语 / 非主流平台（GitLab、ModelScope
私有仓库、国内魔搭社区非公开仓库等）有同 base 对；(e) 仅在 paper 表格中报告的
ablation、未发布 safetensors 的 paper-only 权重包含同 base 对。

如果未来发现满足 `feasibility.md` §1.2 硬性判定标准 1-5 的同 base 公开模型对
（不论是现有 family 的新 release 还是新 family 发布），本 no-go 结论需被推翻。
`tests/test_gqa_vs_mha_no_go.py` 与 `tests/test_gqa_vs_mha_audit_consistency.py`
守住本结论不被意外削弱。

## Roadmap 影响

- `docs/plans/roadmap.md` 候选 1 行追加 "状态更新（list item F，2026-08-30）no-go"。
- `docs/plans/open-issues.md` 暂不处理段新增 GQA vs MHA no-go 结论条目。
- `docs/experiments/gqa-vs-mha/protocol.md` 提供 reproducible search protocol (在 §8
  内已 inline，避免文件重复)。

## 后续若要继续 GQA vs MHA 对比（不在本目标 scope 内）

1. **自训练一对 from-scratch same-base 模型**：从同一随机种子、同一训练数据、同一
   token 量、唯一差异为 `num_kv_heads` 训练两个 ~125M 模型。Ainslie 2023 / SmolLM3 团队
   做过的 recipe，但需 8×H100 + 数千 GPU-hour，远超本 lab 算力预算。
2. **复现 uptraining**：从 `meta-llama/Llama-2-7b-hf` 出发做 MHA → GQA-{4,8,16} 转换
   + 5% 额外预训练（按 fpcsong 2025 论文 recipe）。需要 ~50K-100K 训练 token + 数天 GPU。
   也远超本 lab 当前算力。