# Stage Review — Roadmap 更新（2026-08-29，List item C 完成）

## 范围（Scope）

本 stage review 覆盖 `docs/plans/roadmap.md` 的整体重写；目的是把 P4 GRPO + P5-01/P5-02/P5-03 + 最终全链路审查 round-20 + B-audit 22 轮 已交付状态同步到 roadmap，并制定新的"当前阶段 / 下一阶段：5 个候选"。

不在本 stage review 范围内：
- 任何代码、schema、配置、tests 改动（list item C 任务明确"不修改代码、不重跑训练"）；
- 启动 5 个候选中的任何一个（仅 roadmap 文档化候选，需后续 list item 独立激活）；
- 修改 `docs/plans/open-issues.md`（已由 list item B 闭环）；
- 修改 `docs/reports/final-audit.md`（已由 list item B round-20/round-21/round-22 闭环）。

## 修改 summary（high-level diff）

`docs/plans/roadmap.md` 整体重写（130 行 → 230+ 行），关键变更：

| 段落 | 旧 | 新 |
|---|---|---|
| 已完成阶段 | 列表至 P3 D2 + SFT MVP + P2 evaluator | 列表追加 P4 GRPO + P5-02 + P5-03 + 最终全链路审查 round-20 + B-audit 22 轮 Open-issues 关闭 |
| 当前阶段 | "P5 推理后端接入 + 大模型起点"（与下文"已交付阶段：P4 GRPO + P5 后端"重复） | "路线图刷新（2026-08-29，B 完成之后）" + 累计已交付 table |
| 下一阶段 | "已交付阶段：P4 GRPO + P5 后端（2026-08-29）"（仅列已交付 items） | "下一阶段：5 个候选（按推荐顺序）" + 5 个候选详细 scope / 退出条件 / 风险 + 推荐激活顺序 |
| 暂缓阶段 | 9 项 | 12 项（新增完整 RLHF/DPO、模型量化） |

## 5 个候选详细 scope（roadmap 同步要点）

| # | 候选 | 范围 | 退出条件 | 风险 |
|---|---|---|---|---|
| 1 | GQA 公开模型对比 | 1-2 个 GQA-only 模型 × P5-02 benchmark subset × P1-05 + P2 | eval JSON + reward JSON + GQA vs MHA 4 轴对比表 + protocol doc | 难找同 base GQA/MHA pair；退化为不同 base 对比 |
| 2 | 简化 MLA | `architecture_lab/models/mla_dense.py` + Dense baseline n4 + dropout 0.1 n6 + RoPE 50k n7 四向对比 + 5000 步 | `docs/experiments/mla-simplified/` + protocol + stage review | 2.10M 规模 MLA 收益可能不显著 |
| 3 | P5-04 双后端基准对比 | 同 baseline × 5 公开模型 × 2 后端 × 4 轴对比（latency/throughput/reward_binary/reward_layered） | `scripts/eval_backend_comparison.py` + experiment README + protocol + stage review | vLLM 在小 batch/small model 可能不显著优于 Transformers |
| 4 | 长训练曲线 100k | 100k 步 baseline + medium × 3 seed（log_interval=1000 / validation_interval=5000） | `docs/experiments/n13-dense-100k-curve/` + protocol + stage review | 100k 步 baseline 训练时长显著（参考 N12 100k large ~3-4h）；需评估 GPU 预算 |
| 5 | 真实 OWT 评测 | OWT data 上 5 公开模型 per-token loss 对比（不涉 D2 / tool calling；纯 LM 评估） | OWT 数据源 + 5 模型 per-token loss 对比表 + experiment README + stage review | OWT 数据未在仓库（gitignored）；需保证路径 hash 与 manifest 一致 |

## 推荐激活顺序

按"实现路径连续性 + 已交付基础复用率"：

1. **候选 3 (P5-04 双后端基准对比)** — **最高优先级**；直接复用 P5-02/P5-03 eval scripts，仅补一个对比 CLI。1-2 个 list item 内完成。
2. **候选 1 (GQA 公开模型对比)** — 次高优先级；同样复用 P5-02 eval script + P1-05 + P2。依赖"找得到 GQA vs MHA 同 base 对"先决条件。
3. **候选 5 (真实 OWT 评测)** — 简单独立任务；可作为候选 3 完成后补充。
4. **候选 4 (长训练曲线 100k)** — 中优先级；需要先评估 GPU 资源预算。
5. **候选 2 (简化 MLA)** — 最低优先级；涉及自研模型架构改动，工作量最大；MLA 在 2.10M 收益尚不确定。

如 list item 决策推进，建议从候选 3 启动。

## 验证 contract

本 stage review 不触发新的代码 / schema / 配置 / tests 改动；仅 `docs/plans/roadmap.md` doc-only 修改。

verification（abstract HEAD pointer — auditor runs `git rev-parse HEAD` to verify current value）：

- **live command 1** (working tree clean): `git status --short` → empty。
- **live command 2** (diff is doc-only): `git diff --name-only f1fe61c..HEAD` → 仅含 docs/* + README.md，无 .py / schema / config files。
- **live command 3** (overall tests pass): `python scripts/run_tests.py full` → 386 OK (skipped=3)。
- **live command 4** (audit clean): `python scripts/audit/run_doc_artifact_reconciliation.py` → 0 unresolvable。
- **live command 5** (stage 0): `python scripts/validate_stage0.py --examples` → 9/9 PASS。
- **live command 6** (gitignore): `python scripts/audit/run_gitignore_coverage.py` → 7094/7094 PASS。
- **live command 7** (roadmap 当前 HEAD 状态): `git log -1 --oneline docs/plans/roadmap.md` → 应返回本 stage review 创建的 commit subject。

**expected live verification outputs**: 上述 7 个命令在 round-N 后保持一致；新 commits 不应破坏 doc-only 约束。

## 关联文件

- 修改: `docs/plans/roadmap.md` (整体重写，130 行 → 230+ 行)
- 新增: `docs/plans/reviews/stage-roadmap-update.md` (本文件，stage review record)

## 审查 reviewer 身份

- detached auditor: project-level subagent reviewer, MiniMax-M3 (per AGENTS.md "审查模型固定使用 minimax-cn/MiniMax-M3")
- author self-review: this stage review record
- **detached auditor verdict**: **PENDING independent review** (status: stage review record created, awaiting detached auditor subagent dispatch)
- **detached auditor provider/model**: not yet invoked; per `docs/plans/review-process.md` "提交后触发 stage review" 流程, detached auditor is dispatched after commit. Current commit `b6f8e74`; pending independent review will verify roadmap update is semantically consistent with actual delivered state, with required reviewer verdict (pass / conditional / fail) + provider/model field 填充 in this section once independent review completes.
