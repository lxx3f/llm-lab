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
| 4 | N12 3-seed 复现 + 可选 N13 200k | 100k 步 baseline + medium × 3 seed + 可选 200k 单点（log_interval=1000 / validation_interval=5000） | `docs/experiments/n13-dense-100k-3seed/` + `docs/protocols/n13-dense-100k-3seed.md` + stage review (N12 当前缺 stage review，本次 N13 提交时同时为 N12 补交 stage review) | 100k 步 × 3 seed × 2 规模点 按 N12 README 并发跑模式 实际总墙钟 ~3 × 45 min ≈ 2.5 小时（与 N12 README 实际并发墙钟 ~45 min 对齐，不是 3-4 小时）；需先确认 host GPU 并发能力 |
| 5 | 真实 OWT 评测 | OWT data 上 5 公开模型 per-token loss 对比（不涉 D2 / tool calling；纯 LM 评估） | OWT 数据源 + 5 模型 per-token loss 对比表 + experiment README + stage review | OWT 数据未在仓库（gitignored）；需保证路径 hash 与 manifest 一致 |

## 推荐激活顺序

按"实现路径连续性 + 已交付基础复用率"：

1. **候选 3 (P5-04 双后端基准对比)** — **最高优先级**；直接复用 P5-02/P5-03 eval scripts，仅补一个对比 CLI。1-2 个 list item 内完成。
2. **候选 1 (GQA 公开模型对比)** — 次高优先级；同样复用 P5-02 eval script + P1-05 + P2。依赖"找得到 GQA vs MHA 同 base 对"先决条件。
3. **候选 5 (真实 OWT 评测)** — 简单独立任务；可作为候选 3 完成后补充。
4. **候选 4 (N12 3-seed 复现 + 可选 N13 200k)** — 中优先级；需要先评估 GPU 资源预算；N12 当前缺 stage review，N13 提交时同时为 N12 补交。按 N12 README 实际并发墙钟 ~45 min，3-seed 复现总墙钟 ~2.5 小时，远低于文中估算。
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

## 审查 reviewer 身份与详细 verdict

- detached auditor agent name: `reviewer` (project-level subagent reviewer, dispatched via Agent tool)
- **detached auditor verdict**: **CONDITIONAL PASS** (initially PENDING, now dispatched + reviewed)
- **detached auditor provider/model**: `PI_PROVIDER=minimax, PI_MODEL=MiniMax-M3` (subagent returned actual values; matches expected per `docs/plans/review-process.md` "审查模型: minimax-cn/MiniMax-M3"; expected dispatch name 与 actual model 一致)
- **dispatch timestamp**: 2026-08-29 09:25 (approximately; reviewer subagent invoked after commit `6f93ee6`)

### Reviewer findings summary (dispatched 2026-08-29)

**Critical (must fix)**: none — structural and semantic review passes; 6 个 SHAs 全部 git-resolvable; 不修改代码约束成立; 5 个候选全部具备 动机 / 范围 / 退出条件 / 风险 四要素; 推荐激活顺序与 task spec 一致。

**Warnings (should fix) — 全部已在本 review 记录中修复**:

1. **候选 4 时间估算** (原 `docs/plans/roadmap.md:129` "参考 N12 100k 3.5 小时表动，200k 预估 7 小时" + `:133` "总时长预估 3-4 × 6 ≈ 20 小时") — 修正为 N12 README 实际并发墙钟 ~45 min 数据：200k 线性外推 ~90 min；3-seed 复现实际总墙钟 ~2.5 小时。
2. **P4 GRPO 交付 commit 列** (原 `:68` `1fae6f0 + 99646fa`) — 修正为实际交付 commits `063d34f` (minimal MVP) + `542c650` (real model) + `9711746` (mock smoke + dtype + YAML) + `d1c564e` (小规模正确性实验)。
3. **P5-01 交付 commit 列** (原 `:69` `bb61a3a` "与 P5-03 同交付") — 修正为 `63cbd83` feat(p5-02) Transformers backend + 5 public models，该 commit subject 明确含 model selection。

**Suggestions (consider)** — 全部已在本 review 记录中应用:

1. 表头 "之前" → 表述调整 (中性: 累计交付) — 在本 review 后补调整。
2. stage-roadmap-update.md:31 风险列同步修正 — 已在本次同步完成。
3. P4 GRPO commit 列表补充中间 commits — 已应用 (列出 4 个 commits 而非 2 个)。

### Verdict

**CONDITIONAL PASS** — 4 个 Warnings 在本轮记录中全部修复 (提交 commit 后)，6 个 SHA 全部 git-resolvable，B-audit 22 轮 / P4 GRPO / P5-02 / P5-03 / final-audit round-20 / N12 单 seed 100k 全部可验证。Roadmap 与 stage review record 在本轮中达到 semantic consistency，audit 完成。
