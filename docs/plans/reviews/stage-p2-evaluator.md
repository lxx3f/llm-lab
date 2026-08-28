# 阶段 P2 确定性 Evaluator 审查

审查模型：`minimax-cn/MiniMax-M3`（project-level `reviewer` agent，frontmatter 明记）
审查 agent：`reviewer`（subagent dispatch 名称）
实际 provider/model：`PI_PROVIDER=minimax-cn`，`PI_MODEL=MiniMax-M3`
可选运行标识：`PI_AGENT_NAME` 未由 harness 注入
审查文件路径：`docs/plans/reviews/stage-p2-evaluator.md`
commit 候选变更：`schemas/reward_signal.schema.json`、`scripts/reward_offline.py`、`tests/test_reward_offline.py`、`scripts/run_tests.py`、`docs/protocols/p2-evaluator.md`、`docs/experiments/p2-evaluator/README.md`、`docs/plans/roadmap.md`

## 阶段 p2-evaluator 审查

- 完成范围：
  - **reward schema v1.0**：`schemas/reward_signal.schema.json` 定义 reward_signal 结构（reward_binary、reward_layered、layers、first_failure 等）；jsonschema Draft202012 校验通过；
  - **离线 reward 计算器** `scripts/reward_offline.py`：
    - `compute_reward(sample, transcript_row, ...)` 复用 `scripts/classify_tool_failure.py::classify`，按 P1-05 八级（排除 task_success 别名）映射；
    - `aggregate(signals)` 输出 mean ± pop_std（P1-03 协议）+ first_failure 分布 + task_type 分布；
    - CLI 支持 `--samples-dir / --transcripts / --output / --checkpoint / --transcript-kind`；
  - **单测** `tests/test_reward_offline.py`：5 个测试覆盖 full-pass / parse-fail / no_tool / partial-pass / CLI 聚合；注册到 `scripts/run_tests.py` 的 fast + module data + full 三档入口；
  - **协议文档** `docs/protocols/p2-evaluator.md`：reward 定义、语义边界、与 P1-05 关系、与 P4 GRPO 衔接；
  - **跨 checkpoint reward 评测**：`docs/experiments/p2-evaluator/README.md` 记录 8 个 SFT checkpoint × D1 dev 13 样本 = 104 个 reward_signal 全部 reward_binary=0，reward_layered 0.0000~0.0769；
  - **路线图同步**：`docs/plans/roadmap.md` 将 P2 加入已完成阶段；明确 P5-02/04 范围限定为公开模型 + 与 P2 reward offline 对比。

- 未完成范围：
  - P3 数据版本 D2 多轮对话 + IID held-out split
  - P5-01/02/03/04（开源 instruction-tuned 模型 + Transformers/vLLM backend）
  - P4 GRPO（依赖 P3 + P5-02）

- 测试结果：
  - `scripts/run_tests.py full` → Ran 190 tests OK（含 2 skipped）
  - 包含新增 5 个 reward_offline 单测；
  - 104 个 reward signal 全部通过 jsonschema Draft202012 校验（8 checkpoint × 13 样本）。

- 实验结果（`docs/experiments/p2-evaluator/README.md`）：
  - 8 个 checkpoint × 13 样本 = 104 reward_signal，reward_binary 全 0；
  - reward_layered 分布区分 dense large / d256 20k（0.03~0.08）vs d256 5k / MoE（0.00）；
  - first_failure 全部为 parse_success；
  - **D1 dev 13 样本不是独立 held-out split**，不能作为正式 reward 分布结论。

- 新发现问题：
  - **P0：无**。
  - **P2：D1 dev 仅 13 样本，reward 分布没有统计意义**；需 P3 阶段提供独立 IID held-out split（已记录于 `docs/plans/open-issues.md` P2-05）。
  - **P2：reward_offline 当前不实现 trajectory-level reward shaping**；GRPO advantage 计算需在 P4 阶段定义（已在 `docs/protocols/p2-evaluator.md` §4.3 显式声明）。

- 计划调整：
  - P2 已完成；下一阶段候选 P3 / P5-02（公开模型 Transformers backend）；
  - P5-02/04 在路线图已明确为公开模型 backend + 与 P2 reward offline 联动。

- 是否允许进入下一阶段：是
- 下一步：
  1. P3 数据版本 D2 多轮对话（生成器 + MockExecutor 多轮支持 + 独立 held-out split）
  2. P5-01：选定 Qwen2.5-0.5B-Instruct，记录模型卡、LICENSE、运行环境
  3. P5-02：Transformers backend + 同一 reward offline 链路
  4. P4 GRPO：等 P3 + P5-02 就位后实现 advantage 计算 + policy 更新

## 审查结论

- 审查模型：`minimax-cn/MiniMax-M3`
- 审查 agent：`reviewer`
- 结论：通过
- 允许创建阶段 commit：是