# 阶段 P2 确定性 Evaluator 审查

审查模型：`minimax-cn/MiniMax-M3`（project-level `reviewer` agent，frontmatter 明记）
审查 agent：`reviewer`（subagent dispatch 名称）
实际 provider/model：`PI_PROVIDER=minimax-cn`，`PI_MODEL=MiniMax-M3`
可选运行标识：`PI_AGENT_NAME` 未由 harness 注入
审查文件路径：`docs/plans/reviews/stage-p2-evaluator.md`
commit 候选变更：`schemas/reward_signal.schema.json`、`scripts/reward_offline.py`、`tests/test_reward_offline.py`、`scripts/run_tests.py`、`docs/protocols/p2-evaluator.md`、`docs/experiments/p2-evaluator/README.md`、`docs/plans/roadmap.md`

## 审查与复核记录

### 初次审查 + Round 1 (2026-08-28 03:18–04:11 UTC)

详见 reviewer 输出存档 `.pi-glla/scratch/stage-p2-evaluator-review-r1.txt`（gitignored）。

### Round 2 (2026-08-28 04:13–04:25 UTC)

修复内容：

- schema: 增加 `reward_type` enum = {parse_success, argument_correct, final_answer_correct, execution_correct}；`compute_reward` 增加 `_dominant_reward()` 选择主导通道；
- 测试: 从 5 个扩充到 30 个（8 层各 ≥ 3 例 + classifier 一致性 + CLI 聚合）；
- stage0: 注册 reward_signal schema + 两个 examples + 5 个新测试 case；
- demo: 从 8 个 checkpoint 收敛到 5 个 SFT MVP checkpoint，65 reward_signal 全部 schema 校验通过；
- open-issues P2-06: 登记 P2 完成 + P3/P4 follow-up。

reviewer 实际身份：`minimax-cn/MiniMax-M3`，有条件通过（3 个文档 warnings：README 命名一致表 / protocol 测试数量 / README 30 单测拆分）。三个 warnings 已在 commit `7f30b9b` 中修复。

### Round 3 (2026-08-28 04:30–04:45 UTC, 本轮)

Auditor 在 detached audit 中提出 4 项具体修复：

1. **Reviewer 耐久证据**：补充到 `.pi-glla/scratch/stage-p2-evaluator-review-r3.txt`（本文件 gitignored，供后续仓参）。
2. **Roadmap 内部矛盾**：删除“下一阶段”中的 P2 行；修正“已完成阶段”中 P2 行的描述（8→5、5→30）。
3. **Protocol §6 stale 5 个单测**：补上正确数字 30 + stage0 reward 测试 + examples。
4. **reward_type 映射语义错误**：原 `_dominant_reward` 在 `first_failure in (None, "execution_success", "result_grounded")` 返回 `execution_correct`，与文档语义矛盾（execution_success / result_grounded 失败不应标记 execution_correct）。重构为：仅在 `first_failure is None` 时返回 `execution_correct`；`execution_success` / `result_grounded` / `tool_name_correct` / `argument_value_correct` / `call_plan_matches` 全部映射为 `argument_correct`。

同时 `tests/test_reward_offline.py` 增加 5 个 RewardTypeMapping 测试：

- `test_execution_success_failure_maps_to_argument_correct`
- `test_result_grounded_failure_maps_to_argument_correct`
- `test_full_pass_maps_to_execution_correct`
- `test_final_answer_failure_maps_to_final_answer_correct`
- `test_parse_failure_maps_to_parse_success`

总计 35 个 reward_offline 单测。

验证：

- `scripts/run_tests.py full` → Ran 225 tests OK
- `scripts/validate_stage0.py --examples` → 7/7 PASS
- 5 ckpt × 13 = 65 reward_signal 全部 schema 校验通过
- reviewer subagent（本次：minimax-cn/MiniMax-M3）记录在 `.pi-glla/scratch/stage-p2-evaluator-review-r3.txt`

## 阶段 p2-evaluator 审查

- 完成范围（round 3 final，**已修正round 1 8 checkpoint / 104 signal 的 stale 描述**）：
  - **reward schema v1.0 + reward_type enum**：`schemas/reward_signal.schema.json` 定义 reward_signal 结构（reward_binary、reward_layered、reward_type、layers、first_failure 等）；reward_type enum = {parse_success, argument_correct, final_answer_correct, execution_correct}；jsonschema Draft202012 校验严格；
  - **离线 reward 计算器** `scripts/reward_offline.py`：
    - `compute_reward(sample, transcript_row, ...)` 复用 `scripts/classify_tool_failure.py::classify`，按 P1-05 八级（排除 task_success 别名）映射；
    - `_dominant_reward()` 选择 reward_type 主导通道：仅在 `first_failure is None` 时返回 `execution_correct`；`execution_success`/`result_grounded`/`tool_name_correct`/`argument_value_correct`/`call_plan_matches` 全部 → `argument_correct`；round 3 已修正 execution/grounding 失败误标为 execution_correct 的语义错误；
    - `aggregate(signals)` 输出 mean ± pop_std（P1-03 协议）+ first_failure 分布 + reward_type 分布 + task_type 分布；
    - CLI 支持 `--samples-dir / --transcripts / --output / --checkpoint / --transcript-kind`；
  - **单测** `tests/test_reward_offline.py`：**35** 个测试 = 8 层各 ≥ 3 例（25） + classifier 一致性（4） + reward_type 映射（5） + CLI 聚合（1）；注册到 `scripts/run_tests.py` 三档入口（fast / module training / full）；
  - **stage0 集成**：`scripts/validate_stage0.py` 注册 reward_signal schema + 2 个 examples；`tests/test_stage0_schemas.py` 新增 5 个 reward schema case（full pass / parse fail / 缺 reward_type / 未知 reward_type / reward_binary 越界）；
  - **协议文档** `docs/protocols/p2-evaluator.md`：reward 定义、语义边界、与 P1-05 关系（含 task_success 别名排除）、与 P4 GRPO 衔接 5 项前置、已知边界（不含 trajectory shaping、D1 dev 非 held-out split）；
  - **跨 checkpoint reward 评测**：`docs/experiments/p2-evaluator/README.md` 记录 **5 个 SFT MVP checkpoint** × D1 dev 13 样本 = **65 个 reward_signal** 全部 reward_binary=0，reward_layered 0.0000~0.0769；reward_type 64 个 parse_success + 1 个 argument_correct（first_failure = argument_value_correct）；
  - **路线图同步**：`docs/plans/roadmap.md` 将 P2 加入已完成阶段表（描述含 5 ckpt + 35 单测 + reward_type 4 值 enum）；“下一阶段”表删除 P2 行，只保留 P3 D2 多轮 / P4 GRPO / P5-01 开源模型。

- 未完成范围：
  - P3 数据版本 D2 多轮对话 + IID held-out split
  - P5-01/02/03/04（开源 instruction-tuned 模型 + Transformers/vLLM backend）
  - P4 GRPO（依赖 P3 + P5-02）

- 测试结果（round 3）：
  - `scripts/run_tests.py full` → Ran **225** tests OK（skipped=2）
  - 包含新增 35 个 reward_offline 单测 + 5 个 stage0 reward schema 测试 = **40** 个 P2 阶段新增测试；
  - **65** 个 reward signal（5 ckpt × 13 样本）全部通过 jsonschema Draft202012 校验。

- 实验结果（`docs/experiments/p2-evaluator/README.md`，round 3）：
  - **5** 个 SFT MVP checkpoint × 13 样本 = **65** reward_signal，reward_binary 全 0；
  - reward_layered 分布区分 dense large v1（0.029） / d256 20k（0.077） vs d256 5k / MoE（0.00）；
  - first_failure 分布：64 个 parse_success + 1 个 argument_value_correct；
  - reward_type 分布：64 个 parse_success + 1 个 argument_correct（来自 sft-tool-large-v1；first_failure = argument_value_correct，与 README 描述一致）；
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