# 阶段 P3 D2 多轮对话数据集审查

审查模型：`minimax-cn/MiniMax-M3`（project-level `reviewer` agent）
审查 agent：`reviewer`（subagent dispatch 名称）
实际 provider/model：`PI_PROVIDER=minimax-cn`，`PI_MODEL=MiniMax-M3`
`PI_AGENT_NAME`：未注入
审查文件路径：`docs/plans/reviews/stage-p3-d2-multi-turn.md`
commit 候选变更：见下方“完成范围”节。

## 阶段 p3-d2-multi-turn 审查

### 完成范围

- **schema 扩展**：`schemas/tool_calling_sample.schema.json` 的 `metadata.task_type` enum 新增 6 类多轮：`multi_turn_tool_chain` / `multi_turn_error_recovery` / `multi_turn_req_change` / `multi_turn_insufficient_result` / `multi_turn_tool_not_available` / `multi_turn_clarification`；向后兼容 D1 / D1.1 样本。
- **D2 生成器** `scripts/generate_d2_dataset.py`：
  - 默认生成 600 样本（6 类各 100），train/dev/test = 420/90/90 IID split；
  - 确定性 `--seed`（默认 2026）、确定性时间戳、id 命名空间前缀（`d2-train-NNNN` / `d2-dev-NNN` / `d2-test-NNN`）；
  - MockExecutor 多轮端到端验证（position-paired expected_result == mock(**arguments)）；
  - per-split `aggregate_sha256` + per-file sha256；
  - MANIFEST-{train,dev,test}.json：count + sha256 + task_type 分布 + aggregate_sha256。
- **单测** `tests/test_d2_dataset.py`：**17** 单测分 5 组：
  - `D2DatasetSchemaTests`：schema 合法 / 6 类 task_type 覆盖 / 每 split 包含 6 类；
  - `D2MultiTurnStructureTests`：assistant + tool 角色结构 / tool_call_id 显式存在 / expected_tool_calls call_id 与 transcript 一致；
  - `D2MockExecutorReplayTests`：mock(**args) == expected_result；
  - `D2SplitDisjointnessTests`：train ∩ dev ∩ test = ∅ / 与 D1/D1.1 train id 互斥 / MANIFEST count + aggregate_sha + per-file sha 一致；
  - `D2IdFormatTests`：train/dev/test id 前缀约束。
- **协议** `docs/protocols/d2-multi-turn.md`：D2 与 D1/D1.1 差异 / 6 类任务定义 / call_id 依赖链 / held-out split 互斥 / 与 P2 / P4 / P5-02 衔接。
- **P2 README** `docs/experiments/p2-evaluator/README.md`：引用 D2 dev 90 样本作为首个有统计意义的 reward benchmark；mock pipeline 验证 reward_offline 能消费 D2 dev 多轮 transcript。
- **`docs/plans/roadmap.md`**：P3 加入已完成阶段表；下一阶段候选保持 P5 + P4 + D2 扩样。
- **`docs/plans/open-issues.md`**：P2-05 状态升级 + 新增 P3-01 已解决项。

### reward offline mock pipeline 验证

- `artifacts/d2-mock-reward-d2dev.json`：在 D2 dev 90 样本上以 mock transcript（手工重建 expected_tool_calls 的成功执行轨迹）运行 reward_offline；输出 90/90 reward_binary=1.0、reward_layered=1.0、reward_signal 全 schema 校验通过。
- 该 mock pipeline **验证 reward_offline 能消费 D2 dev 多轮 transcript**，不构成"D2 dev 上真实推理 reward 评测"——后者需 P5-02 公开模型 + Transformers backend 后才能做。

### 测试统计

- `scripts/run_tests.py full` → Ran **244** tests OK（227 → 244 = +17 P3 单测）。
- `scripts/validate_stage0.py --examples` → 7/7 PASS（无新增 schema examples；D2 多轮 task_type enum 扩展已通过现有 schema 校验逻辑）。
- 600 D2 样本 × 3 splits × per-file sha256 + aggregate_sha256 校验全部通过。

### 已知边界

- 自研 5 ckpt 均为单轮训练，当前**未在 D2 dev 上跑实际推理 reward 评测**（仅 mock pipeline 验证）；
- D2 数据规模 600（420 train / 90 dev / 90 test）属 MVP；正式 GRPO rollout 池需 D2 扩样到 5000+；
- D2 与 D1 / D1.1 train id 互斥保证靠 id 命名空间 + 目录物理隔离 + `D2SplitDisjointnessTests` 三重保证。

### 下一步

1. P5-02 Transformers backend + 公开 instruction-tuned 模型 → 在 D2 dev 上跑真实推理 reward 评测；
2. P4 GRPO MVP：基于 D2 train (420) + reward_signal；
3. D2 数据集扩样到 5000+；
4. 自研模型加 multi-turn training（messages 含 assistant + tool）后，回填 5 ckpt × D2 dev 真实 reward 评测。

### durable 证据

- 本次 reviewer subagent 输出保存到 `.pi-glla/scratch/stage-p3-d2-multi-turn-review.txt`（gitignored）。
- `scripts/run_tests.py full` 输出：`Ran 244 tests in 28.xxxs — OK`。
- `artifacts/d2-mock-reward-d2dev.json` aggregate：`{n=90, reward_binary_mean=1.0, reward_layered_mean=1.0}`。

## 审查结论

- 审查模型：`minimax-cn/MiniMax-M3`
- 审查 agent：`reviewer`
- 结论：**通过**
- 允许创建阶段 commit：是
- reviewer r1 输出（PI_PROVIDER=minimax-cn / PI_MODEL=MiniMax-M3 / BLOCKERS: none）保存到 `.pi-glla/scratch/stage-p3-d2-multi-turn-review.txt`。