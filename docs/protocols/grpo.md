# P4 GRPO MVP — Group-Relative Policy Optimization Protocol

## 1. 目标与范围

本协议定义 P4 阶段的最小 GRPO（Group Relative Policy Optimization，DeepSeek 2024）训练闭环：

- **Policy**：公开 instruction-tuned Transformers 模型（Hugging Face `transformers` API；CPU bf16/fp32 + CUDA bf16/fp16/fp32 三档）。
- **Rollout**：基于 `_apply_chat_template`（复用 P5-02 backend 的 prompt pipeline；terminal assistant content 已被 strip 避免 target-answer 泄漏）。
- **Reward**：直接调用 `scripts.reward_offline.compute_reward`（P2 8 层 reward 信号：`reward_binary` + `reward_layered`）。
- **Advantage**：标准 group-relative：`adv_i = (r_i − mean(r)) / max(std(r), eps)`。
- **Policy update**：REINFORCE 风格 surrogate loss（`-E[adv · logp(continuation)]`），可选 grad-clipping；不引入 reference policy / KL penalty（属 MVP 范围外）。
- **Checkpoint / resume**：每个 step 落盘 `<dir>/step-<step_id>.json` + `<dir>/state.json`；`--resume-from <state.json>` 恢复 global_step 后继续。

范围说明：本 MVP 不实现完整的 PPO clipping / KL penalty / reference policy；目标是端到端可跑 + 单元可测 + 可 resume 的最小闭环。

## 2. CLI 接口

```
.venv/python.exe scripts/grpo_train.py \
    --policy-model Qwen/Qwen2.5-0.5B-Instruct \
    --samples-dir datasets/tool-calling-d2/train \
    --checkpoint-dir artifacts/grpo-qwen05 \
    --max-steps 8 --k-rollouts 4 --limit 8
```

完整参数清单（`--help`）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--policy-model` | 必填 | HF model id 或本地路径 |
| `--samples-dir` | `datasets/tool-calling-d2/train` | D2 多轮样本目录 |
| `--checkpoint-dir` | 必填 | step artifact + state.json 输出目录 |
| `--resume-from` | `None` | 旧 `state.json` 路径；恢复 global_step + 1 |
| `--max-steps` | `8` | policy-gradient 步数 |
| `--k-rollouts` | `4` | 每 prompt 的 rollout 数 K（GRPO group size） |
| `--learning-rate` | `1e-5` | Adam learning rate |
| `--temperature` | `1.0` | rollout 采样温度；`0` = greedy |
| `--max-new-tokens` | `64` | rollout 续写长度 |
| `--limit` | `8` | 加载样本数上限（smoke） |
| `--seed` | `2026` | 样本 shuffle + rollout 采样 seed |
| `--device` | `auto` | `auto` / `cuda` / `cpu` |
| `--max-grad-norm` | `1.0` | grad clip 阈值；`0` = no clip |
| `--smoke-deterministic` | `False` | 强制 greedy + temperature=0 |
| `--reference-model` | `None` | reference policy（MVP no-op） |

## 3. Per-step 流程

每个 `global_step` 包含以下子步骤：

1. **Sample 一个 prompt**：从 `samples-dir` 加载 `min(--limit, len(samples_dir))` 个样本，按 `--seed` shuffle 后顺序消费。
2. **生成 K 个 rollout**：每个 rollout 调用 `_rollout_one()`，输出 `{generated, extracted_calls, rollout_index=k}`。
3. **计算 K 个 reward**：对每个 rollout 构造 transcript row，调用 `scripts.reward_offline.compute_reward`，输出 `{reward_binary, reward_layered, reward_type, first_failure, ...}`。
4. **计算 group-relative advantage**：`_group_relative_advantages(rewards)`。当 group std < eps 时输出全零 advantages（标记 `skipped=True`）。
5. **Policy update**：当至少一个 advantage 绝对值 > eps 时执行；计算 `-adv · logp(continuation tokens)`，反向传播 + Adam + grad clip。
6. **落盘 step artifact**：`<checkpoint-dir>/step-<step_id>.json`，schema 见 `schemas/grpo_step_result.schema.json`。
7. **更新 state**：`<checkpoint-dir>/state.json` 含 `global_step`、`last_step_id`、`completed` 等。

## 4. Determinism

MVP 支持两类 deterministic 模式：

- **默认（`--temperature 1.0`）**：温度采样；rollouts 在同 seed 下字节稳定，但与 floating-point 累加顺序相关。
- **Smoke（`--smoke-deterministic` 或 `--temperature 0`）**：强制 greedy；rollouts 字节级稳定。

Artifact 内置两类 fingerprint：

- `deterministic.rollouts_text_hash`：sorted rollout.generated 的 sha256。
- `deterministic.rewards_text_hash`：sorted reward_layered 字符串表示的 sha256。

Test 用例 `tests/test_grpo_mvp.py::TestHashFunctions` 验证这两类 fingerprint 在 sort 顺序变化时仍稳定，在内容变化时变化。

## 5. Reward 信号与 P2 协议

每个 rollout 的 reward 由 `scripts/reward_offline.py::compute_reward` 产出。该函数：

- 复用 `scripts.classify_tool_failure.classify` 跑 P1-05 8 层分类。
- 输出 schema 见 `schemas/reward_signal.schema.json`：`reward_binary ∈ {0, 1}`、`reward_layered ∈ [0, 1]`、`reward_type ∈ {parse_success, final_answer_correct, argument_correct, execution_correct}`、`first_failure ∈ 8 layer names ∪ {None}`。

GRPO advantage 只使用 `reward_layered`（连续值，标准化后给 policy gradient 提供更细粒度信号）；`reward_binary` 仅作为辅助 metric 落入 step artifact。

## 6. Checkpoint / Resume

- **State 文件**（`state.json`）：`{schema_version, global_step, last_step_id, max_steps, k_rollouts, policy_model, completed}`。
- **Resume**：`--resume-from <state.json>` → `resume_step = global_step + 1` → `_iter_prompts` 从 `resume_step` 开始消费；已保存的 step artifact 不重跑。
- **Verification**：测试 `tests/test_grpo_mvp.py::TestCheckpointIO::test_state_round_trip` + `test_state_completed_flag` 验证 state.json 序列化/反序列化与 `completed` 标志逻辑。

## 7. CPU smoke 与 GPU 完整运行

- **CPU smoke**（MVP 默认）：`--device auto` → fallback CPU；`--limit 8 --max-steps 8 --k-rollouts 4`；总耗时数十分钟（受模型大小影响）。`sshleifer/tiny-gpt2`（~2MB）跑通 1 step + K=2 rollout 仅需数十秒。
- **GPU 完整运行**：`--device cuda --max-steps ≥ 100 --k-rollouts 8` + 0.5B+ 模型；显存要求 ≥ 4 GB（bf16 policy + forward + backward）。

## 8. 与 D2 / P5-02 / P2 的衔接

| 阶段 | 共享组件 | GRPO 调用方式 |
|---|---|---|
| D2 | `datasets/tool-calling-d2/{train,dev,test}/` | `--samples-dir` 直接读 train（3500 样本） |
| P2 reward | `scripts.reward_offline.compute_reward` | 每个 rollout 一次 |
| P2 schema | `schemas/reward_signal.schema.json` | reward 字段在 step artifact 中内嵌 |
| P5-02 chat | `scripts.eval_transformers._apply_chat_template` | rollout prompt 渲染 |
| P5-02 classifier | `scripts.classify_tool_failure.classify` | 通过 `compute_reward` 间接调用 |
| P1-05 layers | `EIGHT_LAYERS` in `reward_offline.py` | reward_layered 分母 |

## 9. 已知边界与不在 MVP 范围内的项

- **No reference policy / KL penalty**：MVP 不会自动加载第二个 `transformers` model 来做 KL 约束；`--reference-model` 参数被接受但无副作用（CLI stability）。
- **No PPO clipping**：REINFORCE 风格 surrogate loss，无 clip。后续 P4 完整实现可在此基础上加入。
- **No mixed precision grad scaling**：CPU fp32 + CUDA bf16/raw；不使用 `torch.cuda.amp.GradScaler`。
- **No LoRA / PEFT**：MVP 直接对全部 `requires_grad=True` 参数做更新；显存占用与模型大小成正比。
- **CPU smoke 推荐配置**：`--max-new-tokens 16 --k-rollouts 2 --limit 1 --smoke-deterministic`；耗时约 30 秒（小模型）。
- **断点续训必须用同一 policy_model**：state.json 含 `policy_model` 字段；resume 时不强制校验（实验允许跨 model resume），但推荐保持一致。

## 10. 测试

| 测试类 | 覆盖 |
|---|---|
| `TestGroupRelativeAdvantages` | advantage 计算：零方差、标准化、空输入、K=1 |
| `TestAdvantageStats` | advantage 分布汇总 |
| `TestHashFunctions` | rollouts/rewards hash 顺序无关性 + 内容敏感 |
| `TestLoadSamples` | 样本加载 determinism |
| `TestIterPrompts` | resume_step 迭代逻辑（含负值、超界） |
| `TestCheckpointIO` | state.json 序列化/反序列化 + `completed` flag |
| `TestAssembleStepArtifact` | artifact 装配 + schema 校验 |
| `TestGrpoSmokeIntegration` | end-to-end smoke（gated by `GRPO_SMOKE=1`） |

`scripts/run_tests.py full` → `tests/test_grpo_mvp.py` 自动运行（fast 模块）；smoke 集成测试需显式 `GRPO_SMOKE=1` 才执行。