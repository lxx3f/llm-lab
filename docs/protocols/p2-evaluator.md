# P2 确定性 Evaluator：reward schema + 离线校验

> 状态：阶段交付（2026-08-28）。本协议固定“离线 reward signal”的定义、计算方式与离线校验方法；它是 P4 GRPO 的前置；它把 P1-05 八级分类器转换为 GRPO 可训练的目标值。

## 1. 目标

给定 `(sample, transcript)` 对，**离线地**输出一个确定性 reward signal：

- 不调用任何模型
- 不调用真实工具执行器（MockExecutor 已在 P1-01 / P2-02 transcript 阶段完成）
- 不引入随机性
- 输入相同则输出相同

这套 reward signal 供：

1. **P4 GRPO** 使用，作为组内相对优势的标量目标；
2. **离线 reward 校验** 使用，比较不同 checkpoint 在同一 held-out split 上的 reward 分布；
3. **失败分析** 使用，按 first_failure 分桶统计。

## 2. Reward 定义

reward signal 输出 `schemas/reward_signal.schema.json` 规定的对象，必填字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `reward_binary` | float ∈ [0, 1] | `1.0` 当 `first_failure is None`；否则 `0.0` |
| `reward_layered` | float ∈ [0, 1] | `layer_pass_count / layer_pass_total`（详见 §3）|
| `first_failure` | string \| null | P1-05 八级分类器返回的首个失败层名；`None` 表示全部通过 |
| `layers` | object | 八个 P1-05 层的逐层结果（True / False / None）|
| `layer_pass_count` / `layer_pass_total` | int | 通过层数 / 适用层数（None 不计入）|
| `expected_calls_count` / `predicted_calls_count` | int | 期望 vs 预测的工具调用数 |
| `expected_answer` / `predicted_answer` | string \| null | 期望与预测的最终回答 |

协议版本：

- `schema_version = "1.0"`：信号结构版本；
- `reward_offline_version = "1.0"`：reward 映射版本；改写映射语义时必须 bump。

## 3. 分层 reward 的语义

`reward_layered` 不是简单的 0.125 倍数，而是一个连续指标：

- 分母 `layer_pass_total` 仅计入**适用**的层（None 表示 N/A，跳过）；
- 分子 `layer_pass_count` 计入实际通过的层（True）；
- 当 `expected_calls_count == 0 && predicted_calls_count == 0`（no_tool）且 `expected_answer is None` 时，分母为 0，reward_layered 视为 0；
- `parse_success` 失败的样本后续层全部 N/A，因此 reward_layered 必然 < 1。

> 这与 P1-05 八级分类器保持 1:1 对应，但排除 `task_success`（它仅是 `final_answer_correct` 的 backward-compat 别名，不计入协议定义）。

## 4. 离线校验

### 4.1 CLI

```bash
.venv/python.exe scripts/reward_offline.py \
    --samples-dir datasets/tool-calling-d1/dev \
    --transcripts artifacts/<checkpoint>-eval-d1dev.json \
    --output artifacts/<checkpoint>-eval-d1dev-reward.json \
    --checkpoint <checkpoint-name>
```

- `--transcripts` 期望 `scripts/eval_sft_tool.py` 的输出（含 `rows[].extracted_calls`、`rows[].first_failure` 等）；
- `--samples-dir` 指向样本根目录（每个文件含 `id`、`expected_tool_calls` 等）；
- 输出 JSON 含 `signals[]`（每样本）、`aggregate`（mean ± pop std + first_failure 分布 + 任务类型分布）、`missing_sample_ids`。

### 4.2 校验方法

| 检查 | 命令 / 检查项 |
|---|---|
| **Schema 校验** | `jsonschema` Draft202012 校验 `schemas/reward_signal.schema.json`；CLI 集成测试 `tests/test_reward_offline.py::CLIIntegrationTests::test_cli_aggregate_writes_json` 显式调用 `Draft202012Validator` 校验每个生成的 reward_signal |
| **单元测试** | 37 个测试覆盖：8 层各 ≥ 3 例（共 25；parse_success 4 例、其余 7 层各 3 例） + 与 P1-05 classify 一致性（4 例） + reward_type 映射（7 例：5 主路径 + 2 鲁棒性 / 矛盾输入） + CLI 聚合（1 例）；详见 `tests/test_reward_offline.py` |
| **确定性** | 同 `(sample, transcript)` 重复运行 `compute_reward` 应输出完全一致（`compute_reward` 不调用随机源或全局状态）|
| **与 P1-05 一致性** | `compute_reward` 内置 `classify()` 调用，输出 `first_failure` 与 `layers` 必须与直接调用 `scripts/classify_tool_failure.py::classify` 一致；测试覆盖该路径 |

### 4.3 已知边界

- **D1.1 train 50-sample 抽样不是独立 held-out split**：用 D1.1 train 做 reward 校验仅用于探索性诊断，不能作为模型泛化结论（见 `docs/experiments/sft-tool-mvp/README.md` 与 `docs/plans/open-issues.md` 的 P2-05）；
- **本协议不定义 trajectory-level reward shaping**：单样本 reward 是 0/1 或连续层分数；GRPO 中的 advantage 计算需在 P4 阶段另行定义。

## 5. 与 P1-05 八级分类器的关系

| 项 | P1-05 classify | P2 reward_offline |
|---|---|---|
| 输入 | `(sample, transcript)` | `(sample, transcript_row from eval_sft_tool)` |
| 输出 | `layers, first_failure` | `layers, first_failure, reward_binary, reward_layered, …` |
| 作用 | 失败诊断 | GRPO reward + 离线校验 |
| 计算 | 同 | 同（re-import classify）+ reward 映射 |
| 边界 | 含 `task_success` 别名 | 只输出 8 层（排除 task_success 别名）|

reward_offline 不重写分类逻辑；它只是把分类结果映射成可训练 reward。

## 6. 交付物

- `schemas/reward_signal.schema.json`：信号结构定义（含 reward_type 4 值 enum）
- `scripts/reward_offline.py`：CLI + 离线 reward 计算（reward_binary + reward_layered + reward_type）
- `tests/test_reward_offline.py`：**37** 单测（8 层各 ≥ 3 例 = 25 + classifier 一致性 4 + reward_type 映射 7 + CLI 聚合 1）
- `tests/test_stage0_schemas.py`：新增 5 个 reward schema 测试（full pass / parse fail / 缺 reward_type / 未知 reward_type / reward_binary 越界）
- `examples/reward_signals/*.json`：2 个 reward_signal example（execution_correct + parse_success）
- `scripts/validate_stage0.py`：注册 reward_signal schema 到 stage0 校验
- `docs/experiments/p2-evaluator/README.md`：在 D1 dev 上跨 5 个 SFT MVP checkpoint 的 reward 分布
- `docs/protocols/p2-evaluator.md`：本协议文档

## 7. 与 P4 GRPO 的衔接

P4 必须满足以下前置才能进入：

1. ✅ reward_offline 在 D1 dev 上对所有 SFT checkpoint 输出有效 JSON；
2. ✅ reward 信号满足 schema 校验；
3. ✅ P3 数据版本 D2 多轮对话就绪（2026-08-28：**当前扩样版 D2 dev 750 样本** + IID split + 独立 schema + 5000/3500/750/750 严格命中；当前契约见 `docs/protocols/d2-multi-turn.md` §4）；P2 阶段自研 5 ckpt 历史评测归档于 `docs/plans/open-issues.md` P3-01 段 line 769-841 + `docs/experiments/p2-evaluator/README.md` §6。
4. ✅ P5-02 Transformers backend 让 policy 模型可调用 GRPO rollout（**已交付** HEAD `63cbd83` + round 2 修复 HEAD `b4fd879` + audit round 14 HEAD `66ff9eb`; 详见 `docs/plans/reviews/stage-p5-02-transformers-backend.md`）;P5-03 vLLM backend 亦已交付 (HEAD `bb61a3a` + `docs/plans/reviews/stage-p5-03-vllm-feasibility.md`)。本段原“仍待办”为 P2 阶段交付时的状态描述, 现在已多轮 commits 完成。
5. ✅ D1.1 train 之外已新增独立 held-out split（当前扩样版 D2 dev 750 样本；P2-05 已解决）。