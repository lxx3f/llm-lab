# P3 D2 数据版本：多轮对话 + IID held-out split

> 状态：阶段交付（2026-08-28）。

本协议固定 D2 多轮工具调用数据集的结构、生成、split 边界、与 D1 / D1.1 的互斥关系，以及与 P2 离线 reward 校验的衔接。

## 1. 目标

D2 是 P1-04 数据版本谱系中的 **第二个稳定数据版本**（D1 模板版 → D1.1 LLM 生成版 → **D2 多轮版**），它在 D1 之上做两件事：

1. **真正的多轮结构**：messages 数组包含 `assistant` 角色（带 `tool_calls`）+ `tool` 角色（带 `tool_call_id` + `name` + `content`）的完整对话链，而不是 D1 的"只有 system + user + 期望调用"投影。
2. **独立的 IID held-out split**：train / dev / test 三个 split 通过 sample id 命名空间 + 目录隔离双重保证互斥；与 D1 / D1.1 train 在 id 层不重叠；dev / test 足以支撑有意义的 reward 评测。

D2 直接服务于 **P2-05 / P2-06 显式未解决项** —— P2 阶段已交付的 reward offline 当前只在 D1 dev 13 样本上跑评测，不具备统计意义；D2 dev (90 样本) 是 D1 dev (13 样本) 的 ~7×，是首个能在 held-out split 上做 reward 信号校验的数据集。

## 2. 数据 schema 与 task_type

D2 沿用 `schemas/tool_calling_sample.schema.json`，仅扩展 `metadata.task_type` enum：

| D1 task_type | D2 task_type | 含义 |
|---|---|---|
| `single_tool`, `multi_tool`, `no_tool`, `tool_error`, `insufficient_result`, `requirement_change` | （保留）| D1 单轮/投影 |
| — | `multi_turn_tool_chain` | 链式多工具调用（calc → search → answer），2-3 轮 |
| — | `multi_turn_error_recovery` | 工具返回错误，模型识别后调整或上报 |
| — | `multi_turn_req_change` | 用户在第 2 轮改变需求，模型重新执行 |
| — | `multi_turn_insufficient_result` | 工具返回有限结果，模型追问用户 |
| — | `multi_turn_tool_not_available` | 用户要求的工具不在列表中，模型不调用 |
| — | `multi_turn_clarification` | 用户需求模糊，模型先反问澄清 |

每类 ≥ 100 样本；总规模 **≥ 600 样本**，默认 `--count 600`。

## 3. 多轮 transcript 结构

每个 D2 样本的 `messages` 数组按轮次构造：

```text
[system, user, assistant(tool_calls), tool(result), assistant(tool_calls), tool(result), ..., assistant(content)]
```

- `assistant` 消息携带 `tool_calls[]`，每个 `tool_call.id` 与 `expected_tool_calls[i].call_id` 一一对应；
- `tool` 消息携带 `tool_call_id` + `name` + `content`，其中 `tool_call_id` 引用前面某条 `assistant.tool_calls[].id`；
- `expected_tool_calls` 用 `call_id` + `depends_on` 表达链式依赖，`depends_on[]` 列出所有前置 `call_id`；
- `expected_answer` 关闭整个 transcript。

`scripts/generate_d2_dataset.py` 在生成时端到端通过 `MockExecutor` 验证每个 `expected_tool_calls[i].expected_result == mock(**arguments)`，因此每个 D2 样本既是 schema 合法也是语义合法。

## 4. Held-out split

按 `--count N` 切分（默认 70:15:15）：

| Split | 样本数（默认 600） | 命名空间 | 用途 |
|---|---|---|---|
| train | 420 | `d2-train-NNNN` | SFT 训练 |
| dev | 90 | `d2-dev-NNN` | held-out reward 评测、调参 |
| test | 90 | `d2-test-NNN` | held-out reward 评测、最终验证 |

**互斥保证**：

- id 命名空间前缀不同（`d2-train-` / `d2-dev-` / `d2-test-`）；
- 目录物理隔离（`datasets/tool-calling-d2/{train,dev,test}/`）；
- 与 D1 / D1.1 train id 在 stem 层不重叠（D2 id 是 `d2-*`，D1 是 `d1-*` / `d1llm-*`）。

`tests/test_d2_dataset.py::D2SplitDisjointnessTests` 显式校验：

- `train ∩ dev = train ∩ test = dev ∩ test = ∅`；
- train 与 D1/D1.1 train id 集合无交集；
- 每个 `MANIFEST-{split}.json` 的 `count` / `aggregate_sha256` / 每文件 sha256 与磁盘一致。

## 5. MANIFEST 结构

每个 split 单独一个 MANIFEST 文件：

```json
{
  "schema_version": "1.0",
  "data_version": "D2",
  "split": "train",
  "generator": "scripts/generate_d2_dataset.py",
  "seed": 2026,
  "count": 420,
  "aggregate_sha256": "<hex64 of sorted ids hash>",
  "samples": [
    {
      "path": "train/d2-train-0001.json",
      "sha256": "<hex64>",
      "task_type": "multi_turn_error_recovery",
      "split": "train"
    },
    ...
  ]
}
```

`aggregate_sha256` 是该 split 内按 id 排序后 hash — 可在测试中独立重算并验证。

## 6. 生成与复现

```bash
# 默认生成 600 样本
.venv/python.exe scripts/generate_d2_dataset.py

# 自定义规模（每类至少 1 个，≥ 6）
.venv/python.exe scripts/generate_d2_dataset.py --count 300

# 自定义 seed（默认 2026）
.venv/python.exe scripts/generate_d2_dataset.py --count 600 --seed 2027

# 自定义输出目录
.venv/python.exe scripts/generate_d2_dataset.py --out datasets/tool-calling-d2-test
```

确定性保证：同一 `--seed` 产生相同的 sample 内容 / sha256 / MANIFEST aggregate hash；不同 `--count` 通过 round-robin 在 6 类 task_type 间分配。

## 7. 与 P2 离线 reward 校验的衔接

P2 阶段交付的 `scripts/reward_offline.py` 直接消费 D2 dev 90 样本作为正式 reward 评测输入，命令：

```bash
.venv/python.exe scripts/reward_offline.py \
    --samples-dir datasets/tool-calling-d2/dev \
    --transcripts artifacts/<ckpt>-eval-d2dev.json \
    --output artifacts/<ckpt>-eval-d2dev-reward.json \
    --checkpoint <ckpt-name>
```

D1 dev 13 样本保留作为早期 dev 探针；D2 dev 是首个有统计意义的 reward 评测 split。

## 8. 与 P4 GRPO / P5-02 的衔接

- P4 GRPO：以 D2 train (420 样本) 作为 GRPO rollouts 的 prompt 池；D2 dev / test 作为 advantage 估计的对照基线；
- P5-02 Transformers backend：同一 `--samples-dir` 路径可在公开 instruction-tuned 模型上跑同一 reward offline 链路，与自研模型做公平对比。

## 9. 与 D1 / D1.1 的差异

| 维度 | D1 模板版 | D1.1 LLM 生成版 | **D2 多轮版** |
|---|---|---|---|
| messages 长度 | 2（system + user）| 2（system + user）| 4-8（含 assistant + tool 角色）|
| tool_calls 出现位置 | 仅 `expected_tool_calls` | 同 D1 | 同 D1 + assistant messages 中 |
| tool_call_id 显式存在 | 否 | 否 | 是 |
| depends_on 链 | 是 | 是 | 是（多轮中显式串联）|
| Split | train/dev/test 100/13/13 | train-only 1500 | train/dev/test 420/90/90（IID 独立）|
| Held-out 评测意义 | dev 13 样本（不足）| train-only（不算 held-out）| **dev 90 样本（首个有统计意义）** |
| 与自研模型的 reward 信号差异 | 可跑但样本过少 | 仅 train（不算泛化）| **正式 reward benchmark** |