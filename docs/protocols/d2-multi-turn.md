# P3 D2 数据版本：多轮对话 + IID held-out split

> 状态：阶段交付（2026-08-28）。

本协议固定 D2 多轮工具调用数据集的结构、生成、split 边界、与 D1 / D1.1 的互斥关系，以及与 P2 离线 reward 校验的衔接。

## 1. 目标

D2 是 P1-04 数据版本谱系中的 **第二个稳定数据版本**（D1 模板版 → D1.1 LLM 生成版 → **D2 多轮版**），它在 D1 之上做两件事：

1. **真正的多轮结构**：messages 数组包含 `assistant` 角色（带 `tool_calls`）+ `tool` 角色（带 `tool_call_id` + `name` + `content`）的完整对话链，而不是 D1 的"只有 system + user + 期望调用"投影。
2. **独立的 IID held-out split**：train / dev / test 三个 split 通过 sample id 命名空间 + 目录隔离双重保证互斥；与 D1 / D1.1 train 在 id 层不重叠；dev / test 足以支撑有意义的 reward 评测。

D2 直接服务于 **P2-05 / P2-06 显式未解决项** —— P2 阶段已交付的 reward offline 当前只在 D1 dev 13 样本上跑评测，不具备统计意义；D2 dev (90 样本) 是 D1 dev (13 样本) 的 ~7×，是首个能在 held-out split 上做 reward 信号校验的数据集。

## 2. 数据 schema 与 task_type

- D2 沿用 `schemas/d2_multi_turn_sample.schema.json`，而不是仅扩展 D1 的通用 schema；该独立 schema 强制 6 个规范 task_type 名称、D2 id/call_id 格式和多轮消息字段。

| D1 task_type | D2 task_type | 含义 |
|---|---|---|
| `single_tool`, `multi_tool`, `no_tool`, `tool_error`, `insufficient_result`, `requirement_change` | （保留）| D1 单轮/投影 |
| — | `multi_tool_sequential` | 链式多工具调用（calc → search → answer），2-3 轮 |
| — | `tool_error_response` | 工具返回错误，模型观察并上报，不传播错误结果 |
| — | `req_change_city` | 用户在第 2 轮改变需求，模型重新执行 |
| — | `insufficient_result_search` | 工具返回有限结果，模型追问用户 |
| — | `tool_not_available` | 用户要求的工具不在列表中，模型不调用 |
| — | `error_recovery` | 工具结果不足/失败，模型调整参数后重试 |

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

`scripts/generate_d2_dataset.py` 在生成时先通过独立 D2 schema 校验，再通过依赖图语义校验，最后为每个样本注册工具并调用真实 `MockExecutor.execute_sequence()`；它检查每个 `expected_tool_calls[i].expected_result` 与 executor 返回值一致。因此每个 D2 样本既是 schema 合法也是执行语义合法。

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
- 与 D1 / D1.1 train id 在 stem 层不重叠（D2 id 是 `d2-*`，D1 是 `d1-*` / `d1llm-*`）；
- 生成器和测试均计算同一个 `canonical_content_signature()`：去除 `id` / `call_id` / `tool_call_id` / `depends_on`、`metadata.created_at` 和 `metadata.split` 后，仍保留 task_type、tools、消息内容、工具参数/结果与 expected_answer。600 个签名全局唯一；每类 100/100 唯一；train/dev/test 三组 canonical signature 交集均为空。

`tests/test_d2_dataset.py::D2SplitDisjointnessTests` 显式校验：

- `train ∩ dev = train ∩ test = dev ∩ test = ∅`；
- canonical semantic content 的三组 split 交集为空，且 600 个样本签名全局唯一；
- 每个 task_type 都有 100 个不同 canonical semantic instances；
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
  "created_at": "2026-07-25T17:53:46Z",
  "count": 420,
  "aggregate_sha256": "<hex64 of sorted ids hash>",
  "samples": [
    {
      "path": "train/d2-train-0001.json",
      "sha256": "<hex64>",
      "task_type": "tool_error_response",
      "split": "train"
    },
    ...
  ]
}
```

`aggregate_sha256` 是该 split 内按 id 排序后 hash — 可在测试中独立重算并验证。

### 5.1 时间戳契约

D2 采用与 D1 相同的确定性时间戳公式（与 `scripts/generate_d1_dataset.py` manifest 字段一致）：

```text
ts   = 1785000000 + seed + index
fmt  = datetime.fromtimestamp(ts, tz=UTC).isoformat().replace("+00:00", "Z")
```

其中 `index` 为样本在其 split 内的 1-based 位置（train 1..420、dev 1..90、test 1..90；每个 split 独立编号，与 ``sample.id`` 后缀严格一致）。MANIFEST 的 `created_at` 使用 `index=0`（即 `seed` 本身的 epoch 秒）。选择 split-local 编号而非生成器全局序列的好处是 ``d2-train-0001`` / ``d2-dev-0001`` / ``d2-test-0001`` 都映射到同一个 epoch 起始点，便于跨 split 对齐调试；该选择与 round 8 时间戳契约的"split-local 1-based index"语义一致。

**示例**：seed=2026、index=1 → `2026-07-25T17:53:47Z`；seed=2026、index=420（d2-train-0420） → `2026-08-02T03:28:47Z`；seed=2026、index=90（d2-dev-0090） → `2026-07-25T19:53:17Z`。

该契约保证 `tests/test_d2_dataset.py::D2TimestampContractTests` 中 6 个反向测试全绿：同 `seed+index` 字节相同；不同 `seed` 或不同 `index` 均产出不同字符串；磁盘样本与公式逐字符相等。

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

确定性保证：同一 `--seed` 产生相同的 600 条 canonical content、文件 sha256 和 MANIFEST aggregate hash；不同 `--count` 通过 round-robin 在 6 类 task_type 间分配。生成器写盘前以 `canonical_content_signature()` 执行全局语义去重；去除 ID、时间戳、split、call_id 和依赖引用后，默认数据仍保持 600/600 唯一、每类 100/100 唯一、三 split 无交集。

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

**P3 阶段已在 D2 dev 上完成 5 ckpt × 90 = 450 个真实推理 reward_signal（round 10 IID 分裂后重跑）**：5 个 checkpoint 均 `reward_binary=0.0`；large-v1 的 `reward_layered=0.0099`（87 个 `parse_success`，3 个 `argument_correct`），其余 4 个 checkpoint 为 0.0000。与 round 9 数字相比 large-v1 从 0.0162 → 0.0099、d256-20k 从 0.0042 → 0.0000，变化源于 round 10 的 IID stratified shuffle 重新分配 600 个变体至三个 split，导致样本内容不同（同一内容语义不变，但 dev/test 现在拿到的不是 train 的 70-99 连续片段）。该诚实负结果证明：(a) D2 dev 评测管线可跑通；(b) 单轮模型基本不具多轮工具调用能力；(c) 需 P5-02 公开 instruction-tuned 模型或自研多轮 SFT 才能获得有意义的 reward 分布。完整明细与 reward_type 分布见 `docs/experiments/p2-evaluator/README.md` §6。

`scripts/eval_sft_tool.py --prompt-mode multi_turn` 负责把 D2 多轮 messages 序列化为 SFT 模板历史并生成续写，是 D2 dev 评测的标准推理入口。

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