# P3 D2 数据版本：多轮对话 + IID held-out split

> 状态：阶段交付（2026-08-28；round 14 扩样至 5000 样本后修订）。
> 当前契约：`--count 5000`、train 3500 / dev 750 / test 750、6 类各 ≥833 unique canonical variants。
> 历史 MVP（600 样本）阶段的初始交付文档在 `docs/plans/open-issues.md` P3-01 段（line 769-841）；5000 样本扩样阶段的实施记录与设计决策在 `docs/data/d2-expansion.md`；扩样 stage review 在 `docs/plans/reviews/stage-p3-d2-expansion.md`。

本协议固定 D2 多轮工具调用数据集的结构、生成、split 边界、与 D1 / D1.1 的互斥关系，以及与 P2 离线 reward 校验的衔接。

## 1. 目标

D2 是 P1-04 数据版本谱系中的 **第二个稳定数据版本**（D1 模板版 → D1.1 LLM 生成版 → **D2 多轮版**），它在 D1 之上做两件事：

1. **真正的多轮结构**：messages 数组包含 `assistant` 角色（带 `tool_calls`）+ `tool` 角色（带 `tool_call_id` + `name` + `content`）的完整对话链，而不是 D1 的"只有 system + user + 期望调用"投影。
2. **独立的 IID held-out split**：train / dev / test 三个 split 通过 sample id 命名空间 + 目录隔离双重保证互斥；与 D1 / D1.1 train 在 id 层不重叠；dev / test 足以支撑有意义的 reward 评测。

D2 直接服务于 **P2-05 / P2-06 显式未解决项** —— P2 阶段已交付的 reward offline 当前只在 D1 dev 13 样本上跑评测，不具备统计意义；D2 dev 是 D1 dev (13 样本) 的 **~57×**（750 vs 13），是首个能在 held-out split 上做 reward 信号校验的、有统计意义的数据集。

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

**当前契约（round 14，HEAD `fdfc519`）**：每类 ≥833 unique canonical variants；总规模 **5000 样本**（4 类 833 + 2 类 834），默认 `--count 5000`。Variant pool 扩展与严格嵌套维度公式见 `docs/data/d2-expansion.md`。

## 3. 多轮 transcript 结构

每个 D2 样本的 `messages` 数组按轮次构造：

```text
[system, user, assistant(tool_calls), tool(result), assistant(tool_calls), tool(result), ..., assistant(content)]
```

- `assistant` 消息携带 `tool_calls[]`，每个 `tool_call.id` 与 `expected_tool_calls[i].call_id` 一一对应；
- `tool` 消息携带 `tool_call_id` + `name` + `content`，其中 `tool_call_id` 引用前面某条 `assistant.tool_calls[].id`；
- `expected_tool_calls` 用 `call_id` + `depends_on` 表达链式依赖，`depends_on[]` 列出所有前置 `call_id`；
- `expected_answer` 关闭整个 transcript；按约定它必须与 transcript 中最后一条非工具调用的 `assistant` 消息的 `content` **逐字符相同**。该约定由 `transcript_well_formedness_errors()` 与 `tests/test_d2_dataset.py::D2ExpectedAnswerContractTests` 双重校验：generator 写盘前 `_validate_samples()` 拒绝任何偏离样本；测试覆盖 5000/5000 样本并含反向断言。

`scripts/generate_d2_dataset.py` 在生成时先通过独立 D2 schema 校验，再通过依赖图语义校验，最后为每个样本注册工具并调用真实 `MockExecutor.execute_sequence()`；它检查每个 `expected_tool_calls[i].expected_result` 与 executor 返回值一致。因此每个 D2 样本既是 schema 合法也是执行语义合法。

## 4. Held-out split

按 `--count N` 切分（默认 70:15:15 per-task，round 14 用 `round()` 化保证 dev=test=125）：

| Split | 样本数（5000） | 命名空间 | 用途 |
|---|---|---|---|
| train | **3500** | `d2-train-NNNN` | SFT 训练 + GRPO rollouts |
| dev | **750** | `d2-dev-NNNN` | held-out reward 评测、调参 |
| test | **750** | `d2-test-NNNN` | held-out reward 评测、最终验证 |

**per-class split 算法**（round 14）：`_plan_per_class_counts(5000)` → `(834, 834, 833, 833, 833, 833)`；`assign_split_ids()` 用 `round(per_class × 0.15)` 计算 dev/test，使 833 与 834 类都得到 dev=test=125，train = per_class - 250。最终 per-split 数学：

- 833 类：train=583、dev=125、test=125
- 834 类：train=584、dev=125、test=125
- 总计：train = 4×583 + 2×584 = 3500；dev = 6×125 = 750；test = 6×125 = 750

**互斥保证**：

- id 命名空间前缀不同（`d2-train-` / `d2-dev-` / `d2-test-`）；
- 目录物理隔离（`datasets/tool-calling-d2/{train,dev,test}/`）；
- 与 D1 / D1.1 train id 在 stem 层不重叠（D2 id 是 `d2-*`，D1 是 `d1-*` / `d1llm-*`）；
- 生成器和测试均计算同一个 `canonical_content_signature()`：去除 `id` / `call_id` / `tool_call_id` / `depends_on`、`metadata.created_at` 和 `metadata.split` 后，仍保留 task_type、tools、消息内容、工具参数/结果与 expected_answer。**5000 个签名全局唯一；每类 833/833/833/833/834/834 唯一**；train/dev/test 三组 canonical signature 交集均为空。
- 与 D1 / D1.1 train 的 **跨数据集语义互斥**由 `cross_dataset_signature()` 投影保证：8 字段（task_type、schema_version、user_turns、assistant_turns、tool_turns、tool_names、expected_tool_calls、expected_answer）形状跨 D1/D1.1/D2 完全一致；`tests/test_d2_dataset.py::test_d2_canonical_content_is_disjoint_from_d1_d1llm_train` 验证 D2 三 split 与 D1/D1.1 train 在该投影下交集均为空；`test_cross_dataset_signature_is_comparable_across_d1_d1llm_d2` 保证投影本身可检测重叠（防止 round 11 那种两个不兼容投影退化为 vacuous ∅）；`test_cross_dataset_signature_detects_real_overlap` 为正向控制样本验证投影能真地检测重叠。

`tests/test_d2_dataset.py::D2SplitDisjointnessTests` 显式校验：

- `train ∩ dev = train ∩ test = dev ∩ test = ∅`；
- canonical semantic content 的三组 split 交集为空，且 5000 个样本签名全局唯一；
- 每个 task_type 都有 ≥833 个不同 canonical semantic instances；
- train 与 D1/D1.1 train id 集合无交集；
- 每个 `MANIFEST-{split}.json` 的 `count` / `build_count` / 每文件 sha256 与磁盘一致。

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
  "count": 3500,
  "build_count": 5000,
  "task_types": {
    "tool_not_available": 584,
    "tool_error_response": 584,
    "insufficient_result_search": 583,
    "req_change_city": 583,
    "multi_tool_sequential": 583,
    "error_recovery": 583
  },
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

`build_count` 是生成器 `--count` 参数的实际值（round 14 新增）；`aggregate_sha256` 是该 split 内按 id 排序后 hash — 可在测试中独立重算并验证。

### 5.1 时间戳契约

D2 采用与 D1 相同的确定性时间戳公式（与 `scripts/generate_d1_dataset.py` manifest 字段一致）：

```text
ts   = 1785000000 + seed + index
fmt  = datetime.fromtimestamp(ts, tz=UTC).isoformat().replace("+00:00", "Z")
```

其中 `index` 为样本在其 split 内的 1-based 位置（train 1..3500、dev 1..750、test 1..750；每个 split 独立编号，与 `sample.id` 后缀严格一致）。MANIFEST 的 `created_at` 使用 `index=0`（即 `seed` 本身的 epoch 秒）。选择 split-local 编号而非生成器全局序列好处是 `d2-train-0001` / `d2-dev-0001` / `d2-test-0001` 都映射到同一个 epoch 起始点，便于跨 split 对齐调试；该选择与 round 8 时间戳契约的"split-local 1-based index"语义一致。

**示例**：seed=2026、index=1 → `2026-07-25T17:53:47Z`；seed=2026、index=3500（d2-train-3500） → `2026-08-26T18:20:00Z`；seed=2026、index=750（d2-dev-0750） → `2026-07-26T00:48:50Z`。

该契约保证 `tests/test_d2_dataset.py::D2TimestampContractTests` 中反向测试全绿：同 `seed+index` 字节相同；不同 `seed` 或不同 `index` 均产出不同字符串；磁盘样本与公式逐字符相等。

## 6. 生成与复现

```bash
# 当前默认生成 5000 样本（round 14 契约）
.venv/python.exe scripts/generate_d2_dataset.py --count 5000 --seed 2026 \
    --out datasets/tool-calling-d2

# 自定义规模（每类至少 1 个，≥ 6；建议 ≥6×833=4998 以保持每类 ≥833 unique variants）
.venv/python.exe scripts/generate_d2_dataset.py --count 6000 --seed 2026 \
    --out datasets/tool-calling-d2

# 自定义 seed（默认 2026）
.venv/python.exe scripts/generate_d2_dataset.py --count 5000 --seed 2027 \
    --out datasets/tool-calling-d2

# 自定义输出目录
.venv/python.exe scripts/generate_d2_dataset.py --count 5000 --out datasets/tool-calling-d2-test
```

确定性保证：同一 `--seed` 产生相同的 canonical content、文件 sha256 和 MANIFEST aggregate hash；`--count` 通过 `_plan_per_class_counts()` 自动在 6 类 task_type 间分配（5000 / 6 = 833 remainder 2 → 4 类 833 + 2 类 834）。生成器写盘前以 `canonical_content_signature()` 执行全局语义去重；去除 ID、时间戳、split、call_id 和依赖引用后，默认数据仍保持 5000/5000 唯一、每类 ≥833/833 唯一、三 split 无交集。

## 7. 与 P2 离线 reward 校验的衔接

P2 阶段交付的 `scripts/reward_offline.py` 直接消费 D2 dev **750 样本**作为正式 reward 评测输入，命令：

```bash
.venv/python.exe scripts/reward_offline.py \
    --samples-dir datasets/tool-calling-d2/dev \
    --transcripts artifacts/<ckpt>-eval-d2dev.json \
    --output artifacts/<ckpt>-eval-d2dev-reward.json \
    --checkpoint <ckpt-name>
```

D1 dev 13 样本保留作为早期 dev 探针；D2 dev 750 样本是首个有统计意义的 reward 评测 split（vs D1 dev 的 13 样本 = 57.7× 提升）。

`scripts/eval_sft_tool.py --prompt-mode multi_turn` 负责把 D2 多轮 messages 序列化为 SFT 模板历史并生成续写，是 D2 dev 评测的标准推理入口。P3 阶段自研 5 ckpt 在该 split 上的历史评测（当前 active 范围外）归档于 `docs/plans/open-issues.md` P3-01 段（line 769-841）；P5-02 公开 5 模型在 D2 dev 750 子集 90 上的当前评测见 §7 后续段落。

## 8. 与 P4 GRPO / P5-02 的衔接

- P4 GRPO：以 D2 train (**3500** 样本) 作为 GRPO rollouts 的 prompt 池；D2 dev / test 作为 advantage 估计的对照基线；
- P5-02 Transformers backend：同一 `--samples-dir` 路径可在公开 instruction-tuned 模型上跑同一 reward offline 链路，与自研模型做公平对比。P5-02 阶段实际评测在当前扩样版 D2 dev 750 子集 90 上完成；公开 5 模型 reward_binary 全部 = 0（诚实负结果）、reward_layered 0.33–0.42，详见 `docs/plans/reviews/stage-p5-02-transformers-backend.md` 与 `docs/experiments/p2-evaluator/README.md` §7。

## 9. 与 D1 / D1.1 的差异

| 维度 | D1 模板版 | D1.1 LLM 生成版 | **D2 多轮版** |
|---|---|---|---|
| messages 长度 | 2（system + user）| 2（system + user）| 4-8（含 assistant + tool 角色）|
| tool_calls 出现位置 | 仅 `expected_tool_calls` | 同 D1 | 同 D1 + assistant messages 中 |
| tool_call_id 显式存在 | 否 | 否 | 是 |
| depends_on 链 | 是 | 是 | 是（多轮中显式串联）|
| Split（当前） | train/dev/test 100/13/13 | train-only 1500 | **train/dev/test 3500/750/750（IID 独立）** |
| Held-out 评测意义 | dev 13 样本（不足）| train-only（不算 held-out）| **dev 750 样本（首个有统计意义）** |
| 与自研模型的 reward 信号差异 | 可跑但样本过少 | 仅 train（不算泛化）| **正式 reward benchmark** |
| 适用 GRPO rollout 池规模 | 否 | 否（仅 train）| **3500 样本 train** |
