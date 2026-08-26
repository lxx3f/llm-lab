# P1-05 工具执行失败层级分类协议

> 状态：已解决（2026-08-27，D1 数据集 + 五层分类器落地）。

## 背景

工具调用评测如果只记录"工具执行成功率"和"任务完成率"，无法定位模型失败发生在哪一步。P1-05 固定五层错误分类。

## 五层失败层级

```text
parse_success      ← 模型输出能否解析为结构化 tool calls
→ schema_valid     ← 每个 call 是否匹配工具声明的参数 schema
→ execution_success ← MockExecutor 是否成功执行（含依赖序）
→ result_grounded  ← 实际工具结果是否等于 expected_result
→ task_success     ← final_answer 是否包含 expected_answer
```

**首次失败层** = 模型失败的具体位置。例如：
- `parse_success` 失败：模型输出无法解析；
- `schema_valid` 失败：参数缺字段/类型错；
- `execution_success` 失败：mock 抛错/依赖未完成；
- `result_grounded` 失败：工具返回了错误结果（模型没正确传参）；
- `task_success` 失败：工具调用全对但最终回答错误。

> 完整八级（open-issue P1-05 原文含 tool_name_correct / argument_value_correct / final_answer_correct）在需要更细粒度时扩展；当前五层覆盖 parse→execute→ground→answer 的评测闭环。

## 分类器接口

```bash
.venv/python.exe scripts/classify_tool_failure.py \
    --sample datasets/tool-calling-d1/train/d1-0001.json \
    --transcript artifacts/transcript.json \
    --output artifacts/classification-result.json
```

`transcript.json` 格式：

```json
{
  "tool_calls": [
    {"call_id": "c1", "name": "calculate", "arguments": {"expression": "1+1"},
     "execution_outcome": "success", "result": "2"}
  ],
  "final_answer": "答案是 2"
}
```

输出（`--output`）：`schema_version / sample_id / layers / first_failure`。

## 与 P1-01 mock executor 的关系

- P1-01 MockExecutor 产生 `execution_outcome`（success / mock_not_found / argument_invalid / mock_exception）；
- P1-05 分类器消费这些 outcome 作为 `execution_success` 层的输入；
- `result_grounded` 依赖样例的 `expected_result` 字段（D1 生成器可选写入）。

## D1 数据集（本阶段交付）

`datasets/tool-calling-d1/`：

- 生成器：`scripts/generate_d1_dataset.py`（确定性模板，seed=2026，可复现）；
- **126 样例**（train=100 / dev=13 / test=13），覆盖 6 种 task_type：
  - no_tool 18 / single_tool 36 / multi_tool 18 / tool_error 18 / insufficient_result 18 / requirement_change 18；
- 每个样例 schema v1.0 valid；MANIFEST.json（每文件 sha256 + 聚合 hash + split）；
- `metadata.source = "d1-synthetic-template"`，`pipeline_version = "p1-05-d1-generator"`。

### 模板清单

| task_template | 说明 |
|---|---|
| no_tool_greeting | 直接回答，不调用工具 |
| single_calc | 单一计算工具 |
| single_weather | 单一天气工具 |
| multi_calc_search | 计算 + 搜索依赖序 |
| tool_missing_arg | 工具缺必要参数 → 应报错 |
| insufficient_search | 结果不足 → 追问用户 |
| req_change_city | 中途改要求 → 以最新为准 |

## 校验

- `scripts/run_tests.py full`：tests/test_d1_failure.py（10 tests：manifest / schema / task_type 覆盖 / 确定性 / 五层分类各一）；
- 生成器退出码 0 且 0 schema errors。

## 应用范围

- D0 样例集（3 个手工样例）+ D1 数据集（126 个）均可作为分类器输入；
- 真实推理后端（Transformers/vLLM）接入后，transcript 由解码输出生成；
- 失败分布统计（哪些层失败最多）是后续 P2 评测报告的输入。

## 遗留

- 完整八级分类（tool_name_correct / argument_value_correct / final_answer_correct）未实现——当前五层够用，扩展留到正式评测；
- D1 的 `expected_result` 字段大多为空（模板未固化结果），`result_grounded` 层在真实 mock 执行时启用；
- D1 样例未用真实 LLM 生成（mock-only 约束下用确定性模板），`source` 标注为 d1-synthetic-template。