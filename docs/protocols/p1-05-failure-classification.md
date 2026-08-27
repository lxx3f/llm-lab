# P1-05 工具执行失败层级分类协议

> 状态：已解决（2026-08-27，D1 数据集 + 八级分类器落地）。

## 背景

工具调用评测如果只记录"工具执行成功率"和"任务完成率"，无法定位模型失败发生在哪一步。P1-05 固定八级错误分类。

## 八级失败层级（v3，2026-08-27 修订）

```text
parse_success          ← transcript 结构可解析为 tool_calls 列表（空列表是合法解析）
→ schema_valid         ← 每个 call 的 arguments 符合命名工具的声明 schema
→ tool_name_correct    ← transcript 的工具名 multiset 与 expected 一致（不含顺序）
→ argument_value_correct ← transcript 的 (name, arguments) multiset 与 expected 一致（不含顺序）
→ call_plan_matches    ← 顺序 / depends_on / 数量一致（name/args 已由上层细粒度定位）
→ execution_success    ← MockExecutor 是否成功执行（含依赖序）
→ result_grounded      ← 实际工具结果是否等于 expected_result（未声明时 N/A）
→ final_answer_correct ← final_answer 是否包含 expected_answer（null 时 N/A；
                         ``task_success`` 为同值向后兼容别名）
```

**层语义规则**：

- `no_tool` 样例（expected_tool_calls 为空）：正确输出 = 空 tool_calls；parse/schema/name/args/execution/grounding 平凡通过（result_grounded = True，无结果可验证）；
- `tool_error` 样例分两种：
  - `tool_not_available`：用户请求的工具不在 available tools 列表中 → 模型不应调用任何工具，应报告工具不可用（expected_tool_calls = []）；
  - `tool_error_response`：模型调用一个始终返回 ERROR 响应的工具 → 模型应观察到错误响应并报告，不应传播错误结果（expected_tool_calls 含 1 个调用 + expected_result 为 ERROR 串）；
- `expected_answer: null`：final_answer_correct = None（不可判定），不计入失败；
- 有 expected calls 但未声明 `expected_result`：result_grounded = None（不可判定），不计入失败；
- **定位精度**：错误的工具名（含不在声明工具注册表中的名字）→ tool_name_correct；正确的 name 但错误的参数值 → argument_value_correct；name/args 都正确但顺序/依赖错误 → call_plan_matches；
- **未知工具名**：schema_valid 对该 call 不可判定（None，因无 schema 可查），名字错误由 tool_name_correct 报告；
- **鲁棒性**：非字符串 name（None/int/dict）、任意形状 arguments（None/str/list/非 str key dict）、混合类型 depends_on（str/int）、不可哈希 call_id（dict）全部归一化比较，不崩溃，一律视为与 expected 不匹配；
- **首次失败层**（None 跳过）= 模型失败的具体位置；答案层失败报告 `final_answer_correct`（非别名 `task_success`）。

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
- `result_grounded` 依赖样例的 `expected_result` 字段（D1 生成器对所有有 expected_tool_calls 的样例均写入 deterministic 值）。

## D1 数据集（本阶段交付）

`datasets/tool-calling-d1/`：

- 生成器：`scripts/generate_d1_dataset.py`（确定性模板，seed=2026，可复现）；
- **126 样例**（train=100 / dev=13 / test=13），覆盖 6 种 task_type：
  - no_tool 18 / single_tool 36 / multi_tool 18 / tool_error 18 / insufficient_result 18 / requirement_change 18；
- 每个样例 schema v1.0 valid；MANIFEST.json（每文件 sha256 + 聚合 hash + split）；
- 所有有调用的样例（117/117 expected calls）都有 deterministic `expected_result`（由 `examples.d1_mocks` 计算）；
- 生成器在写盘前调用 MockExecutor 端到端验证所有 expected_tool_calls 执行成功 + result 一致；生成 exit 非 0 代表任何样例语义失败；
- `metadata.source = "d1-synthetic-template"`，`pipeline_version = "p1-05-d1-generator"`。

### 模板清单

| task_template | 说明 |
|---|---|
| no_tool_greeting | 直接回答，不调用工具 |
| single_calc | 单一计算工具 |
| single_weather | 单一天气工具 |
| multi_calc_search | 计算 + 搜索依赖序 |
| tool_not_available | 可用工具中不含所需工具 → 报告不可用（不应调用） |
| tool_error_response | 调用返回 ERROR 响应的工具 → 报告错误（真实工具错误场景） |
| insufficient_search | 结果不足 → 追问用户 |
| req_change_city | 中途改要求 → 以最新为准 |

## 校验

- `scripts/run_tests.py full`：tests/test_d1_failure.py（43 tests：manifest / schema / task_type 覆盖 / 确定性 / 八级分类各一 / 全 task_type 集成 / D1 端到端 mock 执行 / expected_result 一致性 / depends_on 可达性 / result_grounded 实际激活 / MalformedTranscriptRegressionTests 8 个反向断言 / call_id 不匹配 位置配对回归 2 个 / 八级扩展 4 个 / 未知工具名 + 非字符串 name/arguments 鲁棒性 3 个 / call_id+depends_on 归一化 2 个）；
- 生成器退出码 0 且 0 schema errors。

## 应用范围

- D0 样例集（3 个手工样例）+ D1 数据集（126 个）均可作为分类器输入；
- 真实推理后端（Transformers/vLLM）接入后，transcript 由解码输出生成；
- 失败分布统计（哪些层失败最多）是后续 P2 评测报告的输入。

## 遗留

- 真实 LLM 生成的 transcript（D1.1 需 LLM API 凭证）尚未接入；mock-only 约束下用确定性模板 + examples.d1_mocks，`source` 标注为 d1-synthetic-template。canonical D1 已通过 MockExecutor 端到端验证（117/117 expected calls 全部可执行且 result 与 expected_result 一致），result_grounded 层在 canonical 数据上实际激活；
- 八级分类的每一级在 canonical D1 上的失败分布统计（哪些层失败最多）是后续 P2 评测报告的输入（当前无真实模型 transcript，无法产生分布）。