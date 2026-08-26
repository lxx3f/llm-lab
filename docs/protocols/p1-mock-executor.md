# P1-01 Mock Tool Executor 协议

> 状态：P1-01 阶段交付。本文档描述纯进程内 mock 工具执行器的接口、schema 与边界。

## 目标

实现零外部副作用的 mock 工具执行器，为评测链路提供结构化执行层：

- 不依赖 subprocess / 文件系统 / 网络；
- mock 工具是纯 Python 函数（在注册表里声明）；
- 执行结果结构化为 `tool_execution_result.schema.json` v1.0；
- 支持 `expected_tool_calls` 顺序执行 + `depends_on` 依赖图（简单拓扑序）。

**不是**：真实工具执行；LLM 调用；网络 IO；持久化；并发调度。

## 模块结构

```
architecture_lab/execution/
├── __init__.py          # 导出 MockExecutor + validate_execution_result
└── mock_executor.py     # MockExecutor 实现
```

## 接口

```python
class MockExecutor:
    def register_mock(self, tool_name: str, fn: Callable, arguments_schema: dict) -> None
    def registered_tools(self) -> list[str]
    def execute(self, call: dict) -> dict            # 单次调用
    def execute_sequence(self, calls: list[dict]) -> list[dict]  # 依赖序
```

- `call` 至少含 `tool_name` / `call_id` / `arguments`；
- `execute_sequence` 中 `depends_on` 是前序 `call_id` 列表；前序失败则本调用以 `mock_not_found` 跳过（error 注明 "did not complete"）；
- 所有 mock 异常被捕获为 `mock_exception`（不暴露 traceback，只保留 `TypeName: message`）。

## outcome 枚举（P1-01 子集）

| outcome | 触发条件 |
|---|---|
| `success` | mock 函数正常返回 |
| `mock_not_found` | 注册表无该工具，或依赖未完成被跳过 |
| `argument_invalid` | arguments 不符合该工具的 arguments_schema |
| `mock_exception` | mock 函数抛错 |
| `execution_timeout` | P1-02 保留（本阶段不触发）|

## Schema（v1.0）

`schemas/tool_execution_result.schema.json`：

- `schema_version` / `tool_name` / `call_id` / `outcome` / `arguments` / `result` / `error` / `execution_time_ms` / `mock_metadata`
- `outcome=success` 时 `result` 非 null；
- `outcome != success` 时 `error` 非 null；
- `execution_time_ms >= 0`。

## CLI

```bash
.venv/python.exe scripts/run_mock_executor.py \
    --input examples/mock_execution/sample-mock-001.json \
    --mocks examples/mock_execution/sample-mock-001.mocks.json \
    --output artifacts/mock-execution-sample-001-result.json
```

- `--input`：tool_calling_sample schema JSON（读取 `expected_tool_calls`）；
- `--mocks`：mock 注册表（`tool_name -> {function_path: "module:attr", arguments_schema}`），`function_path` 通过 importlib 解析；
- 输出：`tool_execution_result` 数组 JSON。

## Example

`examples/mock_execution/`：

- `sample-mock-001.json`：calculator 任务（add + multiply，依赖序 call-1 → call-2）；
- `sample-mock-001.mocks.json`：add / multiply / divide 注册表；
- `mocks.py`：纯函数实现。

## 失败分层与 P1-03 关系

- P1-01 只实现 outcome 5 枚举（失败模式的最小集合）；
- P1-03 失败分类（parse_success / schema_valid / execution_success / result_grounded / task_success 五级）是后续阶段，不在 P1-01 展开。

## 测试

`tests/test_mock_executor.py`（8 tests）：

1. register + execute success（result=5）
2. unknown tool → mock_not_found
3. invalid arguments → argument_invalid
4. mock exception → mock_exception
5. execute_sequence respects depends_on
6. execute_sequence skips on failed dependency
7. schema rejects invalid outcome
8. CLI end-to-end（sample-mock-001 → 2 个 success）

## 边界

- ❌ 不执行真实工具（subprocess / 网络 / 文件系统 → P1-02）；
- ❌ 不做并发调度；
- ❌ 不做 LLM 调用；
- ❌ 不做语义校验（result 仅 schema 校验）。

## 退出条件

- MockExecutor + schema + CLI + example + 8 tests 全部就绪；
- 1 个 example 端到端 smoke 跑通；
- 协议/实验/审查三类文档落盘；
- `scripts/run_tests.py full` 108 tests passing；
- 一次 stage review 通过 isolated auditor。