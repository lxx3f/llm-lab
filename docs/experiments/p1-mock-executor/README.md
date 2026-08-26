# P1-01 Mock Tool Executor 实验记录

> 状态：P1-01 阶段交付。

## 交付物

| 文件 | 说明 |
|---|---|
| `architecture_lab/execution/__init__.py` | 模块导出 |
| `architecture_lab/execution/mock_executor.py` | MockExecutor（register / execute / execute_sequence）|
| `schemas/tool_execution_result.schema.json` | 执行结果 schema v1.0 |
| `scripts/run_mock_executor.py` | CLI（--input / --mocks / --output）|
| `examples/mock_execution/sample-mock-001.json` | calculator 示例（add + multiply）|
| `examples/mock_execution/sample-mock-001.mocks.json` | mock 注册表 |
| `examples/mock_execution/mocks.py` | 纯函数实现（add / multiply / divide）|
| `tests/test_mock_executor.py` | 8 个单测 |

## 端到端 smoke

```bash
.venv/python.exe scripts/run_mock_executor.py \
    --input examples/mock_execution/sample-mock-001.json \
    --mocks examples/mock_execution/sample-mock-001.mocks.json \
    --output artifacts/mock-execution-sample-001-result.json
```

输出（`artifacts/mock-execution-sample-001-result.json`）：

```json
[
  {"tool_name": "add", "call_id": "call-1", "outcome": "success", "result": 5, ...},
  {"tool_name": "multiply", "call_id": "call-2", "outcome": "success", "result": 20, ...}
]
```

- `add(2, 3) = 5` → `success`；
- `multiply(5, 4) = 20`（depends_on call-1）→ `success`。

## 测试统计

`scripts/run_tests.py full`：**108 tests passed**（100 → 108，+8 mock executor 单测）。

## 设计决策记录

1. **纯进程内**：mock 不碰 subprocess / 文件系统 / 网络，保证零副作用与确定性；
2. **arguments_schema 校验**：每个 mock 声明自己的参数 JSON Schema，`argument_invalid` 在调用前拦截；
3. **依赖序最小实现**：`depends_on` 用简单拓扑序（前序失败则跳过），不做并发；
4. **异常捕获**：mock 异常 → `mock_exception`（保留 `TypeName: message`），不暴露 traceback；
5. **importlib 加载**：mock 注册表用 `"module:attr"` 字符串指向 Python 函数，CLI 通过 importlib 解析——不引入新依赖。

## 后续（不在 P1-01）

- P1-02：沙箱执行器（subprocess + timeout + 资源限制）；
- P1-03：失败分类五级（parse / schema / execution / grounded / task）；
- 真实推理后端对接（模型输出 → tool call 解析 → 执行器）。