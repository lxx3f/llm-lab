# 阶段 0：统一协议

## 目标

1. 工具调用数据：`schemas/tool_calling_sample.schema.json`
2. 模型输出记录：`schemas/model_output.schema.json`
3. 评测结果：`schemas/evaluation_result.schema.json`
4. 实验配置示例：`configs/stage0.example.yaml`

示例文件位于：

- `examples/tool_calling/sample-001.json`：单工具调用；
- `examples/tool_calling/sample-002-no-tool.json`：无需工具；
- `examples/tool_calling/sample-003-multi-tool.json`：有依赖关系的多工具串联；
- `examples/model_outputs/sample-001.json`
- `examples/evaluation_results/sample-001.json`

## 校验方式

环境依赖：

- Python 3.10+
- `jsonschema`
- `PyYAML`（读取配置文件时使用）

在项目根目录运行：

```bash
python scripts/validate_stage0.py --examples
python -m unittest discover -s tests -p "test_*.py" -v
```

第一条命令校验已提交的正例 JSON；第二条命令同时执行正例校验和负例测试。

## Schema 负例测试

`tests/test_stage0_schemas.py` 覆盖以下必须被拒绝的情况：

- 工具定义缺少工具名称；
- 期望工具调用的参数不是对象；
- 评测失败类型不在约定枚举中；
- 工具调用数据缺少来源；
- 工具调用数据缺少许可证。

这些测试只验证结构和元数据完整性，不执行真实工具。

## 版本约定

- 当前 schema 版本：`1.0`
- 当前数据版本示例：`D0`
- 当前管线版本示例：`stage0`
- 时间字段统一使用 ISO 8601 UTC 格式，例如：`2025-08-25T00:00:00Z`
- `experiment_id` 用于关联模型输出、评测结果和实验配置。

## 数据来源和许可证

提交到仓库的示例数据为人工构造，仅用于协议验证，metadata 中记录：

- `source: llm-lab-manual-example`
- `license: CC0-1.0`

真实数据接入时必须替换为可审计的来源、许可证、原始样本 ID 和处理版本，不能沿用示例元数据。

## 指标约定（第一版）

工具调用评测至少记录：

- `tool_selection_accuracy`：预测工具名称与期望工具名称一致的样本比例；
- `argument_accuracy`：工具参数通过定义的参数比较规则的样本比例；
- `schema_pass_rate`：模型输出通过 JSON Schema 的比例；
- `tool_execution_success_rate`：工具调用成功执行的比例；
- `task_success_rate`：工具调用和最终答案均满足任务要求的比例；
- `invalid_call_rate`：无效工具调用数量除以模型工具调用总数，具体分母需在评测报告中注明。

如果指标依赖自定义比较规则，应在实验报告或配置中保存规则版本。

## 当前边界

阶段 0 不包含：

- 推理后端实现；
- 工具执行器；
- 数据采集和清洗逻辑；
- 模型训练；
- 真实 benchmark 结论。
