# LLM Lab

面向大语言模型研究与工程实践的个人实验项目，围绕以下闭环展开：

```text
模型架构实现 → 训练数据构造 → 模型训练/微调 → vLLM 部署 → 统一评测 → 失败分析
```

项目不追求一次性实现完整的大模型训练平台，而是先构建一个规模可控、结果可复现、能够持续扩展的实验体系。

## 项目目标

1. 在现有 CS336 Transformer 实现基础上，加入 MoE、GQA 和简化版 MLA，分析不同架构对模型效果、参数激活量、推理吞吐和 KV Cache 的影响。
2. 基于成熟开源训练框架，构建面向工具调用/结构化指令数据的数据处理管线，覆盖采集、清洗、去重、生成、校验和版本管理。
3. 构建统一模型评测框架，支持 Transformers 与 vLLM 推理后端，评估结构化输出、工具调用和任务完成能力，并进行失败案例分析。
4. 将架构实验、数据处理、SFT/GRPO 训练、部署和评测串联起来，形成可用于研究总结、面试展示和简历项目的实验闭环。

## 总体设计

项目总体路线图（已完成 / 进行中 / 下一阶段 / 暂缓阶段）见 [`docs/plans/roadmap.md`](docs/plans/roadmap.md)。问题和决策记录于 [`docs/plans/open-issues.md`](docs/plans/open-issues.md)；阶段审查记录于 [`docs/plans/reviews/`](docs/plans/reviews/)。

自研架构实验和开源模型后训练不强行使用同一个模型：

- **架构实验线**：使用自研小型模型，重点验证 Dense Transformer、MoE、MHA/GQA/MLA、KV Cache 和推理效率。
- **训练评测线**：使用 Transformers 生态中可直接加载的开源模型，重点进行数据处理、SFT/GRPO、vLLM 部署和能力评测。

这样可以避免自研 MLA/MoE 模型暂时无法接入 vLLM 或成熟训练框架时，阻塞其他模块的开发。

## 计划目录

```text
llm-lab/
├── architecture_lab/     # CS336 扩展：Dense、MoE、GQA、MLA
│   ├── models/
│   ├── train/
│   ├── inference/
│   ├── benchmarks/
│   └── tests/
│
├── data-pipeline/        # 工具调用/结构化指令数据管线
│   ├── collectors/
│   ├── processors/
│   ├── generators/
│   ├── validators/
│   ├── schemas/
│   └── data/
│
├── evaluation-lab/       # 统一推理、评测和失败分析
│   ├── tasks/
│   ├── backends/
│   │   ├── transformers_backend.py
│   │   └── vllm_backend.py
│   ├── metrics/
│   ├── executors/
│   ├── judges/
│   └── outputs/
│
├── serving/              # vLLM 部署和性能测试
├── configs/              # 实验配置
├── scripts/              # 运行入口
├── docs/                 # 计划、协议、环境和实验文档/结果
│   ├── plans/
│   ├── protocols/
│   └── experiments/
├── README.md
```

目录可以按阶段逐步创建，不要求一开始实现完整结构。

## 统一实验协议

### 模型输出记录

```json
{
  "schema_version": "1.0",
  "model": "model-name",
  "prompt": "...",
  "response": "...",
  "generation_config": {},
  "timestamp": "2026-08-29T00:00:00Z",
  "experiment_id": "exp-2026-08-29-001"
}
```

### 工具调用数据

```json
{
  "schema_version": "1.0",
  "id": "sample-001",
  "messages": [
    {
      "role": "user",
      "content": "..."
    }
  ],
  "tools": [],
  "expected_tool_calls": [],
  "expected_answer": "",
  "metadata": {
    "source": "synthetic",
    "license": "MIT",
    "task_type": "single_tool",
    "data_version": "v1",
    "pipeline_version": "v1",
    "created_at": "2026-08-29T00:00:00Z"
  }
}
```

`metadata` 必填子字段（见 `schemas/tool_calling_sample.schema.json` `$defs/metadata`）：`source`、`license`、`task_type`、`data_version`、`pipeline_version`、`created_at`；可选：`original_sample_id`、`validation`。

### 评测结果

```json
{
  "schema_version": "1.0",
  "model": "base-model",
  "task": "tool_calling",
  "metrics": {
    "tool_selection_accuracy": 0.0,
    "argument_accuracy": 0.0,
    "task_success_rate": 0.0,
    "format_error_rate": 0.0
  },
  "failures": [],
  "experiment_id": "exp-2026-08-29-001",
  "timestamp": "2026-08-29T00:00:00Z"
}
```

### 实验配置

使用 YAML 或 JSON 管理模型、数据、推理和评测配置，例如：

```yaml
model:
  name: model-name
  backend: transformers

data:
  version: v0.1

evaluation:
  tasks:
    - tool_selection
    - structured_output

generation:
  temperature: 0.0
  max_new_tokens: 512
```

### 测试与验证

测试按反馈速度分为三档，统一入口为 `scripts/run_tests.py`：

```bash
# 日常快速检查：API/schema/forward 正确性
.venv/python.exe scripts/run_tests.py fast

# 按模块回归
.venv/python.exe scripts/run_tests.py module training
.venv/python.exe scripts/run_tests.py module data

# 阶段审查/commit 前全量检查
.venv/python.exe scripts/run_tests.py full
```

详细范围和新增测试规则见 [`docs/protocols/testing.md`](../protocols/testing.md)。真实 OWT 训练和 benchmark 不属于默认测试档，必须显式运行并记录实验结果。项目路线图与阶段计划见 [`docs/plans/roadmap.md`](../plans/roadmap.md)。

## 模块 A：LLM Architecture Lab
### A1. 固定 Dense Transformer 基线

保留并整理现有 CS336 A1 实现：

- BPE 分词器；
- Decoder-only Transformer；
- RoPE；
- RMSNorm；
- SwiGLU；
- Causal Attention；
- 训练、生成和 checkpoint 保存；
- 基础单元测试。

先记录基线数据：

- 总参数量；
- 训练 loss 和验证 loss；
- 训练吞吐；
- 推理吞吐；
- 显存占用；
- full forward 与 incremental decode 的一致性。

### A2. 实现 MoE

第一版只做单机、单卡或简单 batch 下的 MoE，不实现 Expert Parallel。

实现顺序：

```text
SwiGLU FFN → Router → Top-1 Routing → Top-2 Routing
                         ↓
              Load Balancing Loss
                         ↓
                Capacity / Overflow
```

需要实现或记录：

- Router；
- Top-1/Top-2 expert selection；
- token dispatch/combine；
- capacity factor；
- load balancing loss；
- expert token 数量和负载统计；
- overflow 或 dropped token 处理；
- forward/backward 和 checkpoint 测试。

实验对比：

```text
Dense FFN vs MoE Top-1 vs MoE Top-2
```

记录总参数量、激活参数量、验证集 loss、训练吞吐、专家负载、token 丢弃率和实际显存。

### A3. 实现 MHA、GQA 和简化 MLA

按以下顺序推进：

```text
MHA → GQA → MLA
```

MLA 第一版只要求完成可运行的简化实现和正确性验证，重点关注：

- latent KV compression；
- query/key 维度设计；
- RoPE 处理；
- compressed KV 与位置编码信息；
- prefill/decode 阶段 cache；
- 增量解码。

必须测试：

1. full forward 与 incremental decode 输出一致；
2. 不同 batch size 输出一致；
3. KV Cache 长度和 shape 正确；
4. 训练与推理 shape 正确；
5. 长上下文下没有明显数值异常。

实验对比：

```text
MHA vs GQA vs MLA
```

记录 KV Cache 占用、prefill 延迟、decode 吞吐、验证集 loss 和显存。

## 模块 B：Instruction Data Pipeline

### B1. 第一版数据类型

第一版只支持**工具调用/结构化指令数据**，不同时扩展普通问答、代码、多模态、数学和偏好数据。

先设计 3–5 个可控工具，例如：

- 文件搜索；
- JSON 查询；
- 表格统计；
- 日期计算；
- 文本信息抽取。

任务覆盖：

- 不需要工具；
- 单工具调用；
- 多工具串联；
- 工具参数错误；
- 工具执行失败；
- 用户中途修改要求；
- 工具结果不足以回答问题。

### B2. 数据处理流程

```text
原始数据
→ 格式归一化
→ 来源与许可证记录
→ 精确去重
→ 近似去重
→ 质量过滤
→ 任务分类
→ 指令生成
→ Schema 校验
→ 工具执行验证
→ 数据版本发布
```

每条数据尽量记录：

- 来源和许可证；
- 原始样本 ID；
- 采集/处理时间；
- 任务类型；
- 数据管线版本；
- Schema 校验结果；
- 工具执行结果；
- 最终样本是否通过质量检查。

### B3. 数据生成与验证

不要采用“调用 LLM 后直接保存 JSON”的流程。推荐：

```text
任务模板
→ 参数采样
→ LLM 生成
→ 规则检查
→ JSON Schema 校验
→ 工具执行
→ 最终答案检查
→ 合格样本入库
```

重点验证：

- 工具名称是否存在；
- 参数是否符合 schema；
- 工具是否能够成功执行；
- 调用顺序是否合理；
- 最终答案是否使用真实工具结果；
- 是否编造工具结果；
- 是否发生不必要的工具调用。

### B4. 开源框架使用策略

不从头实现训练框架。优先使用成熟框架，并将自己的工作放在数据处理、质量控制、评测接口和必要的训练扩展上。

- **TRL**：优先用于 SFT、GRPO 和自定义 reward 实验；
- **LLaMA-Factory**：用于快速比较不同模型、LoRA/QLoRA 和数据版本；
- **Transformers**：作为模型、tokenizer 和基础推理接口。

修改顺序：

1. 先通过自定义数据处理器、模板、统计脚本和 callback 完成外围扩展；
2. 只有在接口不满足需求时，才修改或 fork dataset processor、reward function 或 evaluation hook。

### B5. 数据的下游效果验证

至少建立三个数据版本：

```text
D0：原始/基础数据
D1：清洗后的数据
D2：清洗 + 合成 + 执行验证的数据
```

使用同一模型和训练配置进行：

```text
Base → D0 SFT → D1 SFT → D2 SFT
```

统一比较工具选择准确率、参数准确率、Schema 通过率、工具执行成功率、最终任务完成率和无效调用率。

## 模块 C：LLM Evaluation Lab

### C1. 统一推理后端

定义统一接口：

```python
class InferenceBackend:
    def generate(self, messages, tools=None, **kwargs):
        ...
```

实现：

- `TransformersBackend`；
- `vLLMBackend`。

评测逻辑不直接依赖具体推理后端。

### C2. 评测任务和指标

优先实现确定性指标：

- Accuracy；
- Exact Match；
- F1；
- JSON Schema 通过率；
- 工具名称准确率；
- 参数字段准确率；
- 工具执行成功率；
- 任务完成率；
- pass@1。

工具调用专项指标：

- 是否正确选择工具；
- 参数是否正确；
- 工具调用顺序是否合理；
- 是否发生无效调用；
- 是否使用工具返回结果；
- 多轮任务是否保留约束；
- 平均调用次数。

开放式任务再加入 LLM Judge，并保存 judge 模型、版本、prompt、评分标准和人工抽样结果。

### C3. 失败案例分析

每个失败样本保存：

```json
{
  "input": "...",
  "expected": "...",
  "prediction": "...",
  "failure_type": "wrong_tool",
  "raw_output": "...",
  "analysis": "..."
}
```

第一版失败类型：

- 工具选择错误；
- 参数错误；
- JSON 格式错误；
- 不必要调用工具；
- 工具结果未使用；
- 编造工具结果；
- 最终答案错误；
- 多轮约束丢失；
- 重复调用；
- 超出上下文或超时。

## 模块 D：vLLM Serving

### D1. 标准开源模型优先

第一阶段先支持标准 Transformers/vLLM 模型：

- Transformers 加载；
- vLLM 部署；
- 统一生成接口；
- Transformers 与 vLLM 输出对比；
- 延迟、吞吐和显存 benchmark。

记录：

- 模型名称和规模；
- 输入/输出长度；
- 并发数；
- prefill 延迟；
- decode 吞吐；
- 峰值显存；
- 输出一致性。

### D2. 自研架构后续适配

自研 MoE/MLA 模型后续再尝试接入 vLLM，可能涉及：

- Hugging Face Config；
- PreTrainedModel；
- 权重转换；
- vLLM 模型注册；
- attention backend；
- KV Cache 逻辑；
- 权重命名映射。

这不是第一阶段的阻塞条件。自研架构首先使用原生 PyTorch 验证正确性和性能。

## 开发阶段和顺序

### 阶段 0：统一协议

产物：

- 数据 schema；
- 模型输出 schema；
- 评测结果 schema；
- 实验配置格式；
- 最小示例数据和结果。

### 阶段 1：Dense Transformer 基线

固定当前 CS336 实现，补充基准 loss、参数、吞吐、显存和测试记录。

### 阶段 2：MoE 和注意力扩展

依次完成：

1. MoE Top-1；
2. MoE Top-2 和负载均衡；
3. MHA/GQA；
4. 简化 MLA；
5. 正确性测试；
6. 架构对比实验。

### 阶段 3：评测框架骨架

先实现 Transformers backend、vLLM backend、统一输出格式、基础指标和结果汇总。

### 阶段 4：工具调用数据管线

完成采集、清洗、去重、生成、Schema 校验、工具执行验证和数据版本管理。

### 阶段 5：SFT/GRPO 训练

使用开源训练框架训练 D0/D1/D2 数据版本，形成 Base、SFT 和 GRPO 对比。

### 阶段 6：统一评测和报告

使用 Evaluation Lab 评测模型，生成指标表、失败样本和实验报告，并将问题反馈到数据和 reward 设计。

## MVP 范围

### Architecture MVP

- Dense Transformer；
- MoE Top-1；
- MHA；
- 简化 MLA；
- forward/decode 测试；
- Dense/MoE 对比；
- MHA/MLA 对比。

### Data MVP

- 3–5 个工具；
- 1 种工具调用数据格式；
- 清洗；
- 去重；
- 合成；
- Schema 验证；
- 工具执行验证；
- 至少一个数据版本对比实验。

### Evaluation MVP

- Transformers backend；
- vLLM backend；
- 工具选择准确率；
- 参数准确率；
- Schema 通过率；
- 任务完成率；
- 失败类型统计。

第一版暂不要求：

- Expert Parallel；
- 分布式训练；
- 完整工业级 MLA；
- 多模态数据；
- 复杂联网 Agent；
- 大规模数据采集；
- 复杂 LLM Judge；
- 自研模型完整 vLLM 适配。

## 预期成果

### 代码

- 可运行模型、数据和评测代码；
- 单元测试和正确性测试；
- 训练/推理脚本；
- 配置文件；
- 数据 schema；
- benchmark 和结果汇总脚本。

### 报告

每个模块至少包含：

- 问题定义；
- 实现方法；
- 实验配置；
- 对比结果；
- 失败案例；
- 局限性；
- 后续计划。

### 简历项目方向

#### 模型架构项目

基于 PyTorch 扩展 Decoder-only Transformer，实现 MoE FFN、Top-K 路由和 MLA 注意力模块；在统一训练与推理设置下，对比 Dense/MoE 及 MHA/GQA/MLA 的模型效果、激活参数量、吞吐和 KV Cache 占用，并完成标准模型的 vLLM 部署测试。

#### 数据管线项目

基于开源训练框架构建工具调用指令数据管线，完成数据采集、清洗、去重、样本生成、Schema 校验和工具执行验证；通过数据版本对比实验分析不同数据处理策略对 SFT 模型工具选择和任务完成率的影响。

#### 评测项目

构建统一 LLM 评测框架，支持 Transformers 与 vLLM 推理后端，覆盖结构化输出、工具调用和任务完成率等指标；对比 Base、SFT 和 GRPO 模型并归纳典型失败模式。

## 当前状态

- [x] 创建项目基础目录；
- [x] 实现 Dense Transformer 基线初版；
- [x] 完成 Dense baseline 正确性测试和 smoke benchmark；
- [x] 确定工具调用任务和数据 schema；
- [x] 确定评测指标和结果格式；
- [x] 实现 MoE Top-1 MVP；
- [x] 实现 N2 Dense/MoE 公平对比协议（A/B）；
- [x] 实现 N3 统一 benchmark/result 元数据；
- [x] 完成 D1 模板版与 D1.1 LLM 生成版工具调用数据管线（manifest、source provenance、MockExecutor 校验）；
- [x] 完成 P1-02 mock 工具执行器（历史阶段文件名为 P1-01）及 P1-05 八级失败分类器；
- [x] 完成 P1-03 多 seed 评测协议与 eval 聚合器；
- [x] 完成 SFT 工具调用训练 MVP（Dense + MoE，含诚实负结果）；
- [ ] 实现 Transformers/vLLM 统一推理接口；
- [x] 实现最小数据管线；
- [x] 实现最小评测器与八级失败分类；
- [x] 完成第一组端到端实验（D1/D1.1 + Dense/MoE SFT）；

第一版 Dense 训练数据：使用 Stanford CS336 OWT sample，详见 `docs/data/owt-sample.md`。OWT 原始文件位于 `data/raw/owt-sample/`，不会提交到 Git。

## 运行环境

当前项目使用仓库内 `.venv` Conda 环境，而不是 base 或其他项目环境。环境信息和激活方式见 `docs/environment.md`。该环境中的 PyTorch 已验证支持 RTX 5070 Ti 的 `sm_120`，Dense baseline 和 MoE Top-1 smoke benchmark 当前可以使用 CUDA 运行。

## 原则

1. 先做小而完整的闭环，再扩展功能。
2. 自研架构和开源模型训练线解耦，避免互相阻塞。
3. 不把“使用开源框架”本身当作成果，重点记录改动、验证和实验结论。
4. 不把“使用 vLLM 部署”当作成果，重点记录延迟、吞吐、显存和输出一致性。
5. 所有模型效果结论都应有统一数据、固定配置和可复现实验支撑。
6. 数据项目必须记录来源、许可证、处理版本和验证结果。
7. 真实结果出来后再写入简历，不提前使用未验证的指标。
