# 计划问题记录

> 状态：暂存，逐项讨论解决。
>
> 本文只记录当前计划和实现中发现的问题，不在本轮直接修改实现。后续每解决一个问题，应补充：决策、改动、验证命令、结果和遗留风险。

## 使用方式

问题按优先级分为：

- **P0**：会直接影响实验正确性、结果可比性或后续开发；存在未关闭 P0 时不得以“有条件通过”完成阶段审查或创建阶段 commit。审查还必须确认模型身份字段完整，否则按阻塞处理；
- **P1**：会影响复现、评测质量或范围控制；通常需要在相关阶段完成前处理或明确延期；
- **P2**：文档、工程化或后续优化问题。

建议每次只处理一个主题，避免在未确定协议前同时扩展多个模型模块。

## 当前建议顺序

```text
Dense 训练闭环
→ Dense/MoE Top-1 公平对比协议
→ 统一实验元数据和 benchmark schema
→ MoE capacity/decode 语义
→ 工具执行器与语义校验器
→ 最小评测器
→ 数据版本和 SFT
→ Top-2/GQA/简化 MLA
→ GRPO/vLLM 扩展
```

---

## P0：架构实验基础问题

### P0-01 Dense 基线尚未完成真实训练闭环

**讨论结论（2026-08-25）**

不采用 byte-level tokenizer 作为最终方案。用户已有 CS336 Assignment 1 BPE 实现，已从指定仓库完成审查：

```text
https://github.com/lxx3f/assignment1-basics
```

临时审查目录不放入当前项目：

```text
/tmp/assignment1-basics-review
```

仓库中确认存在以下可复用模块：

- `cs336_basics/tokenizer.py`：BPE tokenizer、特殊 token、encode/decode、文件加载；
- `cs336_basics/train_bpe.py`：GPT-2 风格 regex pre-tokenization 和 BPE 训练；
- `cs336_basics/data.py`：随机 batch 采样、checkpoint 保存/加载；
- `cs336_basics/train_lm.py`：训练 loop、验证、tokenizer 准备、数据缓存、续训；
- `cs336_basics/optimizer.py`：AdamW、梯度裁剪、warmup + cosine schedule；
- `tests/`：tokenizer、data、model、optimizer、checkpoint 等参考测试。

该仓库的 `LICENSE` 为 MIT，后续若复制或改写其代码，需要保留许可证和版权声明，并在项目文档中记录来源。当前只复用设计和经过审查的实现思路，不直接整仓库复制。

**新的实施方向**

```text
Assignment 1 BPE 实现
→ 适配到 llm-lab 的 tokenizer 接口
→ 固定 tokenizer artifact 和 vocab_size
→ 文本数据编码缓存
→ Dense 训练 loop
→ validation loss
→ checkpoint save/load
→ generation
```

**需要重点适配的地方**

1. 当前 Dense 模型默认 `vocab_size=256`，接入 BPE 后必须从 tokenizer artifact 动态读取实际词表大小，不能继续硬编码 256；
2. tokenizer 的 vocab、merges、special tokens 和训练语料版本需要一起记录；
3. tokenizer 输出的 special token ID 需要进入模型和 checkpoint 元数据；
4. 训练数据缓存需要绑定 tokenizer hash 和原始数据 hash，避免 tokenizer 变化后错误复用 `.npy`；
5. `train_lm.py` 中的训练逻辑需要适配当前 `DenseTransformer` 的 forward 返回值和配置格式；
6. 原仓库的实现需要补充项目级 checkpoint metadata、实验日志和统一结果 schema；
7. 需要重新审查原仓库中的 `data.py` 边界条件、BPE 复杂度和训练脚本的可复现性，不能因为已有实现就跳过验证。

**历史方案记录**

此前曾创建人工构造 toy corpus 和 `toy-bpe/v0.1.0` artifact。按当前决策，这些本地产物已删除，不再作为项目数据或 artifact 保留；相关历史 commit 仍保留在 Git 历史中，但不作为当前工作树内容。

**本轮改动**

- 新增 `train_bpe_iterable`；
- `train_bpe_from_file` 改为 newline-aligned 分块读取；
- CLI 支持 `--chunk-size-bytes`、`--max-training-bytes` 和 `--source-kind`；
- metadata 记录实际训练范围；
- 增加 special token 边界和分块训练测试；
- artifact 协议见 `docs/protocols/tokenizer-artifact.md`；
- BPE + CLI 测试共 16 个通过。

**验证**

```text
BPE + tokenizer CLI tests: 7 passed
```

**剩余待讨论决策**

- 是否沿用 `<|endoftext|>`，以及是否增加 BOS/EOS/PAD；
- 第一版训练数据采用 TinyStories、已有本地语料，还是项目自建小语料；
- 训练闭环是否优先复用原仓库的训练脚本结构，还是重新写一个更小的项目脚本。

**已决策（2026-08-25）**

- 训练数据：第一版真实语言模型训练改用 Stanford CS336 OWT sample；
- 数据版本：`OWT-SAMPLE-v1`；
- train/validation：使用官方 `owt_train.txt` 和 `owt_valid.txt`；
- 原始数据目录：`data/raw/owt-sample/`，由 `.gitignore` 排除；
- toy 数据：不保留；
- OWT tokenizer：已生成本地 `owt-bpe/v0.2.0`，使用 train 前缀约 64 MiB、vocab size 8192；正式大规模 tokenizer 仍需后续扩展；
- 数据记录：`docs/data/owt-sample.md`；
- OWT token cache：已实现流式 UTF-8 行读取、newline-aligned 截断、uint16 输出和 source/tokenizer hash 绑定；
- 正式范围 OWT token cache：train 512 MiB 上限、validation 64 MiB 上限已生成；
- 正式 cache 实际范围：train 143,918,122 tokens / 536,870,901 bytes，validation 17,999,093 tokens / 67,105,041 bytes；
- Dense batch sampler：已实现连续 token stream 的随机窗口、确定性 epoch 遍历和显式 next-token targets；
- Dense training MVP：已接入动态 tokenizer vocab、validation、checkpoint/resume、generation、scheduler、AMP 和梯度累积；
- 正式结果 schema：已新增并接入 CLI 校验；
- 真实 OWT train/validation 各 1 MiB cache 试跑通过；
- cache 协议见 `docs/protocols/owt-token-cache.md`；
- batch 协议见 `docs/protocols/dense-batching.md`；
- training 协议见 `docs/protocols/dense-training.md`；
- 正式 cache 记录见 `docs/data/owt-sample.md`；

**原问题现状**

当前 Dense Transformer 已完成 forward、loss、KV Cache、增量解码和 smoke benchmark，但还没有完成原计划中的完整基线：

- OWT tokenizer artifact 已生成 `owt-bpe/v0.2.0`，但尚未接入 Dense 训练；
- BPE tokenizer 集成；
- 固定训练数据；
- train/validation split；
- 训练 loop；
- checkpoint 保存和加载；
- 训练 loss 和 validation loss；
- 基于训练模型的 generation 结果。

当前记录的 loss 来自随机初始化模型和随机 token，只能证明计算链路可运行，不能作为模型效果基线。

---

### P0-02 MoE Top-1 目前是 forward MVP，不是完整训练 MVP

**现状**

当前 MoE Top-1 已实现 router、Top-1 expert、capacity、dropped token、auxiliary loss、forward/backward 测试和基础 KV Cache，但还没有：

- 训练 loop；
- checkpoint；
- 真实数据训练；
- Dense/MoE 统一训练对比；
- active parameter 统计；
- capacity factor 实验；
- auxiliary loss 权重实验。

**风险**

当前 `MoE Top-1 MVP` 的命名可能让文档读者误以为已经完成可用于架构结论的 MoE 实验。

**待讨论决策**

README 状态是否拆成：

```text
[x] MoE Top-1 forward MVP
[ ] MoE Top-1 训练闭环
[ ] Dense/MoE Top-1 对比实验
```

---

### P0-03 Dense 与 MoE 的比较协议不公平

**现状**

当前 Dense 和 MoE 配置的 expert FFN hidden size 相同。4 个 experts 使 MoE 总参数显著增加，但每个 token 只激活一个 expert；同时 MoE 的统计和 dropped token 逻辑也会增加额外开销。

**风险**

直接比较 latency、loss 或参数量，会混合以下因素：

- 总参数量；
- 激活参数量；
- router 开销；
- dispatch 开销；
- token dropping；
- benchmark 统计开销。

**待讨论决策**

至少固定两套对比协议：

1. **相同 expert hidden size**：研究总参数增加和稀疏激活的效果；
2. **相同总 FFN 参数量**：研究在近似相同参数预算下 Dense/MoE 的差异。

每套协议都需要明确：

- 总参数量；
- active parameter count；
- 每 token 计算量或近似 FLOPs；
- token budget；
- batch 和 sequence length；
- optimizer 和训练步数。

---

### P0-04 MoE routing statistics 污染 latency benchmark

**现状**

当前 MoE forward 中会把 routing 统计同步到 CPU，例如调用 `.item()`、`.cpu().tolist()`。这些操作可能触发 GPU 同步。Dense benchmark 没有等价的统计开销。

**风险**

当前 Dense/MoE latency 不是同一测量口径，不能直接比较。

**待讨论决策**

增加统计开关，例如：

```python
output, aux_loss, stats = moe(x, collect_stats=False)
```

- 正式 latency benchmark：关闭统计和 CPU 拷贝；
- 专门 routing 分析：打开统计；
- 报告中记录统计开关状态。

---

### P0-05 MoE capacity 与 prefill/decode cache 语义未定义

**现状**

当前 capacity 按每次 forward 的 token 数计算。prefill 和 incremental decode 的 token 数不同，因此同一个模型在两个阶段可能得到不同的 capacity 和 dropped token 行为。

当前增量等价性测试使用了足够大的 capacity，并未覆盖正式配置 `capacity_factor=1.0` 下的严格行为。

**风险**

- prefill 和 decode 输出可能不一致；
- decode 期间 dropped token 比例可能异常；
- KV Cache 正确性结论不完整；
- capacity factor 的实验结果难以解释。

**待讨论决策**

需要选择并记录一种策略：

- 推理阶段关闭 dropping；
- 为 prefill/decode 使用不同 capacity 策略；
- 固定 batch 级 capacity；
- 接受并明确 prefill/decode 不严格等价；
- 为 decode 单独设计 routing/capacity 逻辑。

---

## P1：实验和评测协议问题

### P1-00 历史阶段审查缺少模型和 agent 元数据

**现状**

早期阶段审查文档没有统一记录：

```text
审查模型：minimax-cn/MiniMax-M3
审查 agent：reviewer
```

**处理策略**

历史审查文档不追溯改写；从下一次阶段审查开始，所有审查记录必须包含上述字段。若需要审计历史阶段，只能依据当时的 session 或 commit 记录，不把未验证信息补写为事实。

---

### P1-01 缺少统一 benchmark schema

Dense 和 MoE 当前输出字段不完全一致。后续 GQA、MLA、Top-2 如果继续各自设计 benchmark 输出，结果会难以汇总。

建议统一记录：

- experiment_id；
- model/architecture；
- code/config/dataset hash；
- seed；
- device、GPU、CUDA、PyTorch 版本；
- batch size、sequence length、输入/输出长度；
- warmup 和 measured steps；
- prefill latency；
- decode latency；
- throughput；
- peak memory；
- total parameter count；
- active parameter count；
- loss；
- routing statistics；
- benchmark 是否开启统计、compile 或 CUDA graph。

---

### P1-02 实验元数据和版本信息不完整

当前结果缺少或未统一保存：

- git commit；
- Python 版本；
- PyTorch/CUDA/driver 版本；
- GPU compute capability；
- tokenizer 和 revision；
- dataset version/hash；
- config hash；
- code version；
- 线程数和确定性设置。

缺少这些字段时，后续无法判断两次运行是否真正可比。

---

### P1-03 缺少多 seed 和统计区间

当前 smoke benchmark 使用单个 seed、少量 warmup 和少量 measured steps，只适合链路验证。

正式实验需要确定：

- seed 数量；
- latency 的 p50/p95 或均值/标准差；
- 训练结果的均值和标准差；
- CUDA 同步和 warmup 规则；
- 是否启用 `torch.compile`、AMP 或 CUDA Graph。

在这些规则固定前，不应把 smoke 数字写成正式性能结论。

---

### P1-04 数据版本 D0/D1/D2 定义不够严格

当前 D0/D1/D2 只有处理流程描述，还没有固定：

- 样本数量；
- 工具和任务类型分布；
- train/dev/test split；
- 数据 hash；
- 生成模型及版本；
- 过滤规则版本；
- 模板和参数泄漏检查；
- 质量通过率。

建议同时设计 IID split 和 compositional split，避免工具调用准确率主要来自模板记忆。

---

### P1-05 工具执行成功不等于任务完成

工具调用评测需要明确错误层级：

```text
parse_success
→ tool_name_correct
→ argument_schema_valid
→ argument_value_correct
→ execution_success
→ result_grounded
→ final_answer_correct
→ task_success
```

如果只记录工具执行成功率和任务完成率，无法定位模型失败发生在哪一步。

---

### P1-06 JSON Schema 校验不能替代语义校验

Schema 可以校验字段、类型和枚举，但不能完整校验：

- expected tool 是否存在于 tools；
- arguments 是否符合对应工具 schema；
- depends_on 是否引用已存在 call ID；
- 依赖是否成环；
- 调用顺序是否可执行；
- 最终答案是否使用真实工具结果。

后续需要区分：

```text
Schema validation
Semantic validation
Tool execution validation
Answer validation
```

---

### P1-07 训练框架范围过大

当前计划同时提到 TRL、LLaMA-Factory、Transformers、vLLM 和自研 PyTorch 模型，可能导致配置、模板、checkpoint 和 reward 接口重复。

建议第一阶段明确：

- Transformers：模型加载和基础推理；
- TRL：SFT/GRPO 主训练框架；
- LLaMA-Factory：暂不作为必选依赖；
- vLLM：标准开源模型部署 benchmark。

---

### P1-08 GRPO 依赖评测和执行器先稳定

GRPO 需要可靠的：

- 工具执行器；
- 确定性 evaluator；
- reward 定义；
- reward 离线验证；
- 失败分类；
- 防 reward hacking 检查。

建议顺序为：

```text
工具执行器
→ 确定性 evaluator
→ SFT
→ reward 离线验证
→ GRPO
```

---

### P1-09 vLLM 运行环境需要单独确认

当前开发环境是 Windows Conda。vLLM 是否能在原生 Windows 环境中稳定运行需要单独验证。

建议先实现 Transformers backend，并将 vLLM backend 的环境目标明确为 WSL、Linux 或 Docker，避免它成为当前开发阻塞条件。

---

## P2：范围和文档工程问题

### P2-01 计划目录和实际目录不同步

README 中已有 `data-pipeline/`、`evaluation-lab/`、`serving/` 等规划目录，但当前仓库尚未实现这些模块。

需要明确这些是 planned layout，而不是当前已存在的代码目录。后续可以：

- 在 README 中标注 planned；或
- 随模块启动时再创建目录。

---

### P2-02 README 协议示例与真实 Schema 不一致

README 中的模型输出和评测结果示例缺少真实 Schema 要求的部分字段，例如：

- `schema_version`；
- `experiment_id`；
- `timestamp`。

后续应让 README 直接引用真实示例，或同步更新示例，避免维护两套协议。

---

### P2-03 缺少依赖锁定文件

当前环境虽已在 `.venv` 中验证，但仓库缺少：

- `environment.yml`；
- `requirements.txt` 或 lock file；
- `pyproject.toml`。

特别是 RTX 5070 Ti `sm_120` 依赖特定 PyTorch/CUDA wheel，应记录安装来源和版本。

---

### P2-04 文档产物目录已经统一，但链接需要持续维护

计划、协议、环境说明和实验报告统一放在 `docs/`：

```text
docs/
├── plans/
├── protocols/
├── experiments/
└── environment.md
```

当前代码生成的 JSON 结果仍放在实验文档目录下，后续需要决定是否统一迁移到 `artifacts/` 或 `outputs/`，并同步 `.gitignore` 和 README 链接。

---

## 暂不处理的范围

以下项目计划本身暂不视为当前问题，除非后续实现暴露具体错误：

- Expert Parallel；
- 分布式训练；
- 完整工业级 MLA；
- 多模态数据；
- 复杂联网 Agent；
- 大规模数据采集；
- 复杂 LLM Judge；
- 自研模型完整 vLLM 适配。

## 记录规则

每次解决一个问题后，在对应条目下补充：

```text
状态：已解决 / 部分解决 / 延后
决策：...
改动：...
验证：...
结果：...
遗留风险：...
```
