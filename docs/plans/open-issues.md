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

项目总体路线图（已完成 / 进行中 / 下一阶段 / 暂缓阶段）见 [`roadmap.md`](./roadmap.md)；本文只跟踪问题与决策。

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

**状态更新（2026-08-26）**

```text
状态：部分解决
决策：最小训练闭环已就绪（BPE artifact + token cache + batch sampler + training loop + scheduler/AMP/梯度累积 + checkpoint/resume + generation + formal cache + 结果 schema）；正式 OWT 基线推迟到 roadmap N4，不关闭 P0-01。
改动：
  - architecture_lab/training/dense_training.py
  - architecture_lab/training/results.py
  - scripts/train_dense.py
  - schemas/dense_training_result.schema.json
  - configs/dense_training.example.yaml
  - configs/dense_training.owt-formal.example.yaml
  - docs/protocols/dense-training.md
  - docs/experiments/dense-training-mvp/README.md
验证：
  scripts/run_tests.py full → 50 tests + 5 schema examples passed
  1MiB smoke 与 formal cache 100-step smoke 都已运行并写入 result JSON
遗留风险：
  - 小模型 100-step smoke 仍不是正式基线；
  - checkpoint 使用 PyTorch pickle，后续需评估 weights_only/safetensors；
  - OWT tokenizer 仍为 owt-bpe/v0.2.0（8192 vocab），与正式规模 tokenizer 还有差距。
```

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

**状态更新（2026-08-26）**

N1 训练闭环已完成大阶段审查，审查结论为通过；正式阶段 commit 已创建。

```text
状态：已解决
决策：N1 训练闭环已完成，后续 N2/N3 边界保留在 roadmap 中。
改动：
  - architecture_lab/models/moe_transformer.py：collect_stats、capacity override、MoE generation、active parameter count
  - architecture_lab/training/moe_training.py：MoE train/evaluate/checkpoint/resume/generation
  - architecture_lab/training/moe_results.py：独立结果 builder/writer 与显式 expert capacity
  - scripts/train_moe.py
  - schemas/moe_training_result.schema.json
  - configs/moe_training.example.yaml
  - configs/moe_training.owt-formal.example.yaml
  - tests/test_moe_training.py
结果：
  - 1MiB 与 formal cache 均完成 100 optimizer-step smoke
  - 两个结果 JSON 均通过独立 MoE schema 校验
  - training.expert_capacity 与 generation prefill/decode expert capacity 均已记录
  - generation 记录 prefill capacity_factor=1.0、decode capacity_factor=2.0、collect_stats=false
  - total parameters=656,192；Top-1 active parameters=582,464
遗留风险：
  - formal benchmark、routing stats 分析和 Dense/MoE 公平对比属于 N2
  - 多 seed、统一 benchmark 元数据属于 N3
  - prefill/decode 接受不严格等价
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

**决策（2026-08-26）**

```text
状态：已解决（N2 A/B smoke 已完成）
决策：两套对比协议固定，执行顺序为 A → B。

协议 A：相同 total params
  对齐方式：MoE 通过调整 d_ff 使总参数量接近 Dense。
    Dense d_ff = D
    MoE n_experts=4，每个 expert d_ff = D/4
  对比指标：
    - loss 曲线
    - prefill latency
    - decode latency
    - throughput
    - peak memory
    - active_parameters
    - total_parameters
  额外记录：
    - aux_loss_weight
    - expert_capacity_factor
    - routing stats：expert load、dropped token ratio、load imbalance
    - stats 开闭状态：benchmark 阶段 collect_stats=False；routing stats
      由专门分析脚本在 collect_stats=True 时输出

协议 B：相同 active params
  对齐方式：Dense d_ff 与 MoE active d_ff 对齐。
    Dense d_ff = D_act
    MoE n_experts=4，每个 expert d_ff = D_act
  对比指标：与 A 一致，加 active FLOPs 近似。
  额外记录：与 A 一致。

隐含要求：
  - 两套协议下 Dense/MoE 使用同一 tokenizer 与同一 cache
  - 同一 optimizer、scheduler、AMP、accumulation、seed
  - 报告需同时给出 total/active 参数与 routing stats（避免重复 P0-04）
  - 两套协议均使用同一 tokenizer/cache、seed=42、batch=2、sequence=32、token budget=128
  - A/B 四个 Dense/MoE benchmark result JSON 与独立 routing stats JSON 已落盘并通过 schema 校验
  - latency benchmark 固定 collect_stats=false；routing stats 独立使用 collect_stats=true
  - formal benchmark、N3 元数据、多 seed 统计和严格性能结论仍不在 N2 范围

**修复更新（2026-08-26）**

```text
状态：已解决（auditor objection 修复后复核）
改动：N2 benchmark 现在实际加载并校验 train/validation token cache，记录两个 token-file hash；按 token_budget 执行 AdamW、scheduler、AMP/gradient accumulation 短训练，并在 validation cache 上计算 validation loss；新增端到端 shared-binding/capacity 测试。
验证：`scripts/run_tests.py full` → 69 tests + 5 Stage 0 examples passed；四个 benchmark artifacts 与 routing artifact 均通过对应 schema。
限制：该短训练 smoke 仍不代表正式模型质量、长训练曲线或多 seed 性能结论。
```

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

**决策（2026-08-26）**

```text
状态：已解决（N2 routing stats 已实现并完成独立 smoke）
决策：MoE routing statistics 采集开关机制固定。

接口：
  output, aux_loss, stats = moe(x, collect_stats=False)
  - collect_stats=True：保留原有行为，stats 含 expert load、
    dropped token ratio、load imbalance
  - collect_stats=False：跳过 .item()、.cpu()、同步；stats=None

默认：collect_stats=False，避免默认路径触发 GPU 同步。

训练 loop 调用处必须 collect_stats=False；只允许专门 routing
分析脚本调用 True。

result JSON 与 benchmark JSON 都必须记录 collect_stats 开关状态。

归属：N1 实现接口；N2 已提供 routing stats 分析脚本与报告输出。
```

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

**决策（2026-08-26）**

```text
状态：已解决（N2 benchmark 已显式记录两阶段 capacity）
决策：接受 prefill/decode 不严格等价，并在报告中显式记录。

capacity 策略：
  prefill：capacity_factor=1.0
  decode：capacity_factor=2.0

报告位置：MoE benchmark result JSON 的 routing_stats 下。
  routing_stats.prefill：{capacity_factor, dropped_token_ratio, expert_load, load_imbalance}
  routing_stats.decode：{capacity_factor, dropped_token_ratio, expert_load, load_imbalance}

调用路径：inference 接口提供 prefill/decode 两个调用点，分别传入不同
capacity_factor；不允许隐式使用一个 capacity_factor。

归属：N1 实现接口与默认 capacity_factor；N2 已提供 prefill/decode 各自 benchmark 输出与报告。

```

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

**决策（2026-08-26，更新于 N3 阶段完成）**

```text
状态：已解决
决策：归 roadmap N3。N3 commit 中以 metadata 块统一所有结果 JSON 的 experiment_id、seed、device/GPU/CUDA/PyTorch/Python 版本、config hash、dataset hash、tokenizer revision；N3 审查记录见 docs/plans/reviews/stage-n3-unified-metadata.md。
进展：2026-08-26 reviewer（calculet/gpt-5.6-terra）阶段审查通过 CAN_ENTER_N3 后，N3 已交付。
```

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

**决策（2026-08-26，更新于 N3 阶段完成）**

```text
状态：已解决
决策：归 roadmap N3。N3 commit 中 metadata 块已包含 git_commit、config_sha256、python_version、pytorch_version、cuda_version、gpu_name、gpu_compute_capability、tokenizer_revision、dataset_hash、seed。
进展：2026-08-26 N3 已交付，审查记录：docs/plans/reviews/stage-n3-unified-metadata.md。
```

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

**决策（2026-08-26，更新于 P1-03 阶段完成）**

```text
状态：已解决（2026-08-26）
决策：P1-03 协议已落 docs/protocols/p1-03-multi-seed.md；N5+N6 已做 3-seed 实跑（42/123/7，18 个训练）并验证：
  - 规模 sweep 结论多 seed 稳健（val_min mean 7.26→7.03）；
  - N6 dropout=0.1 的单 seed 优势是噪声（多 seed 下 dropout 影响退化）；
  - large std 最大（0.093），单 seed 低估其真实水平。
默认：3 seeds={42,123,7}；mean/std；N≥5 时可报告 95% CI；torch.compile/CUDA Graph 默认关；AMP bf16 默认开。
进展：2026-08-26 协议 + 3-seed 实跑完成；2026-08-27 N11 长训练 + MoE 曲线已按本协议扩展为 3-seed（见 `docs/experiments/n11-long-multi-seed/README.md`、`docs/experiments/moe-multi-seed/README.md`）；N1-N10 单 seed 文档不追溯改写，只在关键结论标注"单 seed"。
```

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

**决策（2026-08-26，更新于 P1-04 阶段完成）**

```text
状态：已解决（2026-08-26，auditor 复核后）
决策：D0/D1/D2 定义已落 docs/protocols/p1-04-data-version-d0.md；D0 已实现：
  - scripts/build_d0_manifest.py（生成 + 校验 MANIFEST.json）；
  - examples/tool_calling/MANIFEST.json 已生成（3 样例 sha256 + 聚合 hash + metadata）；
  - 3 个样例 metadata.source 统一为 synthetic；
  - tests/test_d0_manifest.py（4 tests）+ tests/test_artifact_provenance.py（2 tests）；
  - D0 覆盖 no_tool / single_tool / multi_tool。
遗留：D1/D2 触发条件 = 进入 P4 SFT/GRPO 前；compositional split 设计留 D2。
```

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

**决策（2026-08-27，P1-05 阶段完成）**

```text
状态：已解决（v3，八级）
决策：八级失败分类已落 docs/protocols/p1-05-failure-classification.md：
  parse_success → schema_valid → tool_name_correct → argument_value_correct → call_plan_matches → execution_success → result_grounded → final_answer_correct；
  scripts/classify_tool_failure.py 实现分类器（--sample / --transcript / --output）；
  D1 数据集已生成（126 样例，6 种 task_type 全覆盖，train/dev/test 100/13/13，所有 single/multi-tool 样例含 expected_result）；
  tests/test_d1_failure.py 44 单测（含集成验证：MockExecutor 执行 expected_tool_calls、expected_result 一致性、depends_on 可达性、D1SemanticIntegrationTests 6 测试、MalformedTranscriptRegressionTests 8 个反向断言、call_id 不一致位置配对回归 2 个、八级扩展 4 个、未知工具名+非字符串 name/arguments 鲁棒性 3 个、混合 depends_on+不可哈希 call_id 回归 2 个、single-call call_id 不匹配 1 个）。
遗留：模板版 D1 保留为确定性基线；D1.1（MiniMax-M3 真实 LLM 生成，126 样例）已交付（2026-08-27）；八级在 canonical D1 上的失败分布统计需真实模型 transcript（P2 评测）。
```

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

**状态更新（2026-08-28，SFT 阶段审查）**

```text
状态：延后，不阻塞当前阶段
决策：P5-03 仅接入公开 instruction-tuned 模型的 vLLM 推理 backend；不实施暂缓范围内的自研模型完整 vLLM 适配。
改动：roadmap.md 与 stage-sft-tool-mvp.md 明确 P5-03 的模型范围和 WSL/Linux/Docker 环境目标。
验证：SFT 阶段审查确认该范围与“自研模型完整 vLLM 适配”暂缓条目不冲突。
结果：Windows 原生 vLLM 可用性不作为 P5-03 的前置假设；先完成 Transformers backend。
遗留风险：仍需在 P5-03 实际确认 WSL/Linux/Docker 的 vLLM 版本、CUDA 和显存兼容性。
```

---

### P1-10 自研小模型未达到可靠工具调用能力

**发现（2026-08-28，SFT 阶段）**

Dense 2M/5M/12M 与 MoE 4-expert 自研模型均完成了训练、生成和 P1-05 八级分类评测，但在 D1 dev 上除一次 1/13 的部分通过外，未形成可靠工具调用能力。该结果是本阶段的诚实负结果，不应表述为 SFT 已提升工具调用质量。

**状态：已记录，阻塞自研模型直接进入 GRPO；不阻塞公开 instruction-tuned 模型后端评测。**

后续决策：

- P5 先使用公开 instruction-tuned 模型作为标准起点；
- 将 Transformers/vLLM 后端与 P1-05 分类器接通后再评估模型能力；
- GRPO 需等待确定性 evaluator、reward 离线校验和更可靠的数据 split。

---

## P2：范围和文档工程问题


### P2-01 计划目录和实际目录不同步

README 中已有 `data-pipeline/`、`evaluation-lab/`、`serving/` 等规划目录，但当前仓库尚未实现这些模块。

需要明确这些是 planned layout，而不是当前已存在的代码目录。后续可以：

- 在 README 中标注 planned；或
- 随模块启动时再创建目录。

**决策（2026-08-26）**

```text
状态：部分解决
决策：保持现状，不修改 README。roadmap.md 已经独立列出计划范围。
```

---

### P2-02 README 协议示例与真实 Schema 不一致

README 中的模型输出和评测结果示例缺少真实 Schema 要求的部分字段，例如：

- `schema_version`；
- `experiment_id`；
- `timestamp`。

后续应让 README 直接引用真实示例，或同步更新示例，避免维护两套协议。

**决策（2026-08-26）**

```text
状态：部分解决
决策：
  - dense_training_result.schema.json 已对齐并接入 CLI 校验。
  - README 中 model output、evaluation result、tool calling 示例
    改为指向 examples/*.json 与 schemas/*.json 链接。
  - README 示例同步 schema_version、experiment_id、timestamp 等必填字段，
    避免两套协议并存。
执行时机：roadmap 当前阶段完成后随 README 一起更新；本轮不修改实现。
```

---

### P2-03 缺少依赖锁定文件

当前环境虽已在 `.venv` 中验证，但仓库缺少：

- `environment.yml`；
- `requirements.txt` 或 lock file；
- `pyproject.toml`。

特别是 RTX 5070 Ti `sm_120` 依赖特定 PyTorch/CUDA wheel，应记录安装来源和版本。

**决策（2026-08-26）**

```text
状态：部分解决
决策：
  - 现在不在仓库新增 pyproject.toml / requirements.txt / environment.yml。
  - 现在在 docs/environment.md 中记录 PyTorch / CUDA / cuDNN / jsonschema / PyYAML
    版本以及 RTX 5070 Ti sm_120 架构信息。仓库优先 trust docs/environment.md。
  - pyproject.toml / requirements.txt / environment.yml 推迟到 roadmap
    N3 之前的工程改进阶段同步考虑。
```

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

### P2-05 SFT held-out 评测 split 仍然偏小

D1 dev 当前只有 13 个样例。D1.1 train 的 50-sample 聚合可以用于探索性诊断，但不是独立 held-out split，不能作为正式泛化质量结论。

- **P3 交付摘要**：D2 多轮对话数据集共 600 样本，train/dev/test = 420/90/90 IID split；P3 专项测试当前为 33 个，使用独立 D2 schema、真实 `MockExecutor.execute_sequence()`、依赖图语义校验和 canonical semantic uniqueness 校验。

P3 已解决本条目的独立 IID held-out split 要求：D2 dev 90 样本可用于 held-out reward pipeline 验证；P3 已完成 5 ckpt × D2 dev 90 = 450 个真实推理 reward_signal，最终多样化数据上的结果见 `docs/experiments/p2-evaluator/README.md` 第 6 节。

后续 P3/P5 需要：

- 在 D2 dev 上用 P5-02 公开 instruction-tuned 模型跑可比较的真实模型推理 reward 评测；
- 报告 P1-05 八级分布、parse success 和完整任务成功率；
- 保留多 seed mean ± 总体标准差，但不把训练集抽样结果冒充泛化指标。

### P2-06 P2 阶段性交付已完成（2026-08-28）

**状态：已解决。**

决策：实现 P2 确定性 evaluator，作为 P4 GRPO 的 reward 前置。

交付：

- `schemas/reward_signal.schema.json`（v1.0）：reward_type enum = {parse_success, argument_correct, final_answer_correct, execution_correct}；jsonschema Draft202012 校验严格；65 reward_signal 全部通过。
- `scripts/reward_offline.py`：`compute_reward()` 复用 P1-05 classify，不重写分类；`aggregate()` 用 pop std（÷N）符合 P1-03；CLI 支持 --samples-dir / --transcripts / --output / --checkpoint / --transcript-kind。
- `tests/test_reward_offline.py`：**37** 单测 = 8 层各 ≥ 3 例（共 25；parse_success 4 例、其他 7 层各 3 例） + classifier 一致性（4 例） + reward_type 映射（7 例：5 主路径 + 2 鲁棒性 / 矛盾输入） + CLI 聚合（1 例）。
- `tests/test_stage0_schemas.py`：新增 5 个 reward schema 测试（full pass / parse fail / 缺 reward_type / 未知 reward_type / reward_binary 越界）。
- `examples/reward_signals/*.json`：2 个 reward_signal example（execution_correct + parse_success）。
- `scripts/validate_stage0.py`：注册 reward_signal schema + 2 examples 到 stage0 校验。
- `docs/protocols/p2-evaluator.md`：reward 定义、与 P1-05 关系（含 task_success 别名排除）、与 P4 GRPO 衔接的 5 项前置、已知边界（不含 trajectory shaping、D1 dev 非 held-out split）。
- `docs/experiments/p2-evaluator/README.md`：5 ckpt × D1 dev 13 样本 = 65 reward_signal 评测表 + reward_type 主导通道分布。

验证：P2 阶段原始结果保持 Ran **227** tests OK（skipped=2）；P3 新增后本仓库 `scripts/run_tests.py full` 的最终结果见 P3-01 条目。

P3/P4 后续动作：

- P3 D2 多轮对话数据集已交付，详见 `### P3-01` 决策记录 + `docs/protocols/d2-multi-turn.md` + `docs/plans/reviews/stage-p3-d2-multi-turn.md`；
- P4 GRPO：等 P5-02 就位后实现 advantage 计算 + policy 更新；
- P3/P4 的 P2 前置已全部满足（reward_signal schema v1.0 + offline reward CLI + 与 P1-05 classifier 一致性）。

### P3-01 D2 多轮对话数据集 + IID held-out split（2026-08-28）

**状态：已解决。**

决策：实现 D2 多轮数据版本，作为 P4 GRPO 的数据集基础；为 P2-05 显式未解决的“独立 IID held-out split”提供首个有统计意义的 reward split。

交付：

- 独立 schema `schemas/d2_multi_turn_sample.schema.json`，task_type enum = tool_not_available / tool_error_response / insufficient_result_search / req_change_city / multi_tool_sequential / error_recovery；向后兼容 D1 / D1.1 样本（D1 继续使用通用 schema）。
- `scripts/generate_d2_dataset.py`：
  - 默认生成 600 样本（6 类各 100），train/dev/test = 420/90/90 IID split；
  - 确定性 `--seed`（默认 2026）、确定性时间戳、id/call_id 命名空间前缀；
  - 使用真实 `MockExecutor.execute_sequence()`，并在写盘前验证 depends_on 的存在性、严格前序、无环和每一步 `expected_result`；
  - 使用 `canonical_content_signature()` + 生成期全局去重，去除 bookkeeping 后 600 个语义签名仍全部唯一；
  - 六类 builder 使用确定性 variant index 与语义场景组合，避免跨 split 重复模板；
  - per-split `aggregate_sha256` + per-file sha256；
  - MANIFEST-{train,dev,test}.json：count + sha256 + task_type 分布 + aggregate_sha256。
- `tests/test_d2_dataset.py`：**33** 单测分 7 组：
  - `D2DatasetSchemaTests` (3)：独立 D2 schema 合法、6 类 task_type 覆盖、每 split 含 6 类；
  - `D2DependencyGraphTests` (5)：depends_on 存在性、严格前序、无环、负例拒绝、乱序执行语义；
  - `D2MultiTurnStructureTests` (5)：multi-turn messages 结构、tool_call_id 显式存在、expected/transcript call_id 一致、最终命名空间；
  - `D2MockExecutorReplayTests` (3)：真实 `MockExecutor.execute_sequence()` replay、乱序拓扑执行、失败依赖阻断；
  - `D2SplitDisjointnessTests` (8)：train/dev/test id/path/canonical semantic content 不重叠、每类 100 个唯一语义实例、与 D1/D1.1 train id 互斥、MANIFEST count / aggregate / per-file sha 一致；
  - `D2TimestampContractTests` (6)：`1785000000 + seed + index` 精确 UTC/Z 格式。
- `docs/protocols/d2-multi-turn.md`：D2 schema 与 D1 / D1.1 差异 + 6 类任务定义 + call_id 依赖链 + held-out split 互斥保证 + 与 P2 reward offline + P4 GRPO + P5-02 衔接。
- `docs/experiments/p2-evaluator/README.md`：引用 D2 dev 90 样本作为首个有统计意义的 reward benchmark；`artifacts/d2-mock-reward-d2dev.json` 验证 reward_offline 能消费 D2 dev 多轮 transcript（90/90 reward_binary=1.0）。
- `docs/plans/roadmap.md`：P3 加入已完成阶段表；当前阶段仍 P5。
- `docs/plans/reviews/stage-p3-d2-multi-turn.md`：阶段审查记录（reviewer r1/r2/r3/r4/r5 PI_PROVIDER=minimax-cn / PI_MODEL=MiniMax-M3 / BLOCKERS: none）；round 3/4 的历史问题已由 round 9 进一步修复。

验证：`scripts/run_tests.py full` → Ran **255** tests OK；`scripts/validate_stage0.py --examples` → 9/9 PASS；600 D2 样本使用独立 D2 schema + 真实 `MockExecutor.execute_sequence()` 验证零错误；D2 dev mock transcript pipeline 90/90 reward_binary=1.0、reward_layered=1.0、reward_type=execution_correct；**5 ckpt × D2 dev 90 = 450 真实推理 reward_signal 已按最终多样化数据重新产出**（large-v1 reward_layered=0.0162，d256-20k reward_layered=0.0042，其余为 0.0000；详见 README 第 6 节）；六类 builder 均通过 canonical semantic uniqueness 校验（600/600 唯一、每类 100/100 唯一、三 split overlap=0）；reviewer r1/r2/r3 通过。

审计 round 9 修复：

- 六类 builder 使用确定性 `variant_index` + 多维语义组合；默认数据 600/600 canonical semantic signatures 唯一，每类 100/100 唯一，train/dev/test 三组 semantic signature 交集均为 0；
- `canonical_content_signature()` 去除 ID、call_id、tool_call_id、depends_on、created_at、split 等 bookkeeping 字段，保留 task_type、tools、消息内容、工具参数/结果和 expected_answer；生成器写盘前拒绝重复签名；
- `tests/test_d2_dataset.py` 新增跨 split canonical overlap、每类 100 个唯一实例、以及“仅修改 bookkeeping 仍必须拒绝”的反向断言；
- 重新生成最终 `datasets/tool-calling-d2/`，并重新运行 5 ckpt × D2 dev reward artifact。

  - 跑完 5 ckpt × D2 dev 推理 + reward_offline（`artifacts/sft-{large-v1,large-night,d256-5k-seed42,d256-20k-seed42,moe-v1}-eval-d2dev-reward.json`，450 signals 全 schema 合法）；
  - 最终结果（**round 9 历史数字** — round 10 IID 后被取代）：large-v1 `reward_layered=0.0162`（86 parse_success / 3 argument_value_correct / 1 tool_name_correct），d256-20k `reward_layered=0.0042`（89 parse_success / 1 tool_name_correct），其余 3 个 checkpoint `reward_layered=0.0000`（90 parse_success）；5 个 checkpoint 均 `reward_binary=0.0`；
  - `schemas/reward_signal.schema.json` task_type enum 扩展 D2 六类多轮；
  - `scripts/eval_sft_tool.py` 新增 `--prompt-mode multi_turn`（多轮历史序列化提示）；
  - `tool_not_available` 与其余 task builder 通过 600/600 canonical semantic uniqueness 校验；
  - `docs/experiments/p2-evaluator/README.md` 用真实 5 ckpt × D2 dev 结果替换 mock 主表（mock 降为管线兼容性验证）。

### 审计 round 10 修复

- **根因**：`assign_split_ids()` 用 contiguous slicing（`samples[:train_n]`），dev/test 拿到每个 task_type 的 variant 70-99 连续切片，与 train 的 0-69 不交叠。train/dev/test 在 14 vs 3 vs 3 个 semantic domains 上分布完全不一致，并非 IID。`validate_semantics()` 只检查 expected ⊆ transcript，未验证跨 message ID 双向引用、name/arguments、ordering。
- **修复**：`build_samples()` 按 build_pos 显式计算 variant（与样本 id 解耦）；六个 builder 接收 `variant: int` 参数；`assign_split_ids()` 改为按 task_type 分组的 seeded shuffle（`random.Random(seed).shuffle()` per-task-type），per-task 严格 70/15/15。`_req_change_city` 改用 16 城市 120 个有序对组合取代 `rng.sample(cities, 2)`。新增 `transcript_well_formedness_errors()` 验证跨 message ID 双向引用、order、name/arguments 匹配，并接入 `_validate_samples()`。
- **测试**：`D2SplitDisjointnessTests` 新增 4 个 IID / canonical uniqueness / per-task disjoint 测试；新增 `D2TranscriptWellFormednessTests` 5 例（含 orphan tool + argument mismatch 反向断言）。`scripts/run_tests.py full` → Ran **264** tests OK；D2 专项 42 tests OK。
- **数据重生成 + 5 ckpt × D2 dev reward 重跑**：清理磁盘残留旧 3-digit 文件后重生成；5 ckpt 重新跑 multi_turn 推理；reward_offline 产出 450 signals schema 合法。最终：large-v1 `reward_layered=0.0099`（87 parse_success + 3 argument_correct），其余 4 个 ckpt `0.0000`；全部 `reward_binary=0.0`。

遗留：

- ~~自研 5 ckpt（单轮训练）尚未在 D2 dev 上跑实际推理 reward 评测——需 P5-02 公开模型 + Transformers backend 后才能验证。~~（已于 round 7 完成：5 ckpt × D2 dev 90 = 450 reward_signal；最终多样化数据上的结果见 `docs/experiments/p2-evaluator/README.md` 第 6 节。）
- D2 数据规模 600（420 train / 90 dev / 90 test）属 MVP；正式 GRPO rollout 池需 D2 扩样到 5000+。

### 审计 round 11 修复

- **根因**：(a) `docs/protocols/d2-multi-turn.md` §5.1 描述的 timestamp `index` 是生成器全局序列位置（train 1..420、dev 421..510），与代码的 split-local index（每 split 独立 1..N）不一致，`d2-train-0001` / `d2-dev-0001` / `d2-test-0001` 三个样本 timestamp 相同；(b) `transcript_well_formedness_errors()` 只比 ID 列表顺序，不验证 message 实际位置（tool message 能否出现在 final answer 之后 / assistant tool_call 能否出现在 final answer 之后），且原有测试缺反向断言；(c) D2 vs D1/D1.1 互斥验证仅在 ID 文件名层，未在 canonical semantic signature 层。
- **修复**：(a) 把 §5.1 改为 split-local 1-based（train 1..420、dev 1..90、test 1..90），与 `sample.id` 后缀对齐，并补充与 round 8 共识的一致性说明；所有 round 9 数字（0.0162 / 0.0042）一律标注为"round 9 历史数字 — round 10 IID 后被取代"。(b) `transcript_well_formedness_errors()` 改为消息位置状态机：逆向定位 final answer 位置，正向扫描维护 pending call_id 队列，tool message 必须与队首匹配后弹出，final answer 之后任何 tool message 或 assistant tool_call 都报错。(c) 新增 `d1_canonical_signature()` 投影函数和 `test_d2_canonical_content_is_disjoint_from_d1_d1llm_train()` 验证 D2 train/dev/test 三 split 与 D1 train / D1.1 train 在 canonical 层交集为空。
- **测试**：`D2TranscriptWellFormednessTests` 新增 2 个反向断言（tool-after-final-answer、asst-tool-call-after-final-answer），`D2SplitDisjointnessTests` 新增 1 个跨数据集 canonical 互斥测试。`scripts/run_tests.py full` → Ran **267** tests OK（264 + 3 round 11）；stage0 9/9；D2 专项 45 tests OK。

### 审计 round 12 修复

- **根因**：(a) round 11 引入的 `d1_canonical_signature()` 与 D2 的 `canonical_content_signature()` 投影形状不兼容——D2 含 `messages` + `metadata` + `tools` 全量，D1 含 `user_turns` + `tool_names` + 部分字段，两个 JSON 字符串的 set intersection 恒为空，无论数据是否真有重叠，跨数据集互斥验证退化为 vacuous truth；(b) round 11 三个 reverse-assertion / argument-mismatch 测试全部用 `self.train[0]`（`d2-train-0001` = `tool_not_available`、无 `expected_tool_calls`），全部 `skip`，但 complete_goal claim 写"反向断言被捕获"，证据虚假。
- **修复**：(a) 在 `scripts/generate_d2_dataset.py` 新增 `cross_dataset_signature(sample)`——8 字段投影（task_type、schema_version、user_turns、assistant_turns、tool_turns、tool_names、expected_tool_calls、expected_answer）跨 D1 / D1.1 / D2 形状完全一致。`canonical_content_signature()` 保持原 D2 全量投影。测试改用 `cross_dataset_signature`，并新增 `test_cross_dataset_signature_is_comparable_across_d1_d1llm_d2`（三数据集 keys 集合相等）+ `test_cross_dataset_signature_detects_real_overlap`（克隆样本投影碰撞，正向控制）。(b) `D2TranscriptWellFormednessTests` 新增 `with_calls` class attr 和 `train_with_calls()` helper（取第一个有 expected_tool_calls 的样本）；三个测试改用 helper，从 skipped → 实际执行并验证。
- **测试**：`scripts/run_tests.py full` → Ran **269** tests OK（skipped=0）；stage0 9/9；D2 专项 47 tests OK（skipped=0）；D2 × D1 / D2 × D1.1 cross_dataset_signature 交集 = ∅；两个 reverse-assertion 测试与 argument-mismatch 测试均 `... ok`（实际执行，非 skip）。

### 审计 round 13 修复

- **根因**：`_insufficient_result_search` 生成器把"中间响应"作为 `expected_answer` 传给 `_sample()`，但 transcript 后续又追加 user turn + 一个 terminal assistant acknowledgement——这两个值不一致，违反 protocol 中 `expected_answer` "closes the entire transcript" 约定。100/100 `insufficient_result_search` 样本受影响（train 70/70、dev 15/15、test 15/15）。现有测试只校验 transcript 结构与 schema，未校验 `expected_answer` 与 final assistant content 一致。
- **修复**：(a) 把 `_insufficient_result_search` 的 final assistant content 抽出为 `final_answer` 变量，同时作为 transcript 最后一条 assistant 消息 content 与 `expected_answer` 参数；(b) `transcript_well_formedness_errors()` 增加 invariant (5)：逆向定位 final assistant content，验证 `sample.expected_answer == final_answer_content`，不等则报错；(c) 新增 `D2ExpectedAnswerContractTests`：`test_expected_answer_equals_final_assistant_content` 遍历 600 样本验证 invariant；`test_transcript_well_formedness_flags_expected_answer_mismatch` 反向断言（修改 expected_answer 后 validator 必须报错）。
- **测试**：`scripts/run_tests.py full` → Ran **271** tests OK (skipped=0)；stage0 9/9；D2 专项 49 tests OK (skipped=0)；600/600 `expected_answer == final assistant content`；重生成数据集 + 5 ckpt × D2 dev reward 重跑（450 signals schema 合法，large-v1 0.0099、其余 4 ckpt 0、全部 binary=0，数字与 round 12 一致）。

---

## P5：公开 instruction-tuned 模型推理后端接入

### P5-02 Transformers backend + 5 个公开 instruction-tuned 模型 × D2 dev reward 评测（2026-08-28）

状态：已交付（2026-08-28 下午修正 target-answer 泄漏 bug 后重新验证；待 detached auditor + minimax-M3 subagent reviewer 联合复审）。
决策：实现 `scripts/eval_transformers.py` Transformers 推理后端，在 D2 dev 90 样本上对 5 个公开 instruction-tuned 模型（SmolLM2-360M/1.7B-Instruct + Qwen2.5-0.5B/1.5B/3B-Instruct）跑真实推理 + reward_offline；修正后 5 模型 reward_binary 全部 = 0（诚实负结果），reward_layered 在 0.33–0.42 区间，仍显著高于自研 5 ckpt 的 0.0。

改动（含 2026-08-28 下午修正）：
- `scripts/eval_transformers.py`（9066 → ~11300 bytes）：AutoModelForCausalLM + AutoTokenizer + 官方 chat_template + greedy + bf16；保留 `extract_tool_calls` 与 `classify_tool_failure` 复用，输出与 `eval_sft_tool.py` 完全一致的 `{summary, rows}` schema 以便 `reward_offline.py` 直接消费；CPU 强制 fp32；CUDA bf16；output 增加 `generated_preview` + 完整 `generated`；`first_failure_distribution` / `parse_success_count` / `parse_success_rate` / 5 个 backend 元数据 (backend/revision/device/dtype/transformers_version/torch_version)；**新增 `_strip_terminal_assistant()` 在 `_apply_chat_template()` 中去除 D2 样本的 gold terminal assistant content message 避免 target-answer 泄漏**。
- `tests/test_transformers_backend.py` 25 个单测（含 5 个 `GoldAnswerLeakageTests` 反向断言）：argparse / 设备解析 / dtype / chat template fallback / OOM fallback / greedy generation (do_sample=False, num_beams=1, pad_token fallback) / `extract_tool_calls None → []` 规范化 / artifact schema 兼容性 / `reward_signal.schema.json` enum 一致 / script end-to-end with mocks / **5 个反向测试证明 gold `expected_answer` 不会出现在 prompt 中、上下文 tool_calls 与 tool 结果消息会保留**。
- `scripts/run_tests.py` `COMMON_TESTS` + `MODULES["training"]` 新增 `test_transformers_backend.py`；运行总数 245 → 270 fast（含全部子模块）。
- `docs/protocols/transformers-backend.md`（NEW）：CLI / 输入契约 / chat template 优先级 / output schema 严格定义 / reward_offline 衔接 / GPU 显存 / 已知边界（包含 target-answer 泄漏修复段） / 5 模型修正后结果表与复现命令。
- `docs/experiments/p2-evaluator/README.md` §7（NEW + 修正）：5 模型矩阵 + 5×90=450 reward_signal 修正后结果表 + 与自研 5 ckpt 横向对比 + 解读（含原版 vs 修正后诚实负结果对比）。
- `docs/plans/roadmap.md` P5-02 行更新：含 25 单测、5 模型、450 signals、全部 reward_binary=0、stage review planned。
- `docs/plans/open-issues.md` P5-02 条目含修复记录。
- `.gitignore` 增 `artifacts/huggingface/` + `artifacts/**/*.json`（避免权重 + eval JSON 入库）。
- `artifacts/{qwen2.5,huggingfacetb-smollm2}-*-eval-d2dev.json` + `-reward.json`：5 × 2 = 10 个 gitignored 产物（90 signals × 5 = 450 reward_signal）。
- `HF_ENDPOINT=https://hf-mirror.com` 镜像下载 5 个模型（~14 GB 总 cache）。

修正后验证（2026-08-28 下午）：
- `scripts/run_tests.py fast` → Ran 270 tests OK（25 个 P5-02 含 5 个 GoldAnswerLeakageTests）。
- `scripts/validate_stage0.py --examples` → 9/9 PASS（reward schema 仍合法）。
- 5 个 `eval-d2dev-reward.json` 全部 schema 合法（450 signals）。
- 修正后 5 模型 aggregate：SmolLM2-360M binary=0/layered=0.4236（从原 0.0111 修正为 0；原伪信号因 target-answer 泄漏产生）；Qwen2.5-0.5B binary=0/layered=0.3634；Qwen2.5-1.5B binary=0/layered=0.3690；Qwen2.5-3B binary=0/layered=0.3333；SmolLM2-1.7B binary=0/layered=0.4236。
- 修正后 5 模型 `no_failure` 全部 = 0（SmolLM2-360M 修复前为 1）。所有模型的 final_answer 都是 prompt 中**不存在**的生成文本，证明 prompt 不含 gold 答案。
- 横向对比：自研 5 ckpt reward_layered 全部 0；公开模型 reward_layered ≥ 0.33。

遗留：
- **reviewer dispatch（minimax-M3）**：将在下一轮 audit 时由 detached auditor 联动 minimax-M3 subagent reviewer 联合复审；持续证据入 `docs/plans/reviews/stage-p5-02-transformers-backend.md`。
- P5-03 vLLM backend 未启动（硬件 / 环境需求超出当前阶段）；待 P5-02 audit 通过后启动。
- D2 数据集扩样到 5000+ **已完成**（见下方 `### P3 D2 扩样到 5000（2026-08-28，round 14）` 段，HEAD `c6eac20`）：5000 samples（4×833 + 2×834）、train 3500 / dev 750 / test 750 严格命中、6 类 ≥833 unique canonical signatures、cross-split canonical disjoint、D2-vs-D1/D1.1 cross_dataset disjoint、datasets/ 已加入 `.gitignore` 并 `git rm --cached` 隔离。
- 全部 5 模型 reward_binary=0 为诚实负结果：D2 expected_answer 与公开模型生成的 final_answer 字面不一致；如需严格一致，可加后处理归一化或引入轻量 evaluator prompt。

---

### P3 D2 扩样到 5000（2026-08-28，round 14）

状态：已交付（round 14 auditor 复查后；待 detached auditor round 14 终审）。
教训：round 13 完成时使用 5004 样本（4/6 类 834 + 2/6 类 832），split 实际为 3498/750/756；detached auditor round 14 否决该交付，要求严格命中 5000/3500/750/750 契约。

修复：
- `scripts/generate_d2_dataset.py`：
  - 新增 `_plan_per_class_counts(total)` helper：5000 / 6 = 833 remainder 2 → 返回 `(834, 834, 833, 833, 833, 833)`。
  - `build_samples()` 改用 **per-class running counter**（`intra_class_index`）作为 `variant`，代替旧公式 `build_pos // len(TASK_TYPES)`。旧公式在总数变化（如 600 → 5000）时会重复返回 `variant=0`，打破 canonical uniqueness。
  - `assign_split_ids()` 用 `round(per_class * 0.15)` 计算 dev/test，使 833/834 类都得到 dev=test=125；train = per_class - dev - test。
  - 其他维度扩展同 round 13。
- `tests/test_d2_dataset.py`：
  - 新增 3500/750/750 split 显式断言（manifest `count` 字段）
  - per-task ≥833 unique canonical signatures 显式断言
  - per-class 总数 833/834 各自映射到 `{833: 583 train}, {834: 584 train}` 期望表
- `docs/data/d2-expansion.md` round 14 修订（重写：5000 样本 + 3500/750/750 split）
- `docs/plans/reviews/stage-p3-d2-expansion.md` round 14 修订（重写）
- `git rm -r --cached datasets/tool-calling-d2` 重新同步（5000 个样本与 3 个 MANIFEST）

验证（round 14）：
- 仿真校验：6/6 类都产出 ≥833 unique canonical signatures（4 类 833/833、2 类 834/834）
- 生成 5000 samples 耗时 34s；train 3500 / dev 750 / test 750 严格命中
- canonical total unique: 5000/5000；train∩dev=0、train∩test=0、dev∩test=0
- D2-vs-D1/D1.1 cross_dataset_signature：3 个 split 均 0 重叠
- `scripts/run_tests.py full`：
 - 本机环境：Ran 296 tests OK（无跳过）
 - isolated auditor 环境：可能报告 `OK (skipped=2)`；2 个跳过均为 `tests/test_n2_benchmark.py:93, 108` 的 `@unittest.skipUnless(_has_working_cuda(), ...)`，是环境驱动跳过（isolated auditor 环境不一定有可用的 CUDA kernel image），不是代码缺陷。其余 294 个测试全 PASS。
- `scripts/validate_stage0.py --examples` → 9/9 PASS

遗留：
- per-class 总数不均（4 类 833、2 类 834）：这是 5000 / 6 不整除的必然结果，已通过 _plan_per_class_counts helper 自动化。
- round 14 status (2026-08-28T16:58):
 - minimax-M3 subagent reviewer 已派发 + 通过：evidence in `.pi-glla/scratch/stage-p3-d2-expansion-r14-review.txt`；11/11 审查项 PASS。
 - detached auditor round 14 状态：初次 audit report 在 16:58:01 返回两条 fixes required（同步不一致 + skipped 计数）。
 - 同步修复：本文档已更新以反映 reviewer PASS 已完成（不是“未启动”）；仅 detached auditor 进行中。
 - `scripts/run_tests.py full` 在 isolated auditor 环境可能报告 `OK (skipped=2)`：两个跳过均为 `tests/test_n2_benchmark.py:93, 108` 的 `@unittest.skipUnless(_has_working_cuda(), ...)`，是环境驱动跳过（isolated auditor 环境不一定有 CUDA），不是代码缺陷。

---

### P3 D2 扩样到 5004（2026-08-28，round 13，已否决）

状态：已被 detached auditor round 14 否决，原因是不命中 5000/3500/750/750 契约（实际 5004/3498/750/756）。保留为“round 13”记录以备查。

改动：
- `scripts/generate_d2_dataset.py`：
  - `_multi_tool_sequential`：expressions 20→50、topics 10→20，严格嵌套 50×20×4×6×4=96000 组合上限
  - `_tool_error_response`：_TRANSLATE_TEXTS 20→50、_TRANSLATE_LANGS 5→10、_TRANSLATE_LANG_NAMES 补齐；4×4 嵌套
  - `_insufficient_result_search`：queries 25→40；严格嵌套 40×2×6×8×4×4=61440 组合上限
  - `_req_change_city`：cities 16→30；严格嵌套 435 pairs×6×6×4=62640 组合上限
  - `_tool_not_available`：_NOT_AVAILABLE_VARIANTS 24→40（新增列车/快递/医院/薪资/会议室/发票/餐馆/电商下单/笔记本壁纸/手机定位/智能家居/新闻头条/合同提醒/报销/电费/火车购票等真实业务场景）；严格嵌套 40×5×4×4=3200 组合上限
  - `_error_recovery`：30×2×4×5×5=6000 组合上限，公式改为严格嵌套
  - `_write_manifests()` 新增 `build_count` 字段（对应 `--count` 参数）以帮助测试参数化
- `tests/test_d2_dataset.py`：
  - 新增 `_expected_total_samples()` / `_expected_per_task_samples()` / `_build_count()` 三个 helper，从 on-disk MANIFEST 派生
  - `test_canonical_semantic_content_is_disjoint_across_splits` 改用 `assertGreaterEqual(>=expected_per_task)` 而非硬编码 100
  - `test_split_assignment_uses_iid_stratified_shuffle` 改用派生 per-class 70/15/15 期望值
  - 两处 `build_samples(600, ...)` 改用 `_build_count()` 派生
- `.gitignore` 新增 `datasets/tool-calling-d2/` 与 `datasets/tool-calling-d2-*/`
- `git rm -r --cached datasets/tool-calling-d2` 移除已追踪的 5004 个样本与 3 个 MANIFEST（磁盘保留）

验证（生成 + 测试）：
- 仿真校验：6 类各 834 unique canonical signatures，6/6 OK
- 生成 5004 samples 耗时 35s，per-task 834；train 3498 / dev 750 / test 756
- canonical total unique: 5004/5004；train∩dev=0、train∩test=0、dev∩test=0
- D2-vs-D1/D1.1 cross_dataset_signature：3 个 split 均 0 重叠
- `scripts/run_tests.py full` → 本机 Ran **296** tests OK；isolated auditor 环境 294 PASS + 2 environment-driven skips（`tests/test_n2_benchmark.py:93, 108` 的 CUDA skipUnless）；`scripts/validate_stage0.py --examples` → 9/9 PASS

遗留：
- split 实际为 3498/750/756（不是字面 3500/750/750），原因是 `count % 6 == 0` 约束下 6×834=5004，per-class floor-split 583/125/126，6 类累加为 3498/750/750/756。这是数学必然；如需严格 3500/750/750 应改用 6×833+2 模式（4 类 833、2 类 834），但会让 canonical uniqueness 检查需要按 per-class 单独断言而非统一 floor-split。已在 `docs/data/d2-expansion.md` §8 明确记录。
- reviewer dispatch 与 detached auditor 还未启动（待本目标下一阶段审查）。
- round 14 实际状态：reviewer dispatch 已完成 + PASS verdict；detached auditor round 14 在 16:58 返回 2 条 fixes required（同步不一致 + skipped 计数）。本 round 14 （16:58 后）的 文档同步修正已涵盖这两点。

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
